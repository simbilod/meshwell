"""Cohort topology builder for full vertical+lateral face sharing.

For each connected z-component (cohort), build a shared topology of
vertices, edges, and faces ONCE, then assemble each sub-prism's solid
as a view into that topology. Adjacent cohort sub-prisms (vertically or
laterally) thereby share TopoDS_Face TShape identity at their interfaces,
letting BOPAlgo's pave-filler skip pairwise intersection work.

See spec docs/superpowers/specs/2026-05-27-cad-occ-cohort-topology-builder-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from meshwell.structured.spec import (
    PhantomShape,
    Slab,
    StructuredPlan,
)


@dataclass
class CohortTopology:
    """Shared topology registries for one cohort.

    Per spec Section 'Architecture'. The five registries:

    - vertices: keyed by (z_plane, xy_corner_id) -> TopoDS_Vertex.
    - horizontal_edges: keyed by (z_plane, arrangement_edge_id) ->
      TopoDS_Wire. Each at the cohort's arrangement edge geometry, placed
      at the given z_plane. For a 2-vertex straight edge this is a 1-edge
      wire; for multi-vertex straight edges it is a multi-edge wire covering
      all intermediate polygon vertices. Arc edges remain a single-edge wire.
    - vertical_edges: keyed by (z_interval_id, xy_corner_id) ->
      TopoDS_Edge. Each connects the bottom-z vertex to the top-z vertex
      at the same xy corner.
    - horizontal_faces: keyed by (z_plane, piece_fingerprint) -> TopoDS_Face.
      piece_fingerprint = tuple(tuple(e) for e in face_partition_edges[i]).
      Two slabs with the same piece_fingerprint at the same z-plane SHARE
      the same TopoDS_Face TShape — this is the vertical face-sharing
      invariant. Each face serves as the TOP of the slab below AND the
      BOTTOM of the slab above when the fingerprints match.
    - lateral_faces: keyed by (z_interval_id, arrangement_edge_id) ->
      list[TopoDS_Face]. Each extrudes one segment of an arrangement edge
      across one slab's z-interval. Multi-vertex arrangement edges produce
      multiple faces (one per segment pair). Single-segment edges produce
      a 1-element list.

    Plus an internal helper:
    - xy_to_corner_id: maps (round(x, 9), round(y, 9)) -> int. Stable
      indexing of the unique XY corners across all arrangement edges,
      used as the key in vertices/vertical_edges registries.
    """

    component_index: int
    plan: StructuredPlan | None  # back-reference for slab/piece lookups
    vertices: dict[tuple[float, int], Any] = field(default_factory=dict)
    horizontal_edges: dict[tuple[float, int], Any] = field(default_factory=dict)
    vertical_edges: dict[tuple[int, int], Any] = field(default_factory=dict)
    horizontal_faces: dict[tuple[float, tuple], Any] = field(default_factory=dict)
    lateral_faces: dict[tuple[int, int], list] = field(default_factory=dict)
    xy_to_corner_id: dict[tuple[float, float], int] = field(default_factory=dict)


def build_cohort_topology(
    plan: StructuredPlan,
    component_index: int,
) -> CohortTopology:
    """Build the shared topology for one cohort.

    Walks the cohort's slabs and arrangement to populate registries of
    vertices, horizontal/vertical edges, and faces. All sub-prisms in the
    cohort are then assembled as views into this topology, so adjacent
    sub-prisms share TopoDS_* TShape identity at their interfaces.

    Handles multi-vertex arrangement edges (which occur when a polygon
    boundary segment bends around a corner not shared with any neighbor):
    horizontal_edges stores a TopoDS_Wire with one edge per consecutive
    vertex pair; lateral_faces stores a list of planar faces (one per
    segment).
    """
    import math

    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakeVertex,
        BRepBuilderAPI_MakeWire,
    )
    from OCP.GC import GC_MakeArcOfCircle
    from OCP.Geom import Geom_CylindricalSurface
    from OCP.gp import gp_Ax2, gp_Ax3, gp_Circ, gp_Dir, gp_Pnt
    from OCP.TopoDS import TopoDS

    def _rev_edge(e):
        """edge.Reversed() returns TopoDS_Shape; cast back to TopoDS_Edge."""
        return TopoDS.Edge_s(e.Reversed())

    def _rev_wire(w):
        """wire.Reversed() returns TopoDS_Shape; cast back to TopoDS_Wire."""
        return TopoDS.Wire_s(w.Reversed())

    topology = CohortTopology(component_index=component_index, plan=plan)

    cohort_slabs = [s for s in plan.slabs if s.component_index == component_index]
    if not cohort_slabs:
        return topology

    arrangement = plan.arrangements[component_index]

    # Collect unique z-planes across the cohort.
    z_planes: set[float] = set()
    for s in cohort_slabs:
        z_planes.add(s.zlo)
        z_planes.add(s.zhi)
    z_planes_sorted = sorted(z_planes)

    # Build a stable xy_corner_id for each unique XY vertex in the arrangement.
    _ROUND = 9
    for arr_edge in arrangement.edges:
        for x, y in arr_edge.vertices:
            key = (round(x, _ROUND), round(y, _ROUND))
            if key not in topology.xy_to_corner_id:
                topology.xy_to_corner_id[key] = len(topology.xy_to_corner_id)

    # Vertex snap: corners that touch arc arrangement edges are snapped onto
    # the fitted circles. A corner can be touched by MULTIPLE arc edges (e.g.
    # the two half-circles of a bisected disc, each with an independently
    # fitted circle whose radius differs slightly). For multi-arc corners we
    # snap to the AVERAGE of all touching arcs' snapped positions and record
    # the max residual distance, then apply that as the vertex's OCC tolerance
    # so BRepBuilderAPI_MakeEdge(arc, v1, v2) accepts the vertex for each arc.
    #
    # xy_to_corner_id stays keyed by the original polygon-derived XY (lookups
    # in later loops use those keys); corner_id_to_xy stores the post-snap XY
    # used when building vertices; corner_id_to_tol stores the per-vertex
    # tolerance to apply.
    from OCP.BRep import BRep_Builder

    corner_id_to_xy: dict[int, tuple[float, float]] = {
        cid: xy for xy, cid in topology.xy_to_corner_id.items()
    }
    corner_id_to_arc_snaps: dict[int, list[tuple[float, float]]] = {}
    for arr_edge in arrangement.edges:
        if arr_edge.circle is None:
            continue
        cx, cy = arr_edge.circle.center
        r = arr_edge.circle.radius
        for endpoint_xy in (arr_edge.vertices[0], arr_edge.vertices[-1]):
            key = (round(endpoint_xy[0], _ROUND), round(endpoint_xy[1], _ROUND))
            cid = topology.xy_to_corner_id[key]
            x, y = corner_id_to_xy[cid]
            dx, dy = x - cx, y - cy
            d = math.hypot(dx, dy)
            if d > 0:
                corner_id_to_arc_snaps.setdefault(cid, []).append(
                    (cx + r * dx / d, cy + r * dy / d)
                )

    corner_id_to_tol: dict[int, float] = {}
    for cid, snaps in corner_id_to_arc_snaps.items():
        avg_x = sum(s[0] for s in snaps) / len(snaps)
        avg_y = sum(s[1] for s in snaps) / len(snaps)
        corner_id_to_xy[cid] = (avg_x, avg_y)
        max_resid = max(math.hypot(s[0] - avg_x, s[1] - avg_y) for s in snaps)
        if max_resid > 0:
            corner_id_to_tol[cid] = max_resid

    # Vertex registry — use snapped XY (when applicable) and bump OCC vertex
    # tolerance to absorb multi-arc snap residual.
    _brep_builder = BRep_Builder()
    _VERTEX_TOL_MARGIN = 1e-7  # safety margin above measured residual
    for cid, (x, y) in corner_id_to_xy.items():
        tol = corner_id_to_tol.get(cid, 0.0)
        for z in z_planes_sorted:
            v = BRepBuilderAPI_MakeVertex(gp_Pnt(x, y, z)).Vertex()
            if tol > 0:
                _brep_builder.UpdateVertex(v, tol + _VERTEX_TOL_MARGIN)
            topology.vertices[(z, cid)] = v

    # Horizontal edge registry — stored as TopoDS_Wire (always).
    #
    # For arc arrangement edges: single arc edge wrapped in a 1-edge wire.
    # For 2-vertex straight edges: single line edge wrapped in a 1-edge wire.
    # For multi-vertex straight edges: one line edge per consecutive vertex
    #   pair, assembled into a multi-edge wire. End vertices (first and last)
    #   use the registered TopoDS_Vertex objects so they share TShape with the
    #   vertical edges; intermediate vertices use fresh gp_Pnt points.
    #
    # Degenerate edges: perturbation can produce sub-nanometer arrangement
    # edges (< 1e-7 in XY distance) that OCC cannot build valid edges from.
    # Such edges are skipped here AND in the lateral/horizontal face loops
    # below (tracked in _skipped_edge_ids).
    _skipped_edge_ids: set[int] = set()
    for arr_edge in arrangement.edges:
        p1 = arr_edge.vertices[0]
        p2 = arr_edge.vertices[-1]
        c1 = topology.xy_to_corner_id[(round(p1[0], _ROUND), round(p1[1], _ROUND))]
        c2 = topology.xy_to_corner_id[(round(p2[0], _ROUND), round(p2[1], _ROUND))]
        # Pre-check for degenerate edges before entering the z-plane loop.
        if arr_edge.circle is None and len(arr_edge.vertices) == 2:
            dist_2v = math.hypot(
                arr_edge.vertices[1][0] - arr_edge.vertices[0][0],
                arr_edge.vertices[1][1] - arr_edge.vertices[0][1],
            )
            if dist_2v < 1e-7 or c1 == c2:
                _skipped_edge_ids.add(arr_edge.edge_id)
                continue
        for z in z_planes_sorted:
            v1 = topology.vertices[(z, c1)]
            v2 = topology.vertices[(z, c2)]
            if arr_edge.circle is not None:
                cx, cy = arr_edge.circle.center
                r = arr_edge.circle.radius
                axis = gp_Ax2(gp_Pnt(cx, cy, z), gp_Dir(0, 0, 1))
                circ = gp_Circ(axis, r)
                # Use snapped positions for arc construction; they lie exactly
                # on the fitted circle so BRepBuilderAPI_MakeEdge(arc, v1, v2)
                # succeeds and the registered vertices become the arc endpoints.
                p1_snapped = corner_id_to_xy[c1]
                p2_snapped = corner_id_to_xy[c2]
                start = gp_Pnt(p1_snapped[0], p1_snapped[1], z)
                end = gp_Pnt(p2_snapped[0], p2_snapped[1], z)
                arc = GC_MakeArcOfCircle(circ, start, end, True).Value()
                edge = BRepBuilderAPI_MakeEdge(arc, v1, v2).Edge()
                mw = BRepBuilderAPI_MakeWire()
                mw.Add(edge)
                topology.horizontal_edges[(z, arr_edge.edge_id)] = mw.Wire()
            elif len(arr_edge.vertices) == 2:
                # Simple 2-vertex straight edge: one segment.
                edge = BRepBuilderAPI_MakeEdge(v1, v2).Edge()
                mw = BRepBuilderAPI_MakeWire()
                mw.Add(edge)
                topology.horizontal_edges[(z, arr_edge.edge_id)] = mw.Wire()
            else:
                # Multi-vertex straight edge: build one edge per consecutive
                # vertex pair. The first and last vertices use the registered
                # TopoDS_Vertex objects (shared with vertical_edges); intermediate
                # vertices are built as fresh gp_Pnt (not registered, since they
                # are interior to the arrangement edge and need no vertical sharing).
                verts_3d = arr_edge.vertices
                n = len(verts_3d)
                mw = BRepBuilderAPI_MakeWire()
                for seg_i in range(n - 1):
                    xi, yi = verts_3d[seg_i]
                    xj, yj = verts_3d[seg_i + 1]
                    # Use registered vertices at the two true endpoints.
                    if seg_i == 0:
                        va = v1
                    else:
                        va = BRepBuilderAPI_MakeVertex(gp_Pnt(xi, yi, z)).Vertex()
                    if seg_i == n - 2:
                        vb = v2
                    else:
                        vb = BRepBuilderAPI_MakeVertex(gp_Pnt(xj, yj, z)).Vertex()
                    seg_edge = BRepBuilderAPI_MakeEdge(va, vb).Edge()
                    mw.Add(seg_edge)
                topology.horizontal_edges[(z, arr_edge.edge_id)] = mw.Wire()

    # Vertical edge registry — per slab, per cohort XY corner.
    # z_interval_id == slab's index in plan.slabs (stable, unique per slab).
    slab_to_index = {id(s): i for i, s in enumerate(plan.slabs)}
    for slab in cohort_slabs:
        slab_index = slab_to_index[id(slab)]
        for corner_id in topology.xy_to_corner_id.values():
            v_lo = topology.vertices[(slab.zlo, corner_id)]
            v_hi = topology.vertices[(slab.zhi, corner_id)]
            topology.vertical_edges[(slab_index, corner_id)] = BRepBuilderAPI_MakeEdge(
                v_lo, v_hi
            ).Edge()

    # Lateral face registry: per (slab_index, arrangement_edge_id).
    # Stored as list[TopoDS_Face] — one face per segment of the arrangement edge.
    # For a simple 2-vertex edge this is a 1-element list (planar quad for straight,
    # cylindrical for arc). For multi-vertex straight edges, one planar quad per
    # consecutive vertex pair.
    #
    # Sharing invariant: two slabs at the SAME z-interval (same zlo, zhi)
    # that share an arrangement edge must reference the SAME TopoDS_Face
    # TShape so that cohort-interior laterals are shared at assembly time.
    # We achieve this via a secondary dedup dict keyed by (zlo, zhi, edge_id).
    _lateral_by_zinterval: dict[tuple[float, float, int], list] = {}
    for slab in cohort_slabs:
        slab_index = slab_to_index[id(slab)]
        for arr_edge in arrangement.edges:
            # Skip degenerate edges that were not registered in horizontal_edges.
            if arr_edge.edge_id in _skipped_edge_ids:
                continue
            key = (slab_index, arr_edge.edge_id)
            if key in topology.lateral_faces:
                continue
            # Reuse an existing lateral face list if one was already built for
            # the same z-interval and arrangement edge (by a laterally-adjacent
            # slab sharing this z-range).
            zkey = (slab.zlo, slab.zhi, arr_edge.edge_id)
            if zkey in _lateral_by_zinterval:
                topology.lateral_faces[key] = _lateral_by_zinterval[zkey]
                continue
            p1 = arr_edge.vertices[0]
            p2 = arr_edge.vertices[-1]
            c1 = topology.xy_to_corner_id[(round(p1[0], _ROUND), round(p1[1], _ROUND))]
            c2 = topology.xy_to_corner_id[(round(p2[0], _ROUND), round(p2[1], _ROUND))]

            if arr_edge.circle is not None:
                # Arc arrangement edge: single cylindrical lateral face.
                bot_wire = topology.horizontal_edges[(slab.zlo, arr_edge.edge_id)]
                top_wire = topology.horizontal_edges[(slab.zhi, arr_edge.edge_id)]
                v_edge_1 = topology.vertical_edges[(slab_index, c1)]
                v_edge_2 = topology.vertical_edges[(slab_index, c2)]
                # bot_wire and top_wire each have exactly one arc edge for arc case.
                from OCP.BRepTools import BRepTools_WireExplorer

                bot_edges = []
                exp = BRepTools_WireExplorer(bot_wire)
                while exp.More():
                    bot_edges.append(exp.Current())
                    exp.Next()
                top_edges = []
                exp = BRepTools_WireExplorer(top_wire)
                while exp.More():
                    top_edges.append(exp.Current())
                    exp.Next()
                mw = BRepBuilderAPI_MakeWire()
                mw.Add(bot_edges[0])
                mw.Add(v_edge_2)
                mw.Add(_rev_edge(top_edges[0]))
                mw.Add(_rev_edge(v_edge_1))
                wire = mw.Wire()
                cx, cy = arr_edge.circle.center
                r = arr_edge.circle.radius
                axis = gp_Ax3(gp_Pnt(cx, cy, slab.zlo), gp_Dir(0, 0, 1))
                surface = Geom_CylindricalSurface(axis, r)
                face = BRepBuilderAPI_MakeFace(surface, wire).Face()
                face_list = [face]
            elif len(arr_edge.vertices) == 2:
                # Simple 2-vertex straight edge: single planar quad.
                bot_wire = topology.horizontal_edges[(slab.zlo, arr_edge.edge_id)]
                top_wire = topology.horizontal_edges[(slab.zhi, arr_edge.edge_id)]
                v_edge_1 = topology.vertical_edges[(slab_index, c1)]
                v_edge_2 = topology.vertical_edges[(slab_index, c2)]
                from OCP.BRepTools import BRepTools_WireExplorer

                bot_edges = []
                exp = BRepTools_WireExplorer(bot_wire)
                while exp.More():
                    bot_edges.append(exp.Current())
                    exp.Next()
                top_edges = []
                exp = BRepTools_WireExplorer(top_wire)
                while exp.More():
                    top_edges.append(exp.Current())
                    exp.Next()
                mw = BRepBuilderAPI_MakeWire()
                mw.Add(bot_edges[0])
                mw.Add(v_edge_2)
                mw.Add(_rev_edge(top_edges[0]))
                mw.Add(_rev_edge(v_edge_1))
                wire = mw.Wire()
                face = BRepBuilderAPI_MakeFace(wire).Face()
                face_list = [face]
            else:
                # Multi-vertex straight edge: one planar quad per segment.
                # Each segment face shares its vertical edges with the corner
                # vertex registry (c1 at start, c2 at end, intermediate corners
                # are local to the segment).
                verts_2d = arr_edge.vertices
                n = len(verts_2d)
                face_list = []
                for seg_i in range(n - 1):
                    xi, yi = verts_2d[seg_i]
                    xj, yj = verts_2d[seg_i + 1]
                    ci_key = (round(xi, _ROUND), round(yi, _ROUND))
                    cj_key = (round(xj, _ROUND), round(yj, _ROUND))
                    ci = topology.xy_to_corner_id[ci_key]
                    cj = topology.xy_to_corner_id[cj_key]
                    # Retrieve or build vertical edges for these intermediate corners.
                    # Endpoint corners already have vertical_edges registered.
                    # Intermediate (non-endpoint) corners were registered in the
                    # vertex registry (all polygon vertices are registered), so their
                    # vertical edges should be available.
                    v_edge_i = topology.vertical_edges[(slab_index, ci)]
                    v_edge_j = topology.vertical_edges[(slab_index, cj)]
                    # Build bottom and top segment edges using registered vertices.
                    va_bot = topology.vertices[(slab.zlo, ci)]
                    vb_bot = topology.vertices[(slab.zlo, cj)]
                    va_top = topology.vertices[(slab.zhi, ci)]
                    vb_top = topology.vertices[(slab.zhi, cj)]
                    bot_seg = BRepBuilderAPI_MakeEdge(va_bot, vb_bot).Edge()
                    top_seg = BRepBuilderAPI_MakeEdge(va_top, vb_top).Edge()
                    mw = BRepBuilderAPI_MakeWire()
                    mw.Add(bot_seg)
                    mw.Add(v_edge_j)
                    mw.Add(_rev_edge(top_seg))
                    mw.Add(_rev_edge(v_edge_i))
                    wire = mw.Wire()
                    face = BRepBuilderAPI_MakeFace(wire).Face()
                    face_list.append(face)

            topology.lateral_faces[key] = face_list
            _lateral_by_zinterval[zkey] = face_list

    # Horizontal face registry: per (z_plane, piece_fingerprint).
    # piece_fingerprint = tuple(tuple(e) for e in piece_edges) — a canonical
    # descriptor of the piece boundary in terms of arrangement edge ids and
    # orientations. Two slabs (even from different entities) with the same
    # face_partition_edges at the same z-plane produce the same geometric face
    # and therefore SHARE it (same TopoDS_Face TShape). This is the key sharing
    # invariant for vertically-stacked cohort sub-prisms.
    #
    # horizontal_edges stores TopoDS_Wire. We extract the edges from each
    # wire and add them individually (optionally reversed), rather than
    # adding the whole wire, to preserve correct face normal orientation.
    #
    # Degenerate case: a single-entity arrangement produces ONE arrangement
    # edge whose vertices form an open chain (polygon boundary minus the
    # closing segment). The wire built from that one edge is open, so
    # BRepBuilderAPI_MakeFace would produce a NULL face. We detect this by
    # tracking the first and last endpoint vertices, and add a closing edge
    # when the wire doesn't close.
    from OCP.BRepTools import BRepTools_WireExplorer

    for slab in cohort_slabs:
        if not slab.face_partition or slab.face_partition_edges is None:
            continue
        for _piece_index, piece_edges in enumerate(slab.face_partition_edges):
            piece_fingerprint = tuple(tuple(e) for e in piece_edges)
            for z in (slab.zlo, slab.zhi):
                key = (z, piece_fingerprint)
                if key in topology.horizontal_faces:
                    continue
                mw = BRepBuilderAPI_MakeWire()
                first_vertex = None
                last_vertex = None
                for arr_edge_id, reversed_orient in piece_edges:
                    # Skip degenerate edges not in the horizontal_edges registry.
                    if arr_edge_id in _skipped_edge_ids:
                        continue
                    wire = topology.horizontal_edges[(z, arr_edge_id)]
                    # Extract edges in wire traversal order, then add them
                    # individually (possibly reversed). This mirrors the original
                    # edge-level _rev_edge approach and preserves face normal direction.
                    wire_edges: list = []
                    exp = BRepTools_WireExplorer(wire)
                    while exp.More():
                        wire_edges.append(exp.Current())
                        exp.Next()
                    if reversed_orient:
                        for edge in reversed(wire_edges):
                            rev = _rev_edge(edge)
                            mw.Add(rev)
                    else:
                        for edge in wire_edges:
                            mw.Add(edge)
                    # Track start/end vertices for closing-edge detection.
                    if wire_edges:
                        if reversed_orient:
                            start_e = _rev_edge(wire_edges[-1])
                            end_e = _rev_edge(wire_edges[0])
                        else:
                            start_e = wire_edges[0]
                            end_e = wire_edges[-1]
                        from OCP.ShapeAnalysis import ShapeAnalysis_Edge

                        sa = ShapeAnalysis_Edge()
                        if first_vertex is None:
                            first_vertex = sa.FirstVertex(start_e)
                        last_vertex = sa.LastVertex(end_e)
                # Check if the wire is closed (first == last vertex TShape).
                # If not, add a closing edge to complete the polygon ring.
                if (
                    first_vertex is not None
                    and last_vertex is not None
                    and not first_vertex.IsSame(last_vertex)
                ):
                    closing_edge = BRepBuilderAPI_MakeEdge(
                        last_vertex, first_vertex
                    ).Edge()
                    mw.Add(closing_edge)
                face_wire = mw.Wire()
                topology.horizontal_faces[key] = BRepBuilderAPI_MakeFace(
                    face_wire
                ).Face()

    return topology


def assemble_cohort_sub_prism(
    topology: CohortTopology,
    slab: Slab,
    piece_index: int,
) -> PhantomShape:
    """Assemble one sub-prism's solid + PhantomShape from the registry."""
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS, TopoDS_Shell, TopoDS_Solid

    from meshwell.structured.spec import EdgeKey, FaceKey, VertexKey

    plan = topology.plan
    if plan is None:
        msg = "CohortTopology.plan must not be None during assembly."
        raise ValueError(msg)
    slab_index = plan.slabs.index(slab)
    piece_edges = slab.face_partition_edges[piece_index]
    piece_fingerprint = tuple(tuple(e) for e in piece_edges)

    bot_face = topology.horizontal_faces[(slab.zlo, piece_fingerprint)]
    top_face = topology.horizontal_faces[(slab.zhi, piece_fingerprint)]

    def _rev_face(f):
        return TopoDS.Face_s(f.Reversed())

    # Build lateral faces per outer arrangement edge, applying orientation.
    # lateral_faces values are list[TopoDS_Face] — iterate all faces in each list.
    # Degenerate edges (skipped during topology build) are absent from the registry;
    # skip them here too (they contribute no lateral face to the solid).
    lateral_faces_oriented: list = []
    input_laterals: dict = {}
    for outer_edge_i, (arr_edge_id, reversed_orient) in enumerate(piece_edges):
        if (slab_index, arr_edge_id) not in topology.lateral_faces:
            continue
        face_list = topology.lateral_faces[(slab_index, arr_edge_id)]
        # For input_laterals tracking, use the first face in the list as the
        # representative (downstream PhantomMap uses this for BOP history).
        # For multi-segment edges, store all faces — input_laterals gets the first.
        first_face = face_list[0]
        oriented_first = _rev_face(first_face) if reversed_orient else first_face
        input_laterals[outer_edge_i] = oriented_first
        for lf in face_list:
            oriented = _rev_face(lf) if reversed_orient else lf
            lateral_faces_oriented.append(oriented)

    # Assemble shell + solid.
    b = BRep_Builder()
    shell = TopoDS_Shell()
    b.MakeShell(shell)
    b.Add(shell, _rev_face(bot_face))  # bottom face's normal points outward (down)
    b.Add(shell, top_face)
    for lf in lateral_faces_oriented:
        b.Add(shell, lf)

    solid = TopoDS_Solid()
    b.MakeSolid(solid)
    # OCC requires the shell to be REVERSED in the solid (outward normals).
    # BRep_Builder.Add adds the shell as-is (FORWARD), which makes the
    # solid "inside out" (negative volume). Reversing the shell before
    # adding gives the correct outward orientation.
    b.Add(solid, TopoDS.Shell_s(shell.Reversed()))

    # Populate PhantomShape input dicts.
    input_faces: dict = {
        FaceKey(slab_index=slab_index, side="bot", piece_index=piece_index): bot_face,
        FaceKey(slab_index=slab_index, side="top", piece_index=piece_index): top_face,
    }

    input_edges: dict = {}
    input_vertices: dict = {}
    arrangement = plan.arrangements[topology.component_index]
    edge_by_id = {e.edge_id: e for e in arrangement.edges}
    _ROUND = 9
    from OCP.BRepTools import BRepTools_WireExplorer

    for corner_i, (arr_edge_id, reversed_orient) in enumerate(piece_edges):
        # Edge per side: store the first TopoDS_Edge from the wire.
        # BOP tracks TopoDS_Edge (not TopoDS_Wire), so we extract the actual
        # edge. For 2-vertex arrangement edges the wire has one edge. For
        # multi-vertex edges, only the first segment edge is stored per key —
        # complete multi-segment tracking requires multiple EdgeKeys (future work).
        # Degenerate edges (not in horizontal_edges) are skipped entirely.
        for side, z in (("bot", slab.zlo), ("top", slab.zhi)):
            wire = topology.horizontal_edges.get((z, arr_edge_id))
            if wire is None:
                continue
            exp = BRepTools_WireExplorer(wire)
            if exp.More():
                first_edge = exp.Current()
                input_edges[
                    EdgeKey(
                        slab_index=slab_index,
                        side=side,
                        piece_index=piece_index,
                        edge_index=corner_i,
                    )
                ] = first_edge
        # Vertex per side: start vertex of the piece-side traversal.
        arr_edge = edge_by_id[arr_edge_id]
        if reversed_orient:
            x, y = arr_edge.vertices[-1]
        else:
            x, y = arr_edge.vertices[0]
        c = topology.xy_to_corner_id[(round(x, _ROUND), round(y, _ROUND))]
        for side, z in (("bot", slab.zlo), ("top", slab.zhi)):
            input_vertices[
                VertexKey(
                    slab_index=slab_index,
                    side=side,
                    piece_index=piece_index,
                    corner_index=corner_i,
                )
            ] = topology.vertices[(z, c)]

    return PhantomShape(
        slab_index=slab_index,
        piece_index=piece_index,
        solid=solid,
        input_faces_by_key=input_faces,
        input_edges_by_key=input_edges,
        input_vertices_by_key=input_vertices,
        input_laterals_by_outer_edge=input_laterals,
    )
