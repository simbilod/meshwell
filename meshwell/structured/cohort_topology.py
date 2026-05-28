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
      TopoDS_Edge. Each at the cohort's arrangement edge geometry, placed
      at the given z_plane.
    - vertical_edges: keyed by (z_interval_id, xy_corner_id) ->
      TopoDS_Edge. Each connects the bottom-z vertex to the top-z vertex
      at the same xy corner.
    - horizontal_faces: keyed by (z_plane, piece_id) -> TopoDS_Face. Each
      is a horizontal face of one cohort piece at one z-plane; serves as
      the TOP of the slab below AND the BOTTOM of the slab above.
    - lateral_faces: keyed by (z_interval_id, arrangement_edge_id) ->
      TopoDS_Face. Each extrudes an arrangement edge across one slab's
      z-interval.

    Plus an internal helper:
    - xy_to_corner_id: maps (round(x, 9), round(y, 9)) -> int. Stable
      indexing of the unique XY corners across all arrangement edges,
      used as the key in vertices/vertical_edges registries.

    piece_id = (source_index, piece_index) — disambiguates pieces within
    this cohort (registries are per-cohort, so component_index is implicit).
    """

    component_index: int
    plan: StructuredPlan | None  # back-reference for slab/piece lookups
    vertices: dict[tuple[float, int], Any] = field(default_factory=dict)
    horizontal_edges: dict[tuple[float, int], Any] = field(default_factory=dict)
    vertical_edges: dict[tuple[int, int], Any] = field(default_factory=dict)
    horizontal_faces: dict[tuple[float, tuple[int, int]], Any] = field(
        default_factory=dict
    )
    lateral_faces: dict[tuple[int, int], Any] = field(default_factory=dict)
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

    This task lands the vertex registry and horizontal-edge registry for
    STRAIGHT arrangement edges. Arc support comes in Task 7; vertical
    edges in Task 4; horizontal faces in Task 5; lateral faces in Task 6.
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

    # Vertex snap: corners that touch an arc arrangement edge are snapped to
    # lie EXACTLY on the fitted circle. Polygon vertices are only approximately
    # on the arc curve; snapping ensures BRepBuilderAPI_MakeEdge(arc, v1, v2)
    # accepts them. xy_to_corner_id stays keyed by original polygon-derived XY
    # (lookups in the lateral-face loop use those keys); corner_id_to_xy stores
    # the (possibly snapped) XY used when building vertices.
    corner_id_to_xy: dict[int, tuple[float, float]] = {
        cid: xy for xy, cid in topology.xy_to_corner_id.items()
    }
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
                corner_id_to_xy[cid] = (cx + r * dx / d, cy + r * dy / d)

    # Vertex registry — use snapped XY (when applicable).
    for cid, (x, y) in corner_id_to_xy.items():
        for z in z_planes_sorted:
            topology.vertices[(z, cid)] = BRepBuilderAPI_MakeVertex(
                gp_Pnt(x, y, z)
            ).Vertex()

    # Horizontal edge registry — straight and arc edges.
    for arr_edge in arrangement.edges:
        p1 = arr_edge.vertices[0]
        p2 = arr_edge.vertices[-1]
        c1 = topology.xy_to_corner_id[(round(p1[0], _ROUND), round(p1[1], _ROUND))]
        c2 = topology.xy_to_corner_id[(round(p2[0], _ROUND), round(p2[1], _ROUND))]
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
            else:
                edge = BRepBuilderAPI_MakeEdge(v1, v2).Edge()
            topology.horizontal_edges[(z, arr_edge.edge_id)] = edge

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
    # Straight edges → planar face; arc edges → cylindrical face.
    #
    # Sharing invariant: two slabs at the SAME z-interval (same zlo, zhi)
    # that share an arrangement edge must reference the SAME TopoDS_Face
    # TShape so that cohort-interior laterals are shared at assembly time.
    # We achieve this via a secondary dedup dict keyed by (zlo, zhi, edge_id).
    _lateral_by_zinterval: dict[tuple[float, float, int], object] = {}
    for slab in cohort_slabs:
        slab_index = slab_to_index[id(slab)]
        for arr_edge in arrangement.edges:
            key = (slab_index, arr_edge.edge_id)
            if key in topology.lateral_faces:
                continue
            # Reuse an existing lateral face if one was already built for
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
            bot_edge = topology.horizontal_edges[(slab.zlo, arr_edge.edge_id)]
            top_edge = topology.horizontal_edges[(slab.zhi, arr_edge.edge_id)]
            v_edge_1 = topology.vertical_edges[(slab_index, c1)]
            v_edge_2 = topology.vertical_edges[(slab_index, c2)]
            mw = BRepBuilderAPI_MakeWire()
            mw.Add(bot_edge)
            mw.Add(v_edge_2)
            mw.Add(_rev_edge(top_edge))
            mw.Add(_rev_edge(v_edge_1))
            wire = mw.Wire()
            if arr_edge.circle is not None:
                cx, cy = arr_edge.circle.center
                r = arr_edge.circle.radius
                axis = gp_Ax3(gp_Pnt(cx, cy, slab.zlo), gp_Dir(0, 0, 1))
                surface = Geom_CylindricalSurface(axis, r)
                face = BRepBuilderAPI_MakeFace(surface, wire).Face()
            else:
                face = BRepBuilderAPI_MakeFace(wire).Face()
            topology.lateral_faces[key] = face
            _lateral_by_zinterval[zkey] = face

    # Horizontal face registry: per (z_plane, piece_id).
    # piece_id = (slab.source_index, piece_index_within_slab).
    # Same piece_id at multiple z-planes means vertically-adjacent pieces
    # of the same entity share the face at the shared z-plane. Different
    # entities (different source_index) at the same XY footprint get
    # separate piece_ids -> separate faces.
    for slab in cohort_slabs:
        if not slab.face_partition or slab.face_partition_edges is None:
            continue
        for piece_index, piece_edges in enumerate(slab.face_partition_edges):
            piece_id = (slab.source_index, piece_index)
            for z in (slab.zlo, slab.zhi):
                key = (z, piece_id)
                if key in topology.horizontal_faces:
                    continue
                mw = BRepBuilderAPI_MakeWire()
                for arr_edge_id, reversed_orient in piece_edges:
                    edge = topology.horizontal_edges[(z, arr_edge_id)]
                    mw.Add(_rev_edge(edge) if reversed_orient else edge)
                wire = mw.Wire()
                topology.horizontal_faces[key] = BRepBuilderAPI_MakeFace(wire).Face()

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
    piece_id = (slab.source_index, piece_index)

    bot_face = topology.horizontal_faces[(slab.zlo, piece_id)]
    top_face = topology.horizontal_faces[(slab.zhi, piece_id)]

    def _rev_face(f):
        return TopoDS.Face_s(f.Reversed())

    # Build lateral faces per outer arrangement edge, applying orientation.
    piece_edges = slab.face_partition_edges[piece_index]
    lateral_faces_oriented: list = []
    input_laterals: dict = {}
    for outer_edge_i, (arr_edge_id, reversed_orient) in enumerate(piece_edges):
        lateral = topology.lateral_faces[(slab_index, arr_edge_id)]
        oriented = _rev_face(lateral) if reversed_orient else lateral
        lateral_faces_oriented.append(oriented)
        input_laterals[outer_edge_i] = oriented

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
    b.Add(solid, shell)

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
    for corner_i, (arr_edge_id, reversed_orient) in enumerate(piece_edges):
        # Edge per side.
        for side, z in (("bot", slab.zlo), ("top", slab.zhi)):
            edge = topology.horizontal_edges[(z, arr_edge_id)]
            input_edges[
                EdgeKey(
                    slab_index=slab_index,
                    side=side,
                    piece_index=piece_index,
                    edge_index=corner_i,
                )
            ] = edge
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
