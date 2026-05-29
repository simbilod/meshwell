"""Phase 3 cohort envelope builder.

For each connected z-component (cohort) of structured slabs, build a
single TopoDS_Solid whose boundary has:

- Top shell of per-piece OCC sub-faces (subdivided by piece boundaries)
- Bottom shell of per-piece OCC sub-faces
- Lateral wall of one OCC face per outline edge (un-subdivided)

The resulting envelope is what cad_occ.fragment_all sees instead of the
per-piece sub-prisms. Per-piece volumes and interior interfaces become
pure gmsh discrete entities at mesh time.

This module is a deliberately stripped subset of cohort_topology.py:
no interior horizontal edges, no interior vertical edges, no interior
lateral faces, no per-piece lateral subdivision. See spec
docs/superpowers/specs/2026-05-28-cad-occ-discrete-internal-cohort-mesh-design.md.

FUTURE WORK: If structured slabs ever need XY-unstructured neighbors,
this builder must subdivide lateral OCC faces along piece-to-piece
interior boundaries that meet the cohort exterior. See "Future work"
in the spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from meshwell.structured.spec import (
    FaceKey,
    StructuredPlan,
)


@dataclass
class CohortEnvelope:
    """Envelope topology + assembled solid for one cohort.

    Registries:
    - vertices: keyed by (z_plane, outline_corner_id) -> TopoDS_Vertex.
      Only outline corners — no interior piece corners.
    - horizontal_edges: keyed by (z_plane, outline_edge_id) -> TopoDS_Wire.
      Only cohort outline edges.
    - vertical_edges: keyed by (zlo, zhi, outline_corner_id) -> TopoDS_Edge.
      Deduped across slabs that share a z-interval (so two adjacent slabs
      sharing an outline edge end up with one TopoDS_Edge per vertical
      corner, letting the shared lateral OCC face close cleanly).
    - top_sub_faces: FaceKey(slab_index, "top", piece_index) -> TopoDS_Face.
      Per-piece top sub-face built from face_partition_provenance.
    - bottom_sub_faces: FaceKey(slab_index, "bot", piece_index) -> TopoDS_Face.
    - lateral_faces: keyed by (slab_index, outline_edge_id) -> list[TopoDS_Face].
      One face per segment for multi-vertex straight outline edges;
      one face per arc outline edge.
    - skipped_edge_ids: set of arrangement edge IDs that were
      degenerate at horizontal-edge build time; later tasks skip
      these when populating downstream registries.

    Plus:
    - outline_xy_to_corner_id: (round(x,9), round(y,9)) -> outline_corner_id.
    - cohort_solid: the assembled TopoDS_Solid (None until assemble_*).
    """

    component_index: int
    plan: StructuredPlan | None
    vertices: dict[tuple[float, int], Any] = field(default_factory=dict)
    horizontal_edges: dict[tuple[float, int], Any] = field(default_factory=dict)
    vertical_edges: dict[tuple[float, float, int], Any] = field(default_factory=dict)
    top_sub_faces: dict[FaceKey, Any] = field(default_factory=dict)
    bottom_sub_faces: dict[FaceKey, Any] = field(default_factory=dict)
    lateral_faces: dict[tuple[int, int], list] = field(default_factory=dict)
    outline_xy_to_corner_id: dict[tuple[float, float], int] = field(
        default_factory=dict
    )
    skipped_edge_ids: set[int] = field(default_factory=set)
    cohort_solid: Any = None
    # Union faces for multi-piece lateral cohorts (set by assemble_cohort_envelope_solid).
    # When 2+ pieces share the same z-level, these are used instead of individual sub-faces.
    bottom_union_face: Any = None
    top_union_face: Any = None
    # Sewn lateral faces: keyed by (slab_index, outline_edge_id).
    # After assemble_cohort_envelope_solid, each entry holds the face(s) from
    # the sewn solid that correspond to the original env.lateral_faces entry.
    # Use these (not env.lateral_faces) as BOP input tags when sewing modifies TShapes.
    sewn_lateral_faces: dict[tuple[int, int], list] = field(default_factory=dict)


def build_cohort_envelope(
    plan: StructuredPlan,
    component_index: int,
) -> CohortEnvelope:
    """Build the cohort envelope for one connected z-component.

    Walks the cohort's slabs and arrangement to populate the outline-only
    vertex/edge registries plus the per-piece top/bottom sub-faces and
    un-subdivided lateral wall. Does NOT assemble the solid — call
    assemble_cohort_envelope_solid for that.
    """
    import math

    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCP.gp import gp_Pnt

    env = CohortEnvelope(component_index=component_index, plan=plan)
    cohort_slabs = [s for s in plan.slabs if s.component_index == component_index]
    if not cohort_slabs:
        return env

    arrangement = plan.arrangements[component_index]

    z_planes: set[float] = set()
    for s in cohort_slabs:
        z_planes.add(s.zlo)
        z_planes.add(s.zhi)
    z_planes_sorted = sorted(z_planes)

    _ROUND = 9
    for arr_edge in arrangement.edges:
        for x, y in arr_edge.vertices:
            key = (round(x, _ROUND), round(y, _ROUND))
            if key not in env.outline_xy_to_corner_id:
                env.outline_xy_to_corner_id[key] = len(env.outline_xy_to_corner_id)

    corner_id_to_xy: dict[int, tuple[float, float]] = {
        cid: xy for xy, cid in env.outline_xy_to_corner_id.items()
    }
    corner_id_to_arc_snaps: dict[int, list[tuple[float, float]]] = {}
    for arr_edge in arrangement.edges:
        if arr_edge.circle is None:
            continue
        cx, cy = arr_edge.circle.center
        r = arr_edge.circle.radius
        for endpoint_xy in (arr_edge.vertices[0], arr_edge.vertices[-1]):
            key = (round(endpoint_xy[0], _ROUND), round(endpoint_xy[1], _ROUND))
            cid = env.outline_xy_to_corner_id[key]
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

    _brep_builder = BRep_Builder()
    _VERTEX_TOL_MARGIN = 1e-7
    for cid, (x, y) in corner_id_to_xy.items():
        tol = corner_id_to_tol.get(cid, 0.0)
        for z in z_planes_sorted:
            v = BRepBuilderAPI_MakeVertex(gp_Pnt(x, y, z)).Vertex()
            if tol > 0:
                _brep_builder.UpdateVertex(v, tol + _VERTEX_TOL_MARGIN)
            env.vertices[(z, cid)] = v

    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeWire,
    )
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeVertex as _MV,
    )
    from OCP.GC import GC_MakeArcOfCircle
    from OCP.gp import gp_Ax2, gp_Circ, gp_Dir

    _skipped_edge_ids = env.skipped_edge_ids
    for arr_edge in arrangement.edges:
        p1 = arr_edge.vertices[0]
        p2 = arr_edge.vertices[-1]
        c1 = env.outline_xy_to_corner_id[(round(p1[0], _ROUND), round(p1[1], _ROUND))]
        c2 = env.outline_xy_to_corner_id[(round(p2[0], _ROUND), round(p2[1], _ROUND))]
        if arr_edge.circle is None and len(arr_edge.vertices) == 2:
            dist_2v = math.hypot(
                arr_edge.vertices[1][0] - arr_edge.vertices[0][0],
                arr_edge.vertices[1][1] - arr_edge.vertices[0][1],
            )
            if dist_2v < 1e-7 or c1 == c2:
                _skipped_edge_ids.add(arr_edge.edge_id)
                continue
        for z in z_planes_sorted:
            v1 = env.vertices[(z, c1)]
            v2 = env.vertices[(z, c2)]
            mw = BRepBuilderAPI_MakeWire()
            if arr_edge.circle is not None:
                cx, cy = arr_edge.circle.center
                r = arr_edge.circle.radius
                axis = gp_Ax2(gp_Pnt(cx, cy, z), gp_Dir(0, 0, 1))
                circ = gp_Circ(axis, r)
                p1_snapped = corner_id_to_xy[c1]
                p2_snapped = corner_id_to_xy[c2]
                start = gp_Pnt(p1_snapped[0], p1_snapped[1], z)
                end = gp_Pnt(p2_snapped[0], p2_snapped[1], z)
                arc = GC_MakeArcOfCircle(circ, start, end, True).Value()
                edge = BRepBuilderAPI_MakeEdge(arc, v1, v2).Edge()
                mw.Add(edge)
            elif len(arr_edge.vertices) == 2:
                edge = BRepBuilderAPI_MakeEdge(v1, v2).Edge()
                mw.Add(edge)
            else:
                verts_3d = arr_edge.vertices
                n = len(verts_3d)
                for seg_i in range(n - 1):
                    xi, yi = verts_3d[seg_i]
                    xj, yj = verts_3d[seg_i + 1]
                    va = v1 if seg_i == 0 else _MV(gp_Pnt(xi, yi, z)).Vertex()
                    vb = v2 if seg_i == n - 2 else _MV(gp_Pnt(xj, yj, z)).Vertex()
                    mw.Add(BRepBuilderAPI_MakeEdge(va, vb).Edge())
            env.horizontal_edges[(z, arr_edge.edge_id)] = mw.Wire()

    for slab in cohort_slabs:
        for corner_id in env.outline_xy_to_corner_id.values():
            zkey = (slab.zlo, slab.zhi, corner_id)
            if zkey in env.vertical_edges:
                continue
            v_lo = env.vertices[(slab.zlo, corner_id)]
            v_hi = env.vertices[(slab.zhi, corner_id)]
            env.vertical_edges[zkey] = BRepBuilderAPI_MakeEdge(v_lo, v_hi).Edge()

    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepFill import BRepFill
    from OCP.BRepTools import BRepTools_WireExplorer
    from OCP.GC import GC_MakeArcOfCircle
    from OCP.TopoDS import TopoDS

    from meshwell.structured.phantom import (
        _make_face_from_polygon_with_arcs,
        _make_face_from_provenance,
    )

    def _rev_edge(e):
        return TopoDS.Edge_s(e.Reversed())

    slab_to_index = {id(s): i for i, s in enumerate(plan.slabs)}
    for slab in cohort_slabs:
        slab_index = slab_to_index[id(slab)]
        if not slab.face_partition:
            continue

        # Single-piece slabs: build top/bot faces from the shared
        # horizontal_edges registry so edges share OCC TShape with the lateral
        # wall. This lets sewing close the solid correctly for arc outlines.
        if len(slab.face_partition) == 1:
            # Collect the arrangement edge IDs that belong to this piece.
            piece_edge_ids: set[int] = set()
            if slab.face_partition_edges is not None and slab.face_partition_edges:
                for eid, _forward in slab.face_partition_edges[0]:
                    piece_edge_ids.add(eid)
            else:
                # Fallback: use all non-skipped arrangement edges.
                for arr_edge in arrangement.edges:
                    if arr_edge.edge_id not in env.skipped_edge_ids:
                        piece_edge_ids.add(arr_edge.edge_id)

            # Build a lookup from edge_id to arr_edge for arc detection.
            arr_edge_by_id = {e.edge_id: e for e in arrangement.edges}

            # Collect all horizontal wire edges for this slab z-plane.
            for z, face_key_str, face_dict in (
                (slab.zlo, "bot", env.bottom_sub_faces),
                (slab.zhi, "top", env.top_sub_faces),
            ):
                # Build a combined wire from the piece's arrangement edges at this z.
                mw = BRepBuilderAPI_MakeWire()
                closing_arc_edge = None
                # Track overall start/end corners for straight open-chain detection.
                _piece_start_corner: int | None = None
                _piece_end_corner: int | None = None
                _piece_needs_straight_close = False
                for eid in piece_edge_ids:
                    if eid in env.skipped_edge_ids:
                        continue
                    wire = env.horizontal_edges[(z, eid)]
                    exp = BRepTools_WireExplorer(wire)
                    while exp.More():
                        mw.Add(exp.Current())
                        exp.Next()
                    arr_edge = arr_edge_by_id[eid]
                    p1 = arr_edge.vertices[0]
                    p2 = arr_edge.vertices[-1]
                    ec1 = env.outline_xy_to_corner_id[
                        (round(p1[0], _ROUND), round(p1[1], _ROUND))
                    ]
                    ec2 = env.outline_xy_to_corner_id[
                        (round(p2[0], _ROUND), round(p2[1], _ROUND))
                    ]
                    if _piece_start_corner is None:
                        _piece_start_corner = ec1
                    _piece_end_corner = ec2
                    if arr_edge.circle is not None and ec1 != ec2:
                        # Open-chain arc: build the closing arc edge from c2 back to c1.
                        cx, cy = arr_edge.circle.center
                        r = arr_edge.circle.radius
                        axis = gp_Ax2(gp_Pnt(cx, cy, z), gp_Dir(0, 0, 1))
                        circ = gp_Circ(axis, r)
                        p2_snapped = corner_id_to_xy[ec2]
                        p1_snapped = corner_id_to_xy[ec1]
                        start_pt = gp_Pnt(p2_snapped[0], p2_snapped[1], z)
                        end_pt = gp_Pnt(p1_snapped[0], p1_snapped[1], z)
                        v_c2 = env.vertices[(z, ec2)]
                        v_c1 = env.vertices[(z, ec1)]
                        closing_arc = GC_MakeArcOfCircle(
                            circ, start_pt, end_pt, True
                        ).Value()
                        closing_arc_edge = BRepBuilderAPI_MakeEdge(
                            closing_arc, v_c2, v_c1
                        ).Edge()
                    elif (
                        arr_edge.circle is None
                        and len(arr_edge.vertices) > 2
                        and ec1 != ec2
                    ):
                        # Open-chain straight multi-vertex edge: need a straight
                        # closing segment from ec2 back to ec1.
                        _piece_needs_straight_close = True
                if closing_arc_edge is not None:
                    mw.Add(closing_arc_edge)
                elif (
                    _piece_needs_straight_close
                    and _piece_start_corner is not None
                    and _piece_end_corner is not None
                    and _piece_start_corner != _piece_end_corner
                ):
                    # Degenerate single-edge open-chain polygon: add closing straight
                    # segment from last corner back to first corner.
                    v_end = env.vertices[(z, _piece_end_corner)]
                    v_start = env.vertices[(z, _piece_start_corner)]
                    mw.Add(BRepBuilderAPI_MakeEdge(v_end, v_start).Edge())
                face = BRepBuilderAPI_MakeFace(mw.Wire()).Face()
                face_dict[FaceKey(slab_index, face_key_str, 0)] = face
            continue

        # Multi-piece slabs: fall back to the existing provenance-based path.
        for piece_index, piece in enumerate(slab.face_partition):
            # Prefer provenance when available (arc-aware). Fall back to
            # building from the polygon directly for non-arc structured
            # slabs (the planner only populates provenance when
            # identify_arcs=True).
            provenance = None
            if slab.face_partition_provenance is not None and piece_index < len(
                slab.face_partition_provenance
            ):
                provenance = slab.face_partition_provenance[piece_index]
            if provenance is not None:
                bot_face = _make_face_from_provenance(provenance, z=slab.zlo)
                top_face = _make_face_from_provenance(provenance, z=slab.zhi)
            else:
                bot_face = _make_face_from_polygon_with_arcs(
                    piece,
                    z=slab.zlo,
                    identify_arcs=slab.identify_arcs,
                    min_arc_points=slab.min_arc_points,
                    arc_tolerance=slab.arc_tolerance,
                )
                top_face = _make_face_from_polygon_with_arcs(
                    piece,
                    z=slab.zhi,
                    identify_arcs=slab.identify_arcs,
                    min_arc_points=slab.min_arc_points,
                    arc_tolerance=slab.arc_tolerance,
                )
            env.bottom_sub_faces[FaceKey(slab_index, "bot", piece_index)] = bot_face
            env.top_sub_faces[FaceKey(slab_index, "top", piece_index)] = top_face

    # Detect interior arrangement edges: those shared by 2+ arrangement faces.
    # Interior edges lie INSIDE the cohort's union footprint and must NOT get
    # lateral faces in the envelope solid (they would make it non-manifold).
    # Only boundary edges (appearing in exactly 1 arrangement face) need lateral faces.
    edge_face_count: dict[int, int] = {}
    for arr_face in arrangement.faces:
        for eid, _fwd in arr_face.boundary:
            edge_face_count[eid] = edge_face_count.get(eid, 0) + 1
    interior_edge_ids: set[int] = {
        eid for eid, cnt in edge_face_count.items() if cnt >= 2
    }

    # Dedup set: track (zlo, zhi, edge_id) so that two same-z-interval slabs
    # don't each build an identical lateral face for the same boundary edge.
    _built_lateral_z_edge: set[tuple[float, float, int]] = set()

    for slab in cohort_slabs:
        slab_index = slab_to_index[id(slab)]
        for arr_edge in arrangement.edges:
            if arr_edge.edge_id in env.skipped_edge_ids:
                continue
            if arr_edge.edge_id in interior_edge_ids:
                continue  # interior edge — no lateral face in the envelope solid
            key = (slab_index, arr_edge.edge_id)
            if key in env.lateral_faces:
                continue
            z_edge_key = (slab.zlo, slab.zhi, arr_edge.edge_id)
            if z_edge_key in _built_lateral_z_edge:
                continue  # already built for this z-interval by another same-z slab
            _built_lateral_z_edge.add(z_edge_key)
            p1 = arr_edge.vertices[0]
            p2 = arr_edge.vertices[-1]
            c1 = env.outline_xy_to_corner_id[
                (round(p1[0], _ROUND), round(p1[1], _ROUND))
            ]
            c2 = env.outline_xy_to_corner_id[
                (round(p2[0], _ROUND), round(p2[1], _ROUND))
            ]

            if arr_edge.circle is not None:
                bot_wire = env.horizontal_edges[(slab.zlo, arr_edge.edge_id)]
                top_wire = env.horizontal_edges[(slab.zhi, arr_edge.edge_id)]
                bot_exp = BRepTools_WireExplorer(bot_wire)
                bot_arc_edge = bot_exp.Current()
                top_exp = BRepTools_WireExplorer(top_wire)
                top_arc_edge = top_exp.Current()
                face = BRepFill.Face_s(bot_arc_edge, top_arc_edge)
                env.lateral_faces[key] = [face]
            elif len(arr_edge.vertices) == 2:
                bot_wire = env.horizontal_edges[(slab.zlo, arr_edge.edge_id)]
                top_wire = env.horizontal_edges[(slab.zhi, arr_edge.edge_id)]
                v_edge_1 = env.vertical_edges[(slab.zlo, slab.zhi, c1)]
                v_edge_2 = env.vertical_edges[(slab.zlo, slab.zhi, c2)]
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
                face = BRepBuilderAPI_MakeFace(mw.Wire()).Face()
                env.lateral_faces[key] = [face]
            else:
                verts_2d = arr_edge.vertices
                n = len(verts_2d)
                # Collect sub-edges from env.horizontal_edges so that the
                # bottom and top segments of each lateral face SHARE TShapes
                # with the boundary sub-faces (union face or single sub-face).
                # This TShape-sharing is required for BRepBuilderAPI_Sewing to
                # produce a solid whose faces can be found via IsSame() after BOP.
                bot_wire = env.horizontal_edges[(slab.zlo, arr_edge.edge_id)]
                top_wire = env.horizontal_edges[(slab.zhi, arr_edge.edge_id)]
                bot_sub_edges = []
                _exp_b = BRepTools_WireExplorer(bot_wire)
                while _exp_b.More():
                    bot_sub_edges.append(_exp_b.Current())
                    _exp_b.Next()
                top_sub_edges = []
                _exp_t = BRepTools_WireExplorer(top_wire)
                while _exp_t.More():
                    top_sub_edges.append(_exp_t.Current())
                    _exp_t.Next()
                face_list = []
                for seg_i in range(n - 1):
                    xi, yi = verts_2d[seg_i]
                    xj, yj = verts_2d[seg_i + 1]
                    ci = env.outline_xy_to_corner_id[
                        (round(xi, _ROUND), round(yi, _ROUND))
                    ]
                    cj = env.outline_xy_to_corner_id[
                        (round(xj, _ROUND), round(yj, _ROUND))
                    ]
                    v_edge_i = env.vertical_edges[(slab.zlo, slab.zhi, ci)]
                    v_edge_j = env.vertical_edges[(slab.zlo, slab.zhi, cj)]
                    # Reuse existing sub-edges to preserve TShape sharing.
                    bot_seg = bot_sub_edges[seg_i]
                    top_seg = top_sub_edges[seg_i]
                    mw = BRepBuilderAPI_MakeWire()
                    mw.Add(bot_seg)
                    mw.Add(v_edge_j)
                    mw.Add(_rev_edge(top_seg))
                    mw.Add(_rev_edge(v_edge_i))
                    face_list.append(BRepBuilderAPI_MakeFace(mw.Wire()).Face())
                env.lateral_faces[key] = face_list

    return env


def _build_union_hz_face(
    env: "CohortEnvelope",
    arrangement: Any,
    interior_edge_ids: "set[int]",
    z: float,
) -> Any:
    """Build a single horizontal OCC face at height ``z`` spanning all arrangement faces.

    Uses the EXISTING wires from ``env.horizontal_edges`` so that the OCC TShapes
    are shared with the lateral faces already in ``env.lateral_faces``.  This
    TShape-sharing is what allows BRepBuilderAPI_Sewing to close the solid without
    regenerating face TShapes (which would break IsSame()-based PhantomMap look-ups).

    Algorithm
    ---------
    1. Collect the *boundary* edge IDs (those that appear in exactly one
       arrangement face) and their required traversal directions.
    2. Build a start-corner → (eid, fwd) lookup and chain the boundary edges
       in connected order using arrangement vertex coordinates.  This guarantees
       that sub-edges are added to ``BRepBuilderAPI_MakeWire`` in a properly
       connected sequence (no ``DisconnectedWire`` fragmentation).
    3. For each edge in chain order, add its sub-edges (from
       ``env.horizontal_edges``) with the correct per-sub-edge orientation.
    4. Build and return a ``BRepBuilderAPI_MakeFace`` from the assembled wire.
    """
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
    from OCP.BRepTools import BRepTools_WireExplorer
    from OCP.TopoDS import TopoDS

    _ROUND = 9

    # Build a lookup from arrangement edge_id to the edge object.
    arr_edge_by_id = {e.edge_id: e for e in arrangement.edges}

    # Determine the required traversal direction for each boundary edge.
    # The direction comes from the single arrangement face that contains the edge.
    edge_required_fwd: dict[int, bool] = {
        eid: fwd
        for arr_face in arrangement.faces
        for eid, fwd in arr_face.boundary
        if eid not in interior_edge_ids and eid not in env.skipped_edge_ids
    }

    if not edge_required_fwd:
        # Degenerate: no boundary edges.  Return an empty/invalid sentinel.
        msg = "_build_union_hz_face: no boundary edges found"
        raise ValueError(msg)

    # Build start-corner → (eid, fwd) for graph traversal.
    # The "start corner" of an edge (in its required direction) is:
    #   fwd=True  → vertices[0]
    #   fwd=False → vertices[-1]
    start_corner_to_edge: dict[int, tuple[int, bool]] = {}
    for eid, fwd in edge_required_fwd.items():
        arr_edge = arr_edge_by_id[eid]
        entry_xy = arr_edge.vertices[0] if fwd else arr_edge.vertices[-1]
        cid = env.outline_xy_to_corner_id[
            (round(entry_xy[0], _ROUND), round(entry_xy[1], _ROUND))
        ]
        start_corner_to_edge[cid] = (eid, fwd)

    # Chain boundary edges in connected order.
    first_eid, first_fwd = next(iter(edge_required_fwd.items()))
    arr_edge0 = arr_edge_by_id[first_eid]
    entry0 = arr_edge0.vertices[0] if first_fwd else arr_edge0.vertices[-1]
    start_cid = env.outline_xy_to_corner_id[
        (round(entry0[0], _ROUND), round(entry0[1], _ROUND))
    ]

    chained: list[tuple[int, bool]] = []
    visited: set[int] = set()
    cur_cid = start_cid
    while True:
        if cur_cid not in start_corner_to_edge:
            break
        eid, fwd = start_corner_to_edge[cur_cid]
        if eid in visited:
            break
        visited.add(eid)
        chained.append((eid, fwd))
        arr_edge = arr_edge_by_id[eid]
        exit_xy = arr_edge.vertices[-1] if fwd else arr_edge.vertices[0]
        cur_cid = env.outline_xy_to_corner_id[
            (round(exit_xy[0], _ROUND), round(exit_xy[1], _ROUND))
        ]
        if cur_cid == start_cid:
            break

    # Build the wire from the chained boundary edges.
    mw = BRepBuilderAPI_MakeWire()
    for eid, fwd in chained:
        wire = env.horizontal_edges.get((z, eid))
        if wire is None:
            continue
        sub_edges = []
        exp = BRepTools_WireExplorer(wire)
        while exp.More():
            sub_edges.append(exp.Current())
            exp.Next()
        if fwd:
            for edge in sub_edges:
                mw.Add(edge)
        else:
            # Reversed traversal: reverse each sub-edge and add in reverse order.
            for edge in reversed(sub_edges):
                mw.Add(TopoDS.Edge_s(edge.Reversed()))

    return BRepBuilderAPI_MakeFace(mw.Wire()).Face()


def assemble_cohort_envelope_solid(env: CohortEnvelope) -> Any:
    """Assemble the cohort envelope's TopoDS_Solid.

    Builds a closed solid from:
      - Bottom sub-faces at the cohort's global zmin (reversed for -Z normal).
      - Top sub-faces at the cohort's global zmax (kept for +Z normal).
      - All lateral faces from the registry.
      - Synthetic closing lateral faces for arrangement edges that form an
        open chain (single-entity cohorts produce one multi-vertex arrangement
        edge whose start != end corner; the closing segment from end back to
        start must be added here to seal the lateral wall).

    Uses BRepBuilderAPI_Sewing to merge geometrically-coincident edges
    between independently-built sub-faces and lateral faces, then wraps
    the resulting shell in a TopoDS_Solid.

    Populates env.cohort_solid in-place and returns it.
    """
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakeWire,
        BRepBuilderAPI_Sewing,
    )
    from OCP.BRepFill import BRepFill
    from OCP.GC import GC_MakeArcOfCircle
    from OCP.gp import gp_Ax2, gp_Circ, gp_Dir, gp_Pnt
    from OCP.TopoDS import TopoDS, TopoDS_Solid

    def _rev_face(f):
        return TopoDS.Face_s(f.Reversed())

    def _rev_edge(e):
        return TopoDS.Edge_s(e.Reversed())

    plan = env.plan
    if plan is None:
        msg = "CohortEnvelope.plan must not be None during assembly."
        raise ValueError(msg)

    cohort_slabs = [s for s in plan.slabs if s.component_index == env.component_index]
    if not cohort_slabs:
        msg = "No slabs in cohort; cannot assemble envelope solid."
        raise ValueError(msg)

    zmin = min(s.zlo for s in cohort_slabs)
    zmax = max(s.zhi for s in cohort_slabs)
    slab_by_index = dict(enumerate(plan.slabs))
    _ROUND = 9

    # Compute arrangement and interior edge IDs early — needed for _build_union_hz_face
    # and the closing-face logic below.
    arrangement = plan.arrangements[env.component_index] if plan.arrangements else None

    interior_edge_ids: set[int] = set()
    if arrangement is not None:
        _edge_face_count: dict[int, int] = {}
        for arr_face in arrangement.faces:
            for eid, _fwd in arr_face.boundary:
                _edge_face_count[eid] = _edge_face_count.get(eid, 0) + 1
        interior_edge_ids = {eid for eid, cnt in _edge_face_count.items() if cnt >= 2}

    sewing = BRepBuilderAPI_Sewing(1e-6)

    # Collect bottom and top sub-faces at cohort boundary z-levels.
    # Interior horizontal faces (shared between stacked slabs) are excluded
    # so they don't create non-manifold geometry in the envelope solid.
    _zmin_bot_faces = [
        face
        for fk, face in env.bottom_sub_faces.items()
        if slab_by_index[fk.slab_index].zlo == zmin
    ]
    _zmax_top_faces = [
        face
        for fk, face in env.top_sub_faces.items()
        if slab_by_index[fk.slab_index].zhi == zmax
    ]

    if len(_zmin_bot_faces) == 1:
        # Single piece at zmin — add the existing sub-face directly.
        sewing.Add(_rev_face(_zmin_bot_faces[0]))
    else:
        # Multiple lateral pieces at zmin — build a union face using the EXISTING
        # OCC wires from env.horizontal_edges so TShapes are shared with the
        # lateral faces (preserving IsSame() matching after BRepBuilderAPI_Sewing).
        _zmin_union_face = _build_union_hz_face(
            env, arrangement, interior_edge_ids, zmin
        )
        env.bottom_union_face = _zmin_union_face
        sewing.Add(_rev_face(_zmin_union_face))

    if len(_zmax_top_faces) == 1:
        # Single piece at zmax — add the existing sub-face directly.
        sewing.Add(_zmax_top_faces[0])
    else:
        # Multiple lateral pieces at zmax — union into one face (same approach).
        _zmax_union_face = _build_union_hz_face(
            env, arrangement, interior_edge_ids, zmax
        )
        env.top_union_face = _zmax_union_face
        sewing.Add(_zmax_union_face)

    # Add all registered lateral faces.
    for face_list in env.lateral_faces.values():
        for face in face_list:
            sewing.Add(face)

    # Synthetic closing lateral faces: when a cohort has only one arrangement
    # edge and that edge is an open chain (start corner != end corner), the
    # registered faces leave one side open.
    #
    # For straight multi-vertex edges: n-1 segment faces in the registry leave
    # the closing segment from end back to start missing.
    # For arc edges: BRepFill::Face_s covers the arc strip but the closing
    # cylindrical strip from c2 back to c1 (the short way around) is missing.
    #
    # For multi-face cohorts (2+ arrangement faces), the outer boundary is
    # already a closed loop formed by all the boundary arrangement edges —
    # no closing faces are needed (and adding them would corrupt the solid).
    if arrangement is not None:
        # Only add closing faces for single-face cohorts: when there are 2+
        # arrangement faces, the boundary edges already form a closed outer loop.
        _asm_needs_closing = len(arrangement.faces) <= 1

        if _asm_needs_closing:
            # Track which (slab, arr_edge) pairs we've already closed so stacked
            # slabs sharing no z-overlap don't double-add.
            _closed = set()
            for slab in cohort_slabs:
                for arr_edge in arrangement.edges:
                    if arr_edge.edge_id in env.skipped_edge_ids:
                        continue
                    p1 = arr_edge.vertices[0]
                    p2 = arr_edge.vertices[-1]
                    c1 = env.outline_xy_to_corner_id[
                        (round(p1[0], _ROUND), round(p1[1], _ROUND))
                    ]
                    c2 = env.outline_xy_to_corner_id[
                        (round(p2[0], _ROUND), round(p2[1], _ROUND))
                    ]
                    if c1 == c2:
                        # Already closed; no extra face needed.
                        continue
                    close_key = (slab.zlo, slab.zhi, arr_edge.edge_id)
                    if close_key in _closed:
                        continue
                    _closed.add(close_key)

                    if arr_edge.circle is not None:
                        # Arc arrangement edge: build a closing cylindrical strip
                        # via BRepFill::Face_s between two closing arc edges
                        # (one at zlo, one at zhi), going from c2 back to c1.
                        cx, cy = arr_edge.circle.center
                        r = arr_edge.circle.radius
                        closing_edges = []
                        corner_id_to_xy_asm: dict[int, tuple[float, float]] = {}
                        for cid in (c1, c2):
                            # Reuse existing vertex positions.
                            v = env.vertices[(slab.zlo, cid)]
                            from OCP.BRep import BRep_Tool

                            pt = BRep_Tool.Pnt_s(v)
                            corner_id_to_xy_asm[cid] = (pt.X(), pt.Y())
                        for z in (slab.zlo, slab.zhi):
                            axis = gp_Ax2(gp_Pnt(cx, cy, z), gp_Dir(0, 0, 1))
                            circ = gp_Circ(axis, r)
                            p2_xy = corner_id_to_xy_asm[c2]
                            p1_xy = corner_id_to_xy_asm[c1]
                            start_pt = gp_Pnt(p2_xy[0], p2_xy[1], z)
                            end_pt = gp_Pnt(p1_xy[0], p1_xy[1], z)
                            v_c2 = env.vertices[(z, c2)]
                            v_c1 = env.vertices[(z, c1)]
                            closing_arc = GC_MakeArcOfCircle(
                                circ, start_pt, end_pt, True
                            ).Value()
                            closing_edge = BRepBuilderAPI_MakeEdge(
                                closing_arc, v_c2, v_c1
                            ).Edge()
                            closing_edges.append(closing_edge)
                        closing_face = BRepFill.Face_s(
                            closing_edges[0], closing_edges[1]
                        )
                        sewing.Add(closing_face)

                    elif len(arr_edge.vertices) > 2:
                        # Straight multi-vertex open chain: close from c2 to c1.
                        v_edge_1 = env.vertical_edges[(slab.zlo, slab.zhi, c1)]
                        v_edge_2 = env.vertical_edges[(slab.zlo, slab.zhi, c2)]
                        va_bot = env.vertices[(slab.zlo, c2)]
                        vb_bot = env.vertices[(slab.zlo, c1)]
                        va_top = env.vertices[(slab.zhi, c2)]
                        vb_top = env.vertices[(slab.zhi, c1)]
                        bot_seg = BRepBuilderAPI_MakeEdge(va_bot, vb_bot).Edge()
                        top_seg = BRepBuilderAPI_MakeEdge(va_top, vb_top).Edge()
                        mw = BRepBuilderAPI_MakeWire()
                        mw.Add(bot_seg)
                        mw.Add(v_edge_1)
                        mw.Add(_rev_edge(top_seg))
                        mw.Add(_rev_edge(v_edge_2))
                        closing_face = BRepBuilderAPI_MakeFace(mw.Wire()).Face()
                        sewing.Add(closing_face)

    sewing.Perform()
    sewn = sewing.SewedShape()

    # Populate env.sewn_lateral_faces: for each lateral face entry, resolve
    # the post-sewing face using sewing.Modified() if the face TShape was
    # regenerated during sewing (which happens for multi-segment lateral faces
    # when geometrically-coincident edges get merged).
    for key, face_list in env.lateral_faces.items():
        sewn_list: list = []
        for face in face_list:
            modified = sewing.Modified(face)
            if not modified.IsNull():
                sewn_list.append(modified)
            else:
                sewn_list.append(face)
        env.sewn_lateral_faces[key] = sewn_list

    # Update horizontal faces (per-piece sub-faces + union faces) to their
    # post-sewing TShapes. BRepBuilderAPI_Sewing may regenerate face TShapes
    # when stitching edges shared between horizontal and lateral faces. If
    # we keep the pre-sewing face, fmap.FindIndex(face) returns 0 after BOP
    # (the face is not in the compound). Use sewing.Modified() to get the
    # canonical post-sewing shape.
    for fk, face in list(env.bottom_sub_faces.items()):
        _mod = sewing.Modified(face)
        if not _mod.IsNull():
            env.bottom_sub_faces[fk] = _mod
    for fk, face in list(env.top_sub_faces.items()):
        _mod = sewing.Modified(face)
        if not _mod.IsNull():
            env.top_sub_faces[fk] = _mod
    if env.bottom_union_face is not None:
        _bot_mod = sewing.Modified(env.bottom_union_face)
        if not _bot_mod.IsNull():
            env.bottom_union_face = _bot_mod
    if env.top_union_face is not None:
        _top_mod = sewing.Modified(env.top_union_face)
        if not _top_mod.IsNull():
            env.top_union_face = _top_mod

    # BRepBuilderAPI_Sewing.SewedShape() returns a TopoDS_Shell when the
    # cohort geometry is a single connected body, but a TopoDS_Compound of
    # multiple TopoDS_Shell objects when the cohort has disjoint XY
    # sub-volumes (e.g. two separated footprints in the same cohort).
    # Iterate sub-shapes via TopExp_Explorer so both cases are handled.
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer

    shells: list = []
    exp = TopExp_Explorer(sewn, TopAbs_SHELL)
    while exp.More():
        shells.append(TopoDS.Shell_s(exp.Current()))
        exp.Next()

    if not shells:
        msg = f"Sewing produced no shells (shape type {sewn.ShapeType()})"
        raise RuntimeError(msg)

    b = BRep_Builder()
    solid = TopoDS_Solid()
    b.MakeSolid(solid)
    for shell in shells:
        b.Add(solid, shell)

    env.cohort_solid = solid
    return solid
