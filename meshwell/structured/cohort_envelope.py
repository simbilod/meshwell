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

    from meshwell.structured.phantom import (
        _make_face_from_polygon_with_arcs,
        _make_face_from_provenance,
    )

    slab_to_index = {id(s): i for i, s in enumerate(plan.slabs)}
    for slab in cohort_slabs:
        slab_index = slab_to_index[id(slab)]
        if not slab.face_partition:
            continue
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

    # Subsequent registries are added in Tasks 6.
    return env


def assemble_cohort_envelope_solid(env: CohortEnvelope) -> Any:
    """Assemble the cohort envelope's TopoDS_Solid from the registries.

    Populates env.cohort_solid in-place and returns it. Implemented in
    Task 7.
    """
    raise NotImplementedError("Implemented in Task 7")
