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

    # Subsequent registries are added in Tasks 3-6.
    return env


def assemble_cohort_envelope_solid(env: CohortEnvelope) -> Any:
    """Assemble the cohort envelope's TopoDS_Solid from the registries.

    Populates env.cohort_solid in-place and returns it. Implemented in
    Task 7.
    """
    raise NotImplementedError("Implemented in Task 7")
