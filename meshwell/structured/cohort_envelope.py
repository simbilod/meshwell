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
    env = CohortEnvelope(component_index=component_index, plan=plan)
    cohort_slabs = [s for s in plan.slabs if s.component_index == component_index]
    if not cohort_slabs:
        return env
    # Population logic is added incrementally by Tasks 2-6.
    return env


def assemble_cohort_envelope_solid(env: CohortEnvelope) -> Any:
    """Assemble the cohort envelope's TopoDS_Solid from the registries.

    Populates env.cohort_solid in-place and returns it. Implemented in
    Task 7.
    """
    raise NotImplementedError("Implemented in Task 7")
