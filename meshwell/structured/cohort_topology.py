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

    Implementation lands in Tasks 3-8. Stub returns an empty topology.
    """
    return CohortTopology(component_index=component_index, plan=plan)


def assemble_cohort_sub_prism(
    topology: CohortTopology,
    slab: Slab,
    piece_index: int,
) -> PhantomShape:
    """Assemble one sub-prism's solid + PhantomShape from the registry.

    Implementation lands in Task 9.
    """
    raise NotImplementedError(
        "assemble_cohort_sub_prism is implemented in Task 9 of the plan."
    )
