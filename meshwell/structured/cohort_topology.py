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
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeVertex
    from OCP.gp import gp_Pnt

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

    # Vertex registry.
    for (x, y), corner_id in topology.xy_to_corner_id.items():
        for z in z_planes_sorted:
            topology.vertices[(z, corner_id)] = BRepBuilderAPI_MakeVertex(
                gp_Pnt(x, y, z)
            ).Vertex()

    # Horizontal edge registry — straight edges only for now (Task 7 adds arc support).
    for arr_edge in arrangement.edges:
        if arr_edge.circle is not None:
            continue  # arc edges deferred to Task 7
        p1 = arr_edge.vertices[0]
        p2 = arr_edge.vertices[-1]
        c1 = topology.xy_to_corner_id[(round(p1[0], _ROUND), round(p1[1], _ROUND))]
        c2 = topology.xy_to_corner_id[(round(p2[0], _ROUND), round(p2[1], _ROUND))]
        for z in z_planes_sorted:
            v1 = topology.vertices[(z, c1)]
            v2 = topology.vertices[(z, c2)]
            topology.horizontal_edges[(z, arr_edge.edge_id)] = BRepBuilderAPI_MakeEdge(
                v1, v2
            ).Edge()

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

    return topology


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
