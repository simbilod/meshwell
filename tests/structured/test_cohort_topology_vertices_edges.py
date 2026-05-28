"""build_cohort_topology populates vertex and horizontal-edge registries."""

from __future__ import annotations

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.cohort_topology import build_cohort_topology
from meshwell.structured.plan import build_plan


def _square(x0, y0, x1, y1):
    return shapely.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _polyprism(name, z0, z1, mesh_order, x0=0, y0=0, x1=1, y1=1):
    return PolyPrism(
        polygons=_square(x0, y0, x1, y1),
        buffers={float(z0): 0.0, float(z1): 0.0},
        physical_name=name,
        mesh_order=mesh_order,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )


def test_vertices_registered_at_each_z_plane():
    """A single square cohort with 2 stacked slabs has 4 corners x 3 z-planes = 12 vertices."""
    plan = build_plan([_polyprism("A", 0, 1, 1), _polyprism("B", 1, 2, 2)])
    topology = build_cohort_topology(plan, component_index=0)
    assert len(topology.vertices) == 12


def test_horizontal_edges_registered_at_each_z_plane():
    """A single square cohort has 4 outer edges x 3 z-planes = 12 horizontal edges."""
    plan = build_plan([_polyprism("A", 0, 1, 1), _polyprism("B", 1, 2, 2)])
    topology = build_cohort_topology(plan, component_index=0)
    assert len(topology.horizontal_edges) == 12


def test_horizontal_edge_uses_registry_vertices():
    """A horizontal edge's endpoints must be the same TShape as the registered vertices."""
    plan = build_plan([_polyprism("A", 0, 1, 1)])
    topology = build_cohort_topology(plan, component_index=0)
    edge_key = next(iter(topology.horizontal_edges))
    edge = topology.horizontal_edges[edge_key]

    from OCP.TopAbs import TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer

    edge_vertex_hashes = set()
    exp = TopExp_Explorer(edge, TopAbs_VERTEX)
    while exp.More():
        edge_vertex_hashes.add(hash(exp.Current()))
        exp.Next()

    registry_vertex_hashes = {hash(v) for v in topology.vertices.values()}
    assert (
        edge_vertex_hashes <= registry_vertex_hashes
    ), "Horizontal edge has endpoints not registered in the vertex registry."


def test_empty_cohort_returns_empty_topology():
    """Cohort with no slabs (component_index out of range) returns empty registries."""
    plan = build_plan([_polyprism("A", 0, 1, 1)])
    topology = build_cohort_topology(plan, component_index=99)
    assert topology.vertices == {}
    assert topology.horizontal_edges == {}
