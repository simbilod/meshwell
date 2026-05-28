"""Sanity check that cohort_topology module exposes expected names."""

from __future__ import annotations

from meshwell.structured.cohort_topology import (
    CohortTopology,
    assemble_cohort_sub_prism,
    build_cohort_topology,
)


def test_cohort_topology_dataclass_has_registries():
    """CohortTopology has the documented registry attributes."""
    t = CohortTopology(
        component_index=0,
        plan=None,
        vertices={},
        horizontal_edges={},
        vertical_edges={},
        horizontal_faces={},
        lateral_faces={},
        xy_to_corner_id={},
    )
    assert t.component_index == 0
    assert t.vertices == {}
    assert t.horizontal_edges == {}
    assert t.vertical_edges == {}
    assert t.horizontal_faces == {}
    assert t.lateral_faces == {}
    assert t.xy_to_corner_id == {}


def test_build_cohort_topology_returns_cohort_topology_for_empty_plan():
    """Stub returns an empty CohortTopology when given a plan with no slabs."""
    from meshwell.structured.spec import StructuredPlan

    plan = StructuredPlan(slabs=(), z_planes=(), overlaps=(), arrangements={})
    topology = build_cohort_topology(plan, component_index=0)
    assert isinstance(topology, CohortTopology)
    assert topology.component_index == 0


def test_assemble_cohort_sub_prism_is_callable():
    """assemble_cohort_sub_prism is importable (implemented in Task 9)."""
    assert callable(assemble_cohort_sub_prism)
