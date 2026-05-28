"""Sanity check that cohort_envelope module exposes expected names."""

from __future__ import annotations


def test_cohort_envelope_module_imports():
    from meshwell.structured.cohort_envelope import (
        CohortEnvelope,
        assemble_cohort_envelope_solid,
        build_cohort_envelope,
    )

    assert callable(build_cohort_envelope)
    assert callable(assemble_cohort_envelope_solid)
    assert CohortEnvelope is not None


def test_cohort_envelope_dataclass_has_registries():
    from meshwell.structured.cohort_envelope import CohortEnvelope

    env = CohortEnvelope(
        component_index=0,
        plan=None,
        vertices={},
        horizontal_edges={},
        vertical_edges={},
        top_sub_faces={},
        bottom_sub_faces={},
        lateral_faces={},
        outline_xy_to_corner_id={},
        cohort_solid=None,
    )
    assert env.component_index == 0
    assert env.cohort_solid is None


def test_build_cohort_envelope_returns_envelope_for_empty_plan():
    from meshwell.structured.cohort_envelope import (
        CohortEnvelope,
        build_cohort_envelope,
    )
    from meshwell.structured.spec import StructuredPlan

    plan = StructuredPlan(slabs=(), z_planes=(), overlaps=(), arrangements={})
    env = build_cohort_envelope(plan, component_index=0)
    assert isinstance(env, CohortEnvelope)
    assert env.component_index == 0
