"""Cohort envelope with concentric arc discs (multi-radius stepped stack).

Spec test #3: verifies the vertex registry's multi-arc snap path (Task 2)
and exercises the envelope builder against the same scene that broke
Phase 2 cohort_topology with StdFail_NotDone before the snap fix.

Status: xfail. Phase 2 also fails on this scene; it requires modelling
annular rings on slab-to-slab interfaces (the larger disc's top minus
the next smaller disc's bottom) as separate OCC faces in the envelope.
Phase 3's current envelope architecture (Tasks 5-7) handles single-radius
arc cohorts and same-footprint stacked slabs but not multi-radius stacking.
Tracked as a follow-up; the multi-arc snap behaviour exercised here is
still verified end-to-end by tests/structured/test_cohort_envelope_vertices.py
::test_multi_arc_vertex_snap_carries_tolerance.
"""

from __future__ import annotations

import math

import pytest
import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.cohort_envelope import (
    assemble_cohort_envelope_solid,
    build_cohort_envelope,
)
from meshwell.structured.plan import build_plan


def _disc(r, n=32):
    return shapely.Polygon(
        [
            (r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n))
            for i in range(n)
        ]
    )


def _arc_slab(r, zlo, zhi, name):
    return PolyPrism(
        polygons=_disc(r),
        buffers={zlo: 0.0, zhi: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
        physical_name=name,
    )


@pytest.mark.xfail(
    reason="Multi-radius stepped stack requires modelling exposed annular "
    "rings on slab-to-slab interfaces. Same limitation as Phase 2 "
    "cohort_topology. Phase 3 envelope architecture currently only handles "
    "single-radius arc cohorts and same-footprint stacks."
)
def test_concentric_arc_discs_envelope_builds():
    from OCP.BRepCheck import BRepCheck_Analyzer

    plan = build_plan(
        [
            _arc_slab(1.0, 0.0, 1.0, "L1"),
            _arc_slab(0.7, 1.0, 2.0, "L2"),
            _arc_slab(0.5, 2.0, 3.0, "L3"),
        ]
    )
    env = build_cohort_envelope(plan, component_index=0)
    solid = assemble_cohort_envelope_solid(env)
    # Solid must be BRepCheck-valid even with the multi-arc corners.
    assert BRepCheck_Analyzer(solid).IsValid()
