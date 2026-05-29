"""Regression test for the multi-arc-corner snap conflict.

The vertex snap (Task 8) initially overwrote multi-arc corners last-write-wins,
causing BRepBuilderAPI_MakeEdge(arc, v1, v2) to fail with StdFail_NotDone for
the first-processed arc whose curve no longer matched the vertex's snapped
position (it was on the last-processed arc's circle instead).

Reproducer: concentric arc discs whose half-arc-fitted circles differ slightly.
This test mirrors the failing-in-production scene
test_stress_stacked_patterns.py::test_stacked_concentric_arc_discs_mesh_clean
but only invokes build_cohort_topology (no mesh) so it runs fast and isolates
the topology builder.
"""

from __future__ import annotations

import math

import pytest
import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.cohort_topology import build_cohort_topology
from meshwell.structured.plan import build_plan
from meshwell.structured.spec import StructuredCohortFootprintMismatchError


def _disc(cx, cy, r, n=32):
    return shapely.Polygon(
        [
            (
                cx + r * math.cos(2 * math.pi * i / n),
                cy + r * math.sin(2 * math.pi * i / n),
            )
            for i in range(n)
        ]
    )


def _arc_slab(r, zlo, zhi, name):
    return PolyPrism(
        polygons=_disc(0, 0, r),
        buffers={zlo: 0.0, zhi: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
        physical_name=name,
    )


@pytest.mark.xfail(
    raises=StructuredCohortFootprintMismatchError,
    reason="Stepped/concentric cohort no longer supported by planner "
    "constancy invariant (added 2026-05-28). See "
    "tests/structured/test_cohort_footprint_constancy.py for the "
    "validator's contract and Phase 3 cohort envelope architecture "
    "for why it's needed.",
)
def test_concentric_arc_disc_cohort_topology_builds_without_stdfail():
    """Stacked concentric arc discs must not crash build_cohort_topology.

    Each disc's circle gets bisected by an arrangement radial cut, producing
    two half-arcs that share an endpoint corner but have independently-fitted
    circles with slightly different radii/centers. The vertex snap must
    produce a position + tolerance that lets BRepBuilderAPI_MakeEdge accept
    BOTH halves' v1/v2.
    """
    entities = [
        _arc_slab(1.0, 0.0, 1.0, "L1"),
        _arc_slab(0.7, 1.0, 2.0, "L2"),
        _arc_slab(0.5, 2.0, 3.0, "L3"),
    ]
    plan = build_plan(entities)
    # build_cohort_topology must complete without raising StdFail_NotDone.
    topology = build_cohort_topology(plan, component_index=0)
    # And produce non-empty registries.
    assert topology.vertices
    assert topology.horizontal_edges
    assert topology.lateral_faces
