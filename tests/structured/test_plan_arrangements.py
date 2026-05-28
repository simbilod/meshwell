"""Verify StructuredPlan.arrangements is populated by build_plan."""

from __future__ import annotations

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.plan import build_plan


def _square(x0, y0, x1, y1):
    return shapely.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _polyprism(name, z0, z1, mesh_order):
    return PolyPrism(
        polygons=_square(0, 0, 1, 1),
        buffers={float(z0): 0.0, float(z1): 0.0},
        physical_name=name,
        mesh_order=mesh_order,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )


def test_plan_arrangements_populated():
    """build_plan exposes per-component StackArrangements."""
    plan = build_plan([_polyprism("A", 0, 1, 1), _polyprism("B", 1, 2, 2)])
    assert hasattr(plan, "arrangements")
    # A and B are in the same cohort -> one arrangement (component_index=0).
    assert 0 in plan.arrangements
    arr = plan.arrangements[0]
    assert hasattr(arr, "edges")
    assert hasattr(arr, "faces")


def test_plan_arrangements_disjoint_components():
    """Disjoint slabs have separate arrangements (one per component)."""
    plan = build_plan([_polyprism("A", 0, 1, 1), _polyprism("C", 10, 11, 2)])
    assert len(plan.arrangements) == 2
    assert 0 in plan.arrangements
    assert 1 in plan.arrangements
