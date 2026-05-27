"""Verify Slab.component_index is populated by build_plan."""

from __future__ import annotations

import pytest
import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.plan import build_plan


def _square(x0, y0, x1, y1):
    return shapely.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


@pytest.fixture
def three_slabs_two_components():
    """Slab A (z=0..1) face-touches Slab B (z=1..2); Slab C (z=10..11) is disjoint."""
    A = PolyPrism(
        polygons=_square(0, 0, 1, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="A",
        mesh_order=1,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )
    B = PolyPrism(
        polygons=_square(0, 0, 1, 1),
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="B",
        mesh_order=2,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )
    C = PolyPrism(
        polygons=_square(0, 0, 1, 1),
        buffers={10.0: 0.0, 11.0: 0.0},
        physical_name="C",
        mesh_order=3,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )
    return [A, B, C]


def test_component_index_groups_touching_slabs(three_slabs_two_components):
    plan = build_plan(three_slabs_two_components)
    by_name = {s.physical_name[0]: s for s in plan.slabs}
    assert by_name["A"].component_index == by_name["B"].component_index
    assert by_name["C"].component_index != by_name["A"].component_index


def test_component_index_is_non_negative(three_slabs_two_components):
    plan = build_plan(three_slabs_two_components)
    for s in plan.slabs:
        assert s.component_index >= 0
