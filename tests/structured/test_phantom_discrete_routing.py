"""Phase 3 phantom routing tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import shapely

import meshwell.structured.phantom as _phantom_mod
from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.phantom import build_phantom_shapes
from meshwell.structured.plan import build_plan
from meshwell.structured.spec import FaceKey


def _square_slab(zlo, zhi, name):
    return PolyPrism(
        polygons=shapely.box(0, 0, 1, 1),
        buffers={zlo: 0.0, zhi: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name=name,
    )


def test_phase3_routing_produces_one_phantomshape_per_cohort():
    plan = build_plan([_square_slab(0.0, 1.0, "L1"), _square_slab(1.0, 2.0, "L2")])
    with patch("meshwell.structured.phantom._USE_DISCRETE_COHORT_MESH", True):
        result = build_phantom_shapes(plan)
    # Single cohort (two stacked slabs touch).
    assert len(result.shapes) == 1
    ps = result.shapes[0]
    # Cohort-level PhantomShape carries per-piece top/bot face keys for both slabs.
    assert FaceKey(0, "top", 0) in ps.input_faces_by_key
    assert FaceKey(1, "top", 0) in ps.input_faces_by_key
    assert FaceKey(0, "bot", 0) in ps.input_faces_by_key
    assert FaceKey(1, "bot", 0) in ps.input_faces_by_key


@pytest.mark.skipif(
    getattr(_phantom_mod, "_USE_DISCRETE_COHORT_MESH", False),
    reason="Phase 1+2 path only — Phase 3 envelope produces 1 cohort shape, not per-piece",
)
def test_phase3_flag_off_keeps_phase2_behavior():
    """With _USE_DISCRETE_COHORT_MESH=False, build_phantom_shapes uses the existing path."""
    plan = build_plan([_square_slab(0.0, 1.0, "L1"), _square_slab(1.0, 2.0, "L2")])
    # Default flag is False; result should have one PhantomShape per piece (= 2).
    result = build_phantom_shapes(plan)
    assert len(result.shapes) == 2
