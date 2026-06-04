"""Tests for validate_cohort_wrapping."""
from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from meshwell.polyprism import PolyPrism
from meshwell.structured.cohort import build_cohorts
from meshwell.structured.collect import collect_structured_slabs
from meshwell.structured.exceptions import CohortNotWrappedError
from meshwell.structured.validators import validate_cohort_wrapping


def _rect(x1, y1, x2, y2):
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def _build(entities):
    structured_slabs, unstructured = collect_structured_slabs(entities)
    cohorts = build_cohorts(structured_slabs)
    return cohorts, unstructured


def test_validator_passes_when_cohort_fully_wrapped():
    cohort_ent = PolyPrism(
        _rect(0, 0, 10, 10),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="cohort",
        structured=True,
        mesh_order=3.0,
    )
    below = PolyPrism(
        _rect(0, 0, 10, 10),
        {-1.0: 0.0, 0.0: 0.0},
        physical_name="below",
        mesh_order=5.0,
    )
    above = PolyPrism(
        _rect(0, 0, 10, 10),
        {1.0: 0.0, 2.0: 0.0},
        physical_name="above",
        mesh_order=5.0,
    )
    cohorts, unstructured = _build([cohort_ent, below, above])
    # Should not raise.
    validate_cohort_wrapping(cohorts, unstructured)


def test_validator_raises_when_no_neighbour_below():
    cohort_ent = PolyPrism(
        _rect(0, 0, 10, 10),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="cohort",
        structured=True,
        mesh_order=3.0,
    )
    above = PolyPrism(
        _rect(0, 0, 10, 10),
        {1.0: 0.0, 2.0: 0.0},
        physical_name="above",
        mesh_order=5.0,
    )
    cohorts, unstructured = _build([cohort_ent, above])
    with pytest.raises(CohortNotWrappedError) as exc_info:
        validate_cohort_wrapping(cohorts, unstructured)
    assert exc_info.value.z_plane == 0.0


def test_validator_raises_when_neighbour_does_not_cover():
    cohort_ent = PolyPrism(
        _rect(0, 0, 10, 10),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="cohort",
        structured=True,
        mesh_order=3.0,
    )
    # Below is smaller than cohort — does not cover.
    below_small = PolyPrism(
        _rect(0, 0, 5, 5),
        {-1.0: 0.0, 0.0: 0.0},
        physical_name="below_small",
        mesh_order=5.0,
    )
    above = PolyPrism(
        _rect(0, 0, 10, 10),
        {1.0: 0.0, 2.0: 0.0},
        physical_name="above",
        mesh_order=5.0,
    )
    cohorts, unstructured = _build([cohort_ent, below_small, above])
    with pytest.raises(CohortNotWrappedError):
        validate_cohort_wrapping(cohorts, unstructured)


def test_validator_passes_when_multiple_neighbours_together_cover():
    cohort_ent = PolyPrism(
        _rect(0, 0, 10, 10),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="cohort",
        structured=True,
        mesh_order=3.0,
    )
    below_left = PolyPrism(
        _rect(0, 0, 5, 10),
        {-1.0: 0.0, 0.0: 0.0},
        physical_name="below_left",
        mesh_order=5.0,
    )
    below_right = PolyPrism(
        _rect(5, 0, 10, 10),
        {-1.0: 0.0, 0.0: 0.0},
        physical_name="below_right",
        mesh_order=5.0,
    )
    above = PolyPrism(
        _rect(0, 0, 10, 10),
        {1.0: 0.0, 2.0: 0.0},
        physical_name="above",
        mesh_order=5.0,
    )
    cohorts, unstructured = _build([cohort_ent, below_left, below_right, above])
    # Should not raise — together they cover.
    validate_cohort_wrapping(cohorts, unstructured)


def test_structured_pre_pass_raises_on_unwrapped_cohort():
    from meshwell.structured.pipeline import structured_pre_pass

    cohort_ent = PolyPrism(
        _rect(0, 0, 10, 10),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="cohort",
        structured=True,
        mesh_order=3.0,
    )
    # Missing both above and below.
    with pytest.raises(CohortNotWrappedError):
        structured_pre_pass([cohort_ent], point_tolerance=1e-3)
