import pytest
from shapely.geometry import Polygon

from meshwell.polyprism import PolyPrism
from meshwell.structured.cohort import build_cohorts
from meshwell.structured.collect import collect_structured_slabs
from meshwell.structured.exceptions import ArcIdentifyConflictError
from meshwell.structured.validators import validate_arc_consistency

SQ_A = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
SQ_OVERLAP = Polygon([(3, 0), (8, 0), (8, 5), (3, 5)])  # shares edge x=3..3
SQ_DISJOINT = Polygon([(20, 20), (25, 20), (25, 25), (20, 25)])


def test_disagree_no_overlap_does_not_raise():
    a = PolyPrism(
        SQ_A, {0: 0, 1: 0}, physical_name="a", structured=True, identify_arcs=True
    )
    b = PolyPrism(
        SQ_DISJOINT,
        {0: 0, 1: 0},
        physical_name="b",
        structured=True,
        identify_arcs=False,
    )
    slabs, _ = collect_structured_slabs([a, b])
    cohorts = build_cohorts(slabs)
    validate_arc_consistency(cohorts)  # no raise


def test_disagree_with_shared_boundary_raises():
    a = PolyPrism(
        SQ_A, {0: 0, 1: 0}, physical_name="a", structured=True, identify_arcs=True
    )
    b = PolyPrism(
        SQ_OVERLAP,
        {0: 0, 1: 0},
        physical_name="b",
        structured=True,
        identify_arcs=False,
    )
    slabs, _ = collect_structured_slabs([a, b])
    cohorts = build_cohorts(slabs)
    with pytest.raises(ArcIdentifyConflictError):
        validate_arc_consistency(cohorts)


def test_agree_does_not_raise():
    a = PolyPrism(
        SQ_A, {0: 0, 1: 0}, physical_name="a", structured=True, identify_arcs=True
    )
    b = PolyPrism(
        SQ_OVERLAP, {0: 0, 1: 0}, physical_name="b", structured=True, identify_arcs=True
    )
    slabs, _ = collect_structured_slabs([a, b])
    cohorts = build_cohorts(slabs)
    validate_arc_consistency(cohorts)
