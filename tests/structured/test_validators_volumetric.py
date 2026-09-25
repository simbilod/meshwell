import pytest
from shapely.geometry import Polygon

from meshwell.polyprism import PolyPrism
from meshwell.structured.cohort import build_cohorts
from meshwell.structured.collect import collect_structured_slabs
from meshwell.structured.exceptions import StructuredVolumetricOverlapError
from meshwell.structured.validators import classify_cohort_overlaps

SQ = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
FAR = Polygon([(100, 100), (110, 100), (110, 110), (100, 110)])
BIG = Polygon([(-5, -5), (15, -5), (15, 15), (-5, 15)])


def s_struct(name, zlo, zhi, poly=SQ, mesh_order=1.0):
    return PolyPrism(
        poly,
        {zlo: 0.0, zhi: 0.0},
        physical_name=name,
        structured=True,
        mesh_order=mesh_order,
    )


def s_unstr(name, zlo, zhi, poly=SQ, mesh_order=5.0):
    return PolyPrism(
        poly, {zlo: 0.0, zhi: 0.0}, physical_name=name, mesh_order=mesh_order
    )


def _classify(entities):
    slabs, _ = collect_structured_slabs(entities)
    return classify_cohort_overlaps(build_cohorts(slabs), entities)


def test_unstructured_above_cohort_passes():
    assert _classify([s_struct("c", 0, 1), s_unstr("cap", 1, 2)]) == {}


def test_unstructured_laterally_disjoint_passes():
    assert _classify([s_struct("c", 0, 2), s_unstr("far", 0, 2, poly=FAR)]) == {}


def test_lower_priority_overlap_is_carved():
    # same z, same XY, lower priority (larger mesh_order) → cohort carves it
    assert _classify([s_struct("c", 0, 1), s_unstr("overlap", 0, 1)]) == {1: {0}}


def test_lower_priority_partial_z_overlap_is_carved():
    # cladding spanning past the cohort in z and XY
    assert _classify([s_struct("c", 0, 1), s_unstr("clad", -1, 2, poly=BIG)]) == {
        1: {0}
    }


def test_none_mesh_order_is_lowest_priority():
    ents = [s_struct("c", 0, 1), s_unstr("clad", 0, 1, poly=BIG, mesh_order=None)]
    assert _classify(ents) == {1: {0}}


def test_stepped_cohort_carves_against_every_overlapping_slab():
    ents = [
        s_struct("lo", 0, 1),
        s_struct("hi", 1, 2, poly=Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])),
        s_unstr("clad", 0, 2, poly=BIG),
    ]
    assert _classify(ents) == {2: {0}}


@pytest.mark.parametrize("mesh_order", [1.0, 0.5])
def test_equal_or_higher_priority_overlap_raises(mesh_order):
    ents = [s_struct("c", 0, 1), s_unstr("overlap", 0, 1, mesh_order=mesh_order)]
    with pytest.raises(StructuredVolumetricOverlapError, match="mesh_order"):
        _classify(ents)


def test_none_vs_none_mesh_order_raises():
    ents = [
        s_struct("c", 0, 1, mesh_order=None),
        s_unstr("overlap", 0, 1, mesh_order=None),
    ]
    with pytest.raises(StructuredVolumetricOverlapError):
        _classify(ents)
