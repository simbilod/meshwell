import pytest
from shapely.geometry import Polygon

from meshwell.polyprism import PolyPrism
from meshwell.structured.cohort import build_cohorts
from meshwell.structured.collect import collect_structured_slabs
from meshwell.structured.exceptions import StructuredVolumetricOverlapError
from meshwell.structured.validators import validate_no_volumetric_cohort_overlap

SQ = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
FAR = Polygon([(100, 100), (110, 100), (110, 110), (100, 110)])


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


def test_unstructured_above_cohort_passes():
    cohort = s_struct("c", 0, 1)
    cap = s_unstr("cap", 1, 2)
    slabs, _ = collect_structured_slabs([cohort, cap])
    cohorts = build_cohorts(slabs)
    validate_no_volumetric_cohort_overlap(cohorts, [cohort, cap])  # no raise


def test_unstructured_laterally_disjoint_passes():
    cohort = s_struct("c", 0, 2)
    far = s_unstr("far", 0, 2, poly=FAR)
    slabs, _ = collect_structured_slabs([cohort, far])
    cohorts = build_cohorts(slabs)
    validate_no_volumetric_cohort_overlap(cohorts, [cohort, far])  # no raise


def test_unstructured_same_zinterval_overlapping_xy_raises():
    cohort = s_struct("c", 0, 1)
    overlap = s_unstr("overlap", 0, 1)  # same z, same XY → volumetric overlap
    slabs, _ = collect_structured_slabs([cohort, overlap])
    cohorts = build_cohorts(slabs)
    with pytest.raises(StructuredVolumetricOverlapError):
        validate_no_volumetric_cohort_overlap(cohorts, [cohort, overlap])
