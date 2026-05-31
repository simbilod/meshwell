import pytest
from shapely.geometry import Polygon

from meshwell.polyprism import PolyPrism
from meshwell.structured.cohort import build_cohorts
from meshwell.structured.collect import collect_structured_slabs
from meshwell.structured.exceptions import StructuredZStackError
from meshwell.structured.validators import validate_z_stacks

SQ = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
FAR = Polygon([(100, 100), (110, 100), (110, 110), (100, 110)])


def s(name, zlo, zhi, poly=SQ, structured=False):
    return PolyPrism(
        polygons=poly,
        buffers={zlo: 0.0, zhi: 0.0},
        physical_name=name,
        structured=structured,
    )


def test_clean_stack_passes():
    cohort_slab = s("c", 0, 1, structured=True)
    cap = s("cap", 1, 2)  # zlo=1 coincides with cohort zhi
    floor = s("floor", -1, 0)  # zhi=0 coincides with cohort zlo
    entities = [cohort_slab, cap, floor]
    slabs, _ = collect_structured_slabs(entities)
    cohorts = build_cohorts(slabs)
    # should not raise
    validate_z_stacks(cohorts, entities)


def test_mid_height_zlo_violates():
    cohort_slab = s("c", 0, 2, structured=True)
    bad = s("bad", 1, 3)  # zlo=1 strictly inside cohort
    entities = [cohort_slab, bad]
    slabs, _ = collect_structured_slabs(entities)
    cohorts = build_cohorts(slabs)
    with pytest.raises(StructuredZStackError) as exc:
        validate_z_stacks(cohorts, entities)
    assert exc.value.z == 1.0
    assert exc.value.entity_index == 1


def test_mid_height_zhi_violates():
    cohort_slab = s("c", 0, 2, structured=True)
    bad = s("bad", -1, 1)  # zhi=1 strictly inside cohort
    entities = [cohort_slab, bad]
    slabs, _ = collect_structured_slabs(entities)
    cohorts = build_cohorts(slabs)
    with pytest.raises(StructuredZStackError):
        validate_z_stacks(cohorts, entities)


def test_mid_height_no_xy_overlap_allowed():
    cohort_slab = s("c", 0, 2, structured=True)
    far_cap = s("far", 1, 3, poly=FAR)  # mid-height but XY-disjoint
    entities = [cohort_slab, far_cap]
    slabs, _ = collect_structured_slabs(entities)
    cohorts = build_cohorts(slabs)
    validate_z_stacks(cohorts, entities)  # no raise


def test_multi_slab_cohort_z_plane_is_legal():
    # cohort has z-planes {0, 1, 2}; an unstructured cap at z=[1,3]
    # has zlo=1 which IS a cohort z-plane → no raise.
    bot = s("bot", 0, 1, structured=True)
    top = s("top", 1, 2, structured=True)
    cap = s("cap", 1, 3)
    entities = [bot, top, cap]
    slabs, _ = collect_structured_slabs(entities)
    cohorts = build_cohorts(slabs)
    validate_z_stacks(cohorts, entities)
