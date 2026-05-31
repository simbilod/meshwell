from shapely.geometry import Polygon

from meshwell.polyprism import PolyPrism
from meshwell.structured.pipeline import structured_pre_pass

SQ = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])


def test_pre_pass_replaces_structured_with_cohort_entity():
    a = PolyPrism(
        polygons=SQ, buffers={0.0: 0.0, 1.0: 0.0}, physical_name="a", structured=True
    )
    b = PolyPrism(
        polygons=SQ, buffers={1.0: 0.0, 2.0: 0.0}, physical_name="b", structured=True
    )
    cap = PolyPrism(polygons=SQ, buffers={2.0: 0.0, 3.0: 0.0}, physical_name="cap")
    state = structured_pre_pass([a, b, cap], point_tolerance=1e-3)
    assert len(state.entities_out) == 2  # one cohort + cap
    from meshwell.structured.cohort_entity import _CohortEntity

    cohort_count = sum(isinstance(e, _CohortEntity) for e in state.entities_out)
    assert cohort_count == 1
    # slab_meta is keyed by ShapeKey of each sub-solid.
    assert len(state.slab_meta) == 2  # one per stacked slab


def test_pre_pass_passthrough_no_structured():
    a = PolyPrism(polygons=SQ, buffers={0.0: 0.0, 1.0: 0.0}, physical_name="a")
    state = structured_pre_pass([a], point_tolerance=1e-3)
    assert state.entities_out == [a]
    assert state.slab_meta == {}
