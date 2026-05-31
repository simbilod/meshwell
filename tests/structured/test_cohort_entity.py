from shapely.geometry import Polygon

from meshwell.polyprism import PolyPrism
from meshwell.structured.build import build_cohort_compound
from meshwell.structured.cohort import build_cohorts
from meshwell.structured.cohort_entity import _CohortEntity
from meshwell.structured.collect import collect_structured_slabs
from meshwell.structured.decompose import decompose_cohorts

SQ = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])


def test_cohort_entity_min_mesh_order():
    a = PolyPrism(
        polygons=SQ,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="a",
        structured=True,
        mesh_order=3.0,
    )
    b = PolyPrism(
        polygons=SQ,
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="b",
        structured=True,
        mesh_order=1.0,
    )
    slabs, unstr = collect_structured_slabs([a, b])
    cohorts = build_cohorts(slabs)
    subs, _ = decompose_cohorts(cohorts, unstr)
    compound, slab_meta = build_cohort_compound(
        cohorts[0],
        subs[0],
        point_tolerance=1e-3,
    )
    ent = _CohortEntity(compound=compound, slab_meta=slab_meta, cohort=cohorts[0])
    assert ent.mesh_order == 1.0  # min across slabs
    assert ent.mesh_bool is True
    assert ent.dimension == 3
    assert ent.physical_name == ("__cohort_0",)
    assert ent.instanciate_occ() is compound
