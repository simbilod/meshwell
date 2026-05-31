from shapely.geometry import Polygon

from meshwell.cad_occ import cad_occ
from meshwell.polyprism import PolyPrism
from meshwell.structured.pipeline import structured_post_pass, structured_pre_pass

SQ = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])


def test_cohort_expands_to_per_slab_entities():
    a = PolyPrism(
        polygons=SQ, buffers={0.0: 0.0, 1.0: 0.0}, physical_name="a", structured=True
    )
    b = PolyPrism(
        polygons=SQ, buffers={1.0: 0.0, 2.0: 0.0}, physical_name="b", structured=True
    )
    state = structured_pre_pass([a, b], point_tolerance=1e-3)
    occ_entities = cad_occ(state.entities_out)
    final = structured_post_pass(occ_entities, state)
    physical_names = {e.physical_name for e in final}
    assert ("a",) in physical_names
    assert ("b",) in physical_names
    assert not any(pn[0].startswith("__cohort_") for pn in physical_names)
