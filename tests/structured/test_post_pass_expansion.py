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
    below = PolyPrism(polygons=SQ, buffers={-1.0: 0.0, 0.0: 0.0}, physical_name="below")
    above = PolyPrism(polygons=SQ, buffers={2.0: 0.0, 3.0: 0.0}, physical_name="above")
    state = structured_pre_pass([a, b, below, above], point_tolerance=1e-3)
    occ_entities = cad_occ(state.entities_out)
    final = structured_post_pass(occ_entities, state)

    # Sub-solid entities carry the user's physical name as the FIRST
    # element of the physical_name tuple; the synthetic
    # ``__cohort_<ci>__slab_<si>`` name is appended for orchestrator
    # tag-lookup but is not the user-facing name.
    solid_first_names = {e.physical_name[0] for e in final if e.dim == 3}
    assert "a" in solid_first_names
    assert "b" in solid_first_names
    assert not any(
        n.startswith("__cohort_") for n in solid_first_names
    ), "sub-solid first physical_name must be the user-facing slab name"

    # The pipeline also emits synthetic dim=2 entities tagging each
    # cohort shell face under a ``__cohort_*`` name; these are
    # bookkeeping entries that the orchestrator strips before the .msh
    # is written.
    face_names = {e.physical_name[0] for e in final if e.dim == 2}
    assert any(
        n.startswith("__cohort_") for n in face_names
    ), "synthetic dim=2 face entities must be emitted"
