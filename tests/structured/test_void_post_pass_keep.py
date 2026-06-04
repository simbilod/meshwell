"""Verify void sub-solids exit the post-pass with keep=False."""
from shapely.geometry import Polygon

from meshwell.cad_occ import cad_occ
from meshwell.polyprism import PolyPrism
from meshwell.structured.pipeline import structured_post_pass, structured_pre_pass

SQ_BIG = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
SQ_SMALL = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])


def test_void_sub_solid_marked_keep_false():
    bg = PolyPrism(
        SQ_BIG,
        {0.0: 0.0, 1.0: 0.0},
        physical_name="bg",
        structured=True,
        mesh_order=2.0,
    )
    hole = PolyPrism(
        SQ_SMALL,
        {0.0: 0.0, 1.0: 0.0},
        physical_name="hole",
        structured=True,
        mesh_order=1.0,
        mesh_bool=False,
    )
    below = PolyPrism(
        SQ_BIG,
        {-1.0: 0.0, 0.0: 0.0},
        physical_name="below",
        mesh_order=5.0,
    )
    above = PolyPrism(
        SQ_BIG,
        {1.0: 0.0, 2.0: 0.0},
        physical_name="above",
        mesh_order=5.0,
    )
    state = structured_pre_pass([bg, hole, below, above], point_tolerance=1e-3)
    occ_entities = cad_occ(state.entities_out)
    final = structured_post_pass(occ_entities, state)
    keep_by_name = {}
    for e in final:
        # Skip synthetic 2D annotators (names starting with __cohort_).
        if e.dim != 3:
            continue
        # The dim=3 entity carries (slab_name, synthetic_name) — get the
        # first (user-facing) name.
        name = e.physical_name[0]
        keep_by_name[name] = e.keep
    assert keep_by_name.get("bg") is True
    assert keep_by_name.get("hole") is False
