"""Verify decompose emits SubPieces for voids (not just for solids)."""
from shapely.geometry import Polygon

from meshwell.polyprism import PolyPrism
from meshwell.structured.cohort import build_cohorts
from meshwell.structured.collect import collect_structured_slabs
from meshwell.structured.decompose import decompose_cohorts

SQ_BIG = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
SQ_SMALL = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])


def test_void_emits_subpiece():
    """A void carving a solid should emit TWO SubPieces.

    One for the solid annular ring AND one for the void's inner region.
    """
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
    slabs, unstr = collect_structured_slabs([bg, hole])
    cohorts = build_cohorts(slabs)
    subpieces_per_cohort, _, _arrangements = decompose_cohorts(cohorts, unstr)
    assert len(subpieces_per_cohort) == 1
    subs = subpieces_per_cohort[0]
    # Expect 2 sub-pieces at z=[0,1]: annular ring + inner disc.
    assert len(subs) == 2
    sources = sorted(sp.source_slab_indices[0] for sp in subs)
    # bg is entity 0, hole is entity 1 in the input list.
    assert sources == [0, 1]


def test_solid_only_still_one_subpiece():
    """Without a void, a single solid emits one SubPiece (unchanged behavior)."""
    bg = PolyPrism(
        SQ_BIG,
        {0.0: 0.0, 1.0: 0.0},
        physical_name="bg",
        structured=True,
        mesh_order=1.0,
    )
    slabs, unstr = collect_structured_slabs([bg])
    cohorts = build_cohorts(slabs)
    subpieces_per_cohort, _, _arrangements = decompose_cohorts(cohorts, unstr)
    assert len(subpieces_per_cohort[0]) == 1
