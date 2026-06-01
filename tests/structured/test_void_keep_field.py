"""Verify SlabMeta carries a `keep` flag (True for solids, False for voids)."""

from shapely.geometry import Polygon

from meshwell.polyprism import PolyPrism
from meshwell.structured.build import build_cohort_compound
from meshwell.structured.cohort import build_cohorts
from meshwell.structured.collect import collect_structured_slabs
from meshwell.structured.decompose import decompose_cohorts
from meshwell.structured.types import ShapeKey, SlabMeta


def test_slab_meta_keep_defaults_true():
    """SlabMeta should default to keep=True so existing call sites are unaffected."""
    key = ShapeKey(tshape_id=1, orientation=0)
    meta = SlabMeta(
        slab_index=0,
        physical_name=("bg",),
        bot_face_key=key,
        top_face_key=key,
        lateral_face_keys=(),
    )
    assert meta.keep is True


def test_slab_meta_keep_can_be_false():
    """SlabMeta with keep=False marks a void sub-solid."""
    key = ShapeKey(tshape_id=1, orientation=0)
    meta = SlabMeta(
        slab_index=0,
        physical_name=("hole",),
        bot_face_key=key,
        top_face_key=key,
        lateral_face_keys=(),
        keep=False,
    )
    assert meta.keep is False


SQ_BIG = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
SQ_SMALL = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])


def test_build_populates_keep_from_source_slab():
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
    subs_per_cohort, _ = decompose_cohorts(cohorts, unstr)
    _, slab_meta = build_cohort_compound(
        cohorts[0],
        subs_per_cohort[0],
        point_tolerance=1e-3,
    )
    by_name = {m.physical_name: m for m in slab_meta.values()}
    assert by_name[("bg",)].keep is True
    assert by_name[("hole",)].keep is False
