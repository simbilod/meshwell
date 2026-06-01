"""Verify SlabMeta carries a `keep` flag (True for solids, False for voids)."""

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
