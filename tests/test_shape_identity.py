"""Collision-free shape identity via ``TopTools_IndexedMapOfShape``.

Pins the semantics that replaced the ``TopTools_ShapeMapHasher`` hash keys
that cad_occ / occ_xao_writer previously used as dict/set identity. Hash
values can collide, silently merging distinct shapes; the registry uses
real ``IsSame`` comparison so identities are collision-free.
"""
from __future__ import annotations

import pytest
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt
from OCP.TopTools import TopTools_IndexedMapOfShape

from meshwell.occ_util import IndexedShapeRegistry, validated_find_index


def test_distinct_shapes_get_distinct_indices():
    """Two distinct simple shapes must receive distinct registry indices."""
    b1 = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1.0, 1.0, 1.0).Shape()
    b2 = BRepPrimAPI_MakeBox(gp_Pnt(2, 0, 0), 1.0, 1.0, 1.0).Shape()
    reg = IndexedShapeRegistry()
    assert reg.index_of(b1) != reg.index_of(b2)


def test_same_shape_and_rewrapped_copy_share_index():
    """A shape and a fresh Python wrapper of its TShape share an index.

    OCP hands back a fresh Python wrapper each time a TopoDS handle is
    produced, so object identity / ``id()`` is not stable. The registry
    keys on the underlying TShape via ``IsSame``.
    """
    box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1.0, 1.0, 1.0).Shape()
    reg = IndexedShapeRegistry()
    first = reg.index_of(box)
    # Same Python handle a second time.
    assert reg.index_of(box) == first
    # Fresh Python wrapper over the identical TShape (distinct py object).
    rewrapped = box.Located(box.Location())
    assert rewrapped is not box
    assert box.IsSame(rewrapped)
    assert reg.index_of(rewrapped) == first


def test_index_is_one_based_and_added_on_first_sight():
    """Indices start at 1 and are assigned in first-seen order."""
    reg = IndexedShapeRegistry()
    b1 = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1.0, 1.0, 1.0).Shape()
    b2 = BRepPrimAPI_MakeBox(gp_Pnt(2, 0, 0), 1.0, 1.0, 1.0).Shape()
    assert reg.index_of(b1) == 1
    assert reg.index_of(b2) == 2
    assert reg.index_of(b1) == 1


def test_oriented_key_distinguishes_orientation():
    """Orientation-aware key separates a shape from its reverse, same index.

    IsSame ignores orientation, so the index component is shared while the
    orientation component differs.
    """
    box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1.0, 1.0, 1.0).Shape()
    reg = IndexedShapeRegistry()
    fwd = reg.oriented_key(box)
    rev = reg.oriented_key(box.Reversed())
    assert fwd != rev
    # Same TShape -> same index component, differing only in orientation.
    assert fwd[0] == rev[0]
    assert fwd[1] != rev[1]


def test_validated_find_index_raises_on_absent_shape():
    """A shape absent from the reference map raises ValueError naming the group.

    Instead of silently serializing a dangling reference=0.
    """
    present = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1.0, 1.0, 1.0).Shape()
    absent = BRepPrimAPI_MakeBox(gp_Pnt(5, 5, 5), 1.0, 1.0, 1.0).Shape()
    ref_map = TopTools_IndexedMapOfShape()
    ref_map.Add(present)
    # Present shape resolves to a positive reference.
    assert validated_find_index(ref_map, present, "some_group") > 0
    with pytest.raises(ValueError, match="my_group"):
        validated_find_index(ref_map, absent, "my_group")
