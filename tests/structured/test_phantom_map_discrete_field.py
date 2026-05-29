"""PhantomMap discrete-face field tests."""

from __future__ import annotations

from meshwell.structured.spec import FaceKey, PhantomMap


def test_phantom_map_has_face_keys_to_discrete_field():
    pm = PhantomMap()
    assert pm.face_keys_to_discrete == {}


def test_phantom_map_can_store_discrete_face_tags():
    pm = PhantomMap()
    pm.face_keys_to_discrete[FaceKey(0, "top", 0)] = 42
    assert pm.face_keys_to_discrete[FaceKey(0, "top", 0)] == 42
