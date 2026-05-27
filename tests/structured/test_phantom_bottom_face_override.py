"""_build_sub_prism with bottom_face_override reuses the provided face's TShape."""

from __future__ import annotations

import shapely
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer

from meshwell.structured.phantom import _build_sub_prism


def _faces(shape):
    out = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        out.append(exp.Current())
        exp.Next()
    return out


def test_bottom_face_override_produces_shared_tshape():
    """Build a prism, take its top face, pass as the next prism's bottom override.

    The resulting two PhantomShape solids must share the interface face's TShape.
    """
    poly = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    A_ps = _build_sub_prism(
        piece=poly,
        zlo=0.0,
        zhi=1.0,
        slab_index=0,
        piece_index=0,
    )
    A_top = A_ps.input_faces_by_key[
        next(k for k in A_ps.input_faces_by_key if k.side == "top")
    ]
    B_ps = _build_sub_prism(
        piece=poly,
        zlo=1.0,
        zhi=2.0,
        slab_index=1,
        piece_index=0,
        bottom_face_override=A_top,
    )

    a_face_hashes = {hash(f) for f in _faces(A_ps.solid)}
    b_face_hashes = {hash(f) for f in _faces(B_ps.solid)}
    shared = a_face_hashes & b_face_hashes
    assert shared, (
        "bottom_face_override did not produce shared TShape identity "
        "between adjacent sub-prisms."
    )
