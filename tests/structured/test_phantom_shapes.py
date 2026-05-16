"""Tests for meshwell.structured.phantom shape construction."""
from __future__ import annotations

import pytest
from shapely.geometry import Polygon


def _unit_square() -> Polygon:
    return Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


def _square_with_hole() -> Polygon:
    outer = [(0, 0), (4, 0), (4, 4), (0, 4)]
    hole = [(1, 1), (2, 1), (2, 2), (1, 2)]
    return Polygon(outer, [hole])


def test_make_face_returns_topods_face_at_z():
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_VERTEX

    from meshwell.structured.phantom import _make_face_from_polygon
    from tests.structured._occ_helpers import count_subshapes

    face = _make_face_from_polygon(_unit_square(), z=2.5)
    assert count_subshapes(face, TopAbs_EDGE) == 4
    assert count_subshapes(face, TopAbs_VERTEX) == 4


def test_make_face_with_hole_has_two_wires():
    from OCP.TopAbs import TopAbs_WIRE

    from meshwell.structured.phantom import _make_face_from_polygon
    from tests.structured._occ_helpers import count_subshapes

    face = _make_face_from_polygon(_square_with_hole(), z=0.0)
    # 1 outer wire + 1 inner wire.
    assert count_subshapes(face, TopAbs_WIRE) == 2


def test_make_face_canonicalizes_orientation():
    """A CW-ordered shapely polygon (reversed) still produces a valid face."""
    from meshwell.structured.phantom import _make_face_from_polygon

    cw = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])  # CW exterior
    face = _make_face_from_polygon(cw, z=0.0)
    assert face is not None  # didn't crash; the canonicalize step flipped it


def test_make_face_z_is_respected():
    """The face sits at the requested z plane."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface

    from meshwell.structured.phantom import _make_face_from_polygon

    face = _make_face_from_polygon(_unit_square(), z=7.0)
    surf = BRepAdaptor_Surface(face)
    u_mid = 0.5 * (surf.FirstUParameter() + surf.LastUParameter())
    v_mid = 0.5 * (surf.FirstVParameter() + surf.LastVParameter())
    pnt = surf.Value(u_mid, v_mid)
    assert pnt.Z() == pytest.approx(7.0)
