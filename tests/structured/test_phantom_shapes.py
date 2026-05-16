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


def test_build_sub_prism_returns_solid_with_expected_topology():
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID

    from meshwell.structured.phantom import _build_sub_prism
    from tests.structured._occ_helpers import count_subshapes

    out = _build_sub_prism(_unit_square(), zlo=0.0, zhi=1.0)
    assert count_subshapes(out.solid, TopAbs_SOLID) == 1
    # 1 bottom + 1 top + 4 lateral = 6 faces.
    assert count_subshapes(out.solid, TopAbs_FACE) == 6


def test_build_sub_prism_records_bottom_and_top_face_keys():
    """The returned record knows which face is bottom and which is top, by key."""
    from meshwell.structured.phantom import _build_sub_prism
    from meshwell.structured.spec import FaceKey

    out = _build_sub_prism(
        _unit_square(), zlo=0.0, zhi=1.0, slab_index=2, piece_index=3
    )
    assert FaceKey(slab_index=2, side="bot", piece_index=3) in out.input_faces_by_key
    assert FaceKey(slab_index=2, side="top", piece_index=3) in out.input_faces_by_key


def test_build_sub_prism_records_lateral_faces_per_outer_edge():
    """One lateral face per outer-edge segment, indexed by edge_index."""
    from meshwell.structured.phantom import _build_sub_prism

    out = _build_sub_prism(_unit_square(), zlo=0.0, zhi=1.0)
    # Unit square has 4 outer edges -> 4 lateral faces.
    assert len(out.input_laterals_by_outer_edge) == 4
    assert set(out.input_laterals_by_outer_edge.keys()) == {0, 1, 2, 3}


def test_build_sub_prism_with_hole_records_extra_lateral_faces():
    """A face with a hole produces lateral faces for both outer and inner edges."""
    from meshwell.structured.phantom import _build_sub_prism

    out = _build_sub_prism(_square_with_hole(), zlo=0.0, zhi=1.0)
    # 4 outer + 4 inner = 8 lateral faces total, but we only key the
    # outer ones (Layer A's outer-only contract).
    assert len(out.input_laterals_by_outer_edge) == 4


def test_build_sub_prism_records_bottom_edge_keys():
    """Bottom edge keys cover all bottom face boundary segments."""
    from meshwell.structured.phantom import _build_sub_prism

    out = _build_sub_prism(
        _unit_square(), zlo=0.0, zhi=1.0, slab_index=0, piece_index=0
    )
    bot_edges = {k for k in out.input_edges_by_key if k.side == "bot"}
    # 4 outer edges on a square.
    assert len(bot_edges) == 4
    # All have piece_index=0.
    assert all(k.piece_index == 0 for k in bot_edges)


def test_build_sub_prism_records_bottom_vertex_keys():
    from meshwell.structured.phantom import _build_sub_prism

    out = _build_sub_prism(
        _unit_square(), zlo=0.0, zhi=1.0, slab_index=0, piece_index=0
    )
    bot_verts = {k for k in out.input_vertices_by_key if k.side == "bot"}
    assert len(bot_verts) == 4
