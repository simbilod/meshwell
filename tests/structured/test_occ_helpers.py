"""Smoke test for the OCC test helpers."""
from __future__ import annotations


def test_make_box_returns_solid_with_expected_topology():
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX

    from tests.structured._occ_helpers import count_subshapes, make_box

    box = make_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    assert count_subshapes(box, TopAbs_SOLID) == 1
    assert count_subshapes(box, TopAbs_FACE) == 6
    assert count_subshapes(box, TopAbs_EDGE) == 12
    assert count_subshapes(box, TopAbs_VERTEX) == 8


def test_make_stick_returns_taller_box():
    from OCP.TopAbs import TopAbs_FACE

    from tests.structured._occ_helpers import count_subshapes, make_stick

    stick = make_stick(0.4, 0.4, -0.1, 1.1, 0.2, 0.2)
    assert count_subshapes(stick, TopAbs_FACE) == 6


def test_list_of_shape_iteration():
    """_list_of_shape_to_list converts OCP iterators to plain Python lists."""
    from OCP.TopTools import TopTools_ListOfShape

    from tests.structured._occ_helpers import _list_of_shape_to_list

    empty = TopTools_ListOfShape()
    assert _list_of_shape_to_list(empty) == []
