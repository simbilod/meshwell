"""Shared CAD helpers: single source of truth for ownership + sort semantics."""
from meshwell.cad_common import normalize_mesh_order, resolve_piece_ownership


def test_lowest_mesh_order_wins():
    owners = resolve_piece_ownership({"p": [(0, 5.0), (1, 2.0), (2, 7.0)]})
    assert owners == {"p": 1}


def test_tie_first_candidate_wins():
    owners = resolve_piece_ownership({"p": [(3, 1.0), (1, 1.0)]})
    assert owners == {"p": 3}


def test_normalize_mesh_order():
    assert normalize_mesh_order(None) == float("inf")
    assert normalize_mesh_order(2.5) == 2.5


def test_backends_share_one_implementation():
    import meshwell.cad_gmsh as g
    import meshwell.cad_occ as o

    assert o._resolve_piece_ownership is g._resolve_piece_ownership
