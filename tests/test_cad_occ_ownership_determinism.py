"""Piece-ownership tie-break must depend only on (mesh_order, ent_idx)."""
from __future__ import annotations

from meshwell.cad_occ import _resolve_piece_ownership


def test_lowest_mesh_order_wins():
    candidates = {"piece_a": [(3, 5.0), (1, 2.0), (7, 8.0)]}
    assert _resolve_piece_ownership(candidates) == {"piece_a": 1}


def test_ties_resolved_by_ent_idx_not_candidate_insertion_order():
    # Two equal mesh_orders; lowest ent_idx must win regardless of input order.
    candidates_a = {"piece": [(7, 2.0), (3, 2.0), (5, 2.0)]}
    candidates_b = {"piece": [(3, 2.0), (5, 2.0), (7, 2.0)]}
    candidates_c = {"piece": [(5, 2.0), (7, 2.0), (3, 2.0)]}
    assert _resolve_piece_ownership(candidates_a) == {"piece": 3}
    assert _resolve_piece_ownership(candidates_b) == {"piece": 3}
    assert _resolve_piece_ownership(candidates_c) == {"piece": 3}


def test_single_candidate():
    candidates = {"piece": [(42, 1.0)]}
    assert _resolve_piece_ownership(candidates) == {"piece": 42}
