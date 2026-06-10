"""Unit tests for AABB candidate-pair pruning in occ_xao_writer.

The pruning has two helpers:
- ``_entity_union_aabbs``: per-entity union of face AABBs.
- ``_candidate_pair_mask``: numpy-vectorized intersection mask.

Together they let ``_compute_physical_groups`` skip the O(N^2) pair
iteration over entity pairs whose AABBs cannot overlap.
"""
from __future__ import annotations

import numpy as np

from meshwell.occ_xao_writer import (
    _candidate_pair_mask,
    _entity_union_aabbs,
)


def test_entity_union_aabbs_single_face():
    """One entity with one face — union AABB equals the face AABB."""
    entity_aabbs = [{42: (0.0, 0.0, 0.0, 1.0, 2.0, 3.0)}]
    union, valid = _entity_union_aabbs(entity_aabbs)
    assert union.shape == (1, 6)
    assert np.allclose(union[0], (0.0, 0.0, 0.0, 1.0, 2.0, 3.0))
    assert valid.tolist() == [True]


def test_entity_union_aabbs_multi_face():
    """Multiple faces — union is element-wise min over mins, max over maxes."""
    entity_aabbs = [
        {
            1: (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            2: (-1.0, 0.5, -2.0, 0.5, 3.0, 0.5),
        }
    ]
    union, valid = _entity_union_aabbs(entity_aabbs)
    assert np.allclose(union[0], (-1.0, 0.0, -2.0, 1.0, 3.0, 1.0))
    assert valid.tolist() == [True]


def test_entity_union_aabbs_empty_entity():
    """Entity with no face AABBs — valid_mask[i] is False."""
    entity_aabbs = [{}]
    union, valid = _entity_union_aabbs(entity_aabbs)
    assert union.shape == (1, 6)
    assert valid.tolist() == [False]
    assert np.all(np.isnan(union[0]))


def test_entity_union_aabbs_mixed():
    """Mixed: some entities populated, one empty."""
    entity_aabbs = [
        {1: (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)},
        {},
        {2: (5.0, 5.0, 5.0, 6.0, 6.0, 6.0)},
    ]
    union, valid = _entity_union_aabbs(entity_aabbs)
    assert union.shape == (3, 6)
    assert valid.tolist() == [True, False, True]
    assert np.allclose(union[0], (0.0, 0.0, 0.0, 1.0, 1.0, 1.0))
    assert np.allclose(union[2], (5.0, 5.0, 5.0, 6.0, 6.0, 6.0))


def _aabb(xmin, ymin, zmin, xmax, ymax, zmax):
    return (xmin, ymin, zmin, xmax, ymax, zmax)


def test_candidate_pair_mask_overlapping():
    """Two AABBs that clearly overlap — pair is included."""
    union = np.array(
        [
            _aabb(0, 0, 0, 1, 1, 1),
            _aabb(0.5, 0.5, 0.5, 2, 2, 2),
        ],
        dtype=float,
    )
    valid = np.array([True, True])
    pairs = _candidate_pair_mask(union, valid, tol=0.0)
    assert pairs.shape == (1, 2)
    assert tuple(pairs[0]) == (0, 1)


def test_candidate_pair_mask_disjoint_far():
    """Two AABBs far apart — pair is excluded."""
    union = np.array(
        [
            _aabb(0, 0, 0, 1, 1, 1),
            _aabb(10, 10, 10, 11, 11, 11),
        ],
        dtype=float,
    )
    valid = np.array([True, True])
    pairs = _candidate_pair_mask(union, valid, tol=1e-2)
    assert pairs.shape == (0, 2)


def test_candidate_pair_mask_edge_touching_within_tol():
    """Two AABBs separated by less than tol — pair is included."""
    union = np.array(
        [
            _aabb(0, 0, 0, 1, 1, 1),
            _aabb(1.005, 0, 0, 2, 1, 1),  # 0.005 gap in x
        ],
        dtype=float,
    )
    valid = np.array([True, True])
    pairs = _candidate_pair_mask(union, valid, tol=1e-2)
    assert pairs.shape == (1, 2)
    assert tuple(pairs[0]) == (0, 1)


def test_candidate_pair_mask_disjoint_at_tol_boundary():
    """Two AABBs separated by more than tol — pair is excluded."""
    union = np.array(
        [
            _aabb(0, 0, 0, 1, 1, 1),
            _aabb(1.02, 0, 0, 2, 1, 1),  # 0.02 gap in x, tol is 0.01
        ],
        dtype=float,
    )
    valid = np.array([True, True])
    pairs = _candidate_pair_mask(union, valid, tol=1e-2)
    assert pairs.shape == (0, 2)


def test_candidate_pair_mask_lexicographic_order():
    """Pairs returned in (i, j) lexicographic order (matches combinations())."""
    # 4 entities arranged so that overlapping pairs are (0,1), (0,2), (1,3).
    union = np.array(
        [
            _aabb(0, 0, 0, 2, 2, 2),  # 0: spans (0..2)
            _aabb(1, 1, 1, 3, 3, 3),  # 1: overlaps 0
            _aabb(0, 0, 0, 1, 1, 1),  # 2: overlaps 0
            _aabb(2.5, 2.5, 2.5, 4, 4, 4),  # 3: overlaps 1, NOT 0 or 2
        ],
        dtype=float,
    )
    valid = np.array([True, True, True, True])
    pairs = _candidate_pair_mask(union, valid, tol=0.0)
    # Expected: (0,1), (0,2), (1,3).
    assert pairs.shape == (3, 2)
    assert [tuple(p) for p in pairs] == [(0, 1), (0, 2), (1, 3)]


def test_candidate_pair_mask_degenerate_entity_always_pairs():
    """Entity with valid=False appears in every pair regardless of AABB."""
    union = np.array(
        [
            _aabb(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),  # 0: degenerate
            _aabb(0, 0, 0, 1, 1, 1),  # 1
            _aabb(100, 100, 100, 101, 101, 101),  # 2: far from 1
        ],
        dtype=float,
    )
    valid = np.array([False, True, True])
    pairs = _candidate_pair_mask(union, valid, tol=1e-2)
    # Expected: (0,1) and (0,2) because 0 is degenerate. (1,2) excluded.
    assert pairs.shape == (2, 2)
    assert [tuple(p) for p in pairs] == [(0, 1), (0, 2)]


def test_candidate_pair_mask_all_degenerate():
    """All entities degenerate — every pair returned."""
    union = np.full((3, 6), np.nan, dtype=float)
    valid = np.array([False, False, False])
    pairs = _candidate_pair_mask(union, valid, tol=1e-2)
    assert pairs.shape == (3, 2)
    assert [tuple(p) for p in pairs] == [(0, 1), (0, 2), (1, 2)]


def test_candidate_pair_mask_empty_input():
    """No entities — empty pair list."""
    union = np.zeros((0, 6), dtype=float)
    valid = np.zeros(0, dtype=bool)
    pairs = _candidate_pair_mask(union, valid, tol=1e-2)
    assert pairs.shape == (0, 2)
