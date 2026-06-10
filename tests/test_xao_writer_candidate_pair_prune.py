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
