"""Tests for the adapt() protocol and structured-sweep adaptation."""
import numpy as np

from meshwell.resolution import (
    _equidistribute,
    _gradation_limit,
    _insert_required,
)


class TestGradationLimit:
    def test_flat_untouched(self):
        h = np.array([0.1, 0.1, 0.1])
        np.testing.assert_allclose(_gradation_limit(h, 1.3), h)

    def test_jump_capped_both_directions(self):
        h = np.array([0.1, 1e6, 0.1])
        out = _gradation_limit(h, 1.3)
        ratios = out[1:] / out[:-1]
        assert np.all(ratios <= 1.3 + 1e-12)
        assert np.all(ratios >= 1 / 1.3 - 1e-12)
        # small values are never increased
        assert out[0] <= 0.1 + 1e-12 and out[2] <= 0.1 + 1e-12

    def test_input_not_mutated(self):
        h = np.array([0.1, 1e6, 0.1])
        _gradation_limit(h, 1.3)
        assert h[1] == 1e6


class TestEquidistribute:
    def test_uniform_reproduces_uniform(self):
        offsets = np.linspace(0.0, 0.4, 5)
        h = np.full(4, 0.1)
        out = _equidistribute(offsets, h)
        np.testing.assert_allclose(out, offsets, atol=1e-12)

    def test_endpoints_pinned(self):
        offsets = np.array([0.0, 0.13, 0.4])
        h = np.array([0.05, 0.11])
        out = _equidistribute(offsets, h)
        assert out[0] == 0.0 and out[-1] == 0.4

    def test_equal_density_increments(self):
        # piecewise target: fine near 0, coarse near 1
        offsets = np.linspace(0.0, 1.0, 11)
        h = 0.02 + 0.2 * (offsets[:-1] + offsets[1:]) / 2
        out = _equidistribute(offsets, h)
        # cumulative ∫ dη / h (staircase) at the new offsets is equally spaced
        cum = np.concatenate([[0.0], np.cumsum(np.diff(offsets) / h)])
        vals = np.interp(out, offsets, cum)
        np.testing.assert_allclose(np.diff(vals), vals[-1] / (len(out) - 1), rtol=1e-9)
        # finer cells where h is smaller
        assert (out[1] - out[0]) < (out[-1] - out[-2])

    def test_min_cells_floor(self):
        offsets = np.array([0.0, 1.0])
        h = np.array([10.0])  # target coarser than the extent
        out = _equidistribute(offsets, h)
        assert len(out) == 3  # 2 cells minimum


class TestInsertRequired:
    def test_noop_when_present(self):
        off = np.array([0.0, 1.0, 2.0])
        np.testing.assert_allclose(_insert_required(off, (1.0,)), off)

    def test_snaps_nearest_interior(self):
        off = np.array([0.0, 0.9, 2.0])
        out = _insert_required(off, (1.0,))
        np.testing.assert_allclose(out, [0.0, 1.0, 2.0])

    def test_never_moves_endpoints(self):
        off = np.array([0.0, 1.0, 2.0])
        out = _insert_required(off, (0.1, 1.9))
        assert out[0] == 0.0 and out[-1] == 2.0
        assert 0.1 in out and 1.9 in out

    def test_many_required_preserves_endpoints(self):
        off = np.array([0.0, 1.0, 2.0])
        out = _insert_required(off, (0.1, 0.2, 0.3))
        assert out[0] == 0.0 and out[-1] == 2.0
        assert all(r in out for r in (0.1, 0.2, 0.3))
        assert np.all(np.diff(out) > 0)  # strictly sorted, no duplicates
