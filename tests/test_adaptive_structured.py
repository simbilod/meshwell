"""Tests for the adapt() protocol and structured-sweep adaptation."""
import warnings

import numpy as np

from meshwell.resolution import (
    DirectSizeSpecification,
    Graded,
    StructuredSweepResolutionSpec,
    SweepAdaptContext,
    ThresholdField,
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


class TestAdaptProtocol:
    def test_base_returns_self_and_warns(self):
        spec = ThresholdField(apply_to="curves", sizemin=0.1, sizemax=1.0, distmax=1.0)
        size_map = np.array([[0.0, 0.0, 0.0, 0.05]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = spec.adapt(size_map)
        assert out is spec
        assert any("adapt" in str(x.message) for x in w)

    def test_direct_spec_swaps_map(self):
        old = np.array([[0.0, 0.0, 0.0, 0.5]])
        new = np.array([[0.0, 0.0, 0.0, 0.1], [1.0, 0.0, 0.0, 0.2]])
        spec = DirectSizeSpecification(refinement_data=old)
        out = spec.adapt(new)
        assert out is not spec
        np.testing.assert_allclose(out.refinement_data, new)
        np.testing.assert_allclose(spec.refinement_data, old)  # original intact


class TestRefine:
    def test_sweep_refine_scalar_and_int(self):
        spec = StructuredSweepResolutionSpec(tangential=1.0, normal={"upper": 2})
        out = spec.refine(0.5)
        assert out.tangential == 0.5
        assert out.normal["upper"] == 4
        # original untouched
        assert spec.tangential == 1.0 and spec.normal["upper"] == 2

    def test_sweep_refine_graded(self):
        spec = StructuredSweepResolutionSpec(
            tangential=1.0, normal={"upper": Graded(h0=0.1, ratio=1.5)}
        )
        out = spec.refine(0.5)
        assert out.normal["upper"].h0 == 0.05
        assert out.normal["upper"].ratio == 1.5

    def test_sweep_refine_arrays(self):
        spec = StructuredSweepResolutionSpec(
            tangential=[0.0, 1.0, 2.0], normal={"upper": [0.0, 0.2, 0.4]}
        )
        out = spec.refine(0.5)
        # uniform arrays double their cell count, endpoints pinned
        np.testing.assert_allclose(out.tangential, np.linspace(0.0, 2.0, 5))
        np.testing.assert_allclose(out.normal["upper"], np.linspace(0.0, 0.4, 5))

    def test_direct_refine_scales_sizes(self):
        data = np.array([[0.0, 0.0, 0.0, 0.4], [1.0, 0.0, 0.0, 0.2]])
        out = DirectSizeSpecification(refinement_data=data).refine(0.5)
        np.testing.assert_allclose(out.refinement_data[:, 3], [0.2, 0.1])
        np.testing.assert_allclose(out.refinement_data[:, :3], data[:, :3])


def _qw_context():
    """Axis-aligned band: x in [0, 4], grown +y from y=1, thickness 0.4."""
    return SweepAdaptContext(
        thickness={"upper": 0.4},
        t_axis=0,
        n_axis=1,
        t0=0.0,
        t1=4.0,
        n0=1.0,
        normal_sign={"upper": 1.0},
    )


def _grid_size_map(size_func, nx=81, ny=41):
    """Dense (x, y, 0, size) map over the band's neighborhood [0,4]x[0.8,1.6]."""
    xs, ys = np.meshgrid(
        np.linspace(0.0, 4.0, nx), np.linspace(0.8, 1.6, ny), indexing="ij"
    )
    pts = np.column_stack([xs.ravel(), ys.ravel()])
    return np.column_stack([pts, np.zeros(len(pts)), size_func(pts[:, 0], pts[:, 1])])


class TestSweepAdapt:
    def test_context_none_returns_self(self):
        spec = StructuredSweepResolutionSpec(tangential=1.0, normal={"upper": 4})
        size_map = _grid_size_map(lambda x, y: np.full_like(x, 0.1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert spec.adapt(size_map) is spec

    def test_no_signal_returns_self(self):
        spec = StructuredSweepResolutionSpec(tangential=1.0, normal={"upper": 4})
        far = np.array([[100.0, 100.0, 0.0, 0.01]])
        assert spec.adapt(far, _qw_context()) is spec

    def test_uniform_signal_reproduces_uniform(self):
        # target 0.1 == current normal size and 1.0 == current tangential
        spec = StructuredSweepResolutionSpec(tangential=1.0, normal={"upper": 4})
        size_map = _grid_size_map(lambda x, y: np.full_like(x, 0.1))
        out = spec.adapt(size_map, _qw_context())
        off = np.asarray(out.normal["upper"])
        assert off[0] == 0.0 and off[-1] == 0.4
        np.testing.assert_allclose(np.diff(off), 0.1, rtol=1e-6)
        # tangential widened to an explicit array spanning [0, 4]
        t = np.asarray(out.tangential)
        assert t[0] == 0.0 and t[-1] == 4.0

    def test_interface_signal_refines_normal_near_interface(self):
        # fine at the attachment (eta=0), coarse away; within the damp clamp
        spec = StructuredSweepResolutionSpec(tangential=1.0, normal={"upper": 4})
        size_map = _grid_size_map(
            lambda x, y: np.clip(0.05 + 0.25 * (y - 1.0), 0.05, 0.2)
        )
        out = spec.adapt(size_map, _qw_context())
        off = np.asarray(out.normal["upper"])
        d = np.diff(off)
        assert d[0] < d[-1]  # finer near the interface
        assert off[0] == 0.0 and off[-1] == 0.4
        # gradation cap holds
        assert np.all(d[1:] / d[:-1] <= 1.3 + 1e-9)
        # int input widened to explicit array
        assert isinstance(out.normal["upper"], list)

    def test_tangential_hotspot_refines_tangentially(self):
        spec = StructuredSweepResolutionSpec(tangential=1.0, normal={"upper": 4})
        size_map = _grid_size_map(
            lambda x, y: np.clip(0.5 + 0.5 * np.abs(x - 2.0), 0.5, 1.0)
        )
        out = spec.adapt(size_map, _qw_context())
        t = np.asarray(out.tangential)
        d = np.diff(t)
        mid = (t[1:] + t[:-1]) / 2
        assert d[np.argmin(np.abs(mid - 2.0))] < d[0]  # denser near x=2

    def test_damping_clamps_pathological_target(self):
        spec = StructuredSweepResolutionSpec(tangential=1.0, normal={"upper": 4})
        size_map = _grid_size_map(lambda x, y: np.full_like(x, 1e-6))
        out = spec.adapt(size_map, _qw_context(), change_max=2.0)
        off = np.asarray(out.normal["upper"])
        # old h = 0.1; clamped target = 0.05 -> exactly 8 cells, not millions
        assert len(off) - 1 == 8

    def test_damping_iteration_converges_monotonically(self):
        spec = StructuredSweepResolutionSpec(tangential=1.0, normal={"upper": 4})
        size_map = _grid_size_map(lambda x, y: np.full_like(x, 0.0125))
        counts = []
        for _ in range(4):
            spec = spec.adapt(size_map, _qw_context(), change_max=2.0)
            counts.append(len(spec.normal["upper"]) - 1)
        assert counts == [8, 16, 32, 32]  # doubles until it hits the target

    def test_graded_input_widens(self):
        spec = StructuredSweepResolutionSpec(
            tangential=1.0, normal={"upper": Graded(h0=0.05, ratio=1.3)}
        )
        size_map = _grid_size_map(lambda x, y: np.full_like(x, 0.1))
        out = spec.adapt(size_map, _qw_context())
        assert isinstance(out.normal["upper"], list)
        off = np.asarray(out.normal["upper"])
        assert off[0] == 0.0 and off[-1] == 0.4

    def test_required_ts_retained(self):
        ctx = _qw_context()
        ctx.required_ts = (1.7,)
        spec = StructuredSweepResolutionSpec(tangential=1.0, normal={"upper": 4})
        size_map = _grid_size_map(lambda x, y: np.full_like(x, 0.1))
        out = spec.adapt(size_map, ctx)
        assert np.any(np.abs(np.asarray(out.tangential) - 1.7) < 1e-9)


from meshwell.remesh import gradation_limit_size_map


class TestSizeMapGradation:
    def test_shock_is_limited(self):
        xs = np.linspace(0.0, 10.0, 101)
        h = np.full_like(xs, 1e6)
        h[50] = 0.01  # one fine point in a sea of huge sizes
        size_map = np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs), h])
        out = gradation_limit_size_map(size_map, max_ratio=1.3)
        g = 1.3 - 1.0
        # pairwise bound against the fine point for its near neighborhood
        d = np.abs(xs - xs[50])
        near = d <= 0.5  # within the kNN propagation horizon
        assert np.all(out[near, 3] <= 0.01 + g * d[near] + 1e-9)
        # fine point itself untouched
        assert out[50, 3] == 0.01

    def test_smooth_map_untouched(self):
        xs = np.linspace(0.0, 1.0, 50)
        h = 0.1 + 0.01 * xs  # gentle slope well within the bound
        size_map = np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs), h])
        out = gradation_limit_size_map(size_map, max_ratio=1.3)
        np.testing.assert_allclose(out[:, 3], h)

    def test_input_not_mutated(self):
        size_map = np.array([[0.0, 0.0, 0.0, 1e6], [0.1, 0.0, 0.0, 0.01]])
        gradation_limit_size_map(size_map, max_ratio=1.3)
        assert size_map[0, 3] == 1e6
