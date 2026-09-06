import numpy as np
import pytest

from meshwell.resolution import (
    Graded,
    StructuredExtrusionResolutionSpec,
    StructuredSweepResolutionSpec,
    resolve_normal_offsets,
)
from meshwell.structured.exceptions import StructuredError, SweepNormalExtentError


def test_resolve_int_uniform():
    out = resolve_normal_offsets(4, thickness=1.0, atol=1e-9)
    np.testing.assert_allclose(out, [0.0, 0.25, 0.5, 0.75, 1.0])


def test_resolve_graded_fills_thickness():
    out = resolve_normal_offsets(Graded(h0=0.1, ratio=2.0), thickness=1.0, atol=1e-9)
    assert out[0] == 0.0
    assert out[-1] == pytest.approx(1.0)
    assert np.all(np.diff(out) > 0)
    # first cell is exactly h0
    assert out[1] == pytest.approx(0.1)


def test_resolve_graded_ratio_one_is_uniform():
    out = resolve_normal_offsets(Graded(h0=0.25, ratio=1.0), thickness=1.0, atol=1e-9)
    np.testing.assert_allclose(out, [0.0, 0.25, 0.5, 0.75, 1.0])


def test_resolve_explicit_array_validated():
    out = resolve_normal_offsets([0.0, 0.2, 1.0], thickness=1.0, atol=1e-9)
    np.testing.assert_allclose(out, [0.0, 0.2, 1.0])
    with pytest.raises(SweepNormalExtentError):
        resolve_normal_offsets([0.0, 0.2, 0.9], thickness=1.0, atol=1e-9)
    with pytest.raises(SweepNormalExtentError):
        resolve_normal_offsets([0.1, 0.2, 1.0], thickness=1.0, atol=1e-9)


def test_sweep_spec_fields_and_noop_apply():
    spec = StructuredSweepResolutionSpec(
        tangential=0.05,
        normal={"well_1": Graded(h0=1e-3, ratio=1.3), "sch": 3},
        element_type="quad",
    )
    assert spec.apply() is None  # no-op


def test_graded_validation():
    with pytest.raises(Exception):
        Graded(h0=-1.0, ratio=1.3)
    with pytest.raises(Exception):
        Graded(h0=1e-3, ratio=0.5)


def test_sweep_normal_extent_error_is_structured_error():
    assert issubclass(SweepNormalExtentError, StructuredError)


def test_resolve_int_below_one_raises():
    with pytest.raises(SweepNormalExtentError):
        resolve_normal_offsets(0, thickness=1.0, atol=1e-9)


def test_extrusion_spec_alias_unchanged():
    spec = StructuredExtrusionResolutionSpec(n_layers=3)
    assert spec.n_layers == 3
    assert spec.apply() is None
    # subclass relationship lets shared machinery treat both uniformly
    assert isinstance(spec, StructuredSweepResolutionSpec)
