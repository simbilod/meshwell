"""Tests for meshwell.structured.spec dataclasses + validators."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from meshwell.structured.spec import StructuredExtrusionResolutionSpec


def test_spec_minimal_valid():
    spec = StructuredExtrusionResolutionSpec(n_layers=[3, 5])
    assert spec.n_layers == [3, 5]
    assert spec.recombine is False


def test_spec_recombine_true():
    spec = StructuredExtrusionResolutionSpec(n_layers=[2], recombine=True)
    assert spec.recombine is True


def test_spec_rejects_empty_n_layers():
    with pytest.raises(ValidationError, match="n_layers"):
        StructuredExtrusionResolutionSpec(n_layers=[])


def test_spec_rejects_non_positive_layer_count():
    with pytest.raises(ValidationError, match="positive"):
        StructuredExtrusionResolutionSpec(n_layers=[3, 0, 4])

    with pytest.raises(ValidationError, match="positive"):
        StructuredExtrusionResolutionSpec(n_layers=[-1])


def test_spec_equality_on_identical_fields():
    """Two specs with identical fields compare equal (pydantic value semantics)."""
    a = StructuredExtrusionResolutionSpec(n_layers=[2])
    b = StructuredExtrusionResolutionSpec(n_layers=[2])
    assert a == b


def test_structured_partition_convergence_error_is_runtime_error():
    """The new convergence error must be a RuntimeError subclass and exportable."""
    from meshwell.structured import StructuredPartitionConvergenceError

    assert issubclass(StructuredPartitionConvergenceError, RuntimeError)
    err = StructuredPartitionConvergenceError("did not converge")
    assert "did not converge" in str(err)
