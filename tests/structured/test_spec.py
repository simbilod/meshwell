"""Tests for meshwell.structured.spec dataclasses + validators."""
from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_spec_minimal_valid():
    from meshwell.structured.spec import StructuredExtrusionResolutionSpec

    spec = StructuredExtrusionResolutionSpec(n_layers=[3, 5])
    assert spec.n_layers == [3, 5]
    assert spec.recombine is False


def test_spec_recombine_true():
    from meshwell.structured.spec import StructuredExtrusionResolutionSpec

    spec = StructuredExtrusionResolutionSpec(n_layers=[2], recombine=True)
    assert spec.recombine is True


def test_spec_rejects_empty_n_layers():
    from meshwell.structured.spec import StructuredExtrusionResolutionSpec

    with pytest.raises(ValidationError, match="n_layers"):
        StructuredExtrusionResolutionSpec(n_layers=[])


def test_spec_rejects_non_positive_layer_count():
    from meshwell.structured.spec import StructuredExtrusionResolutionSpec

    with pytest.raises(ValidationError, match="positive"):
        StructuredExtrusionResolutionSpec(n_layers=[3, 0, 4])

    with pytest.raises(ValidationError, match="positive"):
        StructuredExtrusionResolutionSpec(n_layers=[-1])


def test_spec_is_hashable():
    """Frozen pydantic models are hashable; we use them as dict keys downstream."""
    from meshwell.structured.spec import StructuredExtrusionResolutionSpec

    a = StructuredExtrusionResolutionSpec(n_layers=[2])
    b = StructuredExtrusionResolutionSpec(n_layers=[2])
    # Pydantic equality on identical fields.
    assert a == b
