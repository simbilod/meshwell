import pytest

from meshwell.resolution import StructuredExtrusionResolutionSpec


def test_default_n_layers_is_one():
    s = StructuredExtrusionResolutionSpec()
    assert s.n_layers == 1
    assert s.apply_to == "volumes"


def test_explicit_n_layers():
    s = StructuredExtrusionResolutionSpec(n_layers=4)
    assert s.n_layers == 4


def test_invalid_n_layers_raises():
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        StructuredExtrusionResolutionSpec(n_layers=0)


def test_to_dict_round_trip():
    s = StructuredExtrusionResolutionSpec(n_layers=3)
    d = s.to_dict()
    assert d["n_layers"] == 3
    assert d["resolution_type"] == "StructuredExtrusionResolutionSpec"
