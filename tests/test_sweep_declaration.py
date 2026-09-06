import pytest

from meshwell.structured.exceptions import SweepKeyError
from meshwell.structured.sweep import StructuredSweep


def test_interface_keys():
    s = StructuredSweep(name="qw", on="a___b", thickness={"a": 0.1, "b": 0.2})
    assert s.attachment_kind == "interface"
    assert set(s.sides()) == {"a", "b"}
    with pytest.raises(SweepKeyError):
        StructuredSweep(name="qw", on="a___b", thickness={"c": 0.1})


def test_boundary_keys():
    s = StructuredSweep(name="bot", on="a___None", thickness={"a": 0.1})
    assert s.attachment_kind == "boundary"
    with pytest.raises(SweepKeyError):
        StructuredSweep(name="bot", on="a___None", thickness={"a": 0.1, "None": 0.1})


def test_polyline_keys():
    s = StructuredSweep(name="jn", on="junction_line", thickness={"left": 0.1})
    assert s.attachment_kind == "polyline"
    with pytest.raises(SweepKeyError):
        StructuredSweep(name="jn", on="junction_line", thickness={"a": 0.1})


def test_name_validation():
    with pytest.raises(ValueError):
        StructuredSweep(name="bad|name", on="a___b", thickness={"a": 0.1})
    with pytest.raises(ValueError):
        StructuredSweep(name="qw", on="a___b", thickness={"a": -0.1})


def test_roundtrip_serialization():
    s = StructuredSweep(name="qw", on="a___b", thickness={"a": 0.1})
    d = s.to_dict()
    assert d["type"] == "StructuredSweep"
    s2 = StructuredSweep.from_dict(d)
    assert s2.name == s.name and s2.on == s.on and s2.thickness == s.thickness
