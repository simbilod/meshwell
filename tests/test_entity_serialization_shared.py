"""Shared serialization helpers + consistent physical-name normalization."""
from shapely.geometry import LineString, Polygon

from meshwell.polyline import PolyLine
from meshwell.polyprism import PolyPrism
from meshwell.polysurface import PolySurface
from meshwell.validation import format_physical_name

_SQ = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


def test_list_physical_name_normalized_to_tuple_everywhere():
    ps = PolySurface(polygons=_SQ, physical_name=["a", "b"])
    pl = PolyLine(linestrings=LineString([(0, 0), (1, 1)]), physical_name=["a", "b"])
    pp = PolyPrism(polygons=_SQ, buffers={0.0: 0.0, 1.0: 0.0}, physical_name=["a", "b"])
    assert ps.physical_name == pl.physical_name == pp.physical_name == ("a", "b")


def test_round_trips_unchanged():
    for ent in [
        PolySurface(polygons=_SQ, physical_name="s", mesh_order=2),
        PolyLine(linestrings=LineString([(0, 0), (1, 1)]), physical_name="l"),
        PolyPrism(polygons=_SQ, buffers={0.0: 0.0, 1.0: 0.1}, physical_name="p"),
    ]:
        d = ent.to_dict()
        assert type(ent).from_dict(d).to_dict() == d


def test_format_physical_name_normalizes_list_to_tuple():
    assert format_physical_name(["a", "b"]) == ("a", "b")


def test_format_physical_name_leaves_tuple_and_str_and_none_unchanged():
    assert format_physical_name("a") == ("a",)
    assert format_physical_name(("a", "b")) == ("a", "b")
    assert format_physical_name(None) is None
