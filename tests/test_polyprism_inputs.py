"""PolyPrism input normalization."""
import pytest
from shapely.geometry import MultiPolygon, Polygon

from meshwell.polyprism import PolyPrism

_SQ1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
_SQ2 = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])


def test_polygons_normalized_to_multipolygon():
    inputs = [
        (_SQ1, 1),
        ([_SQ1, _SQ2], 2),
        (MultiPolygon([_SQ1, _SQ2]), 2),
        ([MultiPolygon([_SQ1, _SQ2])], 2),
    ]
    for inp, expected_count in inputs:
        prism = PolyPrism(polygons=inp, buffers={0.0: 0.0, 1.0: 0.0}, physical_name="x")
        assert isinstance(prism.polygons, MultiPolygon)
        assert len(prism.polygons.geoms) == expected_count


def test_list_input_with_nonzero_buffers_builds_buffered_polygons():
    # crashed before: list has no .geoms, so .buffer() was called on a list
    prism = PolyPrism(
        polygons=[_SQ1, _SQ2], buffers={0.0: 0.0, 1.0: 0.1}, physical_name="x"
    )
    assert len(prism.buffered_polygons) == 2
    assert all(len(entry) == 2 for entry in prism.buffered_polygons)


def test_serialization_round_trip_two_polygons():
    prism = PolyPrism(
        polygons=[_SQ1, _SQ2], buffers={0.0: 0.0, 1.0: 0.1}, physical_name="x"
    )
    clone = PolyPrism.from_dict(prism.to_dict())
    assert len(clone.polygons.geoms) == 2
    assert clone.to_dict() == prism.to_dict()


def test_empty_buffers_raises_value_error():
    with pytest.raises(ValueError, match="buffers"):
        PolyPrism(polygons=_SQ1, buffers={}, physical_name="x")
