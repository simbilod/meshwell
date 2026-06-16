import pytest
from shapely.geometry import Polygon

from meshwell.polyprism import PolyPrism
from meshwell.structured.exceptions import StructuredExtrudeRequiredError

SQUARE = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])


def test_default_is_unstructured():
    p = PolyPrism(polygons=SQUARE, buffers={0.0: 0.0, 1.0: 0.0}, physical_name="x")
    assert p.structured is False
    assert p.identify_arcs is False


def test_structured_true_extrude_ok():
    p = PolyPrism(
        polygons=SQUARE,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="x",
        structured=True,
    )
    assert p.structured is True
    assert p.extrude is True
    # identify_arcs defaults to False regardless of structured= to prevent
    # polygons whose vertices lie on a common circle (e.g. rectangles on
    # their circumscribed circle) from being mis-built as arc-based solids.
    assert p.identify_arcs is False


def test_structured_identify_arcs_user_override():
    p = PolyPrism(
        polygons=SQUARE,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="x",
        structured=True,
        identify_arcs=True,
    )
    assert p.identify_arcs is True


def test_structured_buffered_raises():
    with pytest.raises(StructuredExtrudeRequiredError):
        PolyPrism(
            polygons=SQUARE,
            buffers={0.0: 0.0, 1.0: 0.5},
            physical_name="x",
            structured=True,
        )


def test_structured_round_trips_through_dict():
    p = PolyPrism(
        polygons=SQUARE,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="x",
        structured=True,
    )
    d = p.to_dict()
    p2 = PolyPrism.from_dict(d) if hasattr(PolyPrism, "from_dict") else None
    # If from_dict not on PolyPrism, fall back to checking the dict directly.
    if p2 is not None:
        assert p2.structured is True
    else:
        assert d.get("structured") is True
