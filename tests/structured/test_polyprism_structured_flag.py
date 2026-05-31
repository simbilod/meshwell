import pytest
from shapely.geometry import Polygon

from meshwell.polyprism import PolyPrism
from meshwell.structured.exceptions import StructuredExtrudeRequiredError

SQUARE = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])


def test_default_is_unstructured():
    p = PolyPrism(polygons=SQUARE, buffers={0.0: 0.0, 1.0: 0.0}, physical_name="x")
    assert p.structured is False


def test_structured_true_extrude_ok():
    p = PolyPrism(
        polygons=SQUARE,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="x",
        structured=True,
    )
    assert p.structured is True
    assert p.extrude is True
    assert p.identify_arcs is True  # default flips when structured


def test_structured_identify_arcs_user_override():
    p = PolyPrism(
        polygons=SQUARE,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="x",
        structured=True,
        identify_arcs=False,
    )
    assert p.identify_arcs is False


def test_structured_buffered_raises():
    with pytest.raises(StructuredExtrudeRequiredError):
        PolyPrism(
            polygons=SQUARE,
            buffers={0.0: 0.0, 1.0: 0.5},
            physical_name="x",
            structured=True,
        )
