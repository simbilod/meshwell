import pytest
from shapely.geometry import Point

from meshwell.polyprism import PolyPrism


@pytest.mark.skip("Validation of buffers is currently disabled!")
def test_prism_value_error():
    inner_radius = 3
    outer_radius = 5

    inner_circle = Point(0, 0).buffer(inner_radius)
    outer_circle = Point(0, 0).buffer(outer_radius)

    ring = outer_circle.difference(inner_circle)

    polygons = ring
    # Buffers that will cause a change in the topology of the polygon
    buffers = {0: 0, -1: 0, -1.001: 15, -5: 0}

    with pytest.raises(ValueError):  # noqa: PT011
        PolyPrism(polygons=polygons, buffers=buffers)


if __name__ == "__main__":
    test_prism_value_error()
