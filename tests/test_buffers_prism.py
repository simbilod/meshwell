import pytest
from shapely.geometry import Point
from meshwell.model import Model
from meshwell.prism import Prism


def test_prism_value_error():
    inner_radius = 3
    outer_radius = 5

    inner_circle = Point(0, 0).buffer(inner_radius)
    outer_circle = Point(0, 0).buffer(outer_radius)

    ring = outer_circle.difference(inner_circle)

    polygons = ring
    # Buffers that will cause a change in the topology of the polygon
    buffers = {0: 0, -1: 0, -1.001: 10, -5: 0}

    model = Model(n_threads=1)

    with pytest.raises(ValueError):
        Prism(polygons=polygons, buffers=buffers, model=model)
