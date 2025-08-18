from __future__ import annotations

import shapely
from meshwell.polysurface import PolySurface
from meshwell.cad import cad
from meshwell.mesh import mesh


def test_polysurface():
    polygon1 = shapely.Polygon(
        [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
        holes=([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]],),
    )
    polygon2 = shapely.Polygon([[-1, -1], [-2, -1], [-2, -2], [-1, -2], [-1, -1]])
    polygon = shapely.MultiPolygon([polygon1, polygon2])

    polysurface_obj = PolySurface(polygons=polygon, physical_name="polysurface")
    cad(entities_list=[polysurface_obj], output_file="test_polysurface")
    mesh(
        input_file="test_polysurface.xao",
        output_file="test_polysurface.msh",
        dim=2,
        default_characteristic_length=0.5,
        n_threads=1,
    )
