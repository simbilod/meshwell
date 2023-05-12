from __future__ import annotations

import gmsh
import shapely
from meshwell.prism import Prism
from meshwell.model import Model


def test_prism():
    polygon1 = shapely.Polygon(
        [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
        holes=([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]],),
    )
    polygon2 = shapely.Polygon([[-1, -1], [-2, -1], [-2, -2], [-1, -2], [-1, -1]])
    polygon = shapely.MultiPolygon([polygon1, polygon2])

    buffers = {0.0: 0.0, 0.3: 0.1, 1.0: -0.2}

    model = Model()

    Prism(polygons=polygon, buffers=buffers, model=model)
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(3)
