from __future__ import annotations

import gmsh
import shapely
from meshwell.prism import Prism
from meshwell.cad import cad, CAD
import numpy as np


def test_prism():
    polygon1 = shapely.Polygon(
        [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
        holes=([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]],),
    )
    polygon2 = shapely.Polygon([[-1, -1], [-2, -1], [-2, -2], [-1, -2], [-1, -1]])
    polygon = shapely.MultiPolygon([polygon1, polygon2])

    buffers = {0.0: 0.0, 0.3: 0.1, 1.0: -0.2}

    prism_obj = Prism(polygons=polygon, buffers=buffers, physical_name="prism")
    assert prism_obj.extrude is False
    cad(entities_list=[prism_obj], output_file="test_prism")


def test_prism_extruded():
    polygon1 = shapely.Polygon(
        [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
        holes=([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]],),
    )
    polygon2 = shapely.Polygon([[-1, -1], [-2, -1], [-2, -2], [-1, -2], [-1, -1]])
    polygon = shapely.MultiPolygon([polygon1, polygon2])

    buffers = {-1.0: 0.0, 1.0: 0.0}

    prism_obj = Prism(polygons=polygon, buffers=buffers)

    cad_processor = CAD()
    cad_processor._initialize_model()
    entity_dimtags = prism_obj.instanciate(cad_processor)
    assert prism_obj.extrude is True
    dim = entity_dimtags[0][0]
    tag = entity_dimtags[0][1]
    _, _, zmin, _, _, zmax = gmsh.model.getBoundingBox(dim, tag[0])
    assert np.isclose(zmin, min(buffers.keys()))
    assert np.isclose(zmax, max(buffers.keys()))


if __name__ == "__main__":
    test_prism()
