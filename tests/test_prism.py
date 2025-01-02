from __future__ import annotations

import gmsh
import shapely
from meshwell.prism import Prism
from meshwell.model import Model
import numpy as np


def test_prism():
    polygon1 = shapely.Polygon(
        [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
        holes=([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]],),
    )
    polygon2 = shapely.Polygon([[-1, -1], [-2, -1], [-2, -2], [-1, -2], [-1, -1]])
    polygon = shapely.MultiPolygon([polygon1, polygon2])

    buffers = {0.0: 0.0, 0.3: 0.1, 1.0: -0.2}

    model = Model(n_threads=1)
    model._initialize_model()

    prism_obj = Prism(polygons=polygon, buffers=buffers, model=model)
    prism_obj.instanciate()
    assert prism_obj.extrude is False

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write("mesh_prism.msh")


def test_prism_extruded():
    polygon1 = shapely.Polygon(
        [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
        holes=([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]],),
    )
    polygon2 = shapely.Polygon([[-1, -1], [-2, -1], [-2, -2], [-1, -2], [-1, -1]])
    polygon = shapely.MultiPolygon([polygon1, polygon2])

    buffers = {-1.0: 0.0, 1.0: 0.0}

    model = Model(n_threads=1)
    model._initialize_model()

    prism_obj = Prism(polygons=polygon, buffers=buffers, model=model)
    dim, tags = prism_obj.instanciate()[0]
    assert prism_obj.extrude is True

    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tags[0])
    assert np.isclose(zmin, min(buffers.keys()))
    assert np.isclose(zmax, max(buffers.keys()))

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write("mesh_extruded.msh")


if __name__ == "__main__":
    test_prism_extruded()
    # test_prism()
