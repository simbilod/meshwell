from __future__ import annotations

import gmsh
import shapely
from meshwell.prism import Prism
from meshwell.model import Model
from collections import OrderedDict


def test_mesh_3D():
    polygon1 = shapely.Polygon(
        [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
        holes=([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]],),
    )
    polygon2 = shapely.Polygon([[-1, -1], [-2, -1], [-2, -2], [-1, -2], [-1, -1]])
    polygon = shapely.MultiPolygon([polygon1, polygon2])

    buffers = {0.0: 0.0, 1.0: -0.1}

    model = Model()
    poly3D = Prism(polygons=polygon, buffers=buffers, model=model)

    dimtags_dict = OrderedDict(
        {
            "first_physical": [(3, poly3D)],
            "second_entity": [(3, gmsh.model.occ.addSphere(0, 0, 0, 1))],
        }
    )

    resolutions = {
        "first_physical": {"resolution": 0.3},
    }

    model.mesh(
        dimtags_dict=dimtags_dict,
        resolutions=resolutions,
        default_characteristic_length=0.5,
        verbosity=False,
        filename="mesh2D.msh",
    )

    pass


def test_mesh_2D():
    polygon1 = shapely.Polygon(
        [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
        holes=([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]],),
    )
    polygon2 = shapely.Polygon([[-1, -1], [-2, -1], [-2, -2], [-1, -2], [-1, -1]])
    polygon = shapely.MultiPolygon([polygon1, polygon2])

    buffers = {0.0: 0.0, 1.0: -0.1}

    model = Model()
    poly3D = Prism(polygons=polygon, buffers=buffers, model=model)

    dimtags_dict = OrderedDict(
        {
            "first_physical": [(3, poly3D)],
            "second_entity": [(3, gmsh.model.occ.addSphere(0, 0, 0, 1))],
        }
    )

    resolutions = {
        "first_physical": {"resolution": 0.3},
    }

    model.mesh(
        dimtags_dict=dimtags_dict,
        resolutions=resolutions,
        default_characteristic_length=0.5,
        verbosity=False,
        filename="mesh3D.msh",
    )

    pass
