from __future__ import annotations

import gmsh
import shapely
from meshwell.prism import Prism
from collections import OrderedDict

from meshwell.mesh import mesh


def test_mesh_3D():
    polygon1 = shapely.Polygon(
        [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
        holes=([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]],),
    )
    polygon2 = shapely.Polygon([[-1, -1], [-2, -1], [-2, -2], [-1, -2], [-1, -1]])
    polygon = shapely.MultiPolygon([polygon1, polygon2])

    buffers = {0.0: 0.0, 1.0: -0.1}

    gmsh.initialize()
    occ = gmsh.model.occ
    poly3D = Prism(polygons=polygon, buffers=buffers, model=occ)

    dimtags_dict = OrderedDict(
        {
            "first_physical": [(3, poly3D)],
            "second_entity": [(3, occ.addSphere(0, 0, 0, 1))],
        }
    )

    resolutions = {
        "first_physical": {"resolution": 0.3},
    }

    mesh(
        dimtags_dict=dimtags_dict,
        model=occ,
        resolutions=resolutions,
        default_characteristic_length=0.5,
        verbosity=False,
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

    gmsh.initialize()
    occ = gmsh.model.occ
    poly3D = Prism(polygons=polygon, buffers=buffers, model=occ)

    dimtags_dict = OrderedDict(
        {
            "first_physical": [(3, poly3D)],
            "second_entity": [(3, occ.addSphere(0, 0, 0, 1))],
        }
    )

    resolutions = {
        "first_physical": {"resolution": 0.3},
    }

    mesh(
        dimtags_dict=dimtags_dict,
        model=occ,
        resolutions=resolutions,
        default_characteristic_length=0.5,
        verbosity=False,
    )

    pass
