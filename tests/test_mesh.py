from __future__ import annotations

import shapely
from meshwell.prism import Prism
from meshwell.polysurface import PolySurface
from meshwell.model import Model
from meshwell.gmsh_entity import GMSH_entity
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

    gmsh_entity = GMSH_entity(
        gmsh_function=model.occ.addSphere,
        gmsh_function_kwargs={"xc": 0, "yc": 0, "zc": 0, "radius": 1},
        dim=3,
        model=model,
    )

    entities_dict = OrderedDict(
        {
            "first_entity": poly3D,
            "second_entity": gmsh_entity,
        }
    )

    resolutions = {
        "first_entity": {"resolution": 0.3},
    }

    model.mesh(
        entities_dict=entities_dict,
        resolutions=resolutions,
        default_characteristic_length=0.5,
        verbosity=False,
        filename="mesh3D.msh",
    )

    pass


def test_mesh_2D():
    polygon1 = shapely.Polygon(
        [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
        holes=([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]],),
    )
    polygon2 = shapely.Polygon([[-1, -1], [-2, -1], [-2, -2], [-1, -2], [-1, -1]])
    polygon = shapely.MultiPolygon([polygon1, polygon2])

    model = Model()
    poly2D = PolySurface(polygons=polygon, model=model)

    gmsh_entity = GMSH_entity(
        gmsh_function=model.occ.add_rectangle,
        gmsh_function_kwargs={"x": 3, "y": 3, "z": 0, "dx": 1, "dy": 1},
        dim=2,
        model=model,
    )

    entities_dict = OrderedDict(
        {
            "first_entity": poly2D,
            "second_entity": gmsh_entity,
        }
    )

    resolutions = {
        "first_entity": {"resolution": 0.3},
    }

    model.mesh(
        entities_dict=entities_dict,
        resolutions=resolutions,
        default_characteristic_length=0.5,
        verbosity=False,
        filename="mesh2D.msh",
    )

    pass


if __name__ == "__main__":
    test_mesh_3D()
    test_mesh_2D()
