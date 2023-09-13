from __future__ import annotations

import shapely
from meshwell.prism import Prism
from meshwell.polysurface import PolySurface
from meshwell.model import Model
from meshwell.gmsh_entity import GMSH_entity


def test_mesh_3D():
    polygon1 = shapely.Polygon(
        [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
        holes=([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]],),
    )
    polygon2 = shapely.Polygon([[-1, -1], [-2, -1], [-2, -2], [-1, -2], [-1, -1]])
    polygon = shapely.MultiPolygon([polygon1, polygon2])

    buffers = {0.0: 0.0, 1.0: -0.1}

    model = Model()
    poly3D = Prism(
        polygons=polygon,
        buffers=buffers,
        model=model,
        physical_name="first_entity",
        mesh_order=1,
    )

    gmsh_entity = GMSH_entity(
        gmsh_function=model.occ.addSphere,
        gmsh_function_kwargs={"xc": 0, "yc": 0, "zc": 0, "radius": 1},
        dimension=3,
        model=model,
        physical_name="second_entity",
        mesh_order=2,
    )

    entities_list = [poly3D, gmsh_entity]

    model.mesh(
        input_entities_list=entities_list,
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
    poly2D = PolySurface(
        polygons=polygon, model=model, physical_name="first_entity", mesh_order=1
    )

    gmsh_entity = GMSH_entity(
        gmsh_function=model.occ.add_rectangle,
        gmsh_function_kwargs={"x": 3, "y": 3, "z": 0, "dx": 1, "dy": 1},
        dimension=2,
        model=model,
        physical_name="second_entity",
        mesh_order=2,
    )

    entities_list = [poly2D, gmsh_entity]

    model.mesh(
        input_entities_list=entities_list,
        default_characteristic_length=0.5,
        verbosity=False,
        filename="mesh2D.msh",
    )

    pass


def test_mesh_ND():
    z = 1
    polygon1 = shapely.Polygon(
        [[0, 0, z], [2, 0, z], [2, 2, z], [0, 2, z], [0, 0, z]],
        holes=(
            [[0.5, 0.5, z], [1.5, 0.5, z], [1.5, 1.5, z], [0.5, 1.5, z], [0.5, 0.5, z]],
        ),
    )
    polygon2 = shapely.Polygon(
        [[-1, -1, z], [-2, -1, z], [-2, -2, z], [-1, -2, z], [-1, -1, z]]
    )
    polygon = shapely.MultiPolygon([polygon1, polygon2])

    model = Model()
    poly2D = PolySurface(
        polygons=polygon, model=model, physical_name="polygon", mesh_order=1
    )

    box = GMSH_entity(
        gmsh_function=model.occ.add_box,
        gmsh_function_kwargs={"x": 0, "y": 0, "z": 0, "dx": 3, "dy": 3, "dz": 3},
        dimension=3,
        model=model,
        physical_name="box",
        mesh_order=2,
    )

    entities_list = [poly2D, box]

    model.mesh(
        input_entities_list=entities_list,
        default_characteristic_length=0.5,
        verbosity=False,
        filename="meshND.msh",
    )

    pass


if __name__ == "__main__":
    test_mesh_3D()
    test_mesh_2D()
    test_mesh_ND()
