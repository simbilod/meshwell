from __future__ import annotations

import shapely
from meshwell.polysurface import PolySurface
from meshwell.model import Model
from meshwell.gmsh_entity import GMSH_entity


def test_resolution():
    polygon1 = shapely.Polygon(
        [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
        holes=([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]],),
    )
    polygon2 = shapely.Polygon([[-2, -2], [-3, -2], [-3, -3], [-2, -3], [-2, -2]])
    polygon = shapely.MultiPolygon([polygon1, polygon2])

    num_vertices = []

    resolution1 = [
        None,
        {
            "resolution": 0.2,
        },
        {
            "resolution": 0.2,
        },
    ]

    resolution2 = [
        None,
        {
            "resolution": 0.2,
        },
        {
            "resolution": 0.2,
            "DistMax": 1,
            "DistMin": 0.5,
            "SizeMax": 0.5,
            "SizeMin": 0.05,
        },
    ]

    for i in range(len(resolution1)):
        model = Model()
        poly2D = PolySurface(
            polygons=polygon,
            model=model,
            physical_name="first_entity",
            mesh_order=1,
            resolution=resolution1[i],
        )

        gmsh_entity = GMSH_entity(
            gmsh_function=model.occ.add_rectangle,
            gmsh_function_kwargs={"x": 3, "y": 3, "z": 0, "dx": 1, "dy": 1},
            dimension=2,
            model=model,
            physical_name="second_entity",
            mesh_order=2,
            resolution=resolution2[i],
        )

        background = GMSH_entity(
            gmsh_function=model.occ.add_rectangle,
            gmsh_function_kwargs={"x": -5, "y": -5, "z": 0, "dx": 10, "dy": 10},
            dimension=2,
            model=model,
            physical_name="background",
            mesh_order=3,
        )

        entities_list = [poly2D, gmsh_entity, background]

        mesh = model.mesh(
            entities_list=entities_list,
            default_characteristic_length=0.5,
            verbosity=0,
            filename=f"mesh_{i}.msh",
        )

        num_vertices.append(mesh.points.shape[0])

    assert num_vertices[0] < num_vertices[1] < num_vertices[2]


if __name__ == "__main__":
    test_resolution()
