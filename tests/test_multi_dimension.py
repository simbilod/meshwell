from meshwell.model import Model
from meshwell.gmsh_entity import GMSH_entity


def test_multi_dimension_entities():
    model = Model(n_threads=1, filename="test_multi_dimension_entities")

    box = GMSH_entity(
        gmsh_function=model.occ.addBox,
        gmsh_function_kwargs={"x": 0, "y": 0, "z": 0, "dx": 1, "dy": 1, "dz": 1},
        dimension=3,
        model=model,
        physical_name="box3D",
        mesh_order=2,
    )

    plane = GMSH_entity(
        gmsh_function=model.occ.addRectangle,
        gmsh_function_kwargs={"x": 0.25, "y": 0.25, "z": 0.25, "dx": 0.5, "dy": 0.5},
        dimension=2,
        model=model,
        physical_name="rect2D",
        mesh_order=1,
    )

    box2 = GMSH_entity(
        gmsh_function=model.occ.addBox,
        gmsh_function_kwargs={"x": 1, "y": 0, "z": 0, "dx": 1, "dy": 1, "dz": 1},
        dimension=3,
        model=model,
        physical_name="box3D_2",
        mesh_order=3,
    )

    plane2 = GMSH_entity(
        gmsh_function=model.occ.addRectangle,
        gmsh_function_kwargs={"x": 0.5, "y": 0.5, "z": 0.75, "dx": 1, "dy": 0.5},
        dimension=2,
        model=model,
        physical_name="rect2D_2",
        mesh_order=2,
    )

    model.mesh(entities_list=[box, plane, box2, plane2])


if __name__ == "__main__":
    test_multi_dimension_entities()
