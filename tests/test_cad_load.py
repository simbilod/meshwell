from __future__ import annotations

import pytest
import shapely
from pathlib import Path
from meshwell.model import Model
from meshwell.prism import Prism
from meshwell.gmsh_entity import GMSH_entity
from meshwell.resolution import ConstantInField

from meshwell.utils import compare_mesh_headers


def test_load_cad_model():
    # Create initial model with some geometry
    initial_model = Model(n_threads=1)

    # Create a prism from polygons
    polygon1 = shapely.Polygon(
        [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
        holes=([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]],),
    )
    polygon2 = shapely.Polygon([[-1, -1], [-2, -1], [-2, -2], [-1, -2], [-1, -1]])
    polygon = shapely.MultiPolygon([polygon1, polygon2])

    buffers = {0.0: 0.0, 1.0: -0.1}

    prism = Prism(
        polygons=polygon,
        buffers=buffers,
        model=initial_model,
        physical_name="prism_entity",
        mesh_order=1,
        resolutions=[
            ConstantInField(resolution=0.5, apply_to="volumes"),
        ],
    )

    # Create a sphere
    sphere = GMSH_entity(
        gmsh_function=initial_model.occ.addSphere,
        gmsh_function_kwargs={"xc": 0, "yc": 0, "zc": 0, "radius": 1},
        dimension=3,
        model=initial_model,
        physical_name="sphere_entity",
        mesh_order=2,
        resolutions=[
            ConstantInField(resolution=0.3, apply_to="volumes"),
        ],
    )

    entities_list = [prism, sphere]

    # Mesh directly for baseline
    initial_model.mesh(entities_list=entities_list, filename="test.msh")

    # Generate initial CAD and save to file
    initial_model.cad(entities_list=entities_list, filename="test.xao")

    # Create new model to test loading
    load_identical_model = Model("test_load_model_identical", n_threads=1)

    load_identical_model.mesh(
        entities_list=entities_list,
        cad_filename="test.xao",
        from_cad=True,
        filename="test_identical.msh",
    )

    compare_mesh_headers(
        meshfile=Path("test.msh"), other_meshfile=Path("test_identical.msh")
    )

    # Create new model to test loading w/ modified entities
    new_model = Model("test_load_model_modified", n_threads=1)

    # Create new entities with different mesh parameters
    new_prism = Prism(
        polygons=polygon,
        buffers=buffers,
        model=new_model,
        physical_name="prism_entity",
        mesh_order=1,
        resolutions=[
            ConstantInField(resolution=0.2, apply_to="volumes"),  # Different resolution
        ],
    )

    new_sphere = GMSH_entity(
        gmsh_function=new_model.occ.addSphere,
        gmsh_function_kwargs={"xc": 0, "yc": 0, "zc": 0, "radius": 1},
        dimension=3,
        model=new_model,
        physical_name="sphere_entity",
        mesh_order=2,
        resolutions=[
            ConstantInField(
                resolution=0.15, apply_to="volumes"
            ),  # Different resolution
        ],
    )

    new_entities_list = [new_prism, new_sphere]

    # Create new model to test loading
    load_modified_model = Model("test_load_model_modified", n_threads=1)

    load_modified_model.mesh(
        entities_list=new_entities_list,
        cad_filename="test.xao",
        from_cad=True,
        filename="test_modified.msh",
    )

    compare_mesh_headers(
        meshfile=Path("test.msh"), other_meshfile=Path("test_modified.msh")
    )

    # Test loading CAD model
    modified_entities, max_dim = new_model._load_cad_model(
        filename="test.xao", entities_list=new_entities_list
    )

    # Verify results
    assert len(modified_entities) == 2
    assert max_dim == 3

    # Check entities maintained correct properties
    for entity in modified_entities:
        if entity.physical_name == "prism_entity":
            assert entity.resolutions[0].resolution == 0.2  # New resolution
            assert entity.mesh_order == 1
        elif entity.physical_name == "sphere_entity":
            assert entity.resolutions[0].resolution == 0.15  # New resolution
            assert entity.mesh_order == 2


def test_load_cad_model_with_additive_entities(tmp_path):
    """Test that loading fails when additive entities are present"""
    model = Model(n_threads=1)

    # Create base geometry
    sphere = GMSH_entity(
        gmsh_function=model.occ.addSphere,
        gmsh_function_kwargs={"xc": 0, "yc": 0, "zc": 0, "radius": 1},
        dimension=3,
        model=model,
        physical_name="base_sphere",
        mesh_order=1,
    )

    # Create additive geometry
    additive_sphere = GMSH_entity(
        gmsh_function=model.occ.addSphere,
        gmsh_function_kwargs={"xc": 0.5, "yc": 0, "zc": 0, "radius": 0.5},
        dimension=3,
        model=model,
        physical_name="additive_sphere",
        mesh_order=1,
        additive=True,  # Make this an additive entity
    )

    xao_file = tmp_path / "test_additive.xao"
    entities_list = [sphere, additive_sphere]

    # Should raise ValueError due to additive entities
    with pytest.raises(
        ValueError,
        match="Meshing from a loaded CAD file currently does not support additive entities",
    ):
        model._load_cad_model(filename=xao_file, entities_list=entities_list)


if __name__ == "__main__":
    test_load_cad_model()
