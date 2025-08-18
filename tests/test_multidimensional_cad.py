"""
Test multi-dimensional entity processing capabilities.

This test module validates the ability to process entities of different
dimensions (3D volumes, 2D surfaces, 1D curves, 0D points) in a single model
using fragment-based integration.
"""

from __future__ import annotations

import gmsh
from functools import partial
from meshwell.cad import cad
from meshwell.gmsh_entity import GMSH_entity
from meshwell.mesh import mesh
from meshwell.resolution import ConstantInField


def test_volume_with_internal_surface():
    """Test 3D volume with 2D surface fragment integration."""

    # Create a 3D box (volume)
    box_obj = GMSH_entity(
        gmsh_partial_function=partial(
            gmsh.model.occ.add_box,
            x=0,
            y=0,
            z=0,
            dx=2,
            dy=2,
            dz=1,
        ),
        physical_name="box",
        mesh_order=1,
    )

    # Create a 2D surface inside the box
    surface_obj = GMSH_entity(
        gmsh_partial_function=partial(
            gmsh.model.occ.add_rectangle,
            x=0.5,
            y=0.5,
            z=0.5,  # Inside the box
            dx=1,
            dy=1,
        ),
        physical_name="internal_surface",
        mesh_order=2,
    )

    # Process with multi-dimensional capability
    entities = [box_obj, surface_obj]
    cad(
        entities_list=entities,
        output_file="test_volume_with_surface.xao",
        progress_bars=True,
    )
    mesh(
        dim=3,
        input_file="test_volume_with_surface.xao",
        output_file="test_volume_with_surface.msh",
        default_characteristic_length=1,
    )


def test_lower_dim_with_multiple_higher_entities():
    """Test lower-dimension object interacting with multiple higher-level entities."""

    # Create two 3D boxes that don't overlap
    box1_obj = GMSH_entity(
        gmsh_partial_function=partial(
            gmsh.model.occ.add_box,
            x=0,
            y=0,
            z=0,
            dx=1,
            dy=1,
            dz=1,
        ),
        physical_name="box1",
        mesh_order=1,
    )

    box2_obj = GMSH_entity(
        gmsh_partial_function=partial(
            gmsh.model.occ.add_box,
            x=1.5,
            y=0,
            z=0,
            dx=1,
            dy=1,
            dz=1,
        ),
        physical_name="box2",
        mesh_order=2,
    )

    # Create a 2D surface that spans across both boxes
    spanning_surface = GMSH_entity(
        gmsh_partial_function=partial(
            gmsh.model.occ.add_rectangle,
            x=0.5,
            y=0.5,
            z=0.5,  # Through middle of both boxes
            dx=1.5,  # Spans across both
            dy=0.5,
        ),
        physical_name="spanning_surface",
        mesh_order=3,
    )

    # The surface should fragment against both boxes individually
    entities = [box1_obj, box2_obj, spanning_surface]
    cad(
        entities_list=entities,
        output_file="test_lower_dim_multiple_higher.xao",
        progress_bars=True,
    )
    mesh(
        dim=3,
        input_file="test_lower_dim_multiple_higher.xao",
        output_file="test_lower_dim_multiple_higher.msh",
        default_characteristic_length=1,
        resolution_specs={
            "spanning_surface": [ConstantInField(apply_to="surfaces", resolution=0.1)],
        },
    )


def test_surface_boundary_overlap():
    """Test surface that lies exactly on boundary of volume."""

    # Create a 3D box
    box_obj = GMSH_entity(
        gmsh_partial_function=partial(
            gmsh.model.occ.add_box,
            x=0,
            y=0,
            z=0,
            dx=2,
            dy=2,
            dz=1,
        ),
        physical_name="box",
        mesh_order=1,
    )

    # Create a surface that lies exactly on the top face of the box
    boundary_surface = GMSH_entity(
        gmsh_partial_function=partial(
            gmsh.model.occ.add_rectangle,
            x=0.5,
            y=0.5,
            z=1.0,  # Exactly on top face boundary
            dx=1,
            dy=1,
        ),
        physical_name="boundary_surface",
        mesh_order=2,
    )

    entities = [box_obj, boundary_surface]
    cad(
        entities_list=entities,
        output_file="test_boundary_overlap.xao",
        progress_bars=True,
    )
    mesh(
        dim=3,
        input_file="test_boundary_overlap.xao",
        output_file="test_boundary_overlap.msh",
        default_characteristic_length=1,
    )


def test_point_in_multiple_entities():
    """Test 0D point inside multiple overlapping entities."""

    # Create overlapping 3D entities
    box_obj = GMSH_entity(
        gmsh_partial_function=partial(
            gmsh.model.occ.add_box,
            x=0,
            y=0,
            z=0,
            dx=2,
            dy=2,
            dz=2,
        ),
        physical_name="box",
        mesh_order=1,
    )

    sphere_obj = GMSH_entity(
        gmsh_partial_function=partial(
            gmsh.model.occ.add_sphere,
            xc=1,
            yc=1,
            zc=1,
            radius=1.2,
        ),
        physical_name="sphere",
        mesh_order=2,
    )

    # Create a 2D surface in the overlap region
    surface_obj = GMSH_entity(
        gmsh_partial_function=partial(
            gmsh.model.occ.add_rectangle,
            x=0.5,
            y=0.5,
            z=1,
            dx=1,
            dy=1,
        ),
        physical_name="surface",
        mesh_order=3,
    )

    # Create a 0D point in the overlap region
    point_obj = GMSH_entity(
        gmsh_partial_function=partial(
            gmsh.model.occ.add_point,
            x=1,
            y=1,
            z=1,
        ),
        physical_name="central_point",
        mesh_order=4,
    )

    entities = [box_obj, sphere_obj, surface_obj, point_obj]
    cad(
        entities_list=entities,
        output_file="test_point_multiple_entities.xao",
        progress_bars=True,
    )
    mesh(
        dim=3,
        input_file="test_point_multiple_entities.xao",
        output_file="test_point_multiple_entities.msh",
        default_characteristic_length=1,
    )


def test_sequential_fragmentation_complex():
    """Test sequential fragmentation with multiple lower-dim entities."""

    # Base 3D volume
    base_volume = GMSH_entity(
        gmsh_partial_function=partial(
            gmsh.model.occ.add_box,
            x=0,
            y=0,
            z=0,
            dx=3,
            dy=3,
            dz=2,
        ),
        physical_name="base_volume",
        mesh_order=1,
    )

    # First 2D surface
    surface1 = GMSH_entity(
        gmsh_partial_function=partial(
            gmsh.model.occ.add_rectangle,
            x=0.5,
            y=0.5,
            z=1,
            dx=2,
            dy=2,
        ),
        physical_name="surface1",
        mesh_order=2,
    )

    # Second 2D surface that will fragment against modified volume
    surface2 = GMSH_entity(
        gmsh_partial_function=partial(
            gmsh.model.occ.add_rectangle,
            x=1,
            y=1,
            z=0.5,
            dx=1.5,
            dy=1.5,
        ),
        physical_name="surface2",
        mesh_order=3,
    )

    # 1D curve that intersects with fragmented entities
    curve = GMSH_entity(
        gmsh_partial_function=partial(
            gmsh.model.occ.add_circle,
            x=1.5,
            y=1.5,
            z=1.2,
            r=0.8,
        ),
        physical_name="curve",
        mesh_order=4,
    )

    # Multiple 0D points
    point1 = GMSH_entity(
        gmsh_partial_function=partial(
            gmsh.model.occ.add_point,
            x=1.2,
            y=1.2,
            z=1.2,
        ),
        physical_name="point1",
        mesh_order=5,
    )

    point2 = GMSH_entity(
        gmsh_partial_function=partial(
            gmsh.model.occ.add_point,
            x=2,
            y=2,
            z=0.5,
        ),
        physical_name="point2",
        mesh_order=6,
    )

    entities = [base_volume, surface1, surface2, curve, point1, point2]
    cad(
        entities_list=entities,
        output_file="test_sequential_complex.xao",
        progress_bars=True,
    )
    mesh(
        dim=3,
        input_file="test_sequential_complex.xao",
        output_file="test_sequential_complex.msh",
        default_characteristic_length=1,
    )


if __name__ == "__main__":
    # test_volume_with_internal_surface()
    test_lower_dim_with_multiple_higher_entities()
    # test_surface_boundary_overlap()
    # test_point_in_multiple_entities()
    # test_sequential_fragmentation_complex()
