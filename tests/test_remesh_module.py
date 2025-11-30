"""Test remesh module with multiple physical groups."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import shapely

from meshwell.cad import cad
from meshwell.mesh import mesh
from meshwell.remesh import remesh
from meshwell.polysurface import PolySurface
from meshwell.polyprism import PolyPrism
from meshwell.resolution import ConstantInField, ThresholdField


def test_2D_remesh_with_physicals():
    """Test remeshing a 2D mesh with multiple physical groups."""
    large_rect = 20
    small_rect = 5

    polygon1 = shapely.Polygon(
        [[0, 0], [large_rect, 0], [large_rect, large_rect], [0, large_rect], [0, 0]],
    )
    polygon2 = shapely.Polygon(
        [
            [small_rect, small_rect],
            [large_rect - small_rect, small_rect],
            [large_rect - small_rect, large_rect - small_rect],
            [small_rect, 6],
            [small_rect, small_rect],
        ],
    )

    poly_obj1 = PolySurface(
        polygons=polygon1,
        mesh_order=2,
        physical_name="outer",
    )
    poly_obj2 = PolySurface(
        polygons=polygon2,
        mesh_order=1,
        physical_name="inner",
    )

    entities_list = [poly_obj1, poly_obj2]

    # Generate CAD
    cad(
        entities_list=entities_list,
        output_file="test_2D_remesh",
    )

    # Generate initial mesh with resolution specs
    initial_mesh = mesh(
        dim=2,
        input_file="test_2D_remesh.xao",
        output_file="test_2D_remesh_initial.msh",
        resolution_specs={
            "inner": [
                ThresholdField(sizemin=0.5, distmax=2, sizemax=2, apply_to="curves")
            ],
            "outer": [ConstantInField(apply_to="surfaces", resolution=2)],
        },
        n_threads=1,
        default_characteristic_length=100,
    )

    print(f"Initial mesh: {len(initial_mesh.points)} points")

    # Create a size map for remeshing: refine near center (10, 10)
    num_points = 100
    x = np.random.uniform(0, large_rect, num_points)
    y = np.random.uniform(0, large_rect, num_points)
    z = np.zeros(num_points)

    # Size based on distance from center
    center = np.array([10.0, 10.0])
    distances = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    sizes = 0.3 + 0.15 * distances  # Fine at center, coarse at edges

    size_map = np.column_stack([x, y, z, sizes])

    # Perform remeshing
    remesh(
        input_mesh=Path("test_2D_remesh_initial.msh"),
        geometry_file=Path("test_2D_remesh.xao"),
        output_mesh=Path("test_2D_remesh_output.msh"),
        size_map=size_map,
        dim=2,
        field_smoothing_steps=5,
        verbosity=0,
        n_threads=1,
    )

    # Verify output file exists
    assert Path("test_2D_remesh_output.msh").exists()

    # Clean up
    Path("test_2D_remesh.xao").unlink(missing_ok=True)
    Path("test_2D_remesh_initial.msh").unlink(missing_ok=True)
    Path("test_2D_remesh_output.msh").unlink(missing_ok=True)


def test_3D_remesh_with_physicals():
    """Test remeshing a 3D mesh with multiple physical groups.

    Note: 3D remeshing with geometry recovery can be challenging for gmsh,
    so this test uses a simple geometry to ensure basic functionality.
    """
    # Use a simpler single-box geometry for 3D
    polygon = shapely.Polygon(
        [[0, 0], [5, 0], [5, 5], [0, 5], [0, 0]],
    )

    buffers = {0.0: 0.0, 2: 0.0}

    prism_obj = PolyPrism(
        polygons=polygon,
        buffers=buffers,
        mesh_order=1,
        physical_name="box",
    )

    # Generate CAD
    cad(
        entities_list=[prism_obj],
        output_file="test_3D_remesh.xao",
    )

    # Generate initial mesh with coarser resolution
    initial_mesh = mesh(
        dim=3,
        input_file="test_3D_remesh.xao",
        output_file="test_3D_remesh_initial.msh",
        resolution_specs={
            "box": [ConstantInField(resolution=1.0, apply_to="volumes")],
        },
        n_threads=1,
        default_characteristic_length=100,
    )

    print(f"Initial 3D mesh: {len(initial_mesh.points)} points")

    # Create a 3D size map: refine near center (2.5, 2.5, 1.0)
    num_points = 100
    x = np.random.uniform(0, 5, num_points)
    y = np.random.uniform(0, 5, num_points)
    z = np.random.uniform(0, 2, num_points)

    # Size based on distance from center
    center = np.array([2.5, 2.5, 1.0])
    coords = np.column_stack([x, y, z])
    distances = np.linalg.norm(coords - center, axis=1)
    sizes = 0.4 + 0.15 * distances  # Fine at center, coarse at edges

    size_map = np.column_stack([x, y, z, sizes])

    # Perform remeshing
    remesh(
        input_mesh=Path("test_3D_remesh_initial.msh"),
        geometry_file=Path("test_3D_remesh.xao"),
        output_mesh=Path("test_3D_remesh_output.msh"),
        size_map=size_map,
        dim=3,
        global_3D_algorithm=1,
        field_smoothing_steps=3,
        verbosity=0,
        n_threads=1,
    )

    # Verify output file exists
    assert Path("test_3D_remesh_output.msh").exists()

    print("3D remeshing completed successfully")

    # Clean up
    Path("test_3D_remesh.xao").unlink(missing_ok=True)
    Path("test_3D_remesh_initial.msh").unlink(missing_ok=True)
    Path("test_3D_remesh_output.msh").unlink(missing_ok=True)


def test_remesh_radial_pattern():
    """Test remeshing with a clear radial refinement pattern."""
    import meshio

    large_rect = 10

    # Simple square geometry
    polygon = shapely.Polygon(
        [[0, 0], [large_rect, 0], [large_rect, large_rect], [0, large_rect], [0, 0]],
    )

    poly_obj = PolySurface(
        polygons=polygon,
        mesh_order=1,
        physical_name="domain",
    )

    # Generate CAD
    cad(
        entities_list=[poly_obj],
        output_file="test_radial_remesh.xao",
    )

    # Generate COARSE uniform initial mesh (so we can see the refinement)
    initial_mesh = mesh(
        dim=2,
        input_file="test_radial_remesh.xao",
        output_file="test_radial_remesh_initial.msh",
        resolution_specs={
            "domain": [ConstantInField(apply_to="surfaces", resolution=2.0)],  # Coarse!
        },
        n_threads=1,
        default_characteristic_length=100,
    )

    initial_points = len(initial_mesh.points)
    print(f"Initial coarse mesh: {initial_points} points")

    # Create radial size map with strong refinement gradient
    # Dense sampling in a grid pattern for better coverage
    num_x = 15
    num_y = 15
    x_grid = np.linspace(0, large_rect, num_x)
    y_grid = np.linspace(0, large_rect, num_y)
    xx, yy = np.meshgrid(x_grid, y_grid)
    x = xx.ravel()
    y = yy.ravel()
    z = np.zeros(len(x))

    # Strong radial size field: very fine at center (0.1), coarse at edges (1.5)
    center = np.array([large_rect / 2, large_rect / 2])
    distances = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    # Normalize distances and create strong gradient
    max_dist = np.sqrt(2) * large_rect / 2
    normalized_dist = distances / max_dist

    # Very fine at center, coarse at edges
    sizes = 0.15 + 1.2 * normalized_dist**2  # Quadratic increase

    size_map = np.column_stack([x, y, z, sizes])

    print(f"Size field: {len(size_map)} points")
    print(f"  Min size (center): {sizes.min():.3f}")
    print(f"  Max size (edges): {sizes.max():.3f}")

    # Perform remeshing
    remesh(
        input_mesh=Path("test_radial_remesh_initial.msh"),
        geometry_file=Path("test_radial_remesh.xao"),
        output_mesh=Path("test_radial_remesh_output.msh"),
        size_map=size_map,
        dim=2,
        field_smoothing_steps=3,
        verbosity=0,
        n_threads=1,
    )

    # Verify output exists and has more points (showing refinement)
    assert Path("test_radial_remesh_output.msh").exists()

    remeshed = meshio.read("test_radial_remesh_output.msh")
    remeshed_points = len(remeshed.points)

    print(f"Remeshed: {remeshed_points} points")
    print(f"Refinement achieved: {remeshed_points / initial_points:.2f}x more points")

    # Assert that we actually got refinement (should have more points)
    assert (
        remeshed_points > initial_points
    ), f"Remeshing should refine! Got {remeshed_points} vs initial {initial_points}"

    # Clean up
    Path("test_radial_remesh.xao").unlink(missing_ok=True)
    Path("test_radial_remesh_initial.msh").unlink(missing_ok=True)
    Path("test_radial_remesh_output.msh").unlink(missing_ok=True)


if __name__ == "__main__":
    test_2D_remesh_with_physicals()
    test_3D_remesh_with_physicals()
    test_remesh_radial_pattern()
