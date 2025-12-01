"""Test remesh module with multiple physical groups."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import shapely
import meshio

from meshwell.cad import cad
from meshwell.mesh import mesh
from meshwell.remesh import remesh, RemeshingStrategy
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

    # Define strategy: Refine near center (10, 10)
    def center_metric(coords, data=None):
        center = np.array([10.0, 10.0])
        x = coords[:, 0]
        y = coords[:, 1]
        distances = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        # Metric: 1.0 at center, decaying
        return np.maximum(0, 1.0 - distances / 10.0)

    strategy = RemeshingStrategy(
        func=center_metric,
        threshold=0.5,  # Refine within radius 5
        factor=0.3,
        min_size=0.1,
        max_size=2.0,
        field_smoothing_steps=5,
    )

    # Perform remeshing
    remesh(
        input_mesh=Path("test_2D_remesh_initial.msh"),
        geometry_file=Path("test_2D_remesh.xao"),
        output_mesh=Path("test_2D_remesh_output.msh"),
        strategies=[strategy],
        dim=2,
        verbosity=0,
        n_threads=1,
    )

    # Verify output file exists
    assert Path("test_2D_remesh_output.msh").exists()

    # Verify refinement happened (more points)
    final_mesh = meshio.read("test_2D_remesh_output.msh")
    assert len(final_mesh.points) > len(initial_mesh.points)

    # Clean up
    Path("test_2D_remesh.xao").unlink(missing_ok=True)
    Path("test_2D_remesh_initial.msh").unlink(missing_ok=True)
    Path("test_2D_remesh_output.msh").unlink(missing_ok=True)


def test_3D_remesh_with_physicals():
    """Test remeshing a 3D mesh with multiple physical groups."""
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

    # Strategy: Refine near center
    def center_metric_3d(coords, data=None):
        center = np.array([2.5, 2.5, 1.0])
        distances = np.linalg.norm(coords - center, axis=1)
        return np.maximum(0, 1.0 - distances / 3.0)

    strategy = RemeshingStrategy(
        func=center_metric_3d,
        threshold=0.5,
        factor=0.4,
        min_size=0.1,
        max_size=2.0,
        field_smoothing_steps=3,
    )

    # Perform remeshing
    remesh(
        input_mesh=Path("test_3D_remesh_initial.msh"),
        geometry_file=Path("test_3D_remesh.xao"),
        output_mesh=Path("test_3D_remesh_output.msh"),
        strategies=[strategy],
        dim=3,
        global_3D_algorithm=1,
        verbosity=0,
        n_threads=1,
    )

    # Verify output file exists
    assert Path("test_3D_remesh_output.msh").exists()

    final_mesh = meshio.read("test_3D_remesh_output.msh")
    assert len(final_mesh.points) > len(initial_mesh.points)

    print("3D remeshing completed successfully")

    # Clean up
    Path("test_3D_remesh.xao").unlink(missing_ok=True)
    Path("test_3D_remesh_initial.msh").unlink(missing_ok=True)
    Path("test_3D_remesh_output.msh").unlink(missing_ok=True)


def test_remesh_radial_pattern():
    """Test remeshing with a clear radial refinement pattern."""
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

    # Generate COARSE uniform initial mesh
    initial_mesh = mesh(
        dim=2,
        input_file="test_radial_remesh.xao",
        output_file="test_radial_remesh_initial.msh",
        resolution_specs={
            "domain": [ConstantInField(apply_to="surfaces", resolution=2.0)],
        },
        n_threads=1,
        default_characteristic_length=100,
    )

    initial_points = len(initial_mesh.points)
    print(f"Initial coarse mesh: {initial_points} points")

    # Strategy: Radial refinement
    def radial_metric(coords, data=None):
        center = np.array([large_rect / 2, large_rect / 2])
        x = coords[:, 0]
        y = coords[:, 1]
        distances = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        # High metric at center
        return np.maximum(0, 1.0 - distances / 5.0)

    strategy = RemeshingStrategy(
        func=radial_metric,
        threshold=0.2,  # Refine broadly
        factor=0.1,  # Strong refinement
        min_size=0.1,
        max_size=2.0,
        field_smoothing_steps=3,
    )

    # Perform remeshing
    remesh(
        input_mesh=Path("test_radial_remesh_initial.msh"),
        geometry_file=Path("test_radial_remesh.xao"),
        output_mesh=Path("test_radial_remesh_output.msh"),
        strategies=[strategy],
        dim=2,
        verbosity=0,
        n_threads=1,
    )

    # Verify output exists and has more points
    assert Path("test_radial_remesh_output.msh").exists()

    remeshed = meshio.read("test_radial_remesh_output.msh")
    remeshed_points = len(remeshed.points)

    print(f"Remeshed: {remeshed_points} points")
    print(f"Refinement achieved: {remeshed_points / initial_points:.2f}x more points")

    assert (
        remeshed_points > initial_points
    ), f"Remeshing should refine! Got {remeshed_points} vs initial {initial_points}"

    # Clean up
    Path("test_radial_remesh.xao").unlink(missing_ok=True)
    Path("test_radial_remesh_initial.msh").unlink(missing_ok=True)
    Path("test_radial_remesh_output.msh").unlink(missing_ok=True)


def test_multi_strategy_refinement():
    """Test combining multiple strategies (circle and line)."""
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
        output_file="test_multi_strategy.xao",
    )

    # Generate coarse initial mesh
    initial_mesh = mesh(
        dim=2,
        input_file="test_multi_strategy.xao",
        output_file="test_multi_strategy_initial.msh",
        resolution_specs={
            "domain": [ConstantInField(apply_to="surfaces", resolution=2.0)],
        },
        n_threads=1,
        default_characteristic_length=100,
    )

    initial_points = len(initial_mesh.points)
    print(f"Initial mesh: {initial_points} points")

    # Strategy 1: Circle refinement
    circle_center = np.array([3.0, 3.0])
    circle_radius = 1.5

    def circle_metric(coords, data=None):
        if data is not None:
            return data
        x = coords[:, 0]
        y = coords[:, 1]
        dist_from_center = np.sqrt(
            (x - circle_center[0]) ** 2 + (y - circle_center[1]) ** 2
        )
        dist_from_boundary = np.abs(dist_from_center - circle_radius)
        return np.maximum(0, 1.0 - dist_from_boundary / 0.5)

    # Generate grid for circle
    x = np.linspace(0, large_rect, 40)
    y = np.linspace(0, large_rect, 40)
    X, Y = np.meshgrid(x, y)
    circle_coords = np.column_stack([X.ravel(), Y.ravel(), np.zeros_like(X.ravel())])
    circle_data = circle_metric(circle_coords)
    circle_refinement = np.column_stack([circle_coords, circle_data])

    circle_strategy = RemeshingStrategy(
        func=circle_metric,
        threshold=0.6,
        factor=0.2,
        refinement_data=circle_refinement,
        min_size=0.15,
        max_size=2.0,
        field_smoothing_steps=3,
    )

    # Strategy 2: Vertical line refinement
    line_x = 7.0

    def line_metric(coords, data=None):
        if data is not None:
            return data
        x = coords[:, 0]
        dist_from_line = np.abs(x - line_x)
        return np.maximum(0, 1.0 - dist_from_line / 0.3)

    # Generate grid for line
    line_coords = np.column_stack([X.ravel(), Y.ravel(), np.zeros_like(X.ravel())])
    line_data = line_metric(line_coords)
    line_refinement = np.column_stack([line_coords, line_data])

    line_strategy = RemeshingStrategy(
        func=line_metric,
        threshold=0.5,
        factor=0.25,
        refinement_data=line_refinement,
        min_size=0.2,
        max_size=2.0,
        field_smoothing_steps=3,
    )

    # Perform remeshing with both strategies
    remesh(
        input_mesh=Path("test_multi_strategy_initial.msh"),
        geometry_file=Path("test_multi_strategy.xao"),
        output_mesh=Path("test_multi_strategy_output.msh"),
        strategies=[circle_strategy, line_strategy],
        dim=2,
        verbosity=0,
        n_threads=1,
    )

    # Verify output exists and has more points
    assert Path("test_multi_strategy_output.msh").exists()

    remeshed = meshio.read("test_multi_strategy_output.msh")
    remeshed_points = len(remeshed.points)

    print(f"Multi-strategy remeshed: {remeshed_points} points")
    print(f"Refinement achieved: {remeshed_points / initial_points:.2f}x more points")

    assert (
        remeshed_points > initial_points
    ), f"Multi-strategy remeshing should refine! Got {remeshed_points} vs initial {initial_points}"

    # Clean up
    Path("test_multi_strategy.xao").unlink(missing_ok=True)
    Path("test_multi_strategy_initial.msh").unlink(missing_ok=True)
    Path("test_multi_strategy_output.msh").unlink(missing_ok=True)


if __name__ == "__main__":
    test_2D_remesh_with_physicals()
    test_3D_remesh_with_physicals()
    test_remesh_radial_pattern()
    test_multi_strategy_refinement()
