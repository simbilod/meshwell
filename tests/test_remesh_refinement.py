"""Tests for the refinement module."""
import numpy as np
from pathlib import Path
import meshio
from meshwell.remesh import RemeshingStrategy, remesh
from meshwell.cad import cad
from meshwell.mesh import mesh
from meshwell.polysurface import PolySurface
import shapely


def test_remeshing_strategy_dataclass():
    """Test RemeshingStrategy dataclass initialization."""

    def dummy_func(coords, data):
        return np.zeros(len(coords))

    strategy = RemeshingStrategy(
        func=dummy_func, threshold=0.5, factor=0.5, min_size=0.1, max_size=1.0
    )

    assert strategy.threshold == 0.5
    assert strategy.factor == 0.5
    assert strategy.min_size == 0.1


def test_quantitative_refinement():
    """Test that remeshing actually reduces edge lengths in target regions."""
    # 1. Create a simple geometry (square)
    polygon = shapely.Polygon(
        [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
    )
    poly_obj = PolySurface(polygons=polygon, mesh_order=1, physical_name="domain")

    cad(entities_list=[poly_obj], output_file="test_quant.xao")

    # 2. Create coarse initial mesh
    _initial_mesh = mesh(
        dim=2,
        input_file="test_quant.xao",
        output_file="test_quant_initial.msh",
        default_characteristic_length=2.0,
        n_threads=1,
    )

    # 3. Define strategy: Refine x < 5.0
    def region_metric(coords, data=None):
        # Return 1.0 if x < 5.0, else 0.0
        return (coords[:, 0] < 5.0).astype(float)

    strategy = RemeshingStrategy(
        func=region_metric,
        threshold=0.5,
        factor=0.2,  # Should reduce size to ~0.4
        min_size=0.1,
        max_size=2.0,
    )

    # 4. Remesh
    remesh(
        input_mesh=Path("test_quant_initial.msh"),
        geometry_file=Path("test_quant.xao"),
        output_mesh=Path("test_quant_final.msh"),
        strategies=[strategy],
        dim=2,
        n_threads=1,
    )

    # 5. Verify
    final_mesh = meshio.read("test_quant_final.msh")
    points = final_mesh.points

    # Extract edges
    edges = set()
    if "triangle" in final_mesh.cells_dict:
        for elem in final_mesh.cells_dict["triangle"]:
            edges.add(tuple(sorted((elem[0], elem[1]))))
            edges.add(tuple(sorted((elem[1], elem[2]))))
            edges.add(tuple(sorted((elem[2], elem[0]))))

    # Check edge lengths in refined region vs unrefined
    refined_lengths = []
    unrefined_lengths = []

    for n1, n2 in edges:
        p1 = points[n1]
        p2 = points[n2]
        length = np.linalg.norm(p1 - p2)
        center_x = (p1[0] + p2[0]) / 2.0

        if center_x < 4.5:  # Deep inside refined region
            refined_lengths.append(length)
        elif center_x > 5.5:  # Deep inside unrefined region
            unrefined_lengths.append(length)

    mean_refined = np.mean(refined_lengths)
    mean_unrefined = np.mean(unrefined_lengths)

    print(f"Mean refined length: {mean_refined}")
    print(f"Mean unrefined length: {mean_unrefined}")

    # Refined should be significantly smaller
    assert mean_refined < mean_unrefined * 0.5
    # Target size was 2.0 * 0.2 = 0.4.
    assert mean_refined < 0.6  # Allow some slack

    # Clean up
    Path("test_quant.xao").unlink(missing_ok=True)
    Path("test_quant_initial.msh").unlink(missing_ok=True)
    Path("test_quant_final.msh").unlink(missing_ok=True)
