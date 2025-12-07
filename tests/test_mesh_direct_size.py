import numpy as np
from pathlib import Path
import meshio
import shapely
from meshwell.mesh import mesh
from meshwell.resolution import DirectSizeSpecification
from meshwell.polysurface import PolySurface
from meshwell.cad import cad


def test_mesh_direct_size_specification_global():
    """Test DirectSizeSpecification applied globally via mesh()."""
    # 1. Create geometry
    polygon = shapely.Polygon(
        [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
    )
    poly_obj = PolySurface(polygons=polygon, mesh_order=1, physical_name="domain")
    cad(entities_list=[poly_obj], output_file="test_mesh_direct.xao")

    # 2. Create refinement data (N, 4)
    # Gradient in X direction
    x = np.linspace(0, 10, 20)
    y = np.linspace(0, 10, 20)
    X, Y = np.meshgrid(x, y)
    coords = np.column_stack([X.ravel(), Y.ravel(), np.zeros_like(X.ravel())])

    # Size = 0.1 at x=0, 1.0 at x=10
    sizes = 0.1 + (coords[:, 0] / 10.0) * 0.9

    refinement_data = np.column_stack([coords, sizes])

    spec = DirectSizeSpecification(
        refinement_data=refinement_data, min_size=0.1, max_size=1.0
    )

    # 3. Mesh with global spec
    mesh(
        dim=2,
        input_file="test_mesh_direct.xao",
        output_file="test_mesh_direct.msh",
        default_characteristic_length=2.0,
        resolution_specs={None: [spec]},  # Global application
        n_threads=1,
    )

    # 4. Verify
    final_mesh = meshio.read("test_mesh_direct.msh")
    points = final_mesh.points

    # Check sizes at x=0 vs x=10
    edges = set()
    if "triangle" in final_mesh.cells_dict:
        for elem in final_mesh.cells_dict["triangle"]:
            edges.add(tuple(sorted((elem[0], elem[1]))))
            edges.add(tuple(sorted((elem[1], elem[2]))))
            edges.add(tuple(sorted((elem[2], elem[0]))))

    left_lengths = []
    right_lengths = []

    for n1, n2 in edges:
        p1 = points[n1]
        p2 = points[n2]
        length = np.linalg.norm(p1 - p2)
        center_x = (p1[0] + p2[0]) / 2.0

        if center_x < 1.0:
            left_lengths.append(length)
        elif center_x > 9.0:
            right_lengths.append(length)

    mean_left = np.mean(left_lengths)
    mean_right = np.mean(right_lengths)

    print(f"Mean left length (target ~0.1): {mean_left}")
    print(f"Mean right length (target ~1.0): {mean_right}")

    assert mean_left < 0.3
    assert mean_right > 0.5
    assert mean_left < mean_right

    # Clean up
    Path("test_mesh_direct.xao").unlink(missing_ok=True)
    Path("test_mesh_direct.msh").unlink(missing_ok=True)
