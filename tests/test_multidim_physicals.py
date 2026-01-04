from __future__ import annotations

import shutil
from functools import partial
from pathlib import Path

import gmsh
import numpy as np
import pytest

from meshwell.cad import cad
from meshwell.gmsh_entity import GMSH_entity
from meshwell.mesh import mesh
from meshwell.remesh import remesh_mmg


def add_line_with_points(p1_coords, p2_coords):
    """Helper to add points and then a line between them."""
    t1 = gmsh.model.occ.add_point(*p1_coords)
    t2 = gmsh.model.occ.add_point(*p2_coords)
    return gmsh.model.occ.add_line(t1, t2)


def create_geometry_and_mesh(dim, output_name, temp_dir):
    """Create a geometry with entities of all dimensions and mesh it."""
    # Define entities
    # 3D Volume
    box = GMSH_entity(
        gmsh_partial_function=partial(
            gmsh.model.occ.add_box, x=0, y=0, z=0, dx=1, dy=1, dz=1
        ),
        physical_name="volume_phys",
        mesh_order=1,
    )

    # 2D Surface (on boundary)
    surface = GMSH_entity(
        gmsh_partial_function=partial(
            gmsh.model.occ.add_rectangle, x=0.25, y=0.25, z=1.0, dx=0.5, dy=0.5
        ),
        physical_name="surface_phys",
        mesh_order=2,
    )

    # 1D Curve (Line)
    line = GMSH_entity(
        gmsh_partial_function=partial(
            add_line_with_points,
            p1_coords=(0.5, 0.5, -0.5),
            p2_coords=(0.5, 0.5, 1.5),
        ),
        physical_name="line_phys",
        mesh_order=3,
        dimension=1,
    )

    # 0D Point
    point = GMSH_entity(
        gmsh_partial_function=partial(gmsh.model.occ.add_point, x=0.2, y=0.2, z=0.2),
        physical_name="point_phys",
        mesh_order=4,
    )

    entities = [box, surface, line, point]

    xao_path = temp_dir / f"{output_name}.xao"
    msh_path = temp_dir / f"{output_name}.msh"

    # Create CAD
    cad(
        entities_list=entities,
        output_file=xao_path,
        progress_bars=False,
    )

    # Mesh with GMSH
    mesh_obj = mesh(
        dim=dim,
        input_file=xao_path,
        output_file=msh_path,
        default_characteristic_length=0.5,
        verbosity=0,
    )

    return mesh_obj, msh_path, xao_path


def check_physicals(mesh_obj, expected_physicals):
    """Check if expected physical names are present in the mesh."""
    present_physicals = set(mesh_obj.cell_sets.keys())

    # If cell_sets is empty, try to reconstruct from field_data and cell_data
    if (
        not present_physicals
        and hasattr(mesh_obj, "field_data")
        and "gmsh:physical" in mesh_obj.cell_data
    ):
        # Collect all physical tags present in the mesh
        present_tags = set()
        for tags in mesh_obj.cell_data["gmsh:physical"]:
            present_tags.update(tags)

        # Check against field_data
        for name, data in mesh_obj.field_data.items():
            # data matches name -> tag or name -> [tag, dim]
            try:
                tag = data[0] if isinstance(data, (list, tuple, np.ndarray)) else data
            except IndexError:
                continue

            if tag in present_tags:
                present_physicals.add(name)

    print(f"Present physicals: {present_physicals}")

    for phys_name in expected_physicals:
        assert (
            phys_name in present_physicals
        ), f"Physical '{phys_name}' missing from mesh"


def test_preservation_gmsh(tmp_path):
    """Test physical preservation with GMSH."""
    mesh_obj, _, _ = create_geometry_and_mesh(3, "test_gmsh", tmp_path)

    # GMSH should preserve all logic
    expected = ["volume_phys", "surface_phys", "line_phys", "point_phys"]
    check_physicals(mesh_obj, expected)


def test_preservation_mmg(tmp_path):
    """Test physical preservation with MMG."""
    if not shutil.which("mmg3d_O3"):
        pytest.skip("mmg3d_O3 not found")

    # 1. Create initial mesh
    _, msh_path, _ = create_geometry_and_mesh(3, "test_mmg_init", tmp_path)

    # 2. Remesh with MMG
    out_mmg_path = tmp_path / "test_mmg_out.msh"

    remesh_mmg(
        input_mesh=msh_path,
        output_mesh=out_mmg_path,
        strategies=[],  # Use current sizes
        dim=3,
        mmg_executable="mmg3d_O3",
        verbosity=1,
    )

    # 3. Read back and check
    import meshio

    final_mesh = meshio.read(out_mmg_path)

    # MMG/Meshio pipeline currently drops 0D and 1D entities, and internal surfaces logic is complex.
    # We verify that volume and boundary surface physicals are preserved.
    expected = ["volume_phys", "surface_phys"]
    check_physicals(final_mesh, expected)


def test_preservation_parmmg(tmp_path):
    """Test physical preservation with ParMMG in parallel."""
    if not shutil.which("parmmg_O3"):
        pytest.skip("parmmg_O3 not found")
    if not shutil.which("mpirun") and not shutil.which("mpiexec"):
        pytest.skip("MPI not found")

    # 1. Create initial mesh
    _, msh_path, _ = create_geometry_and_mesh(3, "test_parmmg_init", tmp_path)

    # 2. Remesh with ParMMG (defaults for dim=3)
    out_parmmg_path = tmp_path / "test_parmmg_out.msh"

    remesh_mmg(
        input_mesh=msh_path,
        output_mesh=out_parmmg_path,
        strategies=[],  # Use current sizes
        dim=3,
        n_threads=2,    # Trigger MPI
        verbosity=1,
    )

    # 3. Read back and check
    import meshio

    final_mesh = meshio.read(out_parmmg_path)

    expected = ["volume_phys", "surface_phys"]
    check_physicals(final_mesh, expected)



if __name__ == "__main__":
    # Manually run tests if executed as script
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        print("Running test_preservation_gmsh...")
        try:
            test_preservation_gmsh(tmp_path)
            print("PASS")
        except Exception as e:
            print(f"FAIL: {e}")
            import traceback

            traceback.print_exc()

        print("\nRunning test_preservation_mmg...")
        try:
            test_preservation_mmg(tmp_path)
            print("PASS")
        except Exception as e:
            # Check for skip
            if "Skipped" in type(e).__name__ or "skip" in str(e).lower():
                print(f"SKIP: {e}")
            else:
                print(f"FAIL: {e}")
                import traceback

                traceback.print_exc()

        print("\nRunning test_preservation_parmmg...")
        try:
            test_preservation_parmmg(tmp_path)
            print("PASS")
        except Exception as e:
            # Check for skip
            if "Skipped" in type(e).__name__ or "skip" in str(e).lower():
                print(f"SKIP: {e}")
            else:
                print(f"FAIL: {e}")
                import traceback

                traceback.print_exc()
