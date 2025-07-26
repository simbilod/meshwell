#!/usr/bin/env python3
"""Test for gmsh-based adaptive mesh refinement using meshwell-generated mesh.

This example demonstrates the complete workflow:
1. Create geometry and initial mesh using meshwell
2. Apply adaptive refinement using our new gmsh_adaptive_remesh module
"""

import numpy as np
import sys
from pathlib import Path
import shapely

from meshwell.polysurface import PolySurface
from meshwell.cad import cad
from meshwell.mesh import mesh
from meshwell.gmsh_adaptive_remesh import gmsh_adaptive_remesh


def test_function(xyz):
    """Test function with sharp gradient - same as gmsh_adapt_mesh example."""
    a = 6 * (np.hypot(xyz[..., 0] - 0.5, xyz[..., 1] - 0.5) - 0.2)
    f = np.real(np.arctanh(a + 0j))
    return f


def create_meshwell_geometry_and_mesh(lc=0.05):
    """Create geometry and initial mesh using meshwell."""
    print("Creating geometry using meshwell...")

    # Create a simple square geometry using shapely
    square1 = shapely.box(0, 0, 1, 1)
    square2 = shapely.box(0.5, 0.5, 1, 1)

    # Create PolySurface entity
    polysurface1 = PolySurface(
        polygons=square1,
        physical_name="square1",
        mesh_order=2,
    )
    polysurface2 = PolySurface(
        polygons=square2,
        physical_name="square2",
        mesh_order=1,
    )

    # Generate CAD file
    cad_file = Path("meshwell_test_geometry.xao")
    cad(entities_list=[polysurface1, polysurface2], output_file=str(cad_file))

    # Generate initial mesh
    mesh_file = Path("meshwell_test_initial.msh")
    initial_mesh = mesh(
        input_cad_file=str(cad_file),
        output_mesh_file=str(mesh_file),
        dim=2,
        default_characteristic_length=lc,
        verbosity=0,
    )

    return str(cad_file), str(mesh_file), initial_mesh


def main():
    """Main test function demonstrating meshwell + adaptive refinement workflow."""
    print("Testing gmsh-based adaptive mesh refinement with meshwell...")

    # Parse command line arguments
    lc = 0.02
    target_elements = 10000

    argv = sys.argv
    if "-nopopup" in sys.argv:
        argv.remove("-nopopup")

    if len(argv) > 1:
        lc = float(argv[1])
    if len(argv) > 2:
        target_elements = int(argv[2])

    # Step 1: Create initial geometry and mesh using meshwell
    cad_file, mesh_file, _ = create_meshwell_geometry_and_mesh(lc)

    # Step 2: Apply adaptive refinement using our new module
    print("\nStarting adaptive refinement...")
    print(f"Target elements: {target_elements}")

    refined_mesh_file = Path("meshwell_test_refined.msh")
    gmsh_adaptive_remesh(
        input_mesh_file=mesh_file,
        input_cad_file=cad_file,
        target_function=test_function,
        target_elements=target_elements,
        output_mesh_file=str(refined_mesh_file),
        verbosity=1,
    )

    return True


if __name__ == "__main__":
    success = main()
