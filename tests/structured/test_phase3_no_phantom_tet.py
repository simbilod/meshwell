"""Cohort envelope OCC volume must not be tetrahedralized."""

from __future__ import annotations

from unittest.mock import patch

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec


def test_cohort_envelope_volume_not_in_final_mesh(tmp_path):
    """The 3D mesh contains only structured elements, no tets."""
    import gmsh

    from meshwell.orchestrator import generate_mesh

    s1 = PolyPrism(
        polygons=shapely.box(0, 0, 1, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="L1",
    )
    s2 = PolyPrism(
        polygons=shapely.box(0, 0, 1, 1),
        buffers={1.0: 0.0, 2.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="L2",
    )
    out = tmp_path / "phase3.msh"
    with patch("meshwell.structured.phantom._USE_DISCRETE_COHORT_MESH", True):
        generate_mesh(
            [s1, s2], dim=3, output_mesh=str(out), default_characteristic_length=0.5
        )

    gmsh.initialize()
    try:
        gmsh.open(str(out))
        elem_types, _, _ = gmsh.model.mesh.getElements(3)
        # element type 4 = tetrahedron; should be absent.
        assert 4 not in elem_types, (
            f"Found tetrahedra in 3D mesh (cohort envelope was "
            f"tetrahedralized): {elem_types}"
        )
    finally:
        gmsh.finalize()
