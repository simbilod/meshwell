"""Unstructured neighbor above/below shares OCC topology with cohort top/bot."""

from __future__ import annotations

from unittest.mock import patch

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec


def test_unstructured_tet_above_structured_cohort_shares_nodes(tmp_path):
    """Tet-meshed slab above a structured cohort: interface nodes shared."""
    import gmsh
    import numpy as np

    from meshwell.orchestrator import generate_mesh

    structured = PolyPrism(
        polygons=shapely.box(0, 0, 1, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="StructA",
    )
    # Unstructured above.
    unstructured = PolyPrism(
        polygons=shapely.box(0, 0, 1, 1),
        buffers={1.0: 0.0, 2.0: 0.0},
        structured=False,
        physical_name="UnstructB",
    )
    out = tmp_path / "phase3.msh"
    with patch("meshwell.structured.phantom._USE_DISCRETE_COHORT_MESH", True):
        generate_mesh(
            [structured, unstructured],
            dim=3,
            output_mesh=str(out),
            default_characteristic_length=0.5,
        )

    gmsh.initialize()
    try:
        gmsh.open(str(out))
        # No duplicate XY positions at z=1.
        _, all_coords_flat, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(all_coords_flat, dtype=float).reshape(-1, 3)
        at_z1 = np.isclose(coords[:, 2], 1.0, atol=1e-9)
        xy_at_z1 = coords[at_z1, :2]
        unique = np.unique(np.round(xy_at_z1, 9), axis=0)
        assert unique.shape[0] == int(
            at_z1.sum()
        ), "Duplicate nodes between structured + unstructured"
    finally:
        gmsh.finalize()
