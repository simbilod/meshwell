"""Conformal node sharing across interior interfaces."""

from __future__ import annotations

from unittest.mock import patch

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec


def test_horizontal_interface_nodes_shared(tmp_path):
    """The shared interface mesh nodes are the same tags on both sides."""
    import gmsh
    import numpy as np

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
            [s1, s2],
            dim=3,
            output_mesh=str(out),
            default_characteristic_length=0.5,
        )

    gmsh.initialize()
    try:
        gmsh.open(str(out))
        _, all_coords_flat, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(all_coords_flat, dtype=float).reshape(-1, 3)
        at_interface = np.isclose(coords[:, 2], 1.0, atol=1e-9)
        n_interface_nodes = int(at_interface.sum())
        assert n_interface_nodes > 0, "Expected interface plane nodes"

        # No duplicate XY positions at z=1.
        xy_at_iface = coords[at_interface, :2]
        unique = np.unique(np.round(xy_at_iface, 9), axis=0)
        assert (
            unique.shape[0] == n_interface_nodes
        ), "Duplicate interface nodes survived removeDuplicateNodes"
    finally:
        gmsh.finalize()
