"""Horizontal interior interface discrete 2D entity stamping under Phase 3."""

from __future__ import annotations

from unittest.mock import patch

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec


def _square_slab(zlo, zhi, name):
    return PolyPrism(
        polygons=shapely.box(0, 0, 1, 1),
        buffers={zlo: 0.0, zhi: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name=name,
    )


def test_horizontal_interior_interface_materialized(tmp_path):
    """Stacked slabs in one cohort produce a discrete 2D entity at z=1."""
    import gmsh

    from meshwell.orchestrator import generate_mesh

    out = tmp_path / "phase3.msh"
    entities = [_square_slab(0.0, 1.0, "L1"), _square_slab(1.0, 2.0, "L2")]
    with patch("meshwell.structured.phantom._USE_DISCRETE_COHORT_MESH", True):
        generate_mesh(
            entities, dim=3, output_mesh=str(out), default_characteristic_length=0.5
        )

    gmsh.initialize()
    try:
        gmsh.open(str(out))
        groups_2d = gmsh.model.getPhysicalGroups(dim=2)
        names = {gmsh.model.getPhysicalName(2, t) for (_d, t) in groups_2d}
        # Existing meshwell convention names the interface with the triple
        # underscore delimiter, in either order.
        assert (
            "L1___L2" in names or "L2___L1" in names
        ), f"Expected interface physical group, got: {names}"
    finally:
        gmsh.finalize()
