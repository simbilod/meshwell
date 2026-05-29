"""Vertical interior interface materialization under Phase 3."""

from __future__ import annotations

from unittest.mock import patch

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec


def test_vertical_interface_between_lateral_pieces(tmp_path):
    """Two laterally-adjacent slabs produce a vertical interface group.

    Two side-by-side square slabs in one cohort sharing one outline
    edge produce a discrete 2D physical group at the shared edge.
    """
    import gmsh

    from meshwell.orchestrator import generate_mesh

    s1 = PolyPrism(
        polygons=shapely.box(0, 0, 1, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="LeftSlab",
    )
    s2 = PolyPrism(
        polygons=shapely.box(1, 0, 2, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="RightSlab",
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
        groups_2d = gmsh.model.getPhysicalGroups(dim=2)
        names = {gmsh.model.getPhysicalName(2, t) for (_d, t) in groups_2d}
        assert (
            "LeftSlab___RightSlab" in names or "RightSlab___LeftSlab" in names
        ), f"Expected vertical interface physical group, got {names}"
    finally:
        gmsh.finalize()
