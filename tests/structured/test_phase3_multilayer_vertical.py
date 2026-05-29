"""Vertical interfaces stamp correctly when n_layers > 1."""

from __future__ import annotations

from unittest.mock import patch

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec


def test_vertical_interface_multilayer(tmp_path):
    import gmsh

    from meshwell.orchestrator import generate_mesh

    s1 = PolyPrism(
        polygons=shapely.box(0, 0, 1, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[3])],
        physical_name="A",
    )
    s2 = PolyPrism(
        polygons=shapely.box(1, 0, 2, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[3])],
        physical_name="B",
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
        # Find the A___B group; count quads — should be n_layers per
        # horizontal subdivision of the shared edge.
        groups_2d = gmsh.model.getPhysicalGroups(dim=2)
        ab_tag = next(
            t
            for (_d, t) in groups_2d
            if gmsh.model.getPhysicalName(2, t) in ("A___B", "B___A")
        )
        ents = gmsh.model.getEntitiesForPhysicalGroup(2, ab_tag)
        total_quads = 0
        for ent in ents:
            elem_types, elem_tags, _ = gmsh.model.mesh.getElements(2, int(ent))
            for et, ets in zip(elem_types, elem_tags):
                if et == 3:  # quad
                    total_quads += len(ets)
        # The shared edge runs from y=0 to y=1; with default size 0.5 there's
        # at least 1 horizontal subdivision. Each subdivision x 3 layers >= 3.
        assert total_quads >= 3, f"Expected >= 3 quads, got {total_quads}"
    finally:
        gmsh.finalize()
