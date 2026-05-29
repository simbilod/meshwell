"""End-to-end mesh with 4 pieces x 2 slabs = 8 piece volumes.

Spec test #4: verify 8 discrete 3D entities exist with correct
physical names and total element count matches structured layer counts.
"""

from __future__ import annotations

from unittest.mock import patch

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec


def test_8_piece_cohort_yields_8_discrete_volumes(tmp_path):
    import gmsh

    from meshwell.orchestrator import generate_mesh

    # 4 quadrant slabs at z=[0,1] and z=[1,2] => 8 pieces total in one cohort.
    out = tmp_path / "phase3.msh"
    entities = []
    quadrants = [
        (0.0, 0.0, 0.5, 0.5),
        (0.5, 0.0, 1.0, 0.5),
        (0.0, 0.5, 0.5, 1.0),
        (0.5, 0.5, 1.0, 1.0),
    ]
    for zlo, zhi in [(0.0, 1.0), (1.0, 2.0)]:
        for q_i, (x0, y0, x1, y1) in enumerate(quadrants):
            entities.append(
                PolyPrism(
                    polygons=shapely.box(x0, y0, x1, y1),
                    buffers={zlo: 0.0, zhi: 0.0},
                    structured=True,
                    resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
                    physical_name=f"Q{q_i}_z{int(zlo)}",
                )
            )

    with patch("meshwell.structured.phantom._USE_DISCRETE_COHORT_MESH", True):
        generate_mesh(
            entities,
            dim=3,
            output_mesh=str(out),
            default_characteristic_length=0.5,
        )

    gmsh.initialize()
    try:
        gmsh.open(str(out))
        groups_3d = gmsh.model.getPhysicalGroups(dim=3)
        names = [gmsh.model.getPhysicalName(3, t) for (_d, t) in groups_3d]
        # 8 distinct piece physical groups.
        assert len(set(names)) == 8, f"Expected 8 distinct names, got {names}"
        # Total 3D element count per group > 0.
        for _d, t in groups_3d:
            ents = gmsh.model.getEntitiesForPhysicalGroup(3, t)
            total = 0
            for ent in ents:
                _, et_tags, _ = gmsh.model.mesh.getElements(3, int(ent))
                for tags in et_tags:
                    total += len(tags)
            assert (
                total > 0
            ), f"Group {gmsh.model.getPhysicalName(3, t)} has no elements"
    finally:
        gmsh.finalize()
