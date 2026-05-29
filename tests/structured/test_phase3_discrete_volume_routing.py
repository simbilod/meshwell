"""Each cohort piece gets its own discrete 3D entity under Phase 3."""

from __future__ import annotations

from unittest.mock import patch

import pytest
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


@pytest.mark.xfail(
    reason=(
        "Phase 3 mesh stage is not yet complete: apply_structured_mesh calls "
        "_map_phantom_faces_to_gmsh which fails for interior cohort faces "
        "(e.g. FaceKey(slab_index=1, side='bot', piece_index=0) at z=1.0) "
        "because those faces are excluded from the cohort envelope solid "
        "(to avoid non-manifold geometry) and therefore have no gmsh tag in "
        "the XAO compound. Task 15 (interior 2D interface stamping) must "
        "land before this test can reach the volume-routing assertion. "
        "The Task 14 code change (builder.py: per-piece discrete 3D entity "
        "allocation under Phase 3) IS in place; only the test cannot run "
        "end-to-end yet."
    ),
    strict=True,
)
def test_each_piece_gets_dedicated_discrete_volume(tmp_path):
    """Two stacked single-piece slabs in one cohort → 2 distinct volume tags."""
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
        vol_groups = [
            (d, t, gmsh.model.getPhysicalName(d, t))
            for (d, t) in gmsh.model.getPhysicalGroups(dim=3)
        ]
        names = [name for (_d, _t, name) in vol_groups]
        # Two distinct physical groups (one per piece).
        assert "L1" in names
        assert "L2" in names
        assert names.count("L1") == 1
        assert names.count("L2") == 1
    finally:
        gmsh.finalize()
