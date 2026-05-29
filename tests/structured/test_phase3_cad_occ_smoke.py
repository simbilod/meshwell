"""cad_occ fragment smoke test with Phase 3 cohort envelopes.

Goal: confirm that build_phantom_shapes under Phase 3 produces solids
that cad_occ consumes, and that per-piece FaceKeys resolve to gmsh tags
via the PhantomMap.
"""

from __future__ import annotations

from unittest.mock import patch

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.spec import FaceKey


def _square_slab(zlo, zhi, name, mesh_order=1.0):
    return PolyPrism(
        polygons=shapely.box(0, 0, 1, 1),
        buffers={zlo: 0.0, zhi: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name=name,
        mesh_order=mesh_order,
    )


def test_phase3_group_phantom_solids_by_entity_handles_cohort():
    """_group_phantom_solids_by_entity must not IndexError for negative slab_index."""
    from meshwell.structured.phantom import (
        _group_phantom_solids_by_entity,
        build_phantom_shapes,
    )
    from meshwell.structured.plan import build_plan

    entities = [_square_slab(0.0, 1.0, "L1"), _square_slab(1.0, 2.0, "L2")]
    plan = build_plan(entities)
    with patch("meshwell.structured.phantom._USE_DISCRETE_COHORT_MESH", True):
        phantom_result = build_phantom_shapes(plan)

    # Phase 3: one cohort PhantomShape with synthetic slab_index=-1.
    assert len(phantom_result.shapes) == 1
    assert phantom_result.shapes[0].slab_index < 0

    # Must not raise IndexError / KeyError.
    overrides = _group_phantom_solids_by_entity(plan, phantom_result)

    # Both entity source indices (0, 1) must appear.
    assert 0 in overrides
    assert 1 in overrides
    # Exactly one entity gets the envelope solid; the other gets [].
    total_solids = sum(len(v) for v in overrides.values())
    assert total_solids == 1, (
        f"Expected exactly 1 cohort envelope solid across all overrides; "
        f"got {total_solids}: {overrides}"
    )


def test_phase3_extract_phantom_map_populates_all_face_keys():
    """extract_phantom_map walks per-piece FaceKeys from cohort PhantomShape."""
    from OCP.BOPAlgo import BOPAlgo_Builder

    from meshwell.structured.phantom import (
        build_phantom_shapes,
        extract_phantom_map,
    )
    from meshwell.structured.plan import build_plan

    entities = [_square_slab(0.0, 1.0, "L1"), _square_slab(1.0, 2.0, "L2")]
    plan = build_plan(entities)
    with patch("meshwell.structured.phantom._USE_DISCRETE_COHORT_MESH", True):
        phantom_result = build_phantom_shapes(plan)

    ps = phantom_result.shapes[0]
    # All four per-piece keys should be present in the cohort PhantomShape.
    for fk in (
        FaceKey(0, "top", 0),
        FaceKey(0, "bot", 0),
        FaceKey(1, "top", 0),
        FaceKey(1, "bot", 0),
    ):
        assert fk in ps.input_faces_by_key, f"{fk} missing from input_faces_by_key"

    # Run BOP on just the envelope solid (no neighbours).
    builder = BOPAlgo_Builder()
    builder.AddArgument(ps.solid)
    builder.Perform()

    # extract_phantom_map must not raise KeyError / IndexError.
    pmap = extract_phantom_map(phantom_result, builder)

    # All four face keys must be present in output_faces.
    for fk in (
        FaceKey(0, "top", 0),
        FaceKey(0, "bot", 0),
        FaceKey(1, "top", 0),
        FaceKey(1, "bot", 0),
    ):
        assert fk in pmap.output_faces, f"{fk} missing from output_faces"
        assert len(pmap.output_faces[fk]) >= 1, f"{fk} has empty face list"


def test_phase3_envelope_solids_round_trip_through_cad_occ():
    """Build envelope solid → cad_occ → extract_phantom_map resolves all FaceKeys."""
    from meshwell.cad_occ import cad_occ
    from meshwell.structured.phantom import (
        _group_phantom_solids_by_entity,
        build_phantom_shapes,
        extract_phantom_map,
    )
    from meshwell.structured.plan import build_plan

    entities = [
        _square_slab(0.0, 1.0, "L1", mesh_order=1.0),
        _square_slab(1.0, 2.0, "L2", mesh_order=2.0),
    ]
    plan = build_plan(entities)
    with patch("meshwell.structured.phantom._USE_DISCRETE_COHORT_MESH", True):
        phantom_result = build_phantom_shapes(plan)

    # Phase 3: single cohort envelope.
    assert len(phantom_result.shapes) == 1

    overrides = _group_phantom_solids_by_entity(plan, phantom_result)

    captured: list = []
    occ_entities = cad_occ(
        entities,
        entity_shape_overrides=overrides,
        cad_occ_callback=lambda b: captured.append(b),
    )
    assert len(captured) == 1, "cad_occ_callback should fire exactly once"

    pmap = extract_phantom_map(phantom_result, captured[0])

    # All four per-piece FaceKeys must resolve to non-empty face lists.
    for fk in (
        FaceKey(0, "top", 0),
        FaceKey(0, "bot", 0),
        FaceKey(1, "top", 0),
        FaceKey(1, "bot", 0),
    ):
        assert fk in pmap.output_faces, f"{fk} missing from PhantomMap.output_faces"
        assert (
            len(pmap.output_faces[fk]) >= 1
        ), f"{fk} has empty face list in PhantomMap.output_faces"

    # There should be exactly 1 cohort-level structured OCC solid in the result
    # (not 2 separate entity solids).  After cad_occ fragment, the cohort
    # envelope passes through as a single entity solid assigned to the first
    # source entity in the cohort.
    by_name = {le.physical_name[0]: le for le in occ_entities}
    assert "L1" in by_name, f"L1 not in occ_entities; found: {list(by_name)}"
    # L1 should carry exactly 1 shape (the cohort envelope solid).
    assert len(by_name["L1"].shapes) == 1, (
        f"Expected L1 to carry 1 cohort envelope solid; "
        f"got {len(by_name['L1'].shapes)}"
    )
