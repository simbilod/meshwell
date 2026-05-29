"""Tests for builder._map_phantom_faces_to_gmsh (TopExp-index based)."""
from __future__ import annotations

import pytest
from shapely.geometry import Polygon

import meshwell.structured.phantom as _phantom_mod

_PHASE3_ON = getattr(_phantom_mod, "_USE_DISCRETE_COHORT_MESH", False)


def _square(x=0, y=0, w=1, h=1) -> Polygon:
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def _make_occ_labeled_entity(
    shape, dim=3, keep=True, physical_name=("e",), mesh_order=1.0
):
    """Build a minimal OCCLabeledEntity stand-in for testing."""
    from dataclasses import dataclass

    @dataclass
    class _StubEntity:
        shapes: list
        dim: int
        keep: bool
        physical_name: tuple
        mesh_order: float

    return _StubEntity(
        shapes=[shape],
        dim=dim,
        keep=keep,
        physical_name=physical_name,
        mesh_order=mesh_order,
    )


@pytest.mark.skipif(
    _PHASE3_ON,
    reason="Phase 1+2 path — Phase 3 cohort envelope needs plan.arrangements which the handcrafted fixture omits",
)
def test_map_phantom_faces_to_gmsh_single_piece():
    """Single-piece slab: each FaceKey maps to a 1-based TopExp index."""
    from OCP.BOPAlgo import BOPAlgo_Builder

    from meshwell.structured.builder import _map_phantom_faces_to_gmsh
    from meshwell.structured.phantom import build_phantom_shapes, extract_phantom_map
    from meshwell.structured.spec import (
        FaceKey,
        Slab,
        StructuredPlan,
    )

    slab = Slab(
        footprint=_square(0, 0, 2, 2),
        zlo=0.0,
        zhi=1.0,
        physical_name=("s",),
        source_index=0,
        z_interval_index=0,
        mesh_order=1.0,
        face_partition=[_square(0, 0, 2, 2)],
    )
    plan = StructuredPlan(slabs=(slab,), z_planes=(0.0, 1.0), overlaps=())
    phantom_result = build_phantom_shapes(plan)

    builder = BOPAlgo_Builder()
    for ph in phantom_result.shapes:
        builder.AddArgument(ph.solid)
    builder.Perform()
    phantom_map = extract_phantom_map(phantom_result, builder)

    # Wrap the phantom solid as if it were a kept user entity (so the XAO
    # compound includes its shape, mirroring how cad_occ would include
    # neighbour user entities whose solids share TShapes with the phantom
    # post-BOP).
    fake_entities = [
        _make_occ_labeled_entity(phantom_result.shapes[0].solid, dim=3, keep=True)
    ]

    fmap = _map_phantom_faces_to_gmsh(phantom_map, fake_entities)

    # Each FaceKey should have exactly one gmsh tag.
    bot_key = FaceKey(0, "bot", 0)
    top_key = FaceKey(0, "top", 0)
    assert bot_key in fmap
    assert top_key in fmap
    assert len(fmap[bot_key]) == 1
    assert len(fmap[top_key]) == 1
    # Tags are positive 1-based integers.
    assert fmap[bot_key][0] >= 1
    assert fmap[top_key][0] >= 1
    # Bot != top.
    assert fmap[bot_key][0] != fmap[top_key][0]


@pytest.mark.skipif(
    _PHASE3_ON,
    reason="Phase 1+2 path — Phase 3 _map_phantom_faces_to_gmsh silently skips missing faces (interior cohort FaceKeys have no OCC backing)",
)
def test_map_phantom_faces_to_gmsh_missing_face_raises():
    """If compound doesn't contain the phantom face, raise with a clear message.

    This exercises the architectural-error path (face not included in BOP).
    """
    # Build a phantom_map with a fake face object that's not in any entity.
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    from meshwell.structured.builder import _map_phantom_faces_to_gmsh
    from meshwell.structured.spec import FaceKey, PhantomMap

    orphan_box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), gp_Pnt(1, 1, 1)).Solid()
    exp = TopExp_Explorer(orphan_box, TopAbs_FACE)
    orphan_face = TopoDS.Face_s(exp.Current())

    pmap = PhantomMap()
    pmap.output_faces[FaceKey(0, "bot", 0)] = [orphan_face]

    # An entity that doesn't contain the orphan_face.
    other_box = BRepPrimAPI_MakeBox(gp_Pnt(10, 10, 10), gp_Pnt(11, 11, 11)).Solid()
    fake_entities = [_make_occ_labeled_entity(other_box)]

    with pytest.raises(RuntimeError, match="IsSame"):
        _map_phantom_faces_to_gmsh(pmap, fake_entities)
