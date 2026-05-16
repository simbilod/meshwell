"""Tests for builder._map_phantom_faces_to_gmsh."""
from __future__ import annotations

import gmsh
import pytest
from shapely.geometry import Polygon


def _square(x=0, y=0, w=1, h=1) -> Polygon:
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


@pytest.fixture
def gmsh_session():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("test")
    try:
        yield
    finally:
        gmsh.finalize()


def test_map_phantom_faces_to_gmsh_single_piece(gmsh_session):  # noqa: ARG001
    """Single-piece slab: one bottom + one top face match cleanly."""
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
    for s in phantom_result.shapes:
        builder.AddArgument(s.solid)
    builder.Perform()
    phantom_map = extract_phantom_map(phantom_result, builder)

    # Push the phantom into gmsh. importShapesNativePointer may not work in
    # all OCP builds; fall back to a BREP roundtrip via tmp file if it
    # fails. Or use any other available technique.
    try:
        gmsh.model.occ.importShapesNativePointer(
            int(phantom_result.shapes[0].solid.this)
        )
    except Exception:
        # Alternative: serialize to a tmp BREP, then importShapes.
        import tempfile
        from pathlib import Path

        from OCP.BRepTools import BRepTools

        with tempfile.NamedTemporaryFile(suffix=".brep", delete=False) as tf:
            tmp = Path(tf.name)
        BRepTools.Write_s(phantom_result.shapes[0].solid, str(tmp))
        gmsh.model.occ.importShapes(str(tmp))
        tmp.unlink(missing_ok=True)
    gmsh.model.occ.synchronize()

    fmap = _map_phantom_faces_to_gmsh(phantom_map)
    assert FaceKey(0, "bot", 0) in fmap
    assert FaceKey(0, "top", 0) in fmap
    assert len(fmap[FaceKey(0, "bot", 0)]) == 1
    assert len(fmap[FaceKey(0, "top", 0)]) == 1
