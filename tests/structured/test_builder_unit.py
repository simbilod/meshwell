"""Unit tests for builder helpers using direct gmsh fixtures."""
from __future__ import annotations

import gmsh
import numpy as np
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


def _make_box_in_gmsh_and_mesh_2d(z_lo: float, z_hi: float):
    """Create a unit box in gmsh.model.occ, sync, mesh 2D. Returns (bot_face_tag, top_face_tag)."""
    gmsh.model.occ.addBox(0, 0, z_lo, 1, 1, z_hi - z_lo)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.3)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.3)
    gmsh.model.mesh.generate(2)
    bot_tag = top_tag = None
    for dim, tag in gmsh.model.getEntities(2):
        bb = gmsh.model.getBoundingBox(dim, tag)
        if abs(bb[2] - z_lo) < 1e-6 and abs(bb[5] - z_lo) < 1e-6:
            bot_tag = tag
        elif abs(bb[2] - z_hi) < 1e-6 and abs(bb[5] - z_hi) < 1e-6:
            top_tag = tag
    assert bot_tag is not None
    assert top_tag is not None
    return bot_tag, top_tag


def test_stamp_top_face_mesh_replaces_top_with_translated_bottom(
    gmsh_session,  # noqa: ARG001
):
    """Bottom mesh on bot_face -> derived top mesh on top_face with same connectivity."""
    from meshwell.structured.builder import _stamp_top_face_mesh

    bot_tag, top_tag = _make_box_in_gmsh_and_mesh_2d(z_lo=0.0, z_hi=1.0)
    bot_node_tags_before, _, _ = gmsh.model.mesh.getNodes(
        2, bot_tag, includeBoundary=True
    )
    _stamp_top_face_mesh(
        bottom_face_tag=bot_tag,
        top_face_tag=top_tag,
        zlo=0.0,
        zhi=1.0,
    )
    top_node_tags, top_coords, _ = gmsh.model.mesh.getNodes(
        2, top_tag, includeBoundary=True
    )
    assert len(top_node_tags) == len(bot_node_tags_before)
    top_z = np.asarray(top_coords, dtype=float).reshape(-1, 3)[:, 2]
    assert (abs(top_z - 1.0) < 1e-6).all()


def test_stamp_top_face_mesh_produces_matching_triangle_count(
    gmsh_session,  # noqa: ARG001
):
    from meshwell.structured.builder import _stamp_top_face_mesh

    bot_tag, top_tag = _make_box_in_gmsh_and_mesh_2d(z_lo=0.0, z_hi=1.0)
    bot_types_before, bot_tags_before, _ = gmsh.model.mesh.getElements(2, bot_tag)
    n_bot_tris = sum(
        len(t) for et, t in zip(bot_types_before, bot_tags_before) if et == 2
    )

    _stamp_top_face_mesh(
        bottom_face_tag=bot_tag,
        top_face_tag=top_tag,
        zlo=0.0,
        zhi=1.0,
    )
    top_types, top_tags_, _ = gmsh.model.mesh.getElements(2, top_tag)
    n_top_tris = sum(len(t) for et, t in zip(top_types, top_tags_) if et == 2)
    assert n_top_tris == n_bot_tris


def test_build_slab_volume_single_layer_produces_wedges(gmsh_session):  # noqa: ARG001
    """Single-layer slab: triangles in bottom -> wedge prisms in volume."""
    from meshwell.structured.builder import _build_slab_volume, _stamp_top_face_mesh

    bot_tag, top_tag = _make_box_in_gmsh_and_mesh_2d(z_lo=0.0, z_hi=1.0)
    bot_to_top = _stamp_top_face_mesh(bot_tag, top_tag, zlo=0.0, zhi=1.0)

    # Count bottom triangles for comparison.
    bot_types, bot_etags, _ = gmsh.model.mesh.getElements(2, bot_tag)
    n_bot_tris = sum(len(t) for et, t in zip(bot_types, bot_etags) if et == 2)
    assert n_bot_tris > 0

    vol_tag = _build_slab_volume(
        bottom_face_tag=bot_tag,
        bot_to_top_layer_tags=[bot_to_top],
        n_layers=1,
        recombine=False,
    )
    assert vol_tag > 0
    etypes, etags, _ = gmsh.model.mesh.getElements(3, vol_tag)
    n_wedges = sum(len(t) for et, t in zip(etypes, etags) if et == 6)
    assert n_wedges == n_bot_tris


def test_build_slab_volume_multi_layer_skipped(gmsh_session):  # noqa: ARG001
    """Multi-layer path is exercised in Task 6 end-to-end.

    Intermediate node map allocation lives in apply_structured_mesh, not here.
    """
    pytest.skip(
        "multi-layer wiring requires intermediate node maps allocated by "
        "apply_structured_mesh; tested end-to-end in Task 6"
    )
