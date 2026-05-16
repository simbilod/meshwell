"""Tests for Phase-2 key dataclasses in meshwell.structured.spec."""
from __future__ import annotations


def test_face_key_fields():
    from meshwell.structured.spec import FaceKey

    k = FaceKey(slab_index=2, side="top", piece_index=0)
    assert k.slab_index == 2
    assert k.side == "top"
    assert k.piece_index == 0


def test_face_key_hashable():
    from meshwell.structured.spec import FaceKey

    k1 = FaceKey(slab_index=2, side="top", piece_index=0)
    k2 = FaceKey(slab_index=2, side="top", piece_index=0)
    assert k1 == k2
    assert hash(k1) == hash(k2)
    assert {k1: "a"}[k2] == "a"  # usable as dict key


def test_edge_key_includes_edge_index():
    from meshwell.structured.spec import EdgeKey

    k = EdgeKey(slab_index=0, side="bot", piece_index=3, edge_index=2)
    assert k.edge_index == 2


def test_vertex_key_includes_corner_index():
    from meshwell.structured.spec import VertexKey

    k = VertexKey(slab_index=0, side="bot", piece_index=0, corner_index=4)
    assert k.corner_index == 4


def test_lateral_key_fields():
    from meshwell.structured.spec import LateralKey

    k = LateralKey(slab_index=1, outer_edge_index=2)
    assert k.slab_index == 1
    assert k.outer_edge_index == 2


def test_side_only_accepts_bot_or_top_runtime():
    """Side is a typing Literal; runtime acceptance of any str is fine — type checker enforces it."""
    from meshwell.structured.spec import FaceKey

    # Pydantic would reject; plain dataclass does not. Just confirm we can construct.
    k = FaceKey(slab_index=0, side="bot", piece_index=0)
    assert k.side == "bot"


def test_phantom_map_defaults_to_empty_dicts():
    from meshwell.structured.spec import PhantomMap

    m = PhantomMap()
    assert m.output_faces == {}
    assert m.output_edges == {}
    assert m.output_vertices == {}
    assert m.output_laterals == {}


def test_phantom_shape_holds_solid_and_keys():
    """PhantomShape carries the OCP solid + the input-tag bookkeeping for one piece."""
    from meshwell.structured.spec import (
        EdgeKey,
        FaceKey,
        PhantomShape,
        VertexKey,
    )

    # We don't need a real TopoDS here — use a sentinel object.
    sentinel = object()
    s = PhantomShape(
        slab_index=0,
        piece_index=2,
        solid=sentinel,
        input_faces_by_key={
            FaceKey(0, "bot", 2): "fake_face_tag_id_a",
            FaceKey(0, "top", 2): "fake_face_tag_id_b",
        },
        input_edges_by_key={
            EdgeKey(0, "bot", 2, 0): "fake_edge_tag_id_a",
        },
        input_vertices_by_key={
            VertexKey(0, "bot", 2, 0): "fake_vertex_tag_id_a",
        },
        input_laterals_by_outer_edge={0: "fake_lateral_a"},
    )
    assert s.solid is sentinel
    assert FaceKey(0, "bot", 2) in s.input_faces_by_key


def test_phantom_build_result_aggregates_shapes():
    from meshwell.structured.spec import PhantomBuildResult, PhantomShape

    s = PhantomShape(
        slab_index=0,
        piece_index=0,
        solid=object(),
        input_faces_by_key={},
        input_edges_by_key={},
        input_vertices_by_key={},
        input_laterals_by_outer_edge={},
    )
    r = PhantomBuildResult(shapes=(s,))
    assert r.shapes == (s,)
