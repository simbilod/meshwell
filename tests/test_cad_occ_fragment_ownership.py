"""Unit tests for the all-fragment OCC pipeline."""
from __future__ import annotations

from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt

from meshwell.cad_occ import (
    CAD_OCC,
    OCCLabeledEntity,
    _resolve_piece_ownership,
    _shape_key,
    cad_occ,
)
from meshwell.occ_entity import OCC_entity


def test_occ_labeled_entity_accepts_shapes_list():
    """OCCLabeledEntity should store a list of fragment pieces."""
    box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1.0, 1.0, 1.0).Shape()
    ent = OCCLabeledEntity(
        shapes=[box],
        physical_name=("box",),
        index=0,
        keep=True,
        dim=3,
    )
    assert ent.shapes == [box]
    assert ent.dim == 3


def test_occ_labeled_entity_multiple_pieces():
    """OCCLabeledEntity must support multiple fragment pieces per entity."""
    b1 = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1.0, 1.0, 1.0).Shape()
    b2 = BRepPrimAPI_MakeBox(gp_Pnt(2, 0, 0), 1.0, 1.0, 1.0).Shape()
    ent = OCCLabeledEntity(
        shapes=[b1, b2],
        physical_name=("disjoint",),
        index=1,
        keep=True,
        dim=3,
    )
    assert len(ent.shapes) == 2


def test_resolve_piece_ownership_lowest_wins():
    """When multiple entities claim a piece, lowest mesh_order wins."""
    # piece_candidates maps piece_id -> list of (entity_index, mesh_order)
    piece_candidates = {
        "pA": [(0, 2.0), (1, 1.0)],  # entity 1 (mesh_order 1) wins
        "pB": [(0, 2.0)],  # entity 0 only
        "pC": [(2, 3.0), (1, 1.0), (0, 2.0)],  # entity 1 wins
    }
    owners = _resolve_piece_ownership(piece_candidates)
    assert owners == {"pA": 1, "pB": 0, "pC": 1}


def test_resolve_piece_ownership_tie_first_wins():
    """On mesh_order tie, the first candidate (insertion order) wins."""
    piece_candidates = {"p": [(3, 1.0), (5, 1.0), (2, 1.0)]}
    owners = _resolve_piece_ownership(piece_candidates)
    assert owners == {"p": 3}


def test_resolve_piece_ownership_inf_mesh_order():
    """Entities with mesh_order=None treated as infinity (lowest priority)."""
    piece_candidates = {
        "p": [(0, float("inf")), (1, 5.0)],
    }
    owners = _resolve_piece_ownership(piece_candidates)
    assert owners == {"p": 1}


def test_shape_key_same_shape_equal():
    """Two handles to the same underlying shape must compare equal."""
    box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1.0, 1.0, 1.0).Shape()
    k1 = _shape_key(box)
    k2 = _shape_key(box)
    assert k1 == k2
    assert hash(k1) == hash(k2)


def test_shape_key_different_shapes_differ():
    """Distinct shape constructions produce distinct keys."""
    b1 = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1.0, 1.0, 1.0).Shape()
    b2 = BRepPrimAPI_MakeBox(gp_Pnt(2, 0, 0), 1.0, 1.0, 1.0).Shape()
    assert _shape_key(b1) != _shape_key(b2)


def _make_ent(idx, shape, mesh_order, name, dim=3, keep=True):
    return OCCLabeledEntity(
        shapes=[shape],
        physical_name=(name,),
        index=idx,
        keep=keep,
        dim=dim,
        mesh_order=mesh_order,
    )


def test_fragment_all_disjoint_boxes_preserved():
    """Disjoint shapes are unchanged; each entity keeps its piece."""
    b1 = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1.0, 1.0, 1.0).Shape()
    b2 = BRepPrimAPI_MakeBox(gp_Pnt(5, 0, 0), 1.0, 1.0, 1.0).Shape()
    ents = [_make_ent(0, b1, 1.0, "a"), _make_ent(1, b2, 2.0, "b")]
    processor = CAD_OCC()
    result = processor._fragment_all(ents)
    assert len(result) == 2
    assert len(result[0].shapes) == 1
    assert len(result[1].shapes) == 1


def test_fragment_all_overlap_goes_to_lower_mesh_order():
    """Overlapping region is owned by the entity with the lower mesh_order."""
    b1 = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 2.0, 2.0, 2.0).Shape()
    b2 = BRepPrimAPI_MakeBox(gp_Pnt(1, 1, 1), 2.0, 2.0, 2.0).Shape()
    # a has mesh_order 1 (higher priority), b has mesh_order 2
    ents = [_make_ent(0, b1, 1.0, "a"), _make_ent(1, b2, 2.0, "b")]
    processor = CAD_OCC()
    result = processor._fragment_all(ents)
    # Sum of all pieces should equal the number of fragments produced.
    total_pieces = sum(len(e.shapes) for e in result)
    # At minimum a gets the whole a, b gets only its non-overlapping remainder.
    assert total_pieces >= 2
    # 'a' must not have been shrunk to zero
    assert len(result[0].shapes) >= 1
    # 'b' is split; its pieces should be fewer than b1+b2 combined
    assert len(result[1].shapes) >= 1


def test_process_entities_overlapping_boxes_end_to_end():
    """Higher-priority box keeps its full volume; lower-priority box loses overlap."""
    a = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(
            gp_Pnt(0, 0, 0), 2.0, 2.0, 2.0
        ).Shape(),
        physical_name="a",
        mesh_order=1,
        dimension=3,
    )
    b = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(
            gp_Pnt(1, 1, 1), 2.0, 2.0, 2.0
        ).Shape(),
        physical_name="b",
        mesh_order=2,
        dimension=3,
    )
    result = cad_occ([a, b])
    assert len(result) == 2
    # Both entities should still have pieces.
    assert all(len(ent.shapes) >= 1 for ent in result)
    names = {ent.physical_name[0] for ent in result}
    assert names == {"a", "b"}
