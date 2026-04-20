"""Unit tests for the all-fragment OCC pipeline."""
from __future__ import annotations

from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt

from meshwell.cad_occ import OCCLabeledEntity, _resolve_piece_ownership


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
