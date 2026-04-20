"""Unit tests for the all-fragment OCC pipeline."""
from __future__ import annotations

from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt

from meshwell.cad_occ import OCCLabeledEntity


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
