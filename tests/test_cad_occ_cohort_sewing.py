"""Smoke tests for the pre-shared-face cohort BOP design.

The original sewing+compound design failed because BOPAlgo_Builder.Modified()
returns empty for sub-shapes of a compound argument. We pivoted to pre-sharing
TopoDS_Face objects between adjacent sub-prisms at construction time, then
passing each sub-prism as its own argument to BOPAlgo. This file validates
the pivot's foundational assumptions:

1. BRepPrimAPI_MakePrism accepts a TopoDS_Face that came from another
   prism's LastShape(), and the resulting two solids genuinely share the
   interface face's TShape identity.

2. BOPAlgo_Builder handles two solids that share a pre-built interface face
   correctly — Modified() works per-argument, and the shared face survives
   with shared TShape in both solids' fragmented pieces.

These tests are independent of meshwell internals so they run before any
meshwell change ships.
"""

from __future__ import annotations

from OCP.BOPAlgo import BOPAlgo_Builder
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakePrism
from OCP.gp import gp_Pnt, gp_Vec
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS_Face


def _faces(shape):
    out = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        out.append(exp.Current())
        exp.Next()
    return out


def _shape_hash(shape) -> int:
    """OCP 7.8.x: TopoDS_Shape's __hash__ is keyed on TShape* identity."""
    return hash(shape)


def _unit_square_face(z: float) -> TopoDS_Face:
    polygon = BRepBuilderAPI_MakePolygon()
    polygon.Add(gp_Pnt(0.0, 0.0, z))
    polygon.Add(gp_Pnt(1.0, 0.0, z))
    polygon.Add(gp_Pnt(1.0, 1.0, z))
    polygon.Add(gp_Pnt(0.0, 1.0, z))
    polygon.Close()
    wire = polygon.Wire()
    return BRepBuilderAPI_MakeFace(wire).Face()


def test_prism_top_can_be_reused_as_next_prism_bottom():
    """Pre-sharing mechanism: A.LastShape() reused as B's bottom => shared TShape."""
    A_bottom = _unit_square_face(z=0.0)
    A_prism = BRepPrimAPI_MakePrism(A_bottom, gp_Vec(0.0, 0.0, 1.0))
    A_solid = A_prism.Shape()
    A_top = A_prism.LastShape()

    B_prism = BRepPrimAPI_MakePrism(A_top, gp_Vec(0.0, 0.0, 1.0))
    B_solid = B_prism.Shape()

    a_face_hashes = {_shape_hash(f) for f in _faces(A_solid)}
    b_face_hashes = {_shape_hash(f) for f in _faces(B_solid)}
    shared = a_face_hashes & b_face_hashes
    assert shared, (
        "A's top face was NOT reused as B's bottom face — pre-sharing via "
        "BRepPrimAPI_MakePrism(A_top, ...) did not produce shared TShape "
        "identity. The pre-shared-face design cannot work."
    )


def test_bopalgo_modified_works_per_solid_with_shared_interface_face():
    """The critical design-validation test.

    A (z=0..1) and B (z=1..2) share the z=1 interface face (B's bottom is
    A's top, set up by reusing A.LastShape()). C overlaps both. All three
    are added as separate BOPAlgo arguments. Modified(A) and Modified(B)
    must both produce non-empty history — i.e., per-argument fragmentation
    tracking still works when arguments share an internal face TShape.
    """
    A_bottom = _unit_square_face(z=0.0)
    A_prism = BRepPrimAPI_MakePrism(A_bottom, gp_Vec(0.0, 0.0, 1.0))
    A_solid = A_prism.Shape()
    A_top = A_prism.LastShape()

    B_prism = BRepPrimAPI_MakePrism(A_top, gp_Vec(0.0, 0.0, 1.0))
    B_solid = B_prism.Shape()

    a_face_hashes = {_shape_hash(f) for f in _faces(A_solid)}
    b_face_hashes = {_shape_hash(f) for f in _faces(B_solid)}
    assert a_face_hashes & b_face_hashes, "Precondition: A and B must share a face"

    C = BRepPrimAPI_MakeBox(gp_Pnt(0.5, 0.5, 0.5), gp_Pnt(1.5, 1.5, 1.5)).Solid()

    builder = BOPAlgo_Builder()
    builder.AddArgument(A_solid)
    builder.AddArgument(B_solid)
    builder.AddArgument(C)
    builder.Perform()

    a_mod = builder.Modified(A_solid)
    b_mod = builder.Modified(B_solid)

    assert not a_mod.IsEmpty(), (
        "BOPAlgo Modified(A_solid) empty — per-argument history broken when "
        "arguments share an interface face TShape. Design cannot work."
    )
    assert not b_mod.IsEmpty(), (
        "BOPAlgo Modified(B_solid) empty — per-argument history broken when "
        "arguments share an interface face TShape. Design cannot work."
    )


def test_shared_interface_survives_bop_with_inert_neighbor():
    """A.top = B.bottom by construction. D is far away; doesn't touch A or B.

    After BOP, A's and B's resulting pieces must still share at least one
    face TShape (the z=1 interface). Downstream interface tagging depends
    on this.
    """
    A_bottom = _unit_square_face(z=0.0)
    A_prism = BRepPrimAPI_MakePrism(A_bottom, gp_Vec(0.0, 0.0, 1.0))
    A_solid = A_prism.Shape()
    A_top = A_prism.LastShape()

    B_prism = BRepPrimAPI_MakePrism(A_top, gp_Vec(0.0, 0.0, 1.0))
    B_solid = B_prism.Shape()

    D = BRepPrimAPI_MakeBox(gp_Pnt(10, 10, 10), gp_Pnt(11, 11, 11)).Solid()

    builder = BOPAlgo_Builder()
    builder.AddArgument(A_solid)
    builder.AddArgument(B_solid)
    builder.AddArgument(D)
    builder.Perform()

    a_mod = builder.Modified(A_solid)
    b_mod = builder.Modified(B_solid)
    A_pieces = list(a_mod) if not a_mod.IsEmpty() else [A_solid]
    B_pieces = list(b_mod) if not b_mod.IsEmpty() else [B_solid]

    a_face_hashes = set()
    for p in A_pieces:
        a_face_hashes |= {_shape_hash(f) for f in _faces(p)}
    b_face_hashes = set()
    for p in B_pieces:
        b_face_hashes |= {_shape_hash(f) for f in _faces(p)}

    shared = a_face_hashes & b_face_hashes
    assert shared, (
        "After BOP with inert neighbor D, A and B no longer share any face "
        "TShape. The pre-sharing mechanism doesn't survive a BOP pass."
    )
