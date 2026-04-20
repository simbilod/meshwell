"""Tests for the post-fragmentation TShape canonicalization pass."""
from __future__ import annotations

from dataclasses import dataclass

from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
from OCP.TopTools import TopTools_ShapeMapHasher

from meshwell.cad_occ import CAD_OCC
from meshwell.occ_canonicalize import canonicalize_topology
from meshwell.occ_entity import OCC_entity

_HASHER = TopTools_ShapeMapHasher()


@dataclass
class _FakeEnt:
    """Minimal stand-in for OCCLabeledEntity — just needs `.shapes` and `.dim`."""

    shapes: list
    dim: int = 3


def _unique_tshapes(entities, topabs) -> int:
    seen = set()
    for ent in entities:
        for s in ent.shapes:
            e = TopExp_Explorer(s, topabs)
            while e.More():
                seen.add(_HASHER(e.Current()))
                e.Next()
    return len(seen)


def test_canonicalize_two_touching_boxes_shares_all_levels():
    """Two touching boxes share the interface face, edges, and vertices.

    Built independently so the coincident sub-shapes start with distinct
    TShapes; the canonicalization pass must unify them.
    """
    b1 = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1, 1, 1).Shape()
    b2 = BRepPrimAPI_MakeBox(gp_Pnt(1, 0, 0), 1, 1, 1).Shape()
    ents = [_FakeEnt(shapes=[b1]), _FakeEnt(shapes=[b2])]

    pre_v = _unique_tshapes(ents, TopAbs_VERTEX)
    pre_e = _unique_tshapes(ents, TopAbs_EDGE)
    pre_f = _unique_tshapes(ents, TopAbs_FACE)

    stats = canonicalize_topology(ents, point_tolerance=1e-3)
    assert stats == {"vertices": 4, "edges": 4, "faces": 1}

    # 4 interface vertices, 4 interface edges, 1 interface face now shared.
    assert _unique_tshapes(ents, TopAbs_VERTEX) == pre_v - 4
    assert _unique_tshapes(ents, TopAbs_EDGE) == pre_e - 4
    assert _unique_tshapes(ents, TopAbs_FACE) == pre_f - 1


def test_canonicalize_handles_sub_tolerance_drift():
    """Gap < point_tolerance should canonicalize; gap >= tolerance should not."""
    # Below tolerance boundary → merges.
    b1 = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1, 1, 1).Shape()
    b2 = BRepPrimAPI_MakeBox(gp_Pnt(1.0 + 1e-4, 0, 0), 1, 1, 1).Shape()
    ents = [_FakeEnt(shapes=[b1]), _FakeEnt(shapes=[b2])]
    stats = canonicalize_topology(ents, point_tolerance=1e-3)
    assert stats["vertices"] == 4

    # At/above the rounding boundary → does not force-merge distinct geometry.
    b3 = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1, 1, 1).Shape()
    b4 = BRepPrimAPI_MakeBox(gp_Pnt(1.0 + 5e-3, 0, 0), 1, 1, 1).Shape()
    ents = [_FakeEnt(shapes=[b3]), _FakeEnt(shapes=[b4])]
    stats = canonicalize_topology(ents, point_tolerance=1e-3)
    assert stats["vertices"] == 0


def test_canonicalize_noop_on_disjoint_entities():
    """Boxes that genuinely don't share geometry must remain untouched."""
    b1 = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1, 1, 1).Shape()
    b2 = BRepPrimAPI_MakeBox(gp_Pnt(3, 0, 0), 1, 1, 1).Shape()
    ents = [_FakeEnt(shapes=[b1]), _FakeEnt(shapes=[b2])]
    pre = (
        _unique_tshapes(ents, TopAbs_VERTEX),
        _unique_tshapes(ents, TopAbs_EDGE),
        _unique_tshapes(ents, TopAbs_FACE),
    )
    stats = canonicalize_topology(ents, point_tolerance=1e-3)
    assert stats == {"vertices": 0, "edges": 0, "faces": 0}
    post = (
        _unique_tshapes(ents, TopAbs_VERTEX),
        _unique_tshapes(ents, TopAbs_EDGE),
        _unique_tshapes(ents, TopAbs_FACE),
    )
    assert pre == post


def test_canonicalize_topology_opt_in_on_CAD_OCC():
    """CAD_OCC runs the canonicalization pass end-to-end.

    ``canonicalize_topology=True`` leaves two touching boxes sharing their
    x=1 interface face.
    """
    a = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1, 1, 1).Shape(),
        physical_name="a",
        mesh_order=1,
        dimension=3,
    )
    b = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(gp_Pnt(1, 0, 0), 1, 1, 1).Shape(),
        physical_name="b",
        mesh_order=2,
        dimension=3,
    )

    processor = CAD_OCC(point_tolerance=1e-3, canonicalize_topology=True)
    result = processor.process_entities([a, b])
    assert len(result) == 2

    # The canonicalization pass runs after BOPAlgo. For adjacent boxes BOPAlgo
    # already shares the interface face; the canonicalization is a no-op here
    # (stats would be zero) but the resulting entities must still be coherent.
    n_faces = _unique_tshapes(result, TopAbs_FACE)
    # 2 boxes x 6 faces - 1 shared face = 11.
    assert n_faces == 11
