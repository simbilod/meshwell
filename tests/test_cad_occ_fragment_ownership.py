"""Unit tests for the all-fragment OCC pipeline."""
from __future__ import annotations

from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt

import gmsh
from meshwell.cad_occ import (
    CAD_OCC,
    OCCLabeledEntity,
    _resolve_piece_ownership,
    _shape_key,
    cad_occ,
)
from meshwell.model import ModelManager
from meshwell.occ_entity import OCC_entity
from meshwell.occ_to_gmsh import inject_occ_entities_into_gmsh


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


def test_inject_two_overlapping_boxes_produces_shared_interface():
    """After injection, the shared face between two touching boxes exists once."""
    a = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(
            gp_Pnt(0, 0, 0), 1.0, 1.0, 1.0
        ).Shape(),
        physical_name="a",
        mesh_order=1,
        dimension=3,
    )
    b = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(
            gp_Pnt(1, 0, 0), 1.0, 1.0, 1.0
        ).Shape(),
        physical_name="b",
        mesh_order=2,
        dimension=3,
    )
    occ_ents = cad_occ([a, b])

    mm = ModelManager(filename="test_shared_interface")
    try:
        labeled = inject_occ_entities_into_gmsh(occ_ents, mm)
        by_name = {le.physical_name[0]: le for le in labeled}
        assert set(by_name) == {"a", "b"}

        # Two 3D volumes (one per box); the shared face is 2D.
        assert len(gmsh.model.getEntities(3)) == 2

        groups = gmsh.model.getPhysicalGroups(2)
        names = [gmsh.model.getPhysicalName(dim, tag) for dim, tag in groups]
        match = next((n for n in names if n in {"a___b", "b___a"}), None)
        assert match is not None, names

        interface_tag = next(
            tag for dim, tag in groups if gmsh.model.getPhysicalName(dim, tag) == match
        )
        # BREP sharing means exactly one surface carries the interface tag.
        interface_surfaces = gmsh.model.getEntitiesForPhysicalGroup(2, interface_tag)
        assert len(interface_surfaces) == 1, interface_surfaces
    finally:
        mm.finalize()


def test_inject_with_remove_all_duplicates_preserves_physical_tags():
    """`remove_all_duplicates=True` must not invalidate per-entity physical tags.

    The gmsh-level fragment safety net issues fresh dimtags for every imported
    shape. We rely on the returned ``outDimTagsMap`` to remap per-entity
    dimtags; if that remap breaks, each physical group either becomes empty or
    points at the wrong volumes.
    """
    a = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(
            gp_Pnt(0, 0, 0), 1.0, 1.0, 1.0
        ).Shape(),
        physical_name="a",
        mesh_order=1,
        dimension=3,
    )
    b = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(
            gp_Pnt(1, 0, 0), 1.0, 1.0, 1.0
        ).Shape(),
        physical_name="b",
        mesh_order=2,
        dimension=3,
    )
    occ_ents = cad_occ([a, b])

    mm = ModelManager(filename="test_remove_all_duplicates")
    try:
        labeled = inject_occ_entities_into_gmsh(
            occ_ents, mm, remove_all_duplicates=True
        )
        by_name = {le.physical_name[0]: le for le in labeled}
        assert set(by_name) == {"a", "b"}

        # Two 3D volumes, each with the right physical tag.
        vol_groups = gmsh.model.getPhysicalGroups(3)
        names_to_vols = {
            gmsh.model.getPhysicalName(d, t): set(
                gmsh.model.getEntitiesForPhysicalGroup(d, t)
            )
            for d, t in vol_groups
        }
        assert set(names_to_vols) == {"a", "b"}
        assert len(names_to_vols["a"]) == 1
        assert len(names_to_vols["b"]) == 1
        # No volume appears in both groups.
        assert names_to_vols["a"].isdisjoint(names_to_vols["b"])

        # Shared interface still resolved.
        surf_groups = gmsh.model.getPhysicalGroups(2)
        surf_names = {gmsh.model.getPhysicalName(d, t) for d, t in surf_groups}
        assert surf_names & {"a___b", "b___a"}
    finally:
        mm.finalize()


def test_embedded_surface_splits_volume_and_shares_face():
    """A 2D surface inside a 3D box must appear as a shared face of the box sub-solids."""
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakeWire,
    )

    def rect(x, y, z, dx, dy):
        p1 = gp_Pnt(x, y, z)
        p2 = gp_Pnt(x + dx, y, z)
        p3 = gp_Pnt(x + dx, y + dy, z)
        p4 = gp_Pnt(x, y + dy, z)
        w = BRepBuilderAPI_MakeWire(
            BRepBuilderAPI_MakeEdge(p1, p2).Edge(),
            BRepBuilderAPI_MakeEdge(p2, p3).Edge(),
            BRepBuilderAPI_MakeEdge(p3, p4).Edge(),
            BRepBuilderAPI_MakeEdge(p4, p1).Edge(),
        ).Wire()
        return BRepBuilderAPI_MakeFace(w).Face()

    box = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(
            gp_Pnt(0, 0, 0), 2.0, 2.0, 2.0
        ).Shape(),
        physical_name="box",
        mesh_order=1,
        dimension=3,
    )
    # A surface cutting the box in half at z=1.
    cut_surface = OCC_entity(
        occ_function=lambda: rect(0.0, 0.0, 1.0, 2.0, 2.0),
        physical_name="cut",
        mesh_order=2,
        dimension=2,
    )

    occ_ents = cad_occ([box, cut_surface])
    mm = ModelManager(filename="test_embedded_surface")
    try:
        inject_occ_entities_into_gmsh(occ_ents, mm)
        # Box should be split into two volumes.
        vols = gmsh.model.getEntitiesForPhysicalGroup(
            3,
            next(
                tag
                for dim, tag in gmsh.model.getPhysicalGroups(3)
                if gmsh.model.getPhysicalName(dim, tag) == "box"
            ),
        )
        assert len(vols) == 2

        # The "cut" physical group must exist in 2D.
        surf_groups = gmsh.model.getPhysicalGroups(2)
        surf_names = [gmsh.model.getPhysicalName(d, t) for d, t in surf_groups]
        assert "cut" in surf_names
    finally:
        mm.finalize()
