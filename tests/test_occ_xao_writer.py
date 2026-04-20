"""Tests for the fully-tagged XAO writer + injection."""

from __future__ import annotations

import shapely
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt

import gmsh
from meshwell.cad_occ import cad_occ
from meshwell.model import ModelManager
from meshwell.occ_entity import OCC_entity
from meshwell.occ_xao_writer import inject_occ_entities_into_gmsh, write_xao
from meshwell.polyline import PolyLine
from meshwell.polyprism import PolyPrism
from meshwell.polysurface import PolySurface


def test_xao_writer_produces_single_self_contained_file(tmp_path):
    """XAO includes full tagging: entities, A___B interface, A___None exteriors.

    Runs the inputs through ``cad_occ`` so BOPAlgo canonicalises the shared
    TShape at x=1 -- only then does the writer's interface detector fire.
    """
    a = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1, 1, 1).Shape(),
        physical_name="A",
        mesh_order=1,
        dimension=3,
    )
    b = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(gp_Pnt(1, 0, 0), 1, 1, 1).Shape(),
        physical_name="B",
        mesh_order=2,
        dimension=3,
    )
    ents = cad_occ([a, b])
    xao = tmp_path / "m.xao"
    write_xao(ents, xao)

    assert xao.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.xao"]

    content = xao.read_text()
    assert "<![CDATA[" in content
    assert "DBRep_DrawableShape" in content

    gmsh.initialize()
    try:
        gmsh.open(str(xao))
        gmsh.model.occ.synchronize()

        vol_names = {
            gmsh.model.getPhysicalName(d, t) for d, t in gmsh.model.getPhysicalGroups(3)
        }
        assert {"A", "B"} <= vol_names

        surf_names = {
            gmsh.model.getPhysicalName(d, t) for d, t in gmsh.model.getPhysicalGroups(2)
        }
        assert surf_names & {"A___B", "B___A"}
        assert {"A___None", "B___None"} <= surf_names
    finally:
        gmsh.finalize()


def test_inject_full_mixed_scene():
    """Mixed 3D/2D/1D entities end up with correct physical groups + masses."""
    square_A = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    square_B = shapely.Polygon([(1, 0), (2, 0), (2, 1), (1, 1), (1, 0)])
    prism_A = PolyPrism(
        polygons=square_A,
        buffers={0.0: 0.0, 2.0: 0.0},
        physical_name="A",
        mesh_order=1,
    )
    prism_B = PolyPrism(
        polygons=square_B,
        buffers={0.0: 0.0, 2.0: 0.0},
        physical_name="B",
        mesh_order=2,
    )
    cut = PolySurface(
        polygons=square_A,
        physical_name="cut",
        mesh_order=0,
        translation=(0.0, 0.0, 1.0),
    )
    wire = PolyLine(
        linestrings=shapely.LineString([(6, 0), (7, 0)]),
        physical_name="wire",
        mesh_order=3,
    )

    occ_ents = cad_occ([cut, prism_A, prism_B, wire])
    mm = ModelManager(filename="test_xao_mixed")
    try:
        inject_occ_entities_into_gmsh(occ_ents, mm)
        all_names = {
            gmsh.model.getPhysicalName(d, t)
            for dim in (0, 1, 2, 3)
            for d, t in gmsh.model.getPhysicalGroups(dim)
        }
        assert {"A", "B", "cut", "wire"} <= all_names

        # Volume totals per entity match hand-computed values.
        vol_by_name: dict[str, float] = {}
        for d, t in gmsh.model.getPhysicalGroups(3):
            name = gmsh.model.getPhysicalName(d, t)
            total = sum(
                gmsh.model.occ.getMass(d, tag)
                for tag in gmsh.model.getEntitiesForPhysicalGroup(d, t)
            )
            vol_by_name[name] = total
        assert abs(vol_by_name["A"] - 2.0) < 1e-6
        assert abs(vol_by_name["B"] - 2.0) < 1e-6

        # Shared A___B interface exists, and it is area ~ 2.
        surf_groups = gmsh.model.getPhysicalGroups(2)
        interface_name = next(
            (
                gmsh.model.getPhysicalName(d, t)
                for d, t in surf_groups
                if gmsh.model.getPhysicalName(d, t) in {"A___B", "B___A"}
            ),
            None,
        )
        assert interface_name is not None
        interface_tag = next(
            t
            for d, t in surf_groups
            if gmsh.model.getPhysicalName(d, t) == interface_name
        )
        ifc_area = sum(
            gmsh.model.occ.getMass(2, tag)
            for tag in gmsh.model.getEntitiesForPhysicalGroup(2, interface_tag)
        )
        assert abs(ifc_area - 2.0) < 1e-6
    finally:
        mm.finalize()


def test_inject_two_touching_boxes_disjoint_volumes():
    """Per-entity volume lookup produces disjoint gmsh-entity sets."""
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

    mm = ModelManager(filename="test_xao_volumes")
    try:
        inject_occ_entities_into_gmsh(occ_ents, mm)
        names_to_vols = {
            gmsh.model.getPhysicalName(d, t): set(
                gmsh.model.getEntitiesForPhysicalGroup(d, t)
            )
            for d, t in gmsh.model.getPhysicalGroups(3)
        }
        assert set(names_to_vols) == {"a", "b"}
        assert len(names_to_vols["a"]) == 1
        assert len(names_to_vols["b"]) == 1
        assert names_to_vols["a"].isdisjoint(names_to_vols["b"])
        surf_names = {
            gmsh.model.getPhysicalName(d, t) for d, t in gmsh.model.getPhysicalGroups(2)
        }
        assert surf_names & {"a___b", "b___a"}
    finally:
        mm.finalize()


def test_keep_false_entity_removed_but_interface_named():
    """A keep=False helper still names boundaries it shares with kept entities."""
    kept = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1, 1, 1).Shape(),
        physical_name="kept",
        mesh_order=2,
        dimension=3,
    )
    helper = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(gp_Pnt(1, 0, 0), 1, 1, 1).Shape(),
        physical_name="helper",
        mesh_order=1,
        mesh_bool=False,  # keep=False in OCCLabeledEntity.
        dimension=3,
    )
    occ_ents = cad_occ([kept, helper])
    mm = ModelManager(filename="test_xao_keep_false")
    try:
        inject_occ_entities_into_gmsh(occ_ents, mm)
        vol_names = {
            gmsh.model.getPhysicalName(d, t) for d, t in gmsh.model.getPhysicalGroups(3)
        }
        # Only the kept entity is tagged as a 3D physical group.
        assert vol_names == {"kept"}
        # gmsh has only one volume now.
        assert len(gmsh.model.getEntities(3)) == 1
        # The kept___helper interface (or helper___kept) still exists as a
        # named dim-2 group even though the helper solid was removed.
        surf_names = {
            gmsh.model.getPhysicalName(d, t) for d, t in gmsh.model.getPhysicalGroups(2)
        }
        assert surf_names & {"kept___helper", "helper___kept"}
    finally:
        mm.finalize()
