"""Tests for the XAO writer and the use_xao injection path."""
from __future__ import annotations

from dataclasses import dataclass

import shapely
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt

import gmsh
from meshwell.cad_occ import cad_occ
from meshwell.model import ModelManager
from meshwell.occ_entity import OCC_entity
from meshwell.occ_to_gmsh import inject_occ_entities_into_gmsh
from meshwell.occ_xao_writer import write_xao
from meshwell.polyline import PolyLine
from meshwell.polyprism import PolyPrism
from meshwell.polysurface import PolySurface


@dataclass
class _FakeEnt:
    shapes: list
    physical_name: tuple = ("foo",)
    index: int = 0
    keep: bool = True
    dim: int = 3
    mesh_order: float | None = None


def test_xao_writer_produces_single_self_contained_file(tmp_path):
    """Inline-CDATA BREP means the XAO is the only artifact gmsh needs."""
    b1 = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1, 1, 1).Shape()
    b2 = BRepPrimAPI_MakeBox(gp_Pnt(1, 0, 0), 1, 1, 1).Shape()
    ents = [
        _FakeEnt(shapes=[b1], physical_name=("A",), index=0),
        _FakeEnt(shapes=[b2], physical_name=("B",), index=1),
    ]
    xao = tmp_path / "m.xao"
    markers = write_xao(ents, xao)

    assert xao.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.xao"]
    assert set(markers.values()) == {
        "_meshwell_xao_marker_0",
        "_meshwell_xao_marker_1",
    }

    content = xao.read_text()
    assert "<![CDATA[" in content
    assert "DBRep_DrawableShape" in content  # BREP header marker

    gmsh.initialize()
    try:
        gmsh.open(str(xao))
        gmsh.model.occ.synchronize()

        name_to_ents = {}
        for d, t in gmsh.model.getPhysicalGroups(3):
            name_to_ents[gmsh.model.getPhysicalName(d, t)] = list(
                gmsh.model.getEntitiesForPhysicalGroup(d, t)
            )
        assert set(name_to_ents) == set(markers.values())
        assert all(len(v) == 1 for v in name_to_ents.values())
    finally:
        gmsh.finalize()


def test_inject_use_xao_matches_legacy_path_physical_tags():
    """XAO path produces the same physical-group structure as the legacy path.

    Uses a mixed 3D/2D/1D scene and asserts mass + centroid parity across
    both import paths.
    """
    # Same scene as the multi-entity test, pared down.
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

    def run(use_xao: bool) -> dict[str, tuple[float, tuple[float, float, float]]]:
        occ_ents = cad_occ([cut, prism_A, prism_B, wire])
        mm = ModelManager(filename=f"test_use_xao_{use_xao}")
        result: dict[str, tuple[float, tuple[float, float, float]]] = {}
        try:
            inject_occ_entities_into_gmsh(occ_ents, mm, use_xao=use_xao)
            for dim in (1, 2, 3):
                for d, t in gmsh.model.getPhysicalGroups(dim):
                    name = gmsh.model.getPhysicalName(d, t)
                    tot = 0.0
                    sx = sy = sz = 0.0
                    for tag in gmsh.model.getEntitiesForPhysicalGroup(d, t):
                        m = gmsh.model.occ.getMass(d, tag)
                        cx, cy, cz = gmsh.model.occ.getCenterOfMass(d, tag)
                        tot += m
                        sx += cx * m
                        sy += cy * m
                        sz += cz * m
                    if tot > 0:
                        result[name] = (tot, (sx / tot, sy / tot, sz / tot))
        finally:
            mm.finalize()
        return result

    legacy = run(use_xao=False)
    xao = run(use_xao=True)

    assert set(legacy) == set(xao), (set(legacy), set(xao))
    for name in legacy:
        mass_l, c_l = legacy[name]
        mass_x, c_x = xao[name]
        assert abs(mass_l - mass_x) < 1e-6, (name, mass_l, mass_x)
        for a, b in zip(c_l, c_x):
            assert abs(a - b) < 1e-4, (name, c_l, c_x)


def test_inject_use_xao_preserves_per_entity_volumes():
    """XAO marker lookup must produce disjoint per-entity volume groups."""
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

    mm = ModelManager(filename="test_use_xao_volumes")
    try:
        inject_occ_entities_into_gmsh(occ_ents, mm, use_xao=True)
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
        # Shared interface still discovered.
        surf_names = {
            gmsh.model.getPhysicalName(d, t) for d, t in gmsh.model.getPhysicalGroups(2)
        }
        assert surf_names & {"a___b", "b___a"}
    finally:
        mm.finalize()
