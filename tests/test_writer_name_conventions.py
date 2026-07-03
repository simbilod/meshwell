"""Writer semantics must key on explicit flags, not name substrings.

Regression tests for WP4 Task 4: the OCC XAO writer used to decide real
semantics by inspecting entity NAMES:

* any entity whose physical name contained ``"iface"`` had its interface
  groups silently dropped (meant to catch :class:`InterfaceTag` helpers);
* any entity whose names all started with ``"__cohort_"`` was treated as a
  purely-synthetic structured-pipeline annotator and skipped.

Both conventions collide with legitimate user names (``interface_oxide``,
``__cohort_trap``). The writer now keys on ``is_interface_helper`` /
``synthetic_names`` carried on the labeled-entity record instead, so user
names are semantically inert.
"""
from __future__ import annotations

from pathlib import Path

import gmsh
import shapely

from meshwell.cad_occ import cad_occ
from meshwell.interface_tag import InterfaceTag
from meshwell.occ_xao_writer import write_xao
from meshwell.polyprism import PolyPrism


def _physical_names_in_xao(xao_path: Path) -> set[str]:
    """Open ``xao_path`` in gmsh and return the set of physical names."""
    gmsh.initialize()
    try:
        gmsh.open(str(xao_path))
        gmsh.model.occ.synchronize()
        return {
            gmsh.model.getPhysicalName(d, t) for d, t in gmsh.model.getPhysicalGroups()
        }
    finally:
        gmsh.finalize()


def test_user_entity_named_iface_gets_interface_group(tmp_path):
    """A user entity whose name contains 'iface' still gets its A___B group.

    Headline failure of the old ``"iface" in name`` substring check: an
    entity legitimately named ``oxide_iface`` had every one of its
    interfaces silently dropped.

    (The brief's example ``interface_oxide`` does NOT actually contain the
    substring ``"iface"`` -- "interface" has no ``if`` adjacency -- so it was
    never affected; ``oxide_iface`` is the true triggering case and is what
    this test pins. ``interface_oxide`` is covered separately below.)
    """
    A = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    B = shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)])
    buffers = {0.0: 0.0, 2.0: 0.0}
    labeled = cad_occ(
        [
            PolyPrism(
                polygons=A, buffers=buffers, physical_name="oxide_iface", mesh_order=1
            ),
            PolyPrism(polygons=B, buffers=buffers, physical_name="B", mesh_order=2),
        ]
    )
    xao = tmp_path / "oxide_iface.xao"
    write_xao(labeled, xao)
    names = _physical_names_in_xao(xao)

    assert {"oxide_iface", "B"} <= names
    assert (
        "oxide_iface___B" in names or "B___oxide_iface" in names
    ), f"user entity 'oxide_iface' lost its interface group: {sorted(names)}"


def test_user_entity_named_interface_oxide_gets_interface_group(tmp_path):
    """The brief's ``interface_oxide`` name also keeps its interface group.

    ``interface_oxide`` does not contain ``"iface"`` so it never tripped the
    old substring bug, but pinning it guards against a regression if the
    check were ever broadened.
    """
    A = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    B = shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)])
    buffers = {0.0: 0.0, 2.0: 0.0}
    labeled = cad_occ(
        [
            PolyPrism(
                polygons=A,
                buffers=buffers,
                physical_name="interface_oxide",
                mesh_order=1,
            ),
            PolyPrism(polygons=B, buffers=buffers, physical_name="B", mesh_order=2),
        ]
    )
    xao = tmp_path / "interface_oxide.xao"
    write_xao(labeled, xao)
    names = _physical_names_in_xao(xao)

    assert {"interface_oxide", "B"} <= names
    assert (
        "interface_oxide___B" in names or "B___interface_oxide" in names
    ), f"user entity 'interface_oxide' lost its interface group: {sorted(names)}"


def test_user_entity_named_cohort_prefix_gets_interface_group(tmp_path):
    """A user entity named '__cohort_trap' is NOT mistaken for a synthetic annotator.

    The old ``name.startswith('__cohort_')`` check classified it as a
    purely-synthetic structured-pipeline companion and skipped its
    interfaces.
    """
    A = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    B = shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)])
    buffers = {0.0: 0.0, 2.0: 0.0}
    labeled = cad_occ(
        [
            PolyPrism(
                polygons=A, buffers=buffers, physical_name="__cohort_trap", mesh_order=1
            ),
            PolyPrism(polygons=B, buffers=buffers, physical_name="B", mesh_order=2),
        ]
    )
    xao = tmp_path / "cohort_trap.xao"
    write_xao(labeled, xao)
    names = _physical_names_in_xao(xao)

    assert {"__cohort_trap", "B"} <= names
    assert (
        "__cohort_trap___B" in names or "B_____cohort_trap" in names
    ), f"user entity '__cohort_trap' lost its interface group: {sorted(names)}"


def test_interface_helper_flag_suppresses_interface_pairing(tmp_path):
    """InterfaceTag helpers still keep their own group but form no ___iface pair.

    Pins the FLAG (``is_interface_helper``) rather than the substring: the
    helper owns its ``iface`` group, but the writer must not glue it into a
    neighbour pair (no ``A___iface`` / ``iface___B`` etc.).
    """
    from shapely.geometry import LineString

    A = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    B = shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)])
    buffers = {0.0: 0.0, 1.0: 0.0}
    labeled = cad_occ(
        [
            PolyPrism(polygons=A, buffers=buffers, physical_name="A", mesh_order=1),
            PolyPrism(polygons=B, buffers=buffers, physical_name="B", mesh_order=2),
            InterfaceTag(
                linestrings=LineString([(5, 0), (5, 5)]),
                zmin=0.0,
                zmax=1.0,
                physical_name="iface",
                mesh_order=3,
            ),
        ]
    )
    xao = tmp_path / "helper.xao"
    write_xao(labeled, xao)
    names = _physical_names_in_xao(xao)

    assert "iface" in names
    leaked = {n for n in names if "___" in n and "iface" in n}
    assert not leaked, f"InterfaceTag helper leaked into an interface pair: {leaked}"
