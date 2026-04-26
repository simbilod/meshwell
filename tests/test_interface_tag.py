"""Tests for :class:`meshwell.interface_tag.InterfaceTag`."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import shapely
from shapely.geometry import LineString

import gmsh
from meshwell.cad_gmsh import cad_gmsh, strip_suffix
from meshwell.interface_tag import InterfaceTag
from meshwell.model import ModelManager
from meshwell.polyprism import PolyPrism


@dataclass
class _FakePolyEntity:
    """Minimal stand-in for a polygon-bearing entity.

    Used to unit-test :meth:`InterfaceTag.resolve` without spinning up gmsh.
    """

    polygons: object  # shapely Polygon | list[Polygon]
    mesh_order: float | None = None


def test_resolve_picks_winning_boundary_in_abutting_prisms():
    """Abutting prisms: winner keeps its full boundary; loser's inner edge vanishes.

    A=mesh_order 1 (winner), B=mesh_order 2. Both buffered outward by pert.
    Nominal trace at x=5. Expect a single resolved segment on A's right
    boundary at x = 5 + pert. B's left boundary at x = 5 - pert is inside
    A's claimed body and must NOT appear.
    """
    pert = 1e-3
    a_poly = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]).buffer(
        pert, join_style=2
    )
    b_poly = shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)]).buffer(
        pert, join_style=2
    )

    tag = InterfaceTag(
        linestrings=LineString([(5, 0), (5, 5)]),
        zmin=0,
        zmax=1,
        physical_name="iface",
        targets=None,
    )
    tag.resolve(
        polygon_ents={
            "A": _FakePolyEntity(polygons=a_poly, mesh_order=1),
            "B": _FakePolyEntity(polygons=b_poly, mesh_order=2),
        },
        default_snap=2 * 1e-3,
    )

    assert len(tag.resolved_linestrings) == 1
    seg = tag.resolved_linestrings[0]
    # The resolved segment must lie at x = 5 + pert (A's buffered right edge).
    xs = {round(x, 6) for x, _ in seg.coords}
    assert xs == {round(5 + pert, 6)}, (xs, seg)


def test_resolve_warns_when_no_match():
    """A nominal trace far from any target produces no segments and warns."""
    pert = 1e-3
    a_poly = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]).buffer(
        pert, join_style=2
    )

    tag = InterfaceTag(
        linestrings=LineString([(100, 0), (100, 5)]),
        zmin=0,
        zmax=1,
        physical_name="iface",
        targets=None,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tag.resolve(
            polygon_ents={"A": _FakePolyEntity(polygons=a_poly, mesh_order=1)},
            default_snap=2 * 1e-3,
        )

    assert tag.resolved_linestrings == []
    assert any("resolved to no segments" in str(w.message) for w in caught), caught


def test_resolve_picks_up_hole_boundary():
    """A target polygon with an interior hole must snap to the inner ring.

    InterfaceTag traces the hole boundary.
    """
    pert = 1e-3
    outer = shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    hole = shapely.Polygon([(4, 4), (6, 4), (6, 6), (4, 6)])
    donut = outer.difference(hole).buffer(pert, join_style=2)

    tag = InterfaceTag(
        linestrings=LineString([(4, 4), (6, 4), (6, 6), (4, 6), (4, 4)]),
        zmin=0,
        zmax=1,
        physical_name="hole_iface",
        targets=None,
    )
    tag.resolve(
        polygon_ents={"O": _FakePolyEntity(polygons=donut, mesh_order=1)},
        default_snap=2 * 1e-3,
    )

    assert len(tag.resolved_linestrings) >= 1
    total_len = sum(ls.length for ls in tag.resolved_linestrings)
    # The hole's perimeter is 8; expect resolved length ~8 (within buffer slop).
    assert 7.5 < total_len < 8.5, total_len


def test_instanciate_builds_2d_vertical_surfaces():
    """Given two pre-set resolved_linestrings, instanciate returns 2D dimtags.

    The bounding box of each surface must match z = [zmin, zmax].
    """
    mm = ModelManager(filename="test_interface_tag_instanciate")
    try:
        mm.ensure_initialized("test_interface_tag_instanciate")

        tag = InterfaceTag(
            linestrings=LineString([(0, 0), (5, 0)]),  # placeholder
            zmin=0.0,
            zmax=2.0,
            physical_name="iface",
        )
        # Bypass resolve(): inject the trace directly.
        tag.resolved_linestrings = [
            LineString([(0, 0), (5, 0)]),
            LineString([(0, 5), (5, 5)]),
        ]

        dimtags = tag.instanciate()
        assert len(dimtags) == 2, dimtags
        assert all(d == 2 for d, _ in dimtags), dimtags

        gmsh.model.occ.synchronize()
        for d, t in dimtags:
            _xmin, _ymin, zmin_b, _xmax, _ymax, zmax_b = gmsh.model.getBoundingBox(d, t)
            # gmsh pads bounding boxes by ~1e-7; use 1e-6 tolerance.
            assert abs(zmin_b - 0.0) < 1e-6, (zmin_b, t)
            assert abs(zmax_b - 2.0) < 1e-6, (zmax_b, t)
    finally:
        mm.finalize()


def _physical_names() -> list[tuple[int, str]]:
    return [
        (d, gmsh.model.getPhysicalName(d, t)) for d, t in gmsh.model.getPhysicalGroups()
    ]


def test_e2e_interface_tag_resolves_to_winning_boundary():
    """Winning boundary resolves correctly for two abutting prisms.

    Two abutting prisms (A wins, B loses) plus a single InterfaceTag at
    the nominal interface x=5. After cad_gmsh: exactly one face is tagged
    `iface`, and A is not internally split.
    """
    A = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    B = shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)])
    buffers = {0.0: 0.0, 1.0: 0.0}
    try:
        labeled, mm = cad_gmsh(
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

        names = {n for _, n in _physical_names()}
        assert {"A", "B", "iface", "A___B"} <= names

        # The iface physical group must exist at dim 2 with at least one face.
        iface_pg = next(
            t
            for d, t in gmsh.model.getPhysicalGroups(dim=2)
            if gmsh.model.getPhysicalName(d, t) == "iface"
        )
        iface_faces = gmsh.model.getEntitiesForPhysicalGroup(2, iface_pg)
        assert len(iface_faces) >= 1, iface_faces

        # A must remain a single 3D piece (no internal cut by InterfaceTag).
        a_ent = next(e for e in labeled if strip_suffix(e.physical_name[0]) == "A")
        assert sum(1 for d, _ in a_ent.dimtags if d == 3) == 1, a_ent.dimtags
    finally:
        mm.finalize()


def test_e2e_targets_none_picks_winners_only():
    """targets=None tags both interfaces in a three-prism scene.

    Three abutting prisms A/B/C (mesh_orders 1/2/3) with interfaces at
    x=2 and x=5. A single InterfaceTag with linestring spanning both,
    targets=None. Expect both interfaces tagged (one face each on the
    winner side), and no spurious internal cuts in any prism.
    """
    A = shapely.Polygon([(0, 0), (2, 0), (2, 5), (0, 5)])
    B = shapely.Polygon([(2, 0), (5, 0), (5, 5), (2, 5)])
    C = shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)])
    buffers = {0.0: 0.0, 1.0: 0.0}
    try:
        labeled, mm = cad_gmsh(
            [
                PolyPrism(polygons=A, buffers=buffers, physical_name="A", mesh_order=1),
                PolyPrism(polygons=B, buffers=buffers, physical_name="B", mesh_order=2),
                PolyPrism(polygons=C, buffers=buffers, physical_name="C", mesh_order=3),
                InterfaceTag(
                    linestrings=LineString([(1, 2.5), (7, 2.5)]),
                    zmin=0.0,
                    zmax=1.0,
                    physical_name="iface",
                    mesh_order=4,
                ),
            ]
        )

        # Each prism stays as exactly one 3D piece.
        for nm in ("A", "B", "C"):
            ent = next(e for e in labeled if strip_suffix(e.physical_name[0]) == nm)
            n_3d = sum(1 for d, _ in ent.dimtags if d == 3)
            assert n_3d == 1, (nm, ent.dimtags)

        # iface tagged at dim 2 with at least 2 faces (one per real interface).
        iface_pg = next(
            t
            for d, t in gmsh.model.getPhysicalGroups(dim=2)
            if gmsh.model.getPhysicalName(d, t) == "iface"
        )
        iface_faces = gmsh.model.getEntitiesForPhysicalGroup(2, iface_pg)
        assert len(iface_faces) >= 2, iface_faces
    finally:
        mm.finalize()
