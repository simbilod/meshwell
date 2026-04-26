"""Tests for :class:`meshwell.interface_tag.InterfaceTag`."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import shapely
from shapely.geometry import LineString

import gmsh
from meshwell.interface_tag import InterfaceTag
from meshwell.model import ModelManager


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
