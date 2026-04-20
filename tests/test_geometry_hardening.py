"""Regression tests for entity-ingestion input hardening.

Covers seam-duplicate stripping, canonical seam rotation, and
shapely.set_precision snapping.
"""
from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import Polygon

from meshwell.geometry_entity import (
    GeometryEntity,
    _find_canonical_seam,
    _rotate_closed,
    _strip_consecutive_duplicates,
)
from meshwell.polyline import PolyLine
from meshwell.polyprism import PolyPrism
from meshwell.polysurface import PolySurface


def _rounded_rect_coords(w, h, r, n_arc=8):
    hw, hh = w / 2, h / 2
    specs = [
        ((hw - r, hh - r), 0.0),
        ((-hw + r, hh - r), np.pi / 2),
        ((-hw + r, -hh + r), np.pi),
        ((hw - r, -hh + r), 3 * np.pi / 2),
    ]
    coords = []
    for (cx, cy), a0 in specs:
        coords.extend(
            (cx + r * np.cos(a), cy + r * np.sin(a))
            for a in np.linspace(a0, a0 + np.pi / 2, n_arc + 1)
        )
    coords.append(coords[0])
    return coords


def test_strip_consecutive_duplicates_open():
    verts = [(0, 0, 0), (1, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    out = _strip_consecutive_duplicates(verts, tolerance=1e-3)
    assert out == [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]


def test_strip_consecutive_duplicates_preserves_closing_vertex():
    verts = [(0, 0, 0), (1, 0, 0), (1, 0, 0), (1, 1, 0), (0, 0, 0)]
    out = _strip_consecutive_duplicates(verts, tolerance=1e-3)
    assert out[0] == out[-1] == (0, 0, 0)
    assert len(out) == 4  # opener + 2 unique mids + closing repeat


def test_strip_consecutive_duplicates_within_tolerance():
    verts = [(0, 0, 0), (0.0001, 0, 0), (1, 0, 0)]
    out = _strip_consecutive_duplicates(verts, tolerance=1e-3)
    assert out == [(0, 0, 0), (1, 0, 0)]


def test_decompose_vertices_drops_zero_length_segment():
    entity = GeometryEntity(point_tolerance=1e-3)
    verts = [(0, 0, 0), (1, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0)]
    segs = entity.decompose_vertices(verts, identify_arcs=False)
    assert len(segs) == 4
    assert all(
        abs(s.points[0][0] - s.points[1][0]) + abs(s.points[0][1] - s.points[1][1]) > 0
        for s in segs
    )


def test_rotate_closed_basic():
    verts = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0)]
    rotated = _rotate_closed(verts, 2)
    assert rotated[0] == (1, 1, 0)
    assert rotated[0] == rotated[-1]
    assert len(rotated) == len(verts)


def test_find_canonical_seam_picks_sharpest_corner():
    # Square with a clearly sharpest corner at index 1: prev (0,0), cur (1,0),
    # next (1,0.1) — almost-right-angle but subtly sharper than others.
    verts = [(0, 0, 0), (1, 0, 0), (1, 0.1, 0), (0, 0.1, 0), (0, 0, 0)]
    idx = _find_canonical_seam(verts)
    # All four corners are right angles; with lex-min tiebreaker the seam
    # lands on the vertex with the smallest coordinate tuple, i.e. (0,0,0).
    assert verts[idx] == (0, 0, 0)


def test_decompose_rotation_invariant_on_rounded_rect():
    """Rotated coord sequences of a rounded rectangle produce identical arc partitions."""
    entity = GeometryEntity(point_tolerance=1e-3)
    base_2d = _rounded_rect_coords(w=4.0, h=3.0, r=0.6, n_arc=8)
    base = [(x, y, 0.0) for x, y in base_2d]
    # Rotate so the sequence starts mid-arc.
    core = base[:-1]
    rotated = core[4:] + core[:4]
    rotated.append(rotated[0])

    segs_base = entity.decompose_vertices(base, identify_arcs=True, min_arc_points=4)
    segs_rot = entity.decompose_vertices(rotated, identify_arcs=True, min_arc_points=4)

    arcs_base = [s for s in segs_base if s.is_arc]
    arcs_rot = [s for s in segs_rot if s.is_arc]
    assert len(arcs_base) == 4
    assert len(arcs_rot) == 4

    def sig(s):
        return (tuple(round(c, 3) for c in s.center), round(s.radius, 3))

    assert sorted(sig(a) for a in arcs_base) == sorted(sig(a) for a in arcs_rot)


def test_polysurface_applies_set_precision():
    """Off-grid input coords get snapped to the tolerance grid at ingestion."""
    # A square with each corner shifted by a sub-tolerance amount.
    coords = [
        (0.0000007, 0.0),
        (1.0000003, 0.0000002),
        (1.0, 1.0000004),
        (0.0, 1.0),
        (0.0000007, 0.0),
    ]
    ps = PolySurface(polygons=Polygon(coords), point_tolerance=1e-3)
    # After snapping to 1e-3 grid, all the micro-offsets collapse.
    xs = [round(x, 6) for x, _ in ps.polygons[0].exterior.coords]
    ys = [round(y, 6) for _, y in ps.polygons[0].exterior.coords]
    assert set(xs) == {0.0, 1.0}
    assert set(ys) == {0.0, 1.0}


def test_polyline_applies_set_precision():
    ls = shapely.geometry.LineString(
        [(0.0000002, 0.0), (1.0000004, 0.0), (1.0000004, 1.0000005)]
    )
    pl = PolyLine(linestrings=ls, point_tolerance=1e-3)
    coords = [tuple(round(c, 6) for c in pt) for pt in pl.linestrings[0].coords]
    assert coords == [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]


def test_polyprism_applies_set_precision():
    coords = [
        (0.0000007, 0.0),
        (1.0000003, 0.0000002),
        (1.0, 1.0000004),
        (0.0, 1.0),
        (0.0000007, 0.0),
    ]
    pp = PolyPrism(
        polygons=Polygon(coords),
        buffers={0.0: 0.0, 1.0: 0.0},
        point_tolerance=1e-3,
    )
    xs = [round(x, 6) for x, _ in pp.polygons.exterior.coords]
    ys = [round(y, 6) for _, y in pp.polygons.exterior.coords]
    assert set(xs) == {0.0, 1.0}
    assert set(ys) == {0.0, 1.0}


def test_shapely_difference_seam_duplicate_is_stripped():
    """Practical motivator: diffed hole boundary with duplicated seam must decompose cleanly.

    After shapely.difference, the hole ring has a duplicated seam vertex;
    entity ingestion must produce a clean arc decomposition anyway.
    """
    inner_coords = _rounded_rect_coords(4.0, 3.0, 0.6, n_arc=8)
    inner = Polygon(inner_coords[:-1])  # Polygon re-closes automatically
    outer = Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])
    diffed = outer.difference(inner)

    ps = PolyPrism(
        polygons=diffed,
        buffers={0.0: 0.0, 1.0: 0.0},
        point_tolerance=1e-3,
        identify_arcs=True,
    )
    # Fetch the (possibly stripped/snapped) hole and run decomposition.
    # The hole is the single interior ring of the diffed polygon.
    assert len(ps.polygons.interiors) == 1
    hole_verts = [(x, y, 0.0) for x, y in ps.polygons.interiors[0].coords]
    entity = GeometryEntity(point_tolerance=1e-3)
    segs = entity.decompose_vertices(hole_verts, identify_arcs=True, min_arc_points=4)
    n_arcs = sum(1 for s in segs if s.is_arc)
    # The 4 rounded corners should all be recognized as arcs.
    assert n_arcs == 4
