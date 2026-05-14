"""Unit tests for the arc-index / piece-classification helpers."""
from __future__ import annotations

import math

from shapely.geometry import Polygon
from shapely.ops import polygonize, unary_union

from meshwell.structured_polyprism import (
    PieceArcEdge,
    PieceLineEdge,
    _build_arc_index_from_footprint,
    _classify_piece_boundary,
)


def _disc(n: int = 48, r: float = 1.0):
    return Polygon(
        [
            (r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n))
            for k in range(n)
        ]
    )


def test_arc_index_disc_identifies_one_arc():
    fp = _disc(n=48)
    index = _build_arc_index_from_footprint(
        fp, identify_arcs=True, min_arc_points=4, arc_tolerance=0.01
    )
    # Disc has one closed arc (one circle).
    assert len(index.arcs) == 1, index.arcs
    arc = index.arcs[0]
    assert abs(arc.radius - 1.0) < 1e-3
    assert abs(arc.center[0]) < 1e-3
    assert abs(arc.center[1]) < 1e-3
    # Every vertex of the disc exterior is registered.
    for x, y in list(fp.exterior.coords)[:-1]:
        assert (round(x, 3), round(y, 3)) in index.vertex_to_arcs


def test_arc_index_disabled_when_identify_arcs_false():
    fp = _disc(n=48)
    index = _build_arc_index_from_footprint(
        fp, identify_arcs=False, min_arc_points=4, arc_tolerance=0.01
    )
    assert index.arcs == []
    assert index.vertex_to_arcs == {}


def test_arc_index_annulus_identifies_two_arcs():
    outer = [
        (1.0 * math.cos(2 * math.pi * k / 48), 1.0 * math.sin(2 * math.pi * k / 48))
        for k in range(48)
    ]
    inner = [
        (0.4 * math.cos(2 * math.pi * k / 48), 0.4 * math.sin(2 * math.pi * k / 48))
        for k in range(48)
    ]
    fp = Polygon(outer, holes=[inner])
    index = _build_arc_index_from_footprint(
        fp, identify_arcs=True, min_arc_points=4, arc_tolerance=0.01
    )
    radii = sorted(a.radius for a in index.arcs)
    assert len(radii) == 2, radii
    assert abs(radii[0] - 0.4) < 1e-2
    assert abs(radii[1] - 1.0) < 1e-2


def test_classify_disc_cut_by_rectangle():
    """Disc footprint cut by a small interior rectangle.

    The piece surrounding the rectangle has exterior = full-disc arc; the
    rectangle's footprint boundary at the cut becomes line edges on the
    interior hole.
    """
    fp = _disc(n=48, r=1.0)
    rect = Polygon([(-0.3, -0.3), (0.3, -0.3), (0.3, 0.3), (-0.3, 0.3)])
    lines = unary_union([fp.boundary, rect.boundary])
    pieces = [p for p in polygonize(lines) if fp.contains(p.representative_point())]
    assert len(pieces) >= 1
    index = _build_arc_index_from_footprint(
        fp, identify_arcs=True, min_arc_points=4, arc_tolerance=0.01
    )
    for piece in pieces:
        prov = _classify_piece_boundary(piece, index)
        # Each piece's edges must concatenate back to its boundary order.
        rebuilt = []
        for e in prov.exterior_edges:
            for p in e.points:
                if not rebuilt or rebuilt[-1] != p:
                    rebuilt.append(p)
        if rebuilt and rebuilt[0] != rebuilt[-1]:
            rebuilt.append(rebuilt[0])
        # The walked count of unique vertices must equal the polygon's
        # exterior vertex count (minus closing duplicate).
        assert len(rebuilt) - 1 == len(list(piece.exterior.coords)) - 1


def test_classify_pure_disc_one_arc_edge():
    """No splitter: the disc piece's exterior is a single arc edge."""
    fp = _disc(n=48, r=1.0)
    index = _build_arc_index_from_footprint(
        fp, identify_arcs=True, min_arc_points=4, arc_tolerance=0.01
    )
    prov = _classify_piece_boundary(fp, index)
    assert len(prov.exterior_edges) == 1
    assert isinstance(prov.exterior_edges[0], PieceArcEdge)


def test_classify_pure_rectangle_all_lines():
    """No arcs in the input: every piece edge is a line.

    Holds regardless of whether ``identify_arcs`` was passed True.
    """
    rect = Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])
    index = _build_arc_index_from_footprint(
        rect, identify_arcs=True, min_arc_points=4, arc_tolerance=0.01
    )
    prov = _classify_piece_boundary(rect, index)
    assert all(isinstance(e, PieceLineEdge) for e in prov.exterior_edges)
