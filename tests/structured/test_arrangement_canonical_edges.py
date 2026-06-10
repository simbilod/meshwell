"""Tests for _build_canonical_edges in meshwell.structured.decompose."""
from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from meshwell.structured.decompose import _build_canonical_edges


def _rect(x1, y1, x2, y2):
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def _circle(cx, cy, r, n=48):
    a = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    return Polygon([(cx + r * np.cos(t), cy + r * np.sin(t)) for t in a])


def test_two_overlapping_rectangles_split_into_seven_edges():
    """Two overlapping squares share a single cut — expect 7 canonical edges.

    The planar graph has 7 unique edges (4 around left square minus
    shared cut + 4 around right square minus shared cut + 1 shared cut
    = some bookkeeping). Shapely's unary_union nodes at the two crossing
    points where the shared cut meets each square's left/right side.
    """
    merged = unary_union([_rect(0, 0, 6, 10).boundary, _rect(4, 0, 10, 10).boundary])
    edges, lookup = _build_canonical_edges(
        merged,
        z=0.0,
        point_tolerance=1e-3,
        identify_arcs=False,
        min_arc_points=5,
        arc_tolerance=1e-3,
    )
    assert len(edges) == 7
    # Every consecutive vertex pair on every edge is registered exactly once.
    seen = set()
    for ei, edge in enumerate(edges):
        for i in range(len(edge.vertex_keys) - 1):
            pair = frozenset({edge.vertex_keys[i], edge.vertex_keys[i + 1]})
            assert pair not in seen, f"duplicate pair on edge {ei}"
            seen.add(pair)
            assert lookup[pair] == ei
    # Closed standalone edges (none here) are NOT pair-indexed.
    assert all(not e.is_closed for e in edges)


def test_single_closed_disc_yields_one_closed_edge():
    """A standalone circle becomes one closed canonical edge.

    No other arrangement nodes, so is_closed=True; its vertex pairs are
    NOT registered in the lookup (closed-standalone fallback).
    """
    merged = unary_union([_circle(0, 0, 1.0).boundary])
    edges, lookup = _build_canonical_edges(
        merged,
        z=0.0,
        point_tolerance=1e-3,
        identify_arcs=True,
        min_arc_points=5,
        arc_tolerance=1e-3,
    )
    assert len(edges) == 1
    assert edges[0].is_closed
    # OPEN storage convention: first key != last key.
    assert edges[0].vertex_keys[0] != edges[0].vertex_keys[-1]
    # Closed-standalone edges are NOT pair-indexed.
    assert lookup == {}
    # identify_arcs=True with a 48-point circle produces at least one arc.
    assert any(s.is_arc for s in edges[0].segments)


def test_two_overlapping_discs_produce_shared_arc_edges():
    """Two overlapping discs cut each other at two points.

    Shapely's ``unary_union`` of polygon-circle boundaries treats each
    polygon's seam vertex (the angle-0 vertex at ``(cx+r, cy)``) as a
    node where the closed LinearRing splits — so two 48-vertex
    polygon-circles produce 6 arrangement edges, not the 4 you'd
    expect from analytic circles. (The four short edges between a
    seam vertex and a true crossing point may be too short to fit an
    arc; we only require AT LEAST ONE edge to carry arc segments.)
    """
    merged = unary_union([_circle(0, 0, 1.0).boundary, _circle(1.0, 0, 1.0).boundary])
    edges, lookup = _build_canonical_edges(
        merged,
        z=0.0,
        point_tolerance=1e-3,
        identify_arcs=True,
        min_arc_points=5,
        arc_tolerance=1e-3,
    )
    assert len(edges) == 6
    # No closed standalone edges in this configuration.
    assert all(not e.is_closed for e in edges)
    # AT LEAST ONE edge has an arc segment (the long arcs through the
    # disc boundaries are arc-fit; the short edges between a seam and
    # a crossing may be too short for arc detection and stay as line
    # runs).
    assert any(any(s.is_arc for s in e.segments) for e in edges)
    # Every consecutive pair across all edges is registered uniquely.
    n_pairs = sum(len(e.vertex_keys) - 1 for e in edges)
    assert len(lookup) == n_pairs


def test_parallel_edges_between_same_nodes_raise():
    """Parallel edges between the same arrangement nodes raise CanonicalArrangementError.

    Constructed artificially because unary_union normally prevents it —
    but the validator must catch it if the input ever produces it.
    """
    # Two two-vertex LineStrings between the same endpoints, NOT routed
    # through unary_union so they stay as parallel components.
    from shapely.geometry import MultiLineString

    from meshwell.structured.exceptions import CanonicalArrangementError

    merged = MultiLineString(
        [
            LineString([(0, 0), (1, 0)]),
            LineString([(0, 0), (1, 0)]),
        ]
    )
    with pytest.raises(CanonicalArrangementError):
        _build_canonical_edges(
            merged,
            z=0.0,
            point_tolerance=1e-3,
            identify_arcs=False,
            min_arc_points=5,
            arc_tolerance=1e-3,
        )
