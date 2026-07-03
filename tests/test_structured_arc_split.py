"""Both closed-arc split paths must produce identical (start, mid, end) triples."""

import math

import pytest

from meshwell.geometry_entity import decompose_vertices_2d
from meshwell.structured.build import (
    _flatten_decomposition_to_polyline_segments,
    closed_arc_split_indices,
    ring_is_closed,
)

TOL = 1e-3


def _circle_coords(n):
    # closed ring: first point repeated at the end
    pts = [
        (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)) for i in range(n)
    ]
    return [*pts, pts[0]]


@pytest.mark.parametrize("n", [8, 11, 16, 33])
def test_split_indices_match_emit_path_convention(n):
    # the emit path uses n//4, n//2, (3*n)//4 — the helper must pin that
    n_pts = n + 1
    q1, mid, q3 = closed_arc_split_indices(n_pts)
    assert (q1, mid, q3) == (n_pts // 4, n_pts // 2, (n_pts * 3) // 4)


@pytest.mark.parametrize("n", [8, 11, 16, 33])
def test_lateral_flatten_uses_same_arc_midpoints(n):
    coords = _circle_coords(n)
    raw = decompose_vertices_2d(
        coords,
        z=0.0,
        point_tolerance=TOL,
        identify_arcs=True,
        min_arc_points=5,
        arc_tolerance=TOL,
    )
    closed_arcs = [s for s in raw if s.is_arc]
    assert closed_arcs, "decomposition should identify the circle as an arc"

    flat = _flatten_decomposition_to_polyline_segments(raw, TOL)
    arc_segs = [s for s in flat if s.kind == "arc"]
    assert len(arc_segs) == 2, "closed circle must split into two half-arcs"

    pts = closed_arcs[0].points
    q1, _mid, q3 = closed_arc_split_indices(len(pts))

    def key(xy):
        return (round(xy[0] / TOL), round(xy[1] / TOL))

    # first half-arc: start -> q1 -> mid; second: mid -> q3 -> end
    assert key(arc_segs[0].mid) == key((pts[q1][0], pts[q1][1]))
    assert key(arc_segs[1].mid) == key((pts[q3][0], pts[q3][1]))


def test_ring_is_closed_matches_vertex_registry_quantization():
    assert ring_is_closed((0.0, 0.0), (0.0004, -0.0004), 1e-3)
    assert not ring_is_closed((0.0, 0.0), (0.002, 0.0), 1e-3)
