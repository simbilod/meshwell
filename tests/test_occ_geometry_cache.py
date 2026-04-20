"""Unit tests for the OCC geometry cache."""
from __future__ import annotations

from OCP.gp import gp_Pnt
from OCP.TopTools import TopTools_ShapeMapHasher

from meshwell.occ_geometry_cache import OCCGeometryCache

_HASHER = TopTools_ShapeMapHasher()


def test_vertex_reused_within_tolerance():
    cache = OCCGeometryCache(point_tolerance=1e-3)
    v1 = cache.get_vertex(gp_Pnt(0.0, 0.0, 0.0))
    v2 = cache.get_vertex(gp_Pnt(0.0001, 0.0, 0.0))  # within tolerance
    assert _HASHER(v1) == _HASHER(v2)


def test_vertex_distinct_outside_tolerance():
    cache = OCCGeometryCache(point_tolerance=1e-3)
    v1 = cache.get_vertex(gp_Pnt(0.0, 0.0, 0.0))
    v2 = cache.get_vertex(gp_Pnt(0.01, 0.0, 0.0))  # outside tolerance
    assert _HASHER(v1) != _HASHER(v2)


def test_line_edge_reused_same_endpoints():
    cache = OCCGeometryCache(point_tolerance=1e-3)
    p0 = gp_Pnt(0.0, 0.0, 0.0)
    p1 = gp_Pnt(1.0, 0.0, 0.0)
    e1 = cache.get_line_edge(p0, p1)
    e2 = cache.get_line_edge(p0, p1)
    assert _HASHER(e1) == _HASHER(e2)


def test_line_edge_reused_reverse_endpoints():
    cache = OCCGeometryCache(point_tolerance=1e-3)
    p0 = gp_Pnt(0.0, 0.0, 0.0)
    p1 = gp_Pnt(1.0, 0.0, 0.0)
    e_fwd = cache.get_line_edge(p0, p1)
    e_rev = cache.get_line_edge(p1, p0)
    # TShape identity regardless of direction.
    assert _HASHER(e_fwd) == _HASHER(e_rev)


def test_arc_edge_reused_same_params():
    import math

    cache = OCCGeometryCache(point_tolerance=1e-3)
    center = gp_Pnt(0.0, 0.0, 0.0)
    radius = 1.0
    p_start = gp_Pnt(1.0, 0.0, 0.0)
    p_mid = gp_Pnt(math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0)
    p_end = gp_Pnt(0.0, 1.0, 0.0)
    e1 = cache.get_arc_edge(p_start, p_mid, p_end, center, radius)
    e2 = cache.get_arc_edge(p_start, p_mid, p_end, center, radius)
    assert _HASHER(e1) == _HASHER(e2)


def test_arc_edge_distinct_for_opposite_arc():
    import math

    cache = OCCGeometryCache(point_tolerance=1e-3)
    center = gp_Pnt(0.0, 0.0, 0.0)
    radius = 1.0
    p_start = gp_Pnt(1.0, 0.0, 0.0)
    p_end = gp_Pnt(0.0, 1.0, 0.0)
    short_mid = gp_Pnt(math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0)
    long_mid = gp_Pnt(math.cos(5 * math.pi / 4), math.sin(5 * math.pi / 4), 0.0)
    e_short = cache.get_arc_edge(p_start, short_mid, p_end, center, radius)
    e_long = cache.get_arc_edge(p_start, long_mid, p_end, center, radius)
    assert _HASHER(e_short) != _HASHER(e_long)
