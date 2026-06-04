"""Tests for the cohort-global face registry."""
from __future__ import annotations

from shapely.geometry import Polygon

from meshwell.structured.build import EdgeRegistry, FaceRegistry, VertexRegistry


def _square(x0=0.0, y0=0.0, side=1.0):
    return Polygon([(x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side)])


def test_face_registry_key_equal_for_same_polygon_same_z():
    vreg = VertexRegistry(point_tolerance=1e-3)
    ereg = EdgeRegistry(vertices=vreg, point_tolerance=1e-3)
    freg = FaceRegistry(edges=ereg, point_tolerance=1e-3)
    p = _square()
    k1 = freg.key_for_polygon(p, z=0.0)
    k2 = freg.key_for_polygon(p, z=0.0)
    assert k1 == k2


def test_face_registry_key_differs_when_z_differs():
    vreg = VertexRegistry(point_tolerance=1e-3)
    ereg = EdgeRegistry(vertices=vreg, point_tolerance=1e-3)
    freg = FaceRegistry(edges=ereg, point_tolerance=1e-3)
    p = _square()
    assert freg.key_for_polygon(p, z=0.0) != freg.key_for_polygon(p, z=1.0)


def test_face_registry_key_invariant_under_ring_rotation():
    """Ring rotation invariance test.

    Ring [(0,0),(1,0),(1,1),(0,1)] and rotated [(1,0),(1,1),(0,1),(0,0)]
    must produce the same key — same polygon, different starting vertex.
    """
    vreg = VertexRegistry(point_tolerance=1e-3)
    ereg = EdgeRegistry(vertices=vreg, point_tolerance=1e-3)
    freg = FaceRegistry(edges=ereg, point_tolerance=1e-3)
    p_a = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    p_b = Polygon([(1, 0), (1, 1), (0, 1), (0, 0)])
    assert freg.key_for_polygon(p_a, z=0.0) == freg.key_for_polygon(p_b, z=0.0)


def test_face_registry_key_invariant_under_reverse_orientation():
    """Same geometric polygon traversed CCW vs CW must produce the same key."""
    vreg = VertexRegistry(point_tolerance=1e-3)
    ereg = EdgeRegistry(vertices=vreg, point_tolerance=1e-3)
    freg = FaceRegistry(edges=ereg, point_tolerance=1e-3)
    p_ccw = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    p_cw = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    assert freg.key_for_polygon(p_ccw, z=0.0) == freg.key_for_polygon(p_cw, z=0.0)


def test_face_registry_key_distinguishes_polygons_with_and_without_hole():
    vreg = VertexRegistry(point_tolerance=1e-3)
    ereg = EdgeRegistry(vertices=vreg, point_tolerance=1e-3)
    freg = FaceRegistry(edges=ereg, point_tolerance=1e-3)
    outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
    hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
    p_solid = Polygon(outer)
    p_holed = Polygon(outer, holes=[hole])
    assert freg.key_for_polygon(p_solid, z=0.0) != freg.key_for_polygon(p_holed, z=0.0)
