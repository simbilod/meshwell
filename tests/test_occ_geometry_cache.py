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


# --- Face cache tests ---------------------------------------------------


def _square_wire(cache, z=0.0):
    """Return a 4-edge CCW square wire of side 1 built from cached edges."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeWire

    pts = [
        gp_Pnt(0.0, 0.0, z),
        gp_Pnt(1.0, 0.0, z),
        gp_Pnt(1.0, 1.0, z),
        gp_Pnt(0.0, 1.0, z),
    ]
    wb = BRepBuilderAPI_MakeWire()
    for i in range(4):
        wb.Add(cache.get_line_edge(pts[i], pts[(i + 1) % 4]))
    return wb.Wire()


def test_face_reused_for_same_wire():
    cache = OCCGeometryCache(point_tolerance=1e-3)
    w1 = _square_wire(cache)
    w2 = _square_wire(cache)
    f1 = cache.get_face(w1)
    f2 = cache.get_face(w2)
    assert _HASHER(f1) == _HASHER(f2)


def test_face_distinct_with_hole():
    """Adding a hole wire changes the face TShape."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeWire

    cache = OCCGeometryCache(point_tolerance=1e-3)
    outer = _square_wire(cache)

    inner_pts = [
        gp_Pnt(0.25, 0.25, 0.0),
        gp_Pnt(0.75, 0.25, 0.0),
        gp_Pnt(0.75, 0.75, 0.0),
        gp_Pnt(0.25, 0.75, 0.0),
    ]
    wb = BRepBuilderAPI_MakeWire()
    for i in range(4):
        wb.Add(cache.get_line_edge(inner_pts[i], inner_pts[(i + 1) % 4]))
    hole = wb.Wire()
    hole.Reverse()

    solid = cache.get_face(outer)
    with_hole = cache.get_face(outer, (hole,))
    assert _HASHER(solid) != _HASHER(with_hole)


def test_extruded_face_reused_same_edge_and_vec():
    from OCP.gp import gp_Vec

    cache = OCCGeometryCache(point_tolerance=1e-3)
    edge = cache.get_line_edge(gp_Pnt(0, 0, 0), gp_Pnt(1, 0, 0))
    vec = gp_Vec(0, 0, 1)
    f1 = cache.get_extruded_face(edge, vec)
    f2 = cache.get_extruded_face(edge, vec)
    assert _HASHER(f1) == _HASHER(f2)


def test_extruded_face_orientation_agnostic():
    """Two wire traversals of a shared edge must share the swept face TShape.

    Adjacent prisms hit this: both reference the same cached edge, but one
    wire has it oriented ``FORWARD`` while the other has it ``REVERSED``.
    The cache normalizes to a canonical orientation so both calls return
    the same ``TopoDS_Face`` TShape.
    """
    from OCP.gp import gp_Vec
    from OCP.TopAbs import TopAbs_REVERSED

    cache = OCCGeometryCache(point_tolerance=1e-3)
    edge = cache.get_line_edge(gp_Pnt(0, 0, 0), gp_Pnt(1, 0, 0))
    vec = gp_Vec(0, 0, 1)

    f_fwd = cache.get_extruded_face(edge, vec)
    f_rev = cache.get_extruded_face(edge.Oriented(TopAbs_REVERSED), vec)
    assert _HASHER(f_fwd) == _HASHER(f_rev)


def test_extruded_face_distinct_different_vec():
    from OCP.gp import gp_Vec

    cache = OCCGeometryCache(point_tolerance=1e-3)
    edge = cache.get_line_edge(gp_Pnt(0, 0, 0), gp_Pnt(1, 0, 0))
    f1 = cache.get_extruded_face(edge, gp_Vec(0, 0, 1))
    f2 = cache.get_extruded_face(edge, gp_Vec(0, 0, 2))
    assert _HASHER(f1) != _HASHER(f2)
