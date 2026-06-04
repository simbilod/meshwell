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


def test_face_xy_returns_same_tshape_for_same_polygon():
    vreg = VertexRegistry(point_tolerance=1e-3)
    ereg = EdgeRegistry(vertices=vreg, point_tolerance=1e-3)
    freg = FaceRegistry(edges=ereg, point_tolerance=1e-3)
    p1 = _square()
    p2 = _square()  # different Python object, same geometry
    f1 = freg.face_xy(
        p1,
        z=0.0,
        identify_arcs=False,
        min_arc_points=4,
        arc_tolerance=1e-3,
    )
    f2 = freg.face_xy(
        p2,
        z=0.0,
        identify_arcs=False,
        min_arc_points=4,
        arc_tolerance=1e-3,
    )
    # Shared TShape; orientation may differ but IsSame ignores orientation.
    assert f1.IsSame(f2)


def test_face_xy_returns_different_tshape_for_different_z():
    vreg = VertexRegistry(point_tolerance=1e-3)
    ereg = EdgeRegistry(vertices=vreg, point_tolerance=1e-3)
    freg = FaceRegistry(edges=ereg, point_tolerance=1e-3)
    p = _square()
    f_lo = freg.face_xy(
        p,
        z=0.0,
        identify_arcs=False,
        min_arc_points=4,
        arc_tolerance=1e-3,
    )
    f_hi = freg.face_xy(
        p,
        z=1.0,
        identify_arcs=False,
        min_arc_points=4,
        arc_tolerance=1e-3,
    )
    assert not f_lo.IsSame(f_hi)


def test_face_xy_handles_polygon_with_hole():
    outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
    hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
    p = Polygon(outer, holes=[hole])
    vreg = VertexRegistry(point_tolerance=1e-3)
    ereg = EdgeRegistry(vertices=vreg, point_tolerance=1e-3)
    freg = FaceRegistry(edges=ereg, point_tolerance=1e-3)
    f = freg.face_xy(
        p,
        z=0.0,
        identify_arcs=False,
        min_arc_points=4,
        arc_tolerance=1e-3,
    )
    # Face exists and contains exactly two wires (outer + inner).
    from OCP.TopAbs import TopAbs_WIRE
    from OCP.TopExp import TopExp_Explorer

    wires = []
    exp = TopExp_Explorer(f, TopAbs_WIRE)
    while exp.More():
        wires.append(exp.Current())
        exp.Next()
    assert len(wires) == 2


def test_build_horizontal_face_uses_registry_when_provided():
    """Test caching behavior when FaceRegistry is provided.

    When `_build_horizontal_face` is called twice with the same polygon
    and the SAME FaceRegistry, the second call returns the cached face
    rather than constructing a new one.
    """
    from meshwell.structured.build import _build_horizontal_face

    vreg = VertexRegistry(point_tolerance=1e-3)
    ereg = EdgeRegistry(vertices=vreg, point_tolerance=1e-3)
    freg = FaceRegistry(edges=ereg, point_tolerance=1e-3)
    p = _square()
    f1 = _build_horizontal_face(
        p,
        z=0.0,
        ereg=ereg,
        identify_arcs=False,
        min_arc_points=4,
        arc_tolerance=1e-3,
        face_registry=freg,
    )
    f2 = _build_horizontal_face(
        p,
        z=0.0,
        ereg=ereg,
        identify_arcs=False,
        min_arc_points=4,
        arc_tolerance=1e-3,
        face_registry=freg,
    )
    assert f1.IsSame(f2)


def test_build_horizontal_face_without_registry_works_unchanged():
    """Test backward compatibility when face_registry is None.

    When `face_registry=None`, behaviour is identical to the
    pre-refactor signature (no caching, fresh TShape each call).
    """
    from meshwell.structured.build import _build_horizontal_face

    vreg = VertexRegistry(point_tolerance=1e-3)
    ereg = EdgeRegistry(vertices=vreg, point_tolerance=1e-3)
    p = _square()
    f1 = _build_horizontal_face(
        p,
        z=0.0,
        ereg=ereg,
        identify_arcs=False,
        min_arc_points=4,
        arc_tolerance=1e-3,
        face_registry=None,
    )
    f2 = _build_horizontal_face(
        p,
        z=0.0,
        ereg=ereg,
        identify_arcs=False,
        min_arc_points=4,
        arc_tolerance=1e-3,
        face_registry=None,
    )
    # Without registry, two calls produce DIFFERENT TShapes.
    assert not f1.IsSame(f2)


def test_structured_pre_pass_stores_face_registry_per_cohort():
    """Per-cohort 3-tuple registry entry.

    StructuredState.cohort_registries should contain a 3-tuple
    (vertex_registry, edge_registry, face_registry) per cohort.
    """
    from meshwell.polyprism import PolyPrism
    from meshwell.structured.build import FaceRegistry
    from meshwell.structured.pipeline import structured_pre_pass

    A = PolyPrism(
        _square(side=10.0),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="A",
        structured=True,
        mesh_order=3.0,
    )
    state = structured_pre_pass([A], point_tolerance=1e-3)
    assert len(state.cohort_registries) == 1
    entry = state.cohort_registries[0]
    assert len(entry) == 3
    _vreg, ereg, freg = entry
    assert isinstance(freg, FaceRegistry)
    assert freg.edges is ereg
