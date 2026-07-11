"""Tests for pipeline-level arc identification + the canonical geometry pipeline."""
import itertools

import numpy as np
import pytest
from shapely.geometry import Polygon

from meshwell.circle_registry import ArcFit, CircleRegistry, build_circle_registry
from meshwell.geometry_entity import (
    _arc_sense_ccw,
    _circle_circle_junction,
    _line_circle_junction,
    _line_line_junction,
    _offset_point,
    _project_to_circle,
)


def _ring(n: int, radius: float, phase: float = 0.0) -> Polygon:
    pts = [
        (
            radius * np.cos(2 * np.pi * k / n + phase),
            radius * np.sin(2 * np.pi * k / n + phase),
        )
        for k in range(n)
    ]
    return Polygon(pts)


def test_apply_arc_params_stamps_entities():
    from meshwell.cad_common import apply_arc_params
    from meshwell.polyprism import PolyPrism

    p = PolyPrism(
        polygons=_ring(16, 5.0),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="p",
        mesh_order=1,
    )
    assert p.identify_arcs is False  # GeometryEntity class default
    apply_arc_params([p], identify_arcs=True, min_arc_points=7, arc_tolerance=2e-3)
    assert p.identify_arcs is True
    assert p.min_arc_points == 7
    assert p.arc_tolerance == 2e-3


def test_apply_arc_params_stamps_polyline():
    from shapely.geometry import LineString

    from meshwell.cad_common import apply_arc_params
    from meshwell.polyline import PolyLine

    pl = PolyLine(LineString([(0, 0), (1, 0), (1, 1)]), physical_name="pl")
    assert pl.identify_arcs is False  # GeometryEntity class default
    apply_arc_params([pl], identify_arcs=True, min_arc_points=6, arc_tolerance=2e-3)
    assert pl.identify_arcs is True
    assert pl.min_arc_points == 6
    assert pl.arc_tolerance == 2e-3


def test_apply_arc_params_skips_non_extrude_prism():
    from meshwell.cad_common import apply_arc_params
    from meshwell.polyprism import PolyPrism

    p = PolyPrism(
        polygons=_ring(16, 5.0),
        buffers={0.0: 0.1, 1.0: 0.0},
        physical_name="p",
        mesh_order=1,
    )
    apply_arc_params([p], identify_arcs=True)
    assert p.identify_arcs is False  # arcs unsupported with z-varying buffers


def test_polyprism_no_longer_accepts_arc_kwargs():
    from meshwell.polyprism import PolyPrism

    with pytest.raises(TypeError):
        PolyPrism(
            polygons=_ring(16, 5.0),
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="p",
            mesh_order=1,
            identify_arcs=True,
        )


def test_from_dict_ignores_legacy_arc_keys():
    from meshwell.polyprism import PolyPrism

    p = PolyPrism(
        polygons=_ring(16, 5.0),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="p",
        mesh_order=1,
    )
    d = p.to_dict()
    assert "identify_arcs" not in d
    d["identify_arcs"] = True  # legacy serialized scenes carry these keys
    d["min_arc_points"] = 5
    d["arc_tolerance"] = 1e-3
    p2 = PolyPrism.from_dict(d)
    assert p2.identify_arcs is False


def test_cluster_unifies_center_and_radius():
    fits = [
        ArcFit(center=(0.0001, -0.0001), radius=5.0002, npoints=64),
        ArcFit(center=(-0.0001, 0.0002), radius=4.9998, npoints=48),
    ]
    reg = CircleRegistry.from_fits(fits, cluster_tolerance=1e-3)
    hit_a = reg.lookup((0.0001, -0.0001), 5.0002)
    hit_b = reg.lookup((-0.0001, 0.0002), 4.9998)
    assert hit_a == hit_b  # ONE canonical circle
    wx = (64 * 0.0001 + 48 * -0.0001) / 112
    wy = (64 * -0.0001 + 48 * 0.0002) / 112
    wr = (64 * 5.0002 + 48 * 4.9998) / 112
    assert hit_a[0] == pytest.approx((wx, wy), abs=1e-12)
    assert hit_a[1] == pytest.approx(wr, abs=1e-12)


def test_distinct_circles_do_not_cluster():
    fits = [
        ArcFit(center=(0.0, 0.0), radius=5.0, npoints=32),
        ArcFit(center=(0.0, 0.0), radius=5.2, npoints=32),  # concentric, distinct
        ArcFit(center=(20.0, 0.0), radius=5.0, npoints=32),
    ]
    reg = CircleRegistry.from_fits(fits, cluster_tolerance=1e-3)
    assert reg.lookup((0.0, 0.0), 5.0)[1] == pytest.approx(5.0)
    assert reg.lookup((0.0, 0.0), 5.2)[1] == pytest.approx(5.2)
    assert reg.lookup((20.0, 0.0), 5.0)[0] == pytest.approx((20.0, 0.0))


def test_lookup_miss_returns_none():
    reg = CircleRegistry.from_fits(
        [ArcFit(center=(0.0, 0.0), radius=5.0, npoints=32)], cluster_tolerance=1e-3
    )
    assert reg.lookup((3.0, 3.0), 5.0) is None
    assert reg.lookup((0.0, 0.0), 7.0) is None


def test_match_chord_run():
    reg = CircleRegistry.from_fits(
        [ArcFit(center=(0.0, 0.0), radius=5.0, npoints=32)], cluster_tolerance=1e-3
    )
    on_circle = [(5 * np.cos(t), 5 * np.sin(t)) for t in (0.1, 0.2, 0.3)]
    assert reg.match_chord_run(on_circle, tolerance=1e-3) == ((0.0, 0.0), 5.0)
    # Straight edge whose ENDPOINTS graze the circle: interior vertex is
    # on the chord, 2.0 off the circle -> no match.
    straight = [(3.0, 4.0), (3.0, 0.0), (3.0, -4.0)]
    assert reg.match_chord_run(straight, tolerance=1e-3) is None


def test_build_registry_from_entities():
    from meshwell.cad_common import apply_arc_params
    from meshwell.polyprism import PolyPrism

    disc = PolyPrism(
        polygons=_ring(64, 5.0),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="disc",
        mesh_order=1,
    )
    plate = PolyPrism(
        polygons=Polygon(
            [(-9, -9), (9, -9), (9, 9), (-9, 9)],
            holes=[_ring(48, 5.0, phase=0.03).exterior.coords],
        ),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="plate",
        mesh_order=2,
    )
    apply_arc_params([disc, plate], identify_arcs=True)
    reg = build_circle_registry([disc, plate])
    # Both rings discretize the SAME nominal circle -> one cluster.
    # Loose tolerances: PolyPrism snaps inputs to the 1e-3 grid.
    hit = reg.lookup((0.0, 0.0), 5.0)
    assert hit is not None
    assert hit[0] == pytest.approx((0.0, 0.0), abs=1e-4)
    assert hit[1] == pytest.approx(5.0, abs=1e-4)


def test_arc_sense_ccw():
    c = (0.0, 0.0)
    assert _arc_sense_ccw(c, (1, 0), (0, 1), (-1, 0)) is True  # CCW upper half
    assert _arc_sense_ccw(c, (1, 0), (0, -1), (-1, 0)) is False  # CW lower half
    assert _arc_sense_ccw(c, (-1, 0), (0, -1), (1, 0)) is True  # crosses atan2 seam


def test_project_to_circle():
    assert _project_to_circle((0.0, 0.0), 2.0, (3.0, 4.0)) == pytest.approx((1.2, 1.6))
    assert _project_to_circle((1.0, 1.0), 2.0, (1.0, 1.0)) == (1.0, 1.0)  # degenerate


def test_line_circle_junction_transversal_and_tangent():
    j = _line_circle_junction((0.0, 0.0), 1.0, (1.05, 0.0), (5.0, 0.0))
    assert j == pytest.approx((1.0, 0.0), abs=1e-12)
    # y=1 tangent to unit circle: must return the tangency foot.
    j = _line_circle_junction((0.0, 0.0), 1.0, (0.02, 1.0000001), (5.0, 1.0000001))
    assert j == pytest.approx((0.0, 1.0), abs=1e-6)


def test_circle_circle_junction():
    j = _circle_circle_junction((0.0, 0.0), 1.0, (1.0, 0.0), 1.0, (0.5, 0.9))
    assert j == pytest.approx((0.5, np.sqrt(3) / 2), abs=1e-12)


def test_line_line_junction_miter():
    # Offset edges of a 90-degree corner: x=1 line meets y=1 line at (1,1).
    j = _line_line_junction(
        (1.0, -5.0), (1.0, 0.0), (0.0, 1.0), (-5.0, 1.0), (0.0, 0.0)
    )
    assert j == pytest.approx((1.0, 1.0), abs=1e-12)
    # Near-parallel: falls back to the provided point.
    j = _line_line_junction(
        (0.0, 0.0), (1.0, 0.0), (2.0, 1e-15), (3.0, 2e-15), (9.0, 9.0)
    )
    assert j == (9.0, 9.0)


def test_offset_point_right_of_travel():
    # Traveling +x, right of travel is -y.
    assert _offset_point((0.0, 0.0), (1.0, 0.0), 0.1) == pytest.approx((1.0, -0.1))
    # eps=0 is the identity.
    assert _offset_point((0.0, 0.0), (1.0, 0.0), 0.0) == pytest.approx((1.0, 0.0))


def test_promote_short_arc_remnant():
    """A 4-vertex arc span below min_arc_points=5 stays chords in the detector.

    It must still be promoted onto the registry circle.
    """
    from meshwell.geometry_entity import _promote_chord_runs, decompose_vertices_2d

    arc_pts = [
        (5 * np.cos(np.deg2rad(a)), 5 * np.sin(np.deg2rad(a))) for a in (10, 20, 30, 40)
    ]
    ring = [(-6.0, -6.0), (6.0, -6.0), *arc_pts, (-6.0, 6.0), (-6.0, -6.0)]
    segments = decompose_vertices_2d(
        ring,
        z=0.0,
        point_tolerance=1e-3,
        identify_arcs=True,
        min_arc_points=5,
        arc_tolerance=1e-3,
    )
    assert not any(s.is_arc for s in segments)
    reg = CircleRegistry.from_fits(
        [ArcFit(center=(0.0, 0.0), radius=5.0, npoints=32)], cluster_tolerance=1e-3
    )
    promoted = _promote_chord_runs(segments, reg, tolerance=2e-3)
    arcs = [s for s in promoted if s.is_arc]
    assert len(arcs) == 1
    assert len(arcs[0].points) == 4
    for a, b in itertools.pairwise(promoted):
        assert a.points[-1] == b.points[0]  # connectivity preserved


def test_no_promotion_of_genuine_straight_edge():
    from meshwell.geometry_entity import DecompositionSegment, _promote_chord_runs

    segs = [
        DecompositionSegment(points=[(3.0, 4.0, 0.0), (3.0, 0.0, 0.0)], is_arc=False),
        DecompositionSegment(points=[(3.0, 0.0, 0.0), (3.0, -4.0, 0.0)], is_arc=False),
    ]
    reg = CircleRegistry.from_fits(
        [ArcFit(center=(0.0, 0.0), radius=5.0, npoints=32)], cluster_tolerance=1e-3
    )
    assert not any(s.is_arc for s in _promote_chord_runs(segs, reg, tolerance=2e-3))


def test_canonicalize_offsets_full_circle():
    """CCW disc ring at eps=0.01: canonical circle offsets to R+eps."""
    from meshwell.geometry_entity import (
        canonicalize_ring_segments,
        decompose_vertices_2d,
    )

    segments = decompose_vertices_2d(
        list(_ring(64, 5.0).exterior.coords),
        z=0.0,
        point_tolerance=1e-3,
        identify_arcs=True,
        min_arc_points=5,
        arc_tolerance=1e-3,
    )
    reg = CircleRegistry.from_fits(
        [ArcFit(center=(0.0, 0.0), radius=5.0, npoints=64)], cluster_tolerance=1e-3
    )
    out = canonicalize_ring_segments(
        segments, reg, eps=0.01, match_tolerance=2e-3, slack=1e-3
    )
    arcs = [s for s in out if s.is_arc]
    assert arcs
    assert all(s.canonical == ((0.0, 0.0), pytest.approx(5.01)) for s in arcs)


def test_canonicalize_offsets_hole_ring_inward():
    """CW hole ring at eps=0.01: the hole SHRINKS -> R-eps."""
    from meshwell.geometry_entity import (
        canonicalize_ring_segments,
        decompose_vertices_2d,
    )

    cw_coords = list(_ring(64, 5.0).exterior.coords)[::-1]  # CW = OGC hole
    segments = decompose_vertices_2d(
        cw_coords,
        z=0.0,
        point_tolerance=1e-3,
        identify_arcs=True,
        min_arc_points=5,
        arc_tolerance=1e-3,
    )
    reg = CircleRegistry.from_fits(
        [ArcFit(center=(0.0, 0.0), radius=5.0, npoints=64)], cluster_tolerance=1e-3
    )
    out = canonicalize_ring_segments(
        segments, reg, eps=0.01, match_tolerance=2e-3, slack=1e-3
    )
    arcs = [s for s in out if s.is_arc]
    assert arcs
    assert all(s.canonical == ((0.0, 0.0), pytest.approx(4.99)) for s in arcs)


def test_canonicalize_line_ring_miter_offset():
    """Pure-line CCW unit square at eps=0.1: corners land at exact miter points.

    Also checks that eps=0 is the identity (+-1.1 corners either way).
    """
    from meshwell.geometry_entity import (
        canonicalize_ring_segments,
        decompose_vertices_2d,
    )

    square = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0)]
    segments = decompose_vertices_2d(square, z=0.0, point_tolerance=1e-3)
    out = canonicalize_ring_segments(
        segments, None, eps=0.1, match_tolerance=2e-3, slack=1e-3
    )
    xs = [abs(p[0]) for s in out for p in s.points]
    ys = [abs(p[1]) for s in out for p in s.points]
    assert max(xs) == pytest.approx(1.1)
    assert max(ys) == pytest.approx(1.1)
    assert min(xs) == pytest.approx(1.1)
    assert min(ys) == pytest.approx(1.1)

    segments = decompose_vertices_2d(square, z=0.0, point_tolerance=1e-3)
    out0 = canonicalize_ring_segments(
        segments, None, eps=0.0, match_tolerance=2e-3, slack=1e-3
    )
    assert [s.points for s in out0] == [
        [(p[0], p[1], 0.0) for p in pair] for pair in itertools.pairwise(square)
    ]


def _circle_edges(shape):
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomAbs import GeomAbs_CurveType
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    out = []
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        ad = BRepAdaptor_Curve(TopoDS.Edge_s(exp.Current()))
        if ad.GetType() == GeomAbs_CurveType.GeomAbs_Circle:
            circ = ad.Circle()
            loc = circ.Location()
            out.append(((loc.X(), loc.Y()), circ.Radius()))
        exp.Next()
    return out


def test_wire_emits_offset_canonical_circle():
    from meshwell.cad_common import apply_arc_params
    from meshwell.circle_registry import build_circle_registry
    from meshwell.polyprism import PolyPrism

    disc = PolyPrism(
        polygons=_ring(64, 5.0),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="d",
        mesh_order=1,
    )
    apply_arc_params([disc], identify_arcs=True)
    disc.circle_registry = build_circle_registry([disc])
    disc.perturbation = 1e-2  # exaggerated for a visible assertion
    shape = disc.instanciate_occ()
    circles = _circle_edges(shape)
    assert circles
    canonical = disc.circle_registry.clusters[0]
    for (cx, cy), r in circles:
        assert (cx, cy) == pytest.approx(canonical.center, abs=1e-9)
        assert r == pytest.approx(canonical.radius + 1e-2, abs=1e-9)  # CCW disc: R+eps


def test_wire_passthrough_without_registry_or_eps():
    from meshwell.cad_common import apply_arc_params
    from meshwell.polyprism import PolyPrism

    disc = PolyPrism(
        polygons=_ring(64, 5.0),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="d",
        mesh_order=1,
    )
    apply_arc_params([disc], identify_arcs=True)
    assert disc.circle_registry is None
    assert disc.perturbation == 0.0
    for _c, r in _circle_edges(disc.instanciate_occ()):
        assert r == pytest.approx(5.0, abs=1e-6)  # legacy 3-point arcs


def test_rounded_rect_prism_closes_with_registry():
    from meshwell.cad_common import apply_arc_params
    from meshwell.circle_registry import build_circle_registry
    from meshwell.polyprism import PolyPrism

    def rounded_rect(a, b, r, n=12):
        pts = []
        for i, (cx, cy) in enumerate([(a, b), (-a, b), (-a, -b), (a, -b)]):
            pts.extend(
                (cx + r * np.cos(t), cy + r * np.sin(t))
                for t in np.linspace(np.pi / 2 * i, np.pi / 2 * (i + 1), n)
            )
        return Polygon(pts)

    pad = PolyPrism(
        polygons=rounded_rect(3.0, 2.0, 0.8),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="pad",
        mesh_order=1,
    )
    apply_arc_params([pad], identify_arcs=True)
    pad.circle_registry = build_circle_registry([pad])
    pad.perturbation = 1e-5
    shape = pad.instanciate_occ()
    assert shape is not None
    assert len(_circle_edges(shape)) >= 4
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    assert props.Mass() > 0  # wire closed, solid valid
