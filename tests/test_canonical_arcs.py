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


def test_from_dict_warns_on_legacy_arc_keys(caplog):
    """Migration signal: legacy per-entity arc keys log a warning, not silence."""
    import logging

    from shapely.geometry import LineString

    from meshwell.polyline import PolyLine
    from meshwell.polyprism import PolyPrism
    from meshwell.polysurface import PolySurface

    prism = PolyPrism(
        polygons=_ring(16, 5.0),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="p",
        mesh_order=1,
    )
    surf = PolySurface(polygons=_ring(16, 5.0), physical_name="s", mesh_order=1)
    line = PolyLine(LineString([(0, 0), (1, 0), (1, 1)]), physical_name="pl")

    for cls, entity in (
        (PolyPrism, prism),
        (PolySurface, surf),
        (PolyLine, line),
    ):
        d = entity.to_dict()
        d["identify_arcs"] = True
        d["min_arc_points"] = 5
        d["arc_tolerance"] = 1e-3
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="meshwell.geometry_entity"):
            reloaded = cls.from_dict(d)
        assert reloaded is not None  # loading still succeeds
        assert any(
            "legacy" in rec.message and cls.__name__ in rec.message
            for rec in caplog.records
        )


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


def test_two_arc_ring_emits_both_canonical_circles():
    """Arc<->arc junction on the INTERSECTING (exact) path emits BOTH circles.

    The open-arc emitter in ``_make_occ_wire_from_vertices`` reconstructs an
    arc's canonical circle via the 3-point form ``GC_MakeArcOfCircle(start,
    mid_on_circle, end)``, which is exact only when ``start``/``end`` (the
    junction points set by ``canonicalize_ring_segments``) lie ON the
    canonical circle. For an arc<->arc junction that holds when the two
    offset circles genuinely INTERSECT (``_circle_circle_junction``'s exact
    root, not its non-intersecting midpoint-of-two-projections fallback).

    Lens (vesica) shape = disk(center=(-3, 0), r=5) ∩ disk(center=(3, 0),
    r=5): d=6 between centers, r1+r2=10, so the circles intersect at
    (0, +-4) and BOTH of the ring's two arc<->arc junctions land on the
    exact-root path. Each arc's endpoints therefore lie exactly on its own
    canonical circle, so this exercises exactly the gap Finding 2 closes:
    with two arcs (rather than one arc between line segments), the wire's
    only two junctions are both arc<->arc, and both must reconstruct their
    canonical circle exactly for this test to pass.
    """
    from meshwell.cad_common import apply_arc_params
    from meshwell.circle_registry import build_circle_registry
    from meshwell.polyprism import PolyPrism

    c1, c2, radius = (-3.0, 0.0), (3.0, 0.0), 5.0
    half_angle = np.arcsin(4.0 / 5.0)  # circle-circle intersections at (0, +-4)
    n = 24
    arc1 = [
        (c1[0] + radius * np.cos(t), c1[1] + radius * np.sin(t))
        for t in np.linspace(-half_angle, half_angle, n)
    ]
    arc2 = [
        (c2[0] + radius * np.cos(t), c2[1] + radius * np.sin(t))
        for t in np.linspace(np.pi - half_angle, np.pi + half_angle, n)
    ]
    lens = Polygon(arc1 + arc2)
    assert lens.is_valid

    pad = PolyPrism(
        polygons=lens,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="lens",
        mesh_order=1,
    )
    apply_arc_params([pad], identify_arcs=True)
    pad.circle_registry = build_circle_registry([pad])
    assert len(pad.circle_registry.clusters) == 2  # two distinct arcs
    pad.perturbation = 1e-5
    shape = pad.instanciate_occ()

    circles = _circle_edges(shape)
    distinct = {(round(cx, 9), round(cy, 9), round(r, 9)) for (cx, cy), r in circles}
    assert len(distinct) == 2  # both canonical circles survived, not one

    clusters = pad.circle_registry.clusters
    for cx, cy, r in distinct:
        cluster = min(
            clusters, key=lambda cl: np.hypot(cx - cl.center[0], cy - cl.center[1])
        )
        assert (cx, cy) == pytest.approx(cluster.center, abs=1e-9)
        # The lens lies INSIDE both full disks, so each arc behaves like a
        # plain disc's outer boundary (material on the center's side of the
        # curve, per the CCW-disc case in test_wire_emits_offset_canonical_
        # circle) -- both offset R+eps, never R-eps. Verified numerically;
        # asserting the concrete relationship (not a hardcoded sign guess).
        assert r - cluster.radius == pytest.approx(1e-5, abs=1e-9)


def _disc_and_plate(n_disc=64, n_hole=48, phase=0.03):
    from meshwell.polyprism import PolyPrism

    disc = PolyPrism(
        polygons=_ring(n_disc, 5.0),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="disc",
        mesh_order=1,
    )
    plate = PolyPrism(
        polygons=Polygon(
            [(-9, -9), (9, -9), (9, 9), (-9, 9)],
            holes=[_ring(n_hole, 5.0, phase=phase).exterior.coords],
        ),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="plate",
        mesh_order=2,
    )
    return [disc, plate]


def _assert_two_clean_solids(out):
    solids = [e for e in out if e.dim == 3 and e.keep]
    assert sorted(len(e.shapes) for e in solids) == [1, 1]
    return solids


def test_cad_occ_canonical_concentric_default_eps():
    """Mixed discretizations, default eps=1e-5, post-cut boundary.

    Pre-cut, disc and plate-hole each independently emit their OWN
    offset circle (disc=R+eps, plate hole=R-eps) -- uniformly 2*eps
    apart. That 2*eps separation is the PRE-cut invariant: it is what
    guarantees a genuine (non-degenerate) volumetric overlap for
    ``BRepAlgoAPI_Cut`` to engage on, rather than a graze.

    The sequential cut cascade (lower ``mesh_order`` acts as the tool --
    see ``cad_occ.py`` ~line 494-611) then cuts the plate (mesh_order=2)
    against the disc (mesh_order=1): the plate's hole boundary is
    replaced by the disc's (the tool's) surface. This is the designed
    "clean cut" outcome -- the shared interface is the tool's boundary,
    not a re-averaged or re-split radius -- and it collapses the scene
    to exactly ONE surviving circle radius: the disc's offset circle
    R+eps, where R is the registry's canonical (cross-entity fitted)
    nominal radius (~5.0, not exactly 5.0 -- the algebraic circle fit
    over point-tolerance-grid-snapped vertices carries a small, expected
    bias of the same order as the grid). This is a materially different
    (and correct) claim from the removed xfail's premise that both
    offset radii would survive 2*eps apart; the two-radii expectation
    was a plan-authoring error that ignored the cut cascade entirely.
    Note: this collapse is a property of the sequential cut, not of
    ``_resolve_piece_ownership`` (that function only assigns ownership
    of faces already shared/coincident during the final fragment pass;
    it never touches these already-non-overlapping post-cut boundaries).
    """
    from meshwell.cad_occ import cad_occ

    ents = _disc_and_plate()
    out = cad_occ(ents, identify_arcs=True)
    solids = _assert_two_clean_solids(out)
    circles = [c for e in solids for s in e.shapes for c in _circle_edges(s)]
    assert circles
    c0 = circles[0][0]
    radii = set()
    for (cx, cy), r in circles:
        assert (cx, cy) == pytest.approx(c0, abs=1e-9)
        radii.add(round(r, 9))
    assert len(radii) == 1  # sequential cut collapses to ONE shared boundary

    # The survivor is the disc's registered circle offset outward by eps
    # (disc is mesh_order=1, so it is the tool in the cut cascade; a CCW
    # solid ring offsets +eps -- see test_wire_emits_offset_canonical_circle).
    # Compare against the ACTUAL registry radius (not a literal 5.0)
    # since fitting circles on point_tolerance-grid-snapped vertices
    # (default point_tolerance=1e-3) introduces a small algebraic-fit
    # bias of its own -- observed here as ~1.3e-5, i.e. comparable to
    # eps itself. abs=1e-9 is tight because this is a direct comparison
    # against the SAME registry value the pipeline used internally, not
    # an independent re-derivation, so only floating-point noise from
    # the intervening OCC round-trip is expected.
    registry = ents[0].circle_registry
    assert registry is not None
    assert len(registry.clusters) == 1
    expected_radius = registry.clusters[0].radius + 1e-5
    (radius,) = radii
    assert radius == pytest.approx(expected_radius, abs=1e-9)
    # sanity: near nominal -- abs=1e-4 is loose enough to absorb the
    # point_tolerance=1e-3 grid-snap bias in the circle fit itself (the
    # ~1.3e-5 bias noted above), not just floating-point noise.
    assert radius == pytest.approx(5.0 + 1e-5, abs=1e-4)


def test_cad_occ_canonical_exact_eps_zero():
    """eps=0: both sides emit the IDENTICAL circle.

    Ownership resolves by exact coincidence + fragment merge. First
    executable proof of the canonical-exact mode -- if OCC misbehaves
    here, xfail and record the failure mode in Follow-ups; do not
    weaken the assertion.
    """
    from meshwell.cad_occ import cad_occ

    out = cad_occ(_disc_and_plate(), identify_arcs=True, perturbation=0.0)
    solids = _assert_two_clean_solids(out)
    circles = [c for e in solids for s in e.shapes for c in _circle_edges(s)]
    assert len({(round(c[0], 9), round(c[1], 9), round(r, 9)) for c, r in circles}) == 1


def test_cad_occ_promotion_chorded_side():
    """Plate hole is pure chords (classification asymmetry).

    ``identify_arcs`` is flipped off after stamping, simulating a
    classification asymmetry; promotion must pull it onto the disc's
    registered circle.
    """
    from meshwell.cad_common import apply_arc_params
    from meshwell.cad_occ import cad_occ

    ents = _disc_and_plate()
    apply_arc_params(ents, identify_arcs=True)
    ents[1].identify_arcs = False
    out = cad_occ(ents)
    solids = _assert_two_clean_solids(out)
    centers = [c for e in solids for s in e.shapes for c, _r in _circle_edges(s)]
    c0 = centers[0]
    for c in centers[1:]:
        assert c == pytest.approx(c0, abs=1e-9)


def test_fuzzy_defaults_by_regime():
    from meshwell.cad_occ import CAD_OCC

    proc = CAD_OCC(point_tolerance=1e-3, perturbation=1e-5)
    assert proc.cut_fuzzy_value == pytest.approx(0.8e-5)
    proc0 = CAD_OCC(point_tolerance=1e-3, perturbation=0.0)
    assert proc0.cut_fuzzy_value == pytest.approx(0.5e-3)
    assert CAD_OCC(point_tolerance=1e-3).perturbation == pytest.approx(
        1e-5
    )  # None = default


def test_ladder_warns_on_degenerate_zero_cut_fuzzy():
    from meshwell.validation import validate_tolerance_ladder

    with pytest.warns(UserWarning, match="cut_fuzzy_value=0"):
        validate_tolerance_ladder(
            perturbation=0.0, cut_fuzzy_value=0.0, fragment_fuzzy_value=1e-3
        )


def test_generate_mesh_smoke(tmp_path):
    from meshwell.orchestrator import generate_mesh

    generate_mesh(
        entities=_disc_and_plate(),
        dim=3,
        output_mesh=str(tmp_path / "out.msh"),
        identify_arcs=True,
        default_characteristic_length=2.0,
    )
    assert (tmp_path / "out.msh").exists()


def _face_area(shape):
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, props)
    return props.Mass()


def _solid_volume(shape):
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return props.Mass()


def test_polysurface_instanciate_occ_orients_cw_input():
    """CW-wound PolySurface input must still offset OUTWARD, not invert.

    ``PolySurface.instanciate_occ`` never canonicalized ring orientation
    before offsetting (unlike ``PolyPrism.instanciate_occ``'s extrude
    path), so a CW-wound input silently inverted the canonical offset
    direction (material-left-of-travel is orientation-relative): the
    face would SHRINK instead of grow. Build a deliberately CW 2x2
    square (area 4.0), stamp an exaggerated perturbation directly, and
    confirm the emitted face area grows to (2+2*eps)**2.
    """
    from meshwell.polysurface import PolySurface

    cw_square = Polygon([(-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0)])
    assert cw_square.exterior.is_ccw is False
    assert cw_square.area == pytest.approx(4.0)

    surf = PolySurface(polygons=cw_square, physical_name="s", mesh_order=1)
    surf.perturbation = 0.1  # exaggerated for a visible assertion
    shape = surf.instanciate_occ()

    expected_area = (2.0 + 2 * 0.1) ** 2
    assert _face_area(shape) == pytest.approx(expected_area, abs=1e-9)


def test_polyprism_non_extrude_loft_orients_buffered_polygons():
    """Non-extrude (tapered) loft must offset OUTWARD, not invert.

    ``PolyPrism._create_occ_volume`` (the OCC ``BRepOffsetAPI_
    ThruSections`` loft used when buffers vary with z) fed
    ``xy_surface_vertices`` straight from the GEOS-buffered polygon
    without canonicalizing ring orientation first. GEOS buffer output is
    uniformly CW regardless of input winding, so the canonical offset
    (material-left-of-travel) inverted: a perturbation stamp would
    shrink the loft instead of growing it. Compare a stamped vs.
    unstamped tapered prism (buffers vary with z -> extrude=False) and
    confirm the stamped one is LARGER.
    """
    from meshwell.polyprism import PolyPrism

    square = Polygon([(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)])
    buffers = {0.0: 0.0, 1.0: -0.2}

    baseline = PolyPrism(
        polygons=square, buffers=buffers, physical_name="p", mesh_order=1
    )
    assert baseline.extrude is False
    assert baseline.perturbation == 0.0  # class default
    baseline_volume = _solid_volume(baseline.instanciate_occ())

    stamped = PolyPrism(
        polygons=square, buffers=buffers, physical_name="p", mesh_order=1
    )
    stamped.perturbation = 0.1  # exaggerated for a visible assertion
    stamped_volume = _solid_volume(stamped.instanciate_occ())

    assert stamped_volume > baseline_volume
