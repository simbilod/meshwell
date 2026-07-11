"""Tests for pipeline-level arc identification + the canonical geometry pipeline."""
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
