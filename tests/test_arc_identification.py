import gmsh
import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

from meshwell.geometry_entity import GeometryEntity
from meshwell.polyline import PolyLine
from meshwell.polyprism import PolyPrism
from meshwell.polysurface import PolySurface


def test_decompose_vertices_no_arcs():
    """Test decomposition when arc identification is disabled."""
    ge = GeometryEntity()
    vertices = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    segments = ge.decompose_vertices(
        vertices, identify_arcs=False, min_arc_points=4, arc_tolerance=1e-3
    )

    # Should be 3 line segments (0-1, 1-2, 2-3)
    assert len(segments) == 3
    for seg in segments:
        assert not seg.is_arc
        assert len(seg.points) == 2


def test_decompose_vertices_perfect_arc():
    """Test decomposition with a perfect semi-circle."""
    ge = GeometryEntity()
    # Semi-circle with radius 1, center (0,0)
    theta = np.linspace(0, np.pi, 10)
    vertices = [(np.cos(t), np.sin(t), 0) for t in theta]

    segments = ge.decompose_vertices(
        vertices, identify_arcs=True, min_arc_points=4, arc_tolerance=1e-3
    )

    assert len(segments) == 1
    assert segments[0].is_arc
    assert np.allclose(segments[0].center, (0, 0, 0), atol=1e-3)
    assert np.allclose(segments[0].radius, 1.0, atol=1e-3)
    assert len(segments[0].points) == 10


def test_decompose_vertices_noisy_arc():
    """Test decomposition with a noisy arc."""
    ge = GeometryEntity()
    theta = np.linspace(0, np.pi / 2, 10)
    # Add small noise (within tolerance)
    vertices = [(np.cos(t) + 0.0001, np.sin(t) - 0.0001, 0) for t in theta]

    segments = ge.decompose_vertices(
        vertices, identify_arcs=True, min_arc_points=4, arc_tolerance=1e-2
    )

    assert len(segments) == 1
    assert segments[0].is_arc
    assert segments[0].radius > 0.9
    assert segments[0].radius < 1.1


def test_decompose_vertices_mixed():
    """Test decomposition with both lines and arcs."""
    ge = GeometryEntity()

    # Line from (0,0) to (1,0)
    line1 = [(x, 0, 0) for x in np.linspace(0, 1, 5)]
    # Arc from (1,0) to (0,1)
    theta = np.linspace(0, np.pi / 2, 10)
    arc = [(np.cos(t), np.sin(t), 0) for t in theta]

    vertices = line1 + arc[1:]  # Avoid duplicate at (1,0)

    segments = ge.decompose_vertices(
        vertices, identify_arcs=True, min_arc_points=4, arc_tolerance=1e-3
    )

    # Expected: 4 line segments for line1, then 1 arc

    assert any(not seg.is_arc for seg in segments)
    assert any(seg.is_arc for seg in segments)


def test_polyline_arc_instantiate_gmsh():
    """Test PolyLine arc instantiation in GMSH."""
    theta = np.linspace(0, np.pi / 2, 10)
    vertices = [(np.cos(t), np.sin(t), 0) for t in theta]
    ls = LineString(vertices)

    pl = PolyLine(ls, identify_arcs=True, min_arc_points=4, arc_tolerance=1e-3)

    gmsh.initialize()
    gmsh.model.add("test_pl")

    # This calls _create_wire_from_linestring
    dimtags = pl.instanciate()

    assert len(dimtags) == 1
    assert dimtags[0][0] == 1  # Dimension 1

    gmsh.finalize()


def test_polyline_arc_instantiate_occ():
    """Test PolyLine arc instantiation in OCC."""
    theta = np.linspace(0, np.pi / 2, 10)
    vertices = [(np.cos(t), np.sin(t), 0) for t in theta]
    ls = LineString(vertices)

    pl = PolyLine(ls, identify_arcs=True, min_arc_points=4, arc_tolerance=1e-3)

    # This calls instanciate_occ
    shape = pl.instanciate_occ()

    assert shape is not None


def test_polyline_arc_instantiate_gmsh_offgrid():
    """PolyLine arcs must instantiate even when endpoints are off the grid.

    Regression: center-form addCircleArc(start, center, end) failed with
    'Could not create circle arc' because grid-snapped endpoints sit at
    unequal distances from the (grid-rounded) fitted center.
    """
    cx, cy, r = 0.0004437, 0.0007213, 2.0
    theta = np.linspace(0.3, 1.9, 9)
    vertices = [(cx + r * np.cos(t), cy + r * np.sin(t), 0) for t in theta]
    pl = PolyLine(
        LineString(vertices),
        identify_arcs=True,
        min_arc_points=5,
        arc_tolerance=1e-3,
    )

    gmsh.initialize()
    try:
        gmsh.model.add("test_pl_offgrid")
        dimtags = pl.instanciate()
        assert len(dimtags) == 1
        assert dimtags[0][0] == 1
    finally:
        gmsh.finalize()


def test_polysurface_arc_instantiate_gmsh():
    """Test PolySurface arc instantiation in GMSH."""
    theta = np.linspace(0, np.pi / 2, 10)
    vertices = [(np.cos(t), np.sin(t)) for t in theta]
    vertices += [(0, 1), (0, 0), (1, 0)]
    poly = Polygon(vertices)

    ps = PolySurface(poly, identify_arcs=True, min_arc_points=4, arc_tolerance=1e-3)

    gmsh.initialize()
    gmsh.model.add("test_ps")

    dimtags = ps.instanciate()

    assert len(dimtags) == 1
    assert dimtags[0][0] == 2  # Dimension 2

    gmsh.finalize()


def test_polysurface_arc_instantiate_occ():
    """Test PolySurface arc instantiation in OCC."""
    theta = np.linspace(0, np.pi / 2, 10)
    vertices = [(np.cos(t), np.sin(t)) for t in theta]
    vertices += [(0, 1), (0, 0), (1, 0)]
    poly = Polygon(vertices)

    ps = PolySurface(poly, identify_arcs=True, min_arc_points=4, arc_tolerance=1e-3)

    shape = ps.instanciate_occ()

    assert shape is not None


def test_polyprism_arc_instantiate_gmsh():
    """Test PolyPrism arc instantiation in GMSH."""
    theta = np.linspace(0, np.pi / 2, 10)
    vertices = [(np.cos(t), np.sin(t)) for t in theta]
    vertices += [(0, 1), (0, 0), (1, 0)]
    poly = Polygon(vertices)

    pp = PolyPrism(
        poly,
        buffers={0: 0, 1: 0},
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
    )

    gmsh.initialize()
    gmsh.model.add("test_pp")

    # pp.instanciate only touches cad_model when subdivision is set; pass None.
    dimtags = pp.instanciate(None)

    assert len(dimtags) >= 1
    assert dimtags[0][0] == 3  # Dimension 3

    gmsh.finalize()


def test_polyprism_arc_instantiate_occ():
    """Test PolyPrism arc instantiation in OCC."""
    theta = np.linspace(0, np.pi / 2, 10)
    vertices = [(np.cos(t), np.sin(t)) for t in theta]
    vertices += [(0, 1), (0, 0), (1, 0)]
    poly = Polygon(vertices)

    pp = PolyPrism(
        poly,
        buffers={0: 0, 1: 0},
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
    )

    shape = pp.instanciate_occ()

    assert shape is not None


def test_polyprism_arc_error_no_extrude():
    """Test that PolyPrism raises error if identify_arcs=True and extrude=False."""
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    with pytest.raises(
        NotImplementedError,
        match="Arc identification is currently only supported for PolyPrism when extrude=True",
    ):
        PolyPrism(poly, buffers={0: 0, 1: 0.1}, identify_arcs=True)


def test_plot_decomposition_returns_axes():
    """Test that plot_decomposition returns a matplotlib axes object."""
    import matplotlib.pyplot as plt

    theta = np.linspace(0, np.pi / 2, 10)
    vertices = [(np.cos(t), np.sin(t), 0) for t in theta]
    pl = PolyLine(LineString(vertices), identify_arcs=True)

    ax = pl.plot_decomposition()
    assert isinstance(ax, plt.Axes)
    plt.close()


def test_occ_arc_wire_preserves_subtolerance_offsets():
    """The OCC arc path must not round coordinates to the point_tolerance grid.

    Rounding undoes the sub-tolerance perturbation buffer applied by
    cad_common.prepare_entities (buffered coords sit ~1e-5 off grid points
    and round straight back), silently disabling the pre-cut overlap
    strategy for arc-identified entities. Only the dedup KEY may be
    quantized -- the same scheme _add_point_with_tolerance uses for gmsh.
    """
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    from meshwell.geometry_entity import GeometryEntity

    # Pythagorean lattice points on the r=5 circle (all grid-exact at 1e-3),
    # then shift everything by 1e-5 in x: still a perfect circle, but every
    # coordinate is now sub-tolerance off the grid.
    lattice = [(5, 0), (4, 3), (3, 4), (0, 5), (-3, 4), (-4, 3), (-5, 0)]
    verts = [(x + 1e-5, float(y), 0.0) for x, y in lattice]

    ge = GeometryEntity(point_tolerance=1e-3)
    wire = ge._make_occ_wire_from_vertices(
        verts, identify_arcs=True, min_arc_points=5, arc_tolerance=1e-3
    )

    xs = []
    exp = TopExp_Explorer(wire, TopAbs_VERTEX)
    while exp.More():
        p = BRep_Tool.Pnt_s(TopoDS.Vertex_s(exp.Current()))
        xs.append(p.X())
        exp.Next()
    assert xs, "wire has no vertices"
    offsets = [abs(x - round(x, 3)) for x in xs]
    assert max(offsets) == pytest.approx(
        1e-5, rel=0.05
    ), f"sub-tolerance offset was rounded away: offsets={offsets}"


def test_arc_acceptance_bounds_emitted_deviation():
    """Acceptance must bound the emitted circle's MAX sample deviation.

    Detection gates the least-squares RMSE, but the emitted edge only
    interpolates 3 samples; on shallow (sagitta-starved) windows the
    emitted circle deviated from the other samples by more than
    arc_tolerance.
    """
    from meshwell.geometry_entity import (
        _decompose_vertices_3d,
        _three_point_circle_2d,
    )

    rng = np.random.default_rng(7)
    arc_tol = 1e-3
    worst = 0.0
    for _ in range(300):
        r = rng.uniform(5, 100)
        cx, cy = rng.uniform(-1, 1, 2)
        a0 = rng.uniform(0, 2 * np.pi)
        span = rng.uniform(0.05, 0.5)
        n = rng.integers(6, 15)
        t = np.linspace(a0, a0 + span, n)
        pts = np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])
        pts = np.round(pts / 1e-3) * 1e-3  # constructor grid snap
        verts = [(x, y, 0.0) for x, y in pts]
        segs = _decompose_vertices_3d(
            verts,
            point_tolerance=1e-3,
            identify_arcs=True,
            min_arc_points=5,
            arc_tolerance=arc_tol,
        )
        for seg in segs:
            if not seg.is_arc:
                continue
            w = np.array([(p[0], p[1]) for p in seg.points])
            mid = len(w) // 2
            emitted = _three_point_circle_2d(tuple(w[0]), tuple(w[mid]), tuple(w[-1]))
            assert emitted is not None
            (ecx, ecy), er = emitted
            dev = np.abs(np.hypot(w[:, 0] - ecx, w[:, 1] - ecy) - er).max()
            worst = max(worst, dev)
    assert worst <= arc_tol, f"emitted arc deviates {worst:.2e} > {arc_tol:g}"


def test_full_circle_still_detected_with_max_dev_gate():
    """The max-deviation gate must not break closed-circle detection.

    Emission splits closed windows through quarter samples, not the
    start/mid/end 3-point circle.
    """
    from meshwell.geometry_entity import _decompose_vertices_3d

    t = np.linspace(0, 2 * np.pi, 33)
    pts = np.column_stack([2.0 * np.cos(t), 2.0 * np.sin(t)])
    pts = np.round(pts / 1e-3) * 1e-3
    verts = [(x, y, 0.0) for x, y in pts]
    segs = _decompose_vertices_3d(
        verts,
        point_tolerance=1e-3,
        identify_arcs=True,
        min_arc_points=5,
        arc_tolerance=1e-3,
    )
    assert any(s.is_arc for s in segs)
