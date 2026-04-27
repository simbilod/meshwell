import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

import gmsh
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
