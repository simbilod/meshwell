from __future__ import annotations

import shapely
from meshwell.polyline import PolyLine
from meshwell.polysurface import PolySurface
from meshwell.cad import cad
from meshwell.mesh import mesh


def test_polyline_basic():
    """Test basic PolyLine functionality with single and multiple LineStrings."""
    # Single LineString
    line1 = shapely.LineString([(0, 0), (1, 1), (2, 0)])
    polyline1 = PolyLine(linestrings=line1, physical_name="line1")

    cad(entities_list=[polyline1], output_file="test_polyline_single")
    mesh(
        input_file="test_polyline_single.xao",
        output_file="test_polyline_single.msh",
        dim=1,
        default_characteristic_length=0.1,
        n_threads=1,
    )

    # Multiple LineStrings
    line2 = shapely.LineString([(0, -1), (1, -1), (2, -1)])
    line3 = shapely.LineString([(-0.5, 0), (-0.5, 1)])
    polyline_multi = PolyLine(
        linestrings=[line1, line2, line3], physical_name="multi_lines"
    )

    cad(entities_list=[polyline_multi], output_file="test_polyline_multi")
    mesh(
        input_file="test_polyline_multi.xao",
        output_file="test_polyline_multi.msh",
        dim=1,
        default_characteristic_length=0.1,
        n_threads=1,
    )


def test_polyline_embedded_in_polysurface():
    """Test PolyLine embedded within PolySurface boundary."""
    # Create a rectangular surface
    surface_polygon = shapely.Polygon([(-2, -2), (2, -2), (2, 2), (-2, 2), (-2, -2)])
    polysurface = PolySurface(polygons=surface_polygon, physical_name="surface")

    # Create lines within the surface
    embedded_line1 = shapely.LineString([(-1, 0), (1, 0)])  # Horizontal line
    embedded_line2 = shapely.LineString([(0, -1), (0, 1)])  # Vertical line
    polyline = PolyLine(
        linestrings=[embedded_line1, embedded_line2], physical_name="embedded_lines"
    )

    # Create CAD with both surface and embedded lines
    cad(entities_list=[polysurface, polyline], output_file="test_polyline_embedded")
    mesh(
        input_file="test_polyline_embedded.xao",
        output_file="test_polyline_embedded.msh",
        dim=2,  # 2D mesh to capture the surface
        default_characteristic_length=0.2,
        n_threads=1,
    )


def test_polyline_intersecting():
    """Test multiple intersecting PolyLines with different physical names."""
    # Create intersecting lines forming an X
    line1 = shapely.LineString([(-1, -1), (0, 0)])
    line2 = shapely.LineString([(-1, -1), (1, 1)])

    polyline1 = PolyLine(linestrings=line1, physical_name="diagonal1", mesh_order=1)
    polyline2 = PolyLine(linestrings=line2, physical_name="diagonal2", mesh_order=2)

    # Create CAD with intersecting lines
    cad(entities_list=[polyline1, polyline2], output_file="test_polyline_intersecting")
    mesh(
        input_file="test_polyline_intersecting.xao",
        output_file="test_polyline_intersecting.msh",
        dim=1,
        default_characteristic_length=0.1,
        n_threads=1,
    )


def test_polyline_mixed_dimensional():
    """Test mixed 1D PolyLine and 2D PolySurface entities."""
    # Create a simple rectangular surface
    surface_polygon = shapely.Polygon([(-2, -2), (2, -2), (2, 2), (-2, 2), (-2, -2)])
    polysurface = PolySurface(polygons=surface_polygon, physical_name="surface")

    # Create a simple line that doesn't intersect the surface boundary
    simple_line = shapely.LineString(
        [(-1, 0), (1, 0)]
    )  # Horizontal line inside surface
    polyline = PolyLine(linestrings=simple_line, physical_name="embedded_line")

    # Create CAD with mixed dimensions - test separately first
    cad(entities_list=[polysurface], output_file="test_surface_only")
    mesh(
        input_file="test_surface_only.xao",
        output_file="test_surface_only.msh",
        dim=2,
        default_characteristic_length=0.2,
        n_threads=1,
    )

    cad(entities_list=[polyline], output_file="test_line_only")
    mesh(
        input_file="test_line_only.xao",
        output_file="test_line_only.msh",
        dim=1,
        default_characteristic_length=0.2,
        n_threads=1,
    )


def test_polyline_multilinestring():
    """Test PolyLine with MultiLineString geometry."""
    # Create a MultiLineString
    line1 = shapely.LineString([(0, 0), (1, 1)])
    line2 = shapely.LineString([(2, 0), (3, 1)])
    multi_line = shapely.MultiLineString([line1, line2])

    polyline = PolyLine(linestrings=multi_line, physical_name="multi_linestring")

    cad(entities_list=[polyline], output_file="test_polyline_multilinestring")
    mesh(
        input_file="test_polyline_multilinestring.xao",
        output_file="test_polyline_multilinestring.msh",
        dim=1,
        default_characteristic_length=0.1,
        n_threads=1,
    )


if __name__ == "__main__":
    test_polyline_basic()
