import gdstk
import numpy as np
from pathlib import Path

from meshwell.import_gds import read_gds_layers


def write_test_gds(filename):
    """Create a complex GDS with multiple layers and nested structures."""
    # Create library and cell
    lib = gdstk.Library()
    main = lib.new_cell("TOP")

    # Create a complex path on layer 0
    points = [(0, 0), (10, 0), (10, 10), (20, 10)]
    path = gdstk.FlexPath(points, 2, layer=0, ends="round")
    main.add(path)

    # Create some circles on layer 1
    circles = []
    for x in range(0, 30, 10):
        circle = gdstk.ellipse((x, 20), 3, layer=1)
        circles.append(circle)
    main.add(*circles)

    # Create a complex polygon with holes on layer 2
    outer = gdstk.regular_polygon((15, 15), 8, 6, layer=2)
    inner = gdstk.regular_polygon((15, 15), 3, 6, layer=2)
    complex_poly = gdstk.boolean(outer, inner, "not", layer=2)
    main.add(*complex_poly)

    # Create overlapping polygons on layers 3-5
    overlapping_shapes = []

    # Layer 3: Large rectangle
    rect = gdstk.rectangle((5, 25), (25, 35), layer=3)
    overlapping_shapes.append(rect)

    # Layer 4: Overlapping triangle
    triangle_pts = [(10, 28), (20, 28), (15, 33)]
    triangle = gdstk.Polygon(triangle_pts, layer=4)
    overlapping_shapes.append(triangle)

    # Layer 5: Overlapping circle
    circle = gdstk.ellipse((15, 30), 3, layer=5)
    overlapping_shapes.append(circle)

    main.add(*overlapping_shapes)

    # Write the GDS file
    lib.write_gds(filename)


def test_gds_to_shapely():
    """Test creating and importing a complex GDS file."""
    # Create a temporary GDS file
    test_file = Path("test_complex.gds")

    try:
        # Write the complex GDS
        write_test_gds(test_file)

        # Convert to Shapely geometries by layer
        layer_polygons = read_gds_layers(test_file)

        # Verify layer 0 (path)
        path_geom = layer_polygons[(0, 0)]
        assert not path_geom.is_empty, "Path layer should contain geometry"

        # Verify layer 1 (circles)
        circles_geom = layer_polygons[(1, 0)]
        assert len(list(circles_geom.geoms)) == 3, "Should have 3 circles"
        for circle in circles_geom.geoms:
            # Circles should be approximately circular
            assert abs(circle.area - np.pi * 3**2) < 1.0

        # Verify layer 2 (complex polygon with hole)
        complex_geom = layer_polygons[(2, 0)]
        assert (
            not complex_geom.is_empty
        ), "Complex polygon layer should contain geometry"
        assert len(list(complex_geom.geoms)) == 1, "Should have 1 complex polygon"
        complex_poly = list(complex_geom.geoms)[0]
        assert len(complex_poly.interiors) == 1, "Complex polygon should have 1 hole"

        # Verify overlapping shapes on layers 3-5
        rect_geom = layer_polygons[(3, 0)]
        triangle_geom = layer_polygons[(4, 0)]
        circle_geom = layer_polygons[(5, 0)]

        assert not rect_geom.is_empty, "Rectangle layer should contain geometry"
        assert not triangle_geom.is_empty, "Triangle layer should contain geometry"
        assert not circle_geom.is_empty, "Circle layer should contain geometry"

        # Verify overlapping relationships
        assert rect_geom.intersects(triangle_geom), "Rectangle should overlap triangle"
        assert rect_geom.intersects(circle_geom), "Rectangle should overlap circle"
        assert triangle_geom.intersects(circle_geom), "Triangle should overlap circle"

    finally:
        # Cleanup
        if test_file.exists():
            test_file.unlink()


if __name__ == "__main__":
    test_gds_to_shapely()
