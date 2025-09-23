import gdstk
from pathlib import Path
import shapely.geometry as sg
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

        # Verify we got the expected number of layers
        assert len(layer_polygons) == 6, "Expected 6 layers in output"

        # Check that each layer exists and has valid geometry
        for layer in range(6):
            assert (layer, 0) in layer_polygons, f"Layer {layer} missing from output"
            assert not layer_polygons[(layer, 0)].is_empty, (
                f"Layer {layer} has empty geometry"
            )
            assert layer_polygons[(layer, 0)].is_valid, (
                f"Layer {layer} has invalid geometry"
            )

        # Check specific properties of some layers
        # Layer 1 should have 3 circles
        assert isinstance(layer_polygons[(1, 0)], sg.MultiPolygon)
        assert len(list(layer_polygons[(1, 0)].geoms)) == 3

        # Layer 2 should have a polygon with a hole
        layer2_poly = layer_polygons[(2, 0)]
        if isinstance(layer2_poly, sg.MultiPolygon):
            polygon = list(layer2_poly.geoms)[0]
        else:
            polygon = layer2_poly
        assert len(polygon.interiors) > 0

        # Layers 3-5 should overlap
        layer3_poly = layer_polygons[(3, 0)]
        layer4_poly = layer_polygons[(4, 0)]
        layer5_poly = layer_polygons[(5, 0)]
        assert layer3_poly.intersects(layer4_poly)
        assert layer4_poly.intersects(layer5_poly)
        assert layer5_poly.intersects(layer3_poly)

    finally:
        # Cleanup
        if test_file.exists():
            test_file.unlink()


if __name__ == "__main__":
    test_gds_to_shapely()
