import gdstk
import shapely.geometry as sg


def gdstk_to_shapely(cell, layer=None):
    """Convert GDSTK polygons to Shapely geometries."""
    polygons = []

    # Process this layer's polygons in the cell
    for polygon in cell.polygons:
        if polygon.layer != layer:
            continue

        # Convert points to valid shapely polygon
        points = [(float(x), float(y)) for x, y in polygon.points]
        if len(points) >= 3:  # Valid polygon needs at least 3 points
            poly = sg.Polygon(points)
            if not poly.is_valid:
                fixed = poly.buffer(0)  # Fix invalid polygon
                if isinstance(fixed, sg.Polygon):
                    polygons.append(fixed)
                elif isinstance(fixed, sg.MultiPolygon):
                    polygons.extend(list(fixed.geoms))
            else:
                polygons.append(poly)

    # Process all paths in the cell
    for path in cell.paths:
        if path.layer != layer:
            continue

        # Convert path to valid polygon
        poly = path.to_polygons()
        for p in poly:
            points = [(float(x), float(y)) for x, y in p.points]
            if len(points) >= 3:
                shapely_poly = sg.Polygon(points)
                if not shapely_poly.is_valid:
                    fixed = shapely_poly.buffer(0)
                    if isinstance(fixed, sg.Polygon):
                        polygons.append(fixed)
                    elif isinstance(fixed, sg.MultiPolygon):
                        polygons.extend(list(fixed.geoms))
                else:
                    polygons.append(shapely_poly)

    return sg.MultiPolygon(polygons)


def read_gds_layers(gds_file, cell_name=None, layers=None):
    """Read GDS file and convert to Shapely geometries.

    Args:
        gds_file: Path to GDS file
        cell_name: Optional name of cell to process (uses top cell if None)
        layers: Optional list of layer tuples to filter. If None, load all layers.

    Returns:
        Dict of {(gds_layer, gds_datatype): shapely.MultiPolygon}
    """
    # Read the GDS file
    library = gdstk.read_gds(gds_file)

    # Get the specified cell or top cell
    if cell_name:
        cell = library.cells[cell_name]
    else:
        cell = library.top_level()[0]  # Get first top-level cell

    # Get the desired polygons
    layer_to_multipolygons = {}

    if layers is None:
        # Get all unique layers used in the cell
        layers = set()
        for polygon in cell.polygons:
            layers.add((polygon.layer, polygon.datatype))
        for path in cell.paths:
            layers.add((path.layer, path.datatype))

    for layer in layers:
        layer_to_multipolygons[layer] = gdstk_to_shapely(cell, layer)

    return layer_to_multipolygons
