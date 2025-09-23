"""GDS import routines."""
import gdstk
import shapely.geometry as sg
from shapely.ops import unary_union


def gdstk_to_shapely(cell, layer_tuple):
    """Convert GDSTK polygons to Shapely geometries."""
    polygons = []
    layer, datatype = layer_tuple

    # Process this layer's polygons in the cell
    for polygon in cell.polygons:
        if polygon.layer != layer or polygon.datatype != datatype:
            continue

        # Convert points to float tuples
        points = [(float(x), float(y)) for x, y in polygon.points]

        if len(points) >= 3:
            # Create polygon - let Shapely handle holes automatically
            poly = sg.Polygon(points)
            if not poly.is_valid:
                # Use buffer(0) to fix self-intersections and create proper holes
                fixed = poly.buffer(0)
                if isinstance(fixed, sg.Polygon):
                    polygons.append(fixed)
                elif isinstance(fixed, sg.MultiPolygon):
                    polygons.extend(list(fixed.geoms))
            else:
                polygons.append(poly)

    if polygons:
        return unary_union(polygons)
    return sg.MultiPolygon([])


def read_gds_layers(gds_file, cell_name=None, layers=None):
    """Read GDS file and convert to Shapely geometries."""
    # Read the GDS file
    library = gdstk.read_gds(gds_file)

    # Get the specified cell or top cell
    if cell_name:
        cell = next((c for c in library.cells if c.name == cell_name), None)
        if cell is None:
            raise ValueError(f"Cell '{cell_name}' not found in GDS file")
    else:
        cell = library.top_level()[0]

    # Get all layers if none specified
    if layers is None:
        layers = set()
        for polygon in cell.polygons:
            layers.add((polygon.layer, polygon.datatype))
        for path in cell.paths:
            layers.add((path.layer, path.datatype))

    # Process each layer
    layer_to_multipolygons = {}
    for layer_tuple in layers:
        mp = gdstk_to_shapely(cell, layer_tuple)
        layer_to_multipolygons[layer_tuple] = mp

    return layer_to_multipolygons
