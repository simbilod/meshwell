import shapely
from meshwell.geometry import Geometry


class PolySurfaceClass(Geometry):
    """
    Creates bottom-up GMSH polygonal surfaces formed by list of shapely (multi)polygon.

    Attributes:
        polygons: list of shapely (Multi)Polygon
        model: GMSH model to synchronize
    """

    def __init__(
        self,
        model,
        polygons,
    ):
        self.model = model

        # Parse (multi)polygons
        self.polygons = list(
            polygons.geoms if hasattr(polygons, "geoms") else [polygons]
        )

        # Track gmsh entities for bottom-up volume definition
        self.points = {}
        self.segments = {}

    def _parse_coords(self, coords):
        """Chooses z=0 if the provided coordinates are 2D."""
        return (coords[0], coords[1], 0) if len(coords) == 2 else coords

    def get_gmsh_polygons(self):
        """Returns the GMSH surfaces within model from the polygons."""
        return [self._add_surface_with_holes(polygon) for polygon in self.polygons]

    def _add_surface_with_holes(self, polygon):
        """Returns surface, removing intersection with hole surfaces."""
        exterior = self._add_surface(
            [self._parse_coords(coords) for coords in polygon.exterior.coords]
        )
        interior_tags = [
            self._add_surface(
                [self._parse_coords(coords) for coords in interior.coords],
            )
            for interior in polygon.interiors
        ]
        for interior_tag in interior_tags:
            exterior = self.model.cut(
                [(2, exterior)], [(2, interior_tag)], removeObject=True, removeTool=True
            )
            self.model.synchronize()
            exterior = exterior[0][0][1]  # Parse `outDimTags', `outDimTagsMap'
        return exterior


def PolySurface(
    model,
    polygons,
):
    """Functional wrapper around PolySurfaceClass."""
    polysurface = PolySurfaceClass(polygons=polygons, model=model).get_gmsh_polygons()
    model.synchronize()
    return polysurface


if __name__ == "__main__":
    polygon1 = shapely.Polygon(
        [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
        holes=([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]],),
    )
    polygon2 = shapely.Polygon([[-1, -1], [-2, -1], [-2, -2], [-1, -2], [-1, -1]])
    polygon = shapely.MultiPolygon([polygon1, polygon2])

    import gmsh

    occ = gmsh.model.occ
    gmsh.initialize()
    gmsh.option.setNumber("Geometry.OCCBooleanPreserveNumbering", 1)

    poly2D = PolySurface(polygons=polygon, model=occ)
    occ.synchronize()

    gmsh.model.mesh.generate(3)
    gmsh.write("mesh.msh")
