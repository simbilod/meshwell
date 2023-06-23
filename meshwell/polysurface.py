class PolySurface:
    """
    Creates bottom-up GMSH polygonal surfaces formed by list of shapely (multi)polygon.

    Attributes:
        polygons: list of shapely (Multi)Polygon
        model: GMSH model to synchronize
    """

    def __init__(
        self,
        polygons,
        model,
    ):
        # Parse (multi)polygons
        self.polygons = list(
            polygons.geoms if hasattr(polygons, "geoms") else [polygons]
        )

        # Track gmsh entities for bottom-up volume definition
        self.model = model

    def _parse_coords(self, coords):
        """Chooses z=0 if the provided coordinates are 2D."""
        return (coords[0], coords[1], 0) if len(coords) == 2 else coords

    def get_gmsh_polygons(self):
        """Returns the fused GMSH surfaces within model from the polygons."""
        surfaces = [self.add_surface_with_holes(entry) for entry in self.polygons]
        if len(surfaces) <= 1:
            return surfaces
        dimtags = self.model.occ.fuse(
            [(2, surfaces[0])],
            [(2, tag) for tag in surfaces[1:]],
            removeObject=True,
            removeTool=True,
        )[0]
        self.model.occ.synchronize()
        return [tag for dim, tag in dimtags]

    def add_surface_with_holes(self, polygon):
        """Returns surface, removing intersection with hole surfaces."""
        exterior = self.model.add_surface(
            [self._parse_coords(coords) for coords in polygon.exterior.coords]
        )
        interior_tags = [
            self.model.add_surface(
                [self._parse_coords(coords) for coords in interior.coords],
            )
            for interior in polygon.interiors
        ]
        for interior_tag in interior_tags:
            exterior = self.model.occ.cut(
                [(2, exterior)], [(2, interior_tag)], removeObject=True, removeTool=True
            )
            self.model.occ.synchronize()
            exterior = exterior[0][0][1]  # Parse `outDimTags', `outDimTagsMap'
        return exterior

    def instanciate(self):
        polysurface = self.get_gmsh_polygons()
        self.model.occ.synchronize()
        return [(2, polysurface)]


# def PolySurface(
#     polygons,
#     model,
# ):
#     """Functional wrapper around PolySurfaceClass."""
#     polysurface = PolySurfaceClass(polygons=polygons, model=model).get_gmsh_polygons()
#     model.occ.synchronize()
#     return polysurface
