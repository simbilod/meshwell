class Prism:
    """
    Creates a bottom-up GMSH "prism" formed by a polygon associated with (optional) z-dependent grow/shrink operations.

    Attributes:
        polygons: list of shapely (Multi)Polygon
        buffers: dict of {z: buffer} used to shrink/grow base polygons at specified z-values
    """

    def __init__(
        self,
        polygons,
        buffers,
        model,
    ):
        # Model
        self.model = model

        # Parse buffers
        self.buffered_polygons = self._get_buffered_polygons(polygons, buffers)

    def get_gmsh_volumes(self):
        """Returns the fused GMSH volumes within model from the polygons and buffers."""
        volumes = [
            self._add_volume_with_holes(entry) for entry in self.buffered_polygons
        ]
        if len(volumes) <= 1:
            self.model.occ.synchronize()
            return volumes
        dimtags = self.model.occ.fuse(
            [(3, volumes[0])],
            [(3, tag) for tag in volumes[1:]],
            removeObject=True,
            removeTool=True,
        )[0]
        self.model.occ.synchronize()
        return [tag for dim, tag in dimtags]

    def _get_buffered_polygons(self, polygons, buffers):
        """Break up polygons on each layer into lists of polygons:z tuples according to buffer entries.

        Arguments (implicit):
            polygons: polygons to bufferize
            buffers: {z: buffer} values to apply to the polygons

        Returns:
            buffered_polygons: list of (z, buffered_polygons)
        """
        all_polygons_list = []
        for polygon in polygons.geoms if hasattr(polygons, "geoms") else [polygons]:
            current_buffers = []
            for z, width_buffer in buffers.items():
                current_buffers.append((z, polygon.buffer(width_buffer, join_style=2)))
            all_polygons_list.append(current_buffers)

        return all_polygons_list

    def _add_volume(self, entry, exterior=True, interior_index=0):
        """Create shape from a list of the same buffered polygon and a list of z-values.
        Args:
            polygons: shapely polygons from the GDS
            zs: list of z-values for each polygon
        Returns:
            ID of the added volume
        """
        bottom_polygon_vertices = self.xy_surface_vertices(
            entry, 0, exterior, interior_index
        )
        gmsh_surfaces = [self.model.add_surface(bottom_polygon_vertices)]

        top_polygon_vertices = self.xy_surface_vertices(
            entry, -1, exterior, interior_index
        )
        gmsh_surfaces.append(self.model.add_surface(top_polygon_vertices))

        # Draw vertical surfaces
        for pair_index in range(len(entry) - 1):
            if exterior:
                bottom_polygon = entry[pair_index][1].exterior.coords
                top_polygon = entry[pair_index + 1][1].exterior.coords
            else:
                bottom_polygon = entry[pair_index][1].interiors[interior_index].coords
                top_polygon = entry[pair_index + 1][1].interiors[interior_index].coords
            bottom_z = entry[pair_index][0]
            top_z = entry[pair_index + 1][0]
            for facet_pt_ind in range(len(bottom_polygon) - 1):
                facet_pt1 = (
                    bottom_polygon[facet_pt_ind][0],
                    bottom_polygon[facet_pt_ind][1],
                    bottom_z,
                )
                facet_pt2 = (
                    bottom_polygon[facet_pt_ind + 1][0],
                    bottom_polygon[facet_pt_ind + 1][1],
                    bottom_z,
                )
                facet_pt3 = (
                    top_polygon[facet_pt_ind + 1][0],
                    top_polygon[facet_pt_ind + 1][1],
                    top_z,
                )
                facet_pt4 = (
                    top_polygon[facet_pt_ind][0],
                    top_polygon[facet_pt_ind][1],
                    top_z,
                )
                facet_vertices = [facet_pt1, facet_pt2, facet_pt3, facet_pt4, facet_pt1]
                gmsh_surfaces.append(self.model.add_surface(facet_vertices))

        # Return volume from closed shell
        surface_loop = self.model.occ.add_surface_loop(gmsh_surfaces)
        return self.model.occ.add_volume([surface_loop])

    def xy_surface_vertices(self, entry, arg1, exterior, interior_index):
        """"""
        # Draw xy surface
        polygon = entry[arg1][1]
        polygon_z = entry[arg1][0]
        return (
            [(x, y, polygon_z) for x, y in polygon.exterior.coords]
            if exterior
            else [
                (x, y, polygon_z) for x, y in polygon.interiors[interior_index].coords
            ]
        )

    def _add_volume_with_holes(self, entry):
        """Returns volume, removing intersection with hole volumes."""
        exterior = self._add_volume(entry, exterior=True)
        interiors = [
            self._add_volume(
                entry,
                exterior=False,
                interior_index=interior_index,
            )
            for interior_index in range(len(entry[0][1].interiors))
        ]
        if interiors:
            for interior in interiors:
                exterior = self.model.occ.cut(
                    [(3, exterior)], [(3, interior)], removeObject=True, removeTool=True
                )
                self.model.occ.synchronize()
                exterior = exterior[0][0][1]  # Parse `outDimTags', `outDimTagsMap'
        return exterior

    def instanciate(self):
        """Returns dim tag from entity."""
        prism = self.get_gmsh_volumes()
        self.model.occ.synchronize()
        return [(3, prism)]


# def Prism(
#     polygons,
#     model,
#     buffers=None,
# ):
#     """Functional wrapper around PrismClass."""
#     prism = PrismClass(
#         polygons=polygons, buffers=buffers, model=model
#     ).get_gmsh_volumes()
#     model.occ.synchronize()
#     return prism
