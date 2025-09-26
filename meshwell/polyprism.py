"""Gmsh polyprism definitions."""
import gmsh
from shapely.geometry import MultiPolygon, Polygon

from meshwell.cad import CAD
from meshwell.geometry_entity import GeometryEntity
from meshwell.validation import format_physical_name


class PolyPrism(GeometryEntity):
    """Creates a bottom-up GMSH "prism" formed by a polygon associated with (optional) z-dependent grow/shrink operations.

    Attributes:
        polygons: list of shapely (Multi)Polygon
        buffers: dict of {z: buffer} used to shrink/grow base polygons at specified z-values
        physical_name: name of the physical this entity will belong to
        mesh_order: priority of the entity if it overlaps with others (lower numbers override higher numbers)
        mesh_bool: if True, entity will be meshed; if not, will not be meshed (useful to tag boundaries)

    """

    def __init__(
        self,
        polygons: Polygon | list[Polygon] | MultiPolygon | list[MultiPolygon],
        buffers: dict[float, float],
        physical_name: str | tuple[str, ...] | None = None,
        mesh_order: float | None = None,
        mesh_bool: bool = True,
        additive: bool = False,
        subdivision: tuple[int, int, int] | None = None,
        point_tolerance: float = 1e-3,
    ):
        # Initialize parent class with point tracking
        super().__init__(point_tolerance=point_tolerance)

        # Parse buffers or prepare extrusion
        if all(buffer == 0 for buffer in buffers.values()):
            self.extrude = True
            self.polygons = polygons
            self.zmin, self.zmax = min(buffers.keys()), max(buffers.keys())
        else:
            self.extrude = False
            self.buffered_polygons: list[
                tuple[float, Polygon]
            ] = self._get_buffered_polygons(polygons, buffers)

        # Store other attributes
        self.mesh_order = mesh_order
        self.additive = additive
        self.dimension = 3
        self.subdivision = subdivision

        # Format physical name
        self.physical_name = format_physical_name(physical_name)
        self.mesh_bool = mesh_bool
        self.additive = additive

    def _create_volumes_directly(self) -> list[int]:
        """Create GMSH volumes directly without using CAD class methods."""
        if self.extrude:
            surfaces = self._create_surfaces_with_holes_at_z(self.polygons, self.zmin)
            surface_dimtags = [(2, surface) for surface in surfaces]
            entities = gmsh.model.occ.extrude(
                surface_dimtags, 0, 0, self.zmax - self.zmin
            )
            volumes = [tag for dim, tag in entities if dim == 3]
        else:
            volumes = [
                self._create_volume_with_holes_directly(entry)
                for entry in self.buffered_polygons
            ]

        # Fuse multiple volumes if needed
        if len(volumes) <= 1:
            return volumes

        fused_dimtags = gmsh.model.occ.fuse(
            [(3, volumes[0])],
            [(3, tag) for tag in volumes[1:]],
            removeObject=True,
            removeTool=True,
        )[0]
        # Clear caches after boolean operations that may invalidate geometry IDs
        self._clear_caches()
        return [tag for dim, tag in fused_dimtags]

    def _get_buffered_polygons(
        self, polygons: list[Polygon], buffers: dict[float, float]
    ) -> list[tuple[float, Polygon]]:
        """Break up polygons on each layer into lists of (z,polygon) tuples according to buffer entries.

        Arguments (implicit):
            polygons: list of (Multi)Polygons to bufferize
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

    def _create_volume_directly(
        self,
        entry: list[tuple[float, Polygon]],
        exterior: bool = True,
        interior_index: int = 0,
    ) -> int:
        """Create volume directly using GMSH calls without CAD class methods."""
        # Draw bottom surface
        bottom_polygon = entry[0][1]
        bottom_z = entry[0][0]
        bottom_polygon_vertices = self.xy_surface_vertices(
            polygon=bottom_polygon,
            polygon_z=bottom_z,
            exterior=exterior,
            interior_index=interior_index,
        )
        gmsh_surfaces = [self._create_surface_from_vertices(bottom_polygon_vertices)]

        # Draw top surface
        top_polygon = entry[-1][1]
        top_z = entry[-1][0]
        top_polygon_vertices = self.xy_surface_vertices(
            polygon=top_polygon,
            polygon_z=top_z,
            exterior=exterior,
            interior_index=interior_index,
        )
        gmsh_surfaces.append(self._create_surface_from_vertices(top_polygon_vertices))

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
                gmsh_surfaces.append(self._create_surface_from_vertices(facet_vertices))

        # Return volume from closed shell
        surface_loop = gmsh.model.occ.addSurfaceLoop(gmsh_surfaces)
        volume = gmsh.model.occ.addVolume([surface_loop])
        gmsh.model.occ.synchronize()
        return volume

    def xy_surface_vertices(
        self,
        polygon: Polygon,
        polygon_z: float,
        exterior: bool,
        interior_index: int,
    ) -> list[tuple[float, float, float]]:
        """Draw xy surface."""
        return (
            [(x, y, polygon_z) for x, y in polygon.exterior.coords]
            if exterior
            else [
                (x, y, polygon_z) for x, y in polygon.interiors[interior_index].coords
            ]
        )

    def _create_volume_with_holes_directly(
        self, entry: list[tuple[float, Polygon]]
    ) -> int:
        """Create volume with holes directly using GMSH calls."""
        exterior = self._create_volume_directly(entry, exterior=True)
        interiors = [
            self._create_volume_directly(
                entry,
                exterior=False,
                interior_index=interior_index,
            )
            for interior_index in range(len(entry[0][1].interiors))
        ]
        if interiors:
            cut_result = gmsh.model.occ.cut(
                [(3, exterior)],
                [(3, interior) for interior in interiors],
                removeObject=True,
                removeTool=True,
            )
            gmsh.model.occ.synchronize()
            # Handle cut result - if cut succeeded, use the result, otherwise keep original
            if cut_result and cut_result[0]:
                exterior = cut_result[0][0][1]  # Parse `outDimTags', `outDimTagsMap'
            # Clear caches after boolean operations that may invalidate geometry IDs
            self._clear_caches()
        return exterior

    def _create_surfaces_with_holes_at_z(self, polygons, z) -> list[int]:
        """Create surfaces with holes at given z level directly using GMSH calls."""
        surfaces = []
        for polygon in polygons.geoms if hasattr(polygons, "geoms") else [polygons]:
            # Create outer surface
            exterior_vertices = [(x, y, z) for x, y in polygon.exterior.coords]
            exterior = self._create_surface_from_vertices(exterior_vertices)

            # Create interior surfaces (holes)
            interior_surfaces = []
            for interior in polygon.interiors:
                interior_vertices = [(x, y, z) for x, y in interior.coords]
                interior_surface = self._create_surface_from_vertices(interior_vertices)
                interior_surfaces.append(interior_surface)

            # Cut holes from exterior surface
            for interior_surface in interior_surfaces:
                cut_result = gmsh.model.occ.cut(
                    [(2, exterior)],
                    [(2, interior_surface)],
                    removeObject=True,
                    removeTool=True,
                )
                gmsh.model.occ.synchronize()
                # Handle cut result - if cut succeeded, use the result, otherwise keep original
                if cut_result and cut_result[0]:
                    exterior = cut_result[0][0][
                        1
                    ]  # Parse `outDimTags', `outDimTagsMap'
                # Clear caches after boolean operations that may invalidate geometry IDs
                self._clear_caches()

            surfaces.append(exterior)
        return surfaces

    def subdivide(self, model, prisms, subdivision):
        """Split the prisms into subprisms according to subdivision."""
        subdivided_prisms = []
        import numpy as np

        global_xmin = np.inf
        global_ymin = np.inf
        global_zmin = np.inf
        global_xmax = -np.inf
        global_ymax = -np.inf
        global_zmax = -np.inf
        for prism in prisms:
            xmin, ymin, zmin, xmax, ymax, zmax = model.occ.getBoundingBox(3, prism)
            if xmin < global_xmin:
                global_xmin = xmin
            if ymin < global_ymin:
                global_ymin = ymin
            if zmin < global_zmin:
                global_zmin = zmin
            if xmax > global_xmax:
                global_xmax = xmax
            if ymax > global_ymax:
                global_ymax = ymax
            if zmax > global_zmax:
                global_zmax = zmax
        dx = (global_xmax - global_xmin) / subdivision[0]
        dy = (global_ymax - global_ymin) / subdivision[1]
        dz = (global_zmax - global_zmin) / subdivision[2]
        prisms_dimtags = {(3, prism) for prism in prisms}
        for x_index in range(subdivision[0]):
            for y_index in range(subdivision[1]):
                for z_index in range(subdivision[2]):
                    tool = model.occ.add_box(
                        global_xmin + x_index * dx,
                        global_ymin + y_index * dy,
                        global_zmin + z_index * dz,
                        dx,
                        dy,
                        dz,
                    )
                    intersection, intersection_map = model.occ.intersect(
                        list(prisms_dimtags),
                        [(3, tool)],
                        removeObject=False,
                        removeTool=True,
                    )
                    prisms_dimtags -= set(intersection_map[0])
                    subdivided_prisms.extend(intersection_map[0] + intersection)
        model.occ.remove(list(prisms_dimtags))
        return subdivided_prisms

    def instanciate(self, cad_model: CAD) -> list[tuple[int, int]]:
        """Create GMSH volumes directly without using CAD class methods."""
        prisms = self._create_volumes_directly()
        gmsh.model.occ.synchronize()
        if self.subdivision is not None:
            return self.subdivide(
                cad_model.model_manager.model, prisms, self.subdivision
            )
        # Return format consistent with original - individual prism dimtags
        return [(3, prism) for prism in prisms]

    def _validate_polygon_buffers(self) -> bool:
        """Check if any buffering operation changes the topology of the polygon."""
        # Get first polygon or multipolygon
        first_geom = self.buffered_polygons[0][0][1]

        # Handle both single polygons and multipolygons
        first_polygons = (
            first_geom.geoms if hasattr(first_geom, "geoms") else [first_geom]
        )

        # Get reference counts from first polygon(s)
        reference_counts = []
        for polygon in first_polygons:
            # Store exterior vertex count and interior vertex counts for this polygon
            polygon_counts = {
                "exterior": len(polygon.exterior.coords),
                "interiors": [len(interior.coords) for interior in polygon.interiors],
            }
            reference_counts.append(polygon_counts)

        # Check each buffered polygon matches reference counts
        for buffered_polygon in self.buffered_polygons[0][1:]:
            geom = buffered_polygon[1]
            polygons = geom.geoms if hasattr(geom, "geoms") else [geom]

            if len(polygons) != len(reference_counts):
                return False

            for polygon, ref_counts in zip(polygons, reference_counts):
                # Check exterior vertices match
                if len(polygon.exterior.coords) != ref_counts["exterior"]:
                    return False

                # Check interior vertices match
                polygon_interior_counts = [
                    len(interior.coords) for interior in polygon.interiors
                ]
                if len(polygon_interior_counts) != len(ref_counts["interiors"]):
                    return False

                for count, ref_count in zip(
                    polygon_interior_counts, ref_counts["interiors"]
                ):
                    if count != ref_count:
                        return False

        return True
