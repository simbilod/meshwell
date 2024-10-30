from typing import List, Dict, Optional, Tuple, Union, Any
from pydantic import BaseModel, Field, ConfigDict
from shapely.geometry import Polygon, MultiPolygon
from meshwell.resolution import ResolutionSpec


class Prism(BaseModel):
    """
    Creates a bottom-up GMSH "prism" formed by a polygon associated with (optional) z-dependent grow/shrink operations.

    Attributes:
        polygons: list of shapely (Multi)Polygon
        buffers: dict of {z: buffer} used to shrink/grow base polygons at specified z-values
        physical_name: name of the physical this entity wil belong to
        mesh_order: priority of the entity if it overlaps with others (lower numbers override higher numbers)
        mesh_bool: if True, entity will be meshed; if not, will not be meshed (useful to tag boundaries)
    """

    polygons: Union[Polygon, List[Polygon], MultiPolygon, List[MultiPolygon]] = Field(
        ...
    )
    buffers: Dict[float, float] = Field(...)
    model: Any
    physical_name: Optional[str | tuple[str, ...]] = Field(None)
    mesh_order: float | None = None
    mesh_bool: bool = Field(True)
    additive: bool = Field(False)
    buffered_polygons: Optional[List[Tuple[float, Polygon]]] = []
    dimension: int = Field(3)
    resolutions: List[ResolutionSpec] | None = None
    extrude: bool = False
    zmin: Optional[float] = 0
    zmax: Optional[float] = 0
    subdivision: tuple[int, int, int] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        polygons: Union[Polygon, List[Polygon], MultiPolygon, List[MultiPolygon]],
        buffers: Dict[float, float],
        model: Any,
        physical_name: Optional[str | tuple[str, ...]] = None,
        mesh_order: float | None = None,
        mesh_bool: bool = True,
        additive: bool = False,
        resolutions: List[ResolutionSpec] | None = None,
        subdivision: tuple[int, int, int] | None = None,
    ):
        super().__init__(
            polygons=polygons,
            buffers=buffers,
            model=model,
            physical_name=physical_name,
            mesh_order=mesh_order,
            mesh_bool=mesh_bool,
            additive=additive,
            resolution=resolutions,
            subdivision=subdivision,
        )

        # Model
        self.model = model

        # Parse buffers or prepare extrusion
        if all(buffer == 0 for buffer in buffers.values()):
            self.extrude = True
            self.polygons = polygons
            self.zmin, self.zmax = min(buffers.keys()), max(buffers.keys())
        else:
            self.extrude = False
            self.buffered_polygons: List[
                Tuple[float, Polygon]
            ] = self._get_buffered_polygons(polygons, buffers)

            # Validate the input
            if not self._validate_polygon_buffers():
                raise ValueError(
                    f"The buffer has modified the polygon vertices! Bad physical: {physical_name}"
                )

        # Mesh order and name
        self.mesh_order = mesh_order

        # Format physical name
        if isinstance(physical_name, str):
            self.physical_name = [physical_name]
        else:
            self.physical_name = physical_name
        self.mesh_bool = mesh_bool
        self.dimension = 3
        self.resolutions = resolutions
        self.additive = additive

    def get_gmsh_volumes(self) -> List[int]:
        """Returns the fused GMSH volumes within model from the polygons and buffers."""
        if self.extrude:
            surfaces = [
                (2, surface)
                for surface in self._add_surfaces_with_holes(self.polygons, self.zmin)
            ]
            entities = self.model.occ.extrude(surfaces, 0, 0, self.zmax - self.zmin)
            volumes = [tag for dim, tag in entities if dim == 3]
        else:
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

    def _get_buffered_polygons(
        self, polygons: List[Polygon], buffers: Dict[float, float]
    ) -> List[Tuple[float, Polygon]]:
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

    def _add_volume(
        self,
        entry: List[Tuple[float, Polygon]],
        exterior: bool = True,
        interior_index: int = 0,
    ) -> int:
        """Create shape from a list of the same buffered polygon and a list of z-values.
        Args:
            polygons: shapely polygons from the GDS
            zs: list of z-values for each polygon
        Returns:
            ID of the added volume
        """
        # Draw bottom surface
        bottom_polygon = entry[0][1]
        bottom_z = entry[0][0]
        bottom_polygon_vertices = self.xy_surface_vertices(
            polygon=bottom_polygon,
            polygon_z=bottom_z,
            exterior=exterior,
            interior_index=interior_index,
        )
        gmsh_surfaces = [self.model.add_surface(bottom_polygon_vertices)]

        # Draw top surface
        top_polygon = entry[-1][1]
        top_z = entry[-1][0]
        top_polygon_vertices = self.xy_surface_vertices(
            polygon=top_polygon,
            polygon_z=top_z,
            exterior=exterior,
            interior_index=interior_index,
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

    def xy_surface_vertices(
        self,
        polygon: Polygon,
        polygon_z: float,
        exterior: bool,
        interior_index: int,
    ) -> List[Tuple[float, float, float]]:
        """"""
        # Draw xy surface
        return (
            [(x, y, polygon_z) for x, y in polygon.exterior.coords]
            if exterior
            else [
                (x, y, polygon_z) for x, y in polygon.interiors[interior_index].coords
            ]
        )

    def _add_volume_with_holes(self, entry: List[Tuple[float, Polygon]]) -> int:
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
            exterior = self.model.occ.cut(
                [(3, exterior)],
                [(3, interior) for interior in interiors],
                removeObject=True,
                removeTool=True,
            )
            self.model.occ.synchronize()
            exterior = exterior[0][0][1]  # Parse `outDimTags', `outDimTagsMap'
        return exterior

    def subdivide(self, prisms, subdivision):
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
            xmin, ymin, zmin, xmax, ymax, zmax = self.model.occ.getBoundingBox(3, prism)
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
        removeObject = False
        for x_index in range(subdivision[0]):
            for y_index in range(subdivision[1]):
                for z_index in range(subdivision[2]):
                    if (
                        x_index == subdivision[0] - 1
                        and y_index == subdivision[1] - 1
                        and z_index == subdivision[2] - 1
                    ):
                        removeObject = True
                    tool = self.model.occ.add_box(
                        xmin + x_index * dx,
                        ymin + y_index * dy,
                        zmin + z_index * dz,
                        dx,
                        dy,
                        dz,
                    )
                    intersection = self.model.occ.intersect(
                        [(3, prism) for prism in prisms],
                        [(3, tool)],
                        removeObject=removeObject,
                        removeTool=True,
                    )[0]
                    subdivided_prisms.extend(intersection)
        return subdivided_prisms

    def instanciate(self) -> List[Tuple[int, int]]:
        """Returns dim tag from entity."""
        prisms = self.get_gmsh_volumes()
        self.model.occ.synchronize()
        if self.subdivision is not None:
            return self.subdivide(prisms, self.subdivision)
        return [(3, prisms)]

    def _validate_polygon_buffers(self) -> bool:
        """Check if any buffering operation changes the topology of the polygon."""
        reference_exterior_vertices = len(
            self.buffered_polygons[0][0][1].exterior.coords
        )
        reference_interiors_vertices = [
            len(interior.coords)
            for interior in self.buffered_polygons[0][0][1].interiors
        ]
        for buffered_polygon in self.buffered_polygons[0][1:]:
            num_points_exterior = len(buffered_polygon[1].exterior.coords)
            num_points_interiors = (
                [
                    len(interior.coords) if interior else 0
                    for interior in buffered_polygon[1].interiors
                ]
                if buffered_polygon[1].interiors
                else [0]
            )
            if num_points_exterior != reference_exterior_vertices:
                return False
            for num_points_interior, reference_interior_vertices in zip(
                num_points_interiors, reference_interiors_vertices
            ):
                if num_points_interior != reference_interior_vertices:
                    return False
        return True

    """
    Extrusion method
    """

    def _add_surfaces_with_holes(self, polygons, z) -> List[int]:
        """Returns surface, removing intersection with hole surfaces."""
        surfaces = []
        for polygon in polygons.geoms if hasattr(polygons, "geoms") else [polygons]:
            # Add outer surface(s)
            exterior = self.model.add_surface(
                self.xy_surface_vertices(
                    polygon, polygon_z=z, exterior=True, interior_index=0
                )
            )
            interiors = [
                self.model.add_surface(
                    self.xy_surface_vertices(
                        polygon,
                        polygon_z=z,
                        exterior=False,
                        interior_index=interior_index,
                    )
                )
                for interior_index in range(len(polygon.interiors))
            ]
            if interiors:
                exterior = self.model.occ.cut(
                    [(2, exterior)],
                    [(2, interior) for interior in interiors],
                    removeObject=True,
                    removeTool=True,
                )
                self.model.occ.synchronize()
                exterior = exterior[0][0][1]  # Parse `outDimTags', `outDimTagsMap'
            surfaces.append(exterior)
        return surfaces


if __name__ == "__main__":
    # Create ring
    from shapely.geometry import Point, box
    from meshwell.model import Model

    inner_radius = 3
    outer_radius = 5

    inner_circle = Point(0, 0).buffer(inner_radius)
    outer_circle = Point(0, 0).buffer(outer_radius)
    opening = box(minx=-2, miny=-6, maxx=2, maxy=-2)

    ring = outer_circle.difference(inner_circle)  # .difference(opening)

    from shapely.plotting import plot_polygon
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot()
    # plot_polygon(outer_circle, ax=ax, add_points=False, color="blue")
    # plot_polygon(inner_circle, ax=ax, add_points=False, color="green")
    plot_polygon(ring, ax=ax, add_points=False, color="red")
    plt.show()

    # Test the Prism class
    polygons = ring  # Add your polygons here
    buffers = {0: 0, -1: 0, -1.001: 1, -5: 0}  # Add your buffers here

    model = Model()

    poly3D = Prism(polygons=polygons, buffers=buffers, model=model)

    model.mesh(entities_dict={"poly3D": poly3D}, filename="mesh3D.msh")
