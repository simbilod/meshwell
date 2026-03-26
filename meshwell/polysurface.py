"""PolySurface definitions."""
from __future__ import annotations

from typing import TYPE_CHECKING

import gmsh
from shapely.geometry import MultiPolygon, Polygon

from meshwell.cad import CAD
from meshwell.geometry_entity import GeometryEntity

if TYPE_CHECKING:
    from OCP.TopoDS import TopoDS_Shape


class PolySurface(GeometryEntity):
    """Creates bottom-up GMSH polygonal surfaces formed by list of shapely (multi)polygon.

    Attributes:
        polygons: list of shapely (Multi)Polygon
        model: GMSH model to synchronize
        physical_name: name of the physical this entity will belong to
        mesh_order: priority of the entity if it overlaps with others (lower numbers override higher numbers)

    """

    def __init__(
        self,
        polygons: Polygon | list[Polygon] | MultiPolygon | list[MultiPolygon],
        physical_name: str | tuple[str, ...] | None = None,
        mesh_order: float | None = None,
        mesh_bool: bool = True,
        additive: bool = False,
        point_tolerance: float = 1e-3,
        identify_arcs: bool = False,
        min_arc_points: int = 4,
        arc_tolerance: float = 1e-3,
    ):
        # Initialize parent class with point tracking
        super().__init__(point_tolerance=point_tolerance)

        # Parse (multi)polygons
        if isinstance(polygons, (Polygon, MultiPolygon)):
            self.polygons = list(
                polygons.geoms if hasattr(polygons, "geoms") else [polygons]
            )
        else:
            self.polygons = []
            for entry in polygons:
                self.polygons.extend(
                    entry.geoms if hasattr(entry, "geoms") else [entry]
                )

        self.mesh_order = mesh_order
        if isinstance(physical_name, str):
            self.physical_name = (physical_name,)
        else:
            self.physical_name = physical_name
        self.mesh_bool = mesh_bool
        self.dimension = 2
        self.additive = additive
        self.identify_arcs = identify_arcs
        self.min_arc_points = min_arc_points
        self.arc_tolerance = arc_tolerance

    def _create_surface_with_holes(self, polygon: Polygon) -> int:
        """Create surface with holes directly using GMSH calls."""
        # Create exterior surface
        exterior_vertices = [
            self._parse_coords(coords) for coords in polygon.exterior.coords
        ]
        exterior = self._create_surface_from_vertices(
            exterior_vertices,
            identify_arcs=self.identify_arcs,
            min_arc_points=self.min_arc_points,
            arc_tolerance=self.arc_tolerance,
        )

        # Create interior surfaces (holes)
        interior_surfaces = []
        for interior in polygon.interiors:
            interior_vertices = [
                self._parse_coords(coords) for coords in interior.coords
            ]
            interior_surface = self._create_surface_from_vertices(
                interior_vertices,
                identify_arcs=self.identify_arcs,
                min_arc_points=self.min_arc_points,
                arc_tolerance=self.arc_tolerance,
            )
            if interior_surface != 0:
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
            exterior = cut_result[0][0][1]  # Parse `outDimTags', `outDimTagsMap'
            # Clear caches after boolean operations that may invalidate geometry IDs
            self._clear_caches()

        return exterior

    def instanciate(
        self,
        cad_model: CAD | None = None,  # noqa: ARG002
    ) -> list[tuple[int, int]]:
        """Create GMSH surfaces directly without using CAD class methods."""
        surfaces = []
        for polygon in self.polygons:
            surface_id = self._create_surface_with_holes(polygon)
            if surface_id != 0:
                surfaces.append(surface_id)

        # Fuse multiple surfaces if needed
        if len(surfaces) <= 1:
            gmsh.model.occ.synchronize()
            return [(2, surfaces[0])] if surfaces else []

        # Fuse all surfaces together
        dimtags = gmsh.model.occ.fuse(
            [(2, surfaces[0])],
            [(2, tag) for tag in surfaces[1:]],
            removeObject=True,
            removeTool=True,
        )[0]
        gmsh.model.occ.synchronize()
        # Clear caches after boolean operations that may invalidate geometry IDs
        self._clear_caches()
        return dimtags

    def instanciate_occ(self) -> TopoDS_Shape:
        """Create OCC surfaces directly using OCP."""
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse

        surfaces = []
        for polygon in self.polygons:
            # Create exterior face
            exterior_vertices = [
                self._parse_coords(coords) for coords in polygon.exterior.coords
            ]
            exterior_face = self._make_occ_face_from_vertices(
                exterior_vertices,
                identify_arcs=self.identify_arcs,
                min_arc_points=self.min_arc_points,
                arc_tolerance=self.arc_tolerance,
            )

            # Create interior surfaces (holes) and cut them
            for interior in polygon.interiors:
                interior_vertices = [
                    self._parse_coords(coords) for coords in interior.coords
                ]
                interior_face = self._make_occ_face_from_vertices(
                    interior_vertices,
                    identify_arcs=self.identify_arcs,
                    min_arc_points=self.min_arc_points,
                    arc_tolerance=self.arc_tolerance,
                )

                cut_api = BRepAlgoAPI_Cut(exterior_face, interior_face)
                cut_api.Build()
                exterior_face = cut_api.Shape()

            surfaces.append(exterior_face)

        if not surfaces:
            return None

        # Fuse multiple surfaces if needed
        result = surfaces[0]
        for surface in surfaces[1:]:
            fuse_api = BRepAlgoAPI_Fuse(result, surface)
            fuse_api.Build()
            result = fuse_api.Shape()

        return result

    def to_dict(self) -> dict:
        """Convert entity to dictionary representation.

        Returns:
            Dictionary containing serializable entity data
        """
        import shapely.wkt
        from shapely.geometry import MultiPolygon

        if isinstance(self.polygons, MultiPolygon):
            polygons_wkt = [shapely.wkt.dumps(p, rounding_precision=12) for p in self.polygons.geoms]
        elif isinstance(self.polygons, list):
            polygons_wkt = [shapely.wkt.dumps(p, rounding_precision=12) for p in self.polygons]
        else:
            polygons_wkt = [shapely.wkt.dumps(self.polygons, rounding_precision=12)]

        return {
            "type": "PolySurface",
            "polygons_wkt": polygons_wkt,
            "physical_name": self.physical_name,
            "mesh_order": self.mesh_order,
            "mesh_bool": self.mesh_bool,
            "additive": self.additive,
            "point_tolerance": self.point_tolerance,
            "identify_arcs": self.identify_arcs,
            "min_arc_points": self.min_arc_points,
            "arc_tolerance": self.arc_tolerance,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PolySurface":
        """Reconstruct entity from dictionary representation.

        Args:
            data: Dictionary containing entity data

        Returns:
            PolySurface instance
        """
        import shapely.wkt
        from shapely.geometry import MultiPolygon

        polygons = [shapely.wkt.loads(wkt) for wkt in data["polygons_wkt"]]
        if len(polygons) > 1:
            polygons = MultiPolygon(polygons)
        else:
            polygons = polygons[0]

        return cls(
            polygons=polygons,
            physical_name=data["physical_name"],
            mesh_order=data["mesh_order"],
            mesh_bool=data["mesh_bool"],
            additive=data["additive"],
            point_tolerance=data["point_tolerance"],
            identify_arcs=data["identify_arcs"],
            min_arc_points=data["min_arc_points"],
            arc_tolerance=data["arc_tolerance"],
        )

    def plot_decomposition(
        self,
        ax=None,
        line_color: str = "blue",
        arc_color: str = "red",
        show_centers: bool = True,
        **kwargs,
    ):
        """Visualize the decomposition of all polygons into lines and arcs."""
        for polygon in self.polygons:
            # Exterior
            vertices = [
                self._parse_coords(coords) for coords in polygon.exterior.coords
            ]
            ax = super().plot_decomposition(
                vertices,
                ax=ax,
                line_color=line_color,
                arc_color=arc_color,
                show_centers=show_centers,
                identify_arcs=self.identify_arcs,
                min_arc_points=self.min_arc_points,
                arc_tolerance=self.arc_tolerance,
                **kwargs,
            )
            # Interiors
            for interior in polygon.interiors:
                vertices = [self._parse_coords(coords) for coords in interior.coords]
                ax = super().plot_decomposition(
                    vertices,
                    ax=ax,
                    line_color=line_color,
                    arc_color=arc_color,
                    show_centers=show_centers,
                    identify_arcs=self.identify_arcs,
                    min_arc_points=self.min_arc_points,
                    arc_tolerance=self.arc_tolerance,
                    **kwargs,
                )
        return ax
