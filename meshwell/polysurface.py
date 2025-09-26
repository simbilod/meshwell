"""PolySurface definitions."""
import gmsh
from shapely.geometry import MultiPolygon, Polygon

from meshwell.cad import CAD
from meshwell.geometry_entity import GeometryEntity


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

    def _create_surface_with_holes(self, polygon: Polygon) -> int:
        """Create surface with holes directly using GMSH calls."""
        # Create exterior surface
        exterior_vertices = [
            self._parse_coords(coords) for coords in polygon.exterior.coords
        ]
        exterior = self._create_surface_from_vertices(exterior_vertices)

        # Create interior surfaces (holes)
        interior_surfaces = []
        for interior in polygon.interiors:
            interior_vertices = [
                self._parse_coords(coords) for coords in interior.coords
            ]
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
