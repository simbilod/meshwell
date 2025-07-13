from shapely.geometry import Polygon, MultiPolygon
from typing import List, Optional, Union, Tuple


class PolySurface:
    """
    Creates bottom-up GMSH polygonal surfaces formed by list of shapely (multi)polygon.

    Attributes:
        polygons: list of shapely (Multi)Polygon
        model: GMSH model to synchronize
        physical_name: name of the physical this entity wil belong to
        mesh_order: priority of the entity if it overlaps with others (lower numbers override higher numbers)
    """

    def __init__(
        self,
        polygons: Union[Polygon, List[Polygon], MultiPolygon, List[MultiPolygon]],
        physical_name: Optional[str | tuple[str, ...]] = None,
        mesh_order: float | None = None,
        mesh_bool: bool = True,
        additive: bool = False,
    ):
        # Parse (multi)polygons
        self.polygons = list(
            polygons.geoms if hasattr(polygons, "geoms") else [polygons]
        )

        self.mesh_order = mesh_order
        if isinstance(physical_name, str):
            self.physical_name = (physical_name,)
        else:
            self.physical_name = physical_name
        self.mesh_bool = mesh_bool
        self.dimension = 2
        self.additive = additive

    def _parse_coords(self, coords: Tuple[float, float]) -> Tuple[float, float, float]:
        """Chooses z=0 if the provided coordinates are 2D."""
        return (coords[0], coords[1], 0) if len(coords) == 2 else coords

    def get_gmsh_polygons(self, model) -> List[int]:
        """Returns the fused GMSH surfaces within model from the polygons."""
        surfaces = [
            self.add_surface_with_holes(entry, model) for entry in self.polygons
        ]
        if len(surfaces) <= 1:
            return surfaces
        dimtags = model.occ.fuse(
            [(2, surfaces[0])],
            [(2, tag) for tag in surfaces[1:]],
            removeObject=True,
            removeTool=True,
        )[0]
        model.occ.synchronize()
        return [tag for dim, tag in dimtags]

    def add_surface_with_holes(self, polygon: Polygon, model) -> int:
        """Returns surface, removing intersection with hole surfaces."""
        exterior = model.add_surface(
            [self._parse_coords(coords) for coords in polygon.exterior.coords]
        )
        interior_tags = [
            model.add_surface(
                [self._parse_coords(coords) for coords in interior.coords],
            )
            for interior in polygon.interiors
        ]
        for interior_tag in interior_tags:
            exterior = model.occ.cut(
                [(2, exterior)], [(2, interior_tag)], removeObject=True, removeTool=True
            )
            model.occ.synchronize()
            exterior = exterior[0][0][1]  # Parse `outDimTags', `outDimTagsMap'
        return exterior

    def instanciate(self, cad_model) -> List[Tuple[int, int]]:
        polysurface = self.get_gmsh_polygons(cad_model)
        cad_model.model.occ.synchronize()
        return [(2, polysurface)]
