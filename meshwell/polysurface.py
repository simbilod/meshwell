from pydantic import BaseModel, Field, ConfigDict
from shapely.geometry import Polygon, MultiPolygon
from typing import List, Optional, Union, Any, Tuple
from meshwell.resolution import ResolutionSpec


class PolySurface(BaseModel):
    """
    Creates bottom-up GMSH polygonal surfaces formed by list of shapely (multi)polygon.

    Attributes:
        polygons: list of shapely (Multi)Polygon
        model: GMSH model to synchronize
        physical_name: name of the physical this entity wil belong to
        mesh_order: priority of the entity if it overlaps with others (lower numbers override higher numbers)
    """

    polygons: Union[Polygon, List[Polygon], MultiPolygon, List[MultiPolygon]] = Field(
        ...
    )
    model: Any
    physical_name: Optional[str | tuple[str, ...]] = Field(None)
    mesh_order: float | None = None
    mesh_bool: bool = Field(True)
    additive: bool = Field(False)
    dimension: int = Field(2)
    resolutions: List[ResolutionSpec] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        polygons: Union[Polygon, List[Polygon], MultiPolygon, List[MultiPolygon]],
        model: Any,
        physical_name: Optional[str | tuple[str, ...]] = None,
        mesh_order: float | None = None,
        mesh_bool: bool = True,
        additive: bool = False,
        resolutions: List[ResolutionSpec] | None = None,
    ):
        super().__init__(
            polygons=polygons,
            model=model,
            physical_name=physical_name,
            mesh_order=mesh_order,
            mesh_bool=mesh_bool,
            additive=additive,
            resolution=resolutions,
        )

        # Parse (multi)polygons
        self.polygons = list(
            polygons.geoms if hasattr(polygons, "geoms") else [polygons]
        )

        # Track gmsh entities for bottom-up volume definition
        self.model = model

        self.mesh_order = mesh_order
        if isinstance(physical_name, str):
            self.physical_name = [physical_name]
        else:
            self.physical_name = physical_name
        self.mesh_bool = mesh_bool
        self.dimension = 2
        self.resolutions = resolutions
        self.additive = additive

    def _parse_coords(self, coords: Tuple[float, float]) -> Tuple[float, float, float]:
        """Chooses z=0 if the provided coordinates are 2D."""
        return (coords[0], coords[1], 0) if len(coords) == 2 else coords

    def get_gmsh_polygons(self) -> List[int]:
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

    def add_surface_with_holes(self, polygon: Polygon) -> int:
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

    def instanciate(self) -> List[Tuple[int, int]]:
        polysurface = self.get_gmsh_polygons()
        self.model.occ.synchronize()
        return [(2, polysurface)]
