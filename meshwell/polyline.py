from shapely.geometry import LineString, MultiLineString
from typing import List, Optional, Union, Tuple


class PolyLine:
    """
    Creates bottom-up GMSH wires formed by list of shapely (multi)linestring.

    Attributes:
        linestrings: list of shapely (Multi)LineString
        physical_name: name of the physical this entity will belong to
        mesh_order: priority of the entity if it overlaps with others (lower numbers override higher numbers)
    """

    def __init__(
        self,
        linestrings: Union[
            LineString, List[LineString], MultiLineString, List[MultiLineString]
        ],
        physical_name: Optional[str | tuple[str, ...]] = None,
        mesh_order: float | None = None,
        mesh_bool: bool = True,
        additive: bool = False,
    ):
        # Parse (multi)linestrings
        if isinstance(linestrings, list):
            # Handle list of LineString/MultiLineString objects
            self.linestrings = []
            for item in linestrings:
                if hasattr(item, "geoms"):  # MultiLineString
                    self.linestrings.extend(list(item.geoms))
                else:  # LineString
                    self.linestrings.append(item)
        elif hasattr(linestrings, "geoms"):  # Single MultiLineString
            self.linestrings = list(linestrings.geoms)
        else:  # Single LineString
            self.linestrings = [linestrings]

        self.mesh_order = mesh_order
        if isinstance(physical_name, str):
            self.physical_name = (physical_name,)
        else:
            self.physical_name = physical_name
        self.mesh_bool = mesh_bool
        self.dimension = 1
        self.additive = additive

    def _parse_coords(self, coords: Tuple[float, float]) -> Tuple[float, float, float]:
        """Chooses z=0 if the provided coordinates are 2D."""
        return (coords[0], coords[1], 0) if len(coords) == 2 else coords

    def get_gmsh_wires(self, cad_model) -> List[int]:
        """Returns the GMSH wires within model from the linestrings."""
        edges = [
            self.add_wire(linestring, cad_model) for linestring in self.linestrings
        ]
        cad_model.model_manager.occ.synchronize()
        return edges

    def add_wire(self, linestring: LineString, cad_model) -> int:
        """Returns wire from linestring coordinates."""
        wire = cad_model.wire_from_vertices(
            [self._parse_coords(coords) for coords in linestring.coords]
        )
        return wire

    def instanciate(self, cad_model) -> List[Tuple[int, int]]:
        wires = self.get_gmsh_wires(cad_model)
        cad_model.model_manager.occ.synchronize()
        return [(1, wire) for wire in wires]
