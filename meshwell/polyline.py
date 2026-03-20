"""Gmsh wire definitions."""
from __future__ import annotations

from typing import TYPE_CHECKING

import gmsh
from shapely.geometry import LineString, MultiLineString

from meshwell.cad import CAD
from meshwell.geometry_entity import GeometryEntity

if TYPE_CHECKING:
    from OCP.TopoDS import TopoDS_Shape


class PolyLine(GeometryEntity):
    """Creates bottom-up GMSH wires formed by list of shapely (multi)linestring.

    Attributes:
        linestrings: list of shapely (Multi)LineString
        physical_name: name of the physical this entity will belong to
        mesh_order: priority of the entity if it overlaps with others (lower numbers override higher numbers)

    """

    def __init__(
        self,
        linestrings: LineString
        | list[LineString]
        | MultiLineString
        | list[MultiLineString],
        physical_name: str | tuple[str, ...] | None = None,
        mesh_order: float | None = None,
        mesh_bool: bool = True,
        additive: bool = False,
        point_tolerance: float = 1e-3,
    ):
        # Initialize parent class with point tracking
        super().__init__(point_tolerance=point_tolerance)

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

    def _create_wire_from_linestring(self, linestring: LineString) -> int:
        """Create a GMSH wire directly from linestring coordinates."""
        vertices = [self._parse_coords(coords) for coords in linestring.coords]

        # Create points with deduplication
        points = self._create_points_from_vertices(vertices)

        # Create lines between consecutive points
        lines = []
        for i in range(len(points) - 1):
            line_id = self._add_line_with_cache(points[i], points[i + 1])
            if line_id != 0:
                lines.append(line_id)

        # Create wire from lines
        if not lines:
            return 0
        if len(lines) == 1:
            # For a single line, we can return it as-is since GMSH treats it as a wire
            return lines[0]
        # For multiple lines, create a proper wire
        return gmsh.model.occ.addWire(lines)

    def instanciate(
        self,
        cad_model: CAD | None = None,  # noqa: ARG002
    ) -> list[tuple[int, int]]:
        """Create GMSH wires directly without using CAD class methods."""
        wires = []
        for linestring in self.linestrings:
            wire_id = self._create_wire_from_linestring(linestring)
            wires.append(wire_id)

        gmsh.model.occ.synchronize()
        return [(1, wire) for wire in wires]

    def instanciate_occ(self) -> TopoDS_Shape:
        """Create OCC wires directly using OCP."""
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse

        wires = []
        for linestring in self.linestrings:
            vertices = [self._parse_coords(coords) for coords in linestring.coords]
            wire = self._make_occ_wire_from_vertices(vertices)
            wires.append(wire)

        if not wires:
            return None

        # Fuse multiple wires if needed
        result = wires[0]
        for wire in wires[1:]:
            fuse_api = BRepAlgoAPI_Fuse(result, wire)
            fuse_api.Build()
            result = fuse_api.Shape()

        return result
