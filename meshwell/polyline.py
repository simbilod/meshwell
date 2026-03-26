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
        identify_arcs: bool = False,
        min_arc_points: int = 4,
        arc_tolerance: float = 1e-3,
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
        self.identify_arcs = identify_arcs
        self.min_arc_points = min_arc_points
        self.arc_tolerance = arc_tolerance

    def _create_wire_from_linestring(self, linestring: LineString) -> int:
        """Create a GMSH wire directly from linestring coordinates."""
        vertices = [self._parse_coords(coords) for coords in linestring.coords]

        if not self.identify_arcs:
            # ORIGINAL BEHAVIOR
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

        # ARC IDENTIFICATION BEHAVIOR
        # Decompose vertices into segments
        segments = self.decompose_vertices(
            vertices,
            identify_arcs=self.identify_arcs,
            min_arc_points=self.min_arc_points,
            arc_tolerance=self.arc_tolerance,
        )

        entities = []
        for seg in segments:
            if seg.is_arc:
                # Create arc
                start_pt = self._add_point_with_tolerance(*seg.points[0])
                center_pt = self._add_point_with_tolerance(*seg.center)
                end_pt = self._add_point_with_tolerance(*seg.points[-1])
                arc_id = gmsh.model.occ.addCircleArc(start_pt, center_pt, end_pt)
                if arc_id != 0:
                    entities.append(arc_id)
            else:
                # Create line
                p1 = self._add_point_with_tolerance(*seg.points[0])
                p2 = self._add_point_with_tolerance(*seg.points[1])
                line_id = self._add_line_with_cache(p1, p2)
                if line_id != 0:
                    entities.append(line_id)

        # Create wire from lines/arcs
        if not entities:
            return 0
        if len(entities) == 1:
            return entities[0]
        return gmsh.model.occ.addWire(entities)

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
            wire = self._make_occ_wire_from_vertices(
                vertices,
                identify_arcs=self.identify_arcs,
                min_arc_points=self.min_arc_points,
                arc_tolerance=self.arc_tolerance,
            )
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

    def to_dict(self) -> dict:
        """Convert entity to dictionary representation.

        Returns:
            Dictionary containing serializable entity data
        """
        import shapely.wkt
        from shapely.geometry import MultiLineString

        if isinstance(self.linestrings, MultiLineString):
            linestrings_wkt = [
                shapely.wkt.dumps(ls, rounding_precision=12)
                for ls in self.linestrings.geoms
            ]
        elif isinstance(self.linestrings, list):
            linestrings_wkt = [
                shapely.wkt.dumps(ls, rounding_precision=12) for ls in self.linestrings
            ]
        else:
            linestrings_wkt = [
                shapely.wkt.dumps(self.linestrings, rounding_precision=12)
            ]

        return {
            "type": "PolyLine",
            "linestrings_wkt": linestrings_wkt,
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
    def from_dict(cls, data: dict) -> "PolyLine":
        """Reconstruct entity from dictionary representation.

        Args:
            data: Dictionary containing entity data

        Returns:
            PolyLine instance
        """
        import shapely.wkt
        from shapely.geometry import MultiLineString

        linestrings = [shapely.wkt.loads(wkt) for wkt in data["linestrings_wkt"]]
        if len(linestrings) > 1:
            linestrings = MultiLineString(linestrings)
        else:
            linestrings = linestrings[0]

        return cls(
            linestrings=linestrings,
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
        """Visualize the decomposition of all linestrings into lines and arcs."""
        for linestring in self.linestrings:
            vertices = [self._parse_coords(coords) for coords in linestring.coords]
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
