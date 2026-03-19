"""Class definition for entities that create GMHS geometry."""
from __future__ import annotations

from typing import TYPE_CHECKING

import gmsh

if TYPE_CHECKING:
    from OCP.gp import gp_Pnt
    from OCP.TopoDS import TopoDS_Face, TopoDS_Shape, TopoDS_Wire

from meshwell.cad import CAD


class GeometryEntity:
    """Base class for geometry entities that create GMSH geometry directly.

    Provides shared functionality for point deduplication and coordinate parsing
    to ensure consistent geometry creation across PolyLine, PolySurface, and PolyPrism.
    """

    def __init__(self, point_tolerance: float = 1e-3):
        """Initialize geometry entity with point tracking."""
        self.point_tolerance = point_tolerance
        # Track created points to avoid duplicates (strictly instance-local)
        self._points: dict[tuple[float, float, float], int] = {}
        # Track created lines to avoid duplicates and ensure proper edge sharing (strictly instance-local)
        self._lines: dict[tuple[int, int], int] = {}

    def _parse_coords(self, coords: tuple[float, float]) -> tuple[float, float, float]:
        """Convert 2D coordinates to 3D, choosing z=0 if not provided."""
        return (coords[0], coords[1], 0) if len(coords) == 2 else coords

    def _add_point_with_tolerance(self, x: float, y: float, z: float) -> int:
        """Add a point to the model, or reuse a previously-defined point.

        Args:
            x: x-coordinate
            y: y-coordinate
            z: z-coordinate

        Returns:
            GMSH point ID

        """
        key = (x, y, z)

        if key in self._points:
            # Verify that the cached point still exists
            point_tag = self._points[key]
            # Check existence first to avoid GMSH error logging
            if (0, point_tag) in gmsh.model.getEntities(0):
                return point_tag

        # Create new point if not in cache or if cached point is stale/invalid
        tag = gmsh.model.occ.addPoint(x, y, z)
        self._points[key] = tag
        return tag

    def _add_line_with_cache(self, point1_id: int, point2_id: int) -> int:
        """Add a line to the model, or reuse a previously-defined line between the same points.

        Args:
            point1_id: GMSH point ID #1
            point2_id: GMSH point ID #2

        Returns:
            GMSH line ID

        """
        # Initialize local cache if no shared cache is set
        if self._lines is None:
            self._lines = {}

        # Create ordered key (smaller point ID first for consistency)
        key = tuple(sorted([point1_id, point2_id]))

        if key not in self._lines:
            # Create new line
            self._lines[key] = gmsh.model.occ.addLine(point1_id, point2_id)

        return self._lines[key]

    def _create_points_from_vertices(
        self, vertices: list[tuple[float, float, float]]
    ) -> list[int]:
        """Create GMSH points from vertex coordinates, reusing existing points within tolerance.

        Args:
            vertices: List of (x, y, z) coordinates

        Returns:
            List of GMSH point IDs

        """
        points = []
        for x, y, z in vertices:
            point_id = self._add_point_with_tolerance(x, y, z)
            points.append(point_id)
        return points

    def _create_surface_from_vertices(
        self, vertices: list[tuple[float, float, float]]
    ) -> int:
        """Create a GMSH surface from vertex coordinates with point and line reuse."""
        points = self._create_points_from_vertices(vertices)

        # Create lines between consecutive points (closed loop) with caching
        lines = []
        for i in range(len(points) - 1):  # Skip last point as it should equal first
            line_id = self._add_line_with_cache(points[i], points[i + 1])
            lines.append(line_id)

        # Create closed loop and surface
        loop_id = gmsh.model.occ.addCurveLoop(lines)
        return gmsh.model.occ.addPlaneSurface([loop_id])

    def _clear_caches(self):
        """Clear the point and line caches - useful after boolean operations that may invalidate geometry."""
        if self._points is not None:
            self._points.clear()
        if self._lines is not None:
            self._lines.clear()

    def _make_occ_points(
        self, vertices: list[tuple[float, float, float]]
    ) -> list[gp_Pnt]:
        """Convert vertex coordinates to OCP gp_Pnt objects."""
        from OCP.gp import gp_Pnt

        return [gp_Pnt(v[0], v[1], v[2]) for v in vertices]

    def _make_occ_wire_from_vertices(
        self, vertices: list[tuple[float, float, float]]
    ) -> TopoDS_Wire:
        """Create an OCC wire from vertex coordinates."""
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire

        points = self._make_occ_points(vertices)
        wire_builder = BRepBuilderAPI_MakeWire()
        for i in range(len(points) - 1):
            edge = BRepBuilderAPI_MakeEdge(points[i], points[i + 1]).Edge()
            wire_builder.Add(edge)
        return wire_builder.Wire()

    def _make_occ_face_from_vertices(
        self, vertices: list[tuple[float, float, float]]
    ) -> TopoDS_Face:
        """Create an OCC face from vertex coordinates."""
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace

        wire = self._make_occ_wire_from_vertices(vertices)
        return BRepBuilderAPI_MakeFace(wire).Face()

    def instanciate(self, cad_model: CAD | None = None) -> list[tuple[int, int]]:
        """Create GMSH geometry. To be implemented by subclasses.

        Args:
            cad_model: CAD model (kept for interface compatibility)

        Returns:
            List of (dimension, tag) tuples representing created entities

        """
        raise NotImplementedError("Subclasses must implement instanciate method")

    def instanciate_occ(self) -> TopoDS_Shape:
        """Create OCC geometry. To be implemented by subclasses.

        Returns:
            TopoDS_Shape: Created OCC shape

        """
        raise NotImplementedError("Subclasses must implement instanciate_occ method")
