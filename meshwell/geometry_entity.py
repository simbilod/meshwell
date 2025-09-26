"""Class definition for entities that create GMHS geometry."""
from __future__ import annotations

import gmsh

from meshwell.cad import CAD


class GeometryEntity:
    """Base class for geometry entities that create GMSH geometry directly.

    Provides shared functionality for point deduplication and coordinate parsing
    to ensure consistent geometry creation across PolyLine, PolySurface, and PolyPrism.
    """

    def __init__(self, point_tolerance: float = 1e-3):
        """Initialize geometry entity with point tracking."""
        self.point_tolerance = point_tolerance
        # Track created points to avoid duplicates (can be shared across entities)
        self._points: dict[tuple[float, float, float], int] | None = None
        # Track created lines to avoid duplicates and ensure proper edge sharing
        self._lines: dict[tuple[int, int], int] | None = None

    def _parse_coords(self, coords: tuple[float, float]) -> tuple[float, float, float]:
        """Convert 2D coordinates to 3D, choosing z=0 if not provided."""
        return (coords[0], coords[1], 0) if len(coords) == 2 else coords

    def _set_point_cache(self, point_cache: dict[tuple[float, float, float], int]):
        """Set the shared point cache for this entity."""
        self._points = point_cache

    def _set_line_cache(self, line_cache: dict[tuple[int, int], int]):
        """Set the shared line cache for this entity."""
        self._lines = line_cache

    def _add_point_with_tolerance(self, x: float, y: float, z: float) -> int:
        """Add a point to the model, or reuse a previously-defined point within tolerance.

        Args:
            x: x-coordinate
            y: y-coordinate
            z: z-coordinate

        Returns:
            GMSH point ID

        """
        # Initialize local cache if no shared cache is set
        if self._points is None:
            self._points = {}

        # Snap coordinates to tolerance grid to enable point reuse
        snapped_x = round(x / self.point_tolerance) * self.point_tolerance
        snapped_y = round(y / self.point_tolerance) * self.point_tolerance
        snapped_z = round(z / self.point_tolerance) * self.point_tolerance

        key = (snapped_x, snapped_y, snapped_z)

        if key not in self._points:
            # Create new point with snapped coordinates
            self._points[key] = gmsh.model.occ.addPoint(snapped_x, snapped_y, snapped_z)

        return self._points[key]

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

    def instanciate(self, cad_model: CAD | None = None) -> list[tuple[int, int]]:
        """Create GMSH geometry. To be implemented by subclasses.

        Args:
            cad_model: CAD model (kept for interface compatibility)

        Returns:
            List of (dimension, tag) tuples representing created entities

        """
        raise NotImplementedError("Subclasses must implement instanciate method")
