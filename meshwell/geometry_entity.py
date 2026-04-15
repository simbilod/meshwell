"""Shared utilities for geometries."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import gmsh
import numpy as np

if TYPE_CHECKING:
    from OCP.gp import gp_Pnt
    from OCP.TopoDS import TopoDS_Face, TopoDS_Shape, TopoDS_Wire

from meshwell.cad import CAD


@dataclass
class DecompositionSegment:
    """Dataclass to represent a geometry segment (line or arc)."""

    points: list[tuple[float, float, float]]
    is_arc: bool
    center: tuple[float, float, float] | None = None
    radius: float | None = None


def fit_circle_2d(points: np.ndarray) -> tuple[tuple[float, float], float, float]:
    """Fit a circle to 2D points using algebraic distance (least squares).

    Returns:
        center: (xc, yc)
        radius: R
        residual: Mean squared error
    """
    x = points[:, 0]
    y = points[:, 1]

    # Linear least squares for (xc, yc, R^2 - xc^2 - yc^2)
    # x^2 + y^2 - 2*x*xc - 2*y*yc + xc^2 + yc^2 = R^2
    # x^2 + y^2 = 2*x*xc + 2*y*yc + (R^2 - xc^2 - yc^2)
    # Let A = [2*x, 2*y, ones], b = x^2 + y^2
    a = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x**2 + y**2
    try:
        c, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return (0, 0), 0, np.inf

    xc, yc, k = c
    radius = np.sqrt(k + xc**2 + yc**2)

    # Calculate residual (RMSE)
    distances = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    residual = np.sqrt(np.mean((distances - radius) ** 2))

    return (xc, yc), radius, residual


class GeometryEntity:
    """Base class for geometry entities that create GMSH geometry directly.

    Provides shared functionality for point deduplication and coordinate parsing
    to ensure consistent geometry creation across PolyLine, PolySurface, and PolyPrism.
    """

    def __init__(
        self,
        point_tolerance: float = 1e-3,
        translation: tuple[float, float, float] | None = None,
        rotation_axis: tuple[float, float, float] | None = None,
        rotation_point: tuple[float, float, float] | None = None,
        rotation_angle: float = 0.0,
    ):
        """Initialize geometry entity with point tracking and transformation parameters."""
        self.point_tolerance = point_tolerance
        self.translation = translation
        self.rotation_axis = rotation_axis or (0, 0, 1)
        self.rotation_point = rotation_point
        self.rotation_angle = rotation_angle
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
        x, y, z = float(x), float(y), float(z)
        ndigits = max(0, int(-np.floor(np.log10(self.point_tolerance))))
        key = (round(x, ndigits), round(y, ndigits), round(z, ndigits))

        if key in self._points:
            # Verify that the cached point still exists
            point_tag = self._points[key]
            # Check existence first to avoid GMSH error logging
            if (0, point_tag) in gmsh.model.occ.getEntities(0):
                return point_tag

        # If not in cache or doesn't exist anymore, create it
        point_tag = gmsh.model.occ.addPoint(x, y, z)
        self._points[key] = point_tag
        return point_tag

    def _add_line_with_cache(self, point1_id: int, point2_id: int) -> int:
        """Add a line to the model, or reuse a previously-defined line between the same points.

        Args:
            point1_id: GMSH point ID #1
            point2_id: GMSH point ID #2

        Returns:
            Signed GMSH line ID (negative if reversed), or 0 if points are identical

        """
        if point1_id == point2_id:
            return 0

        # Initialize local cache if no shared cache is set
        if self._lines is None:
            self._lines = {}

        # Create ordered key (smaller point ID first for consistency)
        key = tuple(sorted([point1_id, point2_id]))

        if key not in self._lines:
            # Create new line and store its original orientation
            line_tag = gmsh.model.occ.addLine(point1_id, point2_id)
            self._lines[key] = (line_tag, point1_id, point2_id)

        line_tag, orig_p1, _orig_p2 = self._lines[key]

        # Return signed tag based on requested orientation
        if point1_id == orig_p1:
            return line_tag
        return -line_tag

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
        self,
        vertices: list[tuple[float, float, float]],
        identify_arcs: bool = False,
        min_arc_points: int = 4,
        arc_tolerance: float = 1e-3,
    ) -> int:
        """Create a GMSH surface from vertex coordinates with optional arc identification."""
        if not identify_arcs:
            # ORIGINAL BEHAVIOR
            points = self._create_points_from_vertices(vertices)

            # Create lines between consecutive points (closed loop) with caching
            lines = []
            for i in range(len(points) - 1):  # Skip last point as it should equal first
                line_id = self._add_line_with_cache(points[i], points[i + 1])
                if line_id != 0:
                    lines.append(line_id)

            if not lines:
                return 0

            # Create closed loop and surface
            loop_id = gmsh.model.occ.addCurveLoop(lines)
            return gmsh.model.occ.addPlaneSurface([loop_id])

        # ARC IDENTIFICATION BEHAVIOR
        segments = self.decompose_vertices(
            vertices,
            identify_arcs=identify_arcs,
            min_arc_points=min_arc_points,
            arc_tolerance=arc_tolerance,
        )

        entities = []
        for seg in segments:
            if seg.is_arc:
                start_pt = self._add_point_with_tolerance(*seg.points[0])
                mid_idx = len(seg.points) // 2
                mid_pt = self._add_point_with_tolerance(*seg.points[mid_idx])
                end_pt = self._add_point_with_tolerance(*seg.points[-1])

                if start_pt == end_pt:
                    # Full circle: split into two 180-degree arcs
                    # For a full circle, we need two points on the circle to split it
                    quarter_idx = len(seg.points) // 4
                    three_quarter_idx = (len(seg.points) * 3) // 4
                    p1 = self._add_point_with_tolerance(*seg.points[quarter_idx])
                    p3 = self._add_point_with_tolerance(*seg.points[three_quarter_idx])
                    try:
                        arc1 = gmsh.model.occ.addCircleArc(
                            start_pt, p1, mid_pt, center=False
                        )
                        arc2 = gmsh.model.occ.addCircleArc(
                            mid_pt, p3, end_pt, center=False
                        )
                        entities.append(arc1)
                        entities.append(arc2)
                    except Exception as e:
                        raise RuntimeError(f"Failed to split full circle: {e}") from e
                    continue
                try:
                    arc_id = gmsh.model.occ.addCircleArc(
                        start_pt, mid_pt, end_pt, center=False
                    )
                except Exception as e:
                    raise RuntimeError(f"addCircleArc failed: {e}") from e
                if arc_id != 0:
                    entities.append(arc_id)
            else:
                p1 = self._add_point_with_tolerance(*seg.points[0])
                p2 = self._add_point_with_tolerance(*seg.points[1])
                line_id = self._add_line_with_cache(p1, p2)
                if line_id != 0:
                    entities.append(line_id)

        if not entities:
            return 0

        # Create closed loop and surface
        loop_id = gmsh.model.occ.addCurveLoop(entities)
        return gmsh.model.occ.addPlaneSurface([loop_id])

    def _clear_caches(self):
        """Clear the point and line caches - useful after boolean operations that may invalidate geometry."""
        if self._points is not None:
            self._points.clear()
        if self._lines is not None:
            self._lines.clear()

    def decompose_vertices(
        self,
        vertices: list[tuple[float, float, float]],
        identify_arcs: bool = False,
        min_arc_points: int = 4,
        arc_tolerance: float = 1e-3,
    ) -> list[DecompositionSegment]:
        """Decompose a sequence of vertices into line segments and circular arcs.

        Args:
            vertices: List of (x, y, z) coordinates
            identify_arcs: Whether to attempt arc identification
            min_arc_points: Minimum number of points to form an arc
            arc_tolerance: Tolerance for circle fitting

        Returns:
            List of DecompositionSegment objects
        """
        if not vertices or len(vertices) < 2:
            return []

        if not identify_arcs or len(vertices) < min_arc_points:
            # Fallback to simple line segments
            return [
                DecompositionSegment(
                    points=[vertices[i], vertices[i + 1]], is_arc=False
                )
                for i in range(len(vertices) - 1)
            ]

        segments = []
        i = 0
        n = len(vertices)
        ndigits = max(0, int(-np.floor(np.log10(self.point_tolerance))))

        while i < n - 1:
            # Try to find an arc starting at i
            best_arc = None
            if i + min_arc_points <= n:
                for j in range(i + min_arc_points, n + 1):
                    pts = np.array(vertices[i:j])
                    # Ensure points are not collinear and can be fit to a circle
                    # For simplicity, we assume the arc is in the XY plane
                    center, radius, residual = fit_circle_2d(pts[:, :2])

                    if residual <= arc_tolerance and radius < 1e6:
                        # Ensure it's not a polygon with sharp corners (like a rectangle)
                        valid_arc = True
                        for k in range(1, len(pts) - 1):
                            v1 = pts[k][:2] - pts[k - 1][:2]
                            v2 = pts[k + 1][:2] - pts[k][:2]
                            n1 = np.linalg.norm(v1)
                            n2 = np.linalg.norm(v2)
                            if n1 > 1e-6 and n2 > 1e-6:
                                cos_angle = np.dot(v1, v2) / (n1 * n2)
                                if cos_angle < 0.5:  # Turn angle > 60 degrees
                                    valid_arc = False
                                    break

                        if valid_arc:
                            # Round center and radius to consistent decimal places based on tolerance
                            cx = round(center[0], ndigits)
                            cy = round(center[1], ndigits)
                            r = round(radius, ndigits)

                            # Update best arc candidate for this start point
                            best_arc = DecompositionSegment(
                                points=vertices[i:j],
                                is_arc=True,
                                center=(cx, cy, vertices[i][2]),
                                radius=r,
                            )
                    else:
                        # Stop expanding if fit fails
                        break

            if best_arc:
                segments.append(best_arc)
                # Increment i by the number of points in the arc minus 1
                i += len(best_arc.points) - 1
            else:
                # Add a line segment
                segments.append(
                    DecompositionSegment(
                        points=[vertices[i], vertices[i + 1]], is_arc=False
                    )
                )
                i += 1

        return segments

    def _make_occ_points(
        self, vertices: list[tuple[float, float, float]]
    ) -> list[gp_Pnt]:
        """Convert vertex coordinates to OCP gp_Pnt objects."""
        from OCP.gp import gp_Pnt

        return [gp_Pnt(v[0], v[1], v[2]) for v in vertices]

    def _make_occ_wire_from_vertices(
        self,
        vertices: list[tuple[float, float, float]],
        identify_arcs: bool = False,
        min_arc_points: int = 4,
        arc_tolerance: float = 1e-3,
    ) -> TopoDS_Wire:
        """Create an OCC wire from vertex coordinates with optional arc identification."""
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire

        if not identify_arcs:
            # ORIGINAL BEHAVIOR
            points = self._make_occ_points(vertices)
            wire_builder = BRepBuilderAPI_MakeWire()
            for i in range(len(points) - 1):
                edge = BRepBuilderAPI_MakeEdge(points[i], points[i + 1]).Edge()
                wire_builder.Add(edge)
            return wire_builder.Wire()

        # ARC IDENTIFICATION BEHAVIOR
        from OCP.GC import GC_MakeArcOfCircle
        from OCP.gp import gp_Pnt

        segments = self.decompose_vertices(
            vertices,
            identify_arcs=identify_arcs,
            min_arc_points=min_arc_points,
            arc_tolerance=arc_tolerance,
        )

        wire_builder = BRepBuilderAPI_MakeWire()
        ndigits = max(0, int(-np.floor(np.log10(self.point_tolerance))))

        for seg in segments:
            if seg.is_arc:
                # Round points to consistent decimal places based on tolerance
                # to ensure exact geometrical match between adjacent polygons
                p0 = [round(c, ndigits) for c in seg.points[0]]
                p_start = gp_Pnt(*p0)

                mid_idx = len(seg.points) // 2
                pmid = [round(c, ndigits) for c in seg.points[mid_idx]]
                p_mid = gp_Pnt(*pmid)

                pend = [round(c, ndigits) for c in seg.points[-1]]
                p_end = gp_Pnt(*pend)

                if seg.points[0] == seg.points[-1]:
                    # Full circle: split into two 180-degree arcs
                    quarter_idx = len(seg.points) // 4
                    three_quarter_idx = (len(seg.points) * 3) // 4
                    pq1 = [round(c, ndigits) for c in seg.points[quarter_idx]]
                    p1 = gp_Pnt(*pq1)

                    pq3 = [round(c, ndigits) for c in seg.points[three_quarter_idx]]
                    p3 = gp_Pnt(*pq3)

                    arc_geom1 = GC_MakeArcOfCircle(p_start, p1, p_mid).Value()
                    edge1 = BRepBuilderAPI_MakeEdge(arc_geom1).Edge()
                    wire_builder.Add(edge1)

                    arc_geom2 = GC_MakeArcOfCircle(p_mid, p3, p_end).Value()
                    edge = BRepBuilderAPI_MakeEdge(arc_geom2).Edge()
                else:
                    from OCP.GeomAPI import GeomAPI_ProjectPointOnCurve
                    from OCP.gp import gp_Ax2, gp_Circ, gp_Dir

                    try:
                        center = gp_Pnt(seg.center[0], seg.center[1], seg.center[2])
                        axis = gp_Ax2(center, gp_Dir(0, 0, 1))
                        circle = gp_Circ(axis, seg.radius)

                        # Try to make arc with positive sense
                        arc_geom_pos = GC_MakeArcOfCircle(
                            circle, p_start, p_end, True
                        ).Value()
                        # Check if p_mid is on the positive arc
                        proj = GeomAPI_ProjectPointOnCurve(p_mid, arc_geom_pos)
                        if proj.NbPoints() > 0 and proj.LowerDistance() < 1e-2:
                            arc_geom = arc_geom_pos
                        else:
                            arc_geom = GC_MakeArcOfCircle(
                                circle, p_start, p_end, False
                            ).Value()
                        edge = BRepBuilderAPI_MakeEdge(arc_geom).Edge()
                    except Exception:
                        arc_geom = GC_MakeArcOfCircle(p_start, p_mid, p_end).Value()
                        edge = BRepBuilderAPI_MakeEdge(arc_geom).Edge()
            else:
                p1_rounded = [round(c, ndigits) for c in seg.points[0]]
                p1 = gp_Pnt(*p1_rounded)

                p2_rounded = [round(c, ndigits) for c in seg.points[1]]
                p2 = gp_Pnt(*p2_rounded)
                edge = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
            wire_builder.Add(edge)
        return wire_builder.Wire()

    def _make_occ_face_from_vertices(
        self,
        vertices: list[tuple[float, float, float]],
        identify_arcs: bool = False,
        min_arc_points: int = 4,
        arc_tolerance: float = 1e-3,
    ) -> TopoDS_Face:
        """Create an OCC face from vertex coordinates with optional arc identification."""
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace

        wire = self._make_occ_wire_from_vertices(
            vertices,
            identify_arcs=identify_arcs,
            min_arc_points=min_arc_points,
            arc_tolerance=arc_tolerance,
        )
        return BRepBuilderAPI_MakeFace(wire).Face()

    def plot_decomposition(
        self,
        vertices: list[tuple[float, float, float]],
        ax=None,
        line_color: str = "blue",
        arc_color: str = "red",
        show_centers: bool = True,
        identify_arcs: bool = False,
        min_arc_points: int = 4,
        arc_tolerance: float = 1e-3,
        **kwargs,
    ):
        """Visualize the decomposition of vertices into lines and arcs."""
        from meshwell.visualization import plot_decomposition

        segments = self.decompose_vertices(
            vertices,
            identify_arcs=identify_arcs,
            min_arc_points=min_arc_points,
            arc_tolerance=arc_tolerance,
        )
        return plot_decomposition(
            segments,
            ax=ax,
            line_color=line_color,
            arc_color=arc_color,
            show_centers=show_centers,
            **kwargs,
        )

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

    def _get_rotation_point(self, geometries) -> tuple[float, float, float]:
        """Calculate centroid if rotation_point is None."""
        if self.rotation_point is not None:
            return self.rotation_point

        from shapely.ops import unary_union

        combined = unary_union(geometries)
        centroid = combined.centroid
        return (centroid.x, centroid.y, 0.0)  # Assume 2D plane for pivot

    def _apply_transformation_gmsh(
        self, dimtags: list[tuple[int, int]], rotation_point: tuple[float, float, float]
    ) -> list[tuple[int, int]]:
        """Apply rotation and translation to GMSH entities."""
        if not dimtags:
            return dimtags
        import numpy as np

        if self.rotation_angle != 0:
            gmsh.model.occ.rotate(
                dimtags,
                *rotation_point,
                *self.rotation_axis,
                np.radians(self.rotation_angle),
            )
        if self.translation:
            gmsh.model.occ.translate(dimtags, *self.translation)
        return dimtags

    def _apply_transformation_occ(
        self, shape: TopoDS_Shape, rotation_point: tuple[float, float, float]
    ) -> TopoDS_Shape:
        """Apply rotation and translation to OCC shape."""
        if shape is None:
            return None
        import numpy as np
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec

        trsf = gp_Trsf()
        if self.rotation_angle != 0:
            axis = gp_Ax1(gp_Pnt(*rotation_point), gp_Dir(*self.rotation_axis))
            trsf.SetRotation(axis, np.radians(self.rotation_angle))

        if self.translation:
            trsf_trans = gp_Trsf()
            trsf_trans.SetTranslation(gp_Vec(*self.translation))
            trsf = trsf_trans * trsf  # Rotate then Translate

        transform_api = BRepBuilderAPI_Transform(shape, trsf)
        return transform_api.Shape()
