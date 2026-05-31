"""Stage 4 — assemble cohort TopoDS_Compound of N sub-solids.

Bottom-up build: unique vertices → unique edges (with arc detection)
→ unique faces (horizontal interior/boundary, lateral) → per-subpiece
TopoDS_Solid → TopoDS_Compound per cohort.

Shared TShapes by CONSTRUCTION (not post-hoc sewing): every face and
edge is built once and referenced by every solid that needs it. This
is what makes cohort internal interfaces conformal without BOP.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeVertex,
)
from OCP.GC import GC_MakeArcOfCircle
from OCP.gp import gp_Pnt
from OCP.TopoDS import TopoDS_Edge, TopoDS_Vertex


@dataclass
class VertexRegistry:
    """Snap-and-dedup vertex store.

    Coordinates are quantized to `point_tolerance` so near-coincident
    vertices map to the same TopoDS_Vertex.
    """

    point_tolerance: float

    def __post_init__(self):
        """Initialise the internal vertex store."""
        self._store: dict[tuple[int, int, int], TopoDS_Vertex] = {}

    def _key(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        s = self.point_tolerance
        return (round(x / s), round(y / s), round(z / s))

    def get_or_create(self, x: float, y: float, z: float) -> TopoDS_Vertex:
        """Return the unique vertex at (x, y, z), creating it if necessary."""
        k = self._key(x, y, z)
        if k not in self._store:
            self._store[k] = BRepBuilderAPI_MakeVertex(gp_Pnt(x, y, z)).Vertex()
        return self._store[k]

    def __len__(self):
        """Return the number of unique vertices in the registry."""
        return len(self._store)


@dataclass
class EdgeRegistry:
    """Unique edge store with arc detection.

    Two flavours:
      - polyline_xy: a 2D polyline at fixed z; runs of vertices on a
        circle (when identify_arcs) build a GC_MakeArcOfCircle edge.
      - vertical: a single edge between two z values at one (x,y).
    """

    vertices: VertexRegistry
    point_tolerance: float

    def __post_init__(self):
        """Initialise the internal edge store."""
        self._store: dict[tuple, TopoDS_Edge] = {}

    def vertical(self, x: float, y: float, z_a: float, z_b: float) -> TopoDS_Edge:
        """Return the unique vertical edge at (x, y) between z_a and z_b."""
        a = self.vertices.get_or_create(x, y, z_a)
        b = self.vertices.get_or_create(x, y, z_b)
        key = ("V", self.vertices._key(x, y, z_a), self.vertices._key(x, y, z_b))
        if key not in self._store:
            self._store[key] = BRepBuilderAPI_MakeEdge(a, b).Edge()
        return self._store[key]

    def line_xy(
        self, x1: float, y1: float, x2: float, y2: float, z: float
    ) -> TopoDS_Edge:
        """Return the unique straight edge between (x1,y1,z) and (x2,y2,z)."""
        a = self.vertices.get_or_create(x1, y1, z)
        b = self.vertices.get_or_create(x2, y2, z)
        k_a = self.vertices._key(x1, y1, z)
        k_b = self.vertices._key(x2, y2, z)
        key = ("L", tuple(sorted([k_a, k_b])))
        if key not in self._store:
            self._store[key] = BRepBuilderAPI_MakeEdge(a, b).Edge()
        return self._store[key]

    def arc_xy(
        self,
        start: tuple[float, float],
        mid: tuple[float, float],
        end: tuple[float, float],
        z: float,
    ) -> TopoDS_Edge:
        """Return a unique arc edge through start, mid, end at height z."""
        sv = self.vertices.get_or_create(*start, z)
        ev = self.vertices.get_or_create(*end, z)
        k_s = self.vertices._key(*start, z)
        k_m = self.vertices._key(*mid, z)
        k_e = self.vertices._key(*end, z)
        key = ("A", k_s, k_m, k_e)
        if key not in self._store:
            # Guard: GC_MakeArcOfCircle cannot handle a full circle (start==end).
            # Caller should split the circle into sub-arcs before calling here.
            p_start = gp_Pnt(start[0], start[1], z)
            p_mid = gp_Pnt(mid[0], mid[1], z)
            p_end = gp_Pnt(end[0], end[1], z)
            builder = GC_MakeArcOfCircle(p_start, p_mid, p_end)
            if not builder.IsDone():
                raise ValueError(
                    f"GC_MakeArcOfCircle failed for start={start} mid={mid} end={end}. "
                    "Full-circle arcs must be split into two half-arcs before calling arc_xy."
                )
            self._store[key] = BRepBuilderAPI_MakeEdge(builder.Value(), sv, ev).Edge()
        return self._store[key]

    def polyline_xy(
        self,
        coords: list[tuple[float, float]],
        z: float,
        identify_arcs: bool,
        min_arc_points: int = 4,
        arc_tolerance: float = 1e-3,
    ) -> list[TopoDS_Edge]:
        """Return the list of edges (lines and/or arcs) covering coords.

        Uses the same arc-detection as GeometryEntity.decompose_vertices
        but inlined to avoid pulling that dep just for the 2D case.
        """
        if not identify_arcs or len(coords) < min_arc_points:
            return [
                self.line_xy(
                    coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1], z
                )
                for i in range(len(coords) - 1)
            ]
        edges: list[TopoDS_Edge] = []
        i, n = 0, len(coords)
        while i < n - 1:
            best = None
            best_j = i + 1
            for j in range(i + min_arc_points, n + 1):
                pts = np.array(coords[i:j])
                cx, cy, r, residual = _fit_circle_2d(pts)
                if residual <= arc_tolerance and r < 1e6:
                    best = (cx, cy, r)
                    best_j = j
                else:
                    break
            if best is not None:
                seg_start = coords[i]
                seg_end = coords[best_j - 1]
                mid_idx = (i + best_j - 1) // 2
                # Full-circle: start and end coincide → split into two half-arcs.
                tol = self.point_tolerance
                if (
                    abs(seg_start[0] - seg_end[0]) < tol
                    and abs(seg_start[1] - seg_end[1]) < tol
                ):
                    q1_idx = (i + mid_idx) // 2
                    q3_idx = (mid_idx + best_j - 1) // 2
                    edges.append(
                        self.arc_xy(seg_start, coords[q1_idx], coords[mid_idx], z)
                    )
                    edges.append(
                        self.arc_xy(coords[mid_idx], coords[q3_idx], seg_end, z)
                    )
                else:
                    edges.append(self.arc_xy(seg_start, coords[mid_idx], seg_end, z))
                i = best_j - 1
            else:
                edges.append(
                    self.line_xy(
                        coords[i][0],
                        coords[i][1],
                        coords[i + 1][0],
                        coords[i + 1][1],
                        z,
                    )
                )
                i += 1
        return edges


def _fit_circle_2d(pts: np.ndarray) -> tuple[float, float, float, float]:
    """Algebraic circle fit. Returns (cx, cy, r, residual)."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = float(np.sqrt(c + cx * cx + cy * cy))
    residual = float(np.std(np.hypot(x - cx, y - cy) - r))
    return float(cx), float(cy), r, residual
