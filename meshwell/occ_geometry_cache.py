"""Shared geometry cache for OCC entity instantiation.

When multiple PolyPrism (or other GeometryEntity) objects share coincident
points/edges/arcs, BOPAlgo_Builder only recognizes them as shared if they
carry the same TShape identity. Rebuilding geometry entity-by-entity gives
each a fresh TShape, so fragmentation treats geometrically-identical
boundaries as distinct. This cache fixes that by quantizing coordinates
and returning a single TopoDS_Vertex / TopoDS_Edge for every reference to
the same canonical geometry.
"""
from __future__ import annotations

import threading
from math import floor, log10
from typing import TYPE_CHECKING

from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeVertex,
)
from OCP.GC import GC_MakeArcOfCircle

if TYPE_CHECKING:
    from OCP.gp import gp_Pnt
    from OCP.TopoDS import TopoDS_Edge, TopoDS_Vertex


def _coord_ndigits(tolerance: float) -> int:
    return max(0, int(-floor(log10(tolerance))))


_CoordKey = tuple[float, float, float]


class OCCGeometryCache:
    """Cache TopoDS_Vertex and TopoDS_Edge keyed by quantized geometry."""

    def __init__(self, point_tolerance: float = 1e-3):
        self.point_tolerance = point_tolerance
        self._ndigits = _coord_ndigits(point_tolerance)
        self._lock = threading.Lock()
        self._vertices: dict[_CoordKey, TopoDS_Vertex] = {}
        self._line_edges: dict[tuple[_CoordKey, _CoordKey], TopoDS_Edge] = {}
        self._arc_edges: dict[
            tuple[_CoordKey, _CoordKey, _CoordKey, _CoordKey, float], TopoDS_Edge
        ] = {}

    def vertex_key(self, pnt: gp_Pnt) -> _CoordKey:
        """Return the quantized coordinate tuple used to identify this point."""
        n = self._ndigits
        return (
            round(float(pnt.X()), n),
            round(float(pnt.Y()), n),
            round(float(pnt.Z()), n),
        )

    def get_vertex(self, pnt: gp_Pnt) -> TopoDS_Vertex:
        """Return the cached TopoDS_Vertex for ``pnt``, creating it on first use."""
        key = self.vertex_key(pnt)
        with self._lock:
            v = self._vertices.get(key)
            if v is None:
                v = BRepBuilderAPI_MakeVertex(pnt).Vertex()
                self._vertices[key] = v
            return v

    def get_line_edge(self, p1: gp_Pnt, p2: gp_Pnt) -> TopoDS_Edge:
        """Return the cached straight-line edge between the two points.

        The edge TShape is independent of traversal direction — callers
        receive the same TShape for ``(p1, p2)`` and ``(p2, p1)``.
        """
        k1 = self.vertex_key(p1)
        k2 = self.vertex_key(p2)
        if k1 <= k2:
            key = (k1, k2)
            v1 = self.get_vertex(p1)
            v2 = self.get_vertex(p2)
        else:
            key = (k2, k1)
            v1 = self.get_vertex(p2)
            v2 = self.get_vertex(p1)
        with self._lock:
            edge = self._line_edges.get(key)
            if edge is None:
                edge = BRepBuilderAPI_MakeEdge(v1, v2).Edge()
                self._line_edges[key] = edge
            return edge

    def get_arc_edge(
        self,
        p_start: gp_Pnt,
        p_mid: gp_Pnt,
        p_end: gp_Pnt,
        center: gp_Pnt,
        radius: float,
    ) -> TopoDS_Edge:
        """Return the cached circular arc defined by start/mid/end + center/radius.

        Arcs with identical endpoints but traversing the opposite side of the
        circle are distinguished by their midpoint, so short and long arcs
        between the same endpoints receive different TShapes.
        """
        k_start = self.vertex_key(p_start)
        k_mid = self.vertex_key(p_mid)
        k_end = self.vertex_key(p_end)
        k_center = self.vertex_key(center)
        r_q = round(float(radius), self._ndigits)
        key = (k_start, k_mid, k_end, k_center, r_q)

        v_start = self.get_vertex(p_start)
        v_end = self.get_vertex(p_end)
        with self._lock:
            edge = self._arc_edges.get(key)
            if edge is None:
                arc_geom = GC_MakeArcOfCircle(p_start, p_mid, p_end).Value()
                edge = BRepBuilderAPI_MakeEdge(arc_geom, v_start, v_end).Edge()
                self._arc_edges[key] = edge
            return edge
