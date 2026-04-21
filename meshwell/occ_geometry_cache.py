"""Shared geometry cache for OCC entity instantiation.

When multiple PolyPrism (or other GeometryEntity) objects share coincident
points/edges/arcs, BOPAlgo_Builder only recognizes them as shared if they
carry the same TShape identity. Rebuilding geometry entity-by-entity gives
each a fresh TShape, so fragmentation treats geometrically-identical
boundaries as distinct. This cache fixes that by quantizing coordinates
and returning a single TopoDS_Vertex / TopoDS_Edge / TopoDS_Face for every
reference to the same canonical geometry.
"""
from __future__ import annotations

import threading
from math import floor, log10
from typing import TYPE_CHECKING

from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeVertex,
)
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.BRepTools import BRepTools_WireExplorer
from OCP.GC import GC_MakeArcOfCircle
from OCP.TopTools import TopTools_ShapeMapHasher

if TYPE_CHECKING:
    from OCP.gp import gp_Pnt, gp_Vec
    from OCP.TopoDS import TopoDS_Edge, TopoDS_Face, TopoDS_Vertex, TopoDS_Wire

_SHAPE_HASHER = TopTools_ShapeMapHasher()


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
        self._faces: dict[tuple, TopoDS_Face] = {}
        self._side_faces: dict[tuple, TopoDS_Face] = {}

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

    def wire_key(self, wire: TopoDS_Wire) -> tuple:
        """Return a hashable identity for a wire.

        Key is the ordered tuple of ``(edge_TShape_hash, orientation_int)``
        obtained via :class:`BRepTools_WireExplorer`. Two wires composed of
        the same cached edges traversed in the same direction collide.

        Callers that want orientation-agnostic identity should normalize
        (e.g. canonicalize to the lexicographically smaller rotation) before
        passing the result to :meth:`get_face`.
        """
        we = BRepTools_WireExplorer(wire)
        edges = []
        while we.More():
            edge = we.Current()
            edges.append((_SHAPE_HASHER(edge), int(edge.Orientation())))
            we.Next()
        return tuple(edges)

    def get_face(
        self,
        outer_wire: TopoDS_Wire,
        hole_wires: tuple[TopoDS_Wire, ...] = (),
    ) -> TopoDS_Face:
        """Return a cached face built from ``outer_wire`` with optional holes.

        Key is ``(outer_wire_key, sorted_hole_wire_keys)``. Two entities whose
        outer-plus-hole wire composition matches edge-by-edge receive the
        same TopoDS_Face TShape -- the prerequisite for BOPAlgo_Builder to
        fuse them as SameDomain rather than emit near-coincident duplicates.

        The cached face's native orientation follows ``BRepBuilderAPI_MakeFace``
        from the given outer wire. Callers that need the opposite normal
        (e.g. the bottom face of a solid shell) should reverse the face
        locally via ``face.Reversed()`` rather than mutating the cache.
        """
        outer_key = self.wire_key(outer_wire)
        hole_keys = tuple(sorted(self.wire_key(w) for w in hole_wires))
        key = (outer_key, hole_keys)
        with self._lock:
            face = self._faces.get(key)
            if face is None:
                mf = BRepBuilderAPI_MakeFace(outer_wire)
                for hw in hole_wires:
                    mf.Add(hw)
                face = mf.Face()
                self._faces[key] = face
            return face

    def get_extruded_face(self, edge: TopoDS_Edge, vec: gp_Vec) -> TopoDS_Face:
        """Return a cached side face swept from ``edge`` along ``vec``.

        Key is orientation-agnostic: the edge's TShape hash plus the
        quantized sweep vector. Two entities that reference the same cached
        edge and extrude by the same vector receive the same side-face
        TShape even if they traverse the edge in opposite directions
        (as happens at a shared boundary between adjacent prisms -- the
        shared edge appears FORWARD in one wire and REVERSED in the other).

        The face's native normal is determined by sweeping the edge in its
        canonical FORWARD orientation. Callers that need the opposite
        normal for a shell should use ``face.Reversed()`` locally.
        """
        from OCP.TopAbs import TopAbs_FORWARD

        edge_key = _SHAPE_HASHER(edge)
        vec_key = (
            round(float(vec.X()), self._ndigits),
            round(float(vec.Y()), self._ndigits),
            round(float(vec.Z()), self._ndigits),
        )
        key = (edge_key, vec_key)
        with self._lock:
            face = self._side_faces.get(key)
            if face is None:
                canonical_edge = edge.Oriented(TopAbs_FORWARD)
                # Returns a face (1D -> 2D sweep). Typed as TopoDS_Shape by
                # OCP, but guaranteed TopAbs_FACE for edge input.
                face = BRepPrimAPI_MakePrism(canonical_edge, vec).Shape()
                self._side_faces[key] = face
            return face

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
