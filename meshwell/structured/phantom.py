"""Phase-2: CAD-stage phantom shape construction + BOP-history-based PhantomMap.

The two public entry points (added in later tasks) are:

- :func:`build_phantom_shapes` — turn a ``StructuredPlan`` into
  per-piece OCP sub-prisms, recording input OCC tags into
  ``PhantomBuildResult``.
- :func:`extract_phantom_map` — given a post-Perform
  ``BOPAlgo_Builder`` (or any builder exposing the Modified() /
  Generated() / IsDeleted() history API), walk the recorded input
  tags to produce the ``PhantomMap``.

Phase 2 does not integrate with ``cad_occ`` (that's Phase 3). All
tests here use OCP directly with handcrafted scenes.
"""
from __future__ import annotations

from typing import Any

from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from meshwell.structured.logging import phase_timed
from meshwell.structured.spec import (
    EdgeKey,
    FaceKey,
    LateralKey,
    PhantomBuildResult,
    PhantomMap,
    PhantomShape,
    StructuredPlan,
    VertexKey,
)

# Default point_tolerance matches GeometryEntity.__init__ default (1e-3).
_POINT_TOLERANCE = 1e-3


def _make_arc_wire_from_coords(
    coords: list[tuple[float, float]],
    z: float,
    identify_arcs: bool,
    min_arc_points: int,
    arc_tolerance: float,
    point_tolerance: float = _POINT_TOLERANCE,
) -> Any:
    """Build an OCC wire at height z from 2-D polygon coords.

    When ``identify_arcs=True`` this calls the same arc-decomposition
    logic used by ``GeometryEntity._make_occ_wire_from_vertices`` so
    that the phantom sub-prism shares TShapes with the original PolyPrism
    shape after BOP.

    When ``identify_arcs=False`` straight-line edges are produced
    (identical behaviour to the original ``_make_face_from_polygon``
    helper).

    ``coords`` must NOT have the closing duplicate vertex (first == last).
    The function appends the closing vertex internally so the result is a
    closed wire.
    """
    import numpy as np
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
    from OCP.gp import gp_Pnt

    from meshwell.geometry_entity import (
        DecompositionSegment,
        _find_canonical_seam,
        _rotate_closed,
        _strip_consecutive_duplicates,
        fit_circle_2d,
    )

    # Build the closed 3-D vertex list.
    vertices_3d: list[tuple[float, float, float]] = [(cx, cy, z) for (cx, cy) in coords]
    # Close the ring for the arc decomposer.
    if vertices_3d and vertices_3d[0] != vertices_3d[-1]:
        vertices_3d.append(vertices_3d[0])

    vertices = _strip_consecutive_duplicates(vertices_3d, point_tolerance)

    if not identify_arcs:
        # Fast straight-line path (same as original _make_face_from_polygon).
        wire_builder = BRepBuilderAPI_MakeWire()
        n = len(vertices) - 1  # last == first for closed ring
        for i in range(n):
            p1 = gp_Pnt(vertices[i][0], vertices[i][1], z)
            p2 = gp_Pnt(vertices[i + 1][0], vertices[i + 1][1], z)
            wire_builder.Add(BRepBuilderAPI_MakeEdge(p1, p2).Edge())
        return wire_builder.Wire()

    # Arc-aware path — mirrors GeometryEntity._make_occ_wire_from_vertices.
    ndigits = max(0, int(-np.floor(np.log10(point_tolerance))))

    # Quantize and re-strip at grid level (same logic as the method).
    quantized: list[tuple[float, float, float]] = []
    for v in vertices:
        q = tuple(round(c, ndigits) for c in v)
        if quantized and q == quantized[-1]:
            continue
        quantized.append(q)
    was_closed = vertices[0] == vertices[-1]
    if was_closed and len(quantized) >= 2 and quantized[0] != quantized[-1]:
        quantized.append(quantized[0])
    vertices = quantized

    # Canonical seam rotation for closed rings.
    if len(vertices) >= max(min_arc_points + 1, 4) and vertices[0] == vertices[-1]:
        seam = _find_canonical_seam(vertices)
        vertices = _rotate_closed(vertices, seam)

    # Decompose into segments using the same greedy arc fitter.
    segments: list[DecompositionSegment] = []
    i = 0
    n = len(vertices)
    while i < n - 1:
        best_arc = None
        if i + min_arc_points <= n:
            for j in range(i + min_arc_points, n + 1):
                pts = np.array(vertices[i:j])
                center, radius, residual = fit_circle_2d(pts[:, :2])
                if residual <= arc_tolerance and radius < 1e6:
                    valid_arc = True
                    for k in range(1, len(pts) - 1):
                        v1 = pts[k][:2] - pts[k - 1][:2]
                        v2 = pts[k + 1][:2] - pts[k][:2]
                        n1 = np.linalg.norm(v1)
                        n2 = np.linalg.norm(v2)
                        if n1 > 1e-6 and n2 > 1e-6:
                            cos_angle = np.dot(v1, v2) / (n1 * n2)
                            if cos_angle < 0.5:
                                valid_arc = False
                                break
                    if valid_arc:
                        cx = round(center[0], ndigits)
                        cy = round(center[1], ndigits)
                        r = round(radius, ndigits)
                        best_arc = DecompositionSegment(
                            points=vertices[i:j],
                            is_arc=True,
                            center=(cx, cy, vertices[i][2]),
                            radius=r,
                        )
                else:
                    break
        if best_arc:
            segments.append(best_arc)
            i += len(best_arc.points) - 1
        else:
            segments.append(
                DecompositionSegment(
                    points=[vertices[i], vertices[i + 1]], is_arc=False
                )
            )
            i += 1

    # Build OCC wire from segments.
    from OCP.GC import GC_MakeArcOfCircle

    wire_builder = BRepBuilderAPI_MakeWire()

    def _rounded_pnt(c):
        return gp_Pnt(*(round(v, ndigits) for v in c))

    for seg in segments:
        if seg.is_arc:
            start_coords = tuple(round(c, ndigits) for c in seg.points[0])
            mid_idx = len(seg.points) // 2
            mid_coords = tuple(round(c, ndigits) for c in seg.points[mid_idx])
            end_coords = tuple(round(c, ndigits) for c in seg.points[-1])
            is_closed = seg.points[0] == seg.points[-1]
            if not is_closed and start_coords == end_coords:
                continue
            p_start = gp_Pnt(*start_coords)
            p_mid = gp_Pnt(*mid_coords)
            p_end = gp_Pnt(*end_coords)
            if is_closed:
                quarter_idx = len(seg.points) // 4
                three_quarter_idx = (len(seg.points) * 3) // 4
                p1 = _rounded_pnt(seg.points[quarter_idx])
                p3 = _rounded_pnt(seg.points[three_quarter_idx])
                arc_geom1 = GC_MakeArcOfCircle(p_start, p1, p_mid).Value()
                edge1 = BRepBuilderAPI_MakeEdge(arc_geom1).Edge()
                arc_geom2 = GC_MakeArcOfCircle(p_mid, p3, p_end).Value()
                edge = BRepBuilderAPI_MakeEdge(arc_geom2).Edge()
                wire_builder.Add(edge1)
            else:
                arc_geom = GC_MakeArcOfCircle(p_start, p_mid, p_end).Value()
                edge = BRepBuilderAPI_MakeEdge(arc_geom).Edge()
        else:
            p1_coords = [round(c, ndigits) for c in seg.points[0]]
            p2_coords = [round(c, ndigits) for c in seg.points[1]]
            if p1_coords == p2_coords:
                continue
            edge = BRepBuilderAPI_MakeEdge(
                gp_Pnt(*p1_coords), gp_Pnt(*p2_coords)
            ).Edge()
        wire_builder.Add(edge)

    return wire_builder.Wire()


def _make_face_from_polygon(polygon: Polygon, z: float) -> Any:
    """Build a planar TopoDS_Face at the given z from a shapely Polygon.

    Handles interior holes (rings) by adding each as a reversed wire to
    the face builder. Forces CCW exterior + CW interior orientation to
    match OCC convention.
    """
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakeWire,
    )
    from OCP.gp import gp_Pnt

    poly = orient(polygon, sign=1.0)

    def _wire_from_coords(coords: list[tuple[float, float]]) -> Any:
        if coords[0] == coords[-1]:
            coords = coords[:-1]
        wire_builder = BRepBuilderAPI_MakeWire()
        for i in range(len(coords)):
            p1 = gp_Pnt(coords[i][0], coords[i][1], z)
            p2 = gp_Pnt(
                coords[(i + 1) % len(coords)][0], coords[(i + 1) % len(coords)][1], z
            )
            edge = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
            wire_builder.Add(edge)
        return wire_builder.Wire()

    outer_wire = _wire_from_coords(list(poly.exterior.coords))
    face_builder = BRepBuilderAPI_MakeFace(outer_wire)
    for ring in poly.interiors:
        inner_wire = _wire_from_coords(list(ring.coords))
        face_builder.Add(inner_wire)
    return face_builder.Face()


def _make_face_from_polygon_with_arcs(
    polygon: Polygon,
    z: float,
    identify_arcs: bool,
    min_arc_points: int,
    arc_tolerance: float,
    point_tolerance: float = _POINT_TOLERANCE,
) -> Any:
    """Build a planar TopoDS_Face at height z, arc-aware when requested.

    When ``identify_arcs=False`` this falls back to the original
    ``_make_face_from_polygon`` path (straight-line edges, no arc logic).

    When ``identify_arcs=True`` the exterior and each interior ring are
    built via ``_make_arc_wire_from_coords``, which produces true OCC arc
    edges matching those built by ``PolyPrism.instanciate_occ`` so that
    BOP can find shared TShapes between the phantom shape and the original.
    """
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace

    if not identify_arcs:
        return _make_face_from_polygon(polygon, z)

    poly = orient(polygon, sign=1.0)

    outer_coords = list(poly.exterior.coords)
    if outer_coords[0] == outer_coords[-1]:
        outer_coords = outer_coords[:-1]
    outer_wire = _make_arc_wire_from_coords(
        outer_coords, z, identify_arcs, min_arc_points, arc_tolerance, point_tolerance
    )
    face_builder = BRepBuilderAPI_MakeFace(outer_wire)
    for ring in poly.interiors:
        inner_coords = list(ring.coords)
        if inner_coords[0] == inner_coords[-1]:
            inner_coords = inner_coords[:-1]
        inner_wire = _make_arc_wire_from_coords(
            inner_coords,
            z,
            identify_arcs,
            min_arc_points,
            arc_tolerance,
            point_tolerance,
        )
        face_builder.Add(inner_wire)
    return face_builder.Face()


def _build_sub_prism(
    piece: Polygon,
    zlo: float,
    zhi: float,
    slab_index: int = 0,
    piece_index: int = 0,
    identify_arcs: bool = False,
    min_arc_points: int = 4,
    arc_tolerance: float = 1e-3,
) -> PhantomShape:
    """Build one OCP sub-prism for a single partition piece.

    Returns a :class:`PhantomShape` carrying:

    - The TopoDS_Solid produced by extruding the piece face from zlo to zhi.
    - The input OCC tags for bottom face, top face, outer-edge edges,
      outer-edge vertices, and lateral faces — keyed by our Phase-2 key
      types so the post-BOP map can index them.

    Inner-ring edges/vertices are NOT keyed (Layer A's outer-only
    contract: lateral OCC faces are 4-corner on the outer boundary; hole
    boundaries are not in the structured pipeline's correspondence map).

    When ``identify_arcs=True`` the face is built with arc-aware OCC
    edges (matching ``PolyPrism.instanciate_occ``) so that BOP sees
    shared TShapes rather than geometrically-overlapping-but-unshared
    boundaries. Edge/lateral enumeration walks the constructed OCC wire
    instead of iterating over polygon vertex pairs, because one arc OCC
    edge may span many polygon vertices.
    """
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.gp import gp_Vec
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
    from OCP.TopoDS import TopoDS

    height = zhi - zlo
    poly = orient(piece, sign=1.0)
    bottom_face = _make_face_from_polygon_with_arcs(
        poly,
        z=zlo,
        identify_arcs=identify_arcs,
        min_arc_points=min_arc_points,
        arc_tolerance=arc_tolerance,
    )
    prism_builder = BRepPrimAPI_MakePrism(bottom_face, gp_Vec(0.0, 0.0, height))
    solid = prism_builder.Shape()
    top_face = prism_builder.LastShape()

    input_faces: dict[FaceKey, Any] = {
        FaceKey(slab_index, "bot", piece_index): bottom_face,
        FaceKey(slab_index, "top", piece_index): top_face,
    }

    # -------------------------------------------------------------------
    # Edge enumeration: walk the OCC wire edges in traversal order.
    #
    # When identify_arcs=False each wire edge corresponds to one polygon
    # segment (classic 1-to-1 mapping). When identify_arcs=True a single
    # arc OCC edge spans many polygon vertices, so we cannot iterate over
    # polygon-vertex pairs — we must walk the wire topology directly.
    #
    # We assign edge_index 0..N-1 from the traversal order of the
    # bottom-face wire (which is shared with the top face by the prism
    # builder's history). ``prism_builder.Generated(bot_edge).First()``
    # gives the corresponding lateral face regardless of how many polygon
    # vertices the edge represents.
    # -------------------------------------------------------------------

    # Collect outer bottom-face edges in wire traversal order.
    # TopExp_Explorer on the face explores ALL edges including interior-ring
    # edges. To get only the outer ring, explore the outer wire directly.
    # The outer wire is the first wire encountered when exploring the face.
    from OCP.TopAbs import TopAbs_WIRE
    from OCP.TopExp import TopExp_Explorer as _TExp

    outer_wire_exp = _TExp(bottom_face, TopAbs_WIRE)
    outer_wire = outer_wire_exp.Current()  # first wire = outer ring

    bot_outer_edges: list[Any] = []
    edge_exp = _TExp(outer_wire, TopAbs_EDGE)
    while edge_exp.More():
        bot_outer_edges.append(edge_exp.Current())
        edge_exp.Next()

    # Collect top-face outer edges in traversal order.
    top_wire_exp = _TExp(top_face, TopAbs_WIRE)
    top_outer_wire = top_wire_exp.Current()
    top_outer_edges: list[Any] = []
    edge_exp = _TExp(top_outer_wire, TopAbs_EDGE)
    while edge_exp.More():
        top_outer_edges.append(edge_exp.Current())
        edge_exp.Next()

    # For each bot outer edge, find the corresponding top edge via the
    # lateral face topology.
    #
    # IMPORTANT: We cannot match by XY endpoint coordinates for arc edges
    # because two arc half-circles share the same two endpoint XY coords
    # (e.g. a full disc split into two 180-degree arcs both have endpoints
    # at (-1,0) and (1,0)). Endpoint-matching would map both bot arcs to
    # the same top arc.
    #
    # Instead, use ``prism_builder.Generated(bot_edge).First()`` to get the
    # lateral face generated by extruding that bot edge, then extract the
    # top edge of that lateral face (the edge at z=zhi). BRepPrimAPI_MakePrism
    # guarantees that Generated() gives the correct lateral face, and the
    # top edge of that face is topologically unique for each bot edge.

    def _top_edge_from_lateral(lateral: Any) -> Any:
        """Return the edge of a lateral face that lies at the top z."""
        for cand in top_outer_edges:
            lat_edge_exp = _TExp(lateral, TopAbs_EDGE)
            while lat_edge_exp.More():
                lat_edge = lat_edge_exp.Current()
                if lat_edge.IsSame(cand):
                    return cand
                lat_edge_exp.Next()
        return None

    input_edges: dict[EdgeKey, Any] = {}
    input_laterals: dict[int, Any] = {}

    for edge_i, bot_edge in enumerate(bot_outer_edges):
        input_edges[EdgeKey(slab_index, "bot", piece_index, edge_i)] = bot_edge
        lateral_face = prism_builder.Generated(bot_edge).First()
        input_laterals[edge_i] = lateral_face

        top_edge = _top_edge_from_lateral(lateral_face)
        if top_edge is None:
            # Fallback: use index-ordered top edge (reliable when top/bot
            # wire traversal orders are consistent, which BRepPrimAPI_MakePrism
            # guarantees for straight extrusion).
            if edge_i < len(top_outer_edges):
                top_edge = top_outer_edges[edge_i]
            else:
                raise RuntimeError(
                    f"_build_sub_prism: no top edge match for bottom edge {edge_i} "
                    f"of piece (slab={slab_index}, piece={piece_index}). "
                    f"This indicates a prism extrusion topology mismatch."
                )
        input_edges[EdgeKey(slab_index, "top", piece_index, edge_i)] = top_edge

    # -------------------------------------------------------------------
    # Inner ring edge tracking (holes/interiors).
    #
    # The Layer A outer-only contract says inner ring edges are NOT
    # added to ``input_laterals_by_outer_edge`` (no structured meshing
    # on inner ring lateral faces). However they MUST appear in
    # ``input_edges_by_key`` so that ``apply_structured_mesh`` can build
    # a complete ``edge_correspondence`` dict covering all boundary edges
    # of the bottom face — otherwise boundary nodes on inner ring edges
    # are absent from ``bot_to_top_tag`` and ``_stamp_top_face_mesh``
    # raises a KeyError when triangle stamping looks them up.
    #
    # We use a convention: inner ring ``r`` edge ``j`` gets
    # edge_index = n_outer + r * _INNER_RING_STRIDE + j.
    # This keeps the indices unique and non-negative.
    # -------------------------------------------------------------------
    _INNER_RING_STRIDE = 100_000  # large enough to not collide with outer edges

    n_outer = len(bot_outer_edges)

    # Walk all wires of the bottom face; skip the first (outer ring = already done).
    bot_all_wires: list[Any] = []
    wire_exp2 = _TExp(bottom_face, TopAbs_WIRE)
    while wire_exp2.More():
        bot_all_wires.append(wire_exp2.Current())
        wire_exp2.Next()

    top_all_wires: list[Any] = []
    wire_exp2 = _TExp(top_face, TopAbs_WIRE)
    while wire_exp2.More():
        top_all_wires.append(wire_exp2.Current())
        wire_exp2.Next()

    def _top_edge_from_lateral_ring(lateral: Any, candidates: list[Any]) -> Any | None:
        """Find the top ring edge that lies on ``lateral``."""
        for cand in candidates:
            lat_edge_exp = _TExp(lateral, TopAbs_EDGE)
            while lat_edge_exp.More():
                lat_edge = lat_edge_exp.Current()
                if lat_edge.IsSame(cand):
                    return cand
                lat_edge_exp.Next()
        return None

    for ring_idx, (bot_ring_wire, top_ring_wire) in enumerate(
        zip(bot_all_wires[1:], top_all_wires[1:]), start=1
    ):
        # Collect ring edges in traversal order.
        bot_ring_edges: list[Any] = []
        e_exp = _TExp(bot_ring_wire, TopAbs_EDGE)
        while e_exp.More():
            bot_ring_edges.append(e_exp.Current())
            e_exp.Next()

        top_ring_edges: list[Any] = []
        e_exp = _TExp(top_ring_wire, TopAbs_EDGE)
        while e_exp.More():
            top_ring_edges.append(e_exp.Current())
            e_exp.Next()

        for j, bot_re in enumerate(bot_ring_edges):
            inner_edge_idx = n_outer + ring_idx * _INNER_RING_STRIDE + j
            input_edges[
                EdgeKey(slab_index, "bot", piece_index, inner_edge_idx)
            ] = bot_re

            # Find the corresponding top ring edge via the generated lateral face.
            inner_lateral = prism_builder.Generated(bot_re).First()
            top_re = _top_edge_from_lateral_ring(inner_lateral, top_ring_edges)
            if top_re is None and j < len(top_ring_edges):
                top_re = top_ring_edges[j]
            if top_re is not None:
                input_edges[
                    EdgeKey(slab_index, "top", piece_index, inner_edge_idx)
                ] = top_re

    # -------------------------------------------------------------------
    # Vertex enumeration: walk bottom/top face vertices, dedupe via IsSame.
    # corner_index is assigned in the same traversal order as edges above,
    # taking the start vertex of each bot edge.
    # -------------------------------------------------------------------

    def _collect_outer_wire_vertices(wire: Any) -> list[Any]:
        """Return deduplicated vertices in wire traversal order."""
        verts: list[Any] = []
        seen_verts: list[Any] = []
        vert_exp = _TExp(wire, TopAbs_VERTEX)
        while vert_exp.More():
            v = TopoDS.Vertex_s(vert_exp.Current())
            if not any(v.IsSame(s) for s in seen_verts):
                seen_verts.append(v)
                verts.append(v)
            vert_exp.Next()
        return verts

    bot_outer_verts = _collect_outer_wire_vertices(outer_wire)
    top_outer_verts = _collect_outer_wire_vertices(top_outer_wire)

    input_vertices: dict[VertexKey, Any] = {}
    for corner_i, v in enumerate(bot_outer_verts):
        input_vertices[VertexKey(slab_index, "bot", piece_index, corner_i)] = v
    for corner_i, v in enumerate(top_outer_verts):
        input_vertices[VertexKey(slab_index, "top", piece_index, corner_i)] = v

    return PhantomShape(
        slab_index=slab_index,
        piece_index=piece_index,
        solid=solid,
        input_faces_by_key=input_faces,
        input_edges_by_key=input_edges,
        input_vertices_by_key=input_vertices,
        input_laterals_by_outer_edge=input_laterals,
    )


@phase_timed("phantom_build")
def build_phantom_shapes(plan: StructuredPlan) -> PhantomBuildResult:
    """For each slab, build one OCP sub-prism per partition piece.

    Returns a :class:`PhantomBuildResult` with shapes in
    (slab_index, piece_index) ascending order for deterministic
    downstream processing.
    """
    shapes: list[PhantomShape] = []
    for slab_index, slab in enumerate(plan.slabs):
        if not slab.face_partition:
            continue

        # Phase 6(a1) guard: split-arc provenance (multiple pieces from an
        # arc-bearing footprint) is not yet implemented. Phase 6(a2) will
        # replace this error with proper per-segment arc tagging.
        if slab.identify_arcs and len(slab.face_partition) > 1:
            raise NotImplementedError(
                f"Slab {slab.physical_name}: identify_arcs=True is only supported "
                f"for single-piece face_partition in Phase 6(a1). Got "
                f"{len(slab.face_partition)} pieces — split-arc provenance is "
                f"Phase 6(a2)."
            )

        for piece_index, piece in enumerate(slab.face_partition):
            shapes.append(
                _build_sub_prism(
                    piece=piece,
                    zlo=slab.zlo,
                    zhi=slab.zhi,
                    slab_index=slab_index,
                    piece_index=piece_index,
                    identify_arcs=slab.identify_arcs,
                    min_arc_points=slab.min_arc_points,
                    arc_tolerance=slab.arc_tolerance,
                )
            )
    return PhantomBuildResult(shapes=tuple(shapes))


_MIDHEIGHT_TOL = 1e-7


def _slab_z_range_for_shape(shape: PhantomShape) -> tuple[float, float]:
    """Recover (zlo, zhi) for a shape from its input bottom/top faces.

    PhantomShape doesn't store zlo/zhi directly; read it from any
    vertex on the bottom face (= zlo) and any vertex on the top face
    (= zhi).
    """
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    bot_key = FaceKey(
        slab_index=shape.slab_index, side="bot", piece_index=shape.piece_index
    )
    top_key = FaceKey(
        slab_index=shape.slab_index, side="top", piece_index=shape.piece_index
    )
    bot_face = shape.input_faces_by_key[bot_key]
    top_face = shape.input_faces_by_key[top_key]

    def _any_z(face: Any) -> float:
        exp = TopExp_Explorer(face, TopAbs_VERTEX)
        v = TopoDS.Vertex_s(exp.Current())
        return BRep_Tool.Pnt_s(v).Z()

    return _any_z(bot_face), _any_z(top_face)


def _has_midheight_vertex(
    builder: Any, lateral_face: Any, zlo: float, zhi: float
) -> bool:
    """True if BOP.Generated(lateral_face) produced any vertex with zlo < z < zhi."""
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_VERTEX
    from OCP.TopoDS import TopoDS

    generated = builder.Generated(lateral_face)
    if generated.IsEmpty():
        return False
    for sub in list(generated):
        if sub.ShapeType() == TopAbs_VERTEX:
            v = TopoDS.Vertex_s(sub)
            z = BRep_Tool.Pnt_s(v).Z()
            if zlo + _MIDHEIGHT_TOL < z < zhi - _MIDHEIGHT_TOL:
                return True
    return False


def _modified_or_unchanged(builder: Any, input_shape: Any) -> list[Any]:
    """Return list of output shapes for input_shape.

    Mirrors the cad_occ.py pattern: if Modified() is empty AND the shape
    is not deleted, the shape passed through unchanged (input == output,
    one element). Otherwise Modified() gives the actual successor list.
    """
    modified = builder.Modified(input_shape)
    if modified.IsEmpty():
        if builder.IsDeleted(input_shape):
            return []
        return [input_shape]
    return list(modified)


@phase_timed("phantom_map")
def extract_phantom_map(
    build_result: PhantomBuildResult,
    builder: Any,
) -> PhantomMap:
    """Walk the post-Perform BOP history to build the PhantomMap.

    For every input OCC tag recorded in ``build_result``, ask the
    ``builder`` (a ``BOPAlgo_Builder`` or any object exposing
    ``Modified(shape)`` / ``IsDeleted(shape)``) what the input became
    in the output.

    Args:
        build_result: From :func:`build_phantom_shapes`.
        builder: Post-Perform BOP builder.

    Returns:
        :class:`PhantomMap` with all four output_*_by_key dicts
        populated. Each value is a list because a single input can
        split into many outputs.
    """
    pmap = PhantomMap()
    for shape in build_result.shapes:
        for face_key, in_face in shape.input_faces_by_key.items():
            pmap.output_faces[face_key] = _modified_or_unchanged(builder, in_face)
        for edge_key, in_edge in shape.input_edges_by_key.items():
            pmap.output_edges[edge_key] = _modified_or_unchanged(builder, in_edge)
        for vert_key, in_vert in shape.input_vertices_by_key.items():
            pmap.output_vertices[vert_key] = _modified_or_unchanged(builder, in_vert)
        for outer_edge_idx, in_lateral in shape.input_laterals_by_outer_edge.items():
            lateral_key = LateralKey(
                slab_index=shape.slab_index,
                piece_index=shape.piece_index,
                outer_edge_index=outer_edge_idx,
            )
            pmap.output_laterals[lateral_key] = _modified_or_unchanged(
                builder, in_lateral
            )
        slab_zlo, slab_zhi = _slab_z_range_for_shape(shape)
        for outer_edge_idx, in_lateral in shape.input_laterals_by_outer_edge.items():
            lateral_key = LateralKey(
                slab_index=shape.slab_index,
                piece_index=shape.piece_index,
                outer_edge_index=outer_edge_idx,
            )
            pmap.lateral_has_midheight_cut[lateral_key] = _has_midheight_vertex(
                builder, in_lateral, slab_zlo, slab_zhi
            )
    return pmap
