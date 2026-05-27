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

import itertools
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
    PieceArcEdge,
    PieceLineEdge,
    PieceProvenance,
    Slab,
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


def _make_occ_wire_from_labeled_segments(
    segments: list[PieceArcEdge | PieceLineEdge],
    point_tolerance: float = _POINT_TOLERANCE,
) -> Any:
    """Build an OCC wire from a list of already-classified provenance edges.

    Unlike ``_make_arc_wire_from_coords``, this never runs the heuristic
    arc-detector — the provenance is authoritative.

    ``segments`` must be in boundary traversal order; consecutive edges must
    share an endpoint (caller's responsibility). Coordinates are quantized
    to ``point_tolerance`` to match ``_make_arc_wire_from_coords``.
    """
    import numpy as np
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
    from OCP.GC import GC_MakeArcOfCircle
    from OCP.gp import gp_Pnt

    ndigits = max(0, int(-np.floor(np.log10(point_tolerance))))

    def _qpnt(coords: tuple[float, float, float]) -> gp_Pnt:
        return gp_Pnt(*(round(c, ndigits) for c in coords))

    def _qkey(coords: tuple[float, float, float]) -> tuple:
        return tuple(round(c, ndigits) for c in coords)

    wire_builder = BRepBuilderAPI_MakeWire()
    for seg in segments:
        if isinstance(seg, PieceArcEdge):
            pts = list(seg.points)
            if len(pts) < 2:
                continue
            is_closed = _qkey(pts[0]) == _qkey(pts[-1])
            if is_closed:
                if len(pts) < 4:
                    continue  # degenerate closed arc with too few unique points
                # Full circle: two arcs (matches _make_arc_wire_from_coords behaviour).
                quarter = len(pts) // 4
                mid = len(pts) // 2
                three_quarter = (len(pts) * 3) // 4
                p_start = _qpnt(pts[0])
                p_q = _qpnt(pts[quarter])
                p_mid = _qpnt(pts[mid])
                p_3q = _qpnt(pts[three_quarter])
                p_end = _qpnt(pts[-1])
                arc1 = GC_MakeArcOfCircle(p_start, p_q, p_mid).Value()
                arc2 = GC_MakeArcOfCircle(p_mid, p_3q, p_end).Value()
                wire_builder.Add(BRepBuilderAPI_MakeEdge(arc1).Edge())
                wire_builder.Add(BRepBuilderAPI_MakeEdge(arc2).Edge())
            else:
                if len(pts) < 2:
                    continue
                if _qkey(pts[0]) == _qkey(pts[-1]):
                    continue
                if len(pts) == 2:
                    # Only two points on arc — build as line (degenerate arc).
                    wire_builder.Add(
                        BRepBuilderAPI_MakeEdge(_qpnt(pts[0]), _qpnt(pts[-1])).Edge()
                    )
                else:
                    mid = len(pts) // 2
                    p_start = _qpnt(pts[0])
                    p_mid = _qpnt(pts[mid])
                    p_end = _qpnt(pts[-1])
                    arc_geom = GC_MakeArcOfCircle(p_start, p_mid, p_end).Value()
                    wire_builder.Add(BRepBuilderAPI_MakeEdge(arc_geom).Edge())
        elif isinstance(seg, PieceLineEdge):
            p1 = _qpnt(seg.points[0])
            p2 = _qpnt(seg.points[1])
            if _qkey(seg.points[0]) == _qkey(seg.points[1]):
                continue
            wire_builder.Add(BRepBuilderAPI_MakeEdge(p1, p2).Edge())
        else:
            raise TypeError(
                f"_make_occ_wire_from_labeled_segments: unknown segment "
                f"type {type(seg).__name__}"
            )
    return wire_builder.Wire()


def _shift_provenance_edge(
    edge: PieceArcEdge | PieceLineEdge,
    z: float,
) -> PieceArcEdge | PieceLineEdge:
    """Stamp height ``z`` onto each edge's points.

    Provenance is computed at z=0 (face_partition is a 2-D polygon set),
    so we raise to the slab's actual zlo here for OCC consumption.
    """
    if isinstance(edge, PieceArcEdge):
        return PieceArcEdge(
            points=tuple((p[0], p[1], z) for p in edge.points),
            center=(edge.center[0], edge.center[1], z),
            radius=edge.radius,
        )
    return PieceLineEdge(
        points=(
            (edge.points[0][0], edge.points[0][1], z),
            (edge.points[1][0], edge.points[1][1], z),
        ),
    )


def _make_face_from_provenance(
    provenance: PieceProvenance,
    z: float,
    point_tolerance: float = _POINT_TOLERANCE,
) -> Any:
    """Build a planar TopoDS_Face at height z from a PieceProvenance.

    Exterior ring is built from provenance.exterior_edges; each interior
    ring from provenance.interior_edges[i]. Arc edges are built with
    GC_MakeArcOfCircle, line edges with BRepBuilderAPI_MakeEdge.
    """
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace

    outer_edges = [_shift_provenance_edge(e, z) for e in provenance.exterior_edges]
    outer_wire = _make_occ_wire_from_labeled_segments(outer_edges, point_tolerance)
    face_builder = BRepBuilderAPI_MakeFace(outer_wire)
    for ring_edges in provenance.interior_edges:
        shifted = [_shift_provenance_edge(e, z) for e in ring_edges]
        hole_wire = _make_occ_wire_from_labeled_segments(shifted, point_tolerance)
        face_builder.Add(hole_wire)
    return face_builder.Face()


def _polygon_face_cache_key(
    poly: Polygon,
    identify_arcs: bool,
    min_arc_points: int,
    arc_tolerance: float,
) -> tuple:
    """Hashable key for caching a face built from polygon coords + arc settings."""
    ext = tuple((round(x, 9), round(y, 9)) for x, y in tuple(poly.exterior.coords)[:-1])
    interiors = tuple(
        tuple((round(x, 9), round(y, 9)) for x, y in tuple(ring.coords)[:-1])
        for ring in poly.interiors
    )
    return (
        "poly",
        ext,
        interiors,
        identify_arcs,
        min_arc_points,
        arc_tolerance,
    )


def _provenance_face_cache_key(prov: "PieceProvenance") -> tuple:
    """Hashable key for caching a face built from provenance — XY-only, z ignored.

    The cached face is built at z=0; consumers translate to the slab's
    zlo via ``_face_at_z``.
    """

    def _edge_key(e: "PieceArcEdge | PieceLineEdge") -> tuple:
        if isinstance(e, PieceArcEdge):
            return (
                "arc",
                tuple((p[0], p[1]) for p in e.points),
                (e.center[0], e.center[1]),
                e.radius,
            )
        return ("line", tuple((p[0], p[1]) for p in e.points))

    ext = tuple(_edge_key(e) for e in prov.exterior_edges)
    interiors = tuple(tuple(_edge_key(e) for e in ring) for ring in prov.interior_edges)
    return ("prov", ext, interiors)


# Tolerance for "vertically touching" — two slabs are in the same vertical
# stack iff abs(upper.zlo - lower.zhi) < _Z_TOL_VERT.
_Z_TOL_VERT = 1e-9


def _group_slabs_into_vertical_stacks(
    plan: StructuredPlan,
) -> list[list[tuple[Slab, int]]]:
    """Group sub-prism pieces into vertical stacks for pre-shared-face construction.

    A "stack" is a sequence of (slab, piece_index) pairs such that:
      - All pairs share the same component_index (cohort).
      - All pairs have polygon-face-cache-key equality for their piece.
      - Pairs are sorted ascending by slab.zlo.
      - Adjacent pairs satisfy abs(upper.zlo - lower.zhi) < _Z_TOL_VERT.

    Singleton stacks (pieces with no z-touching neighbor) are returned as
    length-1 lists. Each (slab, piece_index) pair appears in exactly one
    stack.

    See spec docs/superpowers/specs/2026-05-27-cad-occ-cohort-preshared-faces-design.md.
    """
    triples: list[tuple[Slab, int, tuple]] = []
    for slab in plan.slabs:
        if not slab.face_partition:
            continue
        provenance_list = slab.face_partition_provenance
        for piece_index, piece in enumerate(slab.face_partition):
            piece_provenance: PieceProvenance | None = None
            if provenance_list is not None and piece_index < len(provenance_list):
                piece_provenance = provenance_list[piece_index]
            if piece_provenance is not None:
                key = _provenance_face_cache_key(piece_provenance)
            else:
                key = _polygon_face_cache_key(
                    orient(piece, sign=1.0),
                    slab.identify_arcs,
                    slab.min_arc_points,
                    slab.arc_tolerance,
                )
            triples.append((slab, piece_index, key))

    buckets: dict[tuple[int, tuple], list[tuple[Slab, int]]] = {}
    for slab, piece_index, key in triples:
        buckets.setdefault((slab.component_index, key), []).append((slab, piece_index))

    stacks: list[list[tuple[Slab, int]]] = []
    for bucket in buckets.values():
        bucket.sort(key=lambda pair: pair[0].zlo)
        current: list[tuple[Slab, int]] = [bucket[0]]
        for prev, curr in itertools.pairwise(bucket):
            prev_slab, _prev_pi = prev
            curr_slab, _curr_pi = curr
            if abs(curr_slab.zlo - prev_slab.zhi) < _Z_TOL_VERT:
                current.append(curr)
            else:
                stacks.append(current)
                current = [curr]
        stacks.append(current)

    return stacks


def _face_at_z(blueprint: Any, z: float) -> Any:
    """Return a deep-copied translation of ``blueprint`` (built at z=0) up to ``z``.

    Uses ``BRepBuilderAPI_Transform(Copy=True)`` so each consumer gets a
    fresh TShape — distinct edges/vertices for downstream per-slab keying.
    The geometry-level work (arc primitives, wire construction, face
    builder) is reused from the cached blueprint, so this is much
    cheaper than rebuilding from coords.
    """
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf, gp_Vec
    from OCP.TopoDS import TopoDS

    trsf = gp_Trsf()
    trsf.SetTranslation(gp_Vec(0.0, 0.0, z))
    builder = BRepBuilderAPI_Transform(blueprint, trsf, True)
    return TopoDS.Face_s(builder.Shape())


def _build_sub_prism(
    piece: Polygon,
    zlo: float,
    zhi: float,
    slab_index: int = 0,
    piece_index: int = 0,
    identify_arcs: bool = False,
    min_arc_points: int = 4,
    arc_tolerance: float = 1e-3,
    provenance: PieceProvenance | None = None,
    face_cache: dict | None = None,
    bottom_face_override: Any | None = None,
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
    # Face construction: when a cache is provided, build the blueprint at
    # z=0 once per unique (XY, arc settings) key, then translate-copy to
    # this slab's zlo for each consumer. On stacked scenes with many
    # slabs sharing the same XY pieces, this avoids rebuilding the wire
    # + arc primitives from scratch per slab.
    # When pre-shared by the caller (e.g., vertically-stacked cohort),
    # reuse the provided face directly. Per spec
    # 2026-05-27-cad-occ-cohort-preshared-faces-design.md.
    if bottom_face_override is not None:
        bottom_face = bottom_face_override
    elif face_cache is not None:
        if provenance is not None:
            key = _provenance_face_cache_key(provenance)
            blueprint = face_cache.get(key)
            if blueprint is None:
                blueprint = _make_face_from_provenance(provenance, z=0.0)
                face_cache[key] = blueprint
        else:
            key = _polygon_face_cache_key(
                poly, identify_arcs, min_arc_points, arc_tolerance
            )
            blueprint = face_cache.get(key)
            if blueprint is None:
                blueprint = _make_face_from_polygon_with_arcs(
                    poly,
                    z=0.0,
                    identify_arcs=identify_arcs,
                    min_arc_points=min_arc_points,
                    arc_tolerance=arc_tolerance,
                )
                face_cache[key] = blueprint
        bottom_face = _face_at_z(blueprint, zlo)
    elif provenance is not None:
        # Provenance path: use labeled edges directly — no heuristic re-detection.
        bottom_face = _make_face_from_provenance(provenance, z=zlo)
    else:
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

    # Build a TopTools_IndexedMapOfShape over top_outer_edges once so each
    # lateral->top-edge lookup is O(lateral_edge_count) hashed map lookups,
    # not O(top_edge_count * lateral_edge_count) IsSame loops. On a complex
    # scene this phase was 47% of build_phantom_shapes' time.
    from OCP.TopTools import TopTools_IndexedMapOfShape

    _top_edge_map = TopTools_IndexedMapOfShape()
    for _e in top_outer_edges:
        _top_edge_map.Add(_e)

    def _top_edge_from_lateral(lateral: Any) -> Any:
        """Return the edge of a lateral face that lies at the top z."""
        lat_edge_exp = _TExp(lateral, TopAbs_EDGE)
        while lat_edge_exp.More():
            lat_edge = lat_edge_exp.Current()
            idx = _top_edge_map.FindIndex(lat_edge)
            if idx > 0:
                return top_outer_edges[idx - 1]
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

    # Fast path: a polygon with no interior rings (the common case after
    # the annular-split fix) has a single wire on each of bottom/top face.
    # Skip the wire walks entirely.
    if not piece.interiors:
        bot_all_wires = [outer_wire]
        top_all_wires = [top_outer_wire]
    else:
        bot_all_wires = []
        wire_exp2 = _TExp(bottom_face, TopAbs_WIRE)
        while wire_exp2.More():
            bot_all_wires.append(wire_exp2.Current())
            wire_exp2.Next()

        top_all_wires = []
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
        seen_map = TopTools_IndexedMapOfShape()
        verts: list[Any] = []
        vert_exp = _TExp(wire, TopAbs_VERTEX)
        while vert_exp.More():
            v = TopoDS.Vertex_s(vert_exp.Current())
            prev_size = seen_map.Extent()
            seen_map.Add(v)
            if seen_map.Extent() > prev_size:
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


def _group_phantom_solids_by_entity(
    plan: StructuredPlan,
    phantom_result: PhantomBuildResult,
) -> dict[int, list[Any]]:
    """Group phantom solids by their source entity's input-list position.

    Each ``PhantomShape`` carries ``slab_index``; ``plan.slabs[slab_index]``
    carries ``source_index`` (position in the input ``entities`` list passed
    to ``build_plan``). The returned dict maps ``source_index -> [solids]``
    in ``(slab_index, piece_index)`` ascending order — the same order
    ``build_phantom_shapes`` populates ``phantom_result.shapes``.

    Structured entities whose every slab is fully carved out by Policy B
    (empty resolved_footprint → ``build_phantom_shapes`` skipped them)
    still get an empty entry ``source_index -> []`` so cad_occ's
    ``overridden_indices`` includes them and the per-entity sequential
    cut loop skips them entirely. Without this, cad_occ would fall back
    to ``entity_obj.instanciate_occ()`` for the carved entity, building
    its FULL pre-carve volume and dragging it into BOP fragmentation —
    where it would claim ownership of pieces that the planner already
    assigned to the carving (higher-priority) entity.
    """
    out: dict[int, list[Any]] = {}
    for shape in phantom_result.shapes:
        src = plan.slabs[shape.slab_index].source_index
        out.setdefault(src, []).append(shape.solid)
    for slab in plan.slabs:
        out.setdefault(slab.source_index, [])
    return out


@phase_timed("phantom_build")
def build_phantom_shapes(plan: StructuredPlan) -> PhantomBuildResult:
    """For each slab, build one OCP sub-prism per partition piece.

    Returns a :class:`PhantomBuildResult` with shapes in
    (slab_index, piece_index) ascending order for deterministic
    downstream processing.
    """
    # Per-call cache of bottom-face blueprints (built at z=0) keyed by
    # canonical XY signature + arc settings. Each consumer gets a fresh
    # TopoDS_Face via translate-copy (BRepBuilderAPI_Transform(Copy=True))
    # so per-slab edge tracking still has distinct TShapes.
    face_cache: dict = {}
    shapes: list[PhantomShape] = []
    for slab_index, slab in enumerate(plan.slabs):
        if not slab.face_partition:
            continue

        provenance_list = (
            slab.face_partition_provenance
        )  # None or list[PieceProvenance]

        for piece_index, piece in enumerate(slab.face_partition):
            piece_provenance: PieceProvenance | None = None
            if provenance_list is not None and piece_index < len(provenance_list):
                piece_provenance = provenance_list[piece_index]
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
                    provenance=piece_provenance,
                    face_cache=face_cache,
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
