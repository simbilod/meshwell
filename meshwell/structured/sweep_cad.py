"""CAD-stage sweep pass: region resolution and attachment lookup.

Phase 1 of the structured-sweep CAD pipeline: turn the 2D/1D meshwell
entities into final (mesh_order-resolved) per-physical-name regions, then
resolve a :class:`~meshwell.structured.sweep.StructuredSweep`'s ``on``
target to the straight two-point segment it attaches to. Clip/imprint
(consuming these outputs) is out of scope here.
"""
from __future__ import annotations

import logging

import numpy as np
import shapely
from shapely.geometry import LineString, Point

from meshwell.structured.exceptions import (
    SweepAttachmentNotFoundError,
    SweepCurvedSourceError,
    SweepOverlapError,
)

logger = logging.getLogger(__name__)

_DELIM = "___"


def final_region_polygons(entities) -> dict:
    """Compute final per-physical-name 2D region polygons.

    Applies meshwell's mesh_order precedence: lower ``mesh_order`` wins
    overlaps against higher ``mesh_order`` (and against entities with no
    explicit order, which sort last, in list order).

    Args:
        entities: meshwell 2D/1D geometry entities (e.g. PolySurface,
            PolyLine). Non-2D and mesh_bool=False entities are ignored.

    Returns:
        Mapping of physical name to its final shapely Polygon/MultiPolygon,
        after subtracting the footprint already claimed by higher-priority
        entities.
    """
    surf = [e for e in entities if getattr(e, "dimension", None) == 2 and e.mesh_bool]
    order = sorted(
        range(len(surf)),
        key=lambda i: (surf[i].mesh_order is None, surf[i].mesh_order, i),
    )
    taken = None
    out: dict = {}
    for i in order:
        ent = surf[i]
        poly = shapely.unary_union(ent.polygons)
        if taken is not None:
            poly = poly.difference(taken)
        taken = poly if taken is None else shapely.unary_union([taken, poly])
        for name in ent.physical_name or ():
            out[name] = (
                poly if name not in out else shapely.unary_union([out[name], poly])
            )
    return out


def _as_straight_segment(geom, point_tolerance: float, context: str):
    """Merge and simplify a linear geometry into a single straight segment.

    Args:
        geom: shapely (Multi)LineString (or GeometryCollection thereof).
        point_tolerance: Douglas-Peucker simplification tolerance.
        context: the sweep's ``on`` string, used in error messages.

    Returns:
        Tuple of (p0, p1) numpy arrays.

    Raises:
        SweepCurvedSourceError: geometry is not a single straight segment.
    """
    if geom.geom_type == "GeometryCollection":
        # boundary intersections can mix a shared edge (LineString) with a
        # stray touching corner (Point) elsewhere; only lineal parts are
        # eligible, and any non-lineal part disqualifies the attachment.
        parts = shapely.get_parts(geom)
        lineal = [p for p in parts if p.geom_type in ("LineString", "MultiLineString")]
        if not lineal or len(lineal) != len(parts):
            raise SweepCurvedSourceError(context, geom.geom_type)
        geom = shapely.GeometryCollection(lineal)
    try:
        merged = geom if geom.geom_type == "LineString" else shapely.line_merge(geom)
        if merged.is_empty:
            raise SweepCurvedSourceError(context, merged.geom_type)
        if merged.geom_type == "MultiLineString":
            # Collinear pieces separated by an imprinted vertex on the seam
            # (e.g. a ridge corner splitting an interface into two aligned
            # sub-segments): collapse to the overall straight extent. The
            # clip pass re-subtracts any genuine gap by full-normal-extent.
            ends = _collinear_extent(merged, point_tolerance)
            if ends is None:
                raise SweepCurvedSourceError(context, merged.geom_type)
            return ends
        if merged.geom_type != "LineString":
            raise SweepCurvedSourceError(context, merged.geom_type)
        simple = merged.simplify(point_tolerance)
        coords = list(simple.coords)
    except SweepCurvedSourceError:
        raise
    except Exception as exc:
        raise SweepCurvedSourceError(context, str(exc)) from exc
    if len(coords) != 2:
        raise SweepCurvedSourceError(context, f"{len(coords)}-point polyline")
    return np.asarray(coords[0]), np.asarray(coords[1])


def _collinear_extent(multiline, point_tolerance: float):
    """Extreme endpoints of a collinear MultiLineString, or None if not collinear.

    Every vertex must lie (within ``point_tolerance``) on the line through
    the cloud's principal axis; returns ``(p_min, p_max)`` by projection.
    """
    pts = np.asarray(
        [c for part in multiline.geoms for c in part.coords], dtype=float
    )
    d = pts.max(axis=0) - pts.min(axis=0)
    n = np.linalg.norm(d)
    if n == 0:
        return None
    u = d / n
    p0 = pts[0]
    for p in pts:
        v = p - p0
        if abs(u[0] * v[1] - u[1] * v[0]) > 10 * point_tolerance:
            return None  # not collinear -> genuine curved/branching source
    proj = pts @ u
    return pts[int(np.argmin(proj))], pts[int(np.argmax(proj))]


def resolve_attachment(sweep, entities, region_polys: dict, point_tolerance: float):
    """Resolve a sweep's ``on`` target to its straight attachment segment.

    Args:
        sweep: the StructuredSweep to resolve.
        entities: the full entity list (searched for polyline attachments).
        region_polys: output of :func:`final_region_polygons`.
        point_tolerance: Douglas-Peucker simplification tolerance used when
            collapsing the resolved geometry to two endpoints.

    Returns:
        Tuple of (p0, p1) numpy arrays. For a polyline attachment, p0/p1
        preserve the user's coordinate order (travel direction matters for
        "left"/"right" sides).

    Raises:
        SweepAttachmentNotFoundError: ``on`` doesn't resolve to any
            geometry (missing physical name, or regions not adjacent).
        SweepCurvedSourceError: the resolved attachment is not a single
            straight segment.
    """
    kind = sweep.attachment_kind
    if kind == "polyline":
        for ent in entities:
            if getattr(ent, "dimension", None) == 1 and sweep.on in (
                ent.physical_name or ()
            ):
                line = ent.linestrings[0]
                coords = list(line.coords)
                if len(coords) != 2:
                    # allow collinear multi-point lines
                    coords = list(line.simplify(point_tolerance).coords)
                if len(coords) != 2:
                    raise SweepCurvedSourceError(sweep.on, f"{len(coords)}-point polyline")
                return np.asarray(coords[0]), np.asarray(coords[1])
        raise SweepAttachmentNotFoundError(sweep.name, sweep.on)

    a, b = sweep.on.split(_DELIM)
    if a not in region_polys:
        raise SweepAttachmentNotFoundError(sweep.name, sweep.on)
    if kind == "interface":
        if b not in region_polys:
            raise SweepAttachmentNotFoundError(sweep.name, sweep.on)
        shared = region_polys[a].boundary.intersection(region_polys[b].boundary)
    else:  # boundary
        hull = shapely.unary_union(list(region_polys.values()))
        shared = region_polys[a].boundary.intersection(hull.boundary)
    if shared.is_empty:
        raise SweepAttachmentNotFoundError(sweep.name, sweep.on)
    return _as_straight_segment(shared, point_tolerance, sweep.on)


def side_normal(p0, p1, side: str, sweep, region_polys: dict, point_tolerance: float):
    """Compute the unit normal from the attachment segment into ``side``.

    Args:
        p0: first endpoint of the attachment segment.
        p1: second endpoint of the attachment segment.
        side: sweep side key ("left"/"right" for polyline attachments, or
            a region physical name for interface/boundary attachments).
        sweep: the StructuredSweep (used for its thickness and for error
            messages).
        region_polys: output of :func:`final_region_polygons`.
        point_tolerance: absolute distance tolerance used to nudge the
            probe point off the segment.

    Returns:
        Unit numpy vector pointing from the segment into ``side``.

    Raises:
        SweepAttachmentNotFoundError: ``side`` names a region that isn't
            adjacent to the attachment segment on either side.
    """
    t = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    t = t / np.linalg.norm(t)
    left = np.array([-t[1], t[0]])  # +90 deg: material-left-of-travel
    if side == "left":
        return left
    if side == "right":
        return -left
    mid = (np.asarray(p0) + np.asarray(p1)) / 2.0
    eps = sweep.thickness[side] * 1e-3 + point_tolerance
    if region_polys[side].contains(Point(*(mid + left * eps))):
        return left
    if region_polys[side].contains(Point(*(mid - left * eps))):
        return -left
    raise SweepAttachmentNotFoundError(sweep.name, f"{sweep.on} (side {side} not adjacent)")


def clip_sweep_side(p0, p1, n_dir, thickness, region_poly, point_tolerance):
    """Kept tangential intervals where the FULL normal extent fits in region.

    Any deficit piece (rect minus region) blocks its entire tangential
    shadow — per the design's full-normal-extent rule.

    Args:
        p0: first endpoint of the attachment segment.
        p1: second endpoint of the attachment segment.
        n_dir: unit outward normal for this side.
        thickness: sweep thickness on this side.
        region_poly: the adjacent region's final polygon, used to test
            where the full-thickness band actually has material.
        point_tolerance: absolute distance tolerance for treating a
            deficit piece as a real gap vs. a boundary sliver.

    Returns:
        List of (lo, hi) tangential-arclength intervals (from p0) that
        keep their full normal extent inside ``region_poly``.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    t_hat = (p1 - p0) / np.linalg.norm(p1 - p0)
    length = float(np.linalg.norm(p1 - p0))
    rect = shapely.Polygon(
        [p0, p1, p1 + n_dir * thickness, p0 + n_dir * thickness]
    )
    deficit = rect.difference(region_poly.buffer(point_tolerance))
    kept = [(0.0, length)]
    pieces = getattr(deficit, "geoms", [deficit]) if not deficit.is_empty else []
    for piece in pieces:
        if piece.area <= (10 * point_tolerance) ** 2:
            continue  # tolerance sliver, not a real deficit
        ts = [float(np.dot(np.asarray(c) - p0, t_hat)) for c in piece.exterior.coords]
        blo, bhi = min(ts), max(ts)
        nxt = []
        for lo, hi in kept:
            if bhi <= lo or blo >= hi:
                nxt.append((lo, hi))
                continue
            if blo > lo:
                nxt.append((lo, blo))
            if bhi < hi:
                nxt.append((bhi, hi))
        kept = nxt
    return [(lo, hi) for lo, hi in kept if hi - lo > 10 * point_tolerance]


def sweep_rectangles(p0, p1, n_dir, thickness, intervals):
    """Shapely rectangles for the kept intervals, in the (t, n) frame.

    Args:
        p0: first endpoint of the attachment segment.
        p1: second endpoint of the attachment segment.
        n_dir: unit outward normal for this side.
        thickness: sweep thickness on this side.
        intervals: (lo, hi) tangential-arclength intervals, as returned
            by :func:`clip_sweep_side`.

    Returns:
        List of shapely Polygons, one per interval.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    t_hat = (p1 - p0) / np.linalg.norm(p1 - p0)
    out = []
    for lo, hi in intervals:
        a = p0 + t_hat * lo
        b = p0 + t_hat * hi
        out.append(shapely.Polygon([a, b, b + n_dir * thickness, a + n_dir * thickness]))
    return out


def _polyline_target_region(p0, p1, n_dir, region_polys):
    """For left/right sides: the region containing a probe point off the line."""
    mid = (np.asarray(p0) + np.asarray(p1)) / 2.0
    seg = float(np.linalg.norm(np.asarray(p1) - np.asarray(p0)))
    probe = Point(*(mid + n_dir * seg * 1e-4))
    for poly in region_polys.values():
        if poly.contains(probe):
            return poly
    raise SweepAttachmentNotFoundError("<polyline sweep>", f"no region contains {probe.wkt}")


def sweep_imprint_pass(occ_entities, sweeps, entities, point_tolerance):
    """Clip + imprint every sweep; emit synthetic __sweep/__sweepsrc entities.

    A second, sweeps-only BOP fragment over all dim-2 entity shapes with the
    clipped sweep rectangles as tool faces ("highest mesh order, last").
    Sub-faces inherit their entity's physical name (shapes are replaced by
    their Modified() pieces in place); pieces inside a sweep rectangle
    additionally get a synthetic dim-2 annotator entity, and edges of those
    pieces lying on the source segment get one dim-1 __sweepsrc entity.
    """
    from OCP.BOPAlgo import BOPAlgo_Builder
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakeWire,
    )
    from OCP.BRepGProp import BRepGProp
    from OCP.gp import gp_Pnt
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_ShapeEnum
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    from meshwell.cad_occ import OCCLabeledEntity

    if not sweeps:
        return occ_entities

    region_polys = final_region_polygons(entities)

    # ---- resolve every sweep side into rectangles -----------------------
    resolved = []  # (sweep, side, rect_polygon, p0, p1, n_dir)
    all_rects = []
    for sweep in sweeps:
        p0, p1 = resolve_attachment(sweep, entities, region_polys, point_tolerance)
        for side, thick in sweep.thickness.items():
            n_dir = side_normal(p0, p1, side, sweep, region_polys, point_tolerance)
            target = (
                region_polys[side]
                if side in region_polys
                else _polyline_target_region(p0, p1, n_dir, region_polys)
            )
            intervals = clip_sweep_side(p0, p1, n_dir, thick, target, point_tolerance)
            for rect in sweep_rectangles(p0, p1, n_dir, thick, intervals):
                for other_sweep, _os, other_rect, *_ in resolved:
                    if rect.intersection(other_rect).area > (10 * point_tolerance) ** 2:
                        raise SweepOverlapError(sweep.name, other_sweep.name)
                resolved.append((sweep, side, rect, p0, p1, n_dir))
                all_rects.append(rect)

    if not resolved:
        return occ_entities

    # ---- tool faces -----------------------------------------------------
    def _face_from_polygon(poly):
        wire = BRepBuilderAPI_MakeWire()
        coords = list(poly.exterior.coords)[:-1]
        for a, b in zip(coords, coords[1:] + coords[:1]):
            edge = BRepBuilderAPI_MakeEdge(
                gp_Pnt(a[0], a[1], 0.0), gp_Pnt(b[0], b[1], 0.0)
            ).Edge()
            wire.Add(edge)
        return BRepBuilderAPI_MakeFace(wire.Wire()).Face()

    tools = [_face_from_polygon(r) for r in all_rects]

    # ---- fragment -------------------------------------------------------
    builder = BOPAlgo_Builder()
    for ent in occ_entities:
        if ent.dim != 2:
            continue
        for shape in ent.shapes:
            builder.AddArgument(shape)
    for tool in tools:
        builder.AddArgument(tool)
    builder.SetFuzzyValue(point_tolerance)
    builder.Perform()

    def _pieces(shape):
        modified = builder.Modified(shape)
        if modified.IsEmpty() and not builder.IsDeleted(shape):
            return [shape]
        return list(modified)

    # replace each dim-2 entity's shapes by their pieces (faces only)
    for ent in occ_entities:
        if ent.dim != 2:
            continue
        new_shapes = []
        for shape in ent.shapes:
            for piece in _pieces(shape):
                if piece.ShapeType() == TopAbs_ShapeEnum.TopAbs_FACE:
                    new_shapes.append(piece)
                else:
                    exp = TopExp_Explorer(piece, TopAbs_FACE)
                    while exp.More():
                        new_shapes.append(exp.Current())
                        exp.Next()
        ent.shapes = new_shapes

    # ---- synthetic annotators ------------------------------------------
    def _face_centroid(face):
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, props)
        p = props.CentreOfMass()
        return np.array([p.X(), p.Y()])

    def _edge_midpoint(edge):
        props = GProp_GProps()
        BRepGProp.LinearProperties_s(edge, props)
        p = props.CentreOfMass()
        return np.array([p.X(), p.Y()])

    next_index = max((e.index for e in occ_entities), default=-1) + 1
    out = list(occ_entities)
    per_sweep_counter: dict = {}
    for sweep, side, rect, p0, p1, n_dir in resolved:
        i = per_sweep_counter.get((sweep.name, side), 0)
        source = LineString([p0, p1])
        band_faces = []
        for ent in occ_entities:
            if ent.dim != 2 or not ent.keep:
                continue
            for face in ent.shapes:
                c = _face_centroid(face)
                if rect.contains(Point(*c)):
                    band_faces.append(face)
        for face in band_faces:
            out.append(
                OCCLabeledEntity(
                    shapes=[face],
                    physical_name=(f"__sweep|{sweep.name}|{side}|{i}",),
                    index=next_index,
                    keep=True,
                    dim=2,
                    mesh_order=None,
                )
            )
            next_index += 1
            i += 1
            # source edges of this face
            src_edges = []
            exp = TopExp_Explorer(face, TopAbs_EDGE)
            while exp.More():
                edge = TopoDS.Edge_s(exp.Current())
                if source.distance(Point(*_edge_midpoint(edge))) < 10 * point_tolerance:
                    src_edges.append(edge)
                exp.Next()
            if src_edges:
                out.append(
                    OCCLabeledEntity(
                        shapes=src_edges,
                        physical_name=(f"__sweepsrc|{sweep.name}",),
                        index=next_index,
                        keep=True,
                        dim=1,
                        mesh_order=None,
                    )
                )
                next_index += 1
        per_sweep_counter[(sweep.name, side)] = i
    return out
