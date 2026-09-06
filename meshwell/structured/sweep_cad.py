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
from shapely.geometry import Point

from meshwell.structured.exceptions import (
    SweepAttachmentNotFoundError,
    SweepCurvedSourceError,
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
        if merged.geom_type != "LineString" or merged.is_empty:
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
