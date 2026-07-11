"""Shared utilities for geometries."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import gmsh
import numpy as np

if TYPE_CHECKING:
    from OCP.gp import gp_Pnt
    from OCP.TopoDS import TopoDS_Face, TopoDS_Shape, TopoDS_Wire


# --- Arc-detection heuristics -------------------------------------------------
# A run of vertices is accepted as a circular arc only if it is smooth and
# genuinely curved. These thresholds are heuristic; collecting them here gives
# the greedy detector (``_decompose_vertices_3d``), the closed-circle primitive,
# and the seam finder ONE shared definition instead of scattered bare literals.

# cos(60 deg): if any interior turn along a candidate run is sharper than 60
# degrees the run is treated as polygon corners (e.g. a rectangle), not an arc.
_ARC_MAX_TURN_COS = 0.5

# A least-squares circle fit whose radius exceeds this is a near-straight line
# masquerading as a huge-radius arc; reject it and emit line segments instead.
# Absolute, hence scale-limited: fine for the O(1)-O(1e3) coordinates meshwell
# targets; extreme-coordinate models would need this raised.
_ARC_MAX_RADIUS = 1e6

# ``|2 * signed triangle area|`` below this means three points are collinear, so
# no unique circle passes through them (guards the division in
# ``_three_point_circle_2d``). Absolute FP floor; area scales with length^2, so
# this is likewise scale-limited at extreme coordinate magnitudes.
_COLLINEAR_DET_EPS = 1e-30

# An edge shorter than ``point_tolerance * _DEGENERATE_EDGE_REL`` is treated as
# zero-length when measuring a turn angle (its direction is meaningless).
# Vertices are already deduplicated at ``point_tolerance`` upstream, so in
# practice this only catches floating-point-coincident survivors. Expressing it
# relative to ``point_tolerance`` keeps the guard correct at any coordinate
# scale -- a fixed 1e-6 would wrongly reject real edges in nanometre-scale
# models, and a fixed 1e-12 is inconsistent with the coarser grid.
_DEGENERATE_EDGE_REL = 1e-3


@dataclass
class DecompositionSegment:
    """Dataclass to represent a geometry segment (line or arc)."""

    points: list[tuple[float, float, float]]
    is_arc: bool
    center: tuple[float, float, float] | None = None
    radius: float | None = None
    # Set by canonicalize_ring_segments: the OFFSET canonical circle
    # ((cx, cy), R +/- eps) this arc must be emitted on. None -> legacy
    # 3-point emission (passthrough path only).
    canonical: tuple[tuple[float, float], float] | None = None


def _strip_consecutive_duplicates(
    vertices: list[tuple[float, float, float]],
    tolerance: float,
) -> list[tuple[float, float, float]]:
    """Collapse adjacent vertices coincident within ``tolerance``.

    Preserves the closing vertex (equal to the opening vertex) when the input
    is a closed polyline so downstream consumers still see a closed ring.
    """
    if len(vertices) < 2:
        return list(vertices)
    was_closed = vertices[0] == vertices[-1]
    tol_sq = tolerance * tolerance
    out = [vertices[0]]
    for v in vertices[1:]:
        prev = out[-1]
        dx = v[0] - prev[0]
        dy = v[1] - prev[1]
        dz = v[2] - prev[2]
        if dx * dx + dy * dy + dz * dz > tol_sq:
            out.append(v)
    if was_closed and len(out) >= 2 and out[0] != out[-1]:
        out.append(out[0])
    return out


def _find_canonical_seam(
    vertices: list[tuple[float, float, float]],
    point_tolerance: float = 1e-3,
    sharp_cos_threshold: float = _ARC_MAX_TURN_COS,
) -> int:
    """Return index at which to start a closed polyline so arc runs don't straddle the seam.

    Picks the vertex whose incoming/outgoing edges turn the most sharply
    (smallest cos(angle)). Ties broken by lexicographic order so the result
    is deterministic across rotated equivalent rings. If no corner is sharp
    enough, falls back to the lex-min vertex.
    """
    n = len(vertices) - 1  # last equals first in a closed ring
    if n < 3:
        return 0

    degenerate_edge = point_tolerance * _DEGENERATE_EDGE_REL
    best_idx = 0
    best_cos = 2.0
    best_key: tuple | None = None
    for i in range(n):
        prev = vertices[(i - 1) % n]
        cur = vertices[i]
        nxt = vertices[(i + 1) % n]
        v1x, v1y = cur[0] - prev[0], cur[1] - prev[1]
        v2x, v2y = nxt[0] - cur[0], nxt[1] - cur[1]
        n1 = (v1x * v1x + v1y * v1y) ** 0.5
        n2 = (v2x * v2x + v2y * v2y) ** 0.5
        if n1 < degenerate_edge or n2 < degenerate_edge:
            continue
        cos_a = (v1x * v2x + v1y * v2y) / (n1 * n2)
        key = (cos_a, tuple(round(c, 9) for c in cur))
        if cos_a < best_cos - 1e-9 or (
            abs(cos_a - best_cos) < 1e-9 and (best_key is None or key < best_key)
        ):
            best_cos = cos_a
            best_idx = i
            best_key = key

    if best_cos > sharp_cos_threshold:
        best_idx = min(range(n), key=lambda i: vertices[i])
    return best_idx


def _rotate_closed(
    vertices: list[tuple[float, float, float]], start_idx: int
) -> list[tuple[float, float, float]]:
    """Rotate a closed polyline so it starts at ``start_idx``.

    If ``vertices`` is not closed (first != last) it is rotated in place.
    """
    if start_idx == 0 or len(vertices) < 2:
        return list(vertices)
    closed = vertices[0] == vertices[-1]
    core = vertices[:-1] if closed else list(vertices)
    rotated = core[start_idx:] + core[:start_idx]
    if closed:
        rotated.append(rotated[0])
    return rotated


def decompose_vertices_2d(
    coords_2d: list[tuple[float, float]],
    z: float,
    point_tolerance: float,
    identify_arcs: bool = False,
    min_arc_points: int = 5,
    arc_tolerance: float = 1e-3,
) -> list["DecompositionSegment"]:
    """Decompose a 2D coordinate sequence into line and arc segments.

    This is the free-function form of ``GeometryEntity.decompose_vertices``.
    It lifts 2D coords to 3D (using ``z``) before delegating, so both the
    structured pipeline (which works with explicit z planes) and the
    legacy ``GeometryEntity`` callers (which carry z in the vertex tuples)
    share the SAME arc-detection logic. This is the canonical entry point
    for arc detection — keep all callers aligned on it so that cohort
    construction and unstructured pre-cut produce matching OCC edges.

    Args:
        coords_2d: List of (x, y) 2D coordinates.
        z: z-coordinate to attach to every vertex.
        point_tolerance: Tolerance for vertex deduplication and rounding.
        identify_arcs: Whether to attempt arc identification.
        min_arc_points: Minimum number of points to form an arc.
        arc_tolerance: Tolerance for circle fitting.

    Returns:
        List of DecompositionSegment objects (their ``points`` are 3D
        with the supplied ``z``).
    """
    vertices_3d = [(c[0], c[1], z) for c in coords_2d]
    return _decompose_vertices_3d(
        vertices_3d,
        point_tolerance=point_tolerance,
        identify_arcs=identify_arcs,
        min_arc_points=min_arc_points,
        arc_tolerance=arc_tolerance,
    )


def _decompose_vertices_3d(
    vertices: list[tuple[float, float, float]],
    point_tolerance: float,
    identify_arcs: bool = False,
    min_arc_points: int = 5,
    arc_tolerance: float = 1e-3,
) -> list["DecompositionSegment"]:
    """Free-function form of ``GeometryEntity.decompose_vertices``.

    Takes ``point_tolerance`` explicitly instead of reading it from a
    ``GeometryEntity`` instance, so it can be called from non-entity
    contexts (e.g., the structured EdgeRegistry).
    """
    vertices = _strip_consecutive_duplicates(list(vertices), point_tolerance)
    if not vertices or len(vertices) < 2:
        return []

    if (
        identify_arcs
        and len(vertices) >= max(min_arc_points + 1, 4)
        and vertices[0] == vertices[-1]
    ):
        seam = _find_canonical_seam(vertices, point_tolerance)
        vertices = _rotate_closed(vertices, seam)

    if not identify_arcs or len(vertices) < min_arc_points:
        # Fallback to simple line segments
        return [
            DecompositionSegment(points=[vertices[i], vertices[i + 1]], is_arc=False)
            for i in range(len(vertices) - 1)
        ]

    ndigits = max(0, int(-np.floor(np.log10(point_tolerance))))
    degenerate_edge = point_tolerance * _DEGENERATE_EDGE_REL

    # Up-front closed-circle primitive. If the whole closed ring fits ONE
    # circle within arc_tolerance, emit it directly as a single closed arc
    # and skip the greedy per-window scan. The emission layer already
    # special-cases closed circles (a 360-degree arc can't be one edge, so
    # it's split into two 180-degree arcs); detecting them here keeps the
    # two layers consistent and, crucially, keeps the greedy emitted-circle
    # gate from ever seeing a window that wraps toward closure -- where
    # start ~= end makes the 3-point circle ill-conditioned and the gate
    # would fragment the ring into an arc + chords, yielding an
    # untriangulatable face. A full-ring least-squares fit is the
    # best-conditioned case; non-circular closed rings (rounded rectangles,
    # ellipses) fail the fit and fall through to the greedy scan below.
    if vertices[0] == vertices[-1] and len(vertices) >= min_arc_points + 1:
        ring = np.array(vertices[:-1])
        c_center, c_radius, c_residual = fit_circle_2d(ring[:, :2])
        if c_residual <= arc_tolerance and c_radius < _ARC_MAX_RADIUS:
            c_dev = np.abs(
                np.hypot(ring[:, 0] - c_center[0], ring[:, 1] - c_center[1]) - c_radius
            ).max()
            # Same sharp-corner gate the greedy path applies (``valid_arc``):
            # a coarsely sampled ring (e.g. a hexagon) fits a circle with low
            # residual but turns too sharply at each vertex to be a genuine
            # arc. Without this the primitive would promote coarse polygonal
            # rings to circles that the greedy path (and the emission layer)
            # would otherwise keep as line segments, producing degenerate
            # faces. Ring is closed, so check the wrap-around corner too.
            m = len(ring)
            smooth = True
            for k in range(m):
                a = ring[(k - 1) % m][:2]
                b = ring[k][:2]
                cc = ring[(k + 1) % m][:2]
                v1 = b - a
                v2 = cc - b
                n1 = np.hypot(v1[0], v1[1])
                n2 = np.hypot(v2[0], v2[1])
                if (
                    n1 > degenerate_edge
                    and n2 > degenerate_edge
                    and float(np.dot(v1, v2) / (n1 * n2)) < _ARC_MAX_TURN_COS
                ):
                    smooth = False
                    break
            if c_dev <= arc_tolerance and smooth:
                return [
                    DecompositionSegment(
                        points=vertices,
                        is_arc=True,
                        center=(
                            round(c_center[0], ndigits),
                            round(c_center[1], ndigits),
                            vertices[0][2],
                        ),
                        radius=round(c_radius, ndigits),
                    )
                ]

    segments = []
    i = 0
    n = len(vertices)

    while i < n - 1:
        # Try to find an arc starting at i
        best_arc = None
        if i + min_arc_points <= n:
            for j in range(i + min_arc_points, n + 1):
                pts = np.array(vertices[i:j])
                # Ensure points are not collinear and can be fit to a circle
                # For simplicity, we assume the arc is in the XY plane
                center, radius, residual = fit_circle_2d(pts[:, :2])

                accepted = residual <= arc_tolerance and radius < _ARC_MAX_RADIUS
                if accepted:
                    # The RMSE above validates the least-squares fit, but
                    # emission interpolates only 3 samples (start, mid, end)
                    # -- or the quarter samples for a closed window. Gate on
                    # the MAX deviation of every window sample from both the
                    # fitted circle and (for open windows) the 3-point
                    # circle actually emitted: sagitta-starved windows
                    # amplify a half-grid midpoint error into radius errors
                    # far beyond arc_tolerance.
                    xy = pts[:, :2]
                    dev_fit = np.abs(
                        np.hypot(xy[:, 0] - center[0], xy[:, 1] - center[1]) - radius
                    ).max()
                    accepted = dev_fit <= arc_tolerance
                    window_closed = vertices[i] == vertices[j - 1]
                    if accepted and not window_closed:
                        mid_k = len(xy) // 2
                        emitted = _three_point_circle_2d(
                            tuple(xy[0]), tuple(xy[mid_k]), tuple(xy[-1])
                        )
                        if emitted is None:
                            accepted = False
                        else:
                            (ecx, ecy), er = emitted
                            dev_emit = np.abs(
                                np.hypot(xy[:, 0] - ecx, xy[:, 1] - ecy) - er
                            ).max()
                            accepted = dev_emit <= arc_tolerance
                if accepted:
                    # Ensure it's not a polygon with sharp corners (like a rectangle)
                    valid_arc = True
                    for k in range(1, len(pts) - 1):
                        v1 = pts[k][:2] - pts[k - 1][:2]
                        v2 = pts[k + 1][:2] - pts[k][:2]
                        n1 = np.linalg.norm(v1)
                        n2 = np.linalg.norm(v2)
                        if n1 > degenerate_edge and n2 > degenerate_edge:
                            cos_angle = np.dot(v1, v2) / (n1 * n2)
                            if (
                                cos_angle < _ARC_MAX_TURN_COS
                            ):  # turn sharper than 60 deg
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


def _arc_sense_ccw(center, p_start, p_mid, p_end) -> bool:
    """True iff the arc start->mid->end runs counterclockwise about center.

    Resolves the two-arc trim ambiguity by ANGULAR position of the
    midpoint -- NOT by distance projection, which is degenerate (the
    midpoint lies on the full circle for both trims; see the 3-point-form
    comment in _make_occ_wire_from_vertices).
    """
    two_pi = 2.0 * np.pi
    a0 = np.arctan2(p_start[1] - center[1], p_start[0] - center[0])
    am = np.arctan2(p_mid[1] - center[1], p_mid[0] - center[0])
    a1 = np.arctan2(p_end[1] - center[1], p_end[0] - center[0])
    sweep_ccw = (a1 - a0) % two_pi
    mid_rel = (am - a0) % two_pi
    if sweep_ccw == 0.0:
        return True
    return bool(mid_rel <= sweep_ccw)


def _project_to_circle(center, radius, p):
    dx, dy = p[0] - center[0], p[1] - center[1]
    d = float(np.hypot(dx, dy))
    if d == 0.0:
        return p
    return (center[0] + radius * dx / d, center[1] + radius * dy / d)


def _line_circle_junction(center, radius, p_junction, p_far):
    """Junction of line (p_far -> p_junction) with the circle, nearest p_junction.

    Tangent/degenerate cases return the tangency foot.
    """
    fx, fy = p_far
    dx, dy = p_junction[0] - fx, p_junction[1] - fy
    a = dx * dx + dy * dy
    if a == 0.0:
        return _project_to_circle(center, radius, p_junction)
    ex, ey = fx - center[0], fy - center[1]
    b = 2.0 * (dx * ex + dy * ey)
    c = ex * ex + ey * ey - radius * radius
    disc = b * b - 4.0 * a * c
    if disc <= 0.0:
        t = -(ex * dx + ey * dy) / a
        return _project_to_circle(center, radius, (fx + t * dx, fy + t * dy))
    sq = float(np.sqrt(disc))
    candidates = [
        (fx + t * dx, fy + t * dy) for t in ((-b - sq) / (2 * a), (-b + sq) / (2 * a))
    ]
    return min(
        candidates,
        key=lambda q: (q[0] - p_junction[0]) ** 2 + (q[1] - p_junction[1]) ** 2,
    )


def _circle_circle_junction(c1, r1, c2, r2, p_junction):
    """Intersection of two circles nearest p_junction.

    Midpoint of the two radial projections when they don't intersect
    (same-circle case included: both projections coincide).
    """
    dx, dy = c2[0] - c1[0], c2[1] - c1[1]
    d = float(np.hypot(dx, dy))
    if d == 0.0 or d > r1 + r2 or d < abs(r1 - r2):
        p1 = _project_to_circle(c1, r1, p_junction)
        p2 = _project_to_circle(c2, r2, p_junction)
        return (0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1]))
    a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
    h = float(np.sqrt(max(r1 * r1 - a * a, 0.0)))
    mx, my = c1[0] + a * dx / d, c1[1] + a * dy / d
    candidates = [
        (mx + h * dy / d, my - h * dx / d),
        (mx - h * dy / d, my + h * dx / d),
    ]
    return min(
        candidates,
        key=lambda q: (q[0] - p_junction[0]) ** 2 + (q[1] - p_junction[1]) ** 2,
    )


def _line_line_junction(a1, a2, b1, b2, p_fallback):
    """Intersection (miter point) of infinite lines a1->a2 and b1->b2.

    Return p_fallback when near-parallel (collinear offset edges keep their
    shared endpoint).
    """
    d1x, d1y = a2[0] - a1[0], a2[1] - a1[1]
    d2x, d2y = b2[0] - b1[0], b2[1] - b1[1]
    denom = d1x * d2y - d1y * d2x
    scale = max(abs(d1x), abs(d1y), abs(d2x), abs(d2y), 1e-300)
    if abs(denom) <= 1e-12 * scale * scale:
        return p_fallback
    t = ((b1[0] - a1[0]) * d2y - (b1[1] - a1[1]) * d2x) / denom
    return (a1[0] + t * d1x, a1[1] + t * d1y)


def _offset_point(p_prev, p, eps):
    """Shift ``p`` by ``eps`` along the right-of-travel normal of the segment.

    For segment p_prev -> p: with OGC ring orientation (CCW exterior, CW
    holes) material is LEFT of travel, so right-of-travel is outward.
    """
    dx, dy = p[0] - p_prev[0], p[1] - p_prev[1]
    n = float(np.hypot(dx, dy))
    if n == 0.0 or eps == 0.0:
        return (p[0], p[1])
    return (p[0] + eps * dy / n, p[1] - eps * dx / n)


def _promote_chord_runs(segments, registry, *, tolerance):
    """Replace line runs whose vertices all track one canonical circle with an arc.

    Closes the classification cliffs (min_arc_points remnants, detector
    gate rejections) where one entity keeps chords -- midpoint sag
    R*theta^2/8, far above any boolean clearance -- against another
    entity's arc. Keys on vertex proximity to a registered circle; never
    re-fits. Single-segment runs (no interior vertex) stay untouched:
    two points near a circle are indistinguishable from a straight edge
    grazing it (the probe reports them).
    """
    if not segments or registry is None:
        return segments

    # Rotate so index 0 is an arc when one exists; line runs then never
    # wrap the list seam. (All-line rings may split a seam-wrapping span
    # into two promoted arcs on the same circle -- harmless: same-circle
    # junctions are exact.)
    first_arc = next((i for i, s in enumerate(segments) if s.is_arc), None)
    if first_arc:
        segments = segments[first_arc:] + segments[:first_arc]

    out: list[DecompositionSegment] = []
    i, n = 0, len(segments)
    while i < n:
        if segments[i].is_arc:
            out.append(segments[i])
            i += 1
            continue
        j = i
        while j < n and not segments[j].is_arc:
            j += 1
        run = segments[i:j]
        run_pts = [s.points[0] for s in run] + [run[-1].points[-1]]
        # Greedy sub-run scan INSIDE the line run: a remnant arc can be
        # embedded between genuine straight edges, so whole-run matching
        # would be poisoned by the corners. Monotone: adding a vertex can
        # only raise the max deviation, so break-on-first-miss finds the
        # longest matching window exactly.
        k, m = 0, len(run_pts)
        while k < m - 1:
            hit_k, best_end = None, None
            end = k + 2  # >= 3 vertices: require an interior vertex
            while end < m:
                h = registry.match_chord_run(
                    [(p[0], p[1]) for p in run_pts[k : end + 1]],
                    tolerance=tolerance,
                )
                if h is None:
                    break
                hit_k, best_end = h, end
                end += 1
            if hit_k is not None:
                (cx, cy), r = hit_k
                z = run_pts[k][2]
                out.append(
                    DecompositionSegment(
                        points=run_pts[k : best_end + 1],
                        is_arc=True,
                        center=(cx, cy, z),
                        radius=r,
                    )
                )
                k = best_end
            else:
                out.append(run[k])
                k += 1
        i = j
    return out


def canonicalize_ring_segments(segments, registry, *, eps, match_tolerance, slack):
    """Canonicalize one OGC-oriented ring: promote, assign circles, offset, re-join.

    OGC orientation (CCW exterior, CW holes) puts material LEFT of
    travel: lines offset eps to the RIGHT of travel; an arc whose center
    is left of travel offsets to R+eps, else R-eps. Junction vertices are
    re-derived as intersections of the ADJACENT OFFSET primitives (miter
    for line/line, root selection nearest the original vertex otherwise),
    so every emitted vertex lies exactly on both incident curves.
    """
    segments = _promote_chord_runs(segments, registry, tolerance=match_tolerance)
    n = len(segments)
    if n == 0:
        return segments

    # Assign offset canonical circles to arcs.
    for seg in segments:
        if not seg.is_arc or seg.center is None:
            continue
        hit = None
        if registry is not None:
            hit = registry.lookup(
                (seg.center[0], seg.center[1]), seg.radius, slack=slack
            )
        (cx, cy), r = (
            hit if hit is not None else ((seg.center[0], seg.center[1]), seg.radius)
        )
        p0, p1 = seg.points[0], seg.points[1]
        cross = (p1[0] - p0[0]) * (cy - p0[1]) - (p1[1] - p0[1]) * (cx - p0[0])
        seg.canonical = ((cx, cy), r + eps if cross > 0 else r - eps)

    if eps != 0.0:
        # Offset line endpoints along their own right-of-travel normals.
        for seg in segments:
            if seg.is_arc:
                continue
            a, b = seg.points[0], seg.points[-1]
            z = a[2]
            oa = _offset_point((2 * a[0] - b[0], 2 * a[1] - b[1]), a, eps)
            ob = _offset_point(a, b, eps)
            seg.points = [(oa[0], oa[1], z), (ob[0], ob[1], z)]

    # Re-derive every junction as the exact intersection of the two
    # adjacent offset primitives (arcs keep their original points as trim
    # references; only their endpoints move here).
    if n > 1 or (n == 1 and not segments[0].is_arc):
        for i, seg in enumerate(segments):
            nxt = segments[(i + 1) % n]
            p_orig = nxt.points[0]
            z = p_orig[2]
            if seg.is_arc and seg.canonical and nxt.is_arc and nxt.canonical:
                (c1, r1), (c2, r2) = seg.canonical, nxt.canonical
                j = _circle_circle_junction(c1, r1, c2, r2, (p_orig[0], p_orig[1]))
            elif seg.is_arc and seg.canonical:
                (c1, r1) = seg.canonical
                j = _line_circle_junction(
                    c1,
                    r1,
                    (nxt.points[0][0], nxt.points[0][1]),
                    (nxt.points[-1][0], nxt.points[-1][1]),
                )
            elif nxt.is_arc and nxt.canonical:
                (c2, r2) = nxt.canonical
                j = _line_circle_junction(
                    c2,
                    r2,
                    (seg.points[-1][0], seg.points[-1][1]),
                    (seg.points[0][0], seg.points[0][1]),
                )
            elif not seg.is_arc and not nxt.is_arc:
                j = _line_line_junction(
                    (seg.points[0][0], seg.points[0][1]),
                    (seg.points[-1][0], seg.points[-1][1]),
                    (nxt.points[0][0], nxt.points[0][1]),
                    (nxt.points[-1][0], nxt.points[-1][1]),
                    (seg.points[-1][0], seg.points[-1][1]),
                )
            else:
                continue  # arc without canonical: legacy passthrough
            seg.points[-1] = (j[0], j[1], z)
            nxt.points[0] = (j[0], j[1], z)
    return segments


def _three_point_circle_2d(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
) -> tuple[tuple[float, float], float] | None:
    """Center and radius of the circle through three 2D points.

    This is the circle the downstream arc emitters actually build
    (gmsh ``addCircleArc(..., center=False)`` / OCC ``GC_MakeArcOfCircle``
    through start, mid-sample, end). Returns ``None`` for (near-)collinear
    points.
    """
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < _COLLINEAR_DET_EPS:
        return None
    ux = (
        (ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by)
    ) / d
    uy = (
        (ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax)
    ) / d
    return (ux, uy), float(np.hypot(ax - ux, ay - uy))


class GeometryEntity:
    """Base class for geometry entities that create GMSH geometry directly.

    Provides shared functionality for point deduplication and coordinate parsing
    to ensure consistent geometry creation across PolyLine, PolySurface, and PolyPrism.
    """

    # Pipeline-level parameters, stamped by cad_common.apply_arc_params /
    # CAD_OCC.process_entities. Class defaults keep standalone entity use
    # working (no arcs, no offset, no registry).
    identify_arcs = False
    min_arc_points = 5
    arc_tolerance = 1e-3
    circle_registry = None
    perturbation = 0.0

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
        min_arc_points: int = 5,
        arc_tolerance: float = 1e-3,
    ) -> int:
        """Create a GMSH surface from vertex coordinates with optional arc identification."""
        vertices = _strip_consecutive_duplicates(list(vertices), self.point_tolerance)
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
        min_arc_points: int = 5,
        arc_tolerance: float = 1e-3,
    ) -> list[DecompositionSegment]:
        """Decompose a sequence of vertices into line segments and circular arcs.

        Thin wrapper around the free function ``_decompose_vertices_3d``
        — kept as a method for backward compatibility. New 2D callers
        should prefer ``decompose_vertices_2d`` which lifts (x, y) +
        z into the 3D shape this function consumes.

        Args:
            vertices: List of (x, y, z) coordinates
            identify_arcs: Whether to attempt arc identification
            min_arc_points: Minimum number of points to form an arc
            arc_tolerance: Tolerance for circle fitting

        Returns:
            List of DecompositionSegment objects
        """
        return _decompose_vertices_3d(
            vertices,
            point_tolerance=self.point_tolerance,
            identify_arcs=identify_arcs,
            min_arc_points=min_arc_points,
            arc_tolerance=arc_tolerance,
        )

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
        min_arc_points: int = 5,
        arc_tolerance: float = 1e-3,
    ) -> TopoDS_Wire:
        """Create an OCC wire from vertex coordinates with optional arc identification.

        Vertices and edges are built fresh per call; cross-entity TShape
        sharing is delegated to ``BOPAlgo_Builder``'s fragment pass with
        its fuzzy tolerance, which is how gmsh's own OCC kernel handles
        the same scenario.
        """
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire

        vertices = _strip_consecutive_duplicates(list(vertices), self.point_tolerance)
        if (
            not identify_arcs
            and self.circle_registry is None
            and self.perturbation == 0.0
        ):
            points = self._make_occ_points(vertices)
            wire_builder = BRepBuilderAPI_MakeWire()
            for i in range(len(points) - 1):
                wire_builder.Add(
                    BRepBuilderAPI_MakeEdge(points[i], points[i + 1]).Edge()
                )
            return wire_builder.Wire()

        from OCP.GC import GC_MakeArcOfCircle
        from OCP.gp import gp_Pnt

        ndigits = max(0, int(-np.floor(np.log10(self.point_tolerance))))

        def _key(coords):
            return tuple(round(c, ndigits) for c in coords)

        # Deduplicate at the point_tolerance grid WITHOUT moving surviving
        # coordinates. Rounding actual coordinates would undo the
        # sub-tolerance perturbation buffer applied by
        # cad_common.prepare_entities (buffered coords sit ~perturbation off
        # grid vertices and round straight back), silently disabling the
        # pre-cut overlap strategy for arc-identified entities. Only the
        # dedup KEY is quantized -- the same scheme
        # _add_point_with_tolerance uses on the gmsh side.
        deduped: list[tuple[float, float, float]] = []
        last_key: tuple[float, float, float] | None = None
        for v in vertices:
            k = _key(v)
            if last_key is not None and k == last_key:
                continue
            deduped.append(v)
            last_key = k
        was_closed = vertices[0] == vertices[-1]
        if was_closed and len(deduped) >= 2 and deduped[0] != deduped[-1]:
            deduped.append(deduped[0])
        vertices = deduped

        segments = self.decompose_vertices(
            vertices,
            identify_arcs=identify_arcs,
            min_arc_points=min_arc_points,
            arc_tolerance=arc_tolerance,
        )

        if self.circle_registry is not None or self.perturbation != 0.0:
            segments = canonicalize_ring_segments(
                segments,
                self.circle_registry,
                eps=self.perturbation,
                match_tolerance=arc_tolerance + self.point_tolerance,
                slack=self.point_tolerance,
            )

        wire_builder = BRepBuilderAPI_MakeWire()

        for seg in segments:
            if seg.is_arc:
                mid_idx = len(seg.points) // 2
                is_closed = seg.points[0] == seg.points[-1]
                # Drop degenerate arcs whose endpoints collapse under the
                # quantization grid -- BRepBuilderAPI_MakeEdge raises
                # StdFail_NotDone on a (near-)zero-length edge.
                if not is_closed and _key(seg.points[0]) == _key(seg.points[-1]):
                    continue

                if seg.canonical is not None:
                    from OCP.gp import gp_Ax2, gp_Circ, gp_Dir

                    (ccx, ccy), cr = seg.canonical
                    z = seg.points[0][2]
                    circ = gp_Circ(
                        gp_Ax2(gp_Pnt(ccx, ccy, z), gp_Dir(0.0, 0.0, 1.0)), cr
                    )

                    def _on_circle(p, _ccx=ccx, _ccy=ccy, _cr=cr, _z=z):
                        q = _project_to_circle((_ccx, _ccy), _cr, (p[0], p[1]))
                        return gp_Pnt(q[0], q[1], _z)

                    if is_closed:
                        quarter = len(seg.points) // 4
                        three_q = (len(seg.points) * 3) // 4
                        p0, pm = _on_circle(seg.points[0]), _on_circle(
                            seg.points[mid_idx]
                        )
                        s1 = _arc_sense_ccw(
                            (ccx, ccy),
                            (p0.X(), p0.Y()),
                            seg.points[quarter][:2],
                            (pm.X(), pm.Y()),
                        )
                        s2 = _arc_sense_ccw(
                            (ccx, ccy),
                            (pm.X(), pm.Y()),
                            seg.points[three_q][:2],
                            (p0.X(), p0.Y()),
                        )
                        wire_builder.Add(
                            BRepBuilderAPI_MakeEdge(
                                GC_MakeArcOfCircle(circ, p0, pm, s1).Value()
                            ).Edge()
                        )
                        edge = BRepBuilderAPI_MakeEdge(
                            GC_MakeArcOfCircle(circ, pm, p0, s2).Value()
                        ).Edge()
                    else:
                        sense = _arc_sense_ccw(
                            (ccx, ccy),
                            seg.points[0][:2],
                            seg.points[mid_idx][:2],
                            seg.points[-1][:2],
                        )
                        edge = BRepBuilderAPI_MakeEdge(
                            GC_MakeArcOfCircle(
                                circ,
                                gp_Pnt(*seg.points[0]),
                                gp_Pnt(*seg.points[-1]),
                                sense,
                            ).Value()
                        ).Edge()
                else:
                    # Legacy passthrough: 3-point arcs through the vertices
                    # (unchanged code from today, including its comment).
                    p_start = gp_Pnt(*seg.points[0])
                    p_mid = gp_Pnt(*seg.points[mid_idx])
                    p_end = gp_Pnt(*seg.points[-1])

                    if is_closed:
                        # Full circle: split into two 180-degree arcs.
                        quarter_idx = len(seg.points) // 4
                        three_quarter_idx = (len(seg.points) * 3) // 4
                        p1 = gp_Pnt(*seg.points[quarter_idx])
                        p3 = gp_Pnt(*seg.points[three_quarter_idx])
                        arc_geom1 = GC_MakeArcOfCircle(p_start, p1, p_mid).Value()
                        edge1 = BRepBuilderAPI_MakeEdge(arc_geom1).Edge()
                        arc_geom2 = GC_MakeArcOfCircle(p_mid, p3, p_end).Value()
                        edge = BRepBuilderAPI_MakeEdge(arc_geom2).Edge()
                        wire_builder.Add(edge1)
                    else:
                        # Three-point arc form: GC_MakeArcOfCircle(p_start, p_mid,
                        # p_end) builds the unique arc passing through all three
                        # points, in that order. This avoids the CCW-vs-CW
                        # ambiguity of the (circle, p_start, p_end, sense) form
                        # -- the latter cannot be disambiguated by projecting
                        # p_mid, since p_mid lies on the underlying full circle
                        # and projection ignores the parametric trim of the arc
                        # (gives LowerDistance == 0 for both senses).
                        arc_geom = GC_MakeArcOfCircle(p_start, p_mid, p_end).Value()
                        edge = BRepBuilderAPI_MakeEdge(arc_geom).Edge()
            else:
                # Consecutive points that collapse to the same grid key are
                # already removed by the dedup pass above; guard anyway so a
                # zero-length segment can never reach MakeEdge.
                if _key(seg.points[0]) == _key(seg.points[1]):
                    continue
                edge = BRepBuilderAPI_MakeEdge(
                    gp_Pnt(*seg.points[0]), gp_Pnt(*seg.points[1])
                ).Edge()
            wire_builder.Add(edge)
        return wire_builder.Wire()

    def _make_occ_face_from_vertices(
        self,
        vertices: list[tuple[float, float, float]],
        identify_arcs: bool = False,
        min_arc_points: int = 5,
        arc_tolerance: float = 1e-3,
    ) -> TopoDS_Face:
        """Create an OCC face from a wire built from the given vertices."""
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
        min_arc_points: int = 5,
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

    def instanciate(self, cad_model: Any | None = None) -> list[tuple[int, int]]:
        """Create GMSH geometry. To be implemented by subclasses.

        Args:
            cad_model: Any model (kept for interface compatibility)

        Returns:
            List of (dimension, tag) tuples representing created entities

        """
        raise NotImplementedError("Subclasses must implement instanciate method")

    def instanciate_occ(self) -> TopoDS_Shape:
        """Create OCC geometry. To be implemented by subclasses."""
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
        """Apply rotation and translation to OCC shape.

        Returns ``shape`` unchanged when neither a rotation nor a
        translation is configured -- ``BRepBuilderAPI_Transform`` mints
        fresh TShapes even for an identity transform.
        """
        if shape is None:
            return None
        if self.rotation_angle == 0 and not self.translation:
            return shape

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
