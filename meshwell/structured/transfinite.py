"""Mesh-stage transfinite grids on cohort regions (``TransfiniteResolutionSpec``).

Runs in the cohort pre-2D hook, before ``freeze_lateral_mesh`` calls
``generate(1)``:

1. Collect the column seed faces (regions) of every kept slab whose
   physical carries a ``TransfiniteResolutionSpec``.
2. Validate each region: a single boundary loop with exactly four corners,
   found from curve end tangents (smoothly joined arcs are not corners); the
   curves between corners form four sides (a side may chain several curves
   where a neighbour creates a T-junction).
3. Classify sides long/short per physical by propagation over shared curves.
4. Solve node counts: every curve is one variable shared by the regions
   touching it; periodic chains (stepped cohorts: a seed-face curve that is
   the periodic copy of a lower lateral face's bottom curve) are merged into
   their root. Counts only go up until opposite sides match (finest wins).
5. Emit ``setTransfiniteCurve`` / ``setTransfiniteSurface`` (with corners).

gmsh silently falls back to an unstructured mesh when a surface's
transfinite constraints are inconsistent, so :func:`validate_transfinite_grids`
checks every region's 2D mesh afterwards (pre-3D hook).
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass

import gmsh
import numpy as np

from meshwell.structured.exceptions import (
    StructuredError,
    StructuredTransfiniteGridError,
    StructuredTransfiniteShapeError,
    StructuredTransfiniteSolverError,
)
from meshwell.structured.types import ShapeKey, SlabMeta

logger = logging.getLogger(__name__)

_MAX_SOLVER_ITERATIONS = 100_000
_OTHER = {"long": "short", "short": "long", "both": "both"}


@dataclass
class _Region:
    """One transfinite region (a cohort column seed face)."""

    face: int
    corners: list[int]  # 4 corner point tags, in loop order
    sides: list[list[int]]  # 4 sides, each a list of curve tags


@dataclass(frozen=True)
class TransfiniteGrid:
    """Expected structured grid of one region, checked after 2D meshing."""

    face: int
    n_a: int  # cells along sides 0 / 2
    n_b: int  # cells along sides 1 / 3


# ---------------------------------------------------------------------------
# gmsh geometry helpers
# ---------------------------------------------------------------------------


def _curve_ends(c: int) -> tuple[int, int]:
    """End points ``(at lower parameter, at upper parameter)``.

    Matched geometrically: ``getBoundary(oriented=True)`` follows the OCC
    edge orientation, which can be opposite to the parametrization.
    """
    pts = [abs(t) for _, t in gmsh.model.getBoundary([(1, c)], oriented=False)]
    if len(pts) == 1:
        return pts[0], pts[0]
    lo, _hi = gmsh.model.getParametrizationBounds(1, c)
    x0 = np.asarray(gmsh.model.getValue(1, c, [lo[0]]))
    d = [np.linalg.norm(np.asarray(gmsh.model.getValue(0, p, [])) - x0) for p in pts]
    return (pts[0], pts[1]) if d[0] <= d[1] else (pts[1], pts[0])


def _curve_length(c: int) -> float:
    return float(gmsh.model.occ.getMass(1, c))


def _tangent(c: int, at_start: bool, forward: bool) -> np.ndarray:
    """Unit tangent in traversal direction at the traversal start or end."""
    lo, hi = gmsh.model.getParametrizationBounds(1, c)
    t = lo[0] if (at_start == forward) else hi[0]
    d = np.asarray(gmsh.model.getDerivative(1, c, [t]))
    d = d if forward else -d
    return d / np.linalg.norm(d)


def _face_bbox_xy(face: int) -> tuple[float, float, float, float]:
    xmin, ymin, _z0, xmax, ymax, _z1 = gmsh.model.getBoundingBox(2, face)
    return (xmin, ymin, xmax, ymax)


def _region(face: int, physical: str, corner_tol: float) -> _Region:
    """Validate ``face`` as a four-sided single loop and split it into sides."""

    def _fail(reason: str) -> StructuredTransfiniteShapeError:
        return StructuredTransfiniteShapeError(
            physical, face, reason, bbox=_face_bbox_xy(face)
        )

    curves = [abs(t) for _, t in gmsh.model.getBoundary([(2, face)], oriented=False)]
    ends = {c: _curve_ends(c) for c in curves}
    a, b = ends[curves[0]]
    loop = [(curves[0], a, b, True)]
    left = set(curves[1:])
    while left:
        tail = loop[-1][2]
        if tail == loop[0][1]:
            raise _fail("has more than one boundary loop (a hole or an island)")
        nxt = next((c for c in left if tail in ends[c]), None)
        if nxt is None:
            raise _fail("has a boundary that is not a single closed loop")
        a, b = ends[nxt]
        loop.append((nxt, a, b, True) if a == tail else (nxt, b, a, False))
        left.remove(nxt)
    if loop[-1][2] != loop[0][1]:
        raise _fail("has a boundary that is not a single closed loop")

    corners = []
    for i in range(len(loop)):
        cp, _, _, fp = loop[i - 1]
        cn, _, _, fn = loop[i]
        d0 = _tangent(cp, at_start=False, forward=fp)
        d1 = _tangent(cn, at_start=True, forward=fn)
        turn = math.degrees(math.acos(float(np.clip(np.dot(d0, d1), -1.0, 1.0))))
        if turn > corner_tol:
            corners.append(i)
    if len(corners) != 4:
        raise _fail(f"has {len(corners)} corners (transfinite needs exactly 4)")

    sides = []
    for k in range(4):
        i, j = corners[k], corners[(k + 1) % 4]
        idx = list(range(i, j)) if i < j else list(range(i, len(loop))) + list(range(j))
        sides.append([loop[m][0] for m in idx])
    return _Region(face=face, corners=[loop[i][1] for i in corners], sides=sides)


def _pieces(faces: set[int], regions: dict[int, _Region]) -> list[set[int]]:
    """Group faces into connected pieces (faces sharing a boundary curve)."""
    curves = {f: {c for side in regions[f].sides for c in side} for f in faces}
    pieces: list[set[int]] = []
    left = set(faces)
    while left:
        stack, piece = [min(left)], set()
        left.discard(stack[0])
        while stack:
            f = stack.pop()
            piece.add(f)
            nbrs = {g for g in left if curves[g] & curves[f]}
            left -= nbrs
            stack.extend(sorted(nbrs))
        pieces.append(piece)
    return pieces


def _propagated_classes(
    regions: dict[int, _Region], faces: set[int], physical: str
) -> dict[int, list[str]]:
    """Region -> classes (``long``/``short``/``both``) of its four sides.

    In each connected piece the largest region takes its longer opposite
    pair as long; classes then spread breadth-first across shared curves.
    A conflict marks the region ``both`` (finer spacing on all sides).
    """
    side_of_curve: dict[int, list[tuple[int, int]]] = {}
    for f in faces:
        for k, side in enumerate(regions[f].sides):
            for c in side:
                side_of_curve.setdefault(c, []).append((f, k))
    out: dict[int, list[str]] = {}
    for piece in _pieces(faces, regions):
        seed = max(piece, key=lambda f: (gmsh.model.occ.getMass(2, f), -f))
        lens = [sum(_curve_length(c) for c in s) for s in regions[seed].sides]
        out[seed] = (
            ["long", "short", "long", "short"]
            if lens[0] + lens[2] >= lens[1] + lens[3]
            else ["short", "long", "short", "long"]
        )
        queue = deque([seed])
        while queue:
            f = queue.popleft()
            for k, side in enumerate(regions[f].sides):
                cls = out[f][k]
                other = _OTHER[cls]
                for c in side:
                    for g, kg in side_of_curve.get(c, []):
                        if g == f or g not in piece:
                            continue
                        want = (
                            [cls, other, cls, other]
                            if kg % 2 == 0
                            else [other, cls, other, cls]
                        )
                        if g not in out:
                            out[g] = want
                            queue.append(g)
                        elif out[g] != want and "both" not in out[g]:
                            logger.info(
                                "TransfiniteResolutionSpec on %r: conflicting "
                                "long/short classes on surface %s; using the "
                                "finer spacing on all its sides",
                                physical,
                                g,
                            )
                            out[g] = ["both"] * 4
                            queue.append(g)
    return out


def _periodic_top_to_bot(
    slab_meta: dict[ShapeKey, SlabMeta], face_tag_by_key: dict[ShapeKey, int]
) -> dict[int, int]:
    """Lateral-face top curve -> bottom curve, paired as ``freeze_lateral_mesh`` does."""
    from meshwell.structured.wedge import (
        _classify_lateral_face_edges,
        _face_centroid_z,
    )

    out: dict[int, int] = {}
    for meta in slab_meta.values():
        if not meta.keep:
            continue
        bot = face_tag_by_key.get(meta.bot_face_key)
        top = face_tag_by_key.get(meta.top_face_key)
        if bot is None or top is None:
            continue
        z_bot, z_top = _face_centroid_z(bot), _face_centroid_z(top)
        for fk in meta.lateral_face_keys:
            tag = face_tag_by_key.get(fk)
            if tag is None:
                continue
            b, t, _ = _classify_lateral_face_edges(tag, z_bot, z_top)
            if b is not None and t is not None:
                out.setdefault(t, b)
    return out


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _names(meta: SlabMeta) -> tuple[str, ...]:
    return (
        (meta.physical_name,)
        if isinstance(meta.physical_name, str)
        else tuple(meta.physical_name)
    )


def transfinite_specs(resolution_specs: dict | None) -> dict[str, object]:
    """``{physical: TransfiniteResolutionSpec}``; raises on duplicates per name."""
    from meshwell.resolution import TransfiniteResolutionSpec

    out: dict[str, object] = {}
    for name, specs in (resolution_specs or {}).items():
        found = [s for s in specs if isinstance(s, TransfiniteResolutionSpec)]
        if len(found) > 1:
            raise StructuredError(
                f"physical_name {name!r} has {len(found)} "
                "TransfiniteResolutionSpec entries; expected at most 1."
            )
        if found:
            out[name] = found[0]
    return out


def transfinite_cohort_columns(
    slab_meta: dict[ShapeKey, SlabMeta],
    face_tag_by_key: dict[ShapeKey, int],
    resolution_specs: dict | None,
) -> list[TransfiniteGrid]:
    """Set transfinite curves / surfaces on the regions of transfinite physicals.

    Must run before ``generate(1)`` (i.e. before ``freeze_lateral_mesh``).

    Returns:
        The expected grid of every region, for :func:`validate_transfinite_grids`.

    Raises:
        StructuredError: a spec targets a physical with no kept cohort slab.
        StructuredTransfiniteShapeError: a region is not a four-sided loop, or
            one of its curves belongs to a structured sweep.
        StructuredTransfiniteSolverError: the node counts cannot be balanced.
    """
    from meshwell.structured.wedge import column_seed_faces

    specs = transfinite_specs(resolution_specs)
    if not specs:
        return []

    cohort_names = {n for m in slab_meta.values() if m.keep for n in _names(m)}
    missing = sorted(set(specs) - cohort_names)
    if missing:
        raise StructuredError(
            f"TransfiniteResolutionSpec targets {missing}, which are not "
            "(kept) structured cohort physicals; the spec applies to "
            "structured PolyPrism physical names only."
        )

    # physical -> seed faces it owns (area-filtered like RecombineResolutionSpec).
    seeds = column_seed_faces(slab_meta, face_tag_by_key)
    owned: dict[str, set[int]] = {}
    for key, seed in seeds.items():
        for name in _names(slab_meta[key]):
            spec = specs.get(name)
            if spec is None:
                continue
            if not (spec.min_mass < gmsh.model.occ.getMass(2, seed) < spec.max_mass):
                continue
            owned.setdefault(name, set()).add(seed)
    if not owned:
        return []

    # Validate regions; the corner tolerance is the smallest over its specs.
    corner_tol: dict[int, tuple[float, str]] = {}
    for name, faces in owned.items():
        for f in faces:
            tol = specs[name].corner_angle_tol
            if f not in corner_tol or tol < corner_tol[f][0]:
                corner_tol[f] = (tol, name)
    regions = {
        f: _region(f, name, tol) for f, (tol, name) in sorted(corner_tol.items())
    }

    sweep_curves = _sweep_curve_tags()
    for f, reg in regions.items():
        if sweep_curves & {c for side in reg.sides for c in side}:
            raise StructuredTransfiniteShapeError(
                corner_tol[f][1],
                f,
                "has a boundary curve owned by a structured sweep",
                bbox=_face_bbox_xy(f),
            )

    # Target spacing per region side: finest over the physicals owning it.
    h_side = {f: [math.inf] * 4 for f in regions}
    for name, faces in sorted(owned.items()):
        spec = specs[name]
        hs = {
            "long": spec.long_axis,
            "short": spec.short_axis,
            "both": min(spec.long_axis, spec.short_axis),
        }
        for f, cls in _propagated_classes(regions, faces, name).items():
            h_side[f] = [min(h, hs[c]) for h, c in zip(h_side[f], cls, strict=True)]

    # Periodic chains: solve on the chain root.
    top_to_bot = _periodic_top_to_bot(slab_meta, face_tag_by_key)

    def _chain(c: int) -> list[int]:
        chain = [c]
        while chain[-1] in top_to_bot and top_to_bot[chain[-1]] not in chain:
            chain.append(top_to_bot[chain[-1]])
        return chain

    root = {c: _chain(c)[-1] for reg in regions.values() for s in reg.sides for c in s}
    rsides = {
        f: [[root[c] for c in s] for s in reg.sides] for f, reg in regions.items()
    }
    length = {r: _curve_length(r) for r in set(root.values())}

    n: dict[int, int] = {}
    for f, sides in rsides.items():
        for side, h in zip(sides, h_side[f], strict=True):
            for r in side:
                n[r] = max(n.get(r, 1), max(1, round(length[r] / h)))
    for _ in range(_MAX_SOLVER_ITERATIONS):
        changed = False
        for f in sorted(rsides):
            sides = rsides[f]
            for a, b in ((sides[0], sides[2]), (sides[1], sides[3])):
                na, nb = sum(n[r] for r in a), sum(n[r] for r in b)
                if na != nb:
                    small = a if na < nb else b
                    r = max(small, key=lambda r: (length[r] / n[r], -r))
                    n[r] += 1
                    changed = True
        if not changed:
            break
    else:
        raise StructuredTransfiniteSolverError(_MAX_SOLVER_ITERATIONS)

    # Emit: every curve of a chain gets its root's count (the root masters
    # the periodic copies, so it must carry the count even when it belongs
    # to a non-transfinite column).
    emitted: set[int] = set()
    for c in root:
        for member in _chain(c):
            if member not in emitted:
                emitted.add(member)
                gmsh.model.mesh.setTransfiniteCurve(member, n[root[c]] + 1)
    grids = []
    for f, reg in regions.items():
        gmsh.model.mesh.setTransfiniteSurface(f, "Left", reg.corners)
        grids.append(
            TransfiniteGrid(
                face=f,
                n_a=sum(n[r] for r in rsides[f][0]),
                n_b=sum(n[r] for r in rsides[f][1]),
            )
        )
    logger.debug(
        "transfinite_cohort_columns: %s",
        {g.face: (g.n_a, g.n_b) for g in grids},
    )
    return grids


def _sweep_curve_tags() -> set[int]:
    out: set[int] = set()
    for _dim, gtag in gmsh.model.getPhysicalGroups(2):
        if gmsh.model.getPhysicalName(2, gtag).startswith("__sweep|"):
            for ftag in gmsh.model.getEntitiesForPhysicalGroup(2, gtag):
                for _d, ct in gmsh.model.getBoundary([(2, int(ftag))], oriented=False):
                    out.add(abs(int(ct)))
    return out


def validate_transfinite_grids(grids: list[TransfiniteGrid]) -> None:
    """Raise if gmsh did not mesh a region with its requested structured grid.

    Call after ``generate(2)``. A transfinite region holds ``n_a * n_b`` quads
    (recombined) or ``2 * n_a * n_b`` triangles, and nothing else.
    """
    for g in grids:
        types, elems, _ = gmsh.model.mesh.getElements(2, g.face)
        counts = {
            gmsh.model.mesh.getElementProperties(int(t))[0]: len(e)
            for t, e in zip(types, elems, strict=True)
        }
        n_tri = sum(v for k, v in counts.items() if k.startswith("Triangle"))
        n_quad = sum(v for k, v in counts.items() if k.startswith("Quadrilateral"))
        cells = g.n_a * g.n_b
        if (n_quad, n_tri) not in ((cells, 0), (0, 2 * cells)):
            raise StructuredTransfiniteGridError(
                g.face,
                expected=(
                    f"{g.n_a} x {g.n_b}: {cells} quadrangles or {2 * cells} triangles"
                ),
                got=f"{n_quad} quadrangles and {n_tri} triangles",
            )
