"""Stage 3 — cohort decomposition in shapely.

Stage 3b: collect z-planes (the cohort's slab boundaries after the 3a
validator ran).
Stage 3c: per-z-interval footprint with Policy B carving.
"""

from __future__ import annotations

from typing import Any

import shapely
from shapely.geometry import Polygon
from shapely.ops import polygonize, unary_union

from meshwell.geometry_entity import decompose_vertices_2d
from meshwell.structured.exceptions import CanonicalArrangementError
from meshwell.structured.types import (
    Arrangement,
    ArrangementEdge,
    Cohort,
    StructuredSlab,
    SubPiece,
    VertexKey,
)


def _quantize_key(x: float, y: float, z: float, point_tolerance: float) -> VertexKey:
    """Match VertexRegistry._key's quantization.

    Ensures canonical edges and runtime EdgeRegistry vertices share the
    same key space.
    """
    s = point_tolerance
    return (round(x / s), round(y / s), round(z / s))


def _build_canonical_edges(
    merged,
    z: float,
    point_tolerance: float,
    identify_arcs: bool,
    min_arc_points: int,
    arc_tolerance: float,
) -> tuple[tuple[ArrangementEdge, ...], dict[frozenset[VertexKey], int]]:
    """Build canonical arrangement edges from a noded MultiLineString.

    ``merged`` is the output of ``shapely.ops.unary_union`` over the
    cohort's boundary linework. Each component of ``merged`` becomes
    one ``ArrangementEdge`` — shapely's union nodes at every line
    crossing, so each component already spans between two arrangement
    nodes (or is a closed standalone loop).

    Open edges are stored in a canonical orientation that is invariant
    under input direction reversal: step 1 places the lex-smaller
    endpoint at index 0 (reversing if the lex-min endpoint was at
    index -1); step 2 reverses again if ``keys[-1] < keys[1]``. The
    net effect is a consistent canonical form for both traversal
    directions of the same edge — this is what makes the lookup map
    direction-invariant. (Note: step 2 can land the lex-min endpoint
    back at index -1; we do NOT guarantee "lex-min start", only
    "deterministic canonical form".) Their consecutive vertex pairs are
    registered in ``edge_by_vertex_pair`` for fast O(1) replay lookup.

    Closed standalone edges (single loops with no other arrangement
    nodes — e.g., a disc boundary alone in a cohort) are stored with
    ``is_closed=True`` but are NOT pair-indexed; sub-piece consumers
    detect the missing pairs and fall back to today's greedy per-ring
    fit, which is deterministic for a single closed ring.

    Raises CanonicalArrangementError if two distinct edges register
    the same unordered vertex pair (parallel-edge graph violation).
    """
    from shapely.geometry import LineString, MultiLineString

    if merged.is_empty:
        return (), {}
    if isinstance(merged, LineString):
        components = [merged]
    elif isinstance(merged, MultiLineString):
        components = list(merged.geoms)
    else:
        # GeometryCollection or other: extract the LineString members.
        components = [
            g for g in getattr(merged, "geoms", []) if isinstance(g, LineString)
        ]

    canonical_edges: list[ArrangementEdge] = []
    edge_by_vertex_pair: dict[frozenset[VertexKey], int] = {}

    for component in components:
        coords = [(c[0], c[1]) for c in component.coords]
        if len(coords) < 2:
            continue
        keys = [_quantize_key(x, y, z, point_tolerance) for x, y in coords]
        # Collapse runs that quantize to the same key (sub-tolerance jitter).
        dedup_coords: list[tuple[float, float]] = [coords[0]]
        dedup_keys: list[VertexKey] = [keys[0]]
        for c, k in zip(coords[1:], keys[1:]):
            if k != dedup_keys[-1]:
                dedup_coords.append(c)
                dedup_keys.append(k)
        coords = dedup_coords
        keys = dedup_keys
        if len(coords) < 2:
            continue

        is_closed = keys[0] == keys[-1]
        if is_closed:
            # OPEN storage convention: drop the closing duplicate.
            coords = coords[:-1]
            keys = keys[:-1]
            # Closed standalone: fit segments as a closed ring via
            # decompose_vertices_2d (which seam-canonicalises internally).
            # We pass closed form (first==last) so its seam logic kicks in.
            fit_coords = [*coords, coords[0]]
        else:
            # Open edge: canonicalise so the orientation is invariant
            # under input direction reversal. Step 1: if the lex-min
            # key is at index -1, reverse it to index 0. Step 2: if
            # keys[-1] < keys[1], reverse again. This gives a
            # deterministic canonical form (NOT necessarily lex-min
            # at start) — what matters is consistency across both
            # traversal directions, which is what the pair lookup
            # needs.
            i_min = min(range(len(keys)), key=lambda i: keys[i])
            # An open edge's start/end vertices SHOULD be at indices 0
            # and -1 (arrangement nodes); lex-min should be one of them.
            # If lex-min is interior, the component is mis-noded —
            # treat as-is in input order (this shouldn't happen for
            # unary_union output).
            if i_min != 0 and i_min == len(keys) - 1:
                coords = list(reversed(coords))
                keys = list(reversed(keys))
            # Now ensure direction: if keys[-1] < keys[1] (next vertex
            # after the start is "later" than the end), reverse the tail.
            if len(keys) >= 3 and keys[-1] < keys[1]:
                coords = list(reversed(coords))
                keys = list(reversed(keys))
            fit_coords = list(coords)

        segments = decompose_vertices_2d(
            fit_coords,
            z=z,
            point_tolerance=point_tolerance,
            identify_arcs=identify_arcs,
            min_arc_points=min_arc_points,
            arc_tolerance=arc_tolerance,
        )

        edge = ArrangementEdge(
            vertex_keys=tuple(keys),
            z=z,
            segments=tuple(segments),
            is_closed=is_closed,
        )
        edge_idx = len(canonical_edges)
        canonical_edges.append(edge)

        # Register vertex pairs in the lookup, but ONLY for open edges.
        # Closed standalones are not pair-indexed (see docstring).
        if not is_closed:
            for i in range(len(keys) - 1):
                pair = frozenset({keys[i], keys[i + 1]})
                if pair in edge_by_vertex_pair:
                    raise CanonicalArrangementError(
                        cohort_index=-1,
                        reason=(
                            f"duplicate vertex pair {tuple(pair)} on edges "
                            f"{edge_by_vertex_pair[pair]} and {edge_idx} "
                            "(parallel edges between the same arrangement "
                            "nodes — unary_union output violates the planar "
                            "arrangement assumption)"
                        ),
                    )
                edge_by_vertex_pair[pair] = edge_idx

    return tuple(canonical_edges), edge_by_vertex_pair


def zinterval_footprint(slabs_here: list[StructuredSlab]) -> Polygon:
    """Resolve Policy B carving for one z-interval.

    Sort slabs by (mesh_order, source_index) ascending; lower wins.
    For mesh_bool=True: union (footprint - accumulated).
    For mesh_bool=False (void): subtract footprint from accumulated.
    """
    ordered = sorted(
        slabs_here,
        key=lambda s: (
            s.mesh_order if s.mesh_order is not None else float("inf"),
            s.source_index,
        ),
    )
    acc = Polygon()  # empty
    for s in ordered:
        if s.mesh_bool:
            new = s.footprint.difference(acc)
            acc = unary_union([acc, new])
        else:
            acc = acc.difference(s.footprint)
    return acc


def build_cohort_arrangement(
    cohort_index: int,
    cohort: Cohort,
    adjacent_unstructured: list,
) -> Arrangement:
    """One shapely polygonize over the union of all relevant boundaries.

    `adjacent_unstructured` is a list of shapely line geometries
    (typically `ent.polygons.boundary` for each unstructured PolyPrism
    whose top/bottom z-plane coincides with one of `cohort.z_planes`
    AND whose XY intersects the cohort footprint).

    The returned `Arrangement.polygons` tile the union of all those
    polygons' interiors. The cohort sub-piece extractor
    (``arrangement_subpieces_for_interval``) filters this list by
    z-interval to produce per-interval SubPieces.
    """
    linework = [s.footprint.boundary for s in cohort.slabs] + list(
        adjacent_unstructured
    )
    merged = unary_union(linework)
    pieces = tuple(polygonize(merged))
    return Arrangement(cohort_index=cohort_index, polygons=pieces)


def arrangement_subpieces_for_interval(
    arrangement: Arrangement,
    cohort: Cohort,
    zlo: float,
    zhi: float,
) -> list[SubPiece]:
    """Project arrangement polygons to one z-interval's SubPieces.

    For each polygon in the arrangement:
    1. Compute owner via the cohort's slabs spanning this interval.
    2. If owner is None (polygon is in the arrangement but no slab here
       contains its representative point), drop.
    3. Otherwise emit one SubPiece referencing the polygon by identity.
    """
    candidate_slabs = [s for s in cohort.slabs if s.zlo <= zlo and s.zhi >= zhi]
    if not candidate_slabs:
        return []
    fp = zinterval_footprint(candidate_slabs)
    subs: list[SubPiece] = []
    for p in arrangement.polygons:
        pt = p.representative_point()
        if not fp.contains(pt):
            continue
        owner = _owner_slab(p, candidate_slabs)
        if owner is None:
            continue
        subs.append(
            SubPiece(
                cohort_index=arrangement.cohort_index,
                z_interval=(zlo, zhi),
                sub_polygon=p,
                source_slab_indices=(owner,),
            )
        )
    return subs


def decompose_cohorts(
    cohorts: list[Cohort],
    unstructured_entities: list[Any],
    point_tolerance: float = 1e-3,
) -> tuple[list[list[SubPiece]], list[Any]]:
    """Stage 3 driver — cohort-global arrangement edition.

    Returns:
        - subpieces_per_cohort: parallel to ``cohorts``. Each cohort gets
          a flat list of SubPiece records (one per z-interval x sub-polygon).
        - unstructured_entities: returned UNCHANGED (same list, same
          Python objects). Cohort↔unstructured interfaces are discovered
          downstream by BOP fragment with AABB-rescue fallback for any
          face pairs BOP doesn't auto-match by TShape identity. The
          earlier ``CohortNeighbourUnstructured`` pre-cut machinery was
          deleted on 2026-06-09 (see
          docs/superpowers/specs/2026-06-01-cohort-topology-investigations.md):
          spike evidence showed it INCREASED rescue count on the meander
          stress scene and BLOCKED arc-cohort scenes at fine cl due to
          wrapped-cylinder surface types reaching gmsh as ``Unknown``.
    """
    from meshwell.polyprism import PolyPrism

    # 1. For each cohort, collect adjacent unstructured boundaries to
    # include in the cohort's arrangement linework.
    adjacency_lines_per_cohort: list[list] = []
    for cohort in cohorts:
        lines = []
        for ent in unstructured_entities:
            if not isinstance(ent, PolyPrism) or not ent.extrude:
                continue
            z_keys = sorted(ent.buffers.keys())
            touches = False
            for z in (z_keys[0], z_keys[-1]):
                z_snap = _snap_to_cohort_plane(z, cohort)
                if z_snap is None:
                    continue
                if _cohort_xy_at(cohort, z_snap).intersects(ent.polygons):
                    touches = True
                    break
            if touches:
                # Snap to the same grid the cohort slab footprints were
                # snapped to in structured_pre_pass. Without this, the
                # 1e-5 perturbation from prepare_entities makes the
                # cladding boundary live on a different grid than the
                # cohort, and polygonize produces a thin annulus that no
                # cohort sub-piece covers.
                lines.append(
                    shapely.set_precision(
                        ent.polygons.boundary,
                        grid_size=point_tolerance,
                        mode="valid_output",
                    )
                )
        adjacency_lines_per_cohort.append(lines)

    # 2. Build one arrangement per cohort.
    arrangements: list[Arrangement] = []
    for ci, cohort in enumerate(cohorts):
        arrangements.append(
            build_cohort_arrangement(
                cohort_index=ci,
                cohort=cohort,
                adjacent_unstructured=adjacency_lines_per_cohort[ci],
            )
        )

    # 3. Cohort sub-pieces: per z-interval, project arrangement to SubPieces.
    subpieces_per_cohort: list[list[SubPiece]] = []
    for ci, cohort in enumerate(cohorts):
        cohort_subs: list[SubPiece] = []
        for zlo, zhi in zip(cohort.z_planes[:-1], cohort.z_planes[1:]):
            cohort_subs.extend(
                arrangement_subpieces_for_interval(arrangements[ci], cohort, zlo, zhi)
            )
        subpieces_per_cohort.append(cohort_subs)

    # 4. Unstructured entities: returned unchanged. BOP + AABB rescue
    # handle interface detection downstream.
    return subpieces_per_cohort, list(unstructured_entities)


def _cohort_xy_at(cohort: Cohort, z: float):
    """Return the union of all slab footprints active at height z."""
    polys = [s.footprint for s in cohort.slabs if s.zlo <= z <= s.zhi]
    return unary_union(polys)


def _snap_to_cohort_plane(z: float, cohort: Cohort, tol: float = 1e-9) -> float | None:
    """Return the cohort z-plane within tol of z, or None if none match.

    Used to normalize an unstructured entity's z-extent to a cohort's exact
    z-plane value before strict-inequality footprint queries.
    """
    for zp in cohort.z_planes:
        if abs(z - zp) <= tol:
            return zp
    return None


def _owner_slab(
    sub_polygon: Polygon, candidate_slabs: list[StructuredSlab]
) -> int | None:
    """Pick the slab that owns a sub-piece under Policy B.

    Take the sub_polygon's representative_point, find every candidate
    whose footprint contains it (solids and voids), then resolve the
    same way `zinterval_footprint` does: sort by (mesh_order, source_index)
    ascending and the first wins.

    Voids return their own source_index now — they become first-class
    sub-pieces in the cohort compound, marked keep=False at post-pass
    time so the XAO writer's existing keep=False semantics produce
    `neighbour___void` interface tags.

    Returns the winning slab's source_index, or None if the point is
    outside every candidate's footprint.
    """
    pt = sub_polygon.representative_point()
    here = [s for s in candidate_slabs if s.footprint.contains(pt)]
    if not here:
        return None
    ordered = sorted(
        here,
        key=lambda s: (
            s.mesh_order if s.mesh_order is not None else float("inf"),
            s.source_index,
        ),
    )
    winner = ordered[0]
    return winner.source_index
