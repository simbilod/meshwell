"""Stage 3 — cohort decomposition in shapely.

Stage 3b: collect z-planes (just the cohort's own slab boundaries
after the 3a validator ran).
Stage 3c: per-z-interval footprint with Policy B carving (this task).
Stage 3d: bidirectional pre-cut at shared z-planes (Task 7).
"""

from __future__ import annotations

from copy import copy
from typing import Any

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import polygonize, unary_union

from meshwell.structured.types import Arrangement, Cohort, StructuredSlab, SubPiece


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
    polygons' interiors. Downstream extractors filter this list by
    z-interval (cohort sub-pieces) or by entity footprint (unstructured
    pre-cut). All filters return the exact same Polygon objects.
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


def arrangement_pre_cut_for_entity(
    arrangement: Arrangement,
    entity_polygons,
):
    """Project arrangement polygons through one unstructured entity's footprint.

    Returns:
        - the original `entity_polygons` if no arrangement polygon's
          representative point falls inside it (entity is untouched).
        - the single arrangement Polygon (by Python `is` identity) if
          exactly one matches.
        - a `MultiPolygon` whose members are bit-exactly equal to
          arrangement polygons when multiple match. Shapely 2.x's
          `.geoms` accessor returns fresh wrapper objects each access,
          so Python `is` is NOT preserved through MultiPolygon. The
          underlying GEOS coordinate sequences are shared by reference;
          members satisfy `equals_exact(member, arrangement_poly,
          tolerance=0.0)`. See `Arrangement` docstring for the full
          identity contract.
    """
    inside = [
        p
        for p in arrangement.polygons
        if entity_polygons.contains(p.representative_point())
    ]
    if not inside:
        return entity_polygons
    if len(inside) == 1:
        return inside[0]
    return MultiPolygon(inside)


def decompose_cohorts(
    cohorts: list[Cohort],
    unstructured_entities: list[Any],
) -> tuple[list[list[SubPiece]], list[Any]]:
    """Stage 3 driver — cohort-global arrangement edition.

    Returns:
        - subpieces_per_cohort: parallel to `cohorts`. Each cohort gets
          a flat list of SubPiece records (one per z-interval x sub-polygon).
        - pre_cut_unstructured: same order as input. PolyPrisms that
          share a z-plane with any cohort are returned as shallow
          copies with their `polygons` replaced by a Polygon (or
          MultiPolygon) drawn FROM the cohort's arrangement, by Python
          identity (single-Polygon case) or bit-exact geometric equality
          (MultiPolygon case; see `Arrangement` docstring). Non-touching
          unstructured entities are returned unchanged.
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
                lines.append(ent.polygons.boundary)
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

    # 4. Unstructured pre-cut: each touching entity picks up arrangement
    # polygons by identity. Find each entity's touched cohorts/planes and
    # apply pre-cut from the arrangement(s).
    pre_cut: list[Any] = []
    for ent in unstructured_entities:
        if not isinstance(ent, PolyPrism) or not ent.extrude:
            pre_cut.append(ent)
            continue
        touched: list[tuple[int, float]] = []
        for ci, cohort in enumerate(cohorts):
            for z in (ent.zmin, ent.zmax):
                z_snap = _snap_to_cohort_plane(z, cohort)
                if z_snap is None:
                    continue
                if _cohort_xy_at(cohort, z_snap).intersects(ent.polygons):
                    touched.append((ci, z_snap))
                    break
        if not touched:
            pre_cut.append(ent)
            continue
        # Use the first touched cohort's arrangement as the source of
        # polygons. (Multi-cohort touch is rare; if it happens, the first
        # cohort's arrangement wins. This matches the previous behaviour
        # of merging cut_unions at the entity's touched planes.)
        primary_ci = touched[0][0]
        new_polys = arrangement_pre_cut_for_entity(
            arrangements[primary_ci], ent.polygons
        )
        if new_polys is ent.polygons:
            pre_cut.append(ent)
            continue
        # Arc-detection propagation: unchanged from previous logic.
        arc_bearing_slabs: list[StructuredSlab] = []
        for ci, _ in touched:
            cohort = cohorts[ci]
            arc_bearing_slabs.extend(s for s in cohort.slabs if s.identify_arcs)

        new_ent = copy(ent)
        new_ent.polygons = new_polys
        if arc_bearing_slabs:
            new_ent.identify_arcs = True
            new_ent.arc_tolerance = max(s.arc_tolerance for s in arc_bearing_slabs)
            new_ent.min_arc_points = min(s.min_arc_points for s in arc_bearing_slabs)
        new_ent._cohort_adjacency = touched
        pre_cut.append(new_ent)

    return subpieces_per_cohort, pre_cut


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
