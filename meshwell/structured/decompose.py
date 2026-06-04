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

from meshwell.structured._zmath import approx_in, approx_key
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


def decompose_cohorts(
    cohorts: list[Cohort],
    unstructured_entities: list[Any],
) -> tuple[list[list[SubPiece]], list[Any]]:
    """Stage 3 driver.

    Returns:
        - subpieces_per_cohort: parallel to `cohorts`. Each cohort gets
          a flat list of SubPiece records (one per z-interval x sub-polygon).
        - pre_cut_unstructured: same order as input. PolyPrisms that
          share a z-plane with any cohort are returned as shallow
          copies with their `polygons` replaced by a MultiPolygon
          decomposed to match the cohort sub-faces at the shared plane.
          Non-touching unstructured entities are returned unchanged.
    """
    from meshwell.polyprism import PolyPrism

    # 1. Per-cohort, per-z-interval footprint.
    per_cohort_per_interval_footprint: dict[int, dict[tuple[float, float], object]] = {}
    for ci, cohort in enumerate(cohorts):
        per_cohort_per_interval_footprint[ci] = {}
        for zlo, zhi in zip(cohort.z_planes[:-1], cohort.z_planes[1:]):
            slabs_here = [s for s in cohort.slabs if s.zlo <= zlo and s.zhi >= zhi]
            fp = zinterval_footprint(slabs_here)
            per_cohort_per_interval_footprint[ci][(zlo, zhi)] = fp

    # 2. Build cut_sources[z] = union of cohort-side and unstructured-side
    # XY outlines at z. This is the symmetric data both sides will use.
    cut_sources: dict[float, list] = {}
    for _ci, _cohort in enumerate(cohorts):
        for (zlo, zhi), fp in per_cohort_per_interval_footprint[_ci].items():
            cut_sources.setdefault(zlo, []).append(fp.boundary)
            cut_sources.setdefault(zhi, []).append(fp.boundary)
            # Also add each individual slab's footprint boundary that
            # spans this z-interval (including voids).  Without this,
            # polygonize would emit one sub-piece per Policy-B union
            # region instead of one per per-slab region, so overlapping
            # solids with higher mesh_order would lose their share to
            # the lowest-mesh_order slab, and void carvings would never
            # split the surrounding solid.
            for s in _cohort.slabs:
                if s.zlo <= zlo and s.zhi >= zhi:
                    cut_sources.setdefault(zlo, []).append(s.footprint.boundary)
                    cut_sources.setdefault(zhi, []).append(s.footprint.boundary)
    for ent in unstructured_entities:
        if not isinstance(ent, PolyPrism) or not ent.extrude:
            continue
        z_keys = sorted(ent.buffers.keys())
        for z in (z_keys[0], z_keys[-1]):
            # Only contribute if this entity touches some cohort at z.
            for cohort in cohorts:
                if not approx_in(z, cohort.z_planes):
                    continue
                cohort_xy_at_z = _cohort_xy_at(cohort, z)
                if cohort_xy_at_z.intersects(ent.polygons):
                    # Normalize to the existing key (cohort's exact z-plane value)
                    # so all contributions for the same plane share one dict entry.
                    existing = approx_key(z, cut_sources)
                    key = existing if existing is not None else z
                    cut_sources.setdefault(key, []).append(ent.polygons.boundary)

    cut_unions = {z: unary_union(lines) for z, lines in cut_sources.items()}

    # 3. Cohort side: emit SubPieces.
    subpieces_per_cohort: list[list[SubPiece]] = []
    for ci, cohort in enumerate(cohorts):
        cohort_subs: list[SubPiece] = []
        for (zlo, zhi), fp in per_cohort_per_interval_footprint[ci].items():
            cuts_zlo = cut_unions.get(zlo)
            cuts_zhi = cut_unions.get(zhi)
            boundaries = [fp.boundary]
            if cuts_zlo is not None:
                boundaries.append(cuts_zlo)
            if cuts_zhi is not None:
                boundaries.append(cuts_zhi)
            merged = unary_union(boundaries)
            pieces = list(polygonize(merged))
            # Filter to pieces whose representative_point lies inside fp.
            inside = [p for p in pieces if fp.contains(p.representative_point())]
            candidate_slabs = [s for s in cohort.slabs if s.zlo <= zlo and s.zhi >= zhi]
            for sub_poly in inside:
                owner = _owner_slab(sub_poly, candidate_slabs)
                if owner is None:
                    # The representative point is outside all candidate
                    # footprints — no SubPiece to emit.
                    continue
                cohort_subs.append(
                    SubPiece(
                        cohort_index=ci,
                        z_interval=(zlo, zhi),
                        sub_polygon=sub_poly,
                        source_slab_indices=(owner,),
                    )
                )
        subpieces_per_cohort.append(cohort_subs)

    # 4. Unstructured side: pre-cut PolyPrisms that touch a cohort z-plane.
    pre_cut: list[Any] = []
    for ent in unstructured_entities:
        if not isinstance(ent, PolyPrism) or not ent.extrude:
            pre_cut.append(ent)
            continue
        touched_keys = [
            approx_key(z, cut_unions)
            for z in (ent.zmin, ent.zmax)
            if any(approx_in(z, c.z_planes) for c in cohorts)
            and approx_key(z, cut_unions) is not None
        ]
        if not touched_keys:
            pre_cut.append(ent)
            continue
        all_cuts = unary_union([cut_unions[k] for k in touched_keys])
        merged = unary_union([ent.polygons.boundary, all_cuts])
        pieces = list(polygonize(merged))
        inside = [p for p in pieces if ent.polygons.contains(p.representative_point())]
        if not inside:
            pre_cut.append(ent)
            continue
        new_polys = MultiPolygon(inside) if len(inside) > 1 else inside[0]
        # Determine whether any touching cohort has arc-bearing slabs.
        # If so, propagate arc detection to the pre-cut entity so its boundary
        # (which now follows the cohort's polyline-approximated arc) is built
        # with matching OCC arc edges rather than polyline edges. Without this,
        # the unstructured and cohort sides build geometrically-coincident-but-
        # topologically-different OCC representations on the shared boundary, and
        # BOP cannot merge them within fragment_fuzzy_value.
        arc_bearing_slabs: list[StructuredSlab] = []
        for c in cohorts:
            for z_check in (ent.zmin, ent.zmax):
                if not approx_in(z_check, c.z_planes):
                    continue
                if not _cohort_xy_at(c, z_check).intersects(ent.polygons):
                    continue
                arc_bearing_slabs.extend(s for s in c.slabs if s.identify_arcs)
                break  # one match is enough to flag the cohort

        new_ent = copy(ent)
        new_ent.polygons = new_polys
        if arc_bearing_slabs:
            # Propagate arc detection settings from the most-permissive cohort slab.
            # arc_tolerance: use the LOOSEST tolerance among touching cohort slabs
            # so arc detection on the pre-cut succeeds whenever it succeeded on the
            # cohort side.
            new_ent.identify_arcs = True
            new_ent.arc_tolerance = max(s.arc_tolerance for s in arc_bearing_slabs)
            new_ent.min_arc_points = min(s.min_arc_points for s in arc_bearing_slabs)
        # Tag the pre-cut entity with the cohorts it touches and the
        # shared z-plane for each. Consumed by PolyPrism.instanciate_occ
        # to route the boundary wire at z=z_shared through the cohort's
        # EdgeRegistry, so cohort/neighbour arc/line TShapes match by
        # construction (not by BOP fuzzy detection).
        cohort_adjacency: list[tuple[int, float]] = []
        for ci, c in enumerate(cohorts):
            for z_check in (ent.zmin, ent.zmax):
                if not approx_in(z_check, c.z_planes):
                    continue
                if _cohort_xy_at(c, z_check).intersects(ent.polygons):
                    cohort_adjacency.append((ci, z_check))
                    break
        new_ent._cohort_adjacency = cohort_adjacency
        pre_cut.append(new_ent)

    return subpieces_per_cohort, pre_cut


def _cohort_xy_at(cohort: Cohort, z: float):
    """Return the union of all slab footprints active at height z."""
    polys = [s.footprint for s in cohort.slabs if s.zlo <= z <= s.zhi]
    return unary_union(polys)


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
