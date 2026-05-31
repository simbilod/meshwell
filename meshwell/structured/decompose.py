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
from meshwell.structured.types import Cohort, StructuredSlab, SubPiece


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
            slab_indices = tuple(
                s.source_index for s in cohort.slabs if s.zlo <= zlo and s.zhi >= zhi
            )
            cohort_subs.extend(
                SubPiece(
                    cohort_index=ci,
                    z_interval=(zlo, zhi),
                    sub_polygon=sub_poly,
                    source_slab_indices=slab_indices,
                )
                for sub_poly in inside
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
        new_ent = copy(ent)
        new_ent.polygons = new_polys
        pre_cut.append(new_ent)

    return subpieces_per_cohort, pre_cut


def _cohort_xy_at(cohort: Cohort, z: float):
    """Return the union of all slab footprints active at height z."""
    polys = [s.footprint for s in cohort.slabs if s.zlo <= z <= s.zhi]
    return unary_union(polys)
