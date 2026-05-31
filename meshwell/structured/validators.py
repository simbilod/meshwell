"""Pipeline validators that may raise.

Stage 3a — z-stack: every entity's z-boundary that lands inside a
cohort z-range must coincide with one of the cohort's own z-planes.

Post-BOP shell invariance comes in Task 12.
"""
from __future__ import annotations

from typing import Any

from meshwell.polyprism import PolyPrism
from meshwell.structured.exceptions import StructuredZStackError
from meshwell.structured.types import Cohort


def _entity_z_boundaries(ent: Any) -> list[float]:
    """List of z-boundaries this entity introduces."""
    if isinstance(ent, PolyPrism):
        if ent.extrude:
            return [ent.zmin, ent.zmax]
        return sorted(ent.buffers.keys())
    return []


def _entity_xy_at(ent: Any, z: float):
    """Shapely geometry of entity's footprint at z, or None if no overlap."""
    # v1: only extrude=True PolyPrisms are considered for z-stack
    # overlap checks. Buffered PolyPrisms are not structured-eligible
    # upstream, and their z-boundaries are returned by
    # _entity_z_boundaries but their XY footprint at intermediate z
    # is variable — out of scope for v1.
    if isinstance(ent, PolyPrism) and ent.extrude and ent.zmin <= z <= ent.zmax:
        return ent.polygons
    return None


def _approx_in(z: float, zs: set[float], tol: float = 1e-9) -> bool:
    """Return True if z is within tol of any value in zs."""
    return any(abs(z - zp) <= tol for zp in zs)


def _cohort_xy_at(cohort: Cohort, z: float):
    """Union of cohort slab footprints whose z-interval covers z."""
    from shapely.ops import unary_union

    polys = [s.footprint for s in cohort.slabs if s.zlo <= z <= s.zhi]
    return unary_union(polys)


def validate_z_stacks(cohorts: list[Cohort], entities: list[Any]) -> None:
    """Stage 3a — raise on any mid-height z-boundary intersecting a cohort.

    For every entity's zlo/zhi, check each cohort: if the z lies
    strictly inside the cohort's z-range AND the entity's XY at that
    z intersects the cohort's XY at that z, raise unless z coincides
    with one of the cohort's own z-planes.
    """
    for cohort_idx, cohort in enumerate(cohorts):
        cohort_z_set = set(cohort.z_planes)
        for ent_idx, ent in enumerate(entities):
            boundaries = _entity_z_boundaries(ent)
            if not boundaries:
                continue
            for z in boundaries:
                if not (cohort.zmin < z < cohort.zmax):
                    continue
                if _approx_in(z, cohort_z_set):
                    continue
                ent_xy = _entity_xy_at(ent, z)
                if ent_xy is None:
                    continue
                cohort_xy = _cohort_xy_at(cohort, z)
                if cohort_xy.intersects(ent_xy):
                    raise StructuredZStackError(
                        entity_index=ent_idx, z=z, cohort_index=cohort_idx
                    )
