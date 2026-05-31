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


def _entity_z_range(ent: Any) -> tuple[float, float] | None:
    """Return (zmin, zmax) for an entity that has identifiable z-extent."""
    if isinstance(ent, PolyPrism):
        if ent.extrude:
            return (ent.zmin, ent.zmax)
        zs = sorted(ent.buffers.keys())
        return (zs[0], zs[-1])
    if hasattr(ent, "zmin") and hasattr(ent, "zmax"):
        return (ent.zmin, ent.zmax)
    return None


def _entity_z_boundaries(ent: Any) -> list[float]:
    """List of z-boundaries this entity introduces."""
    if isinstance(ent, PolyPrism):
        if ent.extrude:
            return [ent.zmin, ent.zmax]
        return sorted(ent.buffers.keys())
    return []


def _entity_xy_at(ent: Any, z: float):
    """Shapely geometry of entity's footprint at z, or None if no overlap."""
    if isinstance(ent, PolyPrism) and ent.extrude and ent.zmin <= z <= ent.zmax:
        return ent.polygons
    return None


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
            ent_z_range = _entity_z_range(ent)
            if ent_z_range is None:
                continue
            for z in _entity_z_boundaries(ent):
                if not (cohort.zmin < z < cohort.zmax):
                    continue
                if z in cohort_z_set:
                    continue
                ent_xy = _entity_xy_at(ent, z)
                if ent_xy is None:
                    continue
                cohort_xy = _cohort_xy_at(cohort, z)
                if cohort_xy.intersects(ent_xy):
                    raise StructuredZStackError(
                        entity_index=ent_idx, z=z, cohort_index=cohort_idx
                    )
