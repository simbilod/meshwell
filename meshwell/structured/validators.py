"""Pipeline validators that may raise.

Stage 3a — z-stack: every entity's z-boundary that lands inside a
cohort z-range must coincide with one of the cohort's own z-planes.

Post-BOP shell invariance comes in Task 12.
"""
from __future__ import annotations

from typing import Any

from meshwell.polyprism import PolyPrism
from meshwell.structured._zmath import approx_in
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
                if approx_in(z, cohort_z_set):
                    continue
                ent_xy = _entity_xy_at(ent, z)
                if ent_xy is None:
                    continue
                cohort_xy = _cohort_xy_at(cohort, z)
                if cohort_xy.intersects(ent_xy):
                    raise StructuredZStackError(
                        entity_index=ent_idx, z=z, cohort_index=cohort_idx
                    )


from typing import TYPE_CHECKING, Iterable  # noqa: E402

from meshwell.structured.exceptions import CohortShellModifiedError  # noqa: E402
from meshwell.structured.types import ShapeKey, SlabMeta  # noqa: E402

if TYPE_CHECKING:
    from OCP.TopoDS import TopoDS_Face


def validate_cohort_shells(
    slab_meta: dict[ShapeKey, SlabMeta],
    faces_by_key: "dict[ShapeKey, TopoDS_Face]",
    builder,
) -> None:
    """Stage 5 post-pass validator for cohort shell face invariance.

    Raise if BOP modified any pre-baked cohort shell face into more
    than one piece.

    For each face role (bot/top/lateral) in each SlabMeta, query
    builder.Modified(original_face). Acceptable outcomes:
      - empty + not deleted: face passed through BOP unchanged.
      - single replacement: face merged with a coincident neighbour.
    Unacceptable:
      - multiple replacements: BOP introduced a cut on this shell face.
    """
    for meta in slab_meta.values():
        face_roles: list[tuple[str, ShapeKey]] = [
            ("bot", meta.bot_face_key),
            ("top", meta.top_face_key),
        ] + [(f"lateral_{i}", lk) for i, lk in enumerate(meta.lateral_face_keys)]
        for role, fk in face_roles:
            face = faces_by_key.get(fk)
            if face is None:
                continue
            modified = builder.Modified(face)
            count = sum(1 for _ in _iterate_list(modified))
            if count > 1:
                raise CohortShellModifiedError(
                    slab_index=meta.slab_index,
                    face_role=role,
                    fragment_count=count,
                )


def _iterate_list(lst) -> Iterable:
    """Iterate a TopTools_ListOfShape (or empty list-like)."""
    n = lst.Extent() if hasattr(lst, "Extent") else 0
    if n == 0:
        return iter(())
    out: list = []
    try:
        out = list(lst)
    except TypeError:
        from OCP.TopTools import TopTools_ListIteratorOfListOfShape

        it = TopTools_ListIteratorOfListOfShape(lst)
        while it.More():
            out.append(it.Value())
            it.Next()
    return out
