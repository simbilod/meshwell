"""Pipeline validators that may raise.

z-stack: every entity's z-boundary that lands inside a cohort z-range
must coincide with one of the cohort's own z-planes. Volumetric overlap,
cohort wrapping, and post-BOP shell invariance are also checked here.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from shapely.geometry import Polygon
from shapely.ops import unary_union

from meshwell.polyprism import PolyPrism
from meshwell.structured._zmath import approx_in
from meshwell.structured.decompose import _cohort_xy_at
from meshwell.structured.exceptions import (
    CohortNotWrappedError,
    CohortShellModifiedError,
    StructuredVolumetricOverlapError,
    StructuredZStackError,
)
from meshwell.structured.types import Cohort, ShapeKey, SlabMeta

if TYPE_CHECKING:
    from OCP.TopoDS import TopoDS_Face


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


def validate_no_volumetric_cohort_overlap(
    cohorts: list[Cohort], entities: list[Any], tol: float = 1e-9
) -> None:
    """Raise if a non-cohort entity shares a 3D interior with a cohort.

    For each cohort, for each non-cohort entity: compute the strict
    z-overlap and, if positive, sample a representative z inside the
    open overlap interval and test XY-intersection. If the entity's
    XY at that z intersects the cohort's XY at that z, raise.

    Cohorts and unstructured entities must live in disjoint 3D volumes
    (touching only at z-planes). The cad_occ cut cascade is skipped
    between cohorts and non-cohorts (cohort sub-solids are OCC-invalid
    bottom-up builds), so any genuine 3D overlap would corrupt the
    fragment pass.
    """
    for cohort_idx, cohort in enumerate(cohorts):
        for ent_idx, ent in enumerate(entities):
            # Skip cohort entities themselves and non-extruded primitives
            # that have no z-extent we can reason about.
            if not isinstance(ent, PolyPrism) or not ent.extrude:
                continue
            ent_zmin = ent.zmin
            ent_zmax = ent.zmax
            z_overlap = min(cohort.zmax, ent_zmax) - max(cohort.zmin, ent_zmin)
            if z_overlap <= tol:
                continue
            # Skip if this entity IS one of the cohort's source slabs
            # (a structured slab); we only care about unstructured
            # entities that share cohort z-interior. A structured slab
            # whose source_index matches ent_idx is part of the cohort.
            if any(s.source_index == ent_idx for s in cohort.slabs):
                continue
            # Sample a z strictly inside the open overlap interval.
            z_mid = 0.5 * (max(cohort.zmin, ent_zmin) + min(cohort.zmax, ent_zmax))
            ent_xy = _entity_xy_at(ent, z_mid)
            if ent_xy is None:
                continue
            cohort_xy = _cohort_xy_at(cohort, z_mid)
            if cohort_xy.is_empty:
                continue
            if cohort_xy.intersects(ent_xy):
                inter = cohort_xy.intersection(ent_xy)
                # Skip zero-area boundary touches (lines/points only)
                if hasattr(inter, "area") and inter.area <= tol:
                    continue
                raise StructuredVolumetricOverlapError(
                    entity_index=ent_idx,
                    cohort_index=cohort_idx,
                    z_overlap=z_overlap,
                )


def validate_cohort_wrapping(
    cohorts: "list[Cohort]",
    unstructured_entities: list,
    point_tolerance: float = 1e-3,
) -> None:
    """Verify every cohort z-plane is covered by adjacent unstructured neighbours.

    For each cohort and each of its outer z-planes (``zmin``, ``zmax``),
    walk every cohort sub-piece active at that plane and confirm at
    least one adjacent unstructured PolyPrism's footprint covers it.
    "Covers" = the sub-piece's footprint is contained (within
    ``point_tolerance``) in the union of all adjacent neighbours'
    footprints at that plane.

    The union convention lets multiple neighbours together wrap a
    cohort (e.g., a substrate split into east/west halves).

    Raises ``CohortNotWrappedError`` on the first violation found.
    """
    for ci, cohort in enumerate(cohorts):
        for z_plane in (cohort.z_planes[0], cohort.z_planes[-1]):
            # Collect adjacent neighbours' footprints at this z-plane.
            neighbours_at_z: list[Polygon] = []
            for ent in unstructured_entities:
                if not isinstance(ent, PolyPrism) or not ent.extrude:
                    continue
                z_keys = sorted(ent.buffers.keys())
                if not (
                    approx_in(z_plane, [z_keys[0]]) or approx_in(z_plane, [z_keys[-1]])
                ):
                    continue
                neighbours_at_z.append(ent.polygons)
            if not neighbours_at_z:
                slabs_here = [s for s in cohort.slabs if s.zlo <= z_plane <= s.zhi]
                first_sub = slabs_here[0].source_index if slabs_here else -1
                raise CohortNotWrappedError(
                    cohort_index=ci,
                    z_plane=z_plane,
                    sub_piece_index=first_sub,
                    reason="no adjacent unstructured neighbour at this z-plane",
                )
            cover = unary_union(neighbours_at_z)
            slabs_here = [s for s in cohort.slabs if s.zlo <= z_plane <= s.zhi]
            for slab in slabs_here:
                if not cover.buffer(point_tolerance).contains(slab.footprint):
                    raise CohortNotWrappedError(
                        cohort_index=ci,
                        z_plane=z_plane,
                        sub_piece_index=slab.source_index,
                        reason=(
                            "adjacent neighbour union does not cover slab "
                            "footprint at this z-plane"
                        ),
                    )


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
