"""End-to-end driver for the structured pre-pass and post-pass.

Pre-pass: collect → cohort → validate z-stacks → decompose → build →
swap entities. Returns a StructuredState that the orchestrator
threads forward to the cad_occ and meshing stages.

Post-pass (Task 14): expand cohort OCCLabeledEntity into per-sub-solid
entities + record post-BOP face ShapeKeys.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from meshwell.structured.build import build_cohort_compound
from meshwell.structured.cohort import build_cohorts
from meshwell.structured.cohort_entity import _CohortEntity
from meshwell.structured.collect import collect_structured_slabs
from meshwell.structured.decompose import decompose_cohorts
from meshwell.structured.types import ShapeKey, SlabMeta
from meshwell.structured.validators import (
    validate_arc_consistency,
    validate_z_stacks,
)


@dataclass
class StructuredState:
    """Threaded between pre-pass, cad_occ, and post-pass."""

    entities_out: list[Any]
    slab_meta: dict[ShapeKey, SlabMeta] = field(default_factory=dict)
    cohort_entities: list[_CohortEntity] = field(default_factory=list)


def structured_pre_pass(
    entities: list[Any],
    point_tolerance: float,
) -> StructuredState:
    """Run Stages 1-4 and return entities_out for cad_occ.

    If no structured entities are present, returns the input list
    unchanged with an empty slab_meta.
    """
    structured_slabs, unstructured = collect_structured_slabs(entities)
    if not structured_slabs:
        return StructuredState(entities_out=entities)
    cohorts = build_cohorts(structured_slabs)
    validate_z_stacks(cohorts, entities)
    validate_arc_consistency(cohorts)
    subpieces_per_cohort, pre_cut_unstr = decompose_cohorts(cohorts, unstructured)

    cohort_entities: list[_CohortEntity] = []
    all_slab_meta: dict[ShapeKey, SlabMeta] = {}
    for ci, (cohort, subs) in enumerate(zip(cohorts, subpieces_per_cohort)):
        compound, slab_meta = build_cohort_compound(cohort, subs, point_tolerance)
        ce = _CohortEntity(
            compound=compound,
            slab_meta=slab_meta,
            cohort=cohort,
            cohort_index=ci,
        )
        cohort_entities.append(ce)
        all_slab_meta.update(slab_meta)

    entities_out = cohort_entities + pre_cut_unstr
    return StructuredState(
        entities_out=entities_out,
        slab_meta=all_slab_meta,
        cohort_entities=cohort_entities,
    )


def structured_post_pass(
    occ_entities: list,
    state: StructuredState,
) -> list:
    """Expand every cohort OCCLabeledEntity into per-sub-solid entities.

    Matches each surviving post-BOP shape to its slab_meta entry by
    ShapeKey (fast path) or by bounding-box fingerprint (fallback for
    the multi-cohort case where BOPAlgo_Builder regenerates TShape IDs
    even for geometrically unchanged shapes).

    One OCCLabeledEntity per sub-solid, carrying the source slab's
    physical_name and a synthetic index.
    """
    from meshwell.cad_occ import OCCLabeledEntity
    from meshwell.structured.build import _shape_key

    # Build a bbox-keyed lookup from pre-BOP slab ShapeKeys.
    # Used as fallback when the ShapeKey doesn't survive BOP.
    slab_fp_by_key = _build_slab_fingerprints(state)

    expanded: list = []
    next_index = max((e.index for e in occ_entities), default=-1) + 1
    cohort_pnames = {ce.physical_name for ce in state.cohort_entities}
    for ent in occ_entities:
        if ent.physical_name not in cohort_pnames:
            expanded.append(ent)
            continue
        for shape in ent.shapes:
            key = _shape_key(shape)
            meta = state.slab_meta.get(key)
            if meta is None:
                # Fast-path miss: BOPAlgo_Builder regenerated the TShape.
                # Fall back to spatial fingerprint matching.
                meta = _match_by_bbox(shape, slab_fp_by_key, state.slab_meta)
            if meta is None:
                # Sub-solid has no matching slab (e.g. a void or leftover
                # fragment from a BOP boundary split). Represent it as-is
                # under the cohort name so it still gets meshed.
                expanded.append(_copy_with(ent, [shape], next_index))
                next_index += 1
                continue
            sub_ent = OCCLabeledEntity(
                shapes=[shape],
                physical_name=meta.physical_name,
                index=next_index,
                keep=True,
                dim=3,
                mesh_order=ent.mesh_order,
            )
            expanded.append(sub_ent)
            next_index += 1
    return expanded


def _build_slab_fingerprints(
    state: StructuredState,
) -> dict[ShapeKey, tuple[float, ...]]:
    """Return bbox fingerprint (xmin,ymin,zmin,xmax,ymax,zmax) per slab ShapeKey.

    Walks the pre-BOP cohort compounds to compute bounding boxes for
    every solid whose ShapeKey is in slab_meta.
    """
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    from meshwell.structured.build import _shape_key

    result: dict[ShapeKey, tuple[float, ...]] = {}
    for ce in state.cohort_entities:
        exp = TopExp_Explorer(ce.compound, TopAbs_SOLID)
        while exp.More():
            solid = exp.Current()
            sk = _shape_key(solid)
            if sk in state.slab_meta:
                box = Bnd_Box()
                BRepBndLib.Add_s(solid, box)
                if not box.IsVoid():
                    result[sk] = box.Get()
            exp.Next()
    return result


def _match_by_bbox(
    shape,
    slab_fp_by_key: dict[ShapeKey, tuple[float, ...]],
    slab_meta: "dict[ShapeKey, SlabMeta]",
    tol: float = 1e-3,
) -> "SlabMeta | None":
    """Match a post-BOP solid to a slab_meta entry by bounding box.

    Returns the SlabMeta whose pre-BOP bbox agrees with ``shape``'s
    bbox within ``tol`` on every axis, or None if no match.
    """
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    if box.IsVoid():
        return None
    g = box.Get()  # (xmin, ymin, zmin, xmax, ymax, zmax)

    best_key: "ShapeKey | None" = None
    best_volume_overlap = -1.0
    for sk, fp in slab_fp_by_key.items():
        # Check bbox corner agreement within tolerance.
        if all(abs(g[i] - fp[i]) <= tol for i in range(6)):
            # Volume of bbox overlap (approximation of "most similar").
            overlap_vol = (
                (min(g[3], fp[3]) - max(g[0], fp[0]))
                * (min(g[4], fp[4]) - max(g[1], fp[1]))
                * (min(g[5], fp[5]) - max(g[2], fp[2]))
            )
            if overlap_vol > best_volume_overlap:
                best_volume_overlap = overlap_vol
                best_key = sk

    if best_key is not None:
        return slab_meta.get(best_key)
    return None


def _copy_with(ent, shapes, idx: int):
    from meshwell.cad_occ import OCCLabeledEntity

    return OCCLabeledEntity(
        shapes=list(shapes),
        physical_name=ent.physical_name,
        index=idx,
        keep=ent.keep,
        dim=ent.dim,
        mesh_order=ent.mesh_order,
    )
