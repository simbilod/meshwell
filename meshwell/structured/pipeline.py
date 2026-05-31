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
from meshwell.structured.validators import validate_z_stacks


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
    ShapeKey (cad_occ already preserved sub-solid TShapes via the
    fragment piece-ownership pass). One OCCLabeledEntity per
    sub-solid, carrying the source slab's physical_name and a
    synthetic index.
    """
    from meshwell.cad_occ import OCCLabeledEntity
    from meshwell.structured.build import _shape_key

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
                # Sub-solid was modified by BOP; still represent it.
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
