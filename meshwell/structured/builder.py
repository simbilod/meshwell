"""Phase-3: mesh-stage builder (Layer C).

Public entry points (added incrementally):

- :func:`resolve_mesh_plan` — second-pass over the spec list to attach
  ``n_layers`` and ``recombine`` to each slab; also cross-checks
  Phase-1 OverlapPairs.
- :func:`apply_structured_mesh` — full mesh-stage execution: stamp
  derived top meshes, build discrete 3D entities per slab, run global
  removeDuplicateNodes.
"""
from __future__ import annotations

from typing import Any

from meshwell.structured.spec import (
    StructuredExtrusionResolutionSpec,
    StructuredMeshOverlapError,
    StructuredMeshPlan,
    StructuredPlan,
)


def _spec_of(entity: Any) -> StructuredExtrusionResolutionSpec | None:
    for r in getattr(entity, "resolutions", None) or []:
        if isinstance(r, StructuredExtrusionResolutionSpec):
            return r
    return None


def resolve_mesh_plan(plan: StructuredPlan, entities: list[Any]) -> StructuredMeshPlan:
    """Look up (n_layers, recombine) for each slab via its owning spec.

    Cross-checks every ``OverlapPair`` in the plan: if the loser slab's
    spec n_layers != the winner's, raises
    ``StructuredMeshOverlapError``. This is a paranoid double-check;
    Phase-1's Policy B already catches direct mismatches at plan time.
    """
    n_layers_list: list[int] = []
    recombine_list: list[bool] = []
    for slab in plan.slabs:
        owner = entities[slab.source_index]
        spec = _spec_of(owner)
        if spec is None:
            raise StructuredMeshOverlapError(
                f"Slab {slab.physical_name} source entity has no "
                f"StructuredExtrusionResolutionSpec attached."
            )
        n_layers_list.append(int(spec.n_layers[slab.z_interval_index]))
        recombine_list.append(bool(spec.recombine))

    for op in plan.overlaps:
        winner = plan.slabs[op.winner_slab_index]
        winner_spec = _spec_of(entities[winner.source_index])
        loser_owner = entities[op.loser_source_index]
        loser_spec = _spec_of(loser_owner)
        if winner_spec is None or loser_spec is None:
            continue
        winner_n = winner_spec.n_layers[winner.z_interval_index]
        loser_n = loser_spec.n_layers[op.loser_z_interval_index]
        if winner_n != loser_n:
            raise StructuredMeshOverlapError(
                f"OverlapPair winner {winner.physical_name} "
                f"(n_layers={winner_n}) and loser source_index="
                f"{op.loser_source_index} (n_layers={loser_n}) "
                f"at z={op.z_extent}: n_layers must match for the "
                f"overlap to be valid."
            )

    return StructuredMeshPlan(
        slabs=plan.slabs,
        n_layers=tuple(n_layers_list),
        recombine=tuple(recombine_list),
    )
