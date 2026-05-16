"""Plan stage for the clean structured-polyprism pipeline.

Public surface: ``build_plan(entities) -> StructuredPlan``. Private
helpers handle the pipeline steps (gather, expand, validate, partition).
"""
from __future__ import annotations

from itertools import pairwise
from typing import Any

from meshwell.structured.spec import Slab, StructuredExtrusionResolutionSpec


def gather_structured_entities(
    entities: list[Any],
) -> list[tuple[Any, StructuredExtrusionResolutionSpec, int]]:
    """Return ``(entity, spec, source_index)`` for every structured prism.

    A structured entity is one with ``structured=True`` AND exactly one
    ``StructuredExtrusionResolutionSpec`` in its ``resolutions``. The
    validation at construction time guarantees this when both conditions
    hold, so we just retrieve here.
    """
    out: list[tuple[Any, StructuredExtrusionResolutionSpec, int]] = []
    for idx, ent in enumerate(entities):
        if not getattr(ent, "structured", False):
            continue
        resolutions = getattr(ent, "resolutions", None) or []
        specs = [
            r for r in resolutions if isinstance(r, StructuredExtrusionResolutionSpec)
        ]
        if len(specs) != 1:
            # PolyPrism construction enforces this; defensive check.
            continue
        out.append((ent, specs[0], idx))
    return out


def expand_to_slabs(
    pairs: list[tuple[Any, StructuredExtrusionResolutionSpec, int]],
) -> list[Slab]:
    """One slab per (entity, z-interval) pair.

    n_layers / recombine are NOT stored on the slab - they live on the
    spec and are resolved at mesh time via (source_index, z_interval_index).
    """
    slabs: list[Slab] = []
    for entity, _spec, source_index in pairs:
        z_keys = list(entity.buffers.keys())
        mesh_order = (
            entity.mesh_order if entity.mesh_order is not None else float("inf")
        )
        footprint = entity.polygons
        for z_idx, (zlo, zhi) in enumerate(pairwise(z_keys)):
            slabs.append(
                Slab(
                    footprint=footprint,
                    zlo=float(zlo),
                    zhi=float(zhi),
                    physical_name=entity.physical_name,
                    source_index=source_index,
                    z_interval_index=z_idx,
                    mesh_order=mesh_order,
                    identify_arcs=getattr(entity, "identify_arcs", False),
                    min_arc_points=getattr(entity, "min_arc_points", 4),
                    arc_tolerance=getattr(entity, "arc_tolerance", 1e-3),
                )
            )
    return slabs
