"""Plan stage for the clean structured-polyprism pipeline.

Public surface: ``build_plan(entities) -> StructuredPlan``. Private
helpers handle the pipeline steps (gather, expand, validate, partition).
"""
from __future__ import annotations

from typing import Any

from meshwell.structured.spec import StructuredExtrusionResolutionSpec


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
