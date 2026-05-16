"""Clean structured-polyprism pipeline.

Public surface:

- :class:`StructuredExtrusionResolutionSpec` -- attach to a
  ``PolyPrism(structured=True)`` to specify per-z-interval layer counts.

CAD-stage and mesh-stage internals (``Slab``, ``StructuredPlan``,
``OverlapPair``, the planner) live in submodules and are loaded on demand
by the orchestrator. They are not part of the public surface for end
users in Phase 1.
"""
from __future__ import annotations


def __getattr__(name: str):
    if name == "StructuredExtrusionResolutionSpec":
        from meshwell.structured.spec import StructuredExtrusionResolutionSpec

        return StructuredExtrusionResolutionSpec
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["StructuredExtrusionResolutionSpec"]
