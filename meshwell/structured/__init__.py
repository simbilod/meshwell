"""Clean structured-polyprism pipeline.

Public surface:

- :class:`StructuredExtrusionResolutionSpec` -- attach to a
  ``PolyPrism(structured=True)`` to specify per-z-interval layer counts.
- :func:`build_plan` -- orchestrator-facing entry point: validates
  structured entities, returns a frozen ``StructuredPlan`` for the
  CAD + mesh stages.
"""
from __future__ import annotations

from meshwell.structured.plan import build_plan
from meshwell.structured.spec import StructuredExtrusionResolutionSpec

__all__ = ["StructuredExtrusionResolutionSpec", "build_plan"]
