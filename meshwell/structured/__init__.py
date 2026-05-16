"""Clean structured-polyprism pipeline.

Public surface:

- :class:`StructuredExtrusionResolutionSpec` -- attach to a
  ``PolyPrism(structured=True)`` to specify per-z-interval layer counts.
- :func:`build_plan` -- planner entry point: validates structured
  entities, returns a frozen ``StructuredPlan`` for the CAD + mesh stages.
- :func:`build_phantom_shapes` -- CAD-stage Layer A: build one OCP
  sub-prism per partition piece.
- :func:`extract_phantom_map` -- CAD-stage Layer B: walk
  ``BOPAlgo_Builder`` history to produce the ``PhantomMap`` consumed
  by the mesh stage.
- :class:`PhantomMap` -- the post-BOP correspondence map.
"""
from __future__ import annotations

from meshwell.structured.phantom import build_phantom_shapes, extract_phantom_map
from meshwell.structured.plan import build_plan
from meshwell.structured.spec import PhantomMap, StructuredExtrusionResolutionSpec

__all__ = [
    "PhantomMap",
    "StructuredExtrusionResolutionSpec",
    "build_phantom_shapes",
    "build_plan",
    "extract_phantom_map",
]
