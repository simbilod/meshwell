"""Clean structured-polyprism pipeline.

Public surface:

- :class:`StructuredExtrusionResolutionSpec` -- attach to a
  ``PolyPrism(structured=True)`` to specify per-z-interval layer counts.
- :func:`build_plan` -- planner entry point.
- :func:`build_phantom_shapes` -- CAD-stage Layer A.
- :func:`extract_phantom_map` -- CAD-stage Layer B.
- :class:`PhantomMap` -- post-BOP correspondence map.
- :func:`resolve_mesh_plan` -- mesh-stage parameter resolver.
- :func:`apply_structured_mesh` -- mesh-stage Layer C entry point.
- :class:`StructuredMeshPlan` -- output of resolve_mesh_plan.
"""
from __future__ import annotations

from meshwell.structured.builder import apply_structured_mesh, resolve_mesh_plan
from meshwell.structured.phantom import build_phantom_shapes, extract_phantom_map
from meshwell.structured.plan import build_plan
from meshwell.structured.spec import (
    PhantomMap,
    StructuredArcSplitError,
    StructuredExtrusionResolutionSpec,
    StructuredLateralUnstructuredNeighbourError,
    StructuredMeshPlan,
    StructuredMidHeightCutError,
    StructuredOverlapError,
    StructuredPartitionConvergenceError,
)

__all__ = [
    "PhantomMap",
    "StructuredArcSplitError",
    "StructuredExtrusionResolutionSpec",
    "StructuredLateralUnstructuredNeighbourError",
    "StructuredMeshPlan",
    "StructuredMidHeightCutError",
    "StructuredOverlapError",
    "StructuredPartitionConvergenceError",
    "apply_structured_mesh",
    "build_phantom_shapes",
    "build_plan",
    "extract_phantom_map",
    "resolve_mesh_plan",
]
