"""Structured prism meshing for meshwell.cad_occ.

User-facing: set ``structured=True`` on a PolyPrism, then attach a
``StructuredExtrusionResolutionSpec(n_layers=N)`` to the prism's
physical_name in the ``resolution_specs`` dict passed to
``generate_mesh``. The volume is meshed with wedge elements;
surrounding unstructured regions remain tet-meshed and interfaces
are conformal by construction.
"""
from meshwell.structured.exceptions import (
    CohortNonManifoldError,
    CohortShellModifiedError,
    MixedIdentifyArcsError,
    StructuredEntityTypeError,
    StructuredError,
    StructuredExtrudeRequiredError,
    StructuredLateralNLayersMismatchError,
    StructuredTransfiniteRejectedError,
    StructuredZStackError,
    SubPolygonAssemblyError,
    UnstructuredImprintRequiresPolyPrismError,
    WedgeBotNodeMismatchError,
    WedgeCountMismatchError,
)

__all__ = [
    "CohortNonManifoldError",
    "CohortShellModifiedError",
    "MixedIdentifyArcsError",
    "StructuredEntityTypeError",
    "StructuredError",
    "StructuredExtrudeRequiredError",
    "StructuredLateralNLayersMismatchError",
    "StructuredTransfiniteRejectedError",
    "StructuredZStackError",
    "SubPolygonAssemblyError",
    "UnstructuredImprintRequiresPolyPrismError",
    "WedgeBotNodeMismatchError",
    "WedgeCountMismatchError",
]
