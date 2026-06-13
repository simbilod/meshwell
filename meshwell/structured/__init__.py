"""Structured prism meshing for meshwell.cad_occ.

User-facing: set ``structured=True`` on a PolyPrism, then attach a
``StructuredExtrusionResolutionSpec(n_layers=N)`` to the prism's
physical_name in the ``resolution_specs`` dict passed to
``generate_mesh``. The volume is meshed with wedge elements;
surrounding unstructured regions remain tet-meshed and interfaces
are conformal by construction.

The public surface of this package is the structured-meshing exception
family, all subclasses of ``StructuredError``.
"""
from meshwell.structured.exceptions import (
    CanonicalArrangementError,
    CohortNotWrappedError,
    CohortShellModifiedError,
    MixedIdentifyArcsError,
    StructuredEntityTypeError,
    StructuredError,
    StructuredExtrudeRequiredError,
    StructuredLateralNLayersMismatchError,
    StructuredTransfiniteRejectedError,
    StructuredVoidMeshOrderRequiredError,
    StructuredVolumetricOverlapError,
    StructuredZStackError,
    WedgeBotNodeMismatchError,
    WedgeCountMismatchError,
)

__all__ = [
    "CanonicalArrangementError",
    "CohortNotWrappedError",
    "CohortShellModifiedError",
    "MixedIdentifyArcsError",
    "StructuredEntityTypeError",
    "StructuredError",
    "StructuredExtrudeRequiredError",
    "StructuredLateralNLayersMismatchError",
    "StructuredTransfiniteRejectedError",
    "StructuredVoidMeshOrderRequiredError",
    "StructuredVolumetricOverlapError",
    "StructuredZStackError",
    "WedgeBotNodeMismatchError",
    "WedgeCountMismatchError",
]
