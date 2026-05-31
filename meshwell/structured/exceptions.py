"""All custom errors raised by the structured pipeline.

Each error inherits from StructuredError so callers can catch the
whole family with one except clause.
"""

from __future__ import annotations


class StructuredError(Exception):
    """Base class for all structured-pipeline errors."""


class StructuredExtrudeRequiredError(StructuredError):
    """Raised when a PolyPrism has structured=True but extrude=False."""

    def __init__(self, entity_index: int):
        self.entity_index = entity_index
        super().__init__(
            f"PolyPrism #{entity_index} has structured=True but extrude=False; "
            "only constant-XY-footprint prisms are structured-eligible."
        )


class StructuredEntityTypeError(StructuredError):
    """Raised when a non-PolyPrism entity has structured=True."""

    def __init__(self, entity_index: int, type_name: str):
        self.entity_index = entity_index
        self.type_name = type_name
        super().__init__(
            f"Entity #{entity_index} ({type_name}) has structured=True; "
            "only PolyPrism is supported as a structured entity."
        )


class StructuredZStackError(StructuredError):
    """Raised when an entity z-boundary falls inside a cohort z-interval."""

    def __init__(self, entity_index: int, z: float, cohort_index: int):
        self.entity_index = entity_index
        self.z = z
        self.cohort_index = cohort_index
        super().__init__(
            f"Entity #{entity_index} has a z-boundary at z={z} falling "
            f"strictly inside cohort #{cohort_index} while sharing XY. "
            "v1 requires all entity z-boundaries to coincide with cohort "
            "z-planes; restructure your stack to make z-boundaries explicit."
        )


class UnstructuredImprintRequiresPolyPrismError(StructuredError):
    """Raised when an unstructured entity at a structured z-plane is not a PolyPrism."""

    def __init__(self, entity_index: int, type_name: str, z: float):
        self.entity_index = entity_index
        self.type_name = type_name
        self.z = z
        super().__init__(
            f"Entity #{entity_index} ({type_name}) shares z-plane z={z} with "
            "a structured cohort but is not a PolyPrism(extrude=True); "
            "pre-cut requires a shapely polygon."
        )


class SubPolygonAssemblyError(StructuredError):
    """Raised when sub-polygon assembly fails for a cohort z-interval."""

    def __init__(self, cohort_index: int, z_interval: tuple[float, float], reason: str):
        self.cohort_index = cohort_index
        self.z_interval = z_interval
        self.reason = reason
        super().__init__(
            f"Cohort #{cohort_index} sub-polygon assembly failed at "
            f"z={z_interval}: {reason}"
        )


class CohortNonManifoldError(StructuredError):
    """Raised when a cohort sewn compound has non-manifold edges."""

    def __init__(self, cohort_index: int, edge_count: int):
        self.cohort_index = cohort_index
        self.edge_count = edge_count
        super().__init__(
            f"Cohort #{cohort_index} sewn compound has {edge_count} "
            "non-manifold edges (planner bug — internal face sharing is wrong)."
        )


class CohortShellModifiedError(StructuredError):
    """Raised when BOP modifies a pre-baked cohort shell face."""

    def __init__(
        self,
        slab_index: int,
        face_role: str,
        fragment_count: int,
    ):
        self.slab_index = slab_index
        self.face_role = face_role
        self.fragment_count = fragment_count
        super().__init__(
            f"BOP modified pre-baked cohort shell face (slab #{slab_index}, "
            f"role={face_role}); post-BOP fragment count = {fragment_count}. "
            "Either Stage 3d's pre-cut decomposition was incomplete, or the "
            "fragment_fuzzy_value needs adjustment."
        )


class StructuredLateralNLayersMismatchError(StructuredError):
    """Raised when two structured slabs sharing a lateral face have mismatched n_layers."""

    def __init__(
        self,
        slab_a: int,
        slab_b: int,
        face_tag: int,
        n_layers_a: int,
        n_layers_b: int,
    ):
        self.slab_a = slab_a
        self.slab_b = slab_b
        self.face_tag = face_tag
        self.n_layers_a = n_layers_a
        self.n_layers_b = n_layers_b
        super().__init__(
            f"Lateral face #{face_tag} shared between structured slabs "
            f"#{slab_a} (n_layers={n_layers_a}) and #{slab_b} "
            f"(n_layers={n_layers_b}); n_layers must match."
        )


class StructuredTransfiniteRejectedError(StructuredError):
    """Raised when gmsh rejects a transfinite hint on a lateral face."""

    def __init__(self, face_tag: int, slab_index: int, reason: str):
        self.face_tag = face_tag
        self.slab_index = slab_index
        self.reason = reason
        super().__init__(
            f"gmsh rejected transfinite hint on lateral face #{face_tag} "
            f"(slab #{slab_index}): {reason}"
        )


class WedgeCountMismatchError(StructuredError):
    """Raised when the number of emitted wedges does not match expectations."""

    def __init__(self, slab_index: int, expected: int, got: int):
        self.slab_index = slab_index
        self.expected = expected
        self.got = got
        super().__init__(
            f"Wedge stamp for slab #{slab_index}: expected {expected} "
            f"wedges (bot triangles x n_layers), emitted {got}."
        )


class WedgeBotNodeMismatchError(StructuredError):
    """Raised when bot vertices do not match the bot face mesh node tags."""

    def __init__(self, slab_index: int, mismatched_count: int):
        self.slab_index = slab_index
        self.mismatched_count = mismatched_count
        super().__init__(
            f"Wedge stamp for slab #{slab_index}: {mismatched_count} bot "
            "vertices did not match the bot face mesh node tags."
        )
