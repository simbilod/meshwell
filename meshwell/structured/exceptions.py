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


class StructuredVolumetricOverlapError(StructuredError):
    """Unstructured entity volumetrically overlaps a cohort.

    v1 requires cohorts and unstructured entities to live in disjoint
    3D volumes (sharing only z-plane boundaries). Place the entity
    above, below, or to the side of the cohort, or split the cohort.
    """

    def __init__(self, entity_index, cohort_index, z_overlap):
        self.entity_index = entity_index
        self.cohort_index = cohort_index
        self.z_overlap = z_overlap
        super().__init__(
            f"Entity #{entity_index} has volumetric overlap (Δz={z_overlap:.4g}) "
            f"with cohort #{cohort_index} AND XY-intersection in the overlap. "
            "v1 does not support unstructured material occupying cohort space. "
            "Restructure so the entity sits above, below, or laterally outside the cohort."
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


class StructuredVoidMeshOrderRequiredError(StructuredError):
    """A structured void (mesh_bool=False) must declare mesh_order.

    Without an explicit mesh_order, the void would sort last in the
    Policy B resolution (mesh_order=None -> float("inf")) and carve
    solids that already ran. Voids must explicitly state their priority
    against the solids around them.
    """

    def __init__(self, entity_index: int, physical_name: tuple[str, ...] | str):
        self.entity_index = entity_index
        self.physical_name = physical_name
        super().__init__(
            f"Structured void at entity #{entity_index} "
            f"(physical_name={physical_name!r}) has mesh_bool=False but no "
            "mesh_order. Voids must declare an explicit mesh_order so "
            "Policy B can resolve them against neighbouring solids."
        )


class MixedIdentifyArcsError(StructuredError):
    """Some PolyPrisms have identify_arcs=True and others False.

    When structured entities are present and any PolyPrism opts into
    arc detection, ALL PolyPrisms must opt in. Otherwise an arc-bearing
    boundary can be shared with a polyline-bearing one through the
    cohort pre-cut, producing geometrically-coincident but topologically
    distinct OCC edges that BOP cannot merge.
    """

    def __init__(
        self,
        arcs_true: list[tuple[int, tuple[str, ...] | str]],
        arcs_false: list[tuple[int, tuple[str, ...] | str]],
    ):
        self.arcs_true = arcs_true
        self.arcs_false = arcs_false
        true_desc = ", ".join(f"#{i} ({n!r})" for i, n in arcs_true)
        false_desc = ", ".join(f"#{i} ({n!r})" for i, n in arcs_false)
        super().__init__(
            "Mixed identify_arcs across PolyPrisms with structured entities "
            "present.\n"
            f"  identify_arcs=True : {true_desc}\n"
            f"  identify_arcs=False: {false_desc}\n"
            "Set identify_arcs=True on every PolyPrism, or remove it from "
            "all of them. Mixing leads to inconsistent OCC edges where "
            "shared boundaries are pre-cut by Stage 3d."
        )
