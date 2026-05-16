"""Data model for the clean structured-polyprism pipeline.

Phase 1 ships only ``StructuredExtrusionResolutionSpec`` and the CAD-stage
``Slab`` / ``OverlapPair`` / ``StructuredPlan`` dataclasses. The PhantomMap
(Layer B) and StructuredMeshPlan (mesh-stage) land in Phase 2 / Phase 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from shapely.geometry import MultiPolygon, Polygon


class StructuredExtrusionResolutionSpec(BaseModel):
    """Per-z-interval layer counts for a structured ``PolyPrism``.

    Attached via ``PolyPrism(..., structured=True, resolutions=[spec])``.

    Attributes:
        n_layers: One positive integer per z-interval of the owning
            ``PolyPrism``. Length must equal ``len(buffers) - 1`` where
            ``buffers`` is the prism's z-keys dict. Enforced at planner
            time (not here, because the spec has no reference to the
            owning entity).
        recombine: When True, the slab volume is meshed with hex
            elements (gmsh element type 5) instead of wedges
            (element type 6).
    """

    model_config = ConfigDict(frozen=True)

    n_layers: list[int] = Field(
        ..., min_length=1, description="positive layer counts, one per z-interval"
    )
    recombine: bool = False

    @field_validator("n_layers")
    @classmethod
    def _all_positive(cls, v: list[int]) -> list[int]:
        for i, n in enumerate(v):
            if n <= 0:
                raise ValueError(
                    f"n_layers[{i}] = {n}: layer counts must be positive integers"
                )
        return v


class StructuredOverlapError(ValueError):
    """Raised when two structured slabs volumetrically overlap with mismatched z-extents.

    Policy B rejects all overlap unless z-extents match exactly within tolerance.
    """


class StructuredBufferTaperError(ValueError):
    """Raised when ``PolyPrism(structured=True)`` is used with non-zero buffers.

    Structured mode requires uniform extrusion; tapered geometry is not supported.
    """


@dataclass
class Slab:
    """One structured-polyprism z-interval, CAD-stage data only.

    Mesh parameters (``n_layers``, ``recombine``) are NOT stored here -
    they live on the resolution spec and are resolved in a second pass
    at mesh time. This keeps Slab a pure geometry+identity record.
    """

    footprint: "Polygon | MultiPolygon"
    zlo: float
    zhi: float
    physical_name: tuple[str, ...]
    source_index: int
    z_interval_index: int
    mesh_order: float
    identify_arcs: bool = False
    min_arc_points: int = 4
    arc_tolerance: float = 1e-3
    fragment_fuzzy_value: float | None = None
    # Populated by compute_face_partition (default: one piece = the whole footprint).
    face_partition: list["Polygon"] = field(default_factory=list)


@dataclass(frozen=True)
class OverlapPair:
    """Record of a Policy-B-resolved volumetric overlap.

    The winner slab is in ``StructuredPlan.slabs``; the loser was
    dropped during planning. Mesh stage uses this to verify the loser
    spec's n_layers agreed with the winner's.
    """

    winner_slab_index: int
    loser_source_index: int
    loser_z_interval_index: int
    z_extent: tuple[float, float]


@dataclass(frozen=True)
class StructuredPlan:
    """Frozen output of the planner; consumed by phantom + builder stages."""

    slabs: tuple[Slab, ...]
    z_planes: tuple[float, ...]
    overlaps: tuple[OverlapPair, ...]


Side = Literal["bot", "top"]


@dataclass(frozen=True)
class FaceKey:
    """Identifies an input face by slab/side/piece.

    Survives BOP because it indexes by piece identity, not by OCC tag.
    """

    slab_index: int
    side: Side
    piece_index: int


@dataclass(frozen=True)
class EdgeKey:
    """Identifies an input boundary edge on a piece face."""

    slab_index: int
    side: Side
    piece_index: int
    edge_index: int


@dataclass(frozen=True)
class VertexKey:
    """Identifies an input boundary corner on a piece face."""

    slab_index: int
    side: Side
    piece_index: int
    corner_index: int


@dataclass(frozen=True)
class LateralKey:
    """Identifies an input lateral face on a slab.

    ``outer_edge_index`` indexes into the slab's union-footprint outer
    boundary.
    """

    slab_index: int
    outer_edge_index: int


@dataclass
class PhantomShape:
    """One partition piece's input OCC bookkeeping.

    ``solid`` is the TopoDS_Solid produced by ``BRepPrimAPI_MakePrism``.
    The four ``input_*`` dicts map our Phase-2 key types to the
    corresponding input OCC sub-shapes (TopoDS_Face / TopoDS_Edge /
    TopoDS_Vertex), captured at construction time so we can ask BOP
    history what they became.

    The values are OCC ``TopoDS_*`` objects (declared ``Any`` to avoid
    importing OCP at type-check time when callers may not have it).
    """

    slab_index: int
    piece_index: int
    solid: Any
    input_faces_by_key: dict[FaceKey, Any]
    input_edges_by_key: dict[EdgeKey, Any]
    input_vertices_by_key: dict[VertexKey, Any]
    # Lateral faces are not piece-scoped on the union footprint — they're
    # slab-scoped. But each piece may own zero or more lateral faces
    # (those that coincide with an outer-boundary edge segment). The dict
    # is keyed by the outer_edge_index they map to.
    input_laterals_by_outer_edge: dict[int, Any]


@dataclass(frozen=True)
class PhantomBuildResult:
    """Output of ``build_phantom_shapes(plan)``.

    Contains one ``PhantomShape`` per (slab, piece) pair, in deterministic
    order (slab_index ascending, then piece_index ascending).
    """

    shapes: tuple[PhantomShape, ...]


@dataclass
class PhantomMap:
    """Post-BOP correspondence map.

    Each input element maps to a *list* of output OCC tags because
    BOP may split a single input into many output shapes (e.g. a
    neighbour cut a piece's top face into 3 sub-faces).
    """

    output_faces: dict[FaceKey, list[Any]] = field(default_factory=dict)
    output_edges: dict[EdgeKey, list[Any]] = field(default_factory=dict)
    output_vertices: dict[VertexKey, list[Any]] = field(default_factory=dict)
    output_laterals: dict[LateralKey, list[Any]] = field(default_factory=dict)
    # Per-lateral flag: True iff BOP introduced a new vertex on the
    # lateral face with z strictly between zlo and zhi (a "mid-height
    # cut"). Phase 3's builder uses this to decide which lateral faces
    # to exclude from transfinite hints.
    lateral_has_midheight_cut: dict[LateralKey, bool] = field(default_factory=dict)


class StructuredMeshOverlapError(ValueError):
    """Raised when an overlap-pair winner and loser have different n_layers.

    Plan-stage Policy B catches direct overlap mismatches at slab
    construction. Mesh-stage catches the case where an OverlapPair
    records a winner/loser whose spec n_layers actually disagree —
    paranoid double-check before we commit to the loser-was-dominated
    decision.
    """


@dataclass(frozen=True)
class StructuredMeshPlan:
    """Output of ``resolve_mesh_plan(plan, entities)``.

    Carries the mesh-stage parameters resolved from each slab's owning
    ``StructuredExtrusionResolutionSpec``. Parallel arrays: index i in
    ``n_layers`` / ``recombine`` corresponds to ``plan.slabs[i]``.
    """

    slabs: tuple[Slab, ...]
    n_layers: tuple[int, ...]
    recombine: tuple[bool, ...]
