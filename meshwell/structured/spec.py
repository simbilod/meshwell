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
        recombine_lateral_faces: When True, the slab's lateral OCC faces
            are recombined into quads (gmsh ``setRecombine`` after the
            existing ``setTransfiniteSurface``). Wedge volume cells stay
            as wedges (use ``recombine`` for hex volumes). Use this when
            a downstream solver requires the wedge boundary-face mesh to
            match the wedge element topology (i.e. quads on the three
            lateral faces) instead of the gmsh-default triangulated
            lateral surface mesh. Only safe when each lateral face is
            either external or shared with another structured slab; if a
            tet-meshed neighbour shares the lateral, the recombined
            quads will not match the tet's triangular boundary faces and
            the mesh will be non-conformal across that interface.
    """

    model_config = ConfigDict(frozen=True)

    n_layers: list[int] = Field(
        ..., min_length=1, description="positive layer counts, one per z-interval"
    )
    recombine: bool = False
    recombine_lateral_faces: bool = False

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


class StructuredMidHeightCutError(ValueError):
    """Raised when a neighbour entity would cut a structured slab mid-height.

    A neighbour with z-endpoint (zmin or zmax) strictly inside a structured
    slab's z-extent and with xy footprint intersecting the slab's footprint
    would introduce vertices on the slab's lateral OCC faces at intermediate
    z values. The structured mesh stage cannot produce a conformal wedge
    grid in that case (would need tet/pyramid bridging — Phase 6+).

    Remediation: add the neighbour's z-endpoint as an additional buffer-key
    on the structured polyprism so it becomes a slab boundary (with explicit
    n_layers split), or move the neighbour to share the slab's existing
    zlo/zhi planes.
    """


class StructuredBufferTaperError(ValueError):
    """Raised when ``PolyPrism(structured=True)`` is used with non-zero buffers.

    Structured mode requires uniform extrusion; tapered geometry is not supported.
    """


class StructuredLateralUnstructuredNeighbourError(ValueError):
    """Raised when an unstructured neighbour shares a lateral face with a structured slab.

    A structured slab is meshed with wedge cells (or hex when
    ``recombine=True``). Wedge lateral faces are topologically quads; hex
    lateral faces are quads. An unstructured (tet-meshed) neighbour
    sharing a portion of the slab's lateral surface produces a quad/tri
    face-topology mismatch at the shared interface — non-conformal at
    the volume-element level. Even when gmsh writes the surface mesh as
    triangles on both sides, the underlying wedge cell still has a quad
    lateral face that no tet face matches one-to-one.

    Detected when an entity is not structured, its z-range overlaps the
    slab's [zlo, zhi] with positive extent, and its xy footprint shares
    1D contact with the slab's footprint boundary.

    Remediation (any of):
      - Make the neighbour structured as well (matching footprint
        adjacency and z-planes).
      - Move the neighbour so its xy footprint is disjoint from the
        slab's boundary, or fully separate in z.
      - Future: pyramid-layer bridging between wedge and tet regions
        (not implemented).
    """


class StructuredPartitionConvergenceError(RuntimeError):
    """face_partition fixed-point iteration did not converge within the iteration cap."""


class StructuredArcSplitError(ValueError):
    """Raised when a structured arc slab is split at a non-polygon-vertex position.

    When a neighbour entity's boundary crosses one of the slab's arc edges
    at a position that is NOT a polygon vertex of the original arc, the
    planner's polygon-based partition lands at the polygon-edge crossing
    (chord intersection), while OCC's BOP later cuts the true arc at the
    geometric intersection point. The two differ by a few percent of the
    polygon edge length; BOP then introduces extra OCC sub-faces and
    micro-vertices that the planner did not predict, breaking the
    structured-wedge construction (manifests as 6-corner laterals or
    duplicate-node mappings downstream).

    Remediation (any of):
      - Densify the arc footprint so a polygon vertex falls on the
        neighbour boundary (the disc polygon vertex at angle θ has
        x = r * cos(θ), so choose neighbour cut x accordingly).
      - Align the neighbour boundary with an existing polygon vertex
        (e.g. cut at x=0 for a 32-vertex disc whose vertices include
        (0, r) and (0, -r)).
      - Move the neighbour off the arc footprint entirely.
      - Drop identify_arcs=True for the slab (use the polygonal
        approximation throughout).

    Phase 6(a3) future work would project true arc/seam intersection
    points into the piece polygons at plan time, eliminating the need
    for this validation.
    """


@dataclass(frozen=True)
class CanonicalCircle:
    """Identity of a circular curve shared across arrangement edges.

    Two arrangement edges with CanonicalCircle instances matching on
    (center, radius) within arc_tolerance are sub-arcs of the same
    physical circle. The phantom builder uses (center, radius) plus arc
    endpoints to construct OCC arc geometry; consumers of the same
    circle produce bit-identical TShapes.
    """

    center: tuple[float, float]
    radius: float


@dataclass(frozen=True)
class ArrangementEdge:
    """One non-crossing curve segment in the planar arrangement.

    vertices: ordered XY sample points; >=2 elements. Endpoints are
        vertices[0] and vertices[-1].
    circle: None means a straight line. Not None means a sub-arc of
        the named circle; endpoints lie on it.
    """

    edge_id: int
    vertices: tuple[tuple[float, float], ...]
    circle: "CanonicalCircle | None"


@dataclass
class ArrangementFace:
    """One face of the planar arrangement (a Polygon with no interior holes).

    boundary: ordered list of (edge_id, reversed) tuples describing the
        traversal of the face's outer ring. ``reversed=True`` means the
        edge's vertex sequence is walked in reverse.
    """

    face_id: int
    polygon: "Polygon"
    boundary: list[tuple[int, bool]]


@dataclass
class StackArrangement:
    """Per-z-touching-component planar arrangement; consumed by face-partition assignment."""

    edges: list[ArrangementEdge]
    faces: list[ArrangementFace]


@dataclass(frozen=True)
class PieceArcEdge:
    """One arc segment on a piece boundary.

    ``points``: ordered (x, y, z) coords of all polygon vertices that lie
    on this arc segment, in boundary traversal order. At least 2 points.
    ``center``: (cx, cy, cz) of the fitted circle.
    ``radius``: arc radius.
    """

    points: tuple[tuple[float, float, float], ...]
    center: tuple[float, float, float]
    radius: float


@dataclass(frozen=True)
class PieceLineEdge:
    """One straight line segment on a piece boundary.

    ``points``: ordered (x, y, z) coords. Exactly 2 points (start, end).
    A straight cut introduced by the partition seam, OR an original
    line segment from the slab footprint (when identify_arcs is False
    or the segment didn't fit a circle).
    """

    points: tuple[tuple[float, float, float], tuple[float, float, float]]


@dataclass
class PieceProvenance:
    """Labelled boundary of one face_partition piece.

    ``exterior_edges``: ordered list of arc/line edges along the
    piece's outer ring, in boundary traversal order.
    ``interior_edges``: list of lists (one per interior ring, in order).
    Each inner-ring entry is its own ordered list of arc/line edges.

    A piece with no holes has ``interior_edges = []``.
    """

    exterior_edges: list["PieceArcEdge | PieceLineEdge"]
    interior_edges: list[list["PieceArcEdge | PieceLineEdge"]]


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
    arc_tolerance: float = 1e-3  # legacy absolute backstop
    arc_chord_height_fraction: float = 0.01  # 1% of arc radius
    fragment_fuzzy_value: float | None = None
    # Populated by compute_face_partition (default: one piece = the whole footprint).
    face_partition: list["Polygon"] = field(default_factory=list)
    # Per-piece edge provenance, parallel to face_partition: entry
    # face_partition_provenance[i] describes the boundary edges of
    # face_partition[i]. None means "treat every edge as a straight line"
    # (matching the existing polyline-only behaviour).
    face_partition_provenance: "list[PieceProvenance] | None" = None
    # Populated by Step 0 (sub-level mesh-order resolution) of the new
    # planar-arrangement pipeline. Equal to footprint when no carving
    # applies. None until Step 0 runs.
    resolved_footprint: "Polygon | MultiPolygon | None" = None
    # Populated by Step F (assign-faces-to-slabs). Parallel to face_partition;
    # face_partition_edges[i] is the boundary of face_partition[i] expressed
    # as (edge_id, reversed) tuples into the slab's stack arrangement.
    face_partition_edges: "list[list[tuple[int, bool]]] | None" = None
    # Populated by build_plan via _assign_component_indices. Slabs in the
    # same connected-z-component (same StackArrangement) share an index.
    # Used by the phantom builder to pre-share TopoDS_Face between
    # vertically-stacked sub-prisms (see spec
    # 2026-05-27-cad-occ-cohort-preshared-faces-design.md).
    component_index: int = 0


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
    # Populated by build_plan. Maps component_index -> StackArrangement for
    # each connected z-component. Consumed by the cohort topology builder
    # (Phase 2) to walk arrangement edges and detect cohort-interior vs
    # cohort-exterior boundaries. See spec
    # 2026-05-27-cad-occ-cohort-topology-builder-design.md.
    arrangements: "dict[int, StackArrangement]" = field(default_factory=dict)


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
    """Identifies an input lateral face on a slab piece.

    ``piece_index`` disambiguates per-piece outer edges so that two pieces
    of the same slab both numbered outer_edge_index=0 do not collide.
    ``outer_edge_index`` indexes into that piece's outer boundary.
    """

    slab_index: int
    piece_index: int
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
    ``n_layers`` / ``recombine`` / ``recombine_lateral_faces``
    corresponds to ``plan.slabs[i]``.
    """

    slabs: tuple[Slab, ...]
    n_layers: tuple[int, ...]
    recombine: tuple[bool, ...]
    recombine_lateral_faces: tuple[bool, ...]
