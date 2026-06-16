"""Dataclasses shared across all structured-pipeline stages.

Kept in one module so importers don't have to know which stage owns
which type. All dataclasses are frozen — these records flow through
the pipeline immutably.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shapely.geometry import MultiPolygon, Polygon

    from meshwell.geometry_entity import DecompositionSegment


@dataclass(frozen=True)
class ShapeKey:
    """Stable identity for a TopoDS_Shape used as a dict key.

    TShape pointer + orientation matches cad_occ._shape_key. We
    redeclare it here as a frozen dataclass so it's pickle-safe and
    type-checkable.
    """

    tshape_id: int
    orientation: int


@dataclass(frozen=True)
class StructuredSlab:
    """One z-interval of one structured PolyPrism.

    A single PolyPrism with N+1 buffer keys yields N StructuredSlab
    records, one per adjacent (zlo, zhi) pair.

    n_layers is intentionally absent — it is read at mesh time from
    the StructuredExtrusionResolutionSpec attached to physical_name.
    """

    source_index: int
    footprint: "Polygon | MultiPolygon"
    zlo: float
    zhi: float
    mesh_order: float | None
    mesh_bool: bool
    physical_name: tuple[str, ...]
    identify_arcs: bool
    arc_tolerance: float
    min_arc_points: int


@dataclass(frozen=True)
class Cohort:
    """Connected component of structured slabs (Union-Find)."""

    slabs: tuple[StructuredSlab, ...]
    z_planes: tuple[float, ...]  # sorted unique cohort z-boundaries

    @property
    def zmin(self) -> float:
        """Return the lowest z boundary of this cohort."""
        return self.z_planes[0]

    @property
    def zmax(self) -> float:
        """Return the highest z boundary of this cohort."""
        return self.z_planes[-1]


@dataclass(frozen=True)
class SubPiece:
    """One (z-interval x sub-polygon) cell after decomposition.

    Each SubPiece becomes one TopoDS_Solid in the cohort compound.
    """

    cohort_index: int
    z_interval: tuple[float, float]
    sub_polygon: "Polygon"
    source_slab_indices: tuple[int, ...]


@dataclass(frozen=True)
class SlabMeta:
    """Per-sub-solid metadata used at meshing time.

    Lookup happens by post-BOP ShapeKey of the sub-solid in the
    OCCLabeledEntity's shapes list. n_layers is NOT here — wedge.py
    resolves it from the resolution_specs dict via physical_name.

    `keep` mirrors the source slab's mesh_bool: True for solids whose
    wedges should be stamped, False for voids whose body must be excluded
    from BREP serialization (XAO writer keep=False path).
    """

    slab_index: int
    physical_name: tuple[str, ...]
    bot_face_key: ShapeKey
    top_face_key: ShapeKey
    lateral_face_keys: tuple[ShapeKey, ...]
    keep: bool = True


# Quantized vertex key as used by VertexRegistry._key.
VertexKey = tuple[int, int, int]


@dataclass(frozen=True)
class ArrangementEdge:
    """Canonical curve between two arrangement nodes.

    Arc/line decomposition is fit ONCE on this edge's coords via
    ``meshwell.geometry_entity.decompose_vertices_2d`` and stored in
    ``segments``. Every sub-piece whose ring traverses this edge
    replays these segments instead of running the greedy fitter on its
    own ring — eliminating the seam-dependent mismatches.

    ``vertex_keys`` is stored OPEN even when ``is_closed=True``
    (``vertex_keys[0] != vertex_keys[-1]``); the implicit closing pair
    is registered in ``Arrangement.edge_by_vertex_pair`` only for
    OPEN edges. Closed standalone edges (e.g., a lone disc boundary
    with no other arrangement nodes) are NOT indexed — sub-pieces
    traversing them fall back to the per-ring greedy fit, which is
    already deterministic for a single closed ring.
    """

    vertex_keys: tuple["VertexKey", ...]
    z: float
    segments: tuple[DecompositionSegment, ...] = ()
    is_closed: bool = False


@dataclass(eq=False)
class Arrangement:
    """Cohort-global polygon arrangement.

    `polygons` is the canonical, ordered tuple of shapely.Polygon objects
    produced by one polygonize call over the union of:
      - every cohort slab boundary, and
      - every adjacent unstructured PolyPrism boundary projected to the
        shared z-planes.

    `canonical_edges` and `edge_by_vertex_pair` carry the arrangement's
    unique boundary edges with arcs fit ONCE per edge. Sub-piece wire
    builders look up each consecutive vertex pair in
    `edge_by_vertex_pair` and replay the stored `segments` so two
    sub-pieces sharing an arc-shaped boundary subset emit the same
    OCC TShape by construction.

    Cohort sub-piece extraction consumes this tuple to build each
    sub-solid's boundary wires.

    Identity contract:
    - When a downstream consumer receives a single Polygon (e.g., a
      SubPiece's `sub_polygon` field), it is the SAME Python object
      (`is`) as the matching entry in `polygons`.
    - When the consumer receives a `MultiPolygon`, Shapely 2.x's
      `.geoms` accessor returns fresh Polygon wrappers each access, so
      Python `is` is NOT preserved. However, the underlying GEOS
      coordinate sequences are shared by reference: vertex coordinates
      are bit-exactly equal (`equals_exact(member, arrangement_poly,
      tolerance=0.0)`). Downstream OCC builders that key polygons by
      coordinate hash get identical hashes from both consumers.

    Note: ``edge_by_vertex_pair`` is a mutable dict, so this dataclass
    uses ``eq=False`` rather than ``frozen=True``. That inherits both
    identity-based equality and identity-based hashing from ``object``,
    keeping ``Arrangement`` usable as a dict key without violating
    Python's ``a == b → hash(a) == hash(b)`` contract.
    """

    cohort_index: int
    polygons: tuple["Polygon", ...]
    canonical_edges: tuple[ArrangementEdge, ...] = ()
    edge_by_vertex_pair: dict[frozenset["VertexKey"], int] = field(default_factory=dict)
