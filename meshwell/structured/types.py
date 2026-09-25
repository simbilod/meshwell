"""Dataclasses shared across all structured-pipeline stages.

Kept in one module so importers don't have to know which stage owns
which type. All dataclasses are frozen — these records flow through
the pipeline immutably.
"""
from __future__ import annotations

import re
from collections import defaultdict
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
class StructuredPlane:
    """Horizontal (`StructuredPolySurface`) or vertical (`InterfaceTag`) surface on a cohort."""

    source_index: int
    orientation: str  # "horizontal" | "vertical"
    footprint: object  # Polygon | MultiPolygon (horizontal) or LineString | MultiLineString (vertical)
    zmin: float
    zmax: float
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
    planes: tuple[StructuredPlane, ...] = ()

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


_SUB_KEY_RE = re.compile(r"^__cohort_\d+__slab_\d+$")
_FACE_NAME_RE = re.compile(r"^(__cohort_\d+__slab_\d+)__(bot|top|lat_\d+)$")


@dataclass(frozen=True)
class SlabMeta:
    """Per-sub-solid metadata used at meshing time.

    Lookup happens by post-BOP ShapeKey of the sub-solid in the
    OCCLabeledEntity's shapes list, or by synthetic ``sub_key``
    (``__cohort_{ci}__slab_{si}``) when reconstructed from a loaded XAO.
    n_layers is NOT here — wedge.py resolves it from the resolution_specs
    dict via physical_name.

    `keep` mirrors the source slab's mesh_bool: True for solids whose
    wedges should be stamped, False for voids whose body must be excluded
    from BREP serialization (XAO writer keep=False path).
    """

    slab_index: int
    physical_name: tuple[str, ...]
    bot_face_key: ShapeKey | str
    top_face_key: ShapeKey | str
    lateral_face_keys: tuple[ShapeKey | str, ...]
    keep: bool = True

    def to_synthetic_solid_name(self, cohort_index: int, sub_index: int) -> str:
        """Encode this SlabMeta as a self-describing dim=3 synthetic physical group name."""
        escaped_names = [
            n.replace("%", "%25").replace("|", "%7C") for n in self.physical_name
        ]
        joined_names = "|".join(escaped_names)
        return (
            f"__cohort_{cohort_index}__slab_{sub_index}"
            f"__name_{joined_names}__src_{self.slab_index}"
        )

    @staticmethod
    def to_synthetic_face_name(cohort_index: int, sub_index: int, role: str) -> str:
        """Encode a sub-solid face role (bot, top, lat_<i>) as a dim=2 synthetic name."""
        return f"__cohort_{cohort_index}__slab_{sub_index}__{role}"

    @staticmethod
    def parse_synthetic_solid_name(
        name: str,
    ) -> tuple[str, int, tuple[str, ...]] | None:
        """Parse a dim=3 synthetic group name into ``(sub_key, slab_index, physical_name)``.

        Supports both the self-describing format
        (``__cohort_{ci}__slab_{si}__name_{names}__src_{slab_index}``)
        and the legacy bare format (``__cohort_{ci}__slab_{si}``).
        Returns ``None`` if ``name`` is not a cohort sub-solid name.
        """
        if not name.startswith("__cohort_"):
            return None
        if "__src_" in name and "__name_" in name:
            head, src_str = name.rsplit("__src_", 1)
            try:
                slab_index = int(src_str)
            except ValueError:
                return None
            if "__name_" not in head:
                return None
            sub_key, names_str = head.split("__name_", 1)
            if not _SUB_KEY_RE.match(sub_key):
                return None
            physical_name = (
                tuple(
                    part.replace("%7C", "|").replace("%25", "%")
                    for part in names_str.split("|")
                )
                if names_str
                else ()
            )
            return sub_key, slab_index, physical_name
        if _SUB_KEY_RE.match(name):
            slab_index = int(name.rsplit("_", 1)[1])
            return name, slab_index, ()
        return None

    @staticmethod
    def parse_synthetic_face_name(name: str) -> tuple[str, str] | None:
        """Parse a dim=2 synthetic face group name into ``(sub_key, role)``."""
        m = _FACE_NAME_RE.match(name)
        if m is None:
            return None
        return m.group(1), m.group(2)

    @classmethod
    def from_synthetic_groups(
        cls,
        solid_groups: dict[str, int],
        face_groups: dict[str, int],
        fallback_names_by_vol: dict[int, list[str]] | None = None,
    ) -> tuple[dict[str, SlabMeta], dict[str, int], dict[str, int]]:
        """Reconstruct ``(slab_meta, face_tag_by_key, sub_solid_tag_by_key)`` from XAO groups.

        Args:
            solid_groups: ``{synthetic_solid_group_name: gmsh_volume_tag}`` for dim=3 groups.
            face_groups: ``{synthetic_face_group_name: gmsh_face_tag}`` for dim=2 groups.
            fallback_names_by_vol: Optional ``{gmsh_volume_tag: [real_name, ...]}`` used
                only when loading legacy bare ``__cohort_{ci}__slab_{si}`` solid names.

        Returns:
            ``(slab_meta, face_tag_by_key, sub_solid_tag_by_key)`` keyed by ``sub_key``
            (``__cohort_{ci}__slab_{si}``) and synthetic face names.
        """
        face_tag_by_key: dict[str, int] = {}
        bot_by_sub: dict[str, str] = {}
        top_by_sub: dict[str, str] = {}
        lats_by_sub: dict[str, list[tuple[int, str]]] = defaultdict(list)

        for fname, ftag in face_groups.items():
            parsed_face = cls.parse_synthetic_face_name(fname)
            if parsed_face is None:
                continue
            sub_key, role = parsed_face
            face_tag_by_key[fname] = int(ftag)
            if role == "bot":
                bot_by_sub[sub_key] = fname
            elif role == "top":
                top_by_sub[sub_key] = fname
            elif role.startswith("lat_"):
                lat_idx = int(role.split("_", 1)[1])
                lats_by_sub[sub_key].append((lat_idx, fname))

        slab_meta: dict[str, SlabMeta] = {}
        sub_solid_tag_by_key: dict[str, int] = {}

        for sname, vtag in solid_groups.items():
            parsed_solid = cls.parse_synthetic_solid_name(sname)
            if parsed_solid is None:
                continue
            sub_key, slab_index, physical_name = parsed_solid
            if not physical_name and fallback_names_by_vol is not None:
                physical_name = tuple(fallback_names_by_vol.get(int(vtag), ()))

            bot_key = bot_by_sub.get(sub_key, f"{sub_key}__bot")
            top_key = top_by_sub.get(sub_key, f"{sub_key}__top")
            sorted_lats = tuple(
                fname
                for _, fname in sorted(lats_by_sub.get(sub_key, ()), key=lambda x: x[0])
            )

            sub_solid_tag_by_key[sub_key] = int(vtag)
            slab_meta[sub_key] = cls(
                slab_index=slab_index,
                physical_name=physical_name,
                bot_face_key=bot_key,
                top_face_key=top_key,
                lateral_face_keys=sorted_lats,
                keep=True,
            )

        return slab_meta, face_tag_by_key, sub_solid_tag_by_key


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
