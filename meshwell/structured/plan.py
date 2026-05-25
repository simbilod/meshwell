"""Plan stage for the clean structured-polyprism pipeline.

Public surface: ``build_plan(entities) -> StructuredPlan``. Private
helpers handle the pipeline steps (gather, expand, validate, partition).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import shapely
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import polygonize, unary_union

from meshwell.structured.logging import phase_timed
from meshwell.structured.spec import (
    ArrangementEdge,
    ArrangementFace,
    CanonicalCircle,
    OverlapPair,
    PieceArcEdge,
    PieceLineEdge,
    PieceProvenance,
    Slab,
    StackArrangement,
    StructuredExtrusionResolutionSpec,
    StructuredOverlapError,
    StructuredPartitionConvergenceError,
    StructuredPlan,
)

_Z_TOL = 1e-9  # exact-match tolerance for z-extent comparison

# Hard ceiling for the face_partition fixed-point iteration. Convergence is
# typically bounded by the longest face-touching z-chain K; we use
# min(K + 2, _PARTITION_FIXED_POINT_CAP) so pathological scenes still fail
# loud rather than loop forever.
_PARTITION_FIXED_POINT_CAP: int = 16

# Updated by compute_face_partition; read by tests to assert convergence
# bounds. Module-level (single-threaded planner) is acceptable.
_LAST_PARTITION_ITERATIONS: int = 0


def gather_structured_entities(
    entities: list[Any],
) -> list[tuple[Any, StructuredExtrusionResolutionSpec, int]]:
    """Return ``(entity, spec, source_index)`` for every structured prism.

    A structured entity is one with ``structured=True`` AND exactly one
    ``StructuredExtrusionResolutionSpec`` in its ``resolutions``. The
    validation at construction time guarantees this when both conditions
    hold, so we just retrieve here.
    """
    out: list[tuple[Any, StructuredExtrusionResolutionSpec, int]] = []
    for idx, ent in enumerate(entities):
        if not getattr(ent, "structured", False):
            continue
        resolutions = getattr(ent, "resolutions", None) or []
        specs = [
            r for r in resolutions if isinstance(r, StructuredExtrusionResolutionSpec)
        ]
        if len(specs) != 1:
            # PolyPrism construction enforces this; defensive check.
            continue
        out.append((ent, specs[0], idx))
    return out


def expand_to_slabs(
    pairs: list[tuple[Any, StructuredExtrusionResolutionSpec, int]],
) -> list[Slab]:
    """One slab per (entity, z-interval) pair.

    n_layers / recombine are NOT stored on the slab - they live on the
    spec and are resolved at mesh time via (source_index, z_interval_index).
    """
    slabs: list[Slab] = []
    for entity, _spec, source_index in pairs:
        z_keys = list(entity.buffers.keys())
        mesh_order = (
            entity.mesh_order if entity.mesh_order is not None else float("inf")
        )
        footprint = entity.polygons
        for z_idx, (zlo, zhi) in enumerate(pairwise(z_keys)):
            slabs.append(
                Slab(
                    footprint=footprint,
                    zlo=float(zlo),
                    zhi=float(zhi),
                    physical_name=entity.physical_name,
                    source_index=source_index,
                    z_interval_index=z_idx,
                    mesh_order=mesh_order,
                    identify_arcs=getattr(entity, "identify_arcs", False),
                    min_arc_points=getattr(entity, "min_arc_points", 4),
                    arc_tolerance=getattr(entity, "arc_tolerance", 1e-3),
                )
            )
    return slabs


def _z_extent_matches(a: Slab, b: Slab) -> bool:
    return abs(a.zlo - b.zlo) < _Z_TOL and abs(a.zhi - b.zhi) < _Z_TOL


def _footprints_overlap(a: Slab, b: Slab) -> bool:
    """True iff the two slab footprints share positive xy area."""
    inter = a.footprint.intersection(b.footprint)
    return (not inter.is_empty) and inter.area > 0


def _z_volumetric_overlap(a: Slab, b: Slab) -> bool:
    """True iff the two slabs' z-extents overlap with positive measure.

    Face-touching (e.g. a.zhi == b.zlo) is NOT volumetric overlap — stacked
    structured slabs sharing a z-plane are allowed and don't trigger Policy B.
    """
    lo = max(a.zlo, b.zlo)
    hi = min(a.zhi, b.zhi)
    return (hi - lo) > _Z_TOL


def _n_layers_of_slab(slab: Slab, entities: list[Any]) -> int:
    """Look up n_layers for slab via (source_index, z_interval_index)."""
    ent = entities[slab.source_index]
    specs = [
        r
        for r in (getattr(ent, "resolutions", None) or [])
        if isinstance(r, StructuredExtrusionResolutionSpec)
    ]
    return specs[0].n_layers[slab.z_interval_index]


def validate_and_resolve_overlap(
    slabs: list[Slab],
    entities: list[Any],
) -> tuple[list[Slab], list[OverlapPair]]:
    """Apply Policy B: drop volumetric overlap losers, fail on mismatch.

    Returns (kept_slabs, overlap_pairs). Lower mesh_order wins; tie-break
    by source_index then z_interval_index for determinism.
    """
    # Sort by (mesh_order, source_index, z_interval_index): winners first.
    order = sorted(
        range(len(slabs)),
        key=lambda i: (
            slabs[i].mesh_order,
            slabs[i].source_index,
            slabs[i].z_interval_index,
        ),
    )

    kept_indices: list[int] = []
    overlaps: list[OverlapPair] = []
    for idx in order:
        slab = slabs[idx]
        dominated = False
        for k_idx in kept_indices:
            kept = slabs[k_idx]
            # Slabs from the same source entity occupy different z-intervals
            # and cannot volumetrically overlap.
            if kept.source_index == slab.source_index:
                continue
            if not _footprints_overlap(kept, slab):
                continue
            if not _z_volumetric_overlap(kept, slab):
                # Footprints overlap but z-extents are disjoint (stacked or
                # vertically separated) — not volumetric overlap, no Policy B.
                continue
            if not _z_extent_matches(kept, slab):
                raise StructuredOverlapError(
                    f"Volumetric overlap of structured prisms "
                    f"{kept.physical_name} and {slab.physical_name} "
                    f"requires matching z-extents (got "
                    f"[{kept.zlo}, {kept.zhi}] vs "
                    f"[{slab.zlo}, {slab.zhi}]). "
                    f"Adjust the prisms so z-extents match exactly or so "
                    f"footprints do not overlap."
                )
            kept_n = _n_layers_of_slab(kept, entities)
            slab_n = _n_layers_of_slab(slab, entities)
            if kept_n != slab_n:
                raise StructuredOverlapError(
                    f"Volumetric overlap of structured prisms "
                    f"{kept.physical_name} (n_layers={kept_n}) and "
                    f"{slab.physical_name} (n_layers={slab_n}) at "
                    f"z=[{kept.zlo}, {kept.zhi}]: n_layers must agree."
                )
            overlaps.append(
                OverlapPair(
                    winner_slab_index=kept_indices.index(k_idx),
                    loser_source_index=slab.source_index,
                    loser_z_interval_index=slab.z_interval_index,
                    z_extent=(slab.zlo, slab.zhi),
                )
            )
            dominated = True
            break
        if not dominated:
            kept_indices.append(idx)

    kept_slabs = [slabs[i] for i in kept_indices]
    return kept_slabs, overlaps


def _entity_z_range(ent: Any) -> tuple[float, float] | None:
    """Return (zmin, zmax) for an entity that has a buffers/z-range, else None."""
    buffers = getattr(ent, "buffers", None)
    if not buffers:
        return None
    return (min(buffers.keys()), max(buffers.keys()))


def _entity_footprint(ent: Any) -> Polygon | MultiPolygon | None:
    polys = getattr(ent, "polygons", None)
    if polys is None:
        return None
    if isinstance(polys, list):
        flat: list[Polygon] = []
        for p in polys:
            if isinstance(p, MultiPolygon):
                flat.extend(p.geoms)
            elif isinstance(p, Polygon):
                flat.append(p)
        if not flat:
            return None
        return flat[0] if len(flat) == 1 else MultiPolygon(flat)
    return polys


def _resolve_sublevel_mesh_order(slabs: list[Slab], entities: list[Any]) -> None:
    """Set ``slab.resolved_footprint`` in place per sub-level mesh-order carving.

    For each z-interval (grouped by (zlo, zhi) keys), sort kept slabs by
    (mesh_order, source_index) ascending. The first (winner) keeps its
    full footprint. Each subsequent slab's resolved_footprint is its
    original footprint minus the union of every prior winner's
    resolved_footprint at the same z-interval.

    mesh_bool=False entities whose z-range covers the sub-level
    additionally carve out of every kept slab's resolved_footprint.
    They do not themselves carry a resolved_footprint (they're not in
    the slab list — only their boundaries propagate to step C).
    """
    # Group slabs by (zlo, zhi).
    by_interval: dict[tuple[float, float], list[Slab]] = {}
    for s in slabs:
        by_interval.setdefault((s.zlo, s.zhi), []).append(s)

    for (zlo, zhi), group in by_interval.items():
        # Collect mesh_bool=False entities whose z-range covers this interval.
        void_footprints: list[Any] = []
        for ent in entities:
            if getattr(ent, "mesh_bool", True):
                continue
            rng = _entity_z_range(ent)
            if rng is None:
                continue
            ent_zmin, ent_zmax = rng
            if ent_zmin <= zlo + _Z_TOL and ent_zmax >= zhi - _Z_TOL:
                fp = _entity_footprint(ent)
                if fp is not None:
                    void_footprints.append(fp)

        # Sort by (mesh_order, source_index): winners first.
        ordered = sorted(group, key=lambda s: (s.mesh_order, s.source_index))
        accumulated_winners: list[Any] = []
        for slab in ordered:
            resolved = slab.footprint
            if accumulated_winners:
                resolved = resolved.difference(unary_union(accumulated_winners))
            if void_footprints:
                resolved = resolved.difference(unary_union(void_footprints))
            slab.resolved_footprint = resolved
            accumulated_winners.append(slab.footprint)


def _connected_z_components(slabs: list[Slab]) -> list[list[Slab]]:
    """Group slabs into connected components.

    Two slabs are in the same component iff either:
      - they share a z-face (abs(a.zhi - b.zlo) < _Z_TOL or symmetric), or
      - they share the same z-interval (a.zlo == b.zlo AND a.zhi == b.zhi).

    The two-clause rule ensures that same-z-interval lateral neighbours
    (e.g., two structured slabs at z=[0,1] abutting at x=1) are grouped
    together. Without that, their cuts wouldn't propagate to the
    arrangement at all.

    Implementation: Union-Find on slab indices.
    """
    parent = list(range(len(slabs)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i, a in enumerate(slabs):
        for j in range(i + 1, len(slabs)):
            b = slabs[j]
            face_touching = abs(a.zhi - b.zlo) < _Z_TOL or abs(a.zlo - b.zhi) < _Z_TOL
            same_interval = abs(a.zlo - b.zlo) < _Z_TOL and abs(a.zhi - b.zhi) < _Z_TOL
            if face_touching or same_interval:
                union(i, j)

    components_by_root: dict[int, list[Slab]] = {}
    for i, s in enumerate(slabs):
        components_by_root.setdefault(find(i), []).append(s)
    return list(components_by_root.values())


def _neighbours_touching_z(
    z: float, entities: list[Any], skip_indices: set[int], tol: float = 1e-9
) -> list[Polygon | MultiPolygon]:
    """Footprints of entities whose buffers include z (within tol)."""
    out: list[Polygon | MultiPolygon] = []
    for i, ent in enumerate(entities):
        if i in skip_indices:
            continue
        rng = _entity_z_range(ent)
        if rng is None:
            continue
        zmin, zmax = rng
        if abs(zmin - z) < tol or abs(zmax - z) < tol:
            fp = _entity_footprint(ent)
            if fp is not None:
                out.append(fp)
    return out


def _structured_slabs_touching_z(
    z: float,
    slabs: list["Slab"],
    skip_slab_ids: set[int],
    tol: float = 1e-9,
) -> list["Slab"]:
    """Structured slabs whose zlo or zhi equals z within tol.

    Mirrors :func:`_neighbours_touching_z` but walks the slab list (so the
    caller can read each slab's *current* face_partition rather than just
    the entity footprint).
    """
    out: list[Slab] = []
    for s in slabs:
        if id(s) in skip_slab_ids:
            continue
        if abs(s.zlo - z) < tol or abs(s.zhi - z) < tol:
            out.append(s)
    return out


def _collect_cut_sources(
    slab: "Slab",
    slabs: list["Slab"],
    entities: list[Any],
    skip_indices: set[int],
) -> list[Any]:
    """Return shapely boundary geometries that should cut ``slab``'s footprint.

    Two arms:

    - Unstructured z-touching entities contribute their **footprint
      boundary**. (Structured entities are filtered out here; their
      cuts come from the slab arm below, which uses piece boundaries.)
    - Structured z-touching slabs contribute **each piece's boundary**
      from their current ``face_partition``. On iteration 1, every
      slab's face_partition is ``[footprint]`` so this matches today's
      behavior; later iterations refine.
    """
    sources: list[Any] = []

    # Unstructured-entity arm.
    for i, ent in enumerate(entities):
        if i in skip_indices:
            continue
        if getattr(ent, "structured", False):
            continue  # structured slabs handled below
        rng = _entity_z_range(ent)
        if rng is None:
            continue
        zmin, zmax = rng
        if (
            abs(zmin - slab.zlo) < 1e-9
            or abs(zmax - slab.zlo) < 1e-9
            or abs(zmin - slab.zhi) < 1e-9
            or abs(zmax - slab.zhi) < 1e-9
        ):
            fp = _entity_footprint(ent)
            if fp is not None:
                sources.append(fp.boundary)

    # Structured-slab arm.
    skip_slab_ids = {id(slab)}
    for n_slab in _structured_slabs_touching_z(slab.zlo, slabs, skip_slab_ids):
        sources.extend(piece.boundary for piece in n_slab.face_partition)
    for n_slab in _structured_slabs_touching_z(slab.zhi, slabs, skip_slab_ids):
        sources.extend(piece.boundary for piece in n_slab.face_partition)

    return sources


def _collect_inherited_arcs(
    slab: "Slab",
    slabs: list["Slab"],
    skip_slab_ids: set[int],
    arc_indices: "dict[int, _ArcIndex] | None" = None,
) -> list[PieceArcEdge]:
    """Return PieceArcEdge entries inherited from z-touching arc neighbours.

    Returns ``[]`` when the receiving slab has ``identify_arcs=False``
    (no point classifying inherited arcs on a slab that doesn't track them).
    Otherwise walks z-touching structured slabs and collects arcs from two
    sources:

    1. ``face_partition_provenance``: arcs on neighbours that already split
       into multiple pieces and carry per-piece provenance.
    2. ``arc_indices[id(n_slab)]``: the neighbour's footprint-derived arc
       index. This covers single-piece arc slabs (which don't get provenance
       attached but still have arcs on their footprint boundary that need to
       propagate). Only the neighbour's *own* footprint arcs are pulled here
       (``inherited=False``); already-inherited arcs in the neighbour are
       handled in the next propagation step from their original source.
    """
    if not slab.identify_arcs:
        return []

    inherited: list[PieceArcEdge] = []
    for z in (slab.zlo, slab.zhi):
        for n_slab in _structured_slabs_touching_z(z, slabs, skip_slab_ids):
            if not n_slab.identify_arcs:
                continue
            # Source 1: neighbour's provenance (multi-piece slabs).
            if n_slab.face_partition_provenance is not None:
                for prov in n_slab.face_partition_provenance:
                    inherited.extend(
                        edge
                        for edge in prov.exterior_edges
                        if isinstance(edge, PieceArcEdge)
                    )
                    for ring_edges in prov.interior_edges:
                        inherited.extend(
                            edge
                            for edge in ring_edges
                            if isinstance(edge, PieceArcEdge)
                        )
            # Source 2: neighbour's arc index (single-piece arc slabs, where
            # provenance is None but the footprint arcs still need to propagate).
            if arc_indices is not None:
                n_idx = arc_indices.get(id(n_slab))
                if n_idx is not None:
                    for arc in n_idx.arcs:
                        if arc.inherited:
                            continue  # original source handles propagation
                        inherited.append(
                            PieceArcEdge(
                                points=arc.points,
                                center=arc.center,
                                radius=arc.radius,
                            )
                        )
    return inherited


def _validate_no_mid_height_cuts(
    slabs: list[Slab],
    entities: list[Any],
    tol: float = 1e-9,
) -> None:
    """Raise StructuredMidHeightCutError if any neighbour would cut a slab mid-height.

    A mid-height cut occurs when a neighbour entity has a z-endpoint
    (zmin or zmax) strictly inside a slab's (zlo, zhi) range AND its xy
    footprint intersects the slab's footprint with positive area. The
    BOP would then introduce a vertex on the slab's lateral OCC face at
    that intermediate z — the structured mesh stage can't form a
    conformal wedge grid through such a face (Layer C's wedge layering
    only has nodes at the n_layers grid).

    Fail-fast diagnostic: clear remediation message instructs the user
    to add the offending z as a buffer-key on the structured polyprism.
    """
    from meshwell.structured.spec import StructuredMidHeightCutError

    structured_source_indices = {s.source_index for s in slabs}
    for slab in slabs:
        slab_fp = slab.footprint
        for i, ent in enumerate(entities):
            # Skip the slab's own owning entity (and any sibling sub-slab
            # of the same entity — same source_index).
            if i == slab.source_index:
                continue
            # Skip other structured entities — Policy B already catches
            # structured/structured volumetric overlap with mismatched z.
            if i in structured_source_indices and getattr(ent, "structured", False):
                continue
            rng = _entity_z_range(ent)
            if rng is None:
                continue
            zmin, zmax = rng
            cut_zs: list[float] = []
            if slab.zlo + tol < zmin < slab.zhi - tol:
                cut_zs.append(zmin)
            if slab.zlo + tol < zmax < slab.zhi - tol:
                cut_zs.append(zmax)
            if not cut_zs:
                continue
            ent_fp = _entity_footprint(ent)
            if ent_fp is None:
                continue
            inter = slab_fp.intersection(ent_fp)
            if inter.is_empty or inter.area <= 0:
                continue
            # We have a mid-height cut.
            cut_z = cut_zs[0]
            neighbour_name = getattr(ent, "physical_name", ("?",))
            raise StructuredMidHeightCutError(
                f"Neighbour {neighbour_name} (z=[{zmin}, {zmax}]) would "
                f"mid-height-cut structured slab {slab.physical_name} "
                f"(z=[{slab.zlo}, {slab.zhi}]) at z={cut_z}. "
                f"The structured pipeline can't form a conformal wedge "
                f"grid through a lateral face cut at an intermediate z. "
                f"Remediation: add {cut_z} as a buffer-key on the "
                f"structured polyprism {slab.physical_name} (with explicit "
                f"n_layers split for the new sub-interval), or move the "
                f"neighbour to share the slab's existing zlo/zhi planes."
            )


def _validate_no_unstructured_lateral_neighbour(
    slabs: list[Slab],
    entities: list[Any],
    tol: float = 1e-9,
) -> None:
    """Raise if any unstructured neighbour shares a lateral face with a slab.

    A wedge or hex lateral face is topologically a quad; a tet-meshed
    neighbour on the other side presents triangular faces. The two
    don't match one-to-one, breaking volume-element conformality. We
    refuse to plan such a configuration rather than silently producing
    a non-conformal mesh.
    """
    from meshwell.structured.spec import StructuredLateralUnstructuredNeighbourError

    structured_source_indices = {s.source_index for s in slabs}
    for slab in slabs:
        slab_boundary = slab.footprint.boundary
        for i, ent in enumerate(entities):
            if i == slab.source_index:
                continue
            if i in structured_source_indices and getattr(ent, "structured", False):
                continue
            # mesh_bool=False / keep=False entities are carving helpers —
            # not present in the final mesh, so they can't introduce a
            # tet/wedge interface. Check both attribute names since the
            # planner sees user-facing entities (mesh_bool) while internal
            # _MeshEntity uses keep.
            if not getattr(ent, "mesh_bool", True):
                continue
            if not getattr(ent, "keep", True):
                continue
            rng = _entity_z_range(ent)
            if rng is None:
                continue
            zmin, zmax = rng
            z_overlap = min(slab.zhi, zmax) - max(slab.zlo, zmin)
            if z_overlap <= tol:
                continue
            ent_fp = _entity_footprint(ent)
            if ent_fp is None:
                continue
            shared = slab_boundary.intersection(ent_fp)
            if shared.is_empty or shared.length <= tol:
                continue
            neighbour_name = getattr(ent, "physical_name", ("?",))
            raise StructuredLateralUnstructuredNeighbourError(
                f"Unstructured neighbour {neighbour_name} "
                f"(z=[{zmin}, {zmax}]) shares a lateral face with "
                f"structured slab {slab.physical_name} "
                f"(z=[{slab.zlo}, {slab.zhi}]) along a curve of length "
                f"{shared.length:.6g}. Wedge/hex lateral faces are "
                f"topologically quads; tet neighbour faces are tris — "
                f"the mesh would be non-conformal at this interface. "
                f"Remediation: make the neighbour structured, or move "
                f"its xy footprint off the slab's boundary, or separate "
                f"them in z so the z-ranges don't overlap."
            )


@dataclass(frozen=True)
class _IndexedArc:
    """An arc identified on the original footprint, or inherited from a neighbour.

    Full vertex sequence preserved in boundary order. ``inherited=True``
    marks arcs merged from a z-touching neighbour's face_partition_provenance
    (rather than fit from this slab's own footprint). Inherited arcs are
    skipped by :func:`_validate_arc_neighbour_alignment` because the receiving
    slab's polygon vertex grid doesn't correspond to the inherited arc — the
    intersection check would always fail on polygon-vs-arc chord deviation
    that has no meaning for OCC's true-arc cut from the source neighbour.
    """

    arc_id: int
    center: tuple[float, float, float]
    radius: float
    points: tuple[tuple[float, float, float], ...]  # length >= min_arc_points
    inherited: bool = False


@dataclass
class _ArcIndex:
    """Lookup structure for classifying piece-boundary segments.

    ``arcs``: every arc identified on the footprint's exterior + holes.
    ``vertex_to_arcs``: maps rounded (x, y) coord -> list of
    (arc_id, position-in-arc) tuples. A piece-boundary segment
    (p_i, p_{i+1}) is on arc K iff both endpoints share an arc_id with
    *adjacent* positions in K's vertex list.

    Rounding granularity is ``ndigits`` = ``-floor(log10(arc_tolerance))``
    where ``arc_tolerance`` is the same tolerance used by the arc-detection
    heuristic so we match its quantization grid.
    """

    arcs: list[_IndexedArc] = field(default_factory=list)
    vertex_to_arcs: dict[tuple[float, float], list[tuple[int, int]]] = field(
        default_factory=dict
    )
    ndigits: int = 3


def _build_arc_index_from_footprint(
    footprint: Polygon | MultiPolygon,
    identify_arcs: bool,
    min_arc_points: int,
    arc_tolerance: float,
) -> _ArcIndex:
    """Decompose every ring of ``footprint`` and index each arc by its vertices.

    Disabled (returns empty index) when ``identify_arcs`` is False.
    The decomposition uses the same heuristic that ``GeometryEntity``
    uses today, but only on the **pure-arc input** (the user-provided
    polygon), where it is reliable.
    """
    ndigits = max(0, int(-math.floor(math.log10(max(arc_tolerance, 1e-12)))))
    index = _ArcIndex(ndigits=ndigits)
    if not identify_arcs:
        return index

    from meshwell.geometry_entity import GeometryEntity

    components: list[Polygon] = (
        list(footprint.geoms) if hasattr(footprint, "geoms") else [footprint]
    )
    adapter = GeometryEntity(point_tolerance=max(arc_tolerance, 1e-12))
    arc_counter = 0
    for comp in components:
        for ring in [comp.exterior, *comp.interiors]:
            verts: list[tuple[float, float, float]] = [
                (x, y, 0.0) for x, y in list(ring.coords)
            ]
            segments = adapter.decompose_vertices(
                verts,
                identify_arcs=True,
                min_arc_points=min_arc_points,
                arc_tolerance=arc_tolerance,
            )
            for seg in segments:
                if not seg.is_arc:
                    continue
                arc = _IndexedArc(
                    arc_id=arc_counter,
                    center=seg.center,
                    radius=seg.radius,
                    points=tuple(seg.points),
                )
                arc_counter += 1
                index.arcs.append(arc)
                for pos, (x, y, _z) in enumerate(arc.points):
                    key = (round(x, ndigits), round(y, ndigits))
                    index.vertex_to_arcs.setdefault(key, []).append((arc.arc_id, pos))

    return index


def _merge_arc_into_index(index: _ArcIndex, arc_edge: PieceArcEdge) -> None:
    """Append a neighbour-inherited PieceArcEdge to an existing arc index.

    Assigns a fresh ``arc_id`` (continuing the per-index counter) and
    indexes each vertex of the arc so :func:`_classify_piece_boundary`
    can recognize inherited arc edges on the receiving slab's pieces.

    The caller is responsible for any cross-iteration deduplication
    (e.g., skipping arcs that were already merged in a prior pass);
    this helper itself is idempotent on inputs but does not deduplicate.
    """
    arc_id = len(index.arcs)
    indexed = _IndexedArc(
        arc_id=arc_id,
        center=arc_edge.center,
        radius=arc_edge.radius,
        points=tuple(arc_edge.points),
        inherited=True,
    )
    index.arcs.append(indexed)
    for pos, (x, y, _z) in enumerate(indexed.points):
        key = (round(x, index.ndigits), round(y, index.ndigits))
        index.vertex_to_arcs.setdefault(key, []).append((arc_id, pos))


def _classify_piece_boundary(
    piece: Polygon,
    arc_index: _ArcIndex,
) -> PieceProvenance:
    """Walk the piece's exterior + interior rings, labeling each segment.

    For each consecutive pair (v_i, v_i+1) on a piece ring:
    - If BOTH are in arc_index.vertex_to_arcs with a SHARED arc_id AND
      the position difference is consistent with being on that arc,
      this segment is part of an arc — extend the running PieceArcEdge.
    - Otherwise close the running arc edge (if any) and start a
      PieceLineEdge containing this segment.

    Returns PieceProvenance with the resulting exterior and interior
    edge lists.
    """

    def _classify_ring(ring) -> list[PieceArcEdge | PieceLineEdge]:
        coords = list(ring.coords)
        if len(coords) < 2:
            return []
        # Drop the duplicated closing vertex if present.
        if coords[0] == coords[-1]:
            coords = coords[:-1]
        # Strip consecutive near-duplicate vertices using rounded key comparison.
        # Shapely may emit tiny floating-point noise (e.g. 1.2e-16) at seam
        # vertices that makes coords appear distinct while they round to the
        # same arc-index key. Use the arc_index ndigits for deduplication so
        # the vertex sequence matches the index keys exactly.
        ndigits = arc_index.ndigits

        def _key(xy):
            return (round(xy[0], ndigits), round(xy[1], ndigits))

        deduped: list = [coords[0]]
        for c in coords[1:]:
            if _key(c) != _key(deduped[-1]):
                deduped.append(c)
        coords = deduped
        n = len(coords)
        if n < 2:
            return []

        def _segment_arc_id(a_xy, b_xy) -> int | None:
            """Return arc_id if (a, b) is one step along some arc, else None.

            Note: a vertex may appear at multiple positions in the same arc
            when the arc is closed (arc.points[0] == arc.points[-1]) — both
            pos 0 and pos arc_len-1 map to the same coordinate. We check ALL
            (arc_id, pos) combinations for b_xy.
            """
            a_lookup = arc_index.vertex_to_arcs.get(_key(a_xy), [])
            b_lookup = arc_index.vertex_to_arcs.get(_key(b_xy), [])
            if not a_lookup or not b_lookup:
                return None
            # Group b positions by arc_id (multiple positions possible for same arc).
            b_by_arc: dict[int, list[int]] = {}
            for arc_id, pos in b_lookup:
                b_by_arc.setdefault(arc_id, []).append(pos)
            for arc_id, a_pos in a_lookup:
                b_positions = b_by_arc.get(arc_id)
                if not b_positions:
                    continue
                arc = arc_index.arcs[arc_id]
                arc_len = len(arc.points)
                for b_pos in b_positions:
                    # Adjacency: |b_pos - a_pos| == 1
                    if abs(b_pos - a_pos) == 1:
                        return arc_id
                    # Wrap: only when the arc is closed (first == last vertex).
                    if arc.points[0] == arc.points[-1] and {a_pos, b_pos} == {
                        0,
                        arc_len - 2,
                    }:
                        return arc_id
            return None

        edges: list[PieceArcEdge | PieceLineEdge] = []
        i = 0
        visited = 0
        while visited < n:
            j = (i + 1) % n
            a = coords[i]
            b = coords[j]
            arc_id = _segment_arc_id(a, b)
            if arc_id is None:
                edges.append(
                    PieceLineEdge(points=((a[0], a[1], 0.0), (b[0], b[1], 0.0)))
                )
                i = j
                visited += 1
                continue
            # Greedily extend along the same arc, but at most (n - visited)
            # steps total so we never wrap around and double-cover the ring.
            remaining = n - visited
            run_indices = [i, j]
            steps_taken = 1
            while steps_taken < remaining:
                k = (run_indices[-1] + 1) % n
                next_arc = _segment_arc_id(coords[run_indices[-1]], coords[k])
                if next_arc != arc_id:
                    break
                run_indices.append(k)
                steps_taken += 1
            pts = tuple((coords[idx][0], coords[idx][1], 0.0) for idx in run_indices)
            arc = arc_index.arcs[arc_id]
            edges.append(PieceArcEdge(points=pts, center=arc.center, radius=arc.radius))
            # Advance past every vertex in the run except the last.
            steps = len(run_indices) - 1
            visited += steps
            i = run_indices[-1]
        return edges

    exterior_edges = _classify_ring(piece.exterior)
    interior_edges = [_classify_ring(ring) for ring in piece.interiors]
    return PieceProvenance(exterior_edges=exterior_edges, interior_edges=interior_edges)


def _partition_pieces_for_slab(
    slab: "Slab",
    cut_sources: list[Any],
) -> list[Polygon]:
    """Polygonize the slab footprint with the given cut sources.

    Pure function: deterministic given (slab.footprint, cut_sources).
    Returns the new face_partition list (>=1 element).
    """
    if not cut_sources:
        return [slab.footprint]
    all_boundaries = unary_union(cut_sources)
    boundary = slab.footprint.boundary
    combined = unary_union([boundary, all_boundaries.intersection(slab.footprint)])
    raw = list(polygonize(combined))
    pieces = [
        piece for piece in raw if slab.footprint.contains(piece.representative_point())
    ]
    return pieces if pieces else [slab.footprint]


def _attach_face_partition_provenance(
    slabs: list["Slab"],
    arc_indices: dict[int, "_ArcIndex"],
) -> None:
    """Compute provenance for arc slabs with multi-piece partitions.

    Called after the fixed-point loop converges. Uses the final (possibly
    merged) per-slab arc index so inherited arc segments are recognized.
    """
    for slab in slabs:
        if not slab.identify_arcs:
            continue
        if len(slab.face_partition) <= 1:
            slab.face_partition_provenance = None
            continue
        idx = arc_indices.get(id(slab))
        if idx is None:
            continue
        slab.face_partition_provenance = [
            _classify_piece_boundary(piece, idx) for piece in slab.face_partition
        ]


def compute_face_partition(slabs: list[Slab], entities: list[Any]) -> None:
    """Compute slab.face_partition (and face_partition_provenance) in place.

    Uses a fixed-point iteration: each pass collects cut sources from
    z-touching neighbours' *current* face_partition pieces (not just
    their footprint), so cuts introduced one z-step away propagate
    transitively across the stack. Iteration 1 reproduces the single-pass
    behavior because every slab's initial face_partition is its footprint.

    For arc slabs, the per-slab arc index accumulates inherited
    PieceArcEdge entries from neighbour provenance so the classifier
    labels inherited arc segments correctly.
    """
    own_indices_by_slab = {id(s): {s.source_index} for s in slabs}

    # Initialize each slab's face_partition to [footprint] so iteration 1
    # sees the same cut sources today's single-pass code does.
    for slab in slabs:
        slab.face_partition = [slab.footprint]
        slab.face_partition_provenance = None

    # Per-slab arc index, built once from the footprint. Mutates over
    # iterations as inherited arcs are merged in.
    arc_indices: dict[int, _ArcIndex] = {}
    for slab in slabs:
        if slab.identify_arcs:
            arc_indices[id(slab)] = _build_arc_index_from_footprint(
                slab.footprint,
                identify_arcs=True,
                min_arc_points=slab.min_arc_points,
                arc_tolerance=slab.arc_tolerance,
            )

    # Track inherited arcs already merged for each slab, to avoid re-merging
    # the same neighbour PieceArcEdge across iterations.
    merged_arc_keys: dict[int, set[tuple]] = {id(s): set() for s in slabs}

    # Cache cut-source WKB sets to detect convergence per slab.
    cached_wkb: dict[int, frozenset] = {id(s): frozenset() for s in slabs}

    # Track which slabs were still changing in the most recent pass so we
    # can report them in the convergence error.
    changed_last_pass: set[int] = set()

    for _pass in range(_PARTITION_FIXED_POINT_CAP):
        changed = False
        changed_last_pass = set()
        for slab in slabs:
            cut_sources = _collect_cut_sources(
                slab=slab,
                slabs=slabs,
                entities=entities,
                skip_indices=own_indices_by_slab[id(slab)],
            )
            new_wkb = frozenset(geom.wkb for geom in cut_sources)
            if new_wkb == cached_wkb[id(slab)]:
                continue  # stable for this pass
            cached_wkb[id(slab)] = new_wkb
            changed_last_pass.add(id(slab))

            # Merge inherited arcs into this slab's arc index, deduped by
            # (center, radius, sorted vertex tuple).
            if slab.identify_arcs:
                idx = arc_indices[id(slab)]
                seen = merged_arc_keys[id(slab)]
                for arc_edge in _collect_inherited_arcs(
                    slab=slab,
                    slabs=slabs,
                    skip_slab_ids={id(slab)},
                    arc_indices=arc_indices,
                ):
                    key = (
                        arc_edge.center,
                        arc_edge.radius,
                        tuple(sorted(arc_edge.points)),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    _merge_arc_into_index(idx, arc_edge)

            slab.face_partition = _partition_pieces_for_slab(slab, cut_sources)
            changed = True

            # If this slab has arcs and >1 piece, compute provenance now so
            # the NEXT pass can read it via _collect_inherited_arcs (other
            # slabs in this pass have not yet seen this change either, so
            # within-pass ordering doesn't matter).
            if slab.identify_arcs and len(slab.face_partition) > 1:
                idx = arc_indices[id(slab)]
                slab.face_partition_provenance = [
                    _classify_piece_boundary(piece, idx)
                    for piece in slab.face_partition
                ]

        if not changed:
            break
    else:
        unstable = [
            (s.physical_name, s.zlo, s.zhi)
            for s in slabs
            # Slabs whose cut-source set was still changing in the final pass.
            if id(s) in changed_last_pass
        ]
        raise StructuredPartitionConvergenceError(
            f"face_partition did not converge after "
            f"{_PARTITION_FIXED_POINT_CAP} passes; unstable slabs: {unstable}"
        )

    global _LAST_PARTITION_ITERATIONS
    _LAST_PARTITION_ITERATIONS = _pass + 1

    # Final provenance attachment on converged partitions (idempotent for
    # arc slabs that already computed it during the loop).
    _attach_face_partition_provenance(slabs, arc_indices)

    # Validate arc-vs-neighbour alignment AFTER convergence so all
    # transitively-introduced cuts are visible.
    for slab in slabs:
        if not slab.identify_arcs:
            continue
        idx = arc_indices.get(id(slab))
        if idx is None:
            continue
        # Validator filters out arc-bearing neighbours. Their polygon edges
        # are polyline approximations of true arcs; OCC will cut with the
        # actual arc, not the polyline chord. The chord-vs-arc deviation at
        # point_tolerance-snapped polygon vertices has no meaning for the
        # OCC cut, so checking it produces false positives. The neighbour's
        # arc-vs-arc cut is handled correctly downstream regardless.
        skip = own_indices_by_slab[id(slab)] | {
            i for i, ent in enumerate(entities) if getattr(ent, "identify_arcs", False)
        }
        neighbours_lo = _neighbours_touching_z(slab.zlo, entities, skip)
        neighbours_hi = _neighbours_touching_z(slab.zhi, entities, skip)
        all_neighbour_polys = neighbours_lo + neighbours_hi
        if all_neighbour_polys:
            _validate_arc_neighbour_alignment(
                slab,
                idx,
                all_neighbour_polys,
                tol=slab.arc_tolerance,
            )


def _interior_buffer_for_radius(slab: "Slab", r: float) -> float:
    """Compute the arc-vs-neighbour interior buffer for a given radius.

    Replaces the heuristic ``max(arc_tol, 0.05 * r)`` with the cleaner
    ``arc_chord_height_fraction * r``. This makes the buffer
    proportional to the local arc radius rather than dominated by a
    5%-of-radius wildcard or an absolute floor that may not scale with
    the geometry.
    """
    return slab.arc_chord_height_fraction * r


def _validate_arc_neighbour_alignment(
    slab: Slab,
    arc_index: _ArcIndex,
    neighbour_polys: list,
    tol: float,
) -> None:
    """Raise StructuredArcSplitError on non-polygon-vertex arc crossings.

    For each neighbour boundary segment, analytically intersect with each
    arc's true circle; raise if any intersection inside the slab footprint
    is not at an original arc polygon vertex.

    Runs BEFORE polygonize (which can silently miss off-vertex cuts).
    For each arc center + radius, for each neighbour boundary edge that
    overlaps the arc's xy-bbox, compute the analytic segment-circle
    intersection. Each intersection point inside the slab footprint must
    be at an arc polygon vertex; otherwise the polygon-based partition
    will disagree with BOP's true-arc cut downstream.
    """
    from meshwell.structured.spec import StructuredArcSplitError

    if not arc_index.arcs:
        return

    ndigits = arc_index.ndigits
    near_arc_tol = max(tol, 1e-9)

    def _on_polygon_vertex(x: float, y: float) -> bool:
        return (round(x, ndigits), round(y, ndigits)) in arc_index.vertex_to_arcs

    for nb_idx, nb_poly in enumerate(neighbour_polys):
        rings = [list(nb_poly.exterior.coords)]
        rings.extend(list(h.coords) for h in nb_poly.interiors)

        for ring_coords in rings:
            for i in range(len(ring_coords) - 1):
                p1 = ring_coords[i]
                p2 = ring_coords[i + 1]

                for arc in arc_index.arcs:
                    # Inherited arcs come from a neighbour's provenance — the
                    # receiving slab's polygon vertex grid doesn't correspond
                    # to the inherited arc's vertices, so the polygon-vertex
                    # check below would always fail on chord-vs-circle
                    # deviation that has no meaning for OCC's true-arc cut
                    # from the source neighbour. The source slab already
                    # validated its own footprint arcs against its neighbours.
                    if arc.inherited:
                        continue
                    cx, cy = arc.center[0], arc.center[1]
                    r = arc.radius

                    # Quick bbox reject: if segment bbox doesn't overlap arc bbox.
                    arc_xmin = cx - r - near_arc_tol
                    arc_xmax = cx + r + near_arc_tol
                    arc_ymin = cy - r - near_arc_tol
                    arc_ymax = cy + r + near_arc_tol
                    seg_xmin, seg_xmax = sorted([p1[0], p2[0]])
                    seg_ymin, seg_ymax = sorted([p1[1], p2[1]])
                    if (
                        seg_xmax < arc_xmin
                        or seg_xmin > arc_xmax
                        or seg_ymax < arc_ymin
                        or seg_ymin > arc_ymax
                    ):
                        continue

                    # Analytic segment-circle intersection.
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    fx = p1[0] - cx
                    fy = p1[1] - cy
                    a = dx * dx + dy * dy
                    b = 2 * (fx * dx + fy * dy)
                    c = fx * fx + fy * fy - r * r
                    disc = b * b - 4 * a * c
                    if disc < 0 or a == 0:
                        continue
                    sd = disc**0.5
                    # Polygon-arc deviation upper bound: r * (1 - cos(dθ/2)).
                    # For a polygon with min_arc_points covering an arc, dθ
                    # ≤ 2π / min_arc_points; conservatively buffer by 1% of r.
                    interior_buffer = _interior_buffer_for_radius(slab, r)
                    inside_region = slab.footprint.buffer(interior_buffer)
                    for t in ((-b - sd) / (2 * a), (-b + sd) / (2 * a)):
                        if t < -near_arc_tol or t > 1 + near_arc_tol:
                            continue
                        ix = p1[0] + t * dx
                        iy = p1[1] + t * dy
                        # Must be inside the slab footprint (buffered to
                        # absorb polygon-vs-arc deviation up to ~r%).
                        if not inside_region.contains(shapely.geometry.Point(ix, iy)):
                            continue
                        if _on_polygon_vertex(ix, iy):
                            continue
                        raise StructuredArcSplitError(
                            f"Slab {slab.physical_name}: neighbour "
                            f"#{nb_idx} boundary segment "
                            f"({p1[0]:.4f}, {p1[1]:.4f}) -> "
                            f"({p2[0]:.4f}, {p2[1]:.4f}) crosses arc "
                            f"(center=({cx:.4f}, {cy:.4f}), r={r:.4f}) at "
                            f"xy=({ix:.6f}, {iy:.6f}), which is not an "
                            f"original arc polygon vertex of the slab's "
                            f"footprint.\n\n"
                            f"The polygon-based partition will disagree "
                            f"with BOP's true-arc cut, producing extra OCC "
                            f"sub-faces / micro-vertices that break the "
                            f"structured wedge construction.\n\n"
                            f"Remediation (any of):\n"
                            f"  - Align the neighbour boundary with an "
                            f"existing polygon vertex of the arc "
                            f"(e.g. for a 32-vertex disc, x=0 falls on "
                            f"polygon vertices at (0, +r) and (0, -r)).\n"
                            f"  - Densify the arc polygon at construction "
                            f"so a polygon vertex lands at the cut "
                            f"position.\n"
                            f"  - Move the neighbour off the arc "
                            f"footprint.\n"
                            f"  - Set identify_arcs=False on the slab to "
                            f"use the polygon approximation throughout."
                        )


def _validate_arc_partition_aligned(
    slab: Slab,
    arc_index: _ArcIndex,
    tol: float,
) -> None:
    """Legacy / unused: post-polygonize check kept as documentation only.

    Raises StructuredArcSplitError if a piece-boundary vertex lies near
    an arc circle but isn't an arc polygon vertex.

    polygonize introduces a NEW polygon vertex where the neighbour
    boundary's straight edge crosses the slab polygon's chord. That new
    vertex lies on the chord — slightly OFF the true arc circle (offset
    grows with chord length). BOP, on the other hand, cuts the true OCC
    arc at the geometric arc/line intersection point. The two
    intersection positions disagree by up to ~chord_length * (1 - cos(θ/2))
    for a chord subtending angle θ. The mismatch produces extra OCC
    sub-faces / 6-corner laterals downstream.

    Detection: walk each piece's polygon ring. For each vertex that
    lies near one of the original arcs' circles (within arc_tolerance)
    but is NOT in arc_index.vertex_to_arcs (= it's a polygonize-
    introduced crossing, not an original arc polygon vertex), raise.
    """
    from meshwell.structured.spec import StructuredArcSplitError

    if slab.face_partition_provenance is None or not arc_index.arcs:
        return

    ndigits = arc_index.ndigits

    def _key(x: float, y: float) -> tuple[float, float]:
        return (round(x, ndigits), round(y, ndigits))

    # Tolerance for "vertex lies near an arc circle": use arc_tolerance
    # times a safety factor so small polygon-vs-arc offsets are caught.
    near_arc_tol = tol

    for piece_idx, piece in enumerate(slab.face_partition):
        # Walk all rings (exterior + interiors) of the piece polygon.
        rings = [("exterior", list(piece.exterior.coords))]
        for h_idx, interior in enumerate(piece.interiors):
            rings.append((f"interior[{h_idx}]", list(interior.coords)))

        for ring_name, coords in rings:
            for x, y in coords:
                if _key(x, y) in arc_index.vertex_to_arcs:
                    continue  # original arc polygon vertex — fine
                # Check distance to each arc's circle.
                for arc in arc_index.arcs:
                    cx, cy = arc.center[0], arc.center[1]
                    dist_to_circle = abs(
                        ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 - arc.radius
                    )
                    if dist_to_circle < near_arc_tol:
                        # This vertex sits near the arc circle but isn't an
                        # original polygon vertex of it — polygonize crossing.
                        raise StructuredArcSplitError(
                            f"Slab {slab.physical_name} piece {piece_idx} "
                            f"({ring_name}): a neighbour boundary cuts the "
                            f"arc (center=({cx:.4f}, {cy:.4f}), "
                            f"r={arc.radius:.4f}) at xy=({x:.6f}, {y:.6f}), "
                            f"which is not an original arc polygon vertex "
                            f"of the slab's footprint (distance to circle: "
                            f"{dist_to_circle:.2e}). The polygon-based "
                            f"partition will disagree with BOP's true-arc "
                            f"cut, producing extra OCC sub-faces.\n\n"
                            f"Remediation (any of):\n"
                            f"  - Align the neighbour boundary with an "
                            f"existing polygon vertex of the arc "
                            f"(e.g. for a 32-vertex disc, x=0 falls on "
                            f"polygon vertices at (0, +r) and (0, -r)).\n"
                            f"  - Densify the arc polygon at construction "
                            f"so a polygon vertex lands at the cut "
                            f"position.\n"
                            f"  - Move the neighbour off the arc "
                            f"footprint.\n"
                            f"  - Set identify_arcs=False on the slab to "
                            f"use the polygon approximation throughout."
                        )


def _collect_stack_boundaries(stack: list[Slab], entities: list[Any]) -> list[Any]:
    """Boundaries to feed the arrangement for this connected component.

    Returns a list of shapely LineString/MultiLineString geometries:
    - Each stack member slab's resolved_footprint.boundary (falling back
      to slab.footprint.boundary when resolved_footprint is None).
    - Each unstructured entity whose z-range touches any z-plane of the
      stack — its boundary contributes line cuts only (no arcs).
    """
    boundaries: list[Any] = []
    seen_source_indices = {s.source_index for s in stack}

    # Stack member resolved footprints.
    for slab in stack:
        fp = (
            slab.resolved_footprint
            if slab.resolved_footprint is not None
            else slab.footprint
        )
        if not fp.is_empty:
            boundaries.append(fp.boundary)

    # Stack's z-planes.
    z_planes = set()
    for slab in stack:
        z_planes.add(slab.zlo)
        z_planes.add(slab.zhi)

    # Unstructured z-touching entities.
    for i, ent in enumerate(entities):
        if i in seen_source_indices:
            continue
        if getattr(ent, "structured", False):
            continue
        rng = _entity_z_range(ent)
        if rng is None:
            continue
        zmin, zmax = rng
        if any(abs(zmin - z) < _Z_TOL or abs(zmax - z) < _Z_TOL for z in z_planes):
            fp = _entity_footprint(ent)
            if fp is not None:
                boundaries.append(fp.boundary)

    return boundaries


def _planar_arrangement(
    boundaries: list[Any],
) -> tuple[list[ArrangementEdge], list[ArrangementFace]]:
    """Build the planar arrangement from a list of boundary geometries.

    Algorithm:
      1. ``merged = unary_union(boundaries)`` — shapely inserts vertices
         at every curve crossing.
      2. ``polygonize(merged)`` — gives the arrangement faces.
      3. Extract edges: walk each face's exterior ring; consecutive vertex
         pairs are candidate edges. Dedup by canonical (sorted-by-first-
         point, then-by-second-point) coordinate tuples, since each
         internal edge appears on the boundary of exactly two faces.

    Returns (edges, faces) where each face's boundary list references the
    edge_ids in the returned edges list.
    """
    merged = unary_union(boundaries)
    raw_polygons = list(polygonize(merged))

    def _key(p1, p2, ndigits=9):
        a = (round(p1[0], ndigits), round(p1[1], ndigits))
        b = (round(p2[0], ndigits), round(p2[1], ndigits))
        return (a, b) if a <= b else (b, a)

    edge_by_key: dict[tuple, int] = {}
    edges: list[ArrangementEdge] = []
    faces: list[ArrangementFace] = []

    for face_id, poly in enumerate(raw_polygons):
        coords = list(poly.exterior.coords)
        boundary_list: list[tuple[int, bool]] = []
        for i in range(len(coords) - 1):
            p1, p2 = coords[i], coords[i + 1]
            key = _key(p1, p2)
            if key not in edge_by_key:
                a, b = key
                edge = ArrangementEdge(
                    edge_id=len(edges),
                    vertices=(a, b),
                    circle=None,
                )
                edge_by_key[key] = edge.edge_id
                edges.append(edge)
            edge_id = edge_by_key[key]
            # Determine traversal direction: the edge canonical orientation
            # is its vertices[0] -> vertices[-1]. If face walks p1 -> p2 and
            # that equals the canonical direction, reversed=False; else True.
            p1_round = (round(p1[0], 9), round(p1[1], 9))
            reversed_flag = p1_round != edges[edge_id].vertices[0]
            boundary_list.append((edge_id, reversed_flag))
        faces.append(
            ArrangementFace(
                face_id=face_id,
                polygon=poly,
                boundary=boundary_list,
            )
        )

    return edges, faces


@phase_timed("plan")
def build_plan(entities: list[Any]) -> StructuredPlan:
    """Top-level planner: entities -> validated, partitioned StructuredPlan.

    Pipeline:

    1. ``gather_structured_entities`` filters and pairs entities with specs.
    2. ``expand_to_slabs`` produces one raw Slab per (entity, z-interval).
    3. ``validate_and_resolve_overlap`` applies Policy B; drops losers,
       records OverlapPairs. Raises ``StructuredOverlapError`` on mismatch.
    4. ``compute_face_partition`` decorates each surviving slab with its
       xy partition based on touching neighbour entities.

    The returned StructuredPlan is frozen and ready for the phantom +
    builder stages (Phase 2 / Phase 3).
    """
    pairs = gather_structured_entities(entities)
    if not pairs:
        return StructuredPlan(slabs=(), z_planes=(), overlaps=())
    raw_slabs = expand_to_slabs(pairs)
    kept_slabs, overlaps = validate_and_resolve_overlap(raw_slabs, entities)
    _validate_no_mid_height_cuts(kept_slabs, entities)
    _validate_no_unstructured_lateral_neighbour(kept_slabs, entities)
    compute_face_partition(kept_slabs, entities)
    z_set: set[float] = set()
    for s in kept_slabs:
        z_set.add(s.zlo)
        z_set.add(s.zhi)
    z_planes = sorted(z_set)
    return StructuredPlan(
        slabs=tuple(kept_slabs),
        z_planes=tuple(z_planes),
        overlaps=tuple(overlaps),
    )


def _fit_arc_to_edge(
    vertices: tuple[tuple[float, float], ...],
    arc_tolerance: float,
) -> "CanonicalCircle | None":
    """Try to fit a circle through the edge's vertices.

    Returns CanonicalCircle if all vertices lie on a common circle
    within ``arc_tolerance``; else None. Requires >=3 vertices (since
    2 points underdetermine a circle).

    Uses the same circle-fitting routine that GeometryEntity uses for
    arc identification today, so the result is consistent with existing
    arc detection.
    """
    if len(vertices) < 3:
        return None

    import numpy as np

    from meshwell.geometry_entity import fit_circle_2d

    pts = np.array(vertices)
    center, radius, residual = fit_circle_2d(pts)
    if residual > arc_tolerance:
        return None
    if radius > 1e6:  # degenerate — colinear points "fit" infinite radius
        return None
    return CanonicalCircle(
        center=(float(center[0]), float(center[1])), radius=float(radius)
    )


def _coalesce_adjacent_arcs(
    edges: list[ArrangementEdge],
    arc_tolerance: float,
) -> list[ArrangementEdge]:
    """Merge adjacent ArrangementEdges sharing an endpoint and circle.

    Two edges merge iff:
      - both have non-None circles matching within arc_tolerance on
        (center, radius)
      - they share an endpoint (last vertex of one == first vertex of
        another, or any other endpoint pairing)

    Merging is greedy: scan all pairs, merge the first matching pair,
    repeat until no more merges possible. Line edges (circle=None) are
    not merged here.

    Output edge_ids are re-assigned to be contiguous from 0.
    """

    def _circles_match(c1: "CanonicalCircle", c2: "CanonicalCircle") -> bool:
        return (
            abs(c1.center[0] - c2.center[0]) < arc_tolerance
            and abs(c1.center[1] - c2.center[1]) < arc_tolerance
            and abs(c1.radius - c2.radius) < arc_tolerance
        )

    def _endpoints_match(p1, p2, tol=1e-9):
        return abs(p1[0] - p2[0]) < tol and abs(p1[1] - p2[1]) < tol

    def _try_merge(
        e1: ArrangementEdge, e2: ArrangementEdge
    ) -> "ArrangementEdge | None":
        if e1.circle is None or e2.circle is None:
            return None
        if not _circles_match(e1.circle, e2.circle):
            return None
        v1_start, v1_end = e1.vertices[0], e1.vertices[-1]
        v2_start, v2_end = e2.vertices[0], e2.vertices[-1]
        if _endpoints_match(v1_end, v2_start):
            merged_verts = e1.vertices + e2.vertices[1:]
        elif _endpoints_match(v1_end, v2_end):
            merged_verts = e1.vertices + e2.vertices[-2::-1]
        elif _endpoints_match(v1_start, v2_start):
            merged_verts = e1.vertices[::-1] + e2.vertices[1:]
        elif _endpoints_match(v1_start, v2_end):
            merged_verts = e2.vertices + e1.vertices[1:]
        else:
            return None
        return ArrangementEdge(edge_id=-1, vertices=merged_verts, circle=e1.circle)

    work = list(edges)
    while True:
        merged_any = False
        for i in range(len(work)):
            if merged_any:
                break
            for j in range(i + 1, len(work)):
                m = _try_merge(work[i], work[j])
                if m is not None:
                    work = [*work[:i], m, *work[i + 1 : j], *work[j + 1 :]]
                    merged_any = True
                    break
        if not merged_any:
            break

    return [
        ArrangementEdge(edge_id=i, vertices=e.vertices, circle=e.circle)
        for i, e in enumerate(work)
    ]


def _assign_faces_to_slabs(
    faces: list[ArrangementFace],
    stack: list[Slab],
) -> None:
    """Set ``face_partition`` and ``face_partition_edges`` in place per containment.

    For each face, find every slab whose resolved_footprint contains
    face.polygon.representative_point(), and append the face's polygon
    to that slab's face_partition (creating the list if necessary).
    Mirrors the assignment to face_partition_edges (the face's boundary
    list of (edge_id, reversed) tuples).

    A face may be contained in zero slabs (e.g., a hole between
    resolved_footprints created by mesh_bool=False carving). Such faces
    are silently dropped.
    """
    # Initialize slab containers if not already.
    for slab in stack:
        if not slab.face_partition:
            slab.face_partition = []
        if slab.face_partition_edges is None:
            slab.face_partition_edges = []

    for face in faces:
        rep = face.polygon.representative_point()
        for slab in stack:
            fp = (
                slab.resolved_footprint
                if slab.resolved_footprint is not None
                else slab.footprint
            )
            if fp.contains(rep):
                slab.face_partition.append(face.polygon)
                slab.face_partition_edges.append(face.boundary)


def _build_provenance_shim(
    stack: list[Slab],
    arrangement: StackArrangement,
) -> None:
    """Derive face_partition_provenance from face_partition_edges + arrangement.

    Walked once per slab after assignment. The phantom builder consumes
    face_partition_provenance today (PieceArcEdge / PieceLineEdge); this
    shim builds it from the new ArrangementEdge data so phantom.py needs
    no changes.

    Only runs for slabs with identify_arcs=True; others get
    face_partition_provenance=None (matches existing behavior).
    """
    edges_by_id = {e.edge_id: e for e in arrangement.edges}

    for slab in stack:
        if not slab.identify_arcs:
            slab.face_partition_provenance = None
            continue
        if not slab.face_partition_edges or len(slab.face_partition) <= 1:
            slab.face_partition_provenance = None
            continue

        provenances: list[PieceProvenance] = []
        for piece_edges in slab.face_partition_edges:
            ext_edges = []
            for edge_id, reversed_flag in piece_edges:
                arr_edge = edges_by_id[edge_id]
                verts = arr_edge.vertices
                if reversed_flag:
                    verts = verts[::-1]
                if arr_edge.circle is not None:
                    pts_3d = tuple((v[0], v[1], 0.0) for v in verts)
                    ext_edges.append(
                        PieceArcEdge(
                            points=pts_3d,
                            center=(
                                arr_edge.circle.center[0],
                                arr_edge.circle.center[1],
                                0.0,
                            ),
                            radius=arr_edge.circle.radius,
                        )
                    )
                else:
                    p1 = (verts[0][0], verts[0][1], 0.0)
                    p2 = (verts[-1][0], verts[-1][1], 0.0)
                    ext_edges.append(PieceLineEdge(points=(p1, p2)))
            provenances.append(
                PieceProvenance(exterior_edges=ext_edges, interior_edges=[])
            )
        slab.face_partition_provenance = provenances
