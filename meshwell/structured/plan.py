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
    """Apply Policy B: record volumetric overlap pairs, fail on mismatch.

    Returns (all_slabs, overlap_pairs). Lower mesh_order wins; tie-break
    by source_index then z_interval_index for determinism.

    Updated 2026-05-25: no longer drops the loser slab. All slabs are kept
    and the loser's footprint is carved by the winner in Step 0
    (_resolve_sublevel_mesh_order). The OverlapPair is still recorded for
    diagnostics.
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
            break
        # OverlapPair recording (above) is unchanged.
        # New behavior: keep all slabs in the returned list. The loser's
        # footprint is carved by the winner in Step 0 (_resolve_sublevel_mesh_order).
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


_DEFAULT_SNAP_TOL = 1e-9

_MAX_ANNULAR_SPLIT_PASSES = 4


class StructuredArrangementError(RuntimeError):
    """Raised when the planar arrangement cannot resolve annular faces.

    The iteration cap (``_MAX_ANNULAR_SPLIT_PASSES``) fired without
    converging to a fully single-loop face_partition. Typically means
    the face has a topology the helper cannot handle (e.g. multiple
    interior rings that the deterministic ±x radial cuts cannot reach).
    """


def _snap_boundary_coords(geom: Any, tol: float) -> Any:
    """Snap every vertex of ``geom`` to a tolerance grid (pointwise).

    Inputs whose vertices differ by < ``tol`` collapse to bit-identical
    coordinates so ``shapely.unary_union`` recognises shared endpoints
    that floating-point construction (e.g. ``cos(0)`` vs ``cos(2*pi)``)
    would otherwise leave subtly different.
    """
    return shapely.set_precision(geom, grid_size=tol, mode="pointwise")


def _collect_stack_boundaries(stack: list[Slab], entities: list[Any]) -> list[Any]:
    """Boundaries to feed the arrangement for this connected component.

    Returns a list of shapely LineString/MultiLineString geometries:
    - Each stack member slab's resolved_footprint.boundary (falling back
      to slab.footprint.boundary when resolved_footprint is None).
    - Each unstructured entity whose z-range touches any z-plane of the
      stack — its boundary contributes line cuts only (no arcs).
    """
    return [b for b, _ in _collect_stack_boundaries_tagged(stack, entities)]


def _collect_stack_boundaries_tagged(
    stack: list[Slab], entities: list[Any]
) -> list[tuple[Any, bool]]:
    """Same as _collect_stack_boundaries but tags each boundary with identify_arcs.

    Returns ``(boundary_geometry, identify_arcs)`` pairs so downstream
    arc fitting can require at least one arc-bearing source contributor.
    Unstructured entities are tagged ``identify_arcs=False`` unconditionally.
    """
    boundaries: list[tuple[Any, bool]] = []
    seen_source_indices = {s.source_index for s in stack}

    # Stack member resolved footprints.
    for slab in stack:
        fp = (
            slab.resolved_footprint
            if slab.resolved_footprint is not None
            else slab.footprint
        )
        if not fp.is_empty:
            boundaries.append((fp.boundary, slab.identify_arcs))

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
                boundaries.append((fp.boundary, False))

    # Snap every contributing boundary's vertices to the tightest positive
    # ``point_tolerance`` among the stack's entities (or ``_DEFAULT_SNAP_TOL``
    # if none report one). Collapses floating-point near-duplicates (e.g.
    # cos(0) vs cos(2*pi)) so the downstream ``unary_union`` recognises them
    # as the same point.
    tol = _stack_snap_tolerance(stack, entities)
    return [(_snap_boundary_coords(b, tol), ident) for b, ident in boundaries]


def _stack_snap_tolerance(stack: list[Slab], entities: list[Any]) -> float:
    """Tightest positive ``point_tolerance`` of the stack's source entities.

    Falls back to ``_DEFAULT_SNAP_TOL`` (1e-9) when no entity reports a
    positive ``point_tolerance``.
    """
    candidate_tols = [
        pt
        for slab in stack
        if (pt := getattr(entities[slab.source_index], "point_tolerance", None))
        is not None
        and pt > 0
    ]
    return min(candidate_tols) if candidate_tols else _DEFAULT_SNAP_TOL


_RADIAL_CUT_Y_OFFSET = 1e-3


def _generate_radial_cuts_for_annular_face(poly: Polygon) -> list[Any]:
    """For one annular face, produce horizontal cut(s) local to that face.

    Cut is a horizontal line at ``y = hole_centroid_y + _RADIAL_CUT_Y_OFFSET``
    (small offset off the centroid's y so the cut crosses each ring
    between vertices of the polygon arc approximation rather than at
    them) clipped to ``poly``. Only the in-face portion is returned.

    Why off-axis + clipped (instead of either extreme):

    * Axis-aligned cut (y = hole_centroid_y): cut endpoints often land
      on existing polygon vertices (e.g. n-gon arc vertices on the
      angular grid). ``shapely.unary_union`` does NOT split a closed
      ``LinearRing`` at an existing vertex when another curve merely
      touches it there. polygonize then keeps the ring as one loop and
      the annulus survives undivided.
    * Off-axis cut that's NOT clipped: it crosses every face in the
      arrangement (including non-annular ring footprints elsewhere),
      creating sliver pieces at near-tangent crossings.
    * Off-axis + clipped: the cut crosses each ring of the annular face
      at a NEW transverse point (between vertices, since the cut is
      off-axis), unary_union nodes the crossings correctly, polygonize
      fragments the annular face into single-loop sub-pieces — and the
      cut doesn't reach into other faces because it's clipped to ``poly``.
    """
    from shapely.geometry import LineString, MultiLineString

    if not poly.interiors:
        return []
    cx, cy = next(iter(poly.interiors[0].centroid.coords))
    minx, _miny, maxx, _maxy = poly.bounds
    span = max(maxx - minx, 1.0)
    half_extent = span * 2 + 1
    y = cy + _RADIAL_CUT_Y_OFFSET
    long_cut = LineString([(cx - half_extent, y), (cx + half_extent, y)])
    clipped = long_cut.intersection(poly)
    if clipped.is_empty:
        return []
    segments: list = []
    if isinstance(clipped, LineString):
        segments = [clipped]
    elif isinstance(clipped, MultiLineString):
        segments = [
            g for g in clipped.geoms if isinstance(g, LineString) and not g.is_empty
        ]

    # Extend each segment by a tiny overshoot past each clipped endpoint.
    # polygonize won't fragment a closed ring at a touching endpoint — it
    # needs the cut to CROSS through the ring transversely. The overshoot
    # turns each terminating endpoint into a transverse crossing without
    # extending far enough to graze unrelated geometry elsewhere in the
    # arrangement (e.g. ring footprints whose boundaries we shouldn't
    # split with this annular-split cut).
    overshoot = 1e-9 * half_extent
    out: list[Any] = []
    for seg in segments:
        coords = list(seg.coords)
        if len(coords) < 2:
            continue
        (x0, y0), (x1, y1) = coords[0], coords[-1]
        dx, dy = x1 - x0, y1 - y0
        length = (dx * dx + dy * dy) ** 0.5
        if length == 0:
            continue
        ux, uy = dx / length, dy / length
        out.append(
            LineString(
                [
                    (x0 - ux * overshoot, y0 - uy * overshoot),
                    (x1 + ux * overshoot, y1 + uy * overshoot),
                ]
            )
        )
    return out


def _planar_arrangement(
    boundaries: list[Any],
) -> tuple[list[ArrangementEdge], list[ArrangementFace]]:
    """Build the planar arrangement from a list of boundary geometries.

    Algorithm:
      1. ``merged = unary_union(boundaries)`` — splits curves at every
         intersection. Result is a LineString or MultiLineString whose
         components are maximal non-crossing curves (the arrangement edges).
      2. Each component LineString becomes one ArrangementEdge carrying
         ALL its vertices (so downstream arc fitting has enough points).
      3. ``polygonize(merged)`` gives the arrangement faces.
      4. Map each face's boundary to ArrangementEdge ids by walking
         the face exterior coords and matching each segment to its
         containing edge. Consecutive segments on the same edge collapse
         to one boundary entry.
    """
    merged = unary_union(boundaries)
    raw_polygons = list(polygonize(merged))

    # Step 2: extract maximal-run edges.
    line_strings = list(merged.geoms) if hasattr(merged, "geoms") else [merged]

    edges: list[ArrangementEdge] = []
    # vertex_pair_key -> (edge_id, position_in_edge, forward_direction)
    # We index every consecutive pair in every edge so face-segment lookups are O(1).
    pair_to_edge: dict[tuple, tuple[int, int, bool]] = {}

    def _round(p, ndigits=9):
        return (round(p[0], ndigits), round(p[1], ndigits))

    def _pair_key(p1, p2):
        a, b = _round(p1), _round(p2)
        return (a, b) if a <= b else (b, a)

    for ls in line_strings:
        verts = tuple((c[0], c[1]) for c in ls.coords)
        # Drop the closing repeat if present (LinearRing closures).
        if len(verts) >= 2 and _round(verts[0]) == _round(verts[-1]):
            verts = verts[:-1]
            if len(verts) < 2:
                continue
        edge = ArrangementEdge(edge_id=len(edges), vertices=verts, circle=None)
        edges.append(edge)
        for k in range(len(verts) - 1):
            p1, p2 = verts[k], verts[k + 1]
            key = _pair_key(p1, p2)
            # Forward = True iff (p1, p2) traversal == canonical edge direction.
            pair_to_edge[key] = (edge.edge_id, k, _round(p1) == _round(verts[k]))

    # Step 4: walk each face's exterior and map segments to edges.
    faces: list[ArrangementFace] = []
    for face_id, poly in enumerate(raw_polygons):
        coords = list(poly.exterior.coords)
        boundary_list: list[tuple[int, bool]] = []
        prev_edge_id = None
        for i in range(len(coords) - 1):
            p1, p2 = coords[i], coords[i + 1]
            key = _pair_key(p1, p2)
            lookup = pair_to_edge.get(key)
            if lookup is None:
                # Segment doesn't match any edge — shouldn't happen if
                # boundaries are well-formed. Skip with no boundary entry.
                continue
            edge_id, pos, _canonical_forward = lookup
            # Determine reversed_flag: does face direction p1 -> p2 match
            # the edge's canonical direction edge.vertices[pos] -> edge.vertices[pos+1]?
            edge_verts = edges[edge_id].vertices
            edge_p1 = edge_verts[pos]
            edge_p2 = edge_verts[pos + 1]
            if _round(p1) == _round(edge_p1) and _round(p2) == _round(edge_p2):
                reversed_flag = False
            else:
                reversed_flag = True
            if edge_id != prev_edge_id:
                boundary_list.append((edge_id, reversed_flag))
                prev_edge_id = edge_id
            # else: same edge as previous segment, just walking further
            #       along it — no new boundary entry.
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
    _resolve_sublevel_mesh_order(kept_slabs, entities)
    arrangements = build_stack_arrangements(kept_slabs, entities)
    assign_face_partition_from_arrangement(kept_slabs, arrangements)
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
    min_arc_points: int = 4,
) -> "CanonicalCircle | None":
    """Try to fit a circle through the edge's vertices.

    Returns CanonicalCircle if all vertices lie on a common circle
    within ``arc_tolerance``; else None. Requires >=``min_arc_points``
    vertices (3 underdetermines a real arc — any 3 non-colinear points
    fit a circle perfectly, so any sharp polygon corner would be
    spuriously classified as an arc).

    Uses the same circle-fitting routine that GeometryEntity uses for
    arc identification today, so the result is consistent with existing
    arc detection.
    """
    if len(vertices) < max(min_arc_points, 4):
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

    def _round_pt(p, ndigits=9):
        return (round(p[0], ndigits), round(p[1], ndigits))

    # Pre-compute: for each rounded point, the set of non-arc edges
    # that have that point as an endpoint. A shared endpoint that's also
    # a non-arc endpoint is a T-junction (a cut), so adjacent arcs there
    # must NOT be merged across the cut.
    non_arc_endpoints: set = set()
    for e in edges:
        if e.circle is None:
            non_arc_endpoints.add(_round_pt(e.vertices[0]))
            non_arc_endpoints.add(_round_pt(e.vertices[-1]))

    def _try_merge(
        e1: ArrangementEdge, e2: ArrangementEdge
    ) -> "ArrangementEdge | None":
        if e1.circle is None or e2.circle is None:
            return None
        if not _circles_match(e1.circle, e2.circle):
            return None
        v1_start, v1_end = e1.vertices[0], e1.vertices[-1]
        v2_start, v2_end = e2.vertices[0], e2.vertices[-1]
        # Block merging when two arcs share BOTH endpoints (they form a
        # closed loop together — typically halves of a source circle that
        # was split by a cut; merging would erase the cut).
        share_a = _endpoints_match(v1_start, v2_start) or _endpoints_match(
            v1_start, v2_end
        )
        share_b = _endpoints_match(v1_end, v2_start) or _endpoints_match(v1_end, v2_end)
        if share_a and share_b:
            return None
        # Locate the shared endpoint (if any).
        shared = None
        if _endpoints_match(v1_end, v2_start):
            shared = v1_end
            merged_verts = e1.vertices + e2.vertices[1:]
        elif _endpoints_match(v1_end, v2_end):
            shared = v1_end
            merged_verts = e1.vertices + e2.vertices[-2::-1]
        elif _endpoints_match(v1_start, v2_start):
            shared = v1_start
            merged_verts = e1.vertices[::-1] + e2.vertices[1:]
        elif _endpoints_match(v1_start, v2_end):
            shared = v1_start
            merged_verts = e2.vertices + e1.vertices[1:]
        else:
            return None
        # T-junction check: if the shared endpoint is also an endpoint of
        # some non-arc edge in the arrangement, a cut terminates here.
        # The two arcs were deliberately split at the cut — don't merge.
        if _round_pt(shared) in non_arc_endpoints:
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


def build_stack_arrangements(
    slabs: list[Slab],
    entities: list[Any],
) -> dict[int, StackArrangement]:
    """Build one StackArrangement per connected z-touching component.

    Assumes ``_resolve_sublevel_mesh_order`` has already populated each
    slab's ``resolved_footprint``.

    Returns dict mapping component_index -> StackArrangement.
    """
    components = _connected_z_components(slabs)
    arrangements: dict[int, StackArrangement] = {}
    for comp_idx, stack in enumerate(components):
        tagged = _collect_stack_boundaries_tagged(stack, entities)
        if not tagged:
            arrangements[comp_idx] = StackArrangement(edges=[], faces=[])
            continue
        boundaries = [b for b, _ in tagged]
        # Build a "point on an identify_arcs=True boundary" set. We will
        # use it to decide which arrangement edges are candidates for arc
        # fitting (per spec: at least one source entity contributing the
        # edge must have identify_arcs=True).
        arc_source_pts: set = set()
        for boundary, ident in tagged:
            if not ident:
                continue
            for ls in (
                list(boundary.geoms) if hasattr(boundary, "geoms") else [boundary]
            ):
                for c in ls.coords:
                    arc_source_pts.add((round(c[0], 9), round(c[1], 9)))
        # Iterate: any face with non-empty interiors (annular) gets split
        # by two ±x radial cuts. We process ONE annular face per pass —
        # if two annular faces share a ring (concentric nested annuli)
        # their ±x cuts are collinear and ``unary_union`` would merge
        # them into a single longer line, losing the intermediate node
        # at the shared ring and leaving the inner annulus undivided.
        # Splitting one annulus per pass forces each cut to node cleanly
        # against the rings BEFORE the next pass adds a collinear cut.
        for _ in range(_MAX_ANNULAR_SPLIT_PASSES):
            edges, faces = _planar_arrangement(boundaries)
            annular = [f for f in faces if f.polygon.interiors]
            if not annular:
                break
            boundaries.extend(
                _generate_radial_cuts_for_annular_face(annular[0].polygon)
            )
        else:
            raise StructuredArrangementError(
                f"Annular face split did not converge after "
                f"{_MAX_ANNULAR_SPLIT_PASSES} passes for stack component "
                f"{comp_idx}; arrangement still contains face(s) with "
                f"interior ring(s)."
            )
        # Determine effective arc_tolerance for this stack: minimum of
        # all member slabs that have identify_arcs=True.
        arc_tols = [s.arc_tolerance for s in stack if s.identify_arcs]
        tol = min(arc_tols) if arc_tols else 1e-3
        # Minimum vertices per arc: minimum over identify_arcs slabs.
        min_arc_pts_list = [
            getattr(s, "min_arc_points", 4) for s in stack if s.identify_arcs
        ]
        min_arc_pts = min(min_arc_pts_list) if min_arc_pts_list else 4

        def _edge_has_arc_source(edge_verts, _pts=arc_source_pts):
            """Return True iff any vertex lies on an identify_arcs=True boundary.

            An edge is a candidate for arc fitting iff at least one of its
            sample vertices lies on an identify_arcs=True boundary.
            """
            return any((round(v[0], 9), round(v[1], 9)) in _pts for v in edge_verts)

        # Step D — try arc fit per edge (only if any stack member identifies arcs).
        if arc_tols:
            new_edges = []
            for e in edges:
                circle = None
                if _edge_has_arc_source(e.vertices):
                    circle = _fit_arc_to_edge(e.vertices, tol, min_arc_pts)
                new_edges.append(
                    ArrangementEdge(
                        edge_id=e.edge_id, vertices=e.vertices, circle=circle
                    )
                )
            edges = new_edges
            # Step E — coalesce adjacent arcs.
            edges = _coalesce_adjacent_arcs(edges, tol)
            # Rebuild face boundaries to reference new edge IDs.
            edges, faces = _rebuild_face_boundaries(edges, faces)
        arrangements[comp_idx] = StackArrangement(edges=edges, faces=faces)
    return arrangements


def _rebuild_face_boundaries(
    new_edges: list[ArrangementEdge],
    faces: list[ArrangementFace],
) -> tuple[list[ArrangementEdge], list[ArrangementFace]]:
    """After coalesce, faces may reference old edge_ids — re-resolve them.

    For each face, walk its polygon's exterior; for each polygon-edge segment,
    find the new ArrangementEdge whose vertices contain that segment and assign
    the (edge_id, reversed_flag) accordingly.
    """

    def _segment_covered_by_edge(p1, p2, edge_verts, tol=1e-9):
        """Returns reversed flag if p1->p2 traversal lies on edge_verts (canonical) or reversed."""
        for k, v in enumerate(edge_verts):
            if abs(v[0] - p1[0]) < tol and abs(v[1] - p1[1]) < tol:
                if k + 1 < len(edge_verts):
                    n = edge_verts[k + 1]
                    if abs(n[0] - p2[0]) < tol and abs(n[1] - p2[1]) < tol:
                        return False
                if k > 0:
                    p = edge_verts[k - 1]
                    if abs(p[0] - p2[0]) < tol and abs(p[1] - p2[1]) < tol:
                        return True
        return None

    new_faces: list[ArrangementFace] = []
    for face in faces:
        coords = list(face.polygon.exterior.coords)
        new_boundary: list[tuple[int, bool]] = []
        i = 0
        while i < len(coords) - 1:
            p1, p2 = coords[i], coords[i + 1]
            matched = False
            for edge in new_edges:
                rev = _segment_covered_by_edge(p1, p2, edge.vertices)
                if rev is None:
                    continue
                new_boundary.append((edge.edge_id, rev))
                edge_len = len(edge.vertices) - 1
                consumed = 1
                while consumed < edge_len and i + consumed + 1 < len(coords):
                    pn, pn1 = coords[i + consumed], coords[i + consumed + 1]
                    rev2 = _segment_covered_by_edge(pn, pn1, edge.vertices)
                    if rev2 is None or rev2 != rev:
                        break
                    consumed += 1
                i += consumed
                matched = True
                break
            if not matched:
                i += 1
        new_faces.append(
            ArrangementFace(
                face_id=face.face_id, polygon=face.polygon, boundary=new_boundary
            )
        )
    return new_edges, new_faces


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

    def _build_edges_list(
        boundary_list: list[tuple[int, bool]],
    ) -> list:
        out: list = []
        # If the boundary is a single arc edge, the ring closes by wrapping
        # back to the starting vertex. The phantom builder expects
        # ``points[0] == points[-1]`` to recognize a closed arc, so we
        # explicitly append the closing repeat in that case.
        is_lone_closed_arc = (
            len(boundary_list) == 1
            and edges_by_id[boundary_list[0][0]].circle is not None
        )
        for edge_id, reversed_flag in boundary_list:
            arr_edge = edges_by_id[edge_id]
            verts = arr_edge.vertices
            if reversed_flag:
                verts = verts[::-1]
            if arr_edge.circle is not None:
                pts_3d = tuple((v[0], v[1], 0.0) for v in verts)
                if is_lone_closed_arc and pts_3d[0] != pts_3d[-1]:
                    pts_3d = (*pts_3d, pts_3d[0])
                out.append(
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
                out.append(PieceLineEdge(points=(p1, p2)))
        return out

    def _ring_to_boundary(ring_coords) -> list[tuple[int, bool]]:
        """Walk a ring's coords and emit (edge_id, reversed) entries.

        Mirrors _rebuild_face_boundaries' inner loop but operates on a
        raw coord list (works for both exterior and interior rings).
        """
        coords = list(ring_coords)
        boundary: list[tuple[int, bool]] = []
        i = 0

        def _segment_covered_by_edge(p1, p2, edge_verts, tol=1e-9):
            for k, v in enumerate(edge_verts):
                if abs(v[0] - p1[0]) < tol and abs(v[1] - p1[1]) < tol:
                    if k + 1 < len(edge_verts):
                        n = edge_verts[k + 1]
                        if abs(n[0] - p2[0]) < tol and abs(n[1] - p2[1]) < tol:
                            return False
                    if k > 0:
                        p = edge_verts[k - 1]
                        if abs(p[0] - p2[0]) < tol and abs(p[1] - p2[1]) < tol:
                            return True
            return None

        while i < len(coords) - 1:
            p1, p2 = coords[i], coords[i + 1]
            matched = False
            for arr_edge in arrangement.edges:
                rev = _segment_covered_by_edge(p1, p2, arr_edge.vertices)
                if rev is None:
                    continue
                boundary.append((arr_edge.edge_id, rev))
                edge_len = len(arr_edge.vertices) - 1
                consumed = 1
                while consumed < edge_len and i + consumed + 1 < len(coords):
                    pn, pn1 = coords[i + consumed], coords[i + consumed + 1]
                    rev2 = _segment_covered_by_edge(pn, pn1, arr_edge.vertices)
                    if rev2 is None or rev2 != rev:
                        break
                    consumed += 1
                i += consumed
                matched = True
                break
            if not matched:
                i += 1
        return boundary

    for slab in stack:
        if not slab.identify_arcs:
            slab.face_partition_provenance = None
            continue
        if not slab.face_partition_edges or len(slab.face_partition) <= 1:
            slab.face_partition_provenance = None
            continue

        provenances: list[PieceProvenance] = []
        for piece_poly, piece_edges in zip(
            slab.face_partition, slab.face_partition_edges, strict=False
        ):
            ext_edges = _build_edges_list(piece_edges)
            interior_edges: list[list] = []
            # Walk each interior ring of the piece polygon and resolve edges.
            for interior_ring in piece_poly.interiors:
                int_boundary = _ring_to_boundary(interior_ring.coords)
                if int_boundary:
                    interior_edges.append(_build_edges_list(int_boundary))
            provenances.append(
                PieceProvenance(exterior_edges=ext_edges, interior_edges=interior_edges)
            )
        slab.face_partition_provenance = provenances


def assign_face_partition_from_arrangement(
    slabs: list[Slab],
    arrangements: dict[int, StackArrangement],
) -> None:
    """Distribute arrangement faces to slabs and build provenance.

    Maps each slab to its containing component (via the same connected-
    components grouping used by build_stack_arrangements), then runs
    Step F (face assignment) and Step G (provenance shim) per stack.
    """
    components = _connected_z_components(slabs)
    for comp_idx, stack in enumerate(components):
        arrangement = arrangements.get(comp_idx)
        if arrangement is None:
            continue
        # Reset face_partition / face_partition_edges (in case slabs were
        # populated by a previous run).
        for slab in stack:
            slab.face_partition = []
            slab.face_partition_edges = []
        _assign_faces_to_slabs(arrangement.faces, stack)
        _build_provenance_shim(stack, arrangement)
        # Slabs whose face_partition is empty (fully dominated by mesh_bool=False
        # carving or by mesh_order overlap) get a one-piece fallback so phantom
        # build doesn't crash. They'll produce no actual mesh content.
        for slab in stack:
            if not slab.face_partition:
                slab.face_partition = [
                    slab.resolved_footprint
                    if slab.resolved_footprint is not None
                    else slab.footprint
                ]
                slab.face_partition_edges = [[]]
