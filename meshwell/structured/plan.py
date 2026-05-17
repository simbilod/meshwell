"""Plan stage for the clean structured-polyprism pipeline.

Public surface: ``build_plan(entities) -> StructuredPlan``. Private
helpers handle the pipeline steps (gather, expand, validate, partition).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import polygonize, unary_union

from meshwell.structured.logging import phase_timed
from meshwell.structured.spec import (
    OverlapPair,
    PieceArcEdge,
    PieceLineEdge,
    PieceProvenance,
    Slab,
    StructuredExtrusionResolutionSpec,
    StructuredOverlapError,
    StructuredPlan,
)

_Z_TOL = 1e-9  # exact-match tolerance for z-extent comparison


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


@dataclass(frozen=True)
class _IndexedArc:
    """An arc identified on the original footprint.

    Full vertex sequence preserved in boundary order.
    """

    arc_id: int
    center: tuple[float, float, float]
    radius: float
    points: tuple[tuple[float, float, float], ...]  # length >= min_arc_points


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


def compute_face_partition(slabs: list[Slab], entities: list[Any]) -> None:
    """Compute slab.face_partition (and face_partition_provenance) in place.

    For each slab, decompose its footprint into pairwise-disjoint pieces
    based on the union of any neighbouring entity footprints touching
    z=zlo or z=zhi. No-neighbour case: one piece = the whole footprint.

    When ``slab.identify_arcs=True``, also computes
    ``slab.face_partition_provenance``: a parallel list of
    :class:`PieceProvenance` objects that label each piece's boundary
    segments as arc-inherited or straight-cut. The arc index is built once
    from the original footprint (where the heuristic is reliable), avoiding
    mis-classification of seam-cut vertices.
    """
    own_indices_by_slab = {id(s): {s.source_index} for s in slabs}

    for slab in slabs:
        skip = own_indices_by_slab[id(slab)]
        neighbours_lo = _neighbours_touching_z(slab.zlo, entities, skip)
        neighbours_hi = _neighbours_touching_z(slab.zhi, entities, skip)
        all_neighbour_polys = neighbours_lo + neighbours_hi
        if not all_neighbour_polys:
            slab.face_partition = [slab.footprint]
            # Single piece = no split: the heuristic path in phantom.py
            # (_make_arc_wire_from_coords) produces OCC geometry that
            # exactly matches PolyPrism.instanciate_occ. No provenance needed.
            continue
        # Phase 5(d): use individual neighbour boundaries (common refinement) so
        # overlapping neighbours' internal seams appear in the partition. Otherwise
        # BOP cuts the slab face at boundaries the planner didn't anticipate,
        # producing multi-output-face per piece.
        all_boundaries = unary_union([poly.boundary for poly in all_neighbour_polys])
        boundary = slab.footprint.boundary
        combined = unary_union([boundary, all_boundaries.intersection(slab.footprint)])
        raw = list(polygonize(combined))
        pieces = [
            piece
            for piece in raw
            if slab.footprint.contains(piece.representative_point())
        ]
        slab.face_partition = pieces if pieces else [slab.footprint]
        # Provenance is only needed when there are multiple pieces (split case).
        # When there is only one piece, the heuristic path in phantom.py produces
        # the correct OCC geometry without provenance. This avoids TShape mismatches
        # that arise from differences in seam rotation between the provenance arc
        # classifier and the canonical-seam-aware heuristic decomposer.
        if slab.identify_arcs and len(slab.face_partition) > 1:
            arc_index = _build_arc_index_from_footprint(
                slab.footprint,
                identify_arcs=True,
                min_arc_points=slab.min_arc_points,
                arc_tolerance=slab.arc_tolerance,
            )
            slab.face_partition_provenance = [
                _classify_piece_boundary(piece, arc_index)
                for piece in slab.face_partition
            ]


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
