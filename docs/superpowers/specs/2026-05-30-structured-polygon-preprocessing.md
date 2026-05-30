# Structured polygon pre-processing — pipeline summary

**Date:** 2026-05-30
**Scope:** the planner-side work that happens between a user passing a
list of `PolyPrism` entities to `generate_mesh` and Phase 3's
`build_cohort_envelope` consuming a `StructuredPlan`.

The entry point is `build_plan(entities)` in [meshwell/structured/plan.py:1601](meshwell/structured/plan.py#L1601). Everything below describes the data flow inside that call.

## Output: `StructuredPlan`

```python
@dataclass(frozen=True)
class StructuredPlan:
    slabs: tuple[Slab, ...]
    z_planes: tuple[float, ...]
    overlaps: tuple[OverlapPair, ...]
    arrangements: dict[int, StackArrangement]  # keyed by component_index
```

Each `Slab` carries: `footprint`, `resolved_footprint`, `zlo`/`zhi`,
`physical_name`, `source_index`, `component_index`, `face_partition`
(list of piece polygons), `face_partition_edges` (list of
`[(edge_id, fwd), ...]` per piece), `face_partition_provenance`, and
arc parameters (`identify_arcs`, `min_arc_points`, `arc_tolerance`).

## Stage 1 — gather + slice

[gather_structured_entities](meshwell/structured/plan.py#L47) filters
the input list for `structured=True` PolyPrisms with exactly one
`StructuredExtrusionResolutionSpec` in `resolutions`. Errors otherwise.

[expand_to_slabs](meshwell/structured/plan.py#L72) walks each entity's
`buffers.keys()` (sorted z-boundaries) and emits one `Slab` per
adjacent-z pair per entity. The slab's `footprint` starts as the raw
`entity.polygons` — `resolved_footprint` is None at this point.

## Stage 2 — overlap policy + footprint resolution

[validate_and_resolve_overlap](meshwell/structured/plan.py#L137)
records every volumetric overlap as an `OverlapPair`. If two
structured slabs share volume but disagree on z-extents or n_layers,
it raises `StructuredOverlapError`. All overlapping slabs are
retained — no Policy-A carving at this stage.

[_resolve_sublevel_mesh_order](meshwell/structured/plan.py#L239)
populates `slab.resolved_footprint` in place. Per z-interval, sorts
slabs by `(mesh_order, source_index)` ascending; the winner keeps its
full footprint; losers' resolved_footprints are
`footprint - union(prior_winners')`. Also subtracts `mesh_bool=False`
(void) entities. This is **Policy B**: lower mesh_order wins,
geometrically subtracted from higher.

## Stage 3 — cohort detection

[_assign_component_indices](meshwell/structured/plan.py#L286)
runs Union-Find over the slabs. Two slabs are merged into the same
component when they either share a z-plane (face-touch) OR share the
same z-interval (lateral-touch). The result writes
`slab.component_index` — this is the *cohort* — and determines which
`StackArrangement` the slab will belong to.

## Stage 4 — footprint constancy invariant

[_validate_cohort_footprint_constancy](meshwell/structured/plan.py#L339)
computes each cohort's union XY footprint per z-interval and verifies
all intervals match. If they don't, raises
`StructuredCohortFootprintMismatchError`. Reason: Phase 3 models each
cohort as one OCC envelope solid with a constant XY outline; stepped
outlines would break the envelope's assumption. Common remediation:
add low-priority "frame" or "filler" slabs to pad short z-intervals to
match the largest.

## Stage 5 — structural validators

Two checks that catch unmeshable scenes early:

- [_validate_no_mid_height_cuts](meshwell/structured/plan.py#L555):
  raises `StructuredMidHeightCutError` when an unstructured neighbour
  has a z-endpoint strictly inside a structured slab's z-range AND
  overlaps the slab's XY. The slab's lateral OCC walls can't accept
  an intermediate-z vertex (would split a quad).
- [_validate_no_unstructured_lateral_neighbour](meshwell/structured/plan.py#L620):
  wedge lateral quads can't share boundary with tet lateral
  triangles. Raises `StructuredLateralUnstructuredNeighbourError` if
  an unstructured entity shares a 1D contact with a structured slab
  boundary and z-overlaps.

## Stage 6 — arc identification (per slab, before arrangements)

When `slab.identify_arcs=True`:

[_build_arc_index_from_footprint](meshwell/structured/plan.py#L722)
decomposes each ring via `GeometryEntity.decompose_vertices` (same
algorithm as the legacy GeometryEntity arc detection). For each
detected arc, [fit_circle_2d](meshwell/structured/plan.py#L666) fits a
2D circle; if residual ≤ `arc_tolerance` (default 1e-3) and radius is
non-degenerate, a `CanonicalCircle(center, radius)` is recorded. The
index maps `(rounded_xy) → [(arc_id, position_on_arc)]`.

Parameters:
- `min_arc_points` (default 4) — minimum vertices per detected arc
- `arc_tolerance` (default 1e-3) — circle-fit residual threshold

## Stage 7 — face_partition (fixed-point iteration)

This is the heart of the planner. Each slab's `face_partition` is a
list of XY polygons representing the slab's footprint cut by
neighbour footprints projected through it.

[compute_face_partition](meshwell/structured/plan.py#L963) initializes
each slab to `face_partition = [footprint]`, then iterates:

For each slab:
1. Gather **cut sources**: z-touching neighbour `resolved_footprint`s
   (above and below) and any current iteration's neighbour pieces.
2. [_partition_pieces_for_slab](meshwell/structured/plan.py#L1044):
   - `union(cut_sources)` → cut geometry
   - `polygonize(slab_boundary + cut_boundaries)` → candidate pieces
   - Filter to pieces whose `representative_point` lies inside the
     slab's footprint
3. If the resulting pieces changed from the prior iteration,
   continue. Stop when no slab's cut sources change (typically 2-4
   iterations); fail with `StructuredFacePartitionFixedPointError` if
   `_PARTITION_FIXED_POINT_CAP=16` exceeded.

The fixed-point structure is essential for **transitive cuts**: a
neighbour at z=2 might cut a slab at z=1, which then cuts its own
neighbour at z=0 in the next iteration.

## Stage 8 — arc inheritance across z-steps

While the partition iterates, each piece's boundary is walked by
[_classify_piece_boundary](meshwell/structured/plan.py#L801) and
labeled as a sequence of `PieceArcEdge | PieceLineEdge` segments.
Inherited arcs from z-neighbours are merged into a slab's arc index
via [_merge_arc_into_index](meshwell/structured/plan.py#L1042) so that
a slab without `identify_arcs=True` still gets correct arc provenance
when its neighbour's arc cuts through it.

`face_partition_provenance` is the list of `PieceProvenance` records
emitted by this walk.

## Stage 9 — per-cohort planar arrangement

[build_stack_arrangements](meshwell/structured/plan.py#L1852) builds
one `StackArrangement` per cohort. For each component_index:

1. [_collect_stack_boundaries_tagged](meshwell/structured/plan.py#L1363):
   gather every slab's `resolved_footprint.boundary` plus every
   unstructured entity touching the cohort's z-extent. Each boundary
   is tagged `identify_arcs=True/False`.

2. [_planar_arrangement](meshwell/structured/plan.py#L1510):
   `shapely.unary_union(boundaries)` splits all curves at
   intersections, producing **arrangement edges**. Each LineString
   becomes one `ArrangementEdge(edge_id, vertices, circle=None)`.
   `shapely.polygonize` then yields arrangement faces; each face's
   boundary is walked and edges matched to arrangement edge_ids,
   producing `ArrangementFace(face_id, polygon, boundary=[(eid, fwd)])`.

3. **Arc fitting per edge**
   ([build_stack_arrangements](meshwell/structured/plan.py#L1925)):
   for each arrangement edge whose source had `identify_arcs=True`,
   call `_fit_arc_to_edge`. On success, replace `circle=None` with
   `CanonicalCircle(center, radius)`. Then
   `_coalesce_adjacent_arcs` merges adjacent arcs that share a circle
   and a vertex, and `_unify_concentric_arc_fits` collapses
   near-identical fitted circles to one canonical circle.

4. **Slab assignment**
   ([_assign_faces_to_slabs](meshwell/structured/plan.py#L2192)): each
   arrangement face is assigned to exactly one slab by polygon
   containment match. The slab's `face_partition_edges` is then
   populated with the matched faces' `[(edge_id, fwd)]` tuples.

Slabs that end up with empty `face_partition` fall back to their
`resolved_footprint` (so single-piece slabs still mesh).

## Output guarantees

After `build_plan` returns successfully:

- Every slab has `resolved_footprint`, `component_index`, and one of
  `face_partition` (non-empty if the slab has neighbours that cut it)
  or `face_partition = [resolved_footprint]`.
- `face_partition_edges` is populated per piece with arrangement edge
  ids that traverse the piece's boundary. **Not guaranteed in chain
  order** — Phase 3 must chain them via corner connectivity. See
  `_chain_piece_edges_by_corner` in
  [meshwell/structured/cohort_envelope.py](meshwell/structured/cohort_envelope.py).
- `face_partition_provenance` carries `PieceArcEdge` for arc segments
  inherited or detected — enough to reconstruct arc geometry in OCC
  later without re-fitting.
- Every cohort's union XY footprint is constant across its z-intervals.
- All structured arrangements have arc edges labeled with their
  fitted circles where applicable.

## Why this matters for Phase 3

The Phase 3 cohort envelope consumes the StructuredPlan via:

- `plan.arrangements[cohort_index].edges` → `env.horizontal_edges`
  keyed by (z, edge_id). Each arrangement edge becomes an OCC wire at
  each z-plane. Arc edges use `GC_MakeArcOfCircle(start, middle, end)`
  with the planner's mid-vertex (fixed in commit `dca2258`).
- `slab.face_partition_edges[piece_index]` → per-piece sub-face wire
  (chained via corner connectivity).
- `slab.face_partition_provenance[piece_index]` → arc-aware fallback
  for the legacy `_make_face_from_provenance` path.

The footprint constancy invariant is what lets Phase 3 model one OCC
envelope solid per cohort instead of per-z-range sub-solids — though
the multi-layer-cohort + neighbour issue documented in
[2026-05-30-phase3-followup.md](docs/superpowers/followup/2026-05-30-phase3-followup.md)
suggests this assumption may need to be revisited.
