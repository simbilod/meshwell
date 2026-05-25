# Structured planner: planar-arrangement preprocessing with canonical edges

**Date:** 2026-05-25
**Supersedes:** `docs/superpowers/specs/2026-05-25-cad-occ-structured-annular-piece-split-design.md`
**Builds on:** `docs/superpowers/specs/2026-05-21-cad-occ-structured-all-layer-intersections-design.md` (the fixed-point iteration it introduced is removed by this spec)

**Scope:** restructure `meshwell/structured/plan.py` so the planner first computes a planar arrangement of all input polygon boundaries per connected stack, fits arcs once globally on the arrangement edges, and uses those canonical edges as the single source of geometric truth from face_partition through phantom build. Eliminates the chord-vs-arc sliver bug at its root.

## Problem

The current planner gives each slab its own copy of the footprint, runs arc fitting per-slab on its own polyline, and then attempts to reconcile cross-slab geometry via a fixed-point iteration that merges inherited arcs into each slab's arc index. This works for purely straight-edge cases but breaks for arc-bearing stacked slabs in two distinct ways:

1. **Polyline-vs-true-arc deviation.** OCC interprets `identify_arcs=True` as a true circular arc, but the planner's `compute_face_partition` polygonizes against polyline chord approximations. After BOP fragmentation, the thin region between the polyline and the analytic arc becomes a sliver sub-face that can't be transfinite-meshed; gmsh falls back to tets there, producing non-conformal wedge/tet interfaces and orphan boundary triangles. Surfaced by `tests/structured/test_stress_stacked_patterns.py::test_stacked_overlapping_ring_segments_mesh_clean`.

2. **Annular face_partition pieces.** Transitive arc cuts (e.g., L2's R=0.7 disc cutting L1's R=1 disc top face) produce face_partition pieces with interior holes. The structured mesher's transfinite logic rejects multi-loop boundary topology. Surfaced by `test_stacked_concentric_arc_discs_mesh_clean`.

Both are symptoms of the same architectural issue: the planner doesn't have a single canonical representation of shared geometry. Each slab independently approximates arcs, and the patchwork inheritance machinery can't guarantee that two slabs sharing an arc agree on exactly which polyline samples lie on it.

## Goal

Replace `compute_face_partition` with a planar-arrangement preprocessing phase that builds canonical edges (arcs and lines) once per connected slab stack. Every slab references these canonical edges. Downstream consumers (phantom builder, mesh builder) use canonical geometry directly. The planner becomes a pure-geometric reasoner with no iteration, no inheritance, no merging.

Concretely:

- Both xfails (`test_stacked_concentric_arc_discs_mesh_clean`, `test_stacked_overlapping_ring_segments_mesh_clean`) flip to passing.
- The fixed-point iteration machinery (`_collect_cut_sources`, `_collect_inherited_arcs`, `_merge_arc_into_index`, `_partition_pieces_for_slab`, `_classify_piece_boundary`, the `_LAST_PARTITION_ITERATIONS` counter, the cap, the convergence error) is removed.
- `_validate_arc_neighbour_alignment` is removed; its concern is eliminated by construction.
- Existing non-arc tests stay green with no assertion changes.
- Existing arc tests stay green; assertion content may need to inspect `ArrangementEdge.circle` instead of `PieceArcEdge`.

Out of scope:
- Changes to `phantom.py` beyond consuming the new `ArrangementEdge` type when building OCC wires.
- Changes to `builder.py`. The mesh stage stays the same.
- Multi-component coupling beyond face-touching. Stacks that don't share a z-face stay independent (their arrangements are disjoint).

## Approach

### Data model — new types in `meshwell/structured/spec.py`

```python
@dataclass(frozen=True)
class CanonicalCircle:
    """Identity of a circular curve shared across arrangement edges.

    Two arrangement edges with CanonicalCircle instances matching on
    (center, radius) within arc_tolerance are guaranteed to be sub-arcs
    of the same physical circle. The phantom builder uses (center, radius)
    plus arc endpoints to construct the OCC arc geometry; all consumers
    of the same circle produce bit-identical TShapes.
    """
    center: tuple[float, float]   # XY only; z is implicit in the slab
    radius: float


@dataclass(frozen=True)
class ArrangementEdge:
    """One non-crossing curve segment in the planar arrangement.

    vertices: ordered XY sample points. >=2 elements. Endpoints are
        vertices[0] and vertices[-1]; interior samples populate the
        polyline approximation used wherever shapely operations are
        still needed (no OCC consumer uses interior samples — they
        reconstruct from circle + endpoints).
    circle: None means the edge is a straight line. Not None means
        the edge is a sub-arc of the named circle; vertices[0] and
        vertices[-1] lie on it.
    """
    edge_id: int
    vertices: tuple[tuple[float, float], ...]
    circle: "CanonicalCircle | None"


@dataclass
class ArrangementFace:
    """One face of the planar arrangement (a polygon with no interior holes)."""
    face_id: int
    polygon: "Polygon"                            # geometric copy for containment
    boundary: list[tuple[int, bool]]              # ordered (edge_id, reversed)


@dataclass
class StackArrangement:
    """Per-component planar arrangement; consumed by the new face-partition step."""
    edges: list[ArrangementEdge]
    faces: list[ArrangementFace]
```

The arrangement collection is `dict[int, StackArrangement]` keyed by an opaque stack id (the connected-component index from Union-Find in step A). Slabs carry no explicit stack_id — the orchestrator passes `(slab, arrangement)` pairs at consumption time, looking up the right arrangement via the slab-to-component map built in step A.

The `Slab` dataclass gains one optional field:

```python
@dataclass
class Slab:
    # ... existing fields ...
    face_partition: list[Polygon] = field(default_factory=list)
    face_partition_edges: list[list[tuple[int, bool]]] | None = None
    # face_partition_edges[i] is the ordered boundary of face_partition[i],
    # expressed as (edge_id, reversed) tuples into the slab's stack
    # arrangement. The phantom builder uses this instead of the legacy
    # face_partition_provenance for arc-aware OCC wire construction.
```

The legacy `face_partition_provenance: list[PieceProvenance] | None` field stays for one release as a derived shim (computed from `face_partition_edges` + arrangement lookup), so phantom.py code that hasn't migrated yet still works. Both fields converge to the same OCC geometry. The shim is removed once phantom.py is fully on the new path.

### Pipeline order in `meshwell/structured/plan.py`

Steps 1–5 (`gather_structured_entities`, `expand_to_slabs`, `validate_and_resolve_overlap`, `_validate_no_mid_height_cuts`, `_validate_no_unstructured_lateral_neighbour`) stay exactly as today.

Step 6 is replaced. Today:

```python
compute_face_partition(kept_slabs, entities)
```

Becomes:

```python
arrangements = build_stack_arrangements(kept_slabs, entities)
assign_face_partition_from_arrangement(kept_slabs, arrangements)
```

`compute_face_partition` is deleted along with its helpers. The two new functions live in `plan.py` next to `expand_to_slabs`.

### Algorithm

**Step A — connected components by z-touch.** Build a graph where nodes are slabs and an edge connects `a, b` iff `abs(a.zhi - b.zlo) < _Z_TOL or abs(a.zlo - b.zhi) < _Z_TOL`. Run Union-Find or DFS to extract connected components. Each component is one stack. A slab with no z-neighbours is a singleton component (its arrangement is just its own footprint, one face, no internal cuts).

**Step B — collect boundaries per stack.** For each component, the boundary set is:

- Every member slab's `footprint.boundary` (the polygon ring with point_tolerance-snapped vertices that PolyPrism already stores).
- Every unstructured entity whose z-range touches any of the component's z-planes. Its `footprint.boundary` is added; unstructured entities never have `identify_arcs=True`, so their contributions are line-only.

The dedup happens implicitly via `unary_union` in step C.

**Step C — planar arrangement.** Compute `merged = unary_union(boundaries)` — a `MultiLineString` with shapely-injected intersection vertices wherever boundaries cross. Then `arrangement_polygons = list(polygonize(merged))` — these are the arrangement faces.

To extract `ArrangementEdge` objects, traverse `merged` and identify *arrangement nodes* (points with three or more incident curves — i.e., crossings or T-junctions). The maximal vertex runs between adjacent arrangement nodes are the edges. Walk each polyline in `merged`, splitting at every arrangement node, giving one `ArrangementEdge` per run.

**Step D — arc fit per edge.** For each `ArrangementEdge`, attempt to fit a circle to its vertices using `GeometryEntity.decompose_vertices` (the same heuristic used today by `_build_arc_index_from_footprint`). Conditions for accepting the fit:

- At least one source entity that contributed this edge has `identify_arcs=True`.
- The fit residual is below `min(source.arc_tolerance for sources)`.
- The fit produces `>= source.min_arc_points` consecutive vertices on the same circle.

If accepted, set `edge.circle = CanonicalCircle(fitted_center_xy, fitted_radius)`. Else `edge.circle = None` (line).

**Step E — coalesce adjacent arcs on the same circle.** Two `ArrangementEdge`s with non-`None` circles that share an endpoint and whose `(center, radius)` match within `arc_tolerance` are merged into one `ArrangementEdge`. The merged vertices are the union of inputs (deduplicated by coordinate match within `point_tolerance`), in boundary-traversal order. The merged `circle` is the consensus — use the source entity's authoritative `arc_index` value if available, else the residual-weighted average of the two fits.

This is the *canonical-circle dedup* step. It guarantees that every consumer of a given physical arc gets the same `(center, radius)` and the same sample point set. The downstream OCC `GC_MakeArcOfCircle` produces bit-identical TShapes from this canonical data, so BOP shares them.

Coalesce iterates until no two edges can merge — typically 1–2 sweeps for any realistic scene.

**Step F — assign faces to slabs.** For each `ArrangementFace`, compute `face.polygon.representative_point()` once. For each slab, append the face to `slab.face_partition` iff `slab.footprint.contains(rep_point)`. The face's boundary list of `(edge_id, reversed)` tuples is copied to `slab.face_partition_edges[i]` for the same `i`.

**Step G — provenance shim.** For backward compatibility with `phantom.py`'s current `PieceProvenance`-based path, derive `slab.face_partition_provenance` from `face_partition_edges` + the arrangement:

- For each piece's boundary, walk the `(edge_id, reversed)` list.
- For each edge with `circle is not None`, construct a `PieceArcEdge` from `(edge.vertices, circle.center + (0.0,), circle.radius)`.
- For each edge with `circle is None`, construct a `PieceLineEdge` from `(edge.vertices[0] + (0,), edge.vertices[-1] + (0,))`.

This shim runs once after step F and lets phantom.py work unchanged. After phantom.py migrates to consume `face_partition_edges` directly, the shim and the `face_partition_provenance` field can be removed.

### Edge cases

**Unstructured neighbours with non-arc straight boundaries.** Contribute to the arrangement as line-only segments. No special handling.

**Slabs with `identify_arcs=False`.** Their footprint boundaries contribute to the arrangement, but the arc-fit acceptance criterion requires at least one source entity with `identify_arcs=True`. So a slab whose only neighbours are non-arc will see only line edges, even if its boundary geometrically resembles an arc. This matches existing behavior.

**Singleton components.** A slab with no z-neighbours: its arrangement has one face = its footprint, edges = its boundary segments after arc fitting. The same machinery still runs; the slab is just a stack of size 1.

**Multipolygons.** A `MultiPolygon` footprint contributes each constituent ring to the boundary set. The arrangement handles them as separate boundaries; assignment-to-slab still works via containment of each face's `representative_point`.

**Numerical noise on shared arcs.** If two entities define arcs with slightly different `(center, radius)` (e.g., R=0.9999 vs R=1.0001 due to construction noise), step D fits each separately. Step E's coalesce merges them within `arc_tolerance`. If the difference exceeds `arc_tolerance`, they stay separate — likely the correct behavior, since the user intended two different curves. A warning can be logged here for diagnostics.

### Validators removed

`_validate_arc_neighbour_alignment` and its supporting code (`_interior_buffer_for_radius`, the post-convergence loop in the old `compute_face_partition`) are removed. The validator's purpose was to catch chord-vs-arc divergence between independently-built slab arc data. With canonical edges, divergence is impossible: every slab uses the same `ArrangementEdge`s, and OCC builds from `(circle.center, circle.radius)` directly. There is nothing to validate.

`_validate_no_mid_height_cuts` and `_validate_no_unstructured_lateral_neighbour` stay unchanged — orthogonal concerns about z-topology that the arrangement preprocessing doesn't address.

### Convergence machinery removed

`_PARTITION_FIXED_POINT_CAP`, `_LAST_PARTITION_ITERATIONS`, `StructuredPartitionConvergenceError`, and the related helpers (`_structured_slabs_touching_z`, `_merge_arc_into_index`, `_collect_cut_sources`, `_collect_inherited_arcs`, `_partition_pieces_for_slab`, `_attach_face_partition_provenance`, `_classify_piece_boundary`, `_build_arc_index_from_footprint`, `_ArcIndex`, `_IndexedArc`) are removed. They served the iteration model that's being replaced.

The convergence-bound test (`test_partition_converges_within_K_plus_two_passes`) and the cap-raise test (`test_partition_raises_if_not_converged`) are removed alongside.

## Code changes

### `meshwell/structured/spec.py`

- Add `CanonicalCircle`, `ArrangementEdge`, `ArrangementFace`, `StackArrangement` dataclasses.
- Add `face_partition_edges` field to `Slab` (default `None`).
- Keep `PieceArcEdge`, `PieceLineEdge`, `PieceProvenance` for now (used by the shim).
- Remove `StructuredPartitionConvergenceError`.

### `meshwell/structured/__init__.py`

- Export the new types (`CanonicalCircle`, `ArrangementEdge`, `ArrangementFace`, `StackArrangement`).
- Remove `StructuredPartitionConvergenceError` from imports and `__all__`. This was added to public exports on 2026-05-21 in commit `5b6e640` on the current feature branch; it has not shipped in any release and no downstream consumer should depend on it. Safe to remove without a deprecation cycle.

### `meshwell/structured/plan.py`

- Add `build_stack_arrangements(slabs, entities) -> dict[stack_id, StackArrangement]`.
- Add `assign_face_partition_from_arrangement(slabs, arrangements)` — sets `face_partition`, `face_partition_edges`, and (via shim) `face_partition_provenance`.
- Add private helpers:
  - `_connected_z_components(slabs, tol) -> list[list[Slab]]`
  - `_collect_stack_boundaries(stack, entities) -> list[LineString]`
  - `_extract_arrangement_edges(merged_linework, source_entity_metadata) -> list[ArrangementEdge]`
  - `_fit_arc_to_edge(edge_vertices, source_entities) -> CanonicalCircle | None`
  - `_coalesce_adjacent_arcs(edges) -> list[ArrangementEdge]`
  - `_assign_face_to_slabs(face, slabs) -> list[Slab]` (containment test)
  - `_build_provenance_shim(face_partition_edges, arrangement) -> list[PieceProvenance]`
- Delete the entire `compute_face_partition` function and its helpers listed in the "Convergence machinery removed" section.
- Update `build_plan` to call `build_stack_arrangements` + `assign_face_partition_from_arrangement` instead of `compute_face_partition`.

### `meshwell/structured/phantom.py`

No changes for this spec — the provenance shim keeps the existing OCC wire construction path working. A follow-up spec can migrate phantom.py to consume `face_partition_edges` directly, then drop the shim and the legacy `PieceProvenance` types.

### `meshwell/structured/builder.py`

No changes. Mesh-stage logic operates on the OCC output produced by phantom.py, which is bit-identical to today's for non-arc cases and finally correct for arc cases (no more slivers).

## Tests

### Primary regression flips

`tests/structured/test_stress_stacked_patterns.py::test_stacked_concentric_arc_discs_mesh_clean`:
- Remove `@pytest.mark.xfail(...)`. Should pass once canonical edges are in place.

`tests/structured/test_stress_stacked_patterns.py::test_stacked_overlapping_ring_segments_mesh_clean`:
- Remove `@pytest.mark.xfail(...)`. Same.

### New plan-only tests in `tests/structured/test_plan_arrangement.py` (new file)

- `test_build_arrangement_single_disc_no_neighbours`: one structured disc, no neighbours. Assert: one face, one edge with `circle != None`.

- `test_build_arrangement_two_overlapping_rings_shares_arc`: two arc-bearing slabs whose outer arcs lie on the same R=1 circle. Assert: the coalesce step merged them — there's exactly one `ArrangementEdge` covering the shared sub-arc, both faces reference it.

- `test_build_arrangement_z_components_isolated`: two stacks at z=[0,1] and z=[10,11] (no face-touching). Assert: two separate `StackArrangement` objects, each containing only its own slab's edges.

- `test_build_arrangement_z_component_transitive`: four face-touching slabs at z=[0..1, 1..2, 2..3, 3..4]. Assert: one `StackArrangement` containing all four slabs; arrangement edges from all four are present.

- `test_assign_face_partition_concentric_discs`: the existing failing scene (decreasing-radius discs). Assert: every `face_partition` piece has zero interior rings; every annular region appears as multiple disk-topology arrangement faces, not as one polygon with a hole.

- `test_assign_face_partition_misaligned_seams`: the 4-layer straight-edge misaligned stack. Assert: each slab's piece count matches today's behavior (regression guard).

### Comprehensive stress test (new, single test in `test_stress_stacked_patterns.py`)

`test_complex_disjoint_arc_stacks_with_keep_mix_mesh_clean(tmp_path)`. Combines every dimension the planar-arrangement architecture is designed to handle, in one scene:

**Two disjoint stacks** (no z-touching between them; verifies the connected-component restriction):

- **Stack A** at z ∈ [0, 3], three sub-levels: [0,1], [1,2], [2,3].
- **Stack B** at z ∈ [10, 12], two sub-levels: [10,11], [11,12].

**Per sub-level, multiple structured polyprisms with mixed keep/mesh_bool and arc-bearing geometry:**

- Stack A, level [0,1]: a half-annulus (R=0.5–1.0, θ=0..π, `identify_arcs=True`, `physical_name="A1_main"`, `mesh_order=1`) overlapping at the same z with a smaller disc (`R=0.3`, center=(0.5, 0.4), `identify_arcs=True`, `physical_name="A1_inset"`, `mesh_order=2`). Mesh-order resolution at the sub-level means the disc carves out a hole in the half-annulus where they overlap. Plus a third entity (a small square at (-0.3, -0.3) to (-0.1, -0.1), `mesh_bool=False`, `physical_name="A1_void_tag"`) — a non-meshed carving marker that should not appear as a wedge volume.
- Stack A, level [1,2]: rotated by π/2 — half-annulus (θ=π/2..3π/2, same R bounds), inset disc at a *different* center (0.4, 0.5), and a different `mesh_bool=False` void tag at (0.1, -0.3). Vertex angular grid aligned with level [0,1]'s but with different angular range so polylines on the shared R=1 / R=0.5 sub-arcs coincide where they overlap.
- Stack A, level [2,3]: rotated by another π/2 — half-annulus (θ=π..2π), no inset, just the half-annulus and a `mesh_bool=False` void tag.

- Stack B, level [10,11]: a disc (R=0.7, center=(0,0), `identify_arcs=True`, `physical_name="B1_disc"`) plus a square cladding (`R^2 < ...` removed) around it via a larger structured square with the disc carved out via mesh_order. Actually simpler: just a disc (R=0.7) and a smaller concentric disc (R=0.3) inside it, mesh_order assigning the inner disc as winner. Use `mesh_bool=True` on both — different physicals.
- Stack B, level [11,12]: an annulus-segment (R=0.3–0.7, θ=π/4..7π/4) with a 90° gap, plus a thin straight bar that crosses through the gap (`mesh_bool=False` tag).

**Sub-level-to-sub-level non-alignment within each stack:**

- The arc seams at z=1 (A1 half-annulus θ=π radial vs A2 half-annulus θ=π/2, 3π/2 radials) introduce transitive cuts that must propagate via the arrangement.
- The disc inset's outer boundary at level [0,1] doesn't coincide with the disc inset at level [1,2] (different centers), so the disc-circle in each level is a separate `CanonicalCircle` — verifying the coalesce step doesn't over-merge.

**Assertions:**

- `build_plan([all entities])` succeeds.
- Two `StackArrangement`s are produced (one per disjoint stack; verified by walking the slab-to-component map exposed for testing).
- All `mesh_bool=False` physicals contribute their boundaries to the relevant arrangements (cuts apply) but do not appear as 3D physicals in the mesh (only as 2D boundary tags if `mesh_bool=False` semantics include surface tagging).
- The arrangement of stack A contains at least one `CanonicalCircle` with `radius≈1.0` (the half-annuli outer arcs, shared across all 3 levels via coalesce).
- The arrangement of stack A contains at least two `CanonicalCircle` entries with `radius≈0.3` (the level [0,1] and level [1,2] insets — *not* coalesced because their centers differ by more than `arc_tolerance`).
- `generate_mesh` produces a mesh with wedge cells and zero tetrahedra in the structured regions (the chord-vs-arc sliver bug is gone).
- Zero orphan boundary triangles (the count function from the existing stress tests).
- All meshable physicals (`mesh_bool=True`) appear in `field_data`.

This test is the spec's load-bearing regression target. If it passes end-to-end, the planar-arrangement architecture has demonstrated correctness on a scene that combines every failure mode the existing iteration model couldn't handle.

### Mesh-order sub-level pre-processing (optional optimization, see "Out of scope")

The user's note on "pre-processing to the sublevel to perform the boolean cuts according to mesh order to simplify first" is captured as a follow-up optimization. The arrangement approach already handles overlapping same-z polyprisms correctly (their boundaries union into the arrangement, and assignment-to-slab via mesh_order at containment-test time produces the right partition). A pre-processing pass could simplify the input by resolving same-z mesh_order conflicts *before* building the arrangement — but the same end-state is achievable post-arrangement by attaching `mesh_order` to faces during step F. Deferred unless profiling shows it's needed.

### Existing-test audit

Run the full `tests/structured/` suite. Tests that inspected `face_partition_provenance` internals (e.g., `test_arc_provenance_propagates_to_neighbour_below`, `test_no_arc_inheritance_when_neighbour_identify_arcs_false`) may need updating to inspect `face_partition_edges` or the shim output. The semantic guarantees they check (arc inheritance, identify_arcs propagation) are preserved by the new model.

The convergence tests (`test_partition_fixed_point_cap_is_module_constant`, `test_partition_converges_within_K_plus_two_passes`, `test_partition_raises_if_not_converged`) are deleted — they test machinery that no longer exists.

## Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| Polygonize step E (coalesce) merges arcs that shouldn't be merged (e.g., two truly distinct circles within `arc_tolerance`) | The arc_tolerance is set explicitly per entity; if two entities have intentionally close circles, the caller is expected to tighten arc_tolerance accordingly. Log a warning when coalesce merges arcs across source entities with different arc_tolerance values. |
| Step D's per-edge arc fit produces inconsistent fits across edges of the same arc due to vertex sampling | Step E's coalesce uses `(center, radius)` consensus from source entity arc_index when available — avoiding the per-edge fit being authoritative. |
| Backward compatibility breaks for phantom.py path | The provenance shim keeps phantom.py running unchanged. Once phantom.py migrates (separate follow-up spec), the shim is removed cleanly. |
| Multi-source arrangement edges have inconsistent `identify_arcs` (e.g., L1 says yes, L2 says no on a shared boundary) | The acceptance rule "at least one source has `identify_arcs=True`" treats this as an arc. The non-arc source will see line classification in `face_partition_edges`, but the OCC arc still gets built — and since the non-arc source's polyline already lies on the canonical circle (by construction), OCC sees the same TShape as the arc source. No conflict. |
| Performance: `unary_union` of N polylines is O(N²) in worst case | Connected-component restriction bounds N per arrangement to slabs in one stack — typically <20 in realistic scenes. Sub-second for any realistic mesh. Revisit only if production scenes exceed this. |
| Step C edge extraction (finding maximal vertex runs between arrangement nodes) is non-trivial in shapely's API | Use shapely's `line_merge` on the unary_union output, then manually split at points with ≥3 incident polylines. Document the extraction algorithm clearly; add unit tests on synthetic linework. |

## Out of scope (future specs)

- **Phantom builder migration to consume `face_partition_edges` directly**, dropping the `PieceProvenance` shim and the legacy `PieceArcEdge` / `PieceLineEdge` types. Separate spec; not blocking on this one.
- **Arc detection on `unary_union` output across slabs** — i.e., recognizing that two slabs' polylines approximate the same arc *even when their vertex sets differ*. Today's `coalesce` step matches arcs only when their fit results agree on `(center, radius)`. If two slabs have very different sampling (e.g., n=8 vs n=32 polylines for the same circle), the fits might still agree, but the merged vertex set will be the union — twice as dense, possibly affecting mesh element count. Acceptable for now.
- **Same-z-interval lateral cut propagation** (the deferred scope (b) from the 2026-05-21 brainstorm). The arrangement approach naturally extends to this: same-z-interval slabs in the same connected component already share an arrangement, so cuts at the lateral interface are visible to both. May resolve as a side-effect; verify when implementing.
