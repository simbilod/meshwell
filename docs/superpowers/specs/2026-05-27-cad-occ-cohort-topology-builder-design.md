# Phase 2 — Cohort topology builder (full vertical + lateral face sharing)

**Status:** design
**Date:** 2026-05-27
**Owner:** simbilod
**Builds on:** `2026-05-27-cad-occ-cohort-preshared-faces-design.md` (Phase 1, vertical-only sharing via `prism.LastShape()` reuse)

## Problem

Phase 1 pre-shares `TopoDS_Face` between vertically-stacked sub-prisms in the same cohort by chaining `BRepPrimAPI_MakePrism(prev.LastShape(), ...)`. That covers the vertical-interface case but leaves cohort-internal *lateral* interfaces to be unified by BOPAlgo at fragment time. Production scenes that mix vertically-stacked AND laterally-adjacent structured slabs still pay the lateral pairwise BOP cost.

We investigated three lateral-sharing mechanisms:
- (A) Sewing post-construction: rejected — Phase 1 already showed sewing+compound loses `Modified()` history. Sewing followed by individual-argument BOP is unproven and would need its own validation.
- (B.1) `BRepTools_ReShape` surgical face replacement: rejected by the user — too narrow, doesn't restructure the construction.
- **(B.2) Full manual assembly via `BRep_Builder`**: chosen. Construct cohort topology once (vertices, edges, faces) and assemble each sub-prism's solid as a view into shared topology.

## Goal

Eliminate pairwise BOP work between sub-prisms of the same cohort by sharing both vertical AND lateral interface faces at construction time. After Phase 2, every cohort-internal interface (whether top/bottom or lateral) has shared `TShape` identity between its two adjacent sub-prisms.

**Success criterion:** on a representative production scene with mixed vertical-stacking + lateral-adjacency cohorts, `_fragment_all` wall time drops measurably (target: ≥3× over baseline; should subsume Phase 1's vertical-only gains).

## Non-goals (this phase)

- Removing `_build_sub_prism` or `_PRESHARE_VERTICAL_FACES`. Phase 2 lands the new path BEHIND a kill-switch (`_USE_COHORT_TOPOLOGY`). The legacy path stays as a fallback and for parity testing. A separate cleanup spec, written after Phase 2 is validated in production, retires the legacy code.
- Changes to cad_occ, the gmsh backend, `OCCLabeledEntity`, or anything downstream of `build_phantom_shapes`.
- Optimization of cohort topology construction itself (e.g., parallelizing across cohorts). First-cut serial.

## Architecture

Per cohort (connected z-component), build a shared topology in a single pass, then assemble each sub-prism's solid as a view into it.

```
For each cohort:
  topology = build_cohort_topology(plan, component_index)
  For each (slab, piece_index) in cohort:
    phantom_shape = assemble_cohort_sub_prism(topology, slab, piece_index)
    emit phantom_shape
```

`build_cohort_topology` returns a `CohortTopology` carrying four registries:

- **Vertices** — keyed by `(z_plane, xy_corner_id)`. One `TopoDS_Vertex` per unique corner of any cohort piece at each z-plane.
- **Horizontal edges** — keyed by `(z_plane, arrangement_edge_id)`. One per cohort arrangement edge at each z-plane, reusing the registry vertices as endpoints. Carries arc/line classification from the arrangement.
- **Vertical edges** — keyed by `(z_interval_id, xy_corner_id)`. One per corner of any cohort piece per slab z-interval; connects the bottom-z vertex to the top-z vertex.
- **Faces** — two kinds:
  - **Horizontal** keyed by `(z_plane, piece_id)`. One per cohort piece at each z-plane it appears in. Serves simultaneously as the top of slab N below and the bottom of slab N+1 above.
  - **Lateral** keyed by `(z_interval_id, arrangement_edge_id)`. One per arrangement edge per slab. Edges bordering two cohort pieces produce a shared lateral; edges on the cohort boundary produce a lateral used by only one piece (still registered for assembly consistency).

`assemble_cohort_sub_prism` pulls the relevant horizontal-face, lateral-face, and edge/vertex entries from the registries, assembles a `TopoDS_Shell` via `BRep_Builder.Add(shell, face)` for each face (with `face.Reversed()` where the prism's outward orientation requires it), and wraps in `TopoDS_Solid` via `BRep_Builder.MakeSolid(shell)`. It then populates the `PhantomShape.input_*_by_key` dicts from the same registry entries so per-slab key lookups yield the shared TShapes.

## Module layout

New file: `meshwell/structured/cohort_topology.py`. Contains:

- `CohortTopology` dataclass — the four registries plus a back-reference to `plan` and `component_index`.
- `build_cohort_topology(plan: StructuredPlan, component_index: int) -> CohortTopology` — pure function.
- `assemble_cohort_sub_prism(topology: CohortTopology, slab: Slab, piece_index: int) -> PhantomShape` — pure function.

`meshwell/structured/phantom.py` change:

- Add `_USE_COHORT_TOPOLOGY: bool = True` module-level constant near the existing `_PRESHARE_VERTICAL_FACES`.
- In `build_phantom_shapes`, branch at the top:
  - `_USE_COHORT_TOPOLOGY=True` (default after Phase 2 lands): group slabs by cohort, call `build_cohort_topology` once per cohort, then `assemble_cohort_sub_prism` per sub-prism.
  - `_USE_COHORT_TOPOLOGY=False`: existing path (Phase 1 vertical chaining if `_PRESHARE_VERTICAL_FACES=True`, otherwise full legacy).

Both paths populate the same `PhantomBuildResult.shapes` tuple in `(slab_index, piece_index)` ascending order.

## Detailed design

### Cohort-interior vs exterior edge detection

For each arrangement edge in the cohort's `StackArrangement.edges`:
- Walk the cohort's pieces; for each piece, check `face_partition_edges[piece_index]` (each entry is `(edge_id, reversed)`) to see whether the edge is referenced.
- **Cohort-interior** = referenced by ≥2 pieces in the cohort. The lateral face produced by extruding this edge is shared.
- **Cohort-exterior** = referenced by exactly 1 piece. The lateral face is built once and used only by that piece (still goes through the topology registry for assembly consistency).

### Edge orientation in piece outer wires

`face_partition_edges` records each piece's boundary as `(edge_id, reversed)` pairs. Two cohort-interior pieces sharing an edge will record opposite `reversed` values for it. The lateral face has one canonical orientation when registered (matching `reversed=False`). At assembly time, each sub-prism adds the lateral face to its shell with `face.Reversed()` if its piece records `reversed=True` for that edge.

### Arc edges and provenance

For each cohort piece, `face_partition_provenance[piece_index]` (when present) classifies each outer edge as arc or line. The cohort topology builder must distill provenance into per-arrangement-edge classification: each arrangement edge inherits the arc/line classification from any piece that references it (one classification per edge — pieces that share an edge must classify it the same way; this is already a planner invariant).

- **Straight edge:** lateral face is a flat quadrilateral. Build via 4 vertices → 4 edges (2 horizontal + 2 vertical) → outer wire → `BRepBuilderAPI_MakeFace(wire)`.
- **Arc edge:** lateral face is a cylindrical strip. Build via 2 horizontal arc edges (one at zlo, one at zhi sharing the same axis and angle) + 2 vertical straight edges → wire → `BRepBuilderAPI_MakeFace(cylindrical_surface, wire)`. The cylindrical surface is constructed once from the arc's center axis + radius, then both arc edges are placed on it.

### Horizontal face construction

For each `(z_plane, piece_id)`, build the horizontal face from the piece's outer-wire arrangement edges (using the registered horizontal edges at that z_plane). For pieces with interior rings (holes), build the interior wires similarly. Use `BRepBuilderAPI_MakeFace(outer_wire, holes=interior_wires)`.

The horizontal face's outer-wire edges are exactly the registered horizontal edges (same TShapes); its vertices match the registered vertices. So when a lateral face is assembled with its bottom edge being one of these registered horizontal edges, the topology is intrinsically consistent.

### Solid assembly per sub-prism

For `(slab, piece_index)`:
1. Look up `bottom_face = horizontal_faces[(slab.zlo, piece_id)]`.
2. Look up `top_face = horizontal_faces[(slab.zhi, piece_id)]`.
3. For each outer arrangement edge of the piece: look up `lateral = lateral_faces[(slab_id, edge_id)]`. Apply `lateral.Reversed()` if the piece records `reversed=True` for the edge.
4. Build `TopoDS_Shell` via `BRep_Builder.MakeShell` + `BRep_Builder.Add(shell, face)` for bottom (reversed because of outward orientation), top, and each lateral.
5. Build `TopoDS_Solid` via `BRep_Builder.MakeSolid` + `BRep_Builder.Add(solid, shell)`.
6. Populate `PhantomShape.input_*_by_key`:
   - `input_faces_by_key[FaceKey(slab_index, "bot", piece_index)]` = bottom_face
   - `input_faces_by_key[FaceKey(slab_index, "top", piece_index)]` = top_face
   - `input_edges_by_key[EdgeKey(slab_index, "bot", piece_index, edge_i)]` = registered horizontal edge for that piece edge at zlo
   - Same for "top" at zhi
   - `input_vertices_by_key` populated from registered vertices
   - `input_laterals_by_outer_edge[outer_edge_index]` = the (possibly reversed) lateral face

### `piece_id` definition

The registries are scoped to a single cohort (one `build_cohort_topology` call per component), so `piece_id` only needs to disambiguate pieces *within* that cohort. Definition: `piece_id = (source_index_of_originating_entity, piece_index_within_slab)`.

This ensures distinct cohort pieces at the same XY footprint that come from different entities (e.g., A and B as adjacent rectangles) get distinct keys. Vertical neighbors that ARE the same piece across slabs (same entity, same piece geometry, contiguous z) get the same `piece_id` because `source_index` and `piece_index` match, so their horizontal faces are unified by the registry.

### Validation gates

**Smoke tests (new file `tests/test_brep_builder_assembly_smoke.py`):**

1. `test_brep_builder_makesolid_from_explicit_faces_bopalgo_modified_works` — pure OCC test. Assemble two solids via `BRep_Builder.MakeSolid` from 6 explicit faces each, sharing one face TShape between them. Add both + an overlapping third box to `BOPAlgo_Builder`. Assert `Modified(solid_A)` and `Modified(solid_B)` both return non-empty per-argument history. This is the load-bearing assumption — if it fails, the design must change.

2. `test_brep_builder_lateral_arc_construction` — pure OCC test. Build a cylindrical lateral face from an arc edge + z-range. Assert the face is valid (`BRepCheck_Analyzer`) and its bottom/top edges are the supplied arc edges (shared TShape).

3. `test_cohort_topology_shared_lateral_face` — meshwell-level. For a small cohort with two pieces sharing an edge, build the topology and assert both pieces' lateral-face registry entries point to the same `TopoDS_Face` TShape.

4. `test_cohort_topology_shared_horizontal_face_vertical_neighbors` — meshwell-level. For a cohort with stacked slabs, assert the horizontal face at the shared z-plane is the same `TopoDS_Face` TShape for the slab below (top) and the slab above (bottom).

**Integration test (`tests/structured/test_cohort_topology_integration.py`):**

End-to-end: build a small mixed scene (3 vertically-stacked + 2 laterally-adjacent + 1 unstructured neighbor). Run `build_phantom_shapes` with `_USE_COHORT_TOPOLOGY=True` and verify:
- All vertically-adjacent cohort pieces share their interface face.
- All laterally-adjacent cohort pieces share their interface lateral.
- The unstructured neighbor's solid has no shared TShapes with cohort pieces.

**Parity test (`tests/test_cohort_topology_parity.py`):**

Run the full pipeline twice on a representative scene:
- First with `_USE_COHORT_TOPOLOGY=False, _PRESHARE_VERTICAL_FACES=False` (full legacy).
- Then with `_USE_COHORT_TOPOLOGY=True` (Phase 2 path).

Assert identical entity-piece-count signature, identical physical names, identical interface-tag groups. The downstream pipeline must be invariant under the optimization.

### Correctness invariants

**Preserved:**
- `PhantomShape` shape and key invariants — same `input_*_by_key` dicts populated.
- `PhantomBuildResult.shapes` ordering: `(slab_index, piece_index)` ascending.
- Downstream consumers iterate `input_*_by_key` per key and call `builder.Modified()` per key — safe under shared TShapes because Modified() returns the correct successor list regardless of how many keys point to the same input.
- `_fragment_all` per-argument `Modified()` history — sub-prisms are still individual BOP arguments.

**New invariants introduced:**
- The cohort topology registries are immutable after `build_cohort_topology` returns; `assemble_cohort_sub_prism` only reads from them.
- Every cohort-internal interface (vertical OR lateral) has shared TShape identity between its two adjacent sub-prisms.

**Risks and mitigations:**
- Manual shell construction can produce invalid topology (non-closed shells, mis-oriented faces). Mitigation: smoke test 1 + 2 catch the basic cases; integration test catches scene-level issues. Also, run `BRepCheck_Analyzer` on every assembled solid in debug builds (gated by an env var).
- Arc lateral construction is the trickiest part. Mitigation: smoke test 2 explicitly validates it; integration and parity tests cover real arc-using cohort scenes.
- Cohort-exterior lateral faces are built once per piece-edge. If a piece has an interior ring (hole), its outer-wire and inner-wire edges all need lateral faces. The design covers this; verify via a test fixture with a holed piece.
- `piece_id` must correctly identify vertically-stacked-same-piece across slabs. Verified in test 4.

## Deferred — Phase 3 cleanup

After Phase 2 is in production and validated:
- Delete `_build_sub_prism`, `_PRESHARE_VERTICAL_FACES`, `_USE_COHORT_TOPOLOGY`.
- Delete `_group_slabs_into_vertical_stacks` (subsumed by cohort topology).
- Remove the legacy branch in `build_phantom_shapes`.
- Update tests that exercise the old path or the toggle.

Phase 3 is a separate spec; it does no algorithmic change, only retires obsolete code.

## Out of scope

- Parallelizing cohort topology construction.
- Sharing across cohorts (cohorts are by definition disjoint — pieces in different cohorts have no shared TShapes).
- Changes to non-structured entity handling.
- Backporting cohort metadata to non-structured paths.

## Measured Results (Task 13)

Scene: 4 lateral stacks × 10 vertical layers each = 40 structured PolyPrism entities, 4 disjoint cohorts.

`build_phantom_shapes` wall time (min of 3 runs after 2 warmup runs):

- Full legacy (`_USE_COHORT_TOPOLOGY=False, _PRESHARE_VERTICAL_FACES=False`): 0.0183s (1.00x)
- Phase 1 vertical-only (`_USE_COHORT_TOPOLOGY=False, _PRESHARE_VERTICAL_FACES=True`): 0.0198s (0.92x)
- Phase 2 vertical + lateral (`_USE_COHORT_TOPOLOGY=True`): 0.0494s (0.37x)

Notes:
- Measurement is `build_phantom_shapes` wall time. The original Phase 1
  success criterion was `_fragment_all` BOP wall time inside cad_occ —
  that requires the full orchestrator run and is left to follow-up.
- The cohort topology builder's construction cost grows with cohort size
  (vertices × z-planes, edges × z-planes). On scenes dominated by many
  small cohorts, this overhead may exceed the BOP-skip savings; on scenes
  with few large cohorts (the production target), savings should
  dominate.
- On this benchmark scene (40 entities, 4 disjoint cohorts of 10 slabs
  each), Phase 2 shows a 2.7× overhead vs legacy within `build_phantom_shapes`.
  This is expected: the scene has no lateral adjacency between cohorts, so
  Phase 2 pays construction cost for shared topology but gets no BOP-skip
  savings in this stage. The savings materialise downstream in `_fragment_all`.
- Default `_USE_COHORT_TOPOLOGY=False` during stabilization. Flip to True
  in Phase 3 cleanup after the concentric-arc snap fix and the
  hanging-scene root-cause are resolved.

## Measured Results — `_fragment_all` BOP wall time (Task 13 follow-up)

Same approach but measuring the actual `_fragment_all` step inside `cad_occ`
(monkey-patched timer; see `scripts/bench_fragment_all.py`). Scenes use a
grid of edge-to-edge laterally-adjacent stacks, each with multiple vertical
layers — i.e., they exercise both vertical and lateral sharing.

| Scene             | Full legacy | Phase 1 (vert only) | Phase 2 (both) |
|-------------------|-------------|---------------------|----------------|
| 4 stacks × 6 layers (24 ent)  | 0.0774s (1.00×) | 0.0673s (1.15×) | 0.0268s (**2.89×**) |
| 4 stacks × 12 layers (48 ent) | 0.1527s (1.00×) | 0.1359s (1.12×) | 0.0492s (**3.11×**) |
| 8 stacks × 6 layers (48 ent)  | 0.1715s (1.00×) | 0.1192s (1.44×) | 0.0471s (**3.64×**) |
| 8 stacks × 12 layers (96 ent) | 0.3525s (1.00×) | 0.2970s (1.19×) | 0.0950s (**3.71×**) |

The Phase 2 success criterion (≥3× over legacy `_fragment_all`) is met on
all 48+ entity scenes. Phase 1 vertical-only sharing gives only modest
improvement (1.12–1.44×), confirming that lateral sharing is the dominant
win — which justifies the Phase 2 design choice over the simpler Phase 1.
