# Phase 3 parity sweep — 2026-05-28

Run with `_USE_DISCRETE_COHORT_MESH=True` against the full
`tests/structured/` suite.

Phase 3 tests (`test_phase3_*.py`) that patch the flag explicitly via
`unittest.mock.patch` are included and all pass (11/11) — the global
flag does not conflict with their internal `patch()` usage.

## Baseline (default off)

- Passed: 290
- Failed: 0
- Skipped: 4
- XFailed: 1 (`test_stacked_concentric_arc_discs_mesh_clean` — known arc annular issue)
- **Total: 295**

## Phase 3 on

Run method: `/tmp/run_phase3_sweep.py` sets `phantom._USE_DISCRETE_COHORT_MESH = True`
before importing pytest; individual stress tests run in isolation to survive segfaults.

- Passed: ~267 (estimated; suite aborts on segfault when run monolithically)
- Failed: 24 (NEW regressions relative to baseline)
- Skipped: 4 (unchanged)
- XFailed: 0 (the 1 baseline xfail now SIGSEGVs — counted in failures)
- **Delta: −24 relative to baseline**

## Failure buckets

### Bucket A — test assumes Phase 1+2 OCC path (7 tests)

These tests encode invariants of the per-piece sub-prism path (per-slab
`PhantomShape`, `slab_index >= 0`, exact shape counts). Under Phase 3 the
cohort envelope path produces one `PhantomShape` per cohort with
`slab_index = -(component_index + 1)`. The tests need to be removed or
adapted when the Phase 1+2 path is deleted (Task 23).

Action: add `pytest.mark.skipif(_USE_DISCRETE_COHORT_MESH, reason="Phase 1+2 path only")` or remove in Task 23.

- `tests/structured/test_phantom_discrete_routing.py::test_phase3_flag_off_keeps_phase2_behavior`
  Explicitly tests that flag-off produces Phase 2 per-piece count; meaningless when flag is globally on.

- `tests/structured/test_phantom_preshared_faces_integration.py::test_stacked_sub_prisms_share_interface_face_tshape`
  Asserts 2 PhantomShapes for a 2-slab stack; Phase 3 produces 1 cohort envelope shape.

- `tests/structured/test_phantom_shapes.py::test_build_phantom_shapes_one_slab_one_piece`
  Asserts `slab_index == 0`; Phase 3 sets `slab_index = -1` for cohort shapes.

- `tests/structured/test_phantom_shapes.py::test_build_phantom_shapes_multi_piece_partition`
  Asserts `len(result.shapes) == 2` for a 2-piece face_partition; Phase 3 collapses to 1 envelope.

- `tests/structured/test_phantom_use_cohort_topology.py::test_cohort_topology_path_produces_shared_lateral_face`
  Uses `cohort_topology_on` fixture then asserts `len(by_slab) == 2`; Phase 3 global flag overrides the fixture routing, yielding 1 cohort shape.

- `tests/structured/test_plan.py::test_fully_carved_slab_leaves_face_partition_empty_and_phantom_skips_it`
  Uses `plan.slabs[s.slab_index].source_index`; crashes for `slab_index = -1` (Phase 3 cohort shape). The assertion `src_indices == {1}` fails because `slab_index = -1` cannot index into `plan.slabs`.

- `tests/structured/test_cad_occ_phantom_hook.py::test_no_sliver_solids_for_ring_quarter_cut`
  Asserts each named entity has 2 shapes (one per face_partition piece); Phase 3 envelope assigns the single cohort solid only to the lowest-priority source (L1), so L2/L3 get `[]` and L1 gets 1 shape, not 2.

### Bucket B — production bug in Phase 3 (17 tests)

These failures indicate genuine issues in the Phase 3 cohort envelope
builder that must be fixed before flipping the default.

#### B1: `TopoDS::Shell` in cohort envelope sewing (4 tests)

`cohort_envelope.py:832` calls `TopoDS.Shell_s(sewn)` but the sewing
result is not a shell — the BRepBuilderAPI_Sewing object left `sewn`
as a compound (multiple disconnected shells) rather than a single
closed shell. Reproduces for scenes with disjoint cohort footprints
(two separate XY polygons stitched into one call).

Root cause: `_sew_cohort_solid` receives faces from geometrically
disconnected sub-volumes (different XY islands in the same cohort)
and the sewer produces a compound of shells, not a single shell.
`TopoDS.Shell_s()` then raises `Standard_TypeMismatch`.

- `tests/structured/test_phantom_shapes.py::test_build_phantom_shapes_is_deterministic_ordering`
  Two disjoint single-slab cohorts (s0 at XY [0,1]×[0,1] and s1 at [10,11]×[0,1]); each forms its own cohort. Both trigger the disconnected-shell failure.

- `tests/structured/test_phantom_shapes.py::test_group_phantom_solids_by_entity_inverts_slab_source_index`
  Same scene (entities A at [0,1]×[0,1], B at [10,11]×[0,1]).

- `tests/structured/test_cad_occ_phantom_hook.py::test_structured_entity_shapes_are_phantom_solids_after_cad_occ`
  Disjoint structured entities in a cad_occ scene.

- `tests/structured/test_cohort_topology_integration.py::test_mixed_cohort_sharing`
  Multi-cohort scene (cohort 1: A/B/C vertical stack at [0,1]×[0,1]; cohort 2: D/E lateral neighbors at [5,7]×[0,1] plus frame slabs). Each cohort builds correctly in isolation but the test invokes `_USE_COHORT_TOPOLOGY=True` fixture alongside the global Phase 3 flag, causing a routing conflict that reaches `_sew_cohort_solid` with disconnected geometry.

#### B2: Phase 5(d) BOP face-count mismatch in arc scenes (5 tests)

`builder.py:1024` raises `RuntimeError` when a cohort slab piece ends up
with more than 1 top or bottom OCC face after BOP fragmentation. This
happens for arc-disc and arc-annulus slabs whose curved top/bottom faces
get split into multiple OCC faces by the BOP cut against neighboring
rectangles. The Phase 3 code path reaches this check but the arc geometry
violates the "exactly 1 bot + 1 top" invariant that the check enforces.

- `tests/structured/test_arc_provenance_helpers.py::TestIntegration::test_disc_split_by_rectangle_top_cover`
  Disc slab covered by a rectangular neighbor; BOP splits top into 3 faces.

- `tests/structured/test_arc_provenance_helpers.py::TestIntegration::test_disc_split_by_two_overlapping_covers`
  Disc slab covered by two rectangles; BOP splits top into 5 faces.

- `tests/structured/test_arc_provenance_helpers.py::TestIntegration::test_annulus_split_by_partial_cover`
  Annulus slab with partial cover; BOP splits top into 4 faces.

- `tests/structured/test_structured_arc_polyprism.py::test_disc_structured_single_piece_meshes`
  Single-piece disc slab; BOP yields bot=[], top=[] (all faces lost — possible empty result from BOP on arc solid).

- `tests/structured/test_structured_arc_polyprism.py::test_split_disc_meshes_with_provenance`
  Split disc with arc provenance; BOP splits top into 3 faces.

#### B3: Interior interface non-conformality (2 tests)

Mesh interfaces between stacked cohort layers are non-conformal: 128
boundary triangles straddle the interface. The discrete cohort mesh path
does not yet stamp interior cut-line geometry onto the inter-layer
interface faces before meshing, so the mesh on each side is independent.

- `tests/structured/test_stress_stacked_patterns.py::test_three_stacked_layers_different_xy_tilings_mesh_clean_wedges`
  3-layer stack with different XY tilings per layer; 128 non-conformal triangles at interfaces.

- `tests/structured/test_stress_stacked_patterns.py::test_three_stacked_layers_with_void_keep_patterns_mesh_clean`
  3-layer stack with void-keep patterns; 128 non-conformal triangles at interfaces.

#### B4: SIGSEGV in mesh-producing stress tests (4 tests)

These tests produce an actual mesh (gmsh.model.mesh.generate()) and crash
the Python process with SIGSEGV. The crash is most likely in gmsh's
transfinite algorithm or its OCC CAD kernel when fed the cohort envelope
solid. Root cause not isolated — may be related to B1 (disconnected shell
→ invalid solid → gmsh CAD kernel assertion failure).

- `tests/structured/test_stress_stacked_patterns.py::test_four_stacked_layers_misaligned_seams_mesh_clean`
  4-layer stack with misaligned seams; segfaults during meshing.

- `tests/structured/test_stress_stacked_patterns.py::test_stacked_concentric_arc_discs_mesh_clean`
  Stacked concentric arc discs (also the baseline xfail, now SIGSEGV).

- `tests/structured/test_stress_stacked_patterns.py::test_stacked_overlapping_ring_segments_mesh_clean`
  Overlapping ring segments; segfaults during meshing.

- `tests/structured/test_stress_stacked_patterns.py::test_stacked_overlapping_ring_segments_with_lower_priority_planes_mesh_clean`
  Ring segments with lower-priority planes; segfaults during meshing.

#### B5: `test_phantom_to_gmsh_map` routing failures (2 tests)

- `tests/structured/test_phantom_to_gmsh_map.py::test_map_phantom_faces_to_gmsh_single_piece`
  `KeyError: 0` at `cohort_envelope.py:110` — `plan.arrangements[0]` is absent
  because the handcrafted `StructuredPlan` fixture does not populate
  `arrangements`. Under Phase 1+2 this code path was not reached; Phase 3
  routes through `_build_cohort_envelope` which calls `plan.arrangements[component_index]`.
  Fix: either populate `arrangements` in the fixture or add a guard in
  `_build_cohort_envelope` for minimal plans.

- `tests/structured/test_phantom_to_gmsh_map.py::test_map_phantom_faces_to_gmsh_missing_face_raises`
  `pytest.raises(RuntimeError)` was not raised — the Phase 3 path does not
  call `_map_phantom_faces_to_gmsh` in the same way the test expects
  (the function is called from builder context, not from `build_phantom_shapes`).

### Bucket C — footprint constancy reject

None. All cohort footprints in the existing test suite are constant across
z-levels (by design or by frame-slab framing). No `FootprintNotConstantError`
rejections observed.

## Summary table

| Bucket | Count | Action |
|--------|-------|--------|
| A — Phase 1+2 path assumption | 7 | Skip-when-Phase3-on or remove in Task 23 |
| B1 — TopoDS::Shell (disconnected cohort) | 4 | Fix `_sew_cohort_solid` to handle disjoint sub-volumes |
| B2 — Phase 5(d) arc BOP face count | 5 | Fix arc-disc/annulus face identification in builder |
| B3 — Interior interface non-conformality | 2 | Interior cut-line stamping for multi-tiling stacks |
| B4 — SIGSEGV in mesh-producing tests | 4 | Root-cause B1/B2 first; retest |
| B5 — phantom_to_gmsh_map routing | 2 | Fix fixture or add guard in `_build_cohort_envelope` |
| C — footprint constancy reject | 0 | N/A |
| **Total** | **24** | |

## Recommendation

- **Do NOT flip default to True** until Bucket B issues are addressed.
- **Most critical blockers:**
  1. **B1 (TopoDS::Shell)** — affects any scene with disjoint cohort
     geometry; the sewing code must handle compound-shell output from
     `BRepBuilderAPI_Sewing` by iterating sub-shapes rather than casting
     with `TopoDS.Shell_s`.
  2. **B4 (SIGSEGV)** — likely downstream of B1; confirm after B1 fix.
  3. **B2 (arc BOP)** — arc scenes are a key use-case; the Phase 5(d)
     face-count assertion needs to be relaxed or the arc top/bot faces
     need to be identified via provenance rather than positional count.
- **Bucket A** (7 tests) and **Bucket B5** (2 tests) are minor and can
  be addressed quickly (one-line skipif markers / fixture fix).
- **Bucket B3** (interior interface non-conformality) is a known
  limitation of the discrete cohort approach for multi-tiling stacks;
  fix requires interior cut-line stamping on shared horizontal faces.
- [x] Fix B1 (`_sew_cohort_solid` disconnected-shell handling) — landed 2026-05-29.
- [ ] Fix B2 (arc top/bot face identification post-BOP).
- [ ] Fix B3 (interior cut-line stamping).
- [ ] Rerun sweep after B1–B3 fixes; expect B4 (SIGSEGV) to resolve.
- [ ] Flip default to True (Task 22) after B1–B4 pass.
- [ ] Task 23: remove Phase 1+2 cohort code + Bucket A tests.

## Update — B1 fix landed (2026-05-29)

`assemble_cohort_envelope_solid` (formerly `_sew_cohort_solid`) now iterates
shells from `BRepBuilderAPI_Sewing.SewedShape()` via `TopExp_Explorer` instead
of casting the result directly to `TopoDS_Shell` with `TopoDS.Shell_s()`.
Disjoint cohorts (compound of shells) now add each shell to the solid as a
separate boundary component. The `Standard_TypeMismatch: Shape is not a
TopoDS_Shell` error is fully eliminated.

### Post-fix Phase 3 sweep (2026-05-29)

Run: `tests/structured/` with `_USE_DISCRETE_COHORT_MESH=True`.
Stress tests (`test_stress_stacked_patterns.py`) still SIGSEGV; counted
individually per-test.

- Bucket A — Phase 1+2 path assumption: **7** (unchanged)
- Bucket B1 — TopoDS::Shell cast: **0** (fixed — `Standard_TypeMismatch` gone)
  - 2 of the 4 former B1 tests now pass outright.
  - 2 of the 4 former B1 tests now fail with a different downstream error
    (multi-shell solid entity grouping / fixture routing mismatch, not the
    cast error); re-bucketed as B1b below.
- Bucket B1b — multi-shell solid not grouped to all entities (new): **2**
  - `test_group_phantom_solids_by_entity_inverts_slab_source_index`: entity B
    gets 0 shapes instead of 1 (`_group_phantom_solids_by_entity` does not
    handle the multi-shell cohort solid correctly).
  - `test_mixed_cohort_sharing`: `KeyError: 'A'` — cohort solid not assigned
    to entity A after multi-shell fix.
- Bucket B2 — Phase 5(d) arc BOP face count: **5** (unchanged)
- Bucket B3 — Interior interface non-conformality: **2** (unchanged)
- Bucket B4 — SIGSEGV in mesh-producing stress tests: **4** (unchanged; still
  crashes even after B1 cast fix — root cause not yet in B1)
- Bucket B5 — phantom_to_gmsh_map routing: **2** (unchanged)
- Total excluding stress-test SIGSEGVs: **25 failed** (was 20 non-SIGSEGV
  failures before; delta is the 2 new B1b failures and 3 additional failures
  in end-to-end tests that were previously masked by the B1 crash).

Baseline (Phase 3 off): **290 passed, 0 failed** — unchanged.
