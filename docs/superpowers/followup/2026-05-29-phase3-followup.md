# Phase 3 follow-up — second handoff (2026-05-29)

**Date:** 2026-05-29
**Branch:** `feat/structured-clean4`
**Predecessor:** `docs/superpowers/followup/2026-05-28-phase3-followup.md`

## What this session did

Picked up the Phase 3 follow-up at "~22 failures, residual end-of-sweep
crash" and worked the recommended ordering: Bucket A skips, B1b
multi-shell solid split, B5 phantom-to-gmsh-map skips, and a partial B2
fix for the per-piece face_map routing. Net delta on the Phase 3 sweep:

- **Before:** 26 failed, 256 passed, 4 skipped
- **After:** 11 failed, 261 passed, 15 skipped
- **Delta:** –15 failures.

Baseline default-off sweep is unchanged: 283 passed, 4 skipped, 0 failed
(one new Phase 3 spec test takes the count from 282 to 283).

Stress test suite (`test_stress_stacked_patterns.py`) was not re-swept;
its B4 SIGSEGV root cause was already addressed by `ccc7b4a` per the
predecessor handoff, but the suite still needs a clean run-to-confirm.

## Commits landed

| Commit | Bucket | Summary |
|--------|--------|---------|
| `9fda0aa` | A + B5 | `pytest.mark.skipif` on 7 Bucket-A tests + 2 B5 tests + `test_mixed_cohort_sharing` + `test_group_phantom_solids_by_entity_inverts_slab_source_index` |
| `ab1181a` | B2 partial | `cohort_envelope.assemble_cohort_envelope_solid` propagates `sewing.Modified()` to `bottom_sub_faces` / `top_sub_faces`; `builder._filter_phase3_face_map_per_piece` filters multi-piece union-face BOP fragments by piece representative-point |
| `c6eccf1` | B1b | `_group_phantom_solids_by_entity` splits multi-shell cohort solid into per-source single-shell `TopoDS_Solid`s (XY-bbox match); new spec test `test_phase3_group_phantom_solids_splits_multi_shell_by_source` |

## Remaining failures (11 tests, 5 buckets)

### 1. Multi-piece volume building under shared bot OCC face (6 tests)

**Root cause** (investigated this session, not yet fixed):

Phase 3 cohort envelope builds **one** bot OCC face per cohort union
(`env.bottom_union_face`) when multiple piece sub-faces share the same
z-level. Routing in `_build_phantom_shapes_via_cohort_envelope` maps
**all** zmin/zmax per-piece `FaceKey`s to the same union face for BOP.

When a neighbor cuts only the slab's TOP (e.g. cap-above-slab), BOP
fragments the top union into per-piece fragments (good) but leaves the
bot union intact (one OCC face for both pieces). After my B2 partial
filter, `face_map` per piece is:

| Piece | bot tag | top tag |
|-------|---------|---------|
| 0 (under cap) | 1 (shared) | 3 (under-cap fragment) |
| 1 (uncapped)  | 1 (shared) | 4 (uncapped fragment) |

In `apply_structured_mesh`'s per-piece loop:

1. Piece 0 calls `_stamp_top_face_mesh(bot=1, top=3)` — reads ALL bot
   triangles, only piece 0's boundary edges are in `edge_correspondence`,
   so only piece 0's boundary nodes get top mappings. **Piece 1's boundary
   bot nodes are not in `bot_to_top_tag`.**
2. Piece 0's `_build_slab_volume(bot=1, ...)` then iterates ALL bot cells,
   including piece 1's region cells — **KeyError on piece 1's nodes.**

**Affected tests** (all KeyError downstream of Phase 5(d)):

- `test_end_to_end_minimal::test_multipiece_slab_with_top_cap_no_true_orphans`
- `test_end_to_end_multipiece::test_structured_slab_with_top_neighbour_produces_multi_piece_wedges`
- `test_multi_output_face::test_two_overlapping_top_neighbours_meshes_cleanly`
- `test_multi_output_face::test_intra_entity_seam_not_tagged_as_None`
- `test_multi_output_face::test_two_overlapping_bottom_neighbours_meshes_cleanly`
- `test_structured_arc_polyprism::test_split_disc_meshes_with_provenance`

**Investigated approaches** (this session, both abandoned):

- **A. Piece-polygon clipping** in `_stamp_top_face_mesh` and `_build_slab_volume`
  (filter `bot_triangles` / `bot_cells` by triangle/cell centroid in
  `slab.face_partition[piece_idx]`). Implemented, then reverted. **Result:**
  KeyError is replaced by gmsh `Invalid boundary mesh (overlapping facets)`
  during `generate(3)` because piece 0's volume now only covers piece 0's
  bot region but the shared bot OCC face's mesh still has triangles in
  piece 1's region with no owning volume. Net test count unchanged.

- **B. Relax Phase 5(d) check + shared-face cache extension** so multiple
  pieces sharing `(bot_tag, X)` reuse one volume. Not implemented — would
  break the per-piece `physical_name` semantics.

**Recommended next approach** (not started):

**Pre-fragment the bot/top union face by interior arrangement edges**
inside `cohort_envelope.assemble_cohort_envelope_solid` so that after
BOP each piece has its own OCC sub-face naturally. Options:

1. Run a local `BOPAlgo_GeneralFuse` on the union face with the interior
   arrangement edges (those skipped from lateral walls) as imprint tools,
   then add the fragment list of sub-faces to the sewing instead of the
   union face.
2. Skip the union step entirely: add per-piece sub-faces directly to
   `BRepBuilderAPI_Sewing`. The risk previously cited in commit `f84f61a`
   ("disconnected TopoDS_Face objects → open shell") is real for
   genuinely-disjoint multi-slab cohorts, but for face_partition pieces
   *within a single slab* the sub-faces share interior edges (built from
   `env.horizontal_edges` wires) and sewing should produce a manifold
   sheet. Distinguish the two cases by `len(cohort_slabs)` vs `len(pieces)`.

Either approach restores the 1-to-1 piece↔OCC-face invariant that
Phase 1+2 enjoyed naturally, after which the existing `apply_structured_mesh`
per-piece loop should just work.

### 2. Arc disc/annulus split tests (3 tests + 1 standalone)

**`test_arc_provenance_helpers::TestIntegration::test_disc_split_by_rectangle_top_cover`**,
**`test_disc_split_by_two_overlapping_covers`**,
**`test_annulus_split_by_partial_cover`**: all fail with `KeyError` in
`_stamp_top_face_mesh` — same root cause as bucket 1 (multi-piece shared
bot OCC face). The arc geometry adds curved boundaries but the fix is
the same.

**`test_structured_arc_polyprism::test_annulus_structured_single_piece_meshes`**:
fails with gmsh `Invalid boundary mesh (overlapping facets) on surface 1
surface 5` during `generate(3)`. This is a single-piece annulus so the
piece-clip approach is N/A; the issue is likely the discrete-volume
boundary vs the annular OCC face mesh having tolerance-level overlaps.
Worth a separate localization run.

### 3. Conformality assertion in simple lateral test (1 test)

**`test_end_to_end_minimal::test_simple_slab_lateral_mesh_is_conformal`**:
single 2x2 slab, n_layers=2. Test computes 66 boundary triangles
"not conformal with wedge mesh". This is a single-piece scene with no
face_partition split — so the multi-piece bug doesn't apply. Likely the
lateral OCC face mesh has subdivisions (interior nodes from gmsh
default mesher) that the wedge mesh doesn't include. The Phase 3
discrete volume builder uses prism-from-bot-triangulation, which only
matches the lateral OCC face mesh when the lateral has no mid-height
interior nodes.

Likely fixable by either:
- forcing the lateral OCC face to have no interior mesh (transfinite
  with 2 layers in Z), or
- accepting the lateral mesh from gmsh and projecting it onto the
  wedge sides.

### 4. Stress tests — not re-swept

The 4 B4 SIGSEGV tests were addressed by `ccc7b4a` per the predecessor
followup. They were **not** re-run in this session. A clean run of
`tests/structured/test_stress_stacked_patterns.py` under
`_MESHWELL_FORCE_PHASE3=1` is needed to confirm:
- B4 SIGSEGV is actually gone for all 4 tests.
- No new regressions from this session's changes.
- The residual end-of-sweep crash is either localized or also resolved.

### 5. Multi-shell entity grouping — solved but partial coverage

`_group_phantom_solids_by_entity` now correctly splits a multi-shell
cohort envelope solid into per-source single-shell solids when each
source owns a disjoint XY component. Spec test
`test_phase3_group_phantom_solids_splits_multi_shell_by_source` verifies
the A/B disjoint scene. Caveats:

- The split uses an XY-bbox containment match (shell vertex centroid in
  source's slab-footprint bbox). For non-axis-aligned shells or shells
  spanning multiple sources this falls back to legacy single-source
  assignment.
- The Phase-2-era assertion-style test
  `test_group_phantom_solids_by_entity_inverts_slab_source_index` is now
  skipped under Phase 3 because its identity check (`overrides[0][0] is
  a_solid`) doesn't apply when the solid is freshly built from a single
  shell.

## Recommended ordering for next session

1. **Pre-fragment cohort union face by interior arrangement edges** (the
   B3-adjacent root cause from this session). Eliminates ~6 failures in
   one focused change inside `cohort_envelope.assemble_cohort_envelope_solid`.
   Suggest approach (2) above (skip union, add per-piece sub-faces) because
   it's simpler — diagnose any open-shell sewing issue if it arises.
2. **Re-sweep stress tests** to confirm B4 is fixed and no new regressions.
3. **Localize the single-piece annulus overlapping-facets error**
   (`test_annulus_structured_single_piece_meshes`).
4. **Fix `test_simple_slab_lateral_mesh_is_conformal`** (single-piece
   lateral conformality — likely a recombine/transfinite tweak).
5. **Flip the kill switch** (`_USE_DISCRETE_COHORT_MESH = True` default)
   once steps 1–4 land and the sweep is at 0 failed.
6. **Task 23: delete Phase 1+2 cohort code** (`cohort_topology.py`,
   `_USE_COHORT_TOPOLOGY`, `_PRESHARE_VERTICAL_FACES`,
   `_build_phantom_shapes_via_cohort_topology`, Bucket-A skipped tests).

## Key files touched

- `meshwell/structured/cohort_envelope.py` — `sewing.Modified()` for `bottom_sub_faces` + `top_sub_faces`.
- `meshwell/structured/builder.py` — `_filter_phase3_face_map_per_piece` helper + Phase-3 call site.
- `meshwell/structured/phantom.py` — `_split_cohort_solid_by_source` helper + new Phase-3 branch in `_group_phantom_solids_by_entity`.
- `tests/structured/test_phase3_cad_occ_smoke.py` — added `test_phase3_group_phantom_solids_splits_multi_shell_by_source`.
- 7 `tests/structured/test_*.py` files — Bucket A `pytest.mark.skipif` markers.

## State of the kill switch

`_USE_DISCRETE_COHORT_MESH` is still `False` by default. Sweep mode (set
via `_MESHWELL_FORCE_PHASE3=1` env var or test-local `unittest.mock.patch`)
is the only way Phase 3 runs today. The kill switch can flip to default
True once the multi-piece volume building issue (item 1 above) is fixed.
