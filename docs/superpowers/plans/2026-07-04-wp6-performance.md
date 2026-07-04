# WP6 — Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the O(n²) / per-element-Python hotspots identified in the spec's WP6 section, with zero behavior change.

**Architecture:** Every change is a semantics-preserving optimization: caching, vectorization, or algorithmic restructuring that must produce identical outputs. Hotspots that currently have NO test coverage (remesh internals, `filter_by_mass`, `plot3D`) get characterization/pinning tests BEFORE being touched. No benchmark gate; the existing behavior suites are the guard, plus a manual spot-check of `tests/test_performance_cad_occ.py` (skipped-by-default print benchmark).

**Tech Stack:** Python 3.13 / uv / pytest; numpy, shapely (GEOS/STRtree), matplotlib collections; pre-commit (black, ruff, codespell) — never `--no-verify`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-03-code-improvements-design.md` (WP6 section). Spec line numbers are stale; this plan's line numbers were re-verified at HEAD 6262139 and are authoritative. Where this plan says the spec's claim was wrong (Task 1, Task 2 sub-target d), the plan governs.
- Baseline at WP6 start: **452 passed / 7 skipped** (known flake `test_backends_mesh_adjacent_3d_equivalently`: rerun in isolation before treating as regression). Run `uv run pytest tests -q` from repo root; exact count match plus any new pinning tests.
- Golden references (`tests/references/`) byte-identical throughout.
- **Zero behavior change.** If an optimization cannot be made output-identical (including edge/boundary cases), implement the safe subset and document what was left, or report BLOCKED. Never silently change grouping/ordering/tolerance semantics.
- **Pinning-before-touching rule:** any function this plan modifies that has no direct test must first get a characterization test capturing CURRENT behavior on a small deterministic input; verify it passes on the unmodified code before optimizing.
- Magic values encountered (e.g. remesh's `1e-5`) keep their current numeric value; promote to a named constant only where the task says so.

---

### Task 1: `cad_occ.py` — cache per-shape bboxes in the cut-cascade loop

**Model:** sonnet (opus review)

**Files:**
- Modify: `meshwell/cad_occ.py` (`process_entities_cut_only`, lines ~429–482; `OCCLabeledEntity` at ~67–89; helper `shape_bbox` lives in `meshwell/occ_util.py:81-96` — do not change it)
- Tests: none created; `tests/test_cad_occ_cut_failure.py`, `tests/test_cad_occ_fragment_ownership.py`, `tests/test_cad_occ.py` and the backend-equivalence suite are the guard

**Reality check (spec correction):** the exact-distance check (`_shapes_actually_overlap`, line ~478) is ALREADY gated behind the AABB check (line ~468). The actual waste is `tb = shape_bbox(ts)` at line ~465 recomputing every prev-shape bbox on every outer iteration. The fix is bbox caching only; do not restructure the distance-check gating.

**Design:** cache bboxes per `OCCLabeledEntity`, invalidation-safe. The subtlety: `prev.shapes` may be REASSIGNED after cutting/fragmentation — a cache computed at instantiation can go stale. Preferred shape: make `shapes` participate in invalidation, e.g. a `bboxes()` method (or property) on `OCCLabeledEntity` that lazily computes `[shape_bbox(s) for s in self.shapes]` and memoizes keyed on identity of the shapes list (recompute if `self.shapes is not` the memoized list, or clear the memo at every assignment site of `.shapes`). Find EVERY site that mutates or reassigns `.shapes` on these entities (grep `\.shapes\s*=` and `\.shapes\.` across meshwell/) and prove in your report that the cache can never serve stale bboxes. Also hoist the per-outer-entity `obj_bboxes` (line ~441) through the same mechanism if trivial; don't force it.

- [ ] **Step 1:** Grep all `.shapes` mutation sites; write the staleness analysis in your report.
- [ ] **Step 2:** Implement the memoized bbox access; replace the line-465 recompute (and line-441 if trivial).
- [ ] **Step 3:** `uv run pytest tests -q` → count match.
- [ ] **Step 4:** Manually run `uv run pytest tests/test_performance_cad_occ.py -q --no-header -s --override-ini="addopts=" -m ""` (unskip by running the test function directly if needed) once before and once after; record both timings in the report (informational, no gate).
- [ ] **Step 5:** Commit — `perf(cad_occ): cache per-entity shape bboxes in cut-cascade loop`

### Task 2: `structured/` GEOS work — accumulation patterns, STRtree pre-check, `_cohort_xy_at` memoization

**Model:** sonnet

**Files:**
- Modify: `meshwell/structured/decompose.py` (`zinterval_footprint` loop ~203–208; `build_cohort_arrangement` loop ~257–270; `decompose_cohorts` pairwise loop ~396–425; `_cohort_xy_at` ~454–457)
- Modify: `meshwell/structured/validators.py` (`validate_z_stacks` call at :69; `validate_no_volumetric_cohort_overlap` call at :114)
- Tests: none created; `tests/structured/test_decompose_footprint.py`, `test_arrangement_canonical_edges.py`, `test_validators_zstack.py`, `test_validators_volumetric.py`, plus the structured suite are the guard

**Sub-targets:**
(a) `zinterval_footprint` and (b) `build_cohort_arrangement`: the spec's "accumulate list, one unary_union at end" does NOT directly apply — each iteration's `difference` needs the running accumulator. Semantics-preserving restructurings that DO work: replace `acc = unary_union([acc, new])` where `new` is disjoint-by-construction with keeping a parts list AND a running union only when the running union is genuinely needed; or batch consecutive same-kind slabs. Implement only what you can argue is output-identical (GEOS `unary_union` output geometry may differ in representation but must be geometrically equal — the tests compare geometric predicates, check what the covering tests actually assert). If no restructuring is provably identical, take the modest win: skip the union when `new.is_empty` / skip the difference when `acc.is_empty`, and document that the full accumulate-once pattern was inapplicable. Do NOT chase representation-identical output at the cost of correctness.
(c) `decompose_cohorts` ~396–425: add a cheap bounds pre-check before `_cohort_xy_at(...).intersects(ent.polygons)` — compare `ent.polygons.bounds` against the cohort's XY extent (max over slab footprint bounds, computable once per cohort) and skip the exact check when the AABBs don't overlap. AABB non-overlap → geometric non-intersection, so this is exactly output-preserving. A full STRtree is overkill at current scales; bounds check suffices (note this deviation from the spec's "STRtree" wording in your report).
(d) `_cohort_xy_at` memoization: add a per-call-site (or module-level per-run, if you can bound its lifetime — prefer passing an explicit dict) cache keyed `(id(cohort), quantized z)` using the existing `quantize_key` from `meshwell/structured/types.py` with the pipeline's `point_tolerance` if reachable at the call sites, else `round(z, 9)` with a comment. Three call sites: validators.py:69, validators.py:114, decompose.py:408. NOTE: the spec also named `structured/build.py:1077-1083` — that reference is DEAD (build.py has no `_cohort_xy_at` call); record the correction in your report, do not touch build.py.

- [ ] **Step 1:** Read all four sites + covering tests; decide (a)/(b) restructuring with written output-identity arguments.
- [ ] **Step 2:** Implement (a)–(d).
- [ ] **Step 3:** `uv run pytest tests -q` → count match.
- [ ] **Step 4:** Commit — `perf(structured): GEOS accumulation trims, bounds pre-check, memoized _cohort_xy_at`

### Task 3: `geometry_entity.py` — incremental circle-fit for arc detection

**Model:** sonnet (opus review)

**Files:**
- Modify: `meshwell/geometry_entity.py` (`_decompose_vertices_3d` lines ~151–242: outer window loop ~188–240, from-scratch `fit_circle_2d` call at ~196, full-window turn-angle re-scan ~201–210; `fit_circle_2d` at ~245–274 — keep it, it stays the single-shot API)
- Tests: `tests/test_arc_identification.py`, `test_arc_fusion.py`, `test_structured_arc_split.py`, `test_geometry_hardening.py` are the guard (good direct coverage exists — no new pinning test needed)

**Design:** `fit_circle_2d` is a Kåsa algebraic least-squares fit: `A=[2x,2y,1]`, `b=x²+y²`, `lstsq`. The normal-equation form `(AᵀA)c = Aᵀb` uses only running sums (Σx, Σy, Σx², Σy², Σxy, Σ(x²+y²)x, …) — maintain those sums incrementally as the window `vertices[i:j]` grows by one point per expansion, solving a 3×3 system per step (np.linalg.solve on the accumulated normal matrix). CAUTION: `lstsq` and normal equations differ numerically for ill-conditioned windows (near-collinear points). The RMSE residual must also be computed incrementally (expand ‖Ac−b‖² via the same sums). Acceptance bar: ALL existing arc tests pass unchanged, including `test_geometry_hardening.py`'s rotation-invariance cases. If numerical drift breaks any test, fall back to a smaller win: incremental sums for the fit but recompute the residual exactly on candidate acceptance only, or keep `lstsq` and only fix the turn-angle re-scan. The turn-angle validity re-scan (~201–210) should check only the newly added corner per expansion — verify by reading the condition that per-corner checks accumulate (each corner's condition doesn't depend on later points); if it does depend on the full window (e.g. against a refitted center), state that and keep the necessary recomputation.
Also: reset the running sums correctly when the window start `i` advances (no stale carryover).

- [ ] **Step 1:** Read the full function; write the incremental-sums derivation and the turn-angle dependency analysis in your report.
- [ ] **Step 2:** Implement; keep `fit_circle_2d` unchanged for external callers.
- [ ] **Step 3:** `uv run pytest tests/test_arc_identification.py tests/test_arc_fusion.py tests/test_structured_arc_split.py tests/test_geometry_hardening.py -q` then full suite → count match.
- [ ] **Step 4:** Commit — `perf(geometry_entity): incremental circle-fit and per-corner turn check in arc detection`

### Task 4: `remesh.py` — vectorize the four hot loops (pinning tests first)

**Model:** sonnet (opus review)

**Files:**
- Modify: `meshwell/remesh.py` (`_extract_edges` ~238–260; `get_current_mesh_sizes` ~262–290 loop at ~276–284; vmap remap in `_extract_gmsh_mesh_data` ~213–236 (dict comp at ~218, lookup loop at ~235); KDTree dedup in `compute_size_field` ~380–417, hard-coded `r=1e-5` at ~395)
- Test: create `tests/test_remesh_internals.py` — these four code paths have NO direct coverage (verified by grep); pinning tests are REQUIRED before touching them

**Pinning tests (write first, verify green on current code):** construct a `Remesher` (or exercise the methods with hand-built small arrays — read how the class is instantiated in `tests/test_remesh_lifecycle.py` and prefer direct unit calls with synthetic `triangles`/`tetrahedra`/`vtags`/`vxyz` arrays): (1) `_extract_edges` on a 2-triangle mesh sharing an edge and a 2-tet mesh → exact expected edge set; (2) `get_current_mesh_sizes` on a known geometry → exact per-node mean edge lengths; (3) the vmap remap: non-contiguous, non-sorted `vtags` → correct element vertex ids; (4) the dedup block in `compute_size_field`: points closer than 1e-5 collapse taking the MIN size, points farther apart survive — include a boundary-ish case (spacing ~2e-5 apart stays separate).

**Vectorizations:** edges via `np.unique(np.sort(pairs, axis=1), axis=0)` built from column stacking; size accumulation via `np.add.at` on sum/count arrays; vmap via `np.searchsorted` over sorted `vtags` (with argsort indirection — `vtags` are not guaranteed sorted); dedup — CAUTION: the spec suggests "tolerance-quantized np.unique" but grid-quantization is NOT equivalent to `query_ball_point` radius grouping (two points 0.9e-5 apart can quantize to different cells; transitive chains group differently). The current KDTree semantics are the contract. Acceptable implementations: keep `cKDTree` but replace the per-point Python loop with a single `tree.query_pairs(r=1e-5)`-based union-find / connected-components grouping IF that reproduces the current greedy `query_ball_point` grouping on the pinning tests — note the current greedy loop is order-dependent; whatever you do must keep the pinning tests green. If exact reproduction requires keeping the loop, vectorize only the min-size reduction inside it and say so. Promote `1e-5` to a module constant `_DEDUP_RADIUS = 1e-5` (same value).

- [ ] **Step 1:** Pinning tests → green on current code (commit them separately or together with the perf change, your call — but they must exist and pass BEFORE the rewrite in your working sequence).
- [ ] **Step 2:** Vectorize the four sites.
- [ ] **Step 3:** Full suite → count match + new tests.
- [ ] **Step 4:** Commit — `perf(remesh): vectorize edge extraction, size accumulation, vertex remap, and dedup grouping`

### Task 5: `filter_by_mass` single `getMass` + visualization collections/dedup

**Model:** sonnet

**Files:**
- Modify: `meshwell/_mesh_entity.py` (`filter_by_mass` inner helper `filter_by_target_and_tags` ~268–288: `getMass` called once to test the bound and again to build the dict — compute once per tag, reuse: `masses = {tag: getMass(...) for tag in tags}` then filter the dict)
- Modify: `meshwell/visualization.py` (`plot2D` per-triangle `ax.fill`+`ax.plot` loop ~175–196 and per-line loop ~225–236 → one `matplotlib.collections.PolyCollection` / `LineCollection` per physical group, preserving per-group color/label/alpha and legend behavior; `plot3D` ~288–303: dedupe scatter vertices via `np.unique(tets.ravel())` indexing into coords, and build the 6-edges-per-tet line set with `np.unique(np.sort(...), axis=0)` dedup before plotting)
- Tests: create `tests/test_mesh_entity_filter_by_mass.py` — `filter_by_mass` has NO coverage (verified): unit test with a stubbed `self.model.occ.getMass` counting calls: correct filtered dict AND each tag's mass computed exactly once (the call-count assertion IS the point of the change; also pins current filtering semantics: strict `min_mass < m < max_mass`). For `plot2D`, existing guard tests in `tests/test_visualization_guards.py` must stay green. For `plot3D` (NO coverage): add one smoke test in the same style as the existing guards (headless backend `matplotlib.use("Agg")` — check how existing viz tests handle it) asserting it runs without crash on a tiny tet mesh and that the scatter point count equals the number of UNIQUE vertices.

- [ ] **Step 1:** Write the `filter_by_mass` call-count test (red against a deliberately-double-calling assertion? No — it will be green on the dict-build if written after; instead: write it asserting once-per-tag, verify it FAILS on current code, then fix) and the `plot3D` smoke test (green after change).
- [ ] **Step 2:** Implement all three changes.
- [ ] **Step 3:** Full suite → count match + new tests.
- [ ] **Step 4:** Commit — `perf(viz,mesh_entity): matplotlib collections, deduped plot3D vertices, single getMass per tag`

### Task 6: `cad_gmsh.py` — Counter-based exterior boundaries, drop redundant subtraction

**Model:** sonnet

**Files:**
- Modify: `meshwell/cad_gmsh.py` (`_tag_entities` exterior-boundary block ~339–358: per-entity inner loop unioning ALL other entities' boundaries is O(n²) — replace with one pass: build `collections.Counter` of boundary tags over all top-dim entities' `boundary_of` sets once; an exterior boundary of entity e is a tag in `boundary_of[e.index]` with total count == 1. Verify this equivalence against the current code's `others` semantics — a tag in ≥2 entities' boundary sets lands in `others` for each of them; also confirm `boundary_of` values are sets (no double-count within one entity). Then drop the `same_material_interfaces` subtraction at ~353, which the code's own comment (~349–352) declares provably redundant — cite the comment's argument in the commit body. The pairwise same-material loop at ~306–328 stays UNCHANGED — it feeds interface naming, out of scope here.)
- Tests: none created; `tests/test_cad_gmsh.py` (boundary/interface naming tests) and the backend-equivalence suite (`tests/test_backend*`) are the guard — run the backend suite explicitly since exterior/interface tagging parity between backends was a WP2 invariant

- [ ] **Step 1:** Write the equivalence argument (Counter==union-of-others semantics) in your report, covering the `keep=False` entity handling — check whether the current inner loop skips any entities and mirror it exactly.
- [ ] **Step 2:** Implement.
- [ ] **Step 3:** `uv run pytest tests/test_cad_gmsh.py tests/ -q` full suite → count match; backend-equivalence tests green (flake rerun rule applies).
- [ ] **Step 4:** Commit — `perf(cad_gmsh): Counter-based exterior boundaries, drop redundant same-material subtraction`

---

## Final verification (orchestrator)

- [ ] `uv run pytest tests -q` fully green; `git diff --stat dfd9cde..HEAD -- tests/references/` empty.
- [ ] Whole-package review of the WP6 range before WP7.
