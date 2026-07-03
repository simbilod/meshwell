# WP4 — Fragile Patterns: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the fragile patterns identified in the 2026-07-03 review: import-time `cpu_count()` defaults, bare/broad excepts used as control flow, name-substring conventions as semantics, hash-as-shape-identity, hidden gmsh lifecycle state, unscaled magic tolerances, and the wedge error-ordering hazards (spec: `docs/superpowers/specs/2026-07-03-code-improvements-design.md`, WP4).

**Architecture:** Seven tasks on `code_improvement`, one commit each (a task may add a second commit in a review fix round). Tasks 1–3 are mechanical/standard; Tasks 4–7 are semantic and carry judgment notes. Unless a task says otherwise, changes are behavior-preserving for all valid inputs — only failure-path behavior (crashes, silences) may change, and each such change needs a pinning test.

**Tech Stack:** Python 3.13, pytest, gmsh, OCP, shapely.

## Global Constraints

- Baseline at WP4 start: **408 passed / 7 skipped, fully green** (commit `dfd9cde`). Any new failure is a regression. Known flake: `test_backends_mesh_adjacent_3d_equivalently` — rerun in isolation before treating as a regression.
- Tests live in `tests/`; run `uv run pytest tests ...` from `/home/simbil/Github/meshwell_structured_manual/meshwell`; full suite once per task before committing (≤ 10 min).
- Pre-commit hooks (black/ruff/codespell); never `--no-verify`.
- **Model dispatch:** per task `Model:` line.
- No golden-reference changes unless a task's notes explicitly authorize regeneration with per-file causal justification.
- Line numbers below predate WP1–WP3 edits — locate by content.

---

### Task 1: Mechanical sweep — `n_threads`, `gmsh.logger`, truthiness, error wording, residual `normalize_mesh_order`

**Model:** haiku

**Files:**
- Modify: `meshwell/mesh.py` (class `Mesh.__init__` + module-level `mesh()` — `n_threads: int = cpu_count()` defaults; the orphan `gmsh.logger.start()` near the retry loop), `meshwell/model.py` (`ModelManager.__init__` n_threads), `meshwell/cad_occ.py` (`CAD_OCC.__init__` + module-level `cad_occ()`), `meshwell/cad_gmsh.py` (`CAD_GMSH.__init__` + module-level wrapper if it has the same default), `meshwell/remesh.py` (every `n_threads: int = cpu_count()` signature), `meshwell/resolution.py` (truthiness: `if self.sizemax and self.distmax:` and the `min_size`/`max_size` checks — change to `is not None` so 0.0 is honored), `meshwell/occ_entity.py` (`instanciate` error message says "the GMSH CAD backend has been removed" — false; reword to "OCC_entity is only supported by the OCC backend (cad_occ); use a gmsh-native entity with cad_gmsh"), `meshwell/structured/decompose.py` (`_policy_b_key` inline mesh_order→inf) and `meshwell/structured/cohort_entity.py` (same) — adopt `cad_common.normalize_mesh_order` (no circular import there; do NOT touch interface_tag.py, which has a real cycle constraint).
- Test: `tests/test_fragile_defaults.py` (create)

**Interfaces:**
- Every `n_threads` parameter becomes `n_threads: int | None = None` and is resolved in the body as `self.n_threads = n_threads if n_threads is not None else (cpu_count() or 1)` (import `cpu_count` where already imported). Forwarding call sites that pass `n_threads=<their value>` keep working because `None` now flows through to the same resolution.

- [ ] **Step 1: Failing tests**

```python
"""Import-time defaults and truthiness hazards stay fixed."""
import inspect


def test_no_import_time_cpu_count_defaults():
    import meshwell.cad_gmsh as cg
    import meshwell.cad_occ as co
    import meshwell.mesh as m
    import meshwell.model as mo
    import meshwell.remesh as r

    for obj in [
        m.Mesh.__init__, m.mesh,
        mo.ModelManager.__init__,
        co.CAD_OCC.__init__, co.cad_occ,
        cg.CAD_GMSH.__init__,
        r.Remesher.__init__,
    ]:
        sig = inspect.signature(obj)
        if "n_threads" in sig.parameters:
            assert sig.parameters["n_threads"].default is None, obj


def test_n_threads_none_resolves_to_positive_int():
    from meshwell.model import ModelManager

    mm = ModelManager(n_threads=None)
    try:
        assert isinstance(mm.n_threads, int) and mm.n_threads >= 1
    finally:
        mm.finalize()


def test_zero_valued_resolution_fields_are_honored():
    # sizemax/distmax of 0.0 must not be skipped by truthiness checks
    import meshwell.resolution as res

    src = inspect.getsource(res)
    assert "if self.sizemax and self.distmax" not in src
```

(Adapt the exact object list to reality — if a listed callable has no `n_threads` parameter, drop it from the loop and note it; if OTHER public callables have the `cpu_count()` default — grep `cpu_count()` across meshwell/ — add them.)

- [ ] **Step 2: RED**, **Step 3: implement** (also delete the unconsumed `gmsh.logger.start()` in mesh.py's meshing routine — grep `logger.start`; it is never stopped or read), **Step 4: full suite green**, **Step 5: commit** — `fix: resolve n_threads at call time; honor zero-valued resolution bounds

Also removes an unconsumed gmsh.logger.start(), rewords a false
error message in OCC_entity.instanciate, and adopts
normalize_mesh_order at the two structured call sites.`

---

### Task 2: remesh.py bare excepts + gmsh lifecycle ownership

**Model:** sonnet

**Files:**
- Modify: `meshwell/remesh.py` (bare `except:` at the `getCurrent` guard, the 3D/2D element-type fallback in `_extract_gmsh_mesh_data`, and the size-map load; `RemeshGMSH.remesh`'s raw `gmsh.open()`; `remesh_mmg` and `compute_total_size_map` missing `finalize()`), `meshwell/model.py` (`_initialize` finalizes a live session unconditionally — add a `warnings.warn` naming the displaced model; add a `load_geometry(path)` method wrapping `gmsh.open`/merge semantics so remesh stops bypassing the manager)
- Test: `tests/test_remesh_lifecycle.py` (create)

**Interfaces:**
- `_extract_gmsh_mesh_data` selects element type by COUNT, not exception: query `gmsh.model.mesh.getElementsByType(4)`; if empty, try type 2; if both empty, `self.triangles = None`. A `KeyError` from the vmap remapping must PROPAGATE (it means corrupt node references, not "try 2D").
- `ModelManager.load_geometry(path: Path | str)` — loads a geometry/mesh file into the managed model (`gmsh.open`), keeping the manager's current-model bookkeeping consistent; `RemeshGMSH.remesh` calls it instead of raw `gmsh.open()`.
- `remesh_mmg` / `compute_total_size_map`: wrap their remesher usage in `try/finally: remesher.finalize()` matching `remesh_gmsh`'s ownership discipline (verify `finalize` semantics — only finalize what the function itself created).

**Judgment notes:** read each bare `except:` and classify what it can actually catch before replacing; the getCurrent guard should catch the specific gmsh exception type (probably `Exception` from the gmsh API — check `gmsh.logger`-adjacent code or test empirically what `gmsh.model.getCurrent()` raises when uninitialized) with a comment. The `ModelManager` warn must not fire during normal single-manager operation (only when a DIFFERENT live session is displaced) — check how `ensure_initialized` re-enters `_initialize` in the same-manager path; if the warn would fire on every re-init, gate it on "gmsh initialized AND not initialized by this manager instance".

- [ ] **Step 1: Failing tests** — (a) meshio→gmsh path: a mesh whose tetra extraction hits a corrupt vmap (monkeypatch `getElementsByType` to return node tags absent from `getNodes`) must RAISE, not silently fall through to 2D; (b) `ModelManager` displacement warning: create manager A, then manager B — `pytest.warns(UserWarning)`; same-manager re-init emits no warning (`warnings.catch_warnings` + `simplefilter("error")`); (c) `remesh_mmg`-owned finalize: after calling with a Path input (monkeypatch the MMG binary invocation if needed — check how existing remesh tests stub it), `gmsh.isInitialized()` is False.
- [ ] **Step 2: RED**, **Step 3: implement**, **Step 4: full suite green** (remesh tests in `tests/` that exercise gmsh/mmg remeshing are the guard), **Step 5: commit** — `fix(remesh): element-type selection by count, managed gmsh lifecycle

Bare excepts no longer swallow KeyError as try-2D control flow;
ModelManager warns when displacing another live session and owns
geometry loading; remesh_mmg/compute_total_size_map finalize what
they create.`

---

### Task 3: cad_common registry/buffer fixes + AABB tolerance derivation

**Model:** sonnet

**Files:**
- Modify: `meshwell/cad_common.py` (`polygon_ents` registry keyed only by FIRST physical name — register under every name in the tuple; mitre-join buffer clipped by bbox inflated by only 1×perturbation — inflate by `5 * perturbation` to cover shapely's default mitre_limit=5), `meshwell/occ_xao_writer.py` (`_DEFAULT_AABB_INTERFACE_TOL` hardcodes the default point_tolerance; make `write_xao`'s tolerance parameter default `None` and derive via the existing `default_interface_aabb_tolerance()` helper from a `point_tolerance` argument — trace who calls `write_xao` and what tolerance they can supply)
- Test: `tests/test_cad_common_registry.py` (create)

**Judgment notes:** for the registry fix, find the actual consumer (InterfaceTag resolution reads the registry) and write the failing test at that level: an InterfaceTag referencing the SECOND name of a two-name entity currently fails to resolve — pin that it now resolves. For the mitre inflation, the failing test is geometric: a sharp-cornered polygon touching the scene bbox whose mitred buffer currently gets shaved — assert corner preservation within tolerance after the fix (construct the scenario from reading `prepare_entities`'s bbox logic; if constructing a genuine shave case proves impractical, a unit test pinning the inflation factor read from the code is the fallback — say which you did). For the AABB tolerance: callers of `write_xao` may not have point_tolerance in scope — if the plumbing is disproportionate, an acceptable outcome is deriving the default inside `write_xao` from an optional `point_tolerance=None` parameter that callers who have it (cad_occ pipeline) now pass, with the old constant as fallback — document what you chose.

- [ ] **Step 1: Failing tests**, **Step 2: RED**, **Step 3: implement**, **Step 4: full suite green**, **Step 5: commit** — `fix(cad): register entities under all physical names; scale mitre bbox and AABB tolerances`

---

### Task 4: Replace name-substring conventions with explicit flags

**Model:** opus

**Files:**
- Modify: `meshwell/occ_xao_writer.py` (the `"iface" in n` substring check that silently drops interfaces for any entity whose name contains "iface"; the `__cohort_` prefix checks in `_is_purely_synthetic`/`_filter_real_names`), `meshwell/interface_tag.py` (source of the iface-helper entities — add the explicit attribute), `meshwell/orchestrator.py` + `meshwell/structured/` cohort-entity construction (source of `__cohort_` names — add the explicit attribute), `meshwell/cad_occ.py` (`OCCLabeledEntity` already has `_is_cohort`; thread whatever flag is needed through to the writer)
- Test: `tests/test_writer_name_conventions.py` (create)

**Interfaces:**
- Entities that are interface helpers carry `is_interface_helper: bool = False` (attribute on the entity object, propagated onto the labeled-entity records the writer sees); synthetic cohort entities carry `is_synthetic: bool = False` (or reuse/extend the existing `_is_cohort` plumbing — read what the writer actually receives and pick the minimal honest carrier; name it in your report).
- The writer's decisions currently keyed on `"iface" in name` / `name.startswith("__cohort_")` key on the flags instead. The NAME conventions may remain as debugging aids but must not carry semantics.

**Judgment notes:** this is the task most likely to have hidden couplings — grep EVERY occurrence of `"iface"` and `__cohort_` in meshwell/ and tests/ first and build a complete map in your report (site → what semantic it carries → what replaces it). Some sites may be pure logging/naming (leave); some tests may pin the substring behavior (update them to pin the flag behavior instead, with justification). A user entity named `interface_oxide` must now get its interface groups written — that's the headline failing test. If the flag cannot reach the writer without changing a serialization format (XAO round-trip), STOP and report BLOCKED with the coupling — do not invent a format change.

- [ ] **Step 1: Failing test** — user entity whose physical name contains "iface" (e.g. `interface_oxide`) gets its `A___B` interface group in the written XAO; synthetic-cohort filtering still works when a user names an entity `__cohort_trap`.
- [ ] **Step 2: RED**, **Step 3: implement**, **Step 4: full suite green**, **Step 5: commit** — `fix(occ_xao_writer): explicit flags replace name-substring semantics`

---

### Task 5: Shape identity via `TopTools_IndexedMapOfShape`

**Model:** opus

**Files:**
- Modify: `meshwell/cad_occ.py` (`_shape_key` — hash+orientation tuple used as dict identity), `meshwell/occ_xao_writer.py` (every `_SHAPE_HASHER`/`TopTools_ShapeMapHasher` site used as a dict/set key: fragment piece ids, `lower_dim_ids`, `topology_local_index`, boundary sid maps; plus the unvalidated `FindIndex` result serialized into group references)
- Test: `tests/test_shape_identity.py` (create)

**Interfaces:**
- One identity registry per pipeline run: an `IndexedShapeRegistry` (thin wrapper over `TopTools_IndexedMapOfShape` — `index_of(shape) -> int` adding on first sight) providing collision-free integer identities via real `IsSame` comparison. Python dicts key on the returned index. Where orientation mattered (cad_occ's key included `Orientation()`), the wrapper key becomes `(index, orientation_int)` — preserve exactly which sites used orientation and which didn't; list them in your report.
- Every `shape_reference_map.FindIndex(shape)` result that gets serialized must be validated `> 0`, raising `ValueError` naming the group, instead of writing reference="0".

**Judgment notes:** this touches the writer's core bookkeeping. Work in small verified steps: first swap `_shape_key` in cad_occ (fragment candidate collection) and run the CAD/ownership tests; then the writer sites one cluster at a time with the XAO test set (`tests/test_occ_xao_writer.py`, `test_xao_writer_*.py`, `test_occ_entity.py`) after each cluster. The registry must not outlive a run (no module-global state — the current `_SHAPE_HASHER` is a module global but stateless; your registry is stateful, so it must be created per `write_xao`/per CAD_OCC instance). If any test depends on hash values being stable across separate registries, that test was depending on the bug — adjudicate and report. Golden XAO references must remain byte-identical — if reference order changes because dict iteration order changed, STOP and reassess (indices from an IndexedMap are insertion-ordered like the hashes' first-seen order, so order should be preserved — verify).

- [ ] **Step 1: Failing test** — construct two distinct simple shapes; assert registry gives distinct indices, same shape (and a re-wrapped copy of the same TShape via a fresh Python wrapper) gives the same index; FindIndex validation raises on a shape absent from the reference map.
- [ ] **Step 2: RED**, **Step 3: implement in verified steps**, **Step 4: full suite green, goldens byte-identical**, **Step 5: commit** — `fix(occ): collision-free shape identity via TopTools_IndexedMapOfShape

Hash values are no longer used as dict identity; XAO group references
are validated before serialization.`

---

### Task 6: Structured tolerance discipline + narrow arc-edge catches

**Model:** opus

**Files:**
- Modify: `meshwell/structured/build.py` (z-plane dicts keyed by raw floats — `z_plane_id_arcs`, `z_plane_arc_tol`, `z_plane_min_arc_pts`, `sub_idx_by_z` and the `arc_params_for_z` lookup: key by `quantize_key(0.0, 0.0, z, point_tolerance)[2]`; `EdgeRegistry.vertical`'s cache key not endpoint-order-invariant — sort the two z keys like `line_xy` does; `_is_arc_edge` + `_build_cylindrical_lateral_face` broad `except Exception` — catch the specific OCC exception types and `warnings.warn` with edge context instead of silently treating as not-an-arc), `meshwell/structured/wedge.py` (`z_tol = 1e-7`, `snap_tolerance = 1e-6` magic numbers; the `setPeriodic` broad catch — narrow to the gmsh error type), `meshwell/structured/pipeline.py` (bbox-match `tol = 1e-3` at two sites)
- Test: `tests/test_structured_tolerances.py` (create)

**Judgment notes:** for each magic tolerance, decide: derive from `point_tolerance` (when it gates geometric matching that must be consistent with the quantization grid) vs. promote to a named module constant with a comment (when it's a genuinely independent numerical guard, e.g. FP-noise epsilon). Do NOT blindly rescale — a tolerance that has been 1e-6 in passing tests encodes validated behavior; derivation must reproduce the current value under default point_tolerance=1e-3 (e.g. `point_tolerance * 1e-3` for 1e-6) or be left as a named constant. List every tolerance with its disposition in your report. The z-plane float-key fix is the load-bearing one: ULP-noise between a subpiece's z and a slab's zhi currently silently falls back to per-slab arc params, breaking cross-plane arc propagation — pin with a test that a z value differing by 1e-12 still hits the same key. For `_is_arc_edge`: determine what OCP actually raises for a non-curve edge (probe empirically) and catch exactly that; anything else propagates.

- [ ] **Step 1: Failing tests** — quantized z-key lookup tolerant to ULP noise; vertical edge cache returns the same edge for (zlo, zhi) and (zhi, zlo); `_is_arc_edge` warns (not silences) on an unexpected failure injected via monkeypatch.
- [ ] **Step 2: RED**, **Step 3: implement**, **Step 4: full suite green (structured suite is the guard)**, **Step 5: commit** — `fix(structured): quantized z-plane keys, order-invariant vertical cache, scaled tolerances, narrow arc catches`

---

### Task 7: Wedge error ordering + placeholder hardening

**Model:** opus

**Files:**
- Modify: `meshwell/structured/wedge.py`: (a) delete the mathematically-unreachable `WedgeCountMismatchError` check (`emitted == expected` always — verify the invariant still holds in current code before deleting); (b) raise the real node-mismatch error BEFORE any `addElementsByType` call so the gmsh model is never left half-mutated; (c) the intermediate-layer mismatch count currently discarded (`this_map, _ = ...`) — check or deliberately document it; (d) `_emit_lateral_face_quads` silent `return`s on row-length mismatch / failed left-right vertical assignment — raise `StructuredTransfiniteRejectedError` (matching the existing bot/top count check) or at minimum warn, and prevent both verticals landing in one slot silently; (e) the step-4.5 placeholder triangle uses the first three boundary nodes which may be collinear — pick three non-collinear nodes and guard the invariant that every placeholder face is later re-stamped (assert or documented reasoning).
- Test: `tests/test_wedge_hardening.py` (create; unit-level with stubs where driving full wedge assembly is impractical — the structured suite integration tests are the behavioral guard)

**Judgment notes:** this file is dense and load-bearing; read the full flow (`wedge.py` end to end) before touching anything. For (b), moving the raise means computing the mismatch check earlier — make sure the data it needs exists before the emission loop; if the check fundamentally requires mid-emission state, the alternative is buffering element batches and committing only after validation — choose the smaller safe change and justify. For (d), grep how callers react to the silent return today: if a silent return currently produces a *working* unstructured fallback mesh in shipping scenes, converting it to a hard raise may break valid workflows — in that case prefer `warnings.warn` + the existing fallback, and say so. Golden/structured tests must stay green; no regeneration is expected for pure error-path changes.

- [ ] **Step 1: Failing tests**, **Step 2: RED**, **Step 3: implement**, **Step 4: full suite green**, **Step 5: commit** — `fix(structured/wedge): fail before mutation, no silent lateral-face degradation`

---

## Final verification (orchestrator)

- [ ] `uv run pytest tests -q` fully green; `git diff --stat dfd9cde..HEAD -- tests/references/` empty.
- [ ] Whole-package review of the WP4 range before WP5.
