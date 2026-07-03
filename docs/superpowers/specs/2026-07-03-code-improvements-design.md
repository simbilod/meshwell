# Code Improvements — Design

**Date:** 2026-07-03
**Branch:** `code_improvement`
**Status:** awaiting user review

## Goal

Fix the correctness bugs, fragile patterns, duplication, performance problems, and
hygiene issues identified in the 2026-07-03 four-subsystem code review, at full scope
(everything except `quality.py`, which is deferred to its own PR).

## Decisions (user-confirmed)

1. **Dead APIs → remove/raise.** Delete the `periodic_entities` plumbing from
   `mesh.py` (parameter, docs, and the never-called `_apply_periodic_boundaries` /
   `_set_periodic_pair` methods). Make `additive=True` raise `NotImplementedError`
   at construction in all entity classes (`OCC_entity`, `PolyPrism`, `PolySurface`,
   `PolyLine`); keep the field in serialization for round-trip compatibility of
   existing `additive=False` dicts.
2. **Tie policy → OCC is source of truth.** On mesh-order ties the pre-fragment cut
   is skipped and the final fragment pass resolves ownership (earlier-declared entity
   wins the overlap). Align `cad_gmsh.py` to skip tie cuts. Final material ownership
   is unchanged in both backends (verified by reading `_fragment_all` +
   `_resolve_piece_ownership`); only boundary-face construction converges.
3. **Same-material `A___A` interfaces → not emitted, in both backends**
   (*provisional — user did not answer this sub-question; the recommended option was
   adopted. Revisit before WP2 lands. Alternative: an `emit_self_interfaces` flag.*)
   Align `cad_gmsh.py:331` to the OCC XAO writer's behavior of skipping
   identically-named neighbor interfaces.
4. **Scope → everything** (correctness + fragile patterns + dedup + perf + hygiene).
5. **`quality.py` → deferred entirely** to a separate future PR (parser rewrite on
   meshio, group-attribution fix, vectorization, module-level `warnings` filter).

## Approach

Staged, independently-reviewable work packages on the existing `code_improvement`
branch, ordered so that correctness lands first and dedup lands before the code it
protects gets further edits. Alternatives considered and rejected: one mega-commit
(unreviewable, high regression risk) and per-file fixes (scatters one logical change
— e.g. `cpu_count` — across many commits).

Each work package: write failing tests first for behavior changes (TDD), fix, run the
full suite, one commit (or a few logically-grouped commits). Test baseline before any
change: 3 stable + 1 flaky pre-existing failures on this branch — record exact IDs
first and require "no new failures" rather than "all green".

## Work packages

### WP1 — Correctness: silent wrong results

| Fix | Where |
|---|---|
| Remove `periodic_entities` plumbing (decision 1) | `mesh.py:109-160, 506` |
| `additive=True` raises `NotImplementedError` (decision 1) | `occ_entity.py`, `polyprism.py`, `polysurface.py`, `polyline.py` |
| Unify closed-arc split: one shared helper (indices + one closedness test) used by both the horizontal-face and lateral-face paths | `structured/build.py:216-234` vs `444-469` |
| Prefer `tetra` over `triangle` (select by dim) when loading meshio meshes for remeshing; populate `vtags`/`triangles_tags` consistently | `remesh.py:188-194` |
| Check `cut_op.IsDone()` and `result.IsNull()` after `BRepAlgoAPI_Cut`; fall back to pre-cut shape with a warning | `cad_occ.py:528-541` |
| Always warn on gmsh cut failure (not only when `progress_bars=True`); narrow the except | `cad_gmsh.py:451-453` |
| Replace `except Exception: return 0` around `addThruSections` with narrow catch + raise/warn including `physical_name` | `polyprism.py:206-214` |
| Hierarchical GDS: use `cell.get_polygons(depth=None, layer=…, datatype=…)`; convert paths or drop path layers from auto-detection; descriptive error on missing top cell | `import_gds.py:13, 49, 55-57` |
| Fix `filter_tags_by_target_dimension`: explicit point case (`getBoundary(..., recursive=True)`), default `[]`, no `UnboundLocalError` | `_mesh_entity.py:186-216` |
| Legacy sharing fallback: normalize `physical_name` to a set, exact-name comparison (no per-character/substring matching); guard at `:360` uses `is None` | `_mesh_entity.py:360, 409, 423` |

### WP2 — Correctness: crash paths + backend alignment

| Fix | Where |
|---|---|
| `resolution_specs` default `None`, normalized to `{}` | `mesh.py:509` |
| Normalize `polygons` input to flat `list[Polygon]` in `PolyPrism.__init__` (fixes list-input crash on buffered path; fixes `from_dict` MultiPolygon crash; enables deleting dead `to_dict` branches in the other classes) | `polyprism.py:164, 651`, `polysurface.py:212-221`, `polyline.py:198-210` |
| Filter zero wire tags in PolyLine like PolySurface does | `polyline.py:151-155` |
| Guard empty cut result in `_create_surface_with_holes` (share polyprism's guarded logic) | `polysurface.py:88, 119` |
| `plot2D`/`plot3D`: guard missing `triangle`/`line` cell blocks and missing group ids (`.get()`) | `visualization.py:152, 161-165, 204, 213-217` |
| Clear `ValueError` for empty `buffers` dict; drop duplicate `self.additive` assignment | `polyprism.py:80-82, 114-115` |
| gmsh backend skips mesh-order-tie cuts (decision 2) | `cad_gmsh.py:437-439` |
| gmsh backend stops emitting `A___A` same-material interfaces (decision 3, provisional) | `cad_gmsh.py:331-332` |
| Add `prepared` flag to `CAD_GMSH.process_entities` (symmetric double-buffer protection) | `cad_gmsh.py:412-416`, `cad_common.py:28-45` |
| Update the "match exactly" docstrings to state the now-true contract | `cad_occ.py:27`, `cad_gmsh.py` header |

### WP3 — Dedup (prevents future drift)

| Fix | Where |
|---|---|
| Move `_resolve_piece_ownership`, the mesh-order sort key, and the `mo → inf` normalization to `cad_common.py`; move `_shape_bbox`/`_shape_aabb` to a shared OCP util | `cad_occ.py:91-108, 200-211, 438-444`, `cad_gmsh.py:63-81, 419-425`, `occ_xao_writer.py:197-203` |
| One shared `quantize_key(x, y, z, tol)` (in `structured/types.py`); `VertexRegistry._key` and the validator closure delegate to it | `structured/build.py:62-64`, `structured/decompose.py:39-49, 492-493` |
| Store `point_tolerance` as a field on `Arrangement`; delete the `_polygon_point_tol` reverse-engineering heuristic | `structured/types.py:136`, `structured/decompose.py:531-548` |
| Hoist into `GeometryEntity`: WKT `to_dict`/`from_dict` helpers, hole-cut routine, `plot_decomposition` loop, `format_physical_name` everywhere (fixes list→tuple normalization inconsistency) | `polyprism.py`, `polysurface.py`, `polyline.py` |
| `process_mesh` tail → `return self.to_meshio()`; `ConstantInField.apply` reuses the MathEval+Restrict batching; one dim→`"…List"` dict constant | `mesh.py:447-452`, `resolution.py:47-59, 103-115`, `_mesh_entity.py:458-465` |
| Extract the near-duplicate interface-recovery branches into one helper (fixes the `_explicit_dim` inconsistency) | `mesh.py:255-291` |

### WP4 — Fragile patterns

| Fix | Where |
|---|---|
| `n_threads` default `None`, resolved as `n_threads or cpu_count() or 1` in `__init__` (5 sites) | `mesh.py:48`, `model.py:21`, `cad_occ.py:117`, `cad_gmsh.py:90`, `remesh.py:122` |
| Replace bare `except:` with narrow exceptions; stop using exceptions as 3D/2D control flow (check element counts) | `remesh.py:172, 224, 230, 313` |
| `_is_arc_edge` / cylindrical-lateral fallbacks: catch OCC error types, warn with context | `structured/build.py:767, 860-862`; `wedge.py:369` |
| Replace `"iface"` substring and `__cohort_` prefix conventions with explicit entity flags (`is_interface_helper`, `is_synthetic`) | `occ_xao_writer.py:231-244, 433` |
| Shape identity via `TopTools_IndexedMapOfShape` (`Add`/`FindIndex`) instead of raw hash values; assert `FindIndex > 0` before serializing group references | `cad_occ.py:79-88`, `occ_xao_writer.py:67, 191, 458, 562-568, 599` |
| `ModelManager`: warn when finalizing another live session; add `load_geometry` so `RemeshGMSH.remesh` stops calling `gmsh.open` behind its back; `remesh_mmg`/`compute_total_size_map` finalize in `finally` | `model.py:82-86`, `remesh.py:479, 804-833` |
| Remove or properly consume `gmsh.logger.start()` | `mesh.py:422-423` |
| `is not None` instead of truthiness for numeric-zero-valid fields | `resolution.py:287, 440-443` |
| Z-plane dicts keyed by `round(z / point_tolerance)`; endpoint-order-invariant vertical-edge cache key; named tolerance constants derived from `point_tolerance` where they gate geometric matching | `structured/build.py:95-102, 1045-1073`, `wedge.py:68, 565`, `pipeline.py:359, 419` |
| Wedge: drop unreachable `WedgeCountMismatchError` check; raise node-mismatch **before** `addElementsByType`; stop discarding intermediate-layer mismatch counts; raise (not silently return) on lateral-face row/vertical assignment failure; non-collinear placeholder triangle + guarded re-stamp invariant | `wedge.py:183-212, 391-417, 719, 768-779` |
| AABB interface tolerance derived from `point_tolerance` (default `None` in `write_xao`) | `occ_xao_writer.py:87-88, 251, 487` |
| Register multi-name entities under every physical name; inflate buffer bbox by `mitre_limit × perturbation` | `cad_common.py:72-111` |
| Reword misleading `OCC_entity.instanciate` error | `occ_entity.py:44-49` |

### WP5 — Dead code removal

`_mesh_entity.py`: shadowed `boundaries()` method, `_fuse_self`. `mesh.py`:
`_restore_structured_sweeps`. `structured/build.py`: unreachable `canon.is_closed`
branch (`:605-628`), duplicated `return edges` (`:274`). `polyprism.py`: dead+buggy
`_validate_polygon_buffers` (delete; the `addThruSections` error handling from WP1
covers its intent) and resolve the `MANUAL_NOTE: delete this` on live `subdivide`
(keep the feature, delete the note, simplify the numpy bbox fold).
`geometry_entity.py:348-350, 469-474`: dead `is None` cache guards.

### WP6 — Performance

| Fix | Where |
|---|---|
| Cache per-shape bboxes on `OCCLabeledEntity` at instantiation (kills O(n²) rebuilds); only run exact-distance check on the specific bbox-overlapping obj shape | `cad_occ.py:495-513` |
| Incremental GEOS union (accumulate list, one `unary_union` at end); drop redundant `difference` in `zinterval_footprint`; STRtree/bounds pre-check for shared-horizontal detection; cache `_cohort_xy_at` per (cohort, quantized z) | `structured/decompose.py:203-210, 262-271, 454-457`, `build.py:1077-1083`, `validators.py:69, 114` |
| Incremental circle-fit (running sums) for arc detection; check only the newly added corner per expansion | `geometry_entity.py:185-241` |
| Vectorize remesh loops: edge extraction (`np.unique(np.sort(...), axis=0)`), size accumulation (`np.add.at`), vmap remapping (`np.searchsorted`); tolerance-quantized `np.unique` dedup instead of per-point KDTree loop with hard-coded `1e-5` | `remesh.py:217-228, 243-254, 268-285, 384-407` |
| `filter_by_mass`: one `getMass` per tag | `_mesh_entity.py:282-294` |
| `plot2D` via `PolyCollection`/`tripcolor` per physical group; dedupe `plot3D` vertices | `visualization.py:174-195, 288-302` |
| `Counter`-based exterior-boundary computation; drop redundant `same_material_interfaces` subtraction | `cad_gmsh.py:350-356` |

### WP7 — Hygiene & packaging

| Fix | Where |
|---|---|
| Convert relative-path tests to `tmp_path` (fixes repo-root litter + xdist collisions); delete existing stray outputs | e.g. `tests/test_multidimensional_cad.py:47, 83, 112`, `test_multiple_physicals.py:49`, `test_mesh_direct_size.py:44` |
| Track `uv.lock` (remove from `.gitignore`) — it pins CI Python 3.13 | `.gitignore` |
| Add `matplotlib`/`numpy` to runtime deps **or** guard imports like `plot3D` guards plotly (pick: guard, keeping deps lean) | `visualization.py:2-3`, `utils.py:149`, `pyproject.toml:19` |
| Replace `cadquery` dep with `cadquery-ocp` (code imports only `OCP`) | `pyproject.toml:19` |
| Config drift: black/mypy target 3.13; drop black/flake8/isort in favor of ruff; dedupe `jupytext`; fix `python_files` glob; move coverage flags out of default `addopts` | `pyproject.toml:36-38, 73, 116, 132-134` |
| `compare_gmsh_files`: byte-compare fast path; include first N diff lines in failure message | `utils.py:56-63, 111-113` |
| Break up `build_cohort_compound` (~390 lines) into `_propagate_arc_params` / `_build_horizontal_faces` / `_build_lateral_faces`; same treatment (milder) for `structured_post_pass` | `structured/build.py:950-1341`, `pipeline.py:153-296` |
| Simplify redundant tolerance re-check in `ModelManager` | `model.py:104-116` |

## Execution strategy — model-tiered subagent dispatch

Implementation is executed by dispatched subagents, each assigned the cheapest model
tier that can do its task reliably. The orchestrating session writes the task spec,
dispatches, and reviews every diff before commit.

- **Haiku** — mechanical, fully-specified edits with no design freedom:
  WP5 (dead-code deletion), the `cpu_count()` 5-site sweep, `is not None`
  truthiness fixes, docstring/error-message rewording, `tmp_path` test conversions,
  pyproject/config cleanup (WP7 except packaging-dep decisions).
- **Sonnet** — standard implementation + tests where the spec defines the behavior:
  most WP1/WP2 crash-path and input-normalization fixes (polyline zero tags,
  polysurface hole guard, visualization guards, `resolution_specs` default,
  GDS import), WP3 dedup hoisting, WP6 vectorization (remesh numpy, GEOS
  accumulation, `PolyCollection`), `Counter`-based exterior boundaries,
  `filter_by_mass`, utils diff messages.
- **Opus** — topology/geometry-critical or cross-cutting semantic changes where a
  wrong-but-plausible implementation would pass shallow review:
  closed-arc split unification (WP1), tie-policy + `A___A` backend alignment and
  reference-mesh regeneration (WP2), shape-identity migration to
  `IndexedMapOfShape` (WP4), wedge error-ordering/placeholder-triangle work (WP4),
  structured tolerance/z-plane key discipline (WP4), `build_cohort_compound`
  decomposition (WP7), magic-string → explicit-flag migration (WP4).

Per-task rules: each dispatched task carries the exact file/line targets and
acceptance tests from this spec; test-writing for a fix may go to a cheaper tier
than the fix itself; any subagent that reports ambiguity escalates back to the
orchestrator instead of guessing. Baseline test failures are recorded once and
included in every task brief so subagents don't chase pre-existing breakage.

## Testing strategy

- Record the exact baseline failures (3 stable + 1 flaky) before WP1; every WP must
  introduce **no new failures**.
- TDD for every behavior change: failing test first (e.g. disc/annulus conformality
  test for the arc-split fix; tie-overlap ownership test pinning "earlier entity
  wins" in both backends; `additive=True` raises; hierarchical-GDS fixture;
  3D-meshio remesh size-field test).
- Backend-equivalence suite (`test_backend_*`) is the guard for WP2; it must pass
  identically for both backends after tie/`A___A` alignment.
- Perf changes (WP6) are covered by existing behavior tests; no benchmark gate, but
  spot-check `test_performance_cad_occ.py`.

## Explicitly deferred

- All of `quality.py` (own PR: meshio-based parsing, group attribution, gradation
  index-alignment, vectorization, `warnings.filterwarnings` removal, `main()` argv).
- Implementing real `additive` fuse semantics and periodic boundaries (removed APIs
  can return as designed features later).
- `structured` closed-ring detection consolidation beyond the shared split helper if
  it balloons (WP1 keeps it minimal; WP3's `quantize_key` covers the root cause).

## Risks

- **`A___A` decision is provisional** — if any downstream workflow consumes those
  groups, flip to the `emit_self_interfaces` flag alternative before WP2 lands.
- Tie-cut alignment changes gmsh-backend face topology for overlapping same-order
  entities; reference meshes in `tests/` may need regeneration
  (`tests/generate_references.py` exists for this).
- Shape-identity migration (hash → `IndexedMapOfShape`) touches the XAO writer's
  core bookkeeping; land it as its own commit with the full XAO test set.
