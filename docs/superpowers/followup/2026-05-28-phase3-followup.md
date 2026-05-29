# Phase 3 follow-up — discrete internal cohort mesh

**Date:** 2026-05-28
**Branch:** `feat/structured-clean4`
**Status at handoff:** feature-complete behind kill-switch; parity gaps remain

## What this is

Phase 3 of the cad_occ structured-cohort optimization arc. Earlier phases:
- **Phase 1** — pre-shared vertical TopoDS_Face between sub-prisms
- **Phase 2** — `cohort_topology` builder (full vertical+lateral OCC sharing)
- **Phase 3** — `cohort_envelope` builder (one OCC solid per cohort; per-piece volumes + interior interfaces become gmsh discrete entities at mesh time)

Phase 3 is gated by `_USE_DISCRETE_COHORT_MESH` in `meshwell/structured/phantom.py` (default `False`).

## Why Phase 3

Phase 2's per-piece OCC sub-prism approach kept hitting OCC tolerance + PCurves + sewing issues for non-trivial cohorts (concentric arcs, multi-piece-per-slab, etc.). The user observation that **internal cohort topology can be tagged + meshed directly rather than going through OCC** unlocked a fundamentally simpler architecture:

- One OCC solid per cohort (the **envelope**): bottom polygon + top polygon + outer perimeter lateral walls.
- Per-piece volumes: gmsh discrete 3D entities with structured elements.
- Interior interfaces (horizontal between stacked slabs, vertical between same-z pieces): gmsh discrete 2D entities with per-piece physical group names (`lower___upper`).

`cad_occ.fragment_all` sees one envelope per cohort instead of N per-piece sub-prisms → far fewer arguments to BOP, no internal sewing/PCurves to manage.

## Architecture

```
build_phantom_shapes
  └─ if _USE_DISCRETE_COHORT_MESH:
       └─ _build_phantom_shapes_via_cohort_envelope(plan)
            └─ for each cohort:
                 ├─ build_cohort_envelope(plan, cidx)  ─ outline-only topology
                 │    ├─ vertex registry (outline corners × z-planes)
                 │    ├─ horizontal edge registry (outline edges × z-planes)
                 │    ├─ vertical edge registry (deduped by (zlo,zhi,corner))
                 │    ├─ per-piece top/bottom OCC sub-faces (for FaceKey lookup)
                 │    └─ lateral wall faces (outline edges, un-subdivided)
                 └─ assemble_cohort_envelope_solid(env)
                      ├─ build bottom_union_face / top_union_face (multi-slab case)
                      ├─ skip interior arrangement edges (≥2 face count)
                      └─ BRepBuilderAPI_Sewing → multi-shell TopoDS_Solid

cad_occ.fragment_all  (one solid per cohort + unstructured neighbors)

apply_structured_mesh  (the Phase 3 hook)
  ├─ pre-pass: create discrete 2D entities for HORIZONTAL interior interfaces
  ├─ per-slab loop:
  │    ├─ stamp top/bottom OCC face mesh (structured grid)
  │    ├─ build per-piece discrete 3D entity (wedge/hex)
  │    └─ assign per-piece physical group
  ├─ _stamp_vertical_interior_interfaces  (n_layers ≥ 1)
  ├─ removeDuplicateNodes  ─ collapse coincident nodes across discrete + OCC faces
  └─ _remove_empty_cohort_envelope_volumes  ─ prune envelope OCC 3D so 3D pass doesn't tet-mesh it
```

## What's done

**23-task plan from `docs/superpowers/plans/2026-05-28-phase3-implementation-plan.md`:**

| # | Task | Commit |
|---|------|--------|
| 1 | cohort_envelope module skeleton | `b55bc8a` |
| 2 | Outline vertex registry (multi-arc snap) | `fe0e742` |
| 3 | Outline horizontal edge registry | `ca747f3`, `8bfc40f` |
| 4 | Outline vertical edge registry | `9ad679e` |
| 5 | Per-piece top/bottom OCC sub-face registry | `f9232e9`, `46d5cff` |
| 6 | Un-subdivided lateral wall registry | `eba7182` |
| 7 | Assemble cohort envelope solid | `51c965c`, `c8a491f` |
| 8 | Spec test #1 — two stacked slabs | `409aa61` |
| 9 | Spec test #2 — arc outline | `ff2611e` |
| 10 | Spec test #3 — concentric arc discs | `ff2611e` |
| — | Footprint constancy validator + xfail recovery | `9b0aebf`, `5cc556d`, `15e67fd` |
| 11 | PhantomMap.face_keys_to_discrete | `7c007d4` |
| 12 | Phase 3 phantom routing branch | `b137a56` |
| 13 | cad_occ integration smoke | `3c61124` |
| 14 | Per-piece discrete 3D entity routing | `2b886c7`, `aec38c3` |
| 15 | Horizontal interior interface stamping | `c54a4bb` |
| 16 | Vertical interior interface stamping | `e89c604`, `dec9447`, `f84f61a`, `8110d75` |
| 17 | Multi-layer vertical interfaces | `350c4bf` |
| 18 | Drop cohort envelope OCC volumes | `b233161` |
| 19 | Spec test #4 — 8 piece volumes | `9ae6c14`, `eeabf62` |
| 20 | Spec tests #5/#6 + bench scripts | `f39607c` |
| 21 | Parity sweep | `6290efd`, `4483ddc`, `bb50652`, `ccc7b4a` |

**Tasks 22-23 (kill-switch flip + cleanup) gated on parity sweep gaps below.**

## Production fixes shipped while implementing

Beyond the planned tasks, several production bugs were diagnosed and fixed:

- **Multi-arc corner snap** (`c87a0a6`) — average snap + per-vertex OCC tolerance.
- **Vertical edge dedup across slabs at same z-interval** (`ca3a361`).
- **Planner-side circle unification** for half-arcs of same logical disc (`9af8999`).
- **BRepFill::Face_s** for arc lateral faces (PCurves issue) (`2f62315`).
- **Multi-shell sewing result** (`4483ddc`) — `TopoDS::Shell` cast on a compound.
- **B4 SIGSEGV** (`ccc7b4a`) — addNodes/addElements corruption from classifying nodes on the wrong-dim entity, plus removeDuplicateNodes ordering.

## Performance

`scripts/bench_cohort_envelope.py` measures `build_phantom_shapes` across the three paths on synthetic NxN grids:

| Scene | Phase 1 | Phase 2 | Phase 3 |
|-------|---------|---------|---------|
| 4x4 × 2 z (32 pieces, 1 cohort) | 0.025s | 0.082s | 0.024s |
| 6x6 × 3 z (108 pieces, 1 cohort) | 0.064s | 0.110s | 0.063s |
| 8x8 × 4 z (256 pieces, 1 cohort) | 0.156s | 0.281s | 0.118s |

`scripts/bench_fragment_all.py` (existing benchmark, extended with Phase 3 mode) measures the actual `cad_occ.fragment_all` time.

## What's not done — parity sweep gaps

Running the full structured suite under `_USE_DISCRETE_COHORT_MESH=True` shows ~22 failures across 5 buckets, documented in `docs/superpowers/plans/2026-05-28-phase3-parity-sweep-report.md`:

### Bucket A — test assumes Phase 1+2 OCC path (7 tests)

Legitimate test invariants tied to per-piece OCC sub-prism existence (e.g. asserts on per-piece OCC entity counts, `bottom_sub_faces` structure). These don't represent production bugs — they should be either:
- Skipped under Phase 3 via `pytest.mark.skipif(_USE_DISCRETE_COHORT_MESH, ...)`
- Removed when Phase 1+2 code is deleted (Task 23)

Easy bucket. ~10 min of work.

### Bucket B1b — entity grouping for multi-shell solids (2 tests)

`_group_phantom_solids_by_entity` in `meshwell/structured/phantom.py` was patched in Task 13 to handle the synthetic `slab_index = -(cidx + 1)` cohort marker, but doesn't yet handle the case where the cohort solid spans multiple `source_index` values (e.g. `test_mixed_cohort_sharing` gets `KeyError: 'A'`). After B1's multi-shell fix, the function needs to assign the multi-shell solid to each relevant entity's `source_index` instead of dropping all but the first.

Likely 30 min — focused fix in one function.

### Bucket B2 — arc BOP face-count mismatch (5 tests)

The `1 bot + 1 top` invariant in Phase 5(d) checking (somewhere in `builder.py` post-cad_occ) is violated for arc-disc/annulus geometries where BOP splits curved faces. The check needs to either relax (allow ≥1 bot, ≥1 top) or aggregate sub-faces via arc provenance.

Likely 1-2 hours — needs investigation of the specific check + how BOP fragments curved geometries.

### Bucket B3 — interior interface non-conformality (2 tests)

`128 bad triangles` reported by the conformality assertion in stress-pattern tests. Likely the interior interface stamping doesn't correctly chain boundary nodes when a piece's bot face has more than 2 outline edges or when a vertical interface meets a horizontal interface at a T-junction.

Likely 1-3 hours — needs scene-specific diagnosis.

### Bucket B5 — phantom_to_gmsh_map routing mismatch (2 tests)

The `extract_phantom_map` walk or downstream consumer (e.g. `_map_phantom_faces_to_gmsh` in `builder.py`) returns the wrong tag list for some FaceKeys under Phase 3. Likely a routing bug in the `face_keys_to_discrete` vs `output_faces` distinction.

Likely 30-60 min.

### Residual end-of-sweep crash (1 test)

The parity sweep timed out / dumped core late in the run after the B4 fix. The exact test wasn't isolated; needs faulthandler-instrumented re-run to localize. May be a new failure mode unmasked by the B4 fix.

## Recommended ordering for follow-up

1. **Bucket A skips** — easy win, takes ~22 failures to ~15.
2. **B1b fix** — small focused fix, builds on B1's multi-shell pattern.
3. **Residual crash localization** — faulthandler-instrumented run to identify which test now segfaults.
4. **B5 routing fix** — focused fix.
5. **B3 conformality** — scene-specific debugging.
6. **B2 arc face-count** — deepest investigation.

Once parity is at "0 failures (all Bucket A skipped, all production fixes in)", proceed to:

7. **Task 22:** flip `_USE_DISCRETE_COHORT_MESH = True` as default.
8. **Task 23:** delete Phase 1+2 cohort code: `cohort_topology.py`, `_USE_COHORT_TOPOLOGY`, `_PRESHARE_VERTICAL_FACES`, `_build_phantom_shapes_via_cohort_topology`, and the related tests.

## Key files

**Phase 3 core:**
- `meshwell/structured/cohort_envelope.py` — envelope builder + assembly
- `meshwell/structured/phantom.py` — kill-switch + routing branch
- `meshwell/structured/builder.py` — mesh-stage discrete entity creation + interior interface stamping
- `meshwell/structured/spec.py` — `PhantomMap.face_keys_to_discrete` field

**Specs and plans:**
- `docs/superpowers/specs/2026-05-28-cad-occ-discrete-internal-cohort-mesh-design.md` — design spec
- `docs/superpowers/specs/2026-05-28-phase3-xao-checkpoint.md` — XAO checkpoint lossiness note
- `docs/superpowers/plans/2026-05-28-phase3-implementation-plan.md` — 23-task plan
- `docs/superpowers/plans/2026-05-28-phase3-parity-sweep-report.md` — parity sweep + B1/B4 fix updates

**Tests:**
- `tests/structured/test_phase3_*.py` — Phase 3 spec tests (all pass with kill-switch on inside test)
- `tests/structured/test_cohort_envelope_*.py` — envelope builder unit tests

**Benches:**
- `scripts/bench_cohort_envelope.py` — Phase 1/2/3 comparison
- `scripts/bench_fragment_all.py` — extended with Phase 3 mode

## Test counts

**At default (`_USE_DISCRETE_COHORT_MESH=False`):**
- 290 passed, 4 skipped, 1 xfailed (annular transfinite, separate issue tracked in MEMORY.md), 0 failures

**Under sweep (`_USE_DISCRETE_COHORT_MESH=True`):**
- ~268 passed, ~22 failed, 4 skipped, 0 xfailed (1 baseline xfail becomes SIGSEGV under Phase 3), 1 residual crash at end-of-sweep

Phase 3 tests themselves (10 files under `tests/structured/test_phase3_*.py`) all pass — they patch the flag explicitly.

## Things to watch when picking this up

1. **Footprint constancy invariant.** `build_plan` rejects cohorts whose union XY footprint changes across z-levels with `FootprintNotConstantError`. Several tests were rescued by wrapping their stepped-cohort scenes in low-priority "frame" slabs at the cohort's max footprint. New Phase 3 tests should follow this pattern.
2. **Multi-shell solids.** `BRepBuilderAPI_Sewing` can return a TopoDS_Compound of shells for disjoint XY components. Don't cast to `TopoDS_Shell` directly; iterate with `TopExp_Explorer`.
3. **Discrete entity node classification.** Adding nodes to a discrete 3D entity via `gmsh.model.mesh.addNodes(3, vol_tag, ...)` corrupts gmsh's `_mesh_vertices` indexing if done before the OCC 2D face mesh that will share those nodes. Classify shared nodes on the 2D OCC face (dim=2) first. See `ccc7b4a` for details.
4. **`removeDuplicateNodes` after volume removal.** Removing OCC 3D volumes via `removeEntities` leaves their 2D children orphaned in gmsh's model state. Calling `removeDuplicateNodes()` globally afterward traverses inconsistent state and SIGSEGVs. Either keep the volumes until after `removeDuplicateNodes`, or pass the discrete-tag list explicitly.
