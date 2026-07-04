# WP5 — Dead Code Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the dead/unreachable code identified in the spec's WP5 section, with proof-of-deadness recorded before each deletion.

**Architecture:** Pure removals plus one comment resolution (polyprism `subdivide`). No behavior changes are permitted except where deleting a dead+buggy validator; every deletion is preceded by a caller grep and followed by the full suite.

**Tech Stack:** Python 3.13 / uv / pytest; pre-commit (black, ruff, codespell) — never `--no-verify`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-03-code-improvements-design.md` (WP5 section). Line numbers in the spec are stale (WP1–4 shifted files); locate by symbol name.
- Tests live in `tests/` at repo root; run `uv run pytest tests -q` from repo root. Baseline at WP5 start: **450 passed / 7 skipped** (known flake `test_backends_mesh_adjacent_3d_equivalently`: rerun in isolation before treating as regression). "No new failures" = exact count match minus any tests that pinned deleted dead code (each such test must be deleted with justification in the report, never weakened).
- Golden references (`tests/references/`) byte-identical throughout.
- **Deadness proof protocol (every deletion):** (1) `grep -rn "<symbol>" meshwell/ tests/ docs/` — record all hits; (2) classify each hit as definition / dead caller / comment; (3) if ANY live caller exists, STOP and report BLOCKED with the call site. Paste the grep output in your report.
- Deletions may orphan imports, exception classes, or helpers — chase one level (delete an import that becomes unused) but no further refactoring. The repo has a contract test (`tests/test_public_api.py::test_all_matches_live_raised_exceptions`) that fails on dead exception classes — if a deletion strands one, retire it fully (exceptions module + `__init__` export + repointed tests), as WP4 Task 7 did.

---

### Task 1: Dead methods in `_mesh_entity.py` and `mesh.py`

**Model:** haiku

**Files:**
- Modify: `meshwell/_mesh_entity.py` (delete `_fuse_self` ~line 127; delete `boundaries()` ~line 177 — the spec calls it "shadowed": verify what shadows it by grepping `def boundaries` and `\.boundaries` across meshwell/ and confirming no call resolves to this definition)
- Modify: `meshwell/mesh.py` (delete `_restore_structured_sweeps` ~line 160)
- Tests: none created; existing suite is the guard

**Interfaces:** Produces nothing new; later tasks are independent.

- [ ] **Step 1:** Run the deadness proof protocol for `_fuse_self`, `boundaries`, `_restore_structured_sweeps`. `boundaries` needs care: the name is generic — check every `.boundaries` attribute access and determine which class it resolves to.
- [ ] **Step 2:** Delete the three methods and any imports/locals they alone used.
- [ ] **Step 3:** `uv run pytest tests -q` → 450 passed / 7 skipped (minus any justified dead-test deletions).
- [ ] **Step 4:** Commit — `refactor: remove dead _fuse_self/boundaries/_restore_structured_sweeps`

### Task 2: Unreachable code in `structured/build.py`

**Model:** haiku

**Files:**
- Modify: `meshwell/structured/build.py` (duplicated `return edges` — currently literal back-to-back at ~lines 303–304, second is unreachable: delete line 304; unreachable `canon.is_closed` branch — the spec's `:605-628` is stale, the branch is now around lines 631–655: read the enclosing function, demonstrate in the report WHY the branch cannot execute (what upstream guarantees `canon.is_closed` state at that point), then delete it. If you cannot prove unreachability, STOP and report BLOCKED with your analysis — do not delete on the spec's word alone.)
- Tests: none created; structured suite (`tests/structured/`) is the guard

**Interfaces:** none.

- [ ] **Step 1:** Prove the duplicated return (trivially unreachable) and the `canon.is_closed` branch (requires flow analysis — write it out).
- [ ] **Step 2:** Delete both.
- [ ] **Step 3:** `uv run pytest tests -q` → count match.
- [ ] **Step 4:** Commit — `refactor(structured): remove unreachable is_closed branch and duplicate return`

### Task 3: `polyprism.py` — dead validator + MANUAL_NOTE resolution on `subdivide`

**Model:** sonnet

**Files:**
- Modify: `meshwell/polyprism.py` (delete `_validate_polygon_buffers` ~line 584 — dead AND buggy; the `addThruSections` error handling added in WP1 covers its intent, note that in the commit body. Resolve `# MANUAL_NOTE: delete this` at ~line 312 above `subdivide` ~line 313: the decision is KEEP the feature, DELETE the note, and simplify the numpy bbox fold inside `subdivide` (the min/max accumulation over prism bounding boxes) to idiomatic numpy without changing results.)
- Test: extend `tests/` coverage of `subdivide` ONLY if it currently has none (check first — if covered, no new test)

**Interfaces:** none.

- [ ] **Step 1:** Deadness proof for `_validate_polygon_buffers`; read `subdivide` end to end and identify the bbox fold.
- [ ] **Step 2:** If `subdivide` is untested, write a pinning test (same subdivision output before/after simplification — capture expected values from the CURRENT code, then verify they hold after).
- [ ] **Step 3:** Delete the validator; delete the note; simplify the fold (e.g. stack the per-prism bounds into an array and take `.min(axis=0)`/`.max(axis=0)` — exact form to match what the fold computes today).
- [ ] **Step 4:** `uv run pytest tests -q` → count match (+ any new pinning test).
- [ ] **Step 5:** Commit — `refactor(polyprism): drop dead _validate_polygon_buffers, resolve subdivide MANUAL_NOTE`

### Task 4: Dead `is None` cache guards in `geometry_entity.py`

**Model:** sonnet

**Files:**
- Modify: `meshwell/geometry_entity.py` (the spec names two guard sites, originally lines 348–350 and 469–474; one is `if self._lines is None: self._lines = {}` in the line-cache helper ~line 350 — find the second by grepping `is None` for lazily-initialized caches. For EACH guard: trace every constructor/assignment of the attribute; the guard is dead only if the attribute is always non-None by construction on every path that reaches the guard. If a path exists where it's None — e.g. an entity built without a shared cache — the guard is LIVE: leave it, and report the finding so the spec claim is corrected in the ledger.)
- Tests: none created; suite is the guard

**Interfaces:** none.

- [ ] **Step 1:** For each candidate guard, write the initialization trace in your report (attribute → where set → can it be None here?).
- [ ] **Step 2:** Delete only proven-dead guards; keep live ones and say so.
- [ ] **Step 3:** `uv run pytest tests -q` → count match.
- [ ] **Step 4:** Commit — `refactor(geometry_entity): remove dead cache-init guards` (only if anything was deleted; if both guards are live, no commit — report NO-OP with the traces)

---

## Final verification (orchestrator)

- [ ] `uv run pytest tests -q` fully green; `git diff --stat dfd9cde..HEAD -- tests/references/` empty.
- [ ] Whole-package review of the WP5 range before WP6.
