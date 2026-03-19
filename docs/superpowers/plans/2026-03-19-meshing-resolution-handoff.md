# Meshing Resolution O(N^2) Handoff Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the O(N^2) iteration over entities when evaluating `sharing` and `not_sharing` resolution constraints.

**Architecture:** Pre-compute reverse lookup dictionaries mapping `Tag -> List[PhysicalNames]` and `Tag -> List[BoundaryTags]` in `Mesh._apply_entity_refinement` before calling `add_refinement_fields_to_model`. Replace nested iterations over `all_entities_dict.items()` with O(1) set operations based on these lookup maps.

**Tech Stack:** Python dictionaries, sets.

---

### Task 1: Add Resolution Performance Benchmark

**Files:**
- Create: `tests/benchmarks/benchmark_resolution_handoff.py`

- [ ] **Step 1: Write benchmark**
Create a test with 50-100 identical blocks side-by-side, each with its own physical name, and a `ResolutionSpec` using `sharing` pointing to all others. Measure time taken by `_apply_entity_refinement`.
- [ ] **Step 2: Commit**
```bash
git add tests/benchmarks/benchmark_resolution_handoff.py
git commit -m "chore: add benchmark for resolution handoff"
```

### Task 2: Build Reverse Indices

**Files:**
- Modify: `meshwell/mesh.py`
- Modify: `meshwell/labeledentity.py`

- [ ] **Step 1: Extract index building**
In `meshwell/mesh.py` `_apply_entity_refinement`, after recovering labels, iterate through `final_entity_list` once to build:
`tag_to_entity_names = defaultdict(list)`
`tag_to_boundaries = {}`
- [ ] **Step 2: Update `add_refinement_fields_to_model` signature**
Modify `add_refinement_fields_to_model` to accept these reverse lookup dictionaries instead of `all_entities_dict`.
- [ ] **Step 3: Refactor the sharing logic**
Inside `add_refinement_fields_to_model`, use set intersections on `tag_to_entity_names` instead of a nested loop over all entities.
- [ ] **Step 4: Run tests**
Run: `pytest tests/test_resolution.py tests/test_multiple_physicals.py`
Expected: PASS
- [ ] **Step 5: Run benchmark**
Run: `pytest tests/benchmarks/benchmark_resolution_handoff.py` to confirm time scales linearly rather than quadratically.
- [ ] **Step 6: Commit**
```bash
git add meshwell/mesh.py meshwell/labeledentity.py
git commit -m "perf: eliminate O(N^2) loop in resolution handoff using indices"
```