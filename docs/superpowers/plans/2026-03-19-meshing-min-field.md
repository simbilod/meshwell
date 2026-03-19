# Meshing Min Field Consolidation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the number of sub-fields attached to the global GMSH `Min` field by grouping identical constant resolution specs.

**Architecture:** In `_apply_entity_refinement`, instead of immediately creating a `MathEval`+`Restrict` field for every entity, we will intercept `ConstantInField` generation. We'll group entities by the `resolution` value, create ONE `MathEval` field per unique size, and ONE `Restrict` field per unique size applied to all associated entity tags.

**Tech Stack:** Python, GMSH fields.

---

### Task 0: Generate Reference Results

**Files:**
- Script: `tests/generate_references.sh`

- [ ] **Step 1: Run reference generation**
Run the reference generation script in an isolated shell from the `tests` directory to compute baseline results for regressions. Make sure `PYTHONPATH` is empty to avoid numpy errors, and that the reference commit is `7d1bdb7844ae85855cdfa5d7ac5a742964e89cfa` in the script.
```bash
(cd tests && export PYTHONPATH="" && bash generate_references.sh)
```

### Task 1: Refactor Resolution Field Aggregation

**Files:**
- Modify: `meshwell/mesh.py`
- Modify: `meshwell/labeledentity.py`
- Modify: `meshwell/resolution.py`

- [ ] **Step 1: Update `ResolutionSpec.apply` signature**
Instead of `apply` instantly creating GMSH fields, we need it to return a description of the field if it is a `ConstantInField`, or create it directly if it's dynamic. Or, handle grouping directly inside `mesh.py`. Let's do it in `mesh.py`.
- [ ] **Step 2: Accumulate Constant Requirements in `mesh.py`**
In `_apply_entity_refinement`, collect `ConstantInField` requests into a dictionary: `constant_sizes[resolution][entity_str].extend(tags)`.
- [ ] **Step 3: Generate Grouped Fields**
After iterating over all entities, iterate over `constant_sizes`. For each `resolution`, create one `MathEval` field, and one `Restrict` field, supplying the consolidated list of tags. Append this field index to `refinement_field_indices`.
- [ ] **Step 4: Run Tests**
Run: `pytest tests/test_resolution.py tests/test_mesh_direct_size.py`
Expected: PASS
- [ ] **Step 5: Verify Field Count**
Optionally add a debug print or test assertion that checks `len(refinement_field_indices)` is bounded by the number of *unique sizes*, not the number of physical entities.
- [ ] **Step 6: Commit**
```bash
git add meshwell/mesh.py meshwell/labeledentity.py meshwell/resolution.py
git commit -m "perf: consolidate constant min fields in meshing"
```
