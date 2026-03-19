# CAD Parallel Instantiation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parallelize the in-memory instantiation of OCC shapes (`instanciate_occ`) to speed up processing of complex shapes.

**Architecture:** Use `concurrent.futures.ThreadPoolExecutor` in `cad_occ.py` to instantiate entities concurrently before feeding them to the boolean solver.

**Tech Stack:** Python, `concurrent.futures`.

---

### Task 0: Generate Reference Results

**Files:**
- Script: `tests/generate_references.sh`

- [ ] **Step 1: Run reference generation**
Run the reference generation script in an isolated shell from the `tests` directory to compute baseline results for regressions. Make sure `PYTHONPATH` is empty to avoid numpy errors, and that the reference commit is `7d1bdb7844ae85855cdfa5d7ac5a742964e89cfa` in the script.
```bash
(cd tests && export PYTHONPATH="" && bash generate_references.sh)
```

### Task 1: Parallelize OCC Instantiation

**Files:**
- Modify: `meshwell/cad_occ.py`
- Create: `tests/benchmarks/benchmark_cad_parallel.py`

- [ ] **Step 1: Write a benchmark test**
Create a benchmark that instantiates a large number of `PolySurface` with complex holes, then run `cad_occ`. Record baseline.
- [ ] **Step 2: Modify `CAD_OCC.process_entities`**
Instead of a sequential `for i, ent_obj in enumerate(structural_entities):`, use `ThreadPoolExecutor(max_workers=self.n_threads)` to `map` the `_instantiate_entity_occ` method.
- [ ] **Step 3: Ensure thread safety**
Verify no shared state is being mutated inside `instanciate_occ`.
- [ ] **Step 4: Run functional tests**
Run: `pytest tests/test_cad_occ.py`
Expected: PASS
- [ ] **Step 5: Run benchmark**
Run: `pytest tests/benchmarks/benchmark_cad_parallel.py` and observe speedup.
- [ ] **Step 6: Commit**
```bash
git add meshwell/cad_occ.py tests/benchmarks/benchmark_cad_parallel.py
git commit -m "perf: parallelize occ entity instantiation"
```
