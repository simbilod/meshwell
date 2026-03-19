# CAD Batching Boolean Tool Operands Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce O(N^2) time complexity in CAD boolean cuts by batching tool operands into a single compound (OCC) or using list inputs (GMSH).

**Architecture:** In `cad_occ.py`, we will group previously processed shapes into a single `TopoDS_Compound` before cutting current shapes against them. In `cad_gmsh.py`, we will pass the entire list of `object_dimtags` and `tool_dimtags` into `gmsh.model.occ.cut` at once.

**Tech Stack:** Python, OCP, GMSH, pytest-benchmark (for benchmarking).

---

### Task 0: Generate Reference Results

**Files:**
- Script: `tests/generate_references.sh`

- [ ] **Step 1: Run reference generation**
Run the reference generation script in an isolated shell from the `tests` directory to compute baseline results for regressions. Make sure `PYTHONPATH` is empty to avoid numpy errors, and that the reference commit is `7d1bdb7844ae85855cdfa5d7ac5a742964e89cfa` in the script.
```bash
(cd tests && export PYTHONPATH="" && bash generate_references.sh)
```

### Task 1: Add Benchmark for CAD Operations

**Files:**
- Create: `tests/benchmarks/benchmark_cad_batching.py`

- [ ] **Step 1: Write benchmark test**
Write a benchmark that creates 50 separate `PolySurface` entities that overlap sequentially, and measures the time taken by `cad_occ` and `cad_gmsh`.
- [ ] **Step 2: Run benchmark before optimisations**
Run: `pytest tests/benchmarks/benchmark_cad_batching.py` and record the timing.
- [ ] **Step 3: Commit benchmark**
```bash
git add tests/benchmarks/benchmark_cad_batching.py
git commit -m "chore: add benchmark for cad boolean scaling"
```

### Task 2: Implement Batch Cutting in OCC Backend

**Files:**
- Modify: `meshwell/cad_occ.py`

- [ ] **Step 1: Update `_process_dimension_group_cuts_occ`**
Modify the logic so `accumulated_shapes` are combined into a single `TopoDS_Compound` before the nested loop. Instead of `for prev_shape in accumulated_shapes: cut_api(...)`, just run one cut per `ent` against the `compound_tool`.
- [ ] **Step 2: Run tests to ensure correctness**
Run: `pytest tests/test_cad_occ.py`
Expected: PASS
- [ ] **Step 3: Run benchmark**
Run: `pytest tests/benchmarks/benchmark_cad_batching.py` and verify speedup.
- [ ] **Step 4: Commit**
```bash
git add meshwell/cad_occ.py
git commit -m "perf: batch tool operands in cad_occ cuts"
```

### Task 3: Implement Batch Cutting in GMSH Backend

**Files:**
- Modify: `meshwell/cad_gmsh.py`

- [ ] **Step 1: Update `_process_dimension_group_cuts`**
Modify the loop. Separate entity instantiation from cutting. Instantiate all entities of the same group first. Then perform a single `gmsh.model.occ.cut` where `objectDimTags` is the entire current group and `toolDimTags` are the accumulated shapes. Map the results back using the `outDimTagsMap`.
- [ ] **Step 2: Update shared cache handling**
Because we separated instantiation from boolean operations, remove the `self._shared_point_cache.clear()` from inside the loop, and clear it only once at the end of the batch operation.
- [ ] **Step 3: Run tests to ensure correctness**
Run: `pytest tests/test_cad.py`
Expected: PASS
- [ ] **Step 4: Run benchmark**
Run: `pytest tests/benchmarks/benchmark_cad_batching.py` and verify speedup.
- [ ] **Step 5: Commit**
```bash
git add meshwell/cad_gmsh.py
git commit -m "perf: batch tool operands and improve caching in cad_gmsh"
```