# Unified CAD Backend Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate GMSH and OCC CAD processing behind a unified backend protocol to allow seamless interchangeability and eliminate redundant disk I/O.

**Architecture:** We will define a `CADBackend` Protocol. We'll create `GmshBackend` and `OccBackend` that implement this protocol. We will also update `meshwell.mesh.mesh` to optionally accept an already populated `ModelManager` instead of strictly requiring an `.xao` disk path. Finally, we'll create a top-level `generate_mesh` orchestrator.

**Tech Stack:** Python 3.10+, typing.Protocol, GMSH, OCP.

---

### Task 0: Generate Reference Results

**Files:**
- Script: `tests/generate_references.sh`

- [ ] **Step 1: Run reference generation**
Run the reference generation script in an isolated shell from the `tests` directory to compute baseline results for regressions. Make sure `PYTHONPATH` is empty to avoid numpy errors, and that the reference commit is `7d1bdb7844ae85855cdfa5d7ac5a742964e89cfa` in the script.
```bash
(cd tests && export PYTHONPATH="" && bash generate_references.sh)
```

### Task 1: Define Backend Protocol

**Files:**
- Create: `meshwell/backend_protocol.py`

- [ ] **Step 1: Write the protocol definition**
Create `meshwell/backend_protocol.py` defining the `CADBackend` protocol with `process_entities`, `save_checkpoint`, and `to_gmsh_model` methods.

- [ ] **Step 2: Commit**
```bash
git add meshwell/backend_protocol.py
git commit -m "feat: define CADBackend protocol"
```

### Task 2: Implement GmshBackend

**Files:**
- Create: `meshwell/backend_gmsh.py`

- [ ] **Step 1: Write GmshBackend**
Create the `GmshBackend` class implementing the protocol by wrapping `meshwell.cad_gmsh.CAD`.

- [ ] **Step 2: Commit**
```bash
git add meshwell/backend_gmsh.py
git commit -m "feat: implement GmshBackend"
```

### Task 3: Implement OccBackend

**Files:**
- Create: `meshwell/backend_occ.py`

- [ ] **Step 1: Write OccBackend**
Create the `OccBackend` class wrapping `meshwell.cad_occ.CAD_OCC` and using `occ_to_gmsh.py` to implement `to_gmsh_model()` and `save_checkpoint()`.

- [ ] **Step 2: Commit**
```bash
git add meshwell/backend_occ.py
git commit -m "feat: implement OccBackend"
```

### Task 4: Update Mesher for In-Memory Consumption

**Files:**
- Modify: `meshwell/mesh.py`
- Test: `tests/test_mesh_in_memory.py`

- [ ] **Step 1: Write the failing test**
Create `tests/test_mesh_in_memory.py` that builds a small `ModelManager` directly and passes it to `mesh()` without `input_file`.
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_mesh_in_memory.py`
Expected: TypeError or file not found related to `input_file`.
- [ ] **Step 3: Modify `mesh.py`**
Make `input_file` optional in `mesh()` and `_initialize_model`. Skip `gmsh.merge()` if no `input_file`.
- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_mesh_in_memory.py`
- [ ] **Step 5: Commit**
```bash
git add meshwell/mesh.py tests/test_mesh_in_memory.py
git commit -m "feat: allow meshing directly from memory ModelManager"
```

### Task 5: Create Unified Orchestrator API

**Files:**
- Modify: `meshwell/__init__.py`
- Create: `meshwell/orchestrator.py`
- Test: `tests/test_unified_pipeline.py`

- [ ] **Step 1: Write orchestrator logic**
Create `meshwell/orchestrator.py` with `generate_mesh(entities, dim, output_mesh, backend="occ", checkpoint_cad=None, **mesh_kwargs)`.
- [ ] **Step 2: Write end-to-end test**
Create `tests/test_unified_pipeline.py` that meshes a simple polysurface using both `backend="gmsh"` and `backend="occ"`.
- [ ] **Step 3: Run test**
Run: `pytest tests/test_unified_pipeline.py`
- [ ] **Step 4: Expose in `__init__.py`**
Add `generate_mesh` to the package exports.
- [ ] **Step 5: Commit**
```bash
git add meshwell/orchestrator.py tests/test_unified_pipeline.py meshwell/__init__.py
git commit -m "feat: create unified generate_mesh orchestrator"
```
