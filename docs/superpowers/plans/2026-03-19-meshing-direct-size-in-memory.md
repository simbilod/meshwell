# Direct Size Specification In-Memory Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bypass writing large `.pos` view files to disk during `DirectSizeSpecification` applications.

**Architecture:** Use the `gmsh.view.add` and `gmsh.view.addListData` API to directly pass NumPy arrays representing the spatial point cloud into GMSH memory.

**Tech Stack:** Python, NumPy, GMSH Python API.

---

### Task 1: Migrate DirectSizeSpecification to In-Memory API

**Files:**
- Modify: `meshwell/resolution.py`
- Modify: `tests/test_mesh_direct_size.py` (ensure exists or create basic test)

- [ ] **Step 1: Replace file writing logic**
In `DirectSizeSpecification.apply`, remove the `tempfile` and `.pos` string writing code.
Instead:
```python
view_tag = gmsh.view.add("size_field")
# GMSH expects a flat list of (x, y, z, val) per point for SP (scalar point)
flattened_data = r_data.flatten().tolist()
gmsh.view.addListData(view_tag, "SP", len(r_data), flattened_data)
```
- [ ] **Step 2: Run tests**
Run: `pytest tests/test_mesh_direct_size.py` (or any existing resolution tests).
Expected: PASS
- [ ] **Step 3: Commit**
```bash
git add meshwell/resolution.py
git commit -m "perf: apply DirectSizeSpecification in-memory without pos files"
```