# Unified CAD Boolean Logic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `CAD_GMSH` to use a single `fragment` operation per dimension instead of sequential cuts, ensuring topological robustness and respecting `mesh_order` priority.

**Architecture:**
1.  Shift `GeometryEntity` to instance-local point/line caches to ensure closed loops while avoiding cross-entity tag collisions.
2.  Implement `mesh_order` priority in `CAD_GMSH` by mapping fragment results back to original entities and selecting the owner with the lowest order.
3.  Utilize GMSH's `Geometry.OCCTolerance` for "fuzzy" geometric matching instead of coordinate snapping.

**Tech Stack:** Python, GMSH (OCC backend), Shapely.

---

### Task 1: Initialize Model-Level Tolerance

**Files:**
- Modify: `meshwell/model.py`

- [ ] **Step 1: Update `ModelManager._initialize` to set OCC tolerance**

```python
# In meshwell/model.py, inside _initialize()
        # Configure threading
        self._configure_threading()

        # Configure OCC tolerance if provided
        if self.point_tolerance is not None:
            gmsh.option.setNumber("Geometry.Tolerance", self.point_tolerance)

        self._is_initialized = True
```

- [ ] **Step 2: Commit model changes**

```bash
git add meshwell/model.py
git commit -m "feat: configure Geometry.Tolerance in ModelManager"
```

### Task 2: Refactor GeometryEntity Caching

**Files:**
- Modify: `meshwell/geometry_entity.py`

- [ ] **Step 1: Remove shared cache setters and switch to local-only initialization**

```python
# In meshwell/geometry_entity.py
# Remove _set_point_cache and _set_line_cache methods.
# Ensure __init__ initializes empty dicts for _points and _lines.
```

- [ ] **Step 2: Remove coordinate snapping from `_add_point_with_tolerance`**
Coordinate snapping is no longer needed because OCC's fuzzy tolerance handles it.

```python
    def _add_point_with_tolerance(self, x: float, y: float, z: float) -> int:
        key = (x, y, z)
        if self._points is None:
            self._points = {}
        if key not in self._points:
            self._points[key] = gmsh.model.occ.addPoint(x, y, z)
        return self._points[key]
```

- [ ] **Step 3: Commit caching changes**

```bash
git add meshwell/geometry_entity.py
git commit -m "refactor: switch to instance-local caching in GeometryEntity"
```

### Task 3: Implement Unified Fragmentation in CAD_GMSH

**Files:**
- Modify: `meshwell/cad_gmsh.py`

- [ ] **Step 1: Remove global cache state from `CAD.__init__` and `_instantiate_entity`**

- [ ] **Step 2: Reimplement `_process_dimension_group_cuts` using unified fragment**

```python
    def _process_dimension_group_cuts(
        self, entity_group: list, progress_bars: bool
    ) -> list[LabeledEntities]:
        """Process entities of same dimension using unified fragment and mesh_order selection."""
        if not entity_group:
            return []

        # 1. Instantiate all entities independently
        labeled_entities_with_objs = []
        for index, entity_obj in entity_group:
            ent = self._instantiate_entity(index, entity_obj, progress_bars)
            labeled_entities_with_objs.append((ent, entity_obj))

        all_dimtags = []
        for ent, _ in labeled_entities_with_objs:
            all_dimtags.extend(ent.dimtags)

        if not all_dimtags:
            return []

        # 2. Single fragment operation to resolve all overlaps
        fragment_result = self.model_manager.model.occ.fragment(all_dimtags, [])
        self.model_manager.model.occ.synchronize()

        if not fragment_result or len(fragment_result) < 2:
            return [ent for ent, _ in labeled_entities_with_objs if ent.dimtags]

        mapping = fragment_result[1]

        # 3. Assign each fragment to the entity with the lowest mesh_order
        piece_to_owners = {} # (dim, tag) -> list of (ent, mesh_order)
        dimtag_idx = 0
        for ent, obj in labeled_entities_with_objs:
            mo = obj.mesh_order if obj.mesh_order is not None else float("inf")
            for _ in ent.dimtags:
                for piece in mapping[dimtag_idx]:
                    if piece not in piece_to_owners:
                        piece_to_owners[piece] = []
                    piece_to_owners[piece].append((ent, mo))
                dimtag_idx += 1

        # Reset entity tags and reassign
        for ent, _ in labeled_entities_with_objs:
            ent.dimtags = []

        for piece, owners in piece_to_owners.items():
            # Find owner with minimum mesh_order
            # In case of tie, first in list (original order) wins
            best_ent = min(owners, key=lambda x: x[1])[0]
            best_ent.dimtags.append(piece)

        return [ent for ent, _ in labeled_entities_with_objs if ent.dimtags]
```

- [ ] **Step 3: Update `_process_dimension_group_fragments` for cross-dimensional consistency**
Ensure that when fragmenting lower-dim entities against higher-dim ones, we maintain the mapping and priority logic if applicable.

- [ ] **Step 4: Commit CAD changes**

```bash
git add meshwell/cad_gmsh.py
git commit -m "feat: implement unified fragmentation and hierarchical integration in CAD_GMSH"
```

### Task 4: Verification and Regression

**Files:**
- Create: `tests/test_boolean_robustness.py`
- Modify: `tests/references/*.msh`, `tests/references/*.xao` (as needed)

- [ ] **Step 1: Write robustness test reproducing the coincident vertex scenario**

- [ ] **Step 2: Run tests and identify regressions**
Run: `pytest tests/test_polysurface.py tests/test_polyline.py tests/test_cad.py tests/test_boolean_robustness.py`

- [ ] **Step 3: Update references for verified improvements**
Since the new logic is more robust and resolves the "wrong node connection" bug, update the reference files where node counts or cell counts have changed but the geometry is confirmed correct.
Run: `python tests/generate_references.py` (or manual copy if needed)

- [ ] **Step 4: Final verification run**
Run: `pytest tests/`

- [ ] **Step 5: Commit final changes**

```bash
git add tests/test_boolean_robustness.py tests/references/
git commit -m "test: update boolean robustness verification and reference baselines"
```
