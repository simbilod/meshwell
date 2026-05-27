# Pre-shared TopoDS_Face Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pre-share `TopoDS_Face` between vertically-stacked sub-prisms within the same structured cohort, so BOPAlgo's pave-filler can recognize shared TShapes and skip pairwise intersection work at the interfaces. Sub-prisms remain individual BOPAlgo arguments — `Modified()` history is preserved.

**Architecture:** The phantom builder iterates sub-prisms grouped by `(component_index, polygon_key)` in bottom-up `zlo` order. Each prism after the first in a vertical stack reuses the previous prism's `LastShape()` as its bottom face. No changes to cad_occ, `OCCLabeledEntity`, or `_group_phantom_solids_by_entity`. Single new field on `Slab`.

**Tech Stack:** Python, OCP (OpenCascade Python bindings), pytest. Key OCP class: `BRepPrimAPI_MakePrism` (its `LastShape()` is the pre-sharing handle).

**Spec:** `docs/superpowers/specs/2026-05-27-cad-occ-cohort-preshared-faces-design.md`

---

## File Structure

**Created:**
- `tests/test_phantom_preshared_faces.py` — end-to-end parity test with kill-switch toggle

**Modified:**
- `meshwell/structured/spec.py` — add `Slab.component_index: int = 0`
- `meshwell/structured/plan.py` — populate `Slab.component_index` in `build_plan` via new helper `_assign_component_indices`
- `meshwell/structured/phantom.py` — add `_PRESHARE_VERTICAL_FACES` constant, `_group_slabs_into_vertical_stacks` helper, `bottom_face_override` parameter on `_build_sub_prism`, refactored `build_phantom_shapes`

**Renamed:**
- `tests/test_cad_occ_cohort_sewing.py` → `tests/test_cad_occ_cohort_preshared_faces.py` (file already validated by smoke tests; rename only)

---

## Task 1: Persist `component_index` on `Slab`

**Files:**
- Modify: `meshwell/structured/spec.py:251-286` (`Slab`)
- Modify: `meshwell/structured/plan.py:286-324` (add `_assign_component_indices`); call from `build_plan` at line ~1547
- Test: `tests/structured/test_plan_component_index.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/structured/test_plan_component_index.py`:

```python
"""Verify Slab.component_index is populated by build_plan."""

from __future__ import annotations

import pytest
import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured.plan import build_plan


def _square(x0, y0, x1, y1):
    return shapely.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


@pytest.fixture
def three_slabs_two_components():
    """Slab A (z=0..1) face-touches Slab B (z=1..2); Slab C (z=10..11) is disjoint."""
    A = PolyPrism(
        polygons=_square(0, 0, 1, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="A",
        mesh_order=1,
        structured=True,
        resolutions=[],
    )
    B = PolyPrism(
        polygons=_square(0, 0, 1, 1),
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="B",
        mesh_order=2,
        structured=True,
        resolutions=[],
    )
    C = PolyPrism(
        polygons=_square(0, 0, 1, 1),
        buffers={10.0: 0.0, 11.0: 0.0},
        physical_name="C",
        mesh_order=3,
        structured=True,
        resolutions=[],
    )
    return [A, B, C]


def test_component_index_groups_touching_slabs(three_slabs_two_components):
    plan = build_plan(three_slabs_two_components)
    by_name = {s.physical_name[0]: s for s in plan.slabs}
    assert by_name["A"].component_index == by_name["B"].component_index
    assert by_name["C"].component_index != by_name["A"].component_index


def test_component_index_is_non_negative(three_slabs_two_components):
    plan = build_plan(three_slabs_two_components)
    for s in plan.slabs:
        assert s.component_index >= 0
```

(If the `PolyPrism(...)` constructor signature in the codebase differs from above, adjust to match — read `meshwell/polyprism.py` for the actual signature and existing structured-PolyPrism test fixtures.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/structured/test_plan_component_index.py -v --no-cov`
Expected: FAIL with `AttributeError: 'Slab' object has no attribute 'component_index'`.

- [ ] **Step 3: Add field to `Slab`**

In `meshwell/structured/spec.py:251-286`, append the new field after `face_partition_edges`:

```python
    # Populated by build_plan via _assign_component_indices. Slabs in the
    # same connected-z-component (same StackArrangement) share an index.
    # Used by the phantom builder to pre-share TopoDS_Face between
    # vertically-stacked sub-prisms (see spec
    # 2026-05-27-cad-occ-cohort-preshared-faces-design.md).
    component_index: int = 0
```

- [ ] **Step 4: Add `_assign_component_indices` helper**

In `meshwell/structured/plan.py`, after `_connected_z_components` (around line 324), add:

```python
def _assign_component_indices(slabs: list[Slab]) -> None:
    """Write Slab.component_index for each slab from _connected_z_components.

    Mutates slabs in place. Components are numbered 0..N-1 in the order
    _connected_z_components returns them.
    """
    components = _connected_z_components(slabs)
    for comp_idx, stack in enumerate(components):
        for s in stack:
            s.component_index = comp_idx
```

- [ ] **Step 5: Call it from `build_plan`**

In `meshwell/structured/plan.py:1525-1559` (`build_plan`), insert after line 1547 (`_resolve_sublevel_mesh_order(...)`) and before line 1548 (`build_stack_arrangements(...)`):

```python
    _assign_component_indices(kept_slabs)
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `python -m pytest tests/structured/test_plan_component_index.py -v --no-cov`
Expected: PASS.

- [ ] **Step 7: Run the structured test suite**

Run: `python -m pytest tests/structured/ -v --no-cov`
Expected: PASS. The new default `component_index = 0` must not break any existing structured test.

- [ ] **Step 8: Commit**

```bash
git add meshwell/structured/spec.py meshwell/structured/plan.py tests/structured/test_plan_component_index.py
git commit -m "feat(structured): persist component_index on Slab"
```

(Repo uses pre-commit hooks; accept formatting fixes if they appear.)

---

## Task 2: `_group_slabs_into_vertical_stacks` helper

**Files:**
- Modify: `meshwell/structured/phantom.py` (add helper near top of file, after imports)
- Test: `tests/structured/test_phantom_vertical_stacks.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/structured/test_phantom_vertical_stacks.py`:

```python
"""_group_slabs_into_vertical_stacks: groups stacked pieces, separates by gap."""

from __future__ import annotations

import pytest
import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured.phantom import _group_slabs_into_vertical_stacks
from meshwell.structured.plan import build_plan


def _square(x0, y0, x1, y1):
    return shapely.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _polyprism(name, z0, z1, mesh_order):
    return PolyPrism(
        polygons=_square(0, 0, 1, 1),
        buffers={float(z0): 0.0, float(z1): 0.0},
        physical_name=name,
        mesh_order=mesh_order,
        structured=True,
        resolutions=[],
    )


def test_vertical_stack_groups_touching_slabs_same_xy():
    """A (z=0..1) and B (z=1..2) same XY -> one stack of length 2."""
    plan = build_plan([_polyprism("A", 0, 1, 1), _polyprism("B", 1, 2, 2)])
    stacks = _group_slabs_into_vertical_stacks(plan)
    multi = [s for s in stacks if len(s) > 1]
    assert len(multi) == 1
    assert len(multi[0]) == 2
    # Stack should be ascending zlo.
    zlos = [slab.zlo for slab, _piece_idx in multi[0]]
    assert zlos == sorted(zlos)


def test_gap_separates_stacks():
    """A (z=0..1) and C (z=10..11): gap >= _Z_TOL -> two separate stacks of length 1."""
    plan = build_plan([_polyprism("A", 0, 1, 1), _polyprism("C", 10, 11, 2)])
    stacks = _group_slabs_into_vertical_stacks(plan)
    # Two singletons, no multi-element stacks.
    assert all(len(s) == 1 for s in stacks)


def test_disjoint_xy_no_stack():
    """Two slabs same Z, different XY -> two singletons (no vertical stack)."""
    A = _polyprism("A", 0, 1, 1)
    B = PolyPrism(
        polygons=_square(5, 5, 6, 6),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="B",
        mesh_order=2,
        structured=True,
        resolutions=[],
    )
    plan = build_plan([A, B])
    stacks = _group_slabs_into_vertical_stacks(plan)
    assert all(len(s) == 1 for s in stacks)
```

- [ ] **Step 2: Run to verify failing**

Run: `python -m pytest tests/structured/test_phantom_vertical_stacks.py -v --no-cov`
Expected: FAIL with `ImportError: cannot import name '_group_slabs_into_vertical_stacks'`.

- [ ] **Step 3: Implement the helper**

In `meshwell/structured/phantom.py`, near the top of the file (after the imports, before `_polygon_face_cache_key`), add:

```python
# Tolerance for "vertically touching" — two slabs are in the same vertical
# stack iff abs(upper.zlo - lower.zhi) < _Z_TOL_VERT.
_Z_TOL_VERT = 1e-9


def _group_slabs_into_vertical_stacks(
    plan: "StructuredPlan",
) -> list[list[tuple["Slab", int]]]:
    """Group sub-prism pieces into vertical stacks for pre-shared-face construction.

    A "stack" is a sequence of (slab, piece_index) pairs such that:
      - All pairs share the same component_index (cohort).
      - All pairs have polygon_face_cache_key equality for their piece.
      - Pairs are sorted ascending by slab.zlo.
      - Adjacent pairs satisfy abs(upper.zlo - lower.zhi) < _Z_TOL_VERT.

    Singleton stacks (pieces with no z-touching neighbor) are returned as
    length-1 lists. Each (slab, piece_index) pair appears in exactly one
    stack.

    See spec 2026-05-27-cad-occ-cohort-preshared-faces-design.md.
    """
    # First: flat list of all (slab, piece_index, polygon_key) triples.
    triples: list[tuple[Slab, int, tuple]] = []
    for slab in plan.slabs:
        if not slab.face_partition:
            continue
        provenance_list = slab.face_partition_provenance
        for piece_index, piece in enumerate(slab.face_partition):
            piece_provenance: PieceProvenance | None = None
            if provenance_list is not None and piece_index < len(provenance_list):
                piece_provenance = provenance_list[piece_index]
            if piece_provenance is not None:
                key = _provenance_face_cache_key(piece_provenance)
            else:
                key = _polygon_face_cache_key(
                    orient(piece, sign=1.0),
                    slab.identify_arcs,
                    slab.min_arc_points,
                    slab.arc_tolerance,
                )
            triples.append((slab, piece_index, key))

    # Group by (component_index, polygon_key); sort each group by zlo.
    buckets: dict[tuple[int, tuple], list[tuple[Slab, int]]] = {}
    for slab, piece_index, key in triples:
        buckets.setdefault((slab.component_index, key), []).append(
            (slab, piece_index)
        )

    # Split each bucket into contiguous z-touching runs.
    stacks: list[list[tuple[Slab, int]]] = []
    for bucket in buckets.values():
        bucket.sort(key=lambda pair: pair[0].zlo)
        current: list[tuple[Slab, int]] = [bucket[0]]
        for prev, curr in zip(bucket, bucket[1:]):
            prev_slab, _prev_pi = prev
            curr_slab, _curr_pi = curr
            if abs(curr_slab.zlo - prev_slab.zhi) < _Z_TOL_VERT:
                current.append(curr)
            else:
                stacks.append(current)
                current = [curr]
        stacks.append(current)

    return stacks
```

The `orient` import should already exist near the top of the file; if not, add `from shapely.geometry.polygon import orient`. Same for `Slab` and `PieceProvenance` — they should be imported via existing structured-spec imports.

- [ ] **Step 4: Run tests to verify passing**

Run: `python -m pytest tests/structured/test_phantom_vertical_stacks.py -v --no-cov`
Expected: all three tests PASS.

- [ ] **Step 5: Run the structured test suite**

Run: `python -m pytest tests/structured/ -v --no-cov`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add meshwell/structured/phantom.py tests/structured/test_phantom_vertical_stacks.py
git commit -m "feat(structured): _group_slabs_into_vertical_stacks helper"
```

---

## Task 3: Add `bottom_face_override` parameter to `_build_sub_prism`

**Files:**
- Modify: `meshwell/structured/phantom.py:480-560` (`_build_sub_prism` signature + bottom-face branch)
- Test: `tests/structured/test_phantom_bottom_face_override.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/structured/test_phantom_bottom_face_override.py`:

```python
"""_build_sub_prism with bottom_face_override reuses the provided face's TShape."""

from __future__ import annotations

import shapely
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.gp import gp_Vec
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer

from meshwell.structured.phantom import _build_sub_prism


def _faces(shape):
    out = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        out.append(exp.Current())
        exp.Next()
    return out


def test_bottom_face_override_produces_shared_tshape():
    """Build a prism, take its top face, pass as the next prism's bottom override.
    The resulting two PhantomShape solids must share the interface face's TShape.
    """
    poly = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    # First sub-prism: z=0..1, no override (normal path).
    A_ps = _build_sub_prism(
        piece=poly,
        zlo=0.0,
        zhi=1.0,
        slab_index=0,
        piece_index=0,
    )
    # Second sub-prism: z=1..2, use A's top face as B's bottom.
    A_top = A_ps.input_faces_by_key[
        next(k for k in A_ps.input_faces_by_key if k.side == "top")
    ]
    B_ps = _build_sub_prism(
        piece=poly,
        zlo=1.0,
        zhi=2.0,
        slab_index=1,
        piece_index=0,
        bottom_face_override=A_top,
    )

    a_face_hashes = {hash(f) for f in _faces(A_ps.solid)}
    b_face_hashes = {hash(f) for f in _faces(B_ps.solid)}
    shared = a_face_hashes & b_face_hashes
    assert shared, (
        "bottom_face_override did not produce shared TShape identity "
        "between adjacent sub-prisms."
    )
```

- [ ] **Step 2: Run to verify failing**

Run: `python -m pytest tests/structured/test_phantom_bottom_face_override.py -v --no-cov`
Expected: FAIL with `TypeError: _build_sub_prism() got an unexpected keyword argument 'bottom_face_override'`.

- [ ] **Step 3: Add the parameter**

In `meshwell/structured/phantom.py:480-491`, modify the `_build_sub_prism` signature:

```python
def _build_sub_prism(
    piece: Polygon,
    zlo: float,
    zhi: float,
    slab_index: int = 0,
    piece_index: int = 0,
    identify_arcs: bool = False,
    min_arc_points: int = 4,
    arc_tolerance: float = 1e-3,
    provenance: PieceProvenance | None = None,
    face_cache: dict | None = None,
    bottom_face_override: Any | None = None,
) -> PhantomShape:
```

Then in the bottom-face branch (lines 519-556), add a short-circuit at the top:

```python
    if bottom_face_override is not None:
        bottom_face = bottom_face_override
    elif face_cache is not None:
        # ... existing cache path (unchanged) ...
    elif provenance is not None:
        # ... existing provenance path (unchanged) ...
    else:
        # ... existing fallback path (unchanged) ...
```

The rest of `_build_sub_prism` (prism construction, edge enumeration, lateral face capture) is unchanged — it operates on `bottom_face` regardless of where the object came from.

Add a one-line comment above the new branch:

```python
    # When pre-shared by the caller (e.g., vertically-stacked cohort),
    # reuse the provided face directly. Per spec
    # 2026-05-27-cad-occ-cohort-preshared-faces-design.md.
```

- [ ] **Step 4: Run the override test to verify passing**

Run: `python -m pytest tests/structured/test_phantom_bottom_face_override.py -v --no-cov`
Expected: PASS.

- [ ] **Step 5: Run the full phantom test suite**

Run: `python -m pytest tests/structured/ -v --no-cov`
Expected: PASS — the new parameter is opt-in; existing code paths unchanged.

- [ ] **Step 6: Commit**

```bash
git add meshwell/structured/phantom.py tests/structured/test_phantom_bottom_face_override.py
git commit -m "feat(structured): _build_sub_prism accepts bottom_face_override"
```

---

## Task 4: Wire `build_phantom_shapes` to use vertical stacks with pre-sharing

**Files:**
- Modify: `meshwell/structured/phantom.py:818-858` (`build_phantom_shapes`)
- Modify: same file — add `_PRESHARE_VERTICAL_FACES = True` module-level constant for kill-switch
- Test: `tests/structured/test_phantom_preshared_faces_integration.py` (new — verifies sharing actually happens in the full pipeline)

- [ ] **Step 1: Write the failing integration test**

Create `tests/structured/test_phantom_preshared_faces_integration.py`:

```python
"""End-to-end: build_phantom_shapes produces shared TShapes between vertically stacked sub-prisms."""

from __future__ import annotations

import shapely
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer

from meshwell.polyprism import PolyPrism
from meshwell.structured.phantom import build_phantom_shapes
from meshwell.structured.plan import build_plan


def _square(x0, y0, x1, y1):
    return shapely.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _faces(shape):
    out = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        out.append(exp.Current())
        exp.Next()
    return out


def test_stacked_sub_prisms_share_interface_face_tshape():
    """Two vertically-stacked PolyPrisms (same XY, touching in z) -> shared TShape."""
    A = PolyPrism(
        polygons=_square(0, 0, 1, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="A",
        mesh_order=1,
        structured=True,
        resolutions=[],
    )
    B = PolyPrism(
        polygons=_square(0, 0, 1, 1),
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="B",
        mesh_order=2,
        structured=True,
        resolutions=[],
    )
    plan = build_plan([A, B])
    result = build_phantom_shapes(plan)
    # Find the two PhantomShapes; one from slab A, one from slab B.
    by_slab = {ps.slab_index: ps for ps in result.shapes}
    assert len(by_slab) == 2
    a_faces = {hash(f) for f in _faces(by_slab[0].solid)}
    b_faces = {hash(f) for f in _faces(by_slab[1].solid)}
    assert a_faces & b_faces, (
        "Vertically-stacked sub-prisms do NOT share interface face TShape "
        "after build_phantom_shapes — pre-sharing not active."
    )
```

- [ ] **Step 2: Run to verify failing**

Run: `python -m pytest tests/structured/test_phantom_preshared_faces_integration.py -v --no-cov`
Expected: FAIL on the `a_faces & b_faces` assertion — the build loop isn't pre-sharing yet.

- [ ] **Step 3: Add the kill-switch constant**

In `meshwell/structured/phantom.py`, near the top (after imports, before `_polygon_face_cache_key`), add:

```python
# Kill-switch for the vertical-stack face pre-sharing optimization.
# When False, build_phantom_shapes builds every sub-prism independently
# (legacy behavior). Used by tests to compare against baseline; default True.
_PRESHARE_VERTICAL_FACES = True
```

- [ ] **Step 4: Refactor `build_phantom_shapes`**

Replace the body of `build_phantom_shapes` in `meshwell/structured/phantom.py:818-858` with:

```python
def build_phantom_shapes(plan: StructuredPlan) -> PhantomBuildResult:
    """For each slab, build one OCP sub-prism per partition piece.

    With _PRESHARE_VERTICAL_FACES=True (the default), vertically-stacked
    pieces in the same cohort share their interface TopoDS_Face: each
    upper sub-prism reuses the prism below's LastShape() as its bottom
    face. This produces shared TShape identity at the interface so
    BOPAlgo's pave-filler can skip the heavy intersection there.

    Returns a PhantomBuildResult with shapes in (slab_index, piece_index)
    ascending order for deterministic downstream processing.
    """
    face_cache: dict = {}

    # Map (slab_index, piece_index) -> PhantomShape, populated below.
    out: dict[tuple[int, int], PhantomShape] = {}

    if _PRESHARE_VERTICAL_FACES:
        stacks = _group_slabs_into_vertical_stacks(plan)
    else:
        # Legacy: every piece is its own length-1 "stack" in slab order.
        stacks = []
        for slab in plan.slabs:
            if not slab.face_partition:
                continue
            for piece_index in range(len(slab.face_partition)):
                stacks.append([(slab, piece_index)])

    for stack in stacks:
        prev_top_face: Any | None = None
        for slab, piece_index in stack:
            slab_index = plan.slabs.index(slab)
            piece = slab.face_partition[piece_index]
            provenance_list = slab.face_partition_provenance
            piece_provenance: PieceProvenance | None = None
            if provenance_list is not None and piece_index < len(provenance_list):
                piece_provenance = provenance_list[piece_index]

            ps = _build_sub_prism(
                piece=piece,
                zlo=slab.zlo,
                zhi=slab.zhi,
                slab_index=slab_index,
                piece_index=piece_index,
                identify_arcs=slab.identify_arcs,
                min_arc_points=slab.min_arc_points,
                arc_tolerance=slab.arc_tolerance,
                provenance=piece_provenance,
                face_cache=face_cache,
                bottom_face_override=prev_top_face,
            )
            out[(slab_index, piece_index)] = ps

            # Top face for next iteration: pull from input_faces_by_key.
            top_key = FaceKey(slab_index=slab_index, side="top", piece_index=piece_index)
            prev_top_face = ps.input_faces_by_key[top_key]

    # Re-sort to (slab_index, piece_index) ascending for deterministic output.
    shapes = [out[k] for k in sorted(out.keys())]
    return PhantomBuildResult(shapes=tuple(shapes))
```

Notes:
- `plan.slabs.index(slab)` is O(N) per call; if `plan.slabs` is large (>1000), build an `{id(slab): index}` lookup once before the loop. For correctness in a first cut, keep it simple.
- The `if not slab.face_partition: continue` guard is implicit because `_group_slabs_into_vertical_stacks` skips slabs with empty `face_partition`.

- [ ] **Step 5: Run the integration test**

Run: `python -m pytest tests/structured/test_phantom_preshared_faces_integration.py -v --no-cov`
Expected: PASS.

- [ ] **Step 6: Run the full structured test suite**

Run: `python -m pytest tests/structured/ -v --no-cov`
Expected: PASS. Pay attention to any test that:
- Checks per-slab face/edge counts (shared faces may change counts).
- Asserts specific TShape identities.
- Runs the full pipeline and compares mesh output.

If a structured test fails: it's likely because pre-sharing changed observable face/edge identity. Investigate before committing — the spec's premise is that downstream tagging accommodates shared TShapes (because `extract_phantom_map` iterates per-key, calling `Modified()` independently). If a test legitimately needs to be updated for shared-TShape reality, do so. If a test catches a real correctness regression, **stop and report**.

- [ ] **Step 7: Run cad_occ + integration tests**

Run: `python -m pytest tests/test_cad_occ.py tests/test_cad_occ_polyprism_overlap_fastpath.py tests/test_cad_occ_same_name_fuse.py tests/test_cad_occ_cohort_sewing.py -v --no-cov`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add meshwell/structured/phantom.py tests/structured/test_phantom_preshared_faces_integration.py
git commit -m "feat(structured): pre-share TopoDS_Face between vertically-stacked sub-prisms"
```

---

## Task 5: Rename validation-gate test file

**Files:**
- Rename: `tests/test_cad_occ_cohort_sewing.py` → `tests/test_cad_occ_cohort_preshared_faces.py`
- Update the module docstring at the top to reflect the new (already implemented) design

- [ ] **Step 1: Rename the file**

```bash
git mv tests/test_cad_occ_cohort_sewing.py tests/test_cad_occ_cohort_preshared_faces.py
```

- [ ] **Step 2: Update the module docstring**

In `tests/test_cad_occ_cohort_preshared_faces.py`, replace the docstring at lines 1-17 with:

```python
"""Smoke tests for the pre-shared-face cohort BOP design.

Validates the two OCC behaviors the design depends on:
1. BRepPrimAPI_MakePrism's LastShape() can be reused as the next prism's
   bottom face, producing genuine shared TopoDS_Face TShape identity.
2. BOPAlgo_Builder.Modified() works per-argument even when arguments
   share an interface face TShape — so per-piece history is intact.

These tests are independent of meshwell internals so they run before any
meshwell change ships.

Spec: docs/superpowers/specs/2026-05-27-cad-occ-cohort-preshared-faces-design.md
"""
```

- [ ] **Step 3: Run renamed tests**

Run: `python -m pytest tests/test_cad_occ_cohort_preshared_faces.py -v --no-cov`
Expected: all three tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_cad_occ_cohort_preshared_faces.py
git commit -m "chore(tests): rename cohort-sewing test file to reflect preshared-faces design"
```

---

## Task 6: End-to-end parity test

**Files:**
- Create: `tests/test_phantom_preshared_faces.py`

- [ ] **Step 1: Write the parity test**

Create `tests/test_phantom_preshared_faces.py`:

```python
"""Parity: full pipeline output with preshared-faces ON vs OFF.

The structured pipeline output must be invariant under the pre-sharing
optimization — same entity piece counts, same physical names, same
interface tagging structure.
"""

from __future__ import annotations

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.polysurface import PolySurface
from meshwell.structured import phantom as phantom_mod
from meshwell.orchestrator import build  # adjust import to actual entry point


def _square(x0, y0, x1, y1):
    return shapely.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _build_scene():
    """Three vertically-stacked structured slabs + one unstructured neighbor."""
    A = PolyPrism(
        polygons=_square(0, 0, 1, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="A",
        mesh_order=1,
        structured=True,
        resolutions=[],
    )
    B = PolyPrism(
        polygons=_square(0, 0, 1, 1),
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="B",
        mesh_order=2,
        structured=True,
        resolutions=[],
    )
    C = PolyPrism(
        polygons=_square(0, 0, 1, 1),
        buffers={2.0: 0.0, 3.0: 0.0},
        physical_name="C",
        mesh_order=3,
        structured=True,
        resolutions=[],
    )
    D = PolyPrism(
        polygons=_square(0.5, 0.5, 2.0, 2.0),
        buffers={0.5: 0.0, 1.5: 0.0},
        physical_name="D",
        mesh_order=4,
        structured=False,
        resolutions=[],
    )
    return [A, B, C, D]


def _entity_signature(occ_entities):
    """Build a comparable signature of entity output: name -> (piece count, dim)."""
    return {
        ent.physical_name: (len(ent.shapes), ent.dim)
        for ent in occ_entities
    }


def test_preshared_faces_off_vs_on_produces_identical_signature(tmp_path):
    entities = _build_scene()

    phantom_mod._PRESHARE_VERTICAL_FACES = False
    try:
        baseline_entities = build(entities, output_dir=tmp_path / "baseline")
        baseline_sig = _entity_signature(baseline_entities)
    finally:
        phantom_mod._PRESHARE_VERTICAL_FACES = True

    new_entities = build(entities, output_dir=tmp_path / "new")
    new_sig = _entity_signature(new_entities)

    assert baseline_sig == new_sig, (
        f"Pipeline output signature differs between pre-sharing OFF and ON.\n"
        f"  baseline: {baseline_sig}\n"
        f"  new:      {new_sig}"
    )
```

**Important:** the `from meshwell.orchestrator import build` call is a stand-in. The implementer must replace it with the project's actual full-pipeline entry point. Look at how `tests/test_cad_occ_phantom_hook.py` or other integration tests drive the pipeline. The goal: run the full path including cad_occ and capture per-entity output, then assert pre-sharing doesn't change the observable result.

If the project doesn't expose a single-function entry point, use the canonical `cad_occ(...)` + structured-orchestration pattern from existing tests.

- [ ] **Step 2: Wire the test to the actual pipeline entry point**

Read `tests/test_cad_occ_phantom_hook.py` and `meshwell/orchestrator.py` to find the canonical full-pipeline call. Adapt the test so it produces a real `occ_entities` (or equivalent) list both times.

- [ ] **Step 3: Run the parity test**

Run: `python -m pytest tests/test_phantom_preshared_faces.py -v --no-cov`
Expected: PASS.

If FAIL: signatures differ → either the pre-sharing changed observable behavior (correctness bug — investigate) OR the signature isn't capturing the right invariant (test bug — refine `_entity_signature`).

- [ ] **Step 4: Commit**

```bash
git add tests/test_phantom_preshared_faces.py
git commit -m "test(structured): parity test for preshared-faces toggle"
```

---

## Task 7: Performance measurement

**Files:**
- Create: `scripts/bench_preshared_faces.py`

- [ ] **Step 1: Write the benchmark**

Create `scripts/bench_preshared_faces.py`:

```python
"""Compare _fragment_all wall time with pre-shared faces ON vs OFF.

Constructs a vertically-stacked structured scene representative of the
production cost surface, then runs the full pipeline twice (OFF/ON),
reporting the wall-clock delta.
"""

from __future__ import annotations

import time

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import phantom as phantom_mod


def _square(x0, y0, x1, y1):
    return shapely.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _build_scene(n_stacks: int = 4, layers_per_stack: int = 10):
    """N_STACKS lateral stacks * LAYERS_PER_STACK vertical layers each."""
    entities = []
    for stack_idx in range(n_stacks):
        x0 = stack_idx * 2.0
        x1 = x0 + 1.0
        for layer_idx in range(layers_per_stack):
            z0 = float(layer_idx)
            z1 = z0 + 1.0
            entities.append(
                PolyPrism(
                    polygons=_square(x0, 0, x1, 1),
                    buffers={z0: 0.0, z1: 0.0},
                    physical_name=f"s{stack_idx}_l{layer_idx}",
                    mesh_order=stack_idx * 100 + layer_idx,
                    structured=True,
                    resolutions=[],
                )
            )
    return entities


def _time_pipeline(entities, preshared: bool) -> float:
    phantom_mod._PRESHARE_VERTICAL_FACES = preshared
    # Replace with actual full-pipeline call as in Task 6.
    from meshwell.orchestrator import build

    t0 = time.perf_counter()
    build(entities, output_dir="/tmp/bench_preshared")
    t1 = time.perf_counter()
    return t1 - t0


if __name__ == "__main__":
    entities = _build_scene()
    off = _time_pipeline(entities, preshared=False)
    on = _time_pipeline(entities, preshared=True)
    print(f"Pre-sharing OFF: {off:.2f}s")
    print(f"Pre-sharing ON:  {on:.2f}s")
    print(f"Speedup:         {off / on:.2f}x")
```

(Adapt the `build` import to the actual pipeline entry point used in Task 6.)

- [ ] **Step 2: Run the benchmark**

Run: `python scripts/bench_preshared_faces.py`

Expected: ≥3× speedup on `_fragment_all` for the structured-heavy scene. If less:
- Confirm pre-sharing is actually being applied (add a debug print in `build_phantom_shapes` and check shared-face count).
- Profile to see whether BOPAlgo time actually decreased; if not, the pave-filler may not be skipping work despite shared TShapes. In that case, the design's benefit hypothesis was wrong, and we need to either pursue lateral pre-sharing (Phase 2) or accept that this optimization is insufficient.

- [ ] **Step 3: Record results in the spec**

Append a "Measured Results" section to `docs/superpowers/specs/2026-05-27-cad-occ-cohort-preshared-faces-design.md`:

```markdown
## Measured Results (Task 7)

- Scene: <N stacks × M layers, brief description>
- Legacy (`_PRESHARE_VERTICAL_FACES=False`): <time>
- Pre-shared (`_PRESHARE_VERTICAL_FACES=True`): <time>
- Speedup: <ratio>
- Notes: <any observations about where time is actually being spent>
```

- [ ] **Step 4: Commit**

```bash
git add scripts/bench_preshared_faces.py docs/superpowers/specs/2026-05-27-cad-occ-cohort-preshared-faces-design.md
git commit -m "perf(structured): benchmark preshared-faces vertical-stack speedup"
```

---

## Wrap-up

After all tasks pass:

- [ ] Final full-suite run: `python -m pytest tests/ -v --no-cov`
- [ ] Manual sanity check on one production scene if available
- [ ] If success criterion (≥3× on `_fragment_all`) not met, file a follow-up to pursue Phase 2 (lateral face pre-sharing)
