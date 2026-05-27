# cad_occ Structured-Cohort Compound BOP Implementation Plan

> **STATUS: SUPERSEDED** 2026-05-27 by `2026-05-27-cad-occ-cohort-preshared-faces-plan.md`.
> Task 0 of this plan revealed the design's core assumption was wrong — `BOPAlgo_Builder.Modified()`
> returns empty for sub-shapes of a compound argument. The replacement plan implements the same
> goal (skip pairwise BOP work on planner-conformed interfaces) via pre-shared TopoDS_Face
> objects between vertically-stacked sub-prisms.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Skip pairwise `BOPAlgo_Builder` work between sub-prisms that the structured planner already conformed, by sewing each cohort into a `TopoDS_Compound` and passing it as a single BOP argument.

**Architecture:** A "cohort" is a connected z-component of structured slabs (already computed by `_connected_z_components`). We persist `component_index` on `Slab` and `PhantomShape`, propagate it via `_group_phantom_solids_by_entity` into `OCCLabeledEntity`, then teach `_fragment_all` to bucket shapes by cohort, sew each bucket (so internal interfaces share TShape identity), and add the sewn compound as one BOP argument.

**Tech Stack:** Python, OCP (OpenCascade Python bindings), pytest. Key OCP classes: `BRepBuilderAPI_Sewing`, `BOPAlgo_Builder`, `TopoDS_Compound`, `TopoDS_Builder`, `TopTools_ShapeMapHasher`.

**Spec:** `docs/superpowers/specs/2026-05-27-cad-occ-structured-cohort-compound-bop-design.md`

---

## File Structure

**Created:**
- `tests/test_cad_occ_cohort_sewing.py` — three smoke tests + parity test (the validation gate)

**Modified:**
- `meshwell/structured/spec.py` — add `component_index` to `Slab` and `PhantomShape`
- `meshwell/structured/plan.py` — populate `Slab.component_index` in `build_plan` (or `build_stack_arrangements`)
- `meshwell/structured/phantom.py` — set `PhantomShape.component_index` at construction; change `_group_phantom_solids_by_entity` return type
- `meshwell/cad_occ.py` — add `shape_cohorts` to `OCCLabeledEntity`; cohort-aware `_fragment_all`; accept cohort tuples in `_instantiate_entity_occ`
- `meshwell/orchestrator.py` — call site of `_group_phantom_solids_by_entity` (no signature change needed since output flows through unchanged interface, but verify)

---

## Task 0: Smoke test — OCC sewing + BOPAlgo compound behavior (validation gate)

**Why first:** The whole design rests on two OCC behaviors we haven't proven: (a) `BRepBuilderAPI_Sewing` produces shared TShape identity at coincident faces, and (b) `BOPAlgo_Builder.Modified(child_of_argument_compound)` returns sensible per-child pieces. If either fails, the plan must change.

**Files:**
- Create: `tests/test_cad_occ_cohort_sewing.py`

- [ ] **Step 1: Write the OCC-only smoke test (no meshwell changes yet)**

```python
"""Smoke tests for the cohort-compound BOP design.

Validates the two OCC behaviors the design depends on:
1. BRepBuilderAPI_Sewing unifies coincident faces into shared TShapes.
2. BOPAlgo_Builder.Modified() works for sub-shapes of a compound argument.

These tests are independent of meshwell internals so they can run before any
meshwell change ships.
"""

from __future__ import annotations

import pytest

OCP = pytest.importorskip("OCP")

from OCP.BOPAlgo import BOPAlgo_Builder
from OCP.BRep import BRep_Builder
from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS_Compound
from OCP.TopTools import TopTools_ShapeMapHasher


def _faces(shape):
    out = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        out.append(exp.Current())
        exp.Next()
    return out


def _hash(shape) -> int:
    return TopTools_ShapeMapHasher().HashCode(shape, 2**31 - 1)


def _compound_of(shapes):
    builder = BRep_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)
    for s in shapes:
        builder.Add(comp, s)
    return comp


def test_sewing_unifies_coincident_faces():
    # Two unit boxes sharing the face x=1 (A: [0,1]^3, B: [1,2]x[0,1]^2).
    A = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), gp_Pnt(1, 1, 1)).Solid()
    B = BRepPrimAPI_MakeBox(gp_Pnt(1, 0, 0), gp_Pnt(2, 1, 1)).Solid()

    compound = _compound_of([A, B])
    sewing = BRepBuilderAPI_Sewing(1e-7)
    sewing.Load(compound)
    sewing.Perform()
    sewed = sewing.SewedShape()

    sewed_faces = _faces(sewed)
    face_hashes = [_hash(f) for f in sewed_faces]
    # 6 + 6 - 2 (one merged from each side) = 10 unique faces if sewing worked
    # OR fewer if same-orientation faces collapse. Either way: total unique
    # TShape identities must be < 12 (sum of original face counts) to prove
    # sewing actually merged something.
    unique = len(set(face_hashes))
    assert unique < 12, (
        f"Sewing did not unify any faces: {unique} unique faces out of "
        f"{len(sewed_faces)} total (expected merging at x=1 interface)."
    )


def test_bopalgo_modified_on_sewn_compound_child():
    # A, B touch at x=1 (cohort); C overlaps A at x in [0.5, 1.5].
    A = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), gp_Pnt(1, 1, 1)).Solid()
    B = BRepPrimAPI_MakeBox(gp_Pnt(1, 0, 0), gp_Pnt(2, 1, 1)).Solid()
    C = BRepPrimAPI_MakeBox(gp_Pnt(0.5, 0.5, 0.5), gp_Pnt(1.5, 1.5, 1.5)).Solid()

    cohort_compound = _compound_of([A, B])
    sewing = BRepBuilderAPI_Sewing(1e-7)
    sewing.Load(cohort_compound)
    sewing.Perform()
    sewed_cohort = sewing.SewedShape()

    # Map A and B to their post-sewing equivalents.
    def _post_sewing(orig):
        modified = sewing.ModifiedSubShape(orig)
        return modified if not modified.IsNull() else orig

    A_sewn = _post_sewing(A)
    B_sewn = _post_sewing(B)

    builder = BOPAlgo_Builder()
    builder.AddArgument(sewed_cohort)
    builder.AddArgument(C)
    builder.Perform()

    # Modified(A_sewn) should return a non-empty TopTools list: A got
    # fragmented by C's overlap.
    a_modified = builder.Modified(A_sewn)
    assert not a_modified.IsEmpty(), (
        "BOPAlgo did not produce Modified history for A (sewn child of compound argument). "
        "The cohort-compound design depends on this behavior."
    )

    # Modified(B_sewn): B does not overlap C; either returns empty-and-not-deleted
    # (the legacy fallback at cad_occ.py:789-790 treats this as "shape passes through"),
    # OR returns [B_sewn]. Both are acceptable. NOT acceptable: IsDeleted == True.
    b_modified = builder.Modified(B_sewn)
    b_deleted = builder.IsDeleted(B_sewn)
    assert not b_deleted, (
        "BOPAlgo deleted B even though it should pass through untouched."
    )
    # Either empty (use original) or contains B_sewn.
    if not b_modified.IsEmpty():
        # If non-empty, must not be "fragmented" — should be a single piece
        # equal to B_sewn under IsSame.
        pieces = list(b_modified)
        assert len(pieces) == 1, (
            f"BOPAlgo fragmented B even though it does not overlap C: {len(pieces)} pieces."
        )


def test_internal_cohort_interface_survives_bop():
    # A and B share the face at x=1. Sew them, run BOP with a non-touching
    # neighbor D, and assert the A-B interface face appears with shared
    # TShape identity in BOTH A's and B's piece face lists.
    A = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), gp_Pnt(1, 1, 1)).Solid()
    B = BRepPrimAPI_MakeBox(gp_Pnt(1, 0, 0), gp_Pnt(2, 1, 1)).Solid()
    D = BRepPrimAPI_MakeBox(gp_Pnt(10, 10, 10), gp_Pnt(11, 11, 11)).Solid()

    cohort_compound = _compound_of([A, B])
    sewing = BRepBuilderAPI_Sewing(1e-7)
    sewing.Load(cohort_compound)
    sewing.Perform()
    sewed_cohort = sewing.SewedShape()

    def _post_sewing(orig):
        modified = sewing.ModifiedSubShape(orig)
        return modified if not modified.IsNull() else orig

    A_sewn = _post_sewing(A)
    B_sewn = _post_sewing(B)

    builder = BOPAlgo_Builder()
    builder.AddArgument(sewed_cohort)
    builder.AddArgument(D)
    builder.Perform()

    # Since neither A nor B was touched by D (D is far away), their face
    # lists should still contain the shared interface face.
    a_face_hashes = {_hash(f) for f in _faces(A_sewn)}
    b_face_hashes = {_hash(f) for f in _faces(B_sewn)}
    shared = a_face_hashes & b_face_hashes
    assert shared, (
        "After sewing, A and B do not share any TShape — the interface face was "
        "not unified. Downstream interface tagging would fail to detect the shared "
        "boundary intra-cohort."
    )
```

- [ ] **Step 2: Run the smoke tests**

Run: `pytest tests/test_cad_occ_cohort_sewing.py -v`
Expected: all three tests PASS. If any FAIL, **stop and notify** — the design needs to be revisited (the deferred Option-2 path, pre-sharing TShapes in the planner, becomes necessary).

- [ ] **Step 3: Commit the smoke tests**

```bash
git add tests/test_cad_occ_cohort_sewing.py
git commit -m "test(cad_occ): smoke tests for cohort-compound BOP design (validation gate)"
```

---

## Task 1: Add `component_index` to `Slab`

**Files:**
- Modify: `meshwell/structured/spec.py:251-286` (`Slab` dataclass)
- Test: `tests/structured/test_plan_component_index.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/structured/test_plan_component_index.py`:

```python
"""Verify Slab.component_index is populated by build_plan."""

from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from meshwell.structured.plan import build_plan


def _box(xlo, xhi, ylo, yhi):
    return Polygon([(xlo, ylo), (xhi, ylo), (xhi, yhi), (xlo, yhi)])


@pytest.fixture
def simple_structured_entities():
    """Build two touching slabs in one cohort and one disjoint slab.

    Slab A: z in [0,1] on the XY box [0,1]x[0,1]
    Slab B: z in [1,2] on the same XY box (face-touches A in z)
    Slab C: z in [10,11] on a far-away XY box (disjoint cohort)
    """
    from meshwell.polysurface import PolySurface

    # Use whatever structured-entity factory the codebase prefers; the goal
    # is to feed three slabs into build_plan and read back their component_index.
    # If PolySurface isn't structured, swap for PolyPrism / the structured factory.
    pytest.skip(
        "Wire up using the project's structured-entity test helpers — "
        "see tests/structured/conftest.py or an existing structured test for the pattern."
    )


def test_component_index_groups_touching_slabs(simple_structured_entities):
    entities = simple_structured_entities
    plan = build_plan(entities)
    # Slab A and B touch in z; should share component_index.
    # Slab C is disjoint; should have a different component_index.
    by_source = {s.source_index: s for s in plan.slabs}
    assert by_source[0].component_index == by_source[1].component_index
    assert by_source[2].component_index != by_source[0].component_index


def test_component_index_is_non_negative(simple_structured_entities):
    entities = simple_structured_entities
    plan = build_plan(entities)
    for s in plan.slabs:
        assert s.component_index >= 0
```

(The implementer should replace the `pytest.skip` with the actual structured-entity construction — look at `tests/structured/test_*.py` for the project's existing fixture pattern; e.g. `test_cad_occ_phantom_hook.py` likely shows it.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/structured/test_plan_component_index.py -v`
Expected: FAIL with `AttributeError: 'Slab' object has no attribute 'component_index'` (or SKIP if fixture not wired — wire it before proceeding).

- [ ] **Step 3: Add the field to `Slab`**

In `meshwell/structured/spec.py:251-286`, add at the end of the `Slab` dataclass (after `face_partition_edges`):

```python
    # Populated by build_plan from _connected_z_components. Slabs in the
    # same connected z-component share an index; pieces within one component
    # are guaranteed mutually conforming by the structured planner.
    # Default 0 lets tests / callers construct Slabs without specifying it;
    # build_plan always overwrites.
    component_index: int = 0
```

- [ ] **Step 4: Populate `component_index` in `build_plan`**

In `meshwell/structured/plan.py:1525-1559` (`build_plan`), insert after line 1547 (`_resolve_sublevel_mesh_order(...)`) and before line 1548 (`build_stack_arrangements(...)`):

```python
    # Tag each slab with its connected-z-component index BEFORE downstream
    # stages run. Cad_occ uses this to bucket sub-prisms into cohorts for
    # the compound-argument BOP optimization (see spec
    # 2026-05-27-cad-occ-structured-cohort-compound-bop-design.md).
    _assign_component_indices(kept_slabs)
```

Then add a helper near `_connected_z_components` (around line 286-324):

```python
def _assign_component_indices(slabs: list[Slab]) -> None:
    """Write Slab.component_index for each slab from _connected_z_components.

    Mutates slabs in place. Component indices are 0..N-1 in the order
    _connected_z_components returns them.
    """
    components = _connected_z_components(slabs)
    for comp_idx, stack in enumerate(components):
        for s in stack:
            s.component_index = comp_idx
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest tests/structured/test_plan_component_index.py -v`
Expected: PASS.

- [ ] **Step 6: Run the full structured test suite to catch regressions**

Run: `pytest tests/structured/ -v`
Expected: PASS for all previously-passing tests. New default `component_index = 0` must not break any frozen-dataclass / equality / serialization expectations.

- [ ] **Step 7: Commit**

```bash
git add meshwell/structured/spec.py meshwell/structured/plan.py tests/structured/test_plan_component_index.py
git commit -m "feat(structured): tag Slab with component_index from _connected_z_components"
```

---

## Task 2: Add `component_index` to `PhantomShape`

**Files:**
- Modify: `meshwell/structured/spec.py:362-380` (`PhantomShape`)
- Modify: `meshwell/structured/phantom.py:776-784` (`_build_sub_prism` return statement)
- Test: extend `tests/structured/test_plan_component_index.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/structured/test_plan_component_index.py`:

```python
def test_phantom_shapes_inherit_slab_component_index(simple_structured_entities):
    from meshwell.structured.phantom import build_phantom_shapes

    entities = simple_structured_entities
    plan = build_plan(entities)
    result = build_phantom_shapes(plan)
    for shape in result.shapes:
        slab = plan.slabs[shape.slab_index]
        assert shape.component_index == slab.component_index
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/structured/test_plan_component_index.py::test_phantom_shapes_inherit_slab_component_index -v`
Expected: FAIL with `AttributeError: 'PhantomShape' object has no attribute 'component_index'`.

- [ ] **Step 3: Add the field to `PhantomShape`**

In `meshwell/structured/spec.py:362-380`, add after `slab_index: int` (line 376):

```python
    slab_index: int
    piece_index: int
    # Component index of the slab this phantom came from; lets cad_occ
    # bucket sub-prisms into cohorts without re-deriving the grouping.
    component_index: int
    solid: Any
```

(Re-order so `component_index` sits next to `slab_index` for readability; reorder other fields if needed to maintain valid dataclass semantics — fields without defaults must come before any with defaults.)

- [ ] **Step 4: Populate `component_index` in `_build_sub_prism`**

In `meshwell/structured/phantom.py:776-784`, change the `PhantomShape(...)` constructor call to include:

```python
    return PhantomShape(
        slab_index=slab_index,
        piece_index=piece_index,
        component_index=component_index,
        solid=solid,
        input_faces_by_key=input_faces,
        input_edges_by_key=input_edges,
        input_vertices_by_key=input_vertices,
        input_laterals_by_outer_edge=input_laterals,
    )
```

Then thread `component_index` into `_build_sub_prism`'s signature. Find `_build_sub_prism`'s `def` line (search for `def _build_sub_prism`) and add the parameter. At each call site, pass `slab.component_index`. Search for `_build_sub_prism(` to find call sites:

```bash
grep -n "_build_sub_prism(" meshwell/structured/phantom.py
```

For each call site, ensure the caller has access to the slab and passes `slab.component_index`.

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest tests/structured/test_plan_component_index.py -v`
Expected: all three tests PASS.

- [ ] **Step 6: Run the full structured test suite**

Run: `pytest tests/structured/ -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add meshwell/structured/spec.py meshwell/structured/phantom.py tests/structured/test_plan_component_index.py
git commit -m "feat(structured): propagate component_index from Slab to PhantomShape"
```

---

## Task 3: Change `_group_phantom_solids_by_entity` return type

**Files:**
- Modify: `meshwell/structured/phantom.py:787-815`
- Test: `tests/structured/test_group_phantom_solids_by_entity.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/structured/test_group_phantom_solids_by_entity.py`:

```python
"""_group_phantom_solids_by_entity now emits (solid, component_index) tuples."""

from __future__ import annotations

import pytest

from meshwell.structured.phantom import (
    _group_phantom_solids_by_entity,
    build_phantom_shapes,
)
from meshwell.structured.plan import build_plan


@pytest.fixture
def simple_structured_entities():
    # Same skip pattern as Task 1; implementer must wire to project fixtures.
    pytest.skip("Wire up structured-entity fixture.")


def test_group_returns_tuples_with_component_index(simple_structured_entities):
    entities = simple_structured_entities
    plan = build_plan(entities)
    result = build_phantom_shapes(plan)
    grouped = _group_phantom_solids_by_entity(plan, result)
    # Every entry is list[tuple[Any, int]].
    for src_idx, entries in grouped.items():
        for entry in entries:
            assert isinstance(entry, tuple), (
                f"entity {src_idx} entry is not a tuple: {entry!r}"
            )
            assert len(entry) == 2, (
                f"entity {src_idx} entry is not 2-tuple: {entry!r}"
            )
            _solid, cohort = entry
            assert isinstance(cohort, int)
            assert cohort >= 0


def test_carved_out_entities_still_get_empty_list(simple_structured_entities):
    # Construct a scenario where some entity has every slab carved out
    # (resolved_footprint empty); _group_phantom_solids_by_entity must still
    # produce an empty list entry for it. Preserve legacy semantics.
    pytest.skip("Add a carve-out scenario fixture or borrow from existing tests.")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/structured/test_group_phantom_solids_by_entity.py -v`
Expected: FAIL on `isinstance(entry, tuple)` (currently entries are raw solids).

- [ ] **Step 3: Update `_group_phantom_solids_by_entity`**

Replace `meshwell/structured/phantom.py:787-815` with:

```python
def _group_phantom_solids_by_entity(
    plan: StructuredPlan,
    phantom_result: PhantomBuildResult,
) -> dict[int, list[tuple[Any, int]]]:
    """Group phantom solids by source entity, tagged with their cohort.

    Returns ``source_index -> [(solid, component_index)]`` in
    ``(slab_index, piece_index)`` ascending order — the same order
    ``build_phantom_shapes`` populates ``phantom_result.shapes``.

    The ``component_index`` is the connected-z-component the slab belongs
    to, used by cad_occ to bucket sub-prisms into cohorts for the
    cohort-compound BOP optimization (see spec
    2026-05-27-cad-occ-structured-cohort-compound-bop-design.md).

    Structured entities whose every slab is fully carved out by Policy B
    still get an empty entry ``source_index -> []`` so cad_occ's
    ``overridden_indices`` includes them. See prior docstring for the
    rationale on why this matters.
    """
    out: dict[int, list[tuple[Any, int]]] = {}
    for shape in phantom_result.shapes:
        slab = plan.slabs[shape.slab_index]
        out.setdefault(slab.source_index, []).append(
            (shape.solid, shape.component_index)
        )
    for slab in plan.slabs:
        out.setdefault(slab.source_index, [])
    return out
```

- [ ] **Step 4: Update all call sites of `_group_phantom_solids_by_entity`**

Search:

```bash
grep -rn "_group_phantom_solids_by_entity" meshwell/ tests/
```

At each call site, the consumer must now handle `list[tuple[solid, cohort]]` instead of `list[solid]`. The primary call site is `meshwell/orchestrator.py:175`:

```python
overrides = _group_phantom_solids_by_entity(plan, phantom_result)
```

`overrides` is passed directly to `cad_occ(entity_shape_overrides=overrides, ...)`. cad_occ must accept the new shape — that's Task 4. For this task, **leave orchestrator unchanged** (the type just becomes `dict[int, list[tuple[Any, int]]]`; cad_occ will be updated in Task 4 to consume it).

If any other call site exists that destructures the old shape, update it to either: (a) extract solids via `[s for s, _ in v]`, or (b) consume the cohort tag if relevant.

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest tests/structured/test_group_phantom_solids_by_entity.py -v`
Expected: PASS.

- [ ] **Step 6: Run structured tests to catch fallout**

Run: `pytest tests/structured/ -v`
Expected: existing tests may BREAK if cad_occ hasn't been updated yet — that's fine, Task 4 fixes them. If any *non-cad-occ* structured test fails (e.g., a test of `_group_phantom_solids_by_entity` itself), it must be fixed before committing.

- [ ] **Step 7: Commit**

```bash
git add meshwell/structured/phantom.py tests/structured/test_group_phantom_solids_by_entity.py
git commit -m "feat(structured): _group_phantom_solids_by_entity returns (solid, cohort) tuples"
```

(Note: this commit may temporarily break cad_occ consumers; Task 4 restores green.)

---

## Task 4: Add `shape_cohorts` to `OCCLabeledEntity`; teach `_instantiate_entity_occ` to consume tuples

**Files:**
- Modify: `meshwell/cad_occ.py:62-86` (`OCCLabeledEntity` dataclass)
- Modify: `meshwell/cad_occ.py` (`_instantiate_entity_occ` — search for its `def`)
- Modify: `meshwell/cad_occ.py:1044-1067` (the override-installation point)
- Test: `tests/test_cad_occ_cohort_metadata.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_cad_occ_cohort_metadata.py`:

```python
"""OCCLabeledEntity carries shape_cohorts in lock-step with shapes."""

from __future__ import annotations

import pytest

from meshwell.cad_occ import OCCLabeledEntity


def test_default_shape_cohorts_is_none_per_shape():
    e = OCCLabeledEntity(
        shapes=[object(), object(), object()],
        physical_name=("foo",),
        index=0,
        keep=True,
        dim=3,
    )
    # After default init, shape_cohorts must be a list of Nones of matching length.
    assert e.shape_cohorts == [None, None, None]


def test_shape_cohorts_parallel_to_shapes():
    e = OCCLabeledEntity(
        shapes=[object(), object()],
        physical_name=("foo",),
        index=0,
        keep=True,
        dim=3,
        shape_cohorts=[7, None],
    )
    assert e.shape_cohorts == [7, None]
    assert len(e.shape_cohorts) == len(e.shapes)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_cad_occ_cohort_metadata.py -v`
Expected: FAIL (`shape_cohorts` doesn't exist on `OCCLabeledEntity`).

- [ ] **Step 3: Add `shape_cohorts` to `OCCLabeledEntity`**

In `meshwell/cad_occ.py:62-86`, modify the dataclass:

```python
@dataclass
class OCCLabeledEntity:
    """Per-entity record produced by :func:`cad_occ`.
    ...  (preserve existing docstring)
    ``shape_cohorts`` is a parallel list to ``shapes`` (same length). Each
    entry is the cohort (connected z-component) index of the corresponding
    shape, or ``None`` if the shape is not part of any structured cohort.
    Used by ``_fragment_all`` to bucket sub-prisms into compound BOP
    arguments (see spec 2026-05-27-cad-occ-structured-cohort-compound-bop-design.md).
    """

    shapes: list[TopoDS_Shape]
    physical_name: tuple[str, ...]
    index: int
    keep: bool
    dim: int
    mesh_order: float | None = None
    overlap_footprint: Any | None = None
    overlap_zrange: tuple[float, float] | None = None
    overlap_exact: bool = False
    structured: bool = False
    shape_cohorts: list[int | None] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Auto-pad shape_cohorts to len(shapes) with None when not provided.
        if len(self.shape_cohorts) != len(self.shapes):
            if not self.shape_cohorts:
                self.shape_cohorts = [None] * len(self.shapes)
            else:
                raise ValueError(
                    f"shape_cohorts length {len(self.shape_cohorts)} "
                    f"!= shapes length {len(self.shapes)}"
                )
```

Add `from dataclasses import field` to the imports if not already present.

- [ ] **Step 4: Run the unit test to verify it passes**

Run: `pytest tests/test_cad_occ_cohort_metadata.py -v`
Expected: PASS.

- [ ] **Step 5: Update `_instantiate_entity_occ` to handle (shape, cohort) tuples**

Find `_instantiate_entity_occ` in `meshwell/cad_occ.py` (search for `def _instantiate_entity_occ`). Its `shape_override` parameter currently accepts `list[Any]`; change to `list[tuple[Any, int]] | None`.

In the implementation: when `shape_override` is not None, build the `OCCLabeledEntity` with:

```python
shapes = [s for s, _ in shape_override]
cohorts = [c for _, c in shape_override]
# ... then construct OCCLabeledEntity with shapes=shapes, shape_cohorts=cohorts
```

When `shape_override` is None, the existing path is unchanged (`shape_cohorts` defaults to `[None] * len(shapes)` via `__post_init__`).

- [ ] **Step 6: Update the override-installation point**

In `meshwell/cad_occ.py:1044-1067` and the `process_entities_cut_only` method signature, ensure `entity_shape_overrides` is typed as `dict[int, list[tuple[Any, int]]] | None` and passes through unchanged into `_instantiate_entity_occ`.

- [ ] **Step 7: Add assertion that overridden entities have cohort != None for all their shapes**

In `_instantiate_entity_occ` (override branch), add:

```python
assert all(c is not None for c in cohorts), (
    f"entity {entity_index}: shape_override entries must carry a non-None "
    f"cohort index (got {cohorts!r})"
)
```

- [ ] **Step 8: Run the broader cad_occ test suite**

Run: `pytest tests/test_cad_occ.py tests/structured/ -v`
Expected: PASS. If structured tests still fail, the override pipeline isn't complete — debug before committing.

- [ ] **Step 9: Commit**

```bash
git add meshwell/cad_occ.py tests/test_cad_occ_cohort_metadata.py
git commit -m "feat(cad_occ): OCCLabeledEntity carries per-shape cohort tags"
```

---

## Task 5: Cohort-aware `_fragment_all`

**Files:**
- Modify: `meshwell/cad_occ.py:724-805` (`_fragment_all`)
- Test: extend `tests/test_cad_occ_cohort_sewing.py` with an integration test

- [ ] **Step 1: Write the failing integration test**

Append to `tests/test_cad_occ_cohort_sewing.py`:

```python
def test_fragment_all_skips_intra_cohort_bop():
    """End-to-end: cohort sub-prisms are bucketed, sewn, and added as one BOP argument.

    Construct a tiny scene:
      Entity 0 (structured): 3 sub-prisms (cohort_id=0) that share faces
      Entity 1 (unstructured): one box overlapping the leftmost sub-prism

    Run cad_occ on this and assert:
      - The leftmost sub-prism gets fragmented (overlap with Entity 1).
      - The other two sub-prisms pass through with their interface face
        still shared (verify via TShape identity).
    """
    pytest.skip(
        "Wire up using project's cad_occ entry; ensure entity_shape_overrides "
        "carries (solid, cohort_id) tuples per Task 3/4. Verify final entity "
        "shape lists have correct piece counts and that the inter-piece interface "
        "TShape is preserved post-fragment."
    )
```

(The implementer must wire this with the project's actual cad_occ test fixtures. Look at `tests/test_cad_occ_polyprism_overlap_fastpath.py` for the pattern.)

- [ ] **Step 2: Run it to confirm SKIP, then wire the fixture**

Run: `pytest tests/test_cad_occ_cohort_sewing.py::test_fragment_all_skips_intra_cohort_bop -v`
Expected: SKIP. Then replace the skip with actual construction. Re-run; should FAIL (cohort path not implemented yet).

- [ ] **Step 3: Implement the cohort-aware path in `_fragment_all`**

Replace `meshwell/cad_occ.py:724-805` with:

```python
def _fragment_all(
    self,
    entities: list[OCCLabeledEntity],
    progress_bars: bool = False,
    extra_occ_shapes: list[Any] | None = None,
    cad_occ_callback: Callable[[Any], None] | None = None,
) -> list[OCCLabeledEntity]:
    """Fragment all entity shapes; assign pieces by mesh_order priority.

    Cohort-aware variant: shapes tagged with a cohort id (from the
    structured planner's connected z-component) are bucketed by cohort,
    sewn into a single TopoDS_Compound, and added as ONE BOPAlgo_Builder
    argument per cohort. This skips pairwise BOP work on internal cohort
    interfaces, which the planner has already conformed. Sewing produces
    shared TShapes at internal interfaces so downstream interface tagging
    works unchanged.

    See spec docs/superpowers/specs/2026-05-27-cad-occ-structured-cohort-compound-bop-design.md.
    """
    if not entities:
        return []
    if len(entities) == 1 and not extra_occ_shapes and cad_occ_callback is None:
        return entities

    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
    from OCP.TopoDS import TopoDS_Compound

    builder = BOPAlgo_Builder()
    builder.SetRunParallel(self.n_threads > 1)
    builder.SetFuzzyValue(self.fragment_fuzzy_value)
    builder.SetNonDestructive(False)

    # Bucket cohort-tagged shapes by cohort id; record their (ent_idx, shape).
    # Non-cohort shapes get added directly.
    cohort_buckets: dict[int, list[tuple[int, TopoDS_Shape]]] = defaultdict(list)
    direct_args: list[tuple[int, TopoDS_Shape]] = []  # (ent_idx, shape)
    for ent_idx, ent in enumerate(entities):
        for shape, cohort in zip(ent.shapes, ent.shape_cohorts):
            if cohort is None:
                direct_args.append((ent_idx, shape))
            else:
                cohort_buckets[cohort].append((ent_idx, shape))

    # Build originals_per_entity in parallel with arg construction.
    # Each entity gets a list of (post_sewing) sub-shapes used as keys to
    # builder.Modified() in the collection loop below.
    originals_per_entity: list[list[TopoDS_Shape]] = [[] for _ in entities]

    # 1. Add non-cohort shapes as today.
    for ent_idx, shape in direct_args:
        builder.AddArgument(shape)
        originals_per_entity[ent_idx].append(shape)

    # 2. For each cohort, sew (if multi-shape) and add as one argument.
    for cohort_id, members in cohort_buckets.items():
        if len(members) == 1:
            # Singleton cohort: nothing to unify; add directly.
            ent_idx, shape = members[0]
            builder.AddArgument(shape)
            originals_per_entity[ent_idx].append(shape)
            continue

        # Build compound of cohort members.
        topo_builder = BRep_Builder()
        compound = TopoDS_Compound()
        topo_builder.MakeCompound(compound)
        for _, shape in members:
            topo_builder.Add(compound, shape)

        # Sew to unify coincident faces into shared TShapes.
        sewing = BRepBuilderAPI_Sewing(self.fragment_fuzzy_value)
        sewing.Load(compound)
        sewing.Perform()
        sewn = sewing.SewedShape()

        # Add the sewn compound as a single BOP argument.
        builder.AddArgument(sewn)

        # Record per-entity post-sewing originals for the collection loop.
        for ent_idx, orig in members:
            modified = sewing.ModifiedSubShape(orig)
            post = modified if not modified.IsNull() else orig
            originals_per_entity[ent_idx].append(post)

    for s in extra_occ_shapes or []:
        builder.AddArgument(s)

    if progress_bars:
        print(
            f"BOPAlgo_Builder.Perform() on {len(entities)} entities "
            f"({len(cohort_buckets)} cohorts, "
            f"{sum(1 for c in cohort_buckets.values() if len(c) > 1)} sewn)…",
            flush=True,
        )
    builder.Perform()

    if cad_occ_callback is not None:
        cad_occ_callback(builder)

    piece_candidates: dict[tuple[int, int], list[tuple[int, float]]] = defaultdict(
        list
    )
    piece_shapes: dict[tuple[int, int], TopoDS_Shape] = {}

    for ent_idx, ent in enumerate(
        tqdm(
            entities,
            desc="Collecting fragment pieces",
            disable=not progress_bars,
            leave=False,
        )
    ):
        mo = ent.mesh_order
        if mo is None:
            mo = float("inf")
        for original in originals_per_entity[ent_idx]:
            modified = builder.Modified(original)
            if modified.IsEmpty() and not builder.IsDeleted(original):
                pieces = [original]
            else:
                pieces = list(modified)
            for piece in pieces:
                k = _shape_key(piece)
                piece_shapes.setdefault(k, piece)
                piece_candidates[k].append((ent_idx, mo))

    owners = _resolve_piece_ownership(piece_candidates)

    for ent in entities:
        ent.shapes = []
        ent.shape_cohorts = []
    for key, ent_idx in owners.items():
        entities[ent_idx].shapes.append(piece_shapes[key])
        # After fragment, cohort metadata is no longer meaningful (pieces
        # may have been split/merged). Reset to None.
        entities[ent_idx].shape_cohorts.append(None)

    return entities
```

- [ ] **Step 4: Run the integration test**

Run: `pytest tests/test_cad_occ_cohort_sewing.py::test_fragment_all_skips_intra_cohort_bop -v`
Expected: PASS.

- [ ] **Step 5: Run the full cad_occ + structured test suite**

Run: `pytest tests/test_cad_occ.py tests/test_cad_occ_polyprism_overlap_fastpath.py tests/test_cad_occ_same_name_fuse.py tests/structured/ -v`
Expected: all PASS. Pay special attention to any test involving structured + interface tagging — that's the surface that could regress.

- [ ] **Step 6: Run the broader integration suite**

Run: `pytest tests/ -x -v -k "not slow and not benchmark"`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add meshwell/cad_occ.py tests/test_cad_occ_cohort_sewing.py
git commit -m "perf(cad_occ): cohort-compound BOP argument for structured sub-prisms"
```

---

## Task 6: End-to-end parity test

**Files:**
- Test: extend `tests/test_cad_occ_cohort_sewing.py`

- [ ] **Step 1: Write the parity test**

Append to `tests/test_cad_occ_cohort_sewing.py`:

```python
def test_cohort_compound_path_parity_with_legacy():
    """Same input → same mesh output (interface tags, physical names, ownership).

    Build a small structured-plus-unstructured scene. Run the full pipeline
    via the project's standard cad_occ entry point. Inspect the XAO output
    (or the equivalent in-memory artifact). Compare against a captured
    baseline from before this change — OR if no baseline exists, run the
    pipeline twice with an env var that disables the cohort path and assert
    bit-identical outputs.

    The cohort path can be disabled by checking an env var inside
    _fragment_all that, when set, forces the legacy flat-argument path.
    Add the env-var hook only for this test if needed; document it as
    test-only.
    """
    pytest.skip(
        "Implementer: either capture a baseline XAO/mesh and diff, or add a "
        "MESHWELL_DISABLE_COHORT_BOP env var hook and run both ways. The latter "
        "is more robust against unrelated mesher changes."
    )
```

- [ ] **Step 2: Implement the parity test**

Recommended approach: add a class-level attribute on the cad_occ caller (e.g., `CAD_OCC.enable_cohort_bop: bool = True`) that the test can toggle, run the pipeline twice (cohort on, cohort off), and assert:

1. Per-entity piece counts identical
2. Per-entity physical_name identical
3. For each entity pair (i, j), the set of shared face TShapes is identical between the two runs

- [ ] **Step 3: Run the parity test**

Run: `pytest tests/test_cad_occ_cohort_sewing.py::test_cohort_compound_path_parity_with_legacy -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_cad_occ_cohort_sewing.py meshwell/cad_occ.py
git commit -m "test(cad_occ): parity test cohort-compound vs legacy BOP path"
```

---

## Task 7: Performance measurement (success-criterion validation)

**Files:**
- Modify: `tests/benchmarks/` or create `scripts/bench_cohort_bop.py`

- [ ] **Step 1: Pick or build a representative scene**

Identify a production-scale scene that:
- Has ≥3 structured cohorts of ≥10 sub-prisms each
- Currently exhibits the slow `_fragment_all` behavior

If one isn't already in `tests/benchmarks/`, construct a synthetic scene that matches (e.g., a 4-cohort stack with 12 sub-prisms per cohort).

- [ ] **Step 2: Write a script that measures `_fragment_all` wall time**

Create `scripts/bench_cohort_bop.py`:

```python
"""Measure _fragment_all wall time with cohort-compound vs legacy path."""

from __future__ import annotations

import time

from meshwell.cad_occ import CAD_OCC

# ... build scene with N structured cohorts of M sub-prisms each ...
# Run cad_occ() twice — once with cohort-compound on, once forced off —
# and print the wall-clock delta on _fragment_all.

if __name__ == "__main__":
    # Implementer: fill in scene construction using project fixtures.
    pass
```

- [ ] **Step 3: Run the benchmark**

Run: `python scripts/bench_cohort_bop.py`
Expected: ≥3× speedup on `_fragment_all` for the structured-heavy scene.

If the speedup falls short:
- Check sewing cost: it may dominate for very large cohorts. If so, consider deferring sewing per-cohort to a thread pool.
- Profile to confirm BOPAlgo time actually decreased; if not, the cohort-bucketing logic isn't kicking in (check that `entity.shape_cohorts` carries non-None values).

- [ ] **Step 4: Record the measurement in the spec**

Append a "Measured Results" section to `docs/superpowers/specs/2026-05-27-cad-occ-structured-cohort-compound-bop-design.md`:

```markdown
## Measured Results (Task 7)

- Scene: <describe>
- Legacy `_fragment_all`: <time>
- Cohort-compound `_fragment_all`: <time>
- Speedup: <ratio>
```

- [ ] **Step 5: Commit**

```bash
git add scripts/bench_cohort_bop.py docs/superpowers/specs/2026-05-27-cad-occ-structured-cohort-compound-bop-design.md
git commit -m "perf(cad_occ): benchmark cohort-compound BOP speedup"
```

---

## Wrap-up

After all tasks pass:

- [ ] Final full-suite run: `pytest tests/ -v`
- [ ] Manual sanity check on one production scene if available
- [ ] If success criterion (≥3× on `_fragment_all`) not met, file a follow-up to revisit Deferred Option (2) per the spec
