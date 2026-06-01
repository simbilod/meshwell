# Void Boundary Tagging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop dropping voids in `decompose.py`; emit them as first-class `keep=False` sub-solids in the cohort compound so meshwell's existing `keep=False` machinery produces `neighbour___hole` interface tags naturally.

**Architecture:** Voids (`mesh_bool=False` structured PolyPrisms) become normal sub-solids in the cohort compound, marked with `keep=False`. Wedge stamping skips them. The XAO writer's existing `keep=False` semantics skip the body in BREP but use the TShapes to name interfaces with kept neighbours.

**Tech Stack:** Python 3.12, shapely, OCP (OpenCASCADE Python bindings), pytest

**Spec:** [docs/superpowers/specs/2026-06-01-void-boundary-tagging-design.md](../specs/2026-06-01-void-boundary-tagging-design.md)

---

## File map

**Modified:**
- `meshwell/structured/types.py` — `SlabMeta` gets a `keep: bool` field
- `meshwell/structured/decompose.py` — `_owner_slab` returns void source_index instead of None
- `meshwell/structured/build.py::build_cohort_compound` — populate `keep` from source slab's `mesh_bool`
- `meshwell/structured/pipeline.py::structured_post_pass` — propagate `keep` to OCCLabeledEntity
- `meshwell/structured/wedge.py::apply_lateral_transfinite_hints` — filter owners by keep=True before n_layers mismatch check
- `meshwell/structured/wedge.py::stamp_wedges` — skip slab_meta entries where keep=False
- `meshwell/structured/exceptions.py` — add `StructuredVoidMeshOrderRequiredError`
- `meshwell/structured/collect.py` — raise the new error when a void has `mesh_order=None`

**Created:**
- `tests/structured/test_void_keep_field.py`
- `tests/structured/test_void_decompose_emits_voids.py`
- `tests/structured/test_void_required_mesh_order.py`
- `tests/structured/test_void_tagging_e2e.py` (10 tests — see Task 10)

---

## Task 0: Add `keep` field to `SlabMeta`

**Files:**
- Modify: `meshwell/structured/types.py:83-96`
- Test: `tests/structured/test_void_keep_field.py`

The field is added first so all later tasks can populate it. Default to `True` so existing call sites stay valid without changes.

- [ ] **Step 1: Write the failing test**

```python
# tests/structured/test_void_keep_field.py
"""Verify SlabMeta carries a `keep` flag (True for solids, False for voids)."""
import pytest

from meshwell.structured.types import ShapeKey, SlabMeta


def test_slab_meta_keep_defaults_true():
    """SlabMeta should default to keep=True so existing call sites are
    unaffected."""
    key = ShapeKey(tshape_id=1, orientation=0)
    meta = SlabMeta(
        slab_index=0,
        physical_name=("bg",),
        bot_face_key=key,
        top_face_key=key,
        lateral_face_keys=(),
    )
    assert meta.keep is True


def test_slab_meta_keep_can_be_false():
    """SlabMeta with keep=False marks a void sub-solid."""
    key = ShapeKey(tshape_id=1, orientation=0)
    meta = SlabMeta(
        slab_index=0,
        physical_name=("hole",),
        bot_face_key=key,
        top_face_key=key,
        lateral_face_keys=(),
        keep=False,
    )
    assert meta.keep is False
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/structured/test_void_keep_field.py -v --no-cov
```
Expected: FAIL on `keep` not being a field of SlabMeta.

- [ ] **Step 3: Add the field**

Edit `meshwell/structured/types.py` to add `keep: bool = True` at the end of `SlabMeta`:

```python
@dataclass(frozen=True)
class SlabMeta:
    """Per-sub-solid metadata used at meshing time.

    Lookup happens by post-BOP ShapeKey of the sub-solid in the
    OCCLabeledEntity's shapes list. n_layers is NOT here — wedge.py
    resolves it from the resolution_specs dict via physical_name.

    `keep` mirrors the source slab's mesh_bool: True for solids whose
    wedges should be stamped, False for voids whose body must be
    excluded from BREP serialization (XAO writer keep=False path).
    """

    slab_index: int
    physical_name: tuple[str, ...]
    bot_face_key: ShapeKey
    top_face_key: ShapeKey
    lateral_face_keys: tuple[ShapeKey, ...]
    keep: bool = True
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/structured/test_void_keep_field.py -v --no-cov
```
Expected: PASS.

Also confirm no existing test broke:
```
pytest tests/structured/ --no-cov -q 2>&1 | tail -3
```
Expected: 87+ pass (was 86), 2 xfailed unchanged.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/types.py tests/structured/test_void_keep_field.py
git commit -m "feat(structured): add keep field to SlabMeta (default True)"
```

---

## Task 1: New exception for void without mesh_order

**Files:**
- Modify: `meshwell/structured/exceptions.py`
- Test: `tests/structured/test_void_required_mesh_order.py`

Voids must declare their priority since they'll now participate in Policy B sub-piece emission. A void with `mesh_order=None` would sort last (`float("inf")`) and then carve solids that already ran, which is confusing.

- [ ] **Step 1: Write the failing test**

```python
# tests/structured/test_void_required_mesh_order.py
"""Verify a void without mesh_order raises StructuredVoidMeshOrderRequiredError."""
import pytest
from shapely.geometry import Polygon

from meshwell.polyprism import PolyPrism
from meshwell.structured.collect import collect_structured_slabs
from meshwell.structured.exceptions import StructuredVoidMeshOrderRequiredError


SQ_BIG = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
SQ_SMALL = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])


def test_void_without_mesh_order_raises():
    bg = PolyPrism(SQ_BIG, {0.0: 0.0, 1.0: 0.0},
                   physical_name="bg", structured=True, mesh_order=1.0)
    hole = PolyPrism(SQ_SMALL, {0.0: 0.0, 1.0: 0.0},
                     physical_name="hole", structured=True, mesh_bool=False)
    with pytest.raises(StructuredVoidMeshOrderRequiredError):
        collect_structured_slabs([bg, hole])


def test_void_with_mesh_order_does_not_raise():
    bg = PolyPrism(SQ_BIG, {0.0: 0.0, 1.0: 0.0},
                   physical_name="bg", structured=True, mesh_order=2.0)
    hole = PolyPrism(SQ_SMALL, {0.0: 0.0, 1.0: 0.0},
                     physical_name="hole", structured=True,
                     mesh_bool=False, mesh_order=1.0)
    collect_structured_slabs([bg, hole])  # no raise


def test_solid_without_mesh_order_does_not_raise():
    """Only voids require mesh_order; solids without it default to inf
    and that's fine."""
    bg = PolyPrism(SQ_BIG, {0.0: 0.0, 1.0: 0.0},
                   physical_name="bg", structured=True)
    collect_structured_slabs([bg])  # no raise
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/structured/test_void_required_mesh_order.py -v --no-cov
```
Expected: FAIL (import error for `StructuredVoidMeshOrderRequiredError`).

- [ ] **Step 3: Add the exception class**

Append to `meshwell/structured/exceptions.py`:

```python
class StructuredVoidMeshOrderRequiredError(StructuredError):
    """A structured void (mesh_bool=False) must declare mesh_order.

    Without an explicit mesh_order, the void would sort last in the
    Policy B resolution (mesh_order=None -> float("inf")) and carve
    solids that already ran. Voids must explicitly state their priority
    against the solids around them.
    """

    def __init__(self, entity_index: int, physical_name: tuple[str, ...] | str):
        self.entity_index = entity_index
        self.physical_name = physical_name
        super().__init__(
            f"Structured void at entity #{entity_index} "
            f"(physical_name={physical_name!r}) has mesh_bool=False but no "
            "mesh_order. Voids must declare an explicit mesh_order so "
            "Policy B can resolve them against neighbouring solids."
        )
```

- [ ] **Step 4: Wire it into `collect_structured_slabs`**

Edit `meshwell/structured/collect.py`. After the existing `extrude` check, add a void-specific mesh_order check.

Locate the loop body where each structured PolyPrism is processed. Insert:

```python
        # Voids (mesh_bool=False) must declare an explicit mesh_order so
        # Policy B can resolve them against surrounding solids; without one
        # they would sort last and carve material that already ran.
        if not ent.mesh_bool and ent.mesh_order is None:
            from meshwell.structured.exceptions import (
                StructuredVoidMeshOrderRequiredError,
            )
            raise StructuredVoidMeshOrderRequiredError(
                entity_index=idx,
                physical_name=ent.physical_name,
            )
```

- [ ] **Step 5: Run test to verify it passes**

```
pytest tests/structured/test_void_required_mesh_order.py -v --no-cov
```
Expected: PASS.

Regression check:
```
pytest tests/structured/ --no-cov -q 2>&1 | tail -3
```
Expected: still ≥86 + 5 new = 91 pass. (Some prior void tests might now raise — investigate if so.)

- [ ] **Step 6: Commit**

```bash
git add meshwell/structured/exceptions.py meshwell/structured/collect.py tests/structured/test_void_required_mesh_order.py
git commit -m "feat(structured): require explicit mesh_order on voids"
```

---

## Task 2: Emit SubPiece for voids in `_owner_slab`

**Files:**
- Modify: `meshwell/structured/decompose.py:203-234`
- Test: `tests/structured/test_void_decompose_emits_voids.py`

Currently `_owner_slab` returns `None` when the winning candidate is a void, causing `decompose_cohorts` to skip emitting a SubPiece for that region. Change it to return the void's source_index so the SubPiece IS emitted (with the void as its source).

- [ ] **Step 1: Write the failing test**

```python
# tests/structured/test_void_decompose_emits_voids.py
"""Verify decompose emits SubPieces for voids (not just for solids)."""
from shapely.geometry import Polygon

from meshwell.polyprism import PolyPrism
from meshwell.structured.cohort import build_cohorts
from meshwell.structured.collect import collect_structured_slabs
from meshwell.structured.decompose import decompose_cohorts


SQ_BIG = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
SQ_SMALL = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])


def test_void_emits_subpiece():
    """A void carving a solid should emit TWO SubPieces: one for the
    solid annular ring AND one for the void's inner region."""
    bg = PolyPrism(SQ_BIG, {0.0: 0.0, 1.0: 0.0},
                   physical_name="bg", structured=True, mesh_order=2.0)
    hole = PolyPrism(SQ_SMALL, {0.0: 0.0, 1.0: 0.0},
                     physical_name="hole", structured=True,
                     mesh_order=1.0, mesh_bool=False)
    slabs, unstr = collect_structured_slabs([bg, hole])
    cohorts = build_cohorts(slabs)
    subpieces_per_cohort, _ = decompose_cohorts(cohorts, unstr)
    assert len(subpieces_per_cohort) == 1
    subs = subpieces_per_cohort[0]
    # Expect 2 sub-pieces at z=[0,1]: annular ring + inner disc.
    assert len(subs) == 2
    sources = sorted(sp.source_slab_indices[0] for sp in subs)
    # bg is entity 0, hole is entity 1 in the input list.
    assert sources == [0, 1]


def test_solid_only_still_one_subpiece():
    """Without a void, a single solid emits one SubPiece (unchanged behavior)."""
    bg = PolyPrism(SQ_BIG, {0.0: 0.0, 1.0: 0.0},
                   physical_name="bg", structured=True, mesh_order=1.0)
    slabs, unstr = collect_structured_slabs([bg])
    cohorts = build_cohorts(slabs)
    subpieces_per_cohort, _ = decompose_cohorts(cohorts, unstr)
    assert len(subpieces_per_cohort[0]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/structured/test_void_decompose_emits_voids.py::test_void_emits_subpiece -v --no-cov
```
Expected: FAIL — only 1 SubPiece emitted (the annular ring); the void's disc is dropped.

`test_solid_only_still_one_subpiece` should already PASS.

- [ ] **Step 3: Modify `_owner_slab`**

Edit `meshwell/structured/decompose.py`. Replace the body of `_owner_slab` so that the winning slab's source_index is returned regardless of `mesh_bool`:

```python
def _owner_slab(
    sub_polygon: Polygon, candidate_slabs: list[StructuredSlab]
) -> int | None:
    """Pick the slab that owns a sub-piece under Policy B.

    Take the sub_polygon's representative_point, find every candidate
    whose footprint contains it (solids and voids), then resolve the
    same way `zinterval_footprint` does:

      - Sort by (mesh_order, source_index) ascending.
      - The first slab in that order wins the point.

    Returns the winning slab's source_index (whether solid or void), or
    None if the point is outside every candidate's footprint. Voids
    return their own source_index now — they become first-class
    sub-pieces in the cohort compound, marked keep=False at post-pass
    time so the XAO writer's existing keep=False semantics produce
    `neighbour___void` interface tags.
    """
    pt = sub_polygon.representative_point()
    here = [s for s in candidate_slabs if s.footprint.contains(pt)]
    if not here:
        return None
    ordered = sorted(
        here,
        key=lambda s: (
            s.mesh_order if s.mesh_order is not None else float("inf"),
            s.source_index,
        ),
    )
    winner = ordered[0]
    return winner.source_index
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/structured/test_void_decompose_emits_voids.py -v --no-cov
```
Expected: PASS (both tests).

- [ ] **Step 5: Run full structured suite — expect some failures here**

```
pytest tests/structured/ --no-cov -q 2>&1 | tail -10
```

After this change, void sub-pieces flow through to `build_cohort_compound`, which builds them as solids. They're not yet `keep=False`, so existing tests that asserted "void does NOT appear as a physical group" may now fail. These will be fixed by Task 4.

If `test_void_carves_solid` (and similar) now claim the void shows up: expected. Move forward.

If unrelated tests fail (`test_lower_mesh_order_void_carves_structured`, etc.), inspect — they may need a small refresh (their assertions assume the void is silently dropped).

- [ ] **Step 6: Commit**

Even if some downstream tests fail, commit the change. We'll patch the downstream behavior in later tasks.

```bash
git add meshwell/structured/decompose.py tests/structured/test_void_decompose_emits_voids.py
git commit -m "feat(structured): emit SubPiece for voids in decompose"
```

---

## Task 3: Populate `keep` in `build_cohort_compound`

**Files:**
- Modify: `meshwell/structured/build.py` (the SlabMeta construction inside `build_cohort_compound`)

Each `SlabMeta` constructed in the build loop should set `keep` to the source slab's `mesh_bool`. Solids get keep=True; voids get keep=False.

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/structured/test_void_keep_field.py:

from shapely.geometry import Polygon

from meshwell.polyprism import PolyPrism
from meshwell.structured.build import build_cohort_compound
from meshwell.structured.cohort import build_cohorts
from meshwell.structured.collect import collect_structured_slabs
from meshwell.structured.decompose import decompose_cohorts


SQ_BIG = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
SQ_SMALL = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])


def test_build_populates_keep_from_source_slab():
    bg = PolyPrism(SQ_BIG, {0.0: 0.0, 1.0: 0.0},
                   physical_name="bg", structured=True, mesh_order=2.0)
    hole = PolyPrism(SQ_SMALL, {0.0: 0.0, 1.0: 0.0},
                     physical_name="hole", structured=True,
                     mesh_order=1.0, mesh_bool=False)
    slabs, unstr = collect_structured_slabs([bg, hole])
    cohorts = build_cohorts(slabs)
    subs_per_cohort, _ = decompose_cohorts(cohorts, unstr)
    _, slab_meta = build_cohort_compound(
        cohorts[0], subs_per_cohort[0], point_tolerance=1e-3,
    )
    by_name = {m.physical_name: m for m in slab_meta.values()}
    assert by_name[("bg",)].keep is True
    assert by_name[("hole",)].keep is False
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/structured/test_void_keep_field.py::test_build_populates_keep_from_source_slab -v --no-cov
```
Expected: FAIL — keep is True for the void (default value, since build doesn't set it yet).

- [ ] **Step 3: Modify `build_cohort_compound`**

Find the `SlabMeta(...)` construction inside the sub-solid assembly loop in `meshwell/structured/build.py::build_cohort_compound`. Look for `slab_meta[_shape_key(solid)] = SlabMeta(...)`. Add `keep=source_slab.mesh_bool` to the construction:

```python
        slab_meta[_shape_key(solid)] = SlabMeta(
            slab_index=source_slab.source_index,
            physical_name=source_slab.physical_name,
            bot_face_key=_shape_key(bot),
            top_face_key=_shape_key(top),
            lateral_face_keys=tuple(_shape_key(f) for f in laterals),
            keep=source_slab.mesh_bool,
        )
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/structured/test_void_keep_field.py -v --no-cov
```
Expected: PASS on all tests in the file.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/build.py tests/structured/test_void_keep_field.py
git commit -m "feat(structured): populate SlabMeta.keep from source slab's mesh_bool"
```

---

## Task 4: Propagate `keep` to OCCLabeledEntity in post-pass

**Files:**
- Modify: `meshwell/structured/pipeline.py::structured_post_pass`

The post-pass currently sets `keep=True` for every per-sub-solid OCCLabeledEntity (line 215). Change it to use the SlabMeta's `keep` field. This is the moment the void becomes a real `keep=False` entity that the XAO writer recognizes.

- [ ] **Step 1: Write the failing test**

```python
# tests/structured/test_void_post_pass_keep.py
"""Verify void sub-solids exit the post-pass with keep=False."""
from shapely.geometry import Polygon

from meshwell.cad_occ import cad_occ
from meshwell.polyprism import PolyPrism
from meshwell.structured.pipeline import structured_post_pass, structured_pre_pass


SQ_BIG = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
SQ_SMALL = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])


def test_void_sub_solid_marked_keep_false():
    bg = PolyPrism(SQ_BIG, {0.0: 0.0, 1.0: 0.0},
                   physical_name="bg", structured=True, mesh_order=2.0)
    hole = PolyPrism(SQ_SMALL, {0.0: 0.0, 1.0: 0.0},
                     physical_name="hole", structured=True,
                     mesh_order=1.0, mesh_bool=False)
    state = structured_pre_pass([bg, hole], point_tolerance=1e-3)
    occ_entities = cad_occ(state.entities_out, prepared=True)
    final = structured_post_pass(occ_entities, state)
    keep_by_name = {}
    for e in final:
        # Skip synthetic 2D annotators (names starting with __cohort_).
        if e.dim != 3:
            continue
        # The dim=3 entity carries (slab_name, synthetic_name) — get the
        # first (user-facing) name.
        name = e.physical_name[0]
        keep_by_name[name] = e.keep
    assert keep_by_name.get("bg") is True
    assert keep_by_name.get("hole") is False
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/structured/test_void_post_pass_keep.py -v --no-cov
```
Expected: FAIL — `hole`'s OCCLabeledEntity has keep=True (since the post-pass hardcodes keep=True).

- [ ] **Step 3: Use `meta.keep` in `structured_post_pass`**

Edit `meshwell/structured/pipeline.py`. In `structured_post_pass`, find the `OCCLabeledEntity` construction that builds `sub_ent`:

```python
                sub_ent = OCCLabeledEntity(
                    shapes=[shape],
                    physical_name=names,
                    index=next_index,
                    keep=True,
                    dim=3,
                    mesh_order=ent.mesh_order,
                )
```

Change `keep=True` to `keep=meta.keep`:

```python
                sub_ent = OCCLabeledEntity(
                    shapes=[shape],
                    physical_name=names,
                    index=next_index,
                    keep=meta.keep,
                    dim=3,
                    mesh_order=ent.mesh_order,
                )
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/structured/test_void_post_pass_keep.py -v --no-cov
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/pipeline.py tests/structured/test_void_post_pass_keep.py
git commit -m "feat(structured): propagate SlabMeta.keep to OCCLabeledEntity in post-pass"
```

---

## Task 5: Skip voids in `stamp_wedges`

**Files:**
- Modify: `meshwell/structured/wedge.py::stamp_wedges`

Voids' sub-solids must not get wedges stamped into them. Once their body is excluded from BREP (Task 4 enables that via `keep=False`), gmsh shouldn't even create a volume tag for them — but the stamp logic should still guard against it explicitly.

- [ ] **Step 1: Write the failing test**

```python
# tests/structured/test_void_no_wedges.py
"""Verify a void volume has no wedges in the output mesh."""
from pathlib import Path

import meshio
from shapely.geometry import Polygon

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.resolution import StructuredExtrusionResolutionSpec


SQ_BIG = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
SQ_SMALL = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])


def test_void_does_not_appear_as_3d_group(tmp_path: Path):
    bg = PolyPrism(SQ_BIG, {0.0: 0.0, 1.0: 0.0},
                   physical_name="bg", structured=True, mesh_order=2.0)
    hole = PolyPrism(SQ_SMALL, {0.0: 0.0, 1.0: 0.0},
                     physical_name="hole", structured=True,
                     mesh_order=1.0, mesh_bool=False)
    generate_mesh(
        [bg, hole], dim=3, output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
        resolution_specs={
            "bg": [StructuredExtrusionResolutionSpec(n_layers=2)],
        },
    )
    m = meshio.read(tmp_path / "out.msh")
    assert "bg" in m.cell_sets
    assert "hole" not in m.cell_sets, (
        "void should NOT appear as a 3D physical group"
    )
    # bg must have wedges.
    bg_sets = m.cell_sets["bg"]
    wedges = sum(
        len(s) for s, b in zip(bg_sets, m.cells)
        if b.type == "wedge" and s is not None
    )
    assert wedges > 0
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/structured/test_void_no_wedges.py -v --no-cov
```
Expected: One of:
- FAIL because gmsh tries to create a volume tag for the void and stamp_wedges crashes.
- FAIL because "hole" appears in cell_sets (it shouldn't).
- FAIL on a different gmsh / OCC error.

The exact failure depends on Task 4's keep=False propagation. Even if Task 4 made the void's body exclude correctly from BREP, the stamp_wedges loop might still try to look it up.

- [ ] **Step 3: Skip keep=False entries in `stamp_wedges`**

Edit `meshwell/structured/wedge.py::stamp_wedges`. After the iteration setup, filter out void slabs:

Find the order-construction loop:
```python
    order: list[tuple[float, ShapeKey, SlabMeta]] = []
    for k, meta in slab_meta.items():
        bot_tag = face_tag_by_key.get(meta.bot_face_key)
        if bot_tag is None:
            continue
        z = _face_centroid_z(bot_tag)
        order.append((z, k, meta))
```

Add a keep filter:

```python
    order: list[tuple[float, ShapeKey, SlabMeta]] = []
    for k, meta in slab_meta.items():
        if not meta.keep:
            # Voids: their bodies are excluded from BREP by the XAO writer
            # (keep=False), so they have no gmsh volume tag and no faces to
            # stamp. Skip them outright.
            continue
        bot_tag = face_tag_by_key.get(meta.bot_face_key)
        if bot_tag is None:
            continue
        z = _face_centroid_z(bot_tag)
        order.append((z, k, meta))
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/structured/test_void_no_wedges.py -v --no-cov
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/wedge.py tests/structured/test_void_no_wedges.py
git commit -m "feat(structured): skip keep=False sub-solids in stamp_wedges"
```

---

## Task 6: Filter keep=False owners in n_layers consistency check

**Files:**
- Modify: `meshwell/structured/wedge.py::apply_lateral_transfinite_hints` (around lines 70-89)

A lateral face shared between a solid (keep=True) and a void (keep=False) shouldn't fire the `n_layers` mismatch check on the void side — the void doesn't dictate vertical resolution.

- [ ] **Step 1: Write the failing test**

```python
# tests/structured/test_void_lateral_n_layers.py
"""Verify n_layers mismatch check ignores keep=False owners."""
from pathlib import Path

import meshio
from shapely.geometry import Polygon

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.resolution import StructuredExtrusionResolutionSpec


SQ_BIG = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
SQ_SMALL = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])


def test_void_does_not_trigger_n_layers_mismatch(tmp_path: Path):
    """The void's lateral face is shared with bg's inner ring. Since the
    void carries no n_layers (no resolution_specs entry for `hole`), the
    mismatch check would otherwise fire. Filtering keep=False owners
    avoids that."""
    bg = PolyPrism(SQ_BIG, {0.0: 0.0, 1.0: 0.0},
                   physical_name="bg", structured=True, mesh_order=2.0)
    hole = PolyPrism(SQ_SMALL, {0.0: 0.0, 1.0: 0.0},
                     physical_name="hole", structured=True,
                     mesh_order=1.0, mesh_bool=False)
    # No resolution_specs entry for "hole"; bg gets 2 layers.
    generate_mesh(
        [bg, hole], dim=3, output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
        resolution_specs={
            "bg": [StructuredExtrusionResolutionSpec(n_layers=2)],
        },
    )
    m = meshio.read(tmp_path / "out.msh")
    # If the test reaches here, the mismatch didn't fire. Sanity check:
    # bg has wedges.
    bg_sets = m.cell_sets["bg"]
    wedges = sum(
        len(s) for s, b in zip(bg_sets, m.cells)
        if b.type == "wedge" and s is not None
    )
    assert wedges > 0
```

- [ ] **Step 2: Run test to verify behavior**

```
pytest tests/structured/test_void_lateral_n_layers.py -v --no-cov
```
This test may already PASS if voids don't show up in `owners_per_face` (e.g. their lateral_face_keys aren't iterated when keep=False). Investigate the result first.

If it FAILS with `StructuredLateralNLayersMismatchError`, proceed with Step 3.
If it PASSES, the filter is already implicit — skip Step 3 but still add the explicit filter for safety. Run the test still — should PASS.

- [ ] **Step 3: Add explicit keep=True filter**

Edit `meshwell/structured/wedge.py::apply_lateral_transfinite_hints`. Find the owners-collection loop:

```python
    owners_per_face: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for meta in slab_meta.values():
        n_layers = resolve_n_layers(meta.physical_name, resolution_specs)
        for fk in meta.lateral_face_keys:
            tag = face_tag_by_key.get(fk)
            if tag is None:
                continue
            owners_per_face[tag].append((meta.slab_index, n_layers))
```

Add a keep check:

```python
    owners_per_face: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for meta in slab_meta.values():
        if not meta.keep:
            # Voids don't dictate vertical resolution; skip them so the
            # n_layers consistency check ignores keep=False owners.
            continue
        n_layers = resolve_n_layers(meta.physical_name, resolution_specs)
        for fk in meta.lateral_face_keys:
            tag = face_tag_by_key.get(fk)
            if tag is None:
                continue
            owners_per_face[tag].append((meta.slab_index, n_layers))
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/structured/test_void_lateral_n_layers.py -v --no-cov
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/wedge.py tests/structured/test_void_lateral_n_layers.py
git commit -m "feat(structured): exclude keep=False owners from lateral n_layers check"
```

---

## Task 7: End-to-end test — case 1 (void inside single structured slab)

**Files:**
- Create: `tests/structured/test_void_tagging_e2e.py`

Verify the simplest end-to-end case: bg square + hole disc void. Expect a `bg___hole` interface group with the inner cylindrical face entities. No `hole` 3D group. No `hole___None` group.

- [ ] **Step 1: Write the failing test**

```python
# tests/structured/test_void_tagging_e2e.py
"""End-to-end void boundary tagging tests."""
from __future__ import annotations

import math
from pathlib import Path

import gmsh
import meshio
import pytest
from shapely.geometry import Polygon

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.resolution import StructuredExtrusionResolutionSpec


def _disc(cx: float, cy: float, r: float, n: int = 48) -> Polygon:
    return Polygon([
        (cx + r * math.cos(2 * math.pi * i / n),
         cy + r * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ])


def _square(x: float, y: float, w: float, h: float) -> Polygon:
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def _physical_names(path: Path) -> set[str]:
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        return {
            gmsh.model.getPhysicalName(dim, tag)
            for dim, tag in gmsh.model.getPhysicalGroups()
        }
    finally:
        gmsh.finalize()


def _has_interface(names: set[str], a: str, b: str) -> bool:
    return f"{a}___{b}" in names or f"{b}___{a}" in names


def _structured_spec(*names: str, n_layers: int = 2) -> dict:
    return {n: [StructuredExtrusionResolutionSpec(n_layers=n_layers)] for n in names}


def test_void_inside_single_structured_slab(tmp_path: Path):
    """bg square at z=[0,1] with a hole disc void. No neighbours.
    Expected: bg___hole lateral. No hole 3D group. No hole___None."""
    bg = PolyPrism(_square(-3, -3, 6, 6), {0.0: 0.0, 1.0: 0.0},
                   physical_name="bg", structured=True, mesh_order=2.0)
    hole = PolyPrism(_disc(0, 0, 1.0), {0.0: 0.0, 1.0: 0.0},
                     physical_name="hole", structured=True,
                     mesh_order=1.0, mesh_bool=False, identify_arcs=True)
    msh = tmp_path / "out.msh"
    generate_mesh(
        [bg, hole], dim=3, output_mesh=msh,
        default_characteristic_length=0.4,
        resolution_specs=_structured_spec("bg"),
    )
    m = meshio.read(msh)
    names = _physical_names(msh)
    assert "bg" in m.cell_sets
    assert "hole" not in m.cell_sets, "void should not have 3D group"
    assert "hole___None" not in names, "void should not have boundary group"
    assert _has_interface(names, "bg", "hole"), (
        f"expected bg___hole lateral; got groups: {sorted(names)}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/structured/test_void_tagging_e2e.py::test_void_inside_single_structured_slab -v --no-cov
```
Expected: depends on prior tasks. Most likely PASS now (all prior tasks done correctly). If it FAILS, debug the keep=False propagation through cad_occ → XAO writer → gmsh load. The XAO writer's existing keep=False machinery should do the work.

If the failure is "interface not found", inspect:
```
python -c "
import gmsh
gmsh.initialize()
gmsh.option.setNumber('General.Terminal', 0)
gmsh.open('<your tmp msh>')
for dim, tag in gmsh.model.getPhysicalGroups():
    print(dim, gmsh.model.getPhysicalName(dim, tag))
gmsh.finalize()
"
```

Common causes if it fails:
- The void's TShape isn't shared with bg's lateral face TShape (build issue — but interior-ring lateral construction should handle this).
- The XAO writer is filtering out the void (check the `_is_purely_synthetic` check — if the void's physical_name is `("hole",)` it's NOT synthetic; should be treated as a real keep=False entity).

- [ ] **Step 3: Fix forward as needed**

If the test FAILS, examine the XAO writer behavior. The cohort entity's physical_name (`__cohort_0`) is in `cohort_pnames`. Each per-sub-solid OCCLabeledEntity has `physical_name = (slab.physical_name[0], synthetic_name)`. So `("hole", "__cohort_0__slab_1")` for the void. Verify the writer's `_filter_real_names` strips the synthetic part and uses just `"hole"` for interface naming.

If a mismatch is found, the fix likely involves the XAO writer's keep=False handling. Investigate, fix, and verify.

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/structured/test_void_tagging_e2e.py::test_void_inside_single_structured_slab -v --no-cov
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/structured/test_void_tagging_e2e.py [+ any writer fixes]
git commit -m "test(structured): e2e void boundary tagging — case 1 (no neighbours)"
```

---

## Task 8: End-to-end tests — cases 2, 3, 4 (void + unstructured neighbours)

**Files:**
- Modify: `tests/structured/test_void_tagging_e2e.py`

Add tests for void with cap above, void with base below, and void sandwiched.

- [ ] **Step 1: Write the tests**

Append to `tests/structured/test_void_tagging_e2e.py`:

```python
def test_void_below_unstructured_cap(tmp_path: Path):
    """bg + void + cap above. Expected: bg___hole + cap___hole."""
    bg = PolyPrism(_square(-3, -3, 6, 6), {0.0: 0.0, 1.0: 0.0},
                   physical_name="bg", structured=True, mesh_order=2.0)
    hole = PolyPrism(_disc(0, 0, 1.0), {0.0: 0.0, 1.0: 0.0},
                     physical_name="hole", structured=True,
                     mesh_order=1.0, mesh_bool=False, identify_arcs=True)
    cap = PolyPrism(_square(-3, -3, 6, 6), {1.0: 0.0, 2.0: 0.0},
                    physical_name="cap", mesh_order=3.0)
    msh = tmp_path / "out.msh"
    generate_mesh(
        [bg, hole, cap], dim=3, output_mesh=msh,
        default_characteristic_length=0.4,
        resolution_specs=_structured_spec("bg"),
    )
    m = meshio.read(msh)
    names = _physical_names(msh)
    assert "bg" in m.cell_sets and "cap" in m.cell_sets
    assert "hole" not in m.cell_sets
    assert _has_interface(names, "bg", "hole"), \
        f"missing bg___hole; groups: {sorted(names)}"
    assert _has_interface(names, "cap", "hole"), \
        f"missing cap___hole; groups: {sorted(names)}"


def test_void_above_unstructured_base(tmp_path: Path):
    """base + bg + void. Expected: bg___hole + base___hole."""
    base = PolyPrism(_square(-3, -3, 6, 6), {-1.0: 0.0, 0.0: 0.0},
                     physical_name="base", mesh_order=3.0)
    bg = PolyPrism(_square(-3, -3, 6, 6), {0.0: 0.0, 1.0: 0.0},
                   physical_name="bg", structured=True, mesh_order=2.0)
    hole = PolyPrism(_disc(0, 0, 1.0), {0.0: 0.0, 1.0: 0.0},
                     physical_name="hole", structured=True,
                     mesh_order=1.0, mesh_bool=False, identify_arcs=True)
    msh = tmp_path / "out.msh"
    generate_mesh(
        [base, bg, hole], dim=3, output_mesh=msh,
        default_characteristic_length=0.4,
        resolution_specs=_structured_spec("bg"),
    )
    m = meshio.read(msh)
    names = _physical_names(msh)
    assert _has_interface(names, "bg", "hole")
    assert _has_interface(names, "base", "hole"), \
        f"missing base___hole; groups: {sorted(names)}"


def test_void_sandwiched_between_unstructured(tmp_path: Path):
    """base + bg + void + cap. All three void interfaces."""
    base = PolyPrism(_square(-3, -3, 6, 6), {-1.0: 0.0, 0.0: 0.0},
                     physical_name="base", mesh_order=3.0)
    bg = PolyPrism(_square(-3, -3, 6, 6), {0.0: 0.0, 1.0: 0.0},
                   physical_name="bg", structured=True, mesh_order=2.0)
    hole = PolyPrism(_disc(0, 0, 1.0), {0.0: 0.0, 1.0: 0.0},
                     physical_name="hole", structured=True,
                     mesh_order=1.0, mesh_bool=False, identify_arcs=True)
    cap = PolyPrism(_square(-3, -3, 6, 6), {1.0: 0.0, 2.0: 0.0},
                    physical_name="cap", mesh_order=3.0)
    msh = tmp_path / "out.msh"
    generate_mesh(
        [base, bg, hole, cap], dim=3, output_mesh=msh,
        default_characteristic_length=0.4,
        resolution_specs=_structured_spec("bg"),
    )
    names = _physical_names(msh)
    assert _has_interface(names, "bg", "hole"), "lateral"
    assert _has_interface(names, "base", "hole"), "void bot"
    assert _has_interface(names, "cap", "hole"), "void top"
```

- [ ] **Step 2: Run the tests**

```
pytest tests/structured/test_void_tagging_e2e.py -v --no-cov
```
Expected: All 4 PASS. Investigate failures — the most likely issue is that the void's top/bot face TShape isn't merging with the neighbour's pre-cut sub-face. The bidirectional pre-cut should give the neighbour a matching sub-prism; arc detection unification should make the OCC arc curves match. If a specific case fails, dump the gmsh groups to understand.

- [ ] **Step 3: Commit**

```bash
git add tests/structured/test_void_tagging_e2e.py
git commit -m "test(structured): e2e void boundary tagging — unstructured neighbours"
```

---

## Task 9: End-to-end tests — cases 5, 6 (void + stacked cohort)

**Files:**
- Modify: `tests/structured/test_void_tagging_e2e.py`

Add tests for void spanning a stacked cohort, and void touching an upper structured slab.

- [ ] **Step 1: Write the tests**

Append to `tests/structured/test_void_tagging_e2e.py`:

```python
def test_void_through_stacked_cohort(tmp_path: Path):
    """A void spanning two stacked slabs. Lateral on BOTH slabs."""
    lower = PolyPrism(_square(-3, -3, 6, 6), {0.0: 0.0, 1.0: 0.0},
                      physical_name="lower", structured=True, mesh_order=2.0)
    upper = PolyPrism(_square(-3, -3, 6, 6), {1.0: 0.0, 2.0: 0.0},
                      physical_name="upper", structured=True, mesh_order=2.0)
    hole = PolyPrism(_disc(0, 0, 1.0), {0.0: 0.0, 2.0: 0.0},
                     physical_name="hole", structured=True,
                     mesh_order=1.0, mesh_bool=False, identify_arcs=True)
    msh = tmp_path / "out.msh"
    generate_mesh(
        [lower, upper, hole], dim=3, output_mesh=msh,
        default_characteristic_length=0.4,
        resolution_specs=_structured_spec("lower", "upper"),
    )
    names = _physical_names(msh)
    assert _has_interface(names, "lower", "hole"), \
        f"missing lower___hole; groups: {sorted(names)}"
    assert _has_interface(names, "upper", "hole"), \
        f"missing upper___hole; groups: {sorted(names)}"


def test_void_below_structured_cohort_slab(tmp_path: Path):
    """Lower has a void; upper is solid above. Expected: lower___hole
    (lateral) + upper___hole (void top at z=1 touching upper's bot)."""
    lower = PolyPrism(_square(-3, -3, 6, 6), {0.0: 0.0, 1.0: 0.0},
                      physical_name="lower", structured=True, mesh_order=2.0)
    upper = PolyPrism(_square(-3, -3, 6, 6), {1.0: 0.0, 2.0: 0.0},
                      physical_name="upper", structured=True, mesh_order=2.0)
    hole = PolyPrism(_disc(0, 0, 1.0), {0.0: 0.0, 1.0: 0.0},
                     physical_name="hole", structured=True,
                     mesh_order=1.0, mesh_bool=False, identify_arcs=True)
    msh = tmp_path / "out.msh"
    generate_mesh(
        [lower, upper, hole], dim=3, output_mesh=msh,
        default_characteristic_length=0.4,
        resolution_specs=_structured_spec("lower", "upper"),
    )
    names = _physical_names(msh)
    assert _has_interface(names, "lower", "hole"), "lateral"
    assert _has_interface(names, "upper", "hole"), (
        f"void top at z=1 should touch upper's bot; got: {sorted(names)}"
    )


def test_void_square_no_arcs(tmp_path: Path):
    """Square void (polyline only). Lateral walls bg___hole."""
    bg = PolyPrism(_square(-3, -3, 6, 6), {0.0: 0.0, 1.0: 0.0},
                   physical_name="bg", structured=True, mesh_order=2.0)
    hole = PolyPrism(_square(-0.5, -0.5, 1, 1), {0.0: 0.0, 1.0: 0.0},
                     physical_name="hole", structured=True,
                     mesh_order=1.0, mesh_bool=False)
    msh = tmp_path / "out.msh"
    generate_mesh(
        [bg, hole], dim=3, output_mesh=msh,
        default_characteristic_length=0.4,
        resolution_specs=_structured_spec("bg"),
    )
    names = _physical_names(msh)
    assert _has_interface(names, "bg", "hole"), \
        f"missing bg___hole (square void); groups: {sorted(names)}"
```

- [ ] **Step 2: Run the tests**

```
pytest tests/structured/test_void_tagging_e2e.py -v --no-cov
```
Expected: All 7 PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/structured/test_void_tagging_e2e.py
git commit -m "test(structured): e2e void boundary tagging — stacked cohort + square"
```

---

## Task 10: End-to-end tests — overlapping voids + arc neighbour pre-cut

**Files:**
- Modify: `tests/structured/test_void_tagging_e2e.py`

Two additional cases:
- **Two overlapping voids** — Policy B winner shows up as the boundary owner.
- **Void with arc-bearing pre-cut neighbour** — verifies the bidirectional pre-cut + arc propagation produces matching OCC curves on both sides.

- [ ] **Step 1: Write the tests**

Append to `tests/structured/test_void_tagging_e2e.py`:

```python
def test_two_overlapping_voids_policy_b(tmp_path: Path):
    """Two voids overlapping: lower-mesh_order wins. The winner is the
    one whose physical_name appears in the bg___hole interface."""
    bg = PolyPrism(_square(-3, -3, 6, 6), {0.0: 0.0, 1.0: 0.0},
                   physical_name="bg", structured=True, mesh_order=3.0)
    # Void A: bigger disc, lower priority (higher mesh_order among voids).
    void_a = PolyPrism(_disc(0, 0, 1.5), {0.0: 0.0, 1.0: 0.0},
                       physical_name="void_a", structured=True,
                       mesh_order=2.0, mesh_bool=False, identify_arcs=True)
    # Void B: smaller disc inside void_a, higher priority (lower mesh_order).
    void_b = PolyPrism(_disc(0, 0, 0.8), {0.0: 0.0, 1.0: 0.0},
                       physical_name="void_b", structured=True,
                       mesh_order=1.0, mesh_bool=False, identify_arcs=True)
    msh = tmp_path / "out.msh"
    generate_mesh(
        [bg, void_a, void_b], dim=3, output_mesh=msh,
        default_characteristic_length=0.4,
        resolution_specs=_structured_spec("bg"),
    )
    m = meshio.read(msh)
    names = _physical_names(msh)
    # Neither void appears as a 3D group.
    assert "void_a" not in m.cell_sets
    assert "void_b" not in m.cell_sets
    # void_b (lower mesh_order) is the inner boundary owner;
    # void_a (higher mesh_order among voids) is the outer ring boundary.
    # Both should show up as interfaces with bg.
    assert _has_interface(names, "bg", "void_b"), \
        f"inner void boundary missing; groups: {sorted(names)}"
    # void_a's ring (between void_b and bg's outer boundary) shows up too.
    assert _has_interface(names, "bg", "void_a") or _has_interface(names, "void_a", "void_b"), (
        f"outer void boundary missing; groups: {sorted(names)}"
    )


def test_void_with_arc_neighbour_pre_cut(tmp_path: Path):
    """A void with an arc-bearing cap above, both involving arcs. The
    bidirectional pre-cut + unified arc detection should make both sides
    have matching OCC arc edges so BOP merges the shared disc face."""
    bg = PolyPrism(_square(-3, -3, 6, 6), {0.0: 0.0, 1.0: 0.0},
                   physical_name="bg", structured=True, mesh_order=2.0)
    hole = PolyPrism(_disc(0, 0, 1.0), {0.0: 0.0, 1.0: 0.0},
                     physical_name="hole", structured=True,
                     mesh_order=1.0, mesh_bool=False, identify_arcs=True)
    # Arc-bearing cap (a larger disc) above. Pre-cut splits it into
    # disc-over-hole + annular-ring-over-bg.
    cap = PolyPrism(_disc(0, 0, 2.5), {1.0: 0.0, 2.0: 0.0},
                    physical_name="cap", mesh_order=3.0, identify_arcs=True)
    msh = tmp_path / "out.msh"
    generate_mesh(
        [bg, hole, cap], dim=3, output_mesh=msh,
        default_characteristic_length=0.4,
        resolution_specs=_structured_spec("bg"),
    )
    names = _physical_names(msh)
    assert _has_interface(names, "bg", "hole"), \
        f"missing bg___hole lateral; groups: {sorted(names)}"
    assert _has_interface(names, "cap", "hole"), \
        f"missing cap___hole (arc-vs-arc); groups: {sorted(names)}"
```

- [ ] **Step 2: Run the tests**

```
pytest tests/structured/test_void_tagging_e2e.py -v --no-cov
```
Expected: All 9 PASS. The arc-vs-arc case might fail if the BOP merge issue from the complex stress xfail surfaces — investigate. The pre-cut on the cap side now uses `identify_arcs=True` (propagated by `decompose_cohorts`), and unified `decompose_vertices_2d` should produce identical arc edges.

- [ ] **Step 3: Commit**

```bash
git add tests/structured/test_void_tagging_e2e.py
git commit -m "test(structured): e2e void tagging — overlapping voids + arc neighbour"
```

---

## Task 11: Verify no regressions

**Files:** none — verification only

- [ ] **Step 1: Run the full structured suite**

```
pytest tests/structured/ --no-cov 2>&1 | tail -10
```
Expected: ≥ 86 (baseline) + 5 (new void unit tests) + 10 (new void e2e tests) = ≥ 101 passing. Original 2 xfails (`test_stress_complex_scene`) unchanged.

If any previously-passing test now fails, investigate:
- Are voids appearing in unexpected places? Some existing tests may have assumed voids are "invisible". With voids now being keep=False (and thus appearing in interface groups), the test may need its assertion updated — but ONLY if the new behavior is correct.
- Is the wedge count for solid slabs now wrong? Possibly the solid sub-piece geometry changed because polygonize now sees voids' boundaries differently. Spot-check.

- [ ] **Step 2: Run demos as smoke tests**

```
python demo_structured.py 2>&1 | tail -5
python demo_curves.py 2>&1 | grep -E "Scene|wedges" | head -15
```
Expected: `demo_structured.py` writes its mesh cleanly. `demo_curves.py` produces single_disc / stacked_discs / annulus_on_disc with similar wedge counts as before (within ~5% due to mesh-density variability).

- [ ] **Step 3: Commit any small fixes**

If any test needed an updated assertion to reflect correct new behavior:

```bash
git add tests/structured/<file>
git commit -m "test(structured): update assertion for void-as-sub-solid behavior"
```

---

## Task 12: Update existing `test_void_carves_solid` if needed

**Files:**
- Modify: `tests/structured/test_stress_arbitrary_slabs.py` (if test_void_carves_solid behavior changed)

The pre-existing `test_void_carves_solid` test asserts:
- "void" NOT in cell_sets
- "solid" IS in cell_sets with wedges > 0
- "cap" IS in cell_sets with wedges > 0
- Solid's wedge area ≈ surviving area (4×4 - 2×2 = 12)

After the void-as-sub-solid change, the solid's wedge count should better reflect the carved area (the sub-piece is the annular ring, polygon area = 12). If the wedge-area fraction assertion was being satisfied loosely before, it should now be tighter — or it may need adjusting if the new mesh density differs.

- [ ] **Step 1: Re-run the test**

```
pytest tests/structured/test_stress_arbitrary_slabs.py::test_void_carves_solid -v --no-cov
```
- If PASS: nothing to do.
- If FAIL: inspect the assertion. If the wedge count is wrong, debug the new sub-piece emission. If it's just a tolerance mismatch, loosen the tolerance modestly.

- [ ] **Step 2: Commit any fix**

If a fix was needed:

```bash
git add tests/structured/test_stress_arbitrary_slabs.py
git commit -m "test(structured): refresh void carves solid assertion"
```

---

## Final verification

- [ ] **Step 1: Full structured suite**

```
pytest tests/structured/ --no-cov 2>&1 | tail -5
```
Expected: ≥ 101 passed, 2 xfailed (arc-merge complex scene unchanged).

- [ ] **Step 2: Demos**

```
python demo_structured.py
python demo_curves.py
```
Both should run cleanly.

- [ ] **Step 3: Cross-check broader tests**

```
pytest tests/test_cad_occ.py tests/test_cad_occ_fragment_ownership.py tests/test_backend_cross_compare.py --no-cov 2>&1 | tail -3
```
Expected: No regressions on the cad_occ-direct test suites.

- [ ] **Step 4: Final commit if any cleanup**

```bash
git status
# review for any uncommitted changes
git add -A
git commit -m "chore: final cleanup for void boundary tagging"
```
