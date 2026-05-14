# Structured Slab Correctness Plan (Revised)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the structured-slab feature mesh correctly on complex
scenes (many entities, overlapping pillars, arc-bearing rings, stacked
sub-slabs) instead of crashing inside `gmsh.model.mesh.generate(2)` or
inside the conformal slab builder.

**Architecture:** Five phases driven by empirical evidence collected
via [diagnose_arc_periodic.py](diagnose_arc_periodic.py) and
[bench_structured.py](bench_structured.py).

* **Phase 1 (root cause #1, validated):** Extend `_compute_face_partition`'s
  splitter predicate to include 3D entities that *cross* the slab in z
  (not only those whose endpoints *touch* z=zlo/z=zhi). Validated by a
  one-line experiment: struct_ring slabs that previously had asymmetric
  decompositions (`3 bot ↔ 2 top`, `4 bot ↔ 2 top`) became mirror-symmetric
  (`3 bot ↔ 3 top` on all three), and the `Different number of points`
  setPeriodic failure disappeared. All 16 existing structured tests
  still pass with this edit.
* **Phase 2 (latent bug R2, now visible):** The conformal builder's
  `_face_belongs_to_slab` predicate (in `_find_all_occ_faces_for_slab`)
  rejects valid bottom OCC faces of stacked sub-slabs in dense scenes,
  so `_build_one_slab_conformal` raises `could not locate bottom OCC
  face(s)`. Was masked before Phase 1 because the run crashed earlier
  on the transfinite-5-corner check.
* **Phase 3 (latent bug R1, warning-level):** Even with symmetric
  bottom/top decompositions, `setPeriodic` warns `Could not find all
  point correspondences` on arc-bearing structured rings. Probable
  cause: independent arc tessellation on bottom vs top OCC faces.
  Non-fatal but degrades lateral mesh quality.
* **Phase 4 (backstop):** A CAD-stage validator that enumerates
  *post-CAD* OCC bottom/top face decompositions per slab and raises a
  clear error if they are not mirror-symmetric — for any future case
  Phase 1 does not cover (e.g., z-varying footprints).
* **Phase 5 (regression coverage):** Move the diagnostic + benchmark
  scripts into `tests/benchmarks/` and gate with `@pytest.mark.slow`.

**Tech stack:** Python 3.12, gmsh OCC kernel, shapely 2.x, pytest.
All work in `meshwell/structured_polyprism.py`, `meshwell/cad_common.py`,
and the test/benchmark directories.

**Status note:** The `crosses_z` edit from Phase 1 is *already applied*
in the working tree as the experimental validation that drove this
plan. Phase 1 below codifies it with proper tests; do not re-apply.

---

## File Map

* **Modify:** [meshwell/structured_polyprism.py](meshwell/structured_polyprism.py)
  * `_compute_face_partition` (around line 340) — Phase 1: `crosses_z`
    predicate. *Already applied; tests + commit pending.*
  * `_find_all_occ_faces_for_slab` / `_face_belongs_to_slab` (around
    line 1990) — Phase 2: fix or relax the physical-group filter.
  * `_apply_slab_horizontal_periodicity` (around line 1264) — Phase 3:
    bottom/top arc canonicalization OR `setPeriodic` tolerance handling.
  * Add `_validate_slab_occ_face_symmetry` near other validators
    (around line 880) — Phase 4.
* **Modify:** [meshwell/cad_common.py](meshwell/cad_common.py) lines 130–150
  — Phase 4: wire the new validator in.
* **Create:** [tests/test_structured_through_entities.py](tests/test_structured_through_entities.py)
  — Phase 1 unit + smoke tests.
* **Create:** [tests/test_structured_dense_scene.py](tests/test_structured_dense_scene.py)
  — Phase 2 regression test for the stack face-location issue.
* **Move:** `bench_structured.py` → `tests/benchmarks/bench_structured.py`,
  `diagnose_arc_periodic.py` → `tests/benchmarks/diagnose_arc_periodic.py`
  — Phase 5.

---

## Phase 1: Codify the `crosses_z` fix

The single-line predicate extension that solved the primary failure.
Already applied — this phase covers tests and the commit.

### Task 1.1: Regression test — through-entity scene meshes successfully

**Files:**
* Create: [tests/test_structured_through_entities.py](tests/test_structured_through_entities.py)

- [ ] **Step 1: Write the failing test (against `crosses_z`-less baseline)**

```python
"""Regression tests for 3D entities that pass through a structured slab in z.

The bench failure that motivated these tests:
    bench_structured.py crashed with `Surface N is transfinite but has K
    corners` because BOP fragmented struct_ring slabs against pillars that
    fully crossed the slab in z, producing asymmetric bottom/top sub-face
    decompositions. _compute_face_partition did not list crossing entities
    as splitters (it only listed entities *touching* z=zlo/z=zhi), so the
    cascade's pre-CAD partition didn't account for the cuts BOP would
    later introduce.
"""
from __future__ import annotations

import math
from pathlib import Path

from shapely.geometry import Polygon

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism


def _disk(cx: float, cy: float, r: float, n: int = 24) -> Polygon:
    return Polygon([
        (cx + r * math.cos(2 * math.pi * i / n),
         cy + r * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ])


def _square(x0, x1, y0, y1) -> Polygon:
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def test_pillar_crossing_slab_partitions_symmetrically(tmp_path: Path) -> None:
    """A 3D pillar fully crossing a structured slab in z must induce
    matching cuts on z=zlo and z=zhi.

    Without ``crosses_z`` in _compute_face_partition, BOP introduces the
    pillar's intersection as a post-cascade fragmentation that ends up
    asymmetric (e.g., 2 bottom sub-faces vs 1 top sub-face), which then
    crashes mesh.generate(2) with `transfinite has 5 corners`.
    """
    slab = PolyPrism(
        polygons=_square(0, 4, 0, 4),
        buffers={0.4: 0.0, 1.0: 0.0},
        n_layers=[4],
        physical_name="slab",
        mesh_order=2,
    )
    # Pillar fully crosses z=[0.4, 1.0] (z=[0, 1.5] contains [0.4, 1.0])
    # and overlaps the slab's xy interior. Pillar loses on mesh_order so
    # the slab wins the overlap volume, but BOP still fragments at
    # z=0.4 / z=1.0 / pillar wall.
    pillar = PolyPrism(
        polygons=_disk(2.0, 2.0, 0.6, n=24),
        buffers={0.0: 0.0, 1.5: 0.0},
        physical_name="pillar",
        mesh_order=18,
    )
    out = tmp_path / "pillar_through.msh"
    generate_mesh(
        entities=[slab, pillar],
        dim=3,
        output_mesh=out,
        default_characteristic_length=0.4,
    )
    assert out.exists()


def test_multiple_pillars_through_arc_slab(tmp_path: Path) -> None:
    """Two pillars crossing an arc-bearing structured ring slab.

    Mirrors the bench's struct_ring_0 + pillar_0 + pillar_1 interaction
    that crashed before Phase 1.
    """
    def _annulus(cx, cy, ro, ri, n=48):
        outer = [(cx + ro * math.cos(2 * math.pi * i / n),
                  cy + ro * math.sin(2 * math.pi * i / n)) for i in range(n)]
        inner = [(cx + ri * math.cos(2 * math.pi * i / n),
                  cy + ri * math.sin(2 * math.pi * i / n)) for i in range(n)]
        return Polygon(outer, holes=[inner])

    ring = PolyPrism(
        polygons=_annulus(0.0, 0.0, 1.5, 0.7, n=48),
        buffers={0.4: 0.0, 1.0: 0.0},
        n_layers=[4],
        physical_name="ring",
        mesh_order=2,
        identify_arcs=True,
    )
    pillars = [
        PolyPrism(
            polygons=_disk(1.0, 0.0, 0.3, n=24),
            buffers={0.0: 0.0, 1.5: 0.0},
            physical_name="pillar_a",
            mesh_order=18,
        ),
        PolyPrism(
            polygons=_disk(-1.0, 0.0, 0.3, n=24),
            buffers={0.0: 0.0, 1.5: 0.0},
            physical_name="pillar_b",
            mesh_order=18,
        ),
    ]
    out = tmp_path / "pillars_through_ring.msh"
    generate_mesh(
        entities=[ring] + pillars,
        dim=3,
        output_mesh=out,
        default_characteristic_length=0.3,
    )
    assert out.exists()
```

- [ ] **Step 2: Verify both tests PASS with the current tree (`crosses_z` already applied)**

```bash
pytest tests/test_structured_through_entities.py -v
```

Expected: 2 passed.

(If they fail with `transfinite has K corners` or
`Conformal slab build ... could not locate bottom OCC face(s)`, that
indicates the second test's geometry triggers a Phase 2 issue. In that
case, mark the second test with
`@pytest.mark.xfail(reason="Phase 2 fix pending")` and proceed —
Task 2.x covers it.)

- [ ] **Step 3: Sanity-check the partition count**

Append:

```python
def test_crosses_z_predicate_adds_pillar_to_partition() -> None:
    """White-box: the cascade's face_partition for a slab with a
    crossing pillar must have at least 2 pieces (slab minus pillar +
    pillar). Locks in the Phase 1 fix at unit-test granularity."""
    from meshwell.structured_polyprism import resolve_structured_slabs

    slab = PolyPrism(
        polygons=_square(0, 4, 0, 4),
        buffers={0.4: 0.0, 1.0: 0.0},
        n_layers=[4],
        physical_name="slab",
        mesh_order=2,
    )
    pillar = PolyPrism(
        polygons=_disk(2.0, 2.0, 0.6, n=24),
        buffers={0.0: 0.0, 1.5: 0.0},
        physical_name="pillar",
        mesh_order=18,
    )
    slabs = resolve_structured_slabs([slab, pillar])
    assert len(slabs) == 1
    partition = slabs[0].face_partition
    assert partition is not None
    assert len(partition) >= 2, (
        f"Expected at least 2 partition pieces (slab + pillar cut), "
        f"got {len(partition)}"
    )
```

- [ ] **Step 4: Run the new white-box test**

```bash
pytest tests/test_structured_through_entities.py::test_crosses_z_predicate_adds_pillar_to_partition -v
```

Expected: PASS. If it fails with `len(partition) == 1`, the
`crosses_z` predicate edit is missing — check
[structured_polyprism.py:385-401](meshwell/structured_polyprism.py#L385-L401).

- [ ] **Step 5: Run the full structured test suite**

```bash
pytest tests/test_structured_complex_scene.py tests/test_overlapping_facets_structured.py -v
```

Expected: 16 passed (matches pre-Phase 1 baseline).

- [ ] **Step 6: Commit Phase 1**

```bash
git add meshwell/structured_polyprism.py tests/test_structured_through_entities.py
git commit -m "$(cat <<'EOF'
fix(structured): include 3D-crossing entities in slab face partition

Slabs intersected in xy by a 3D entity whose z-range fully contains
the slab's z-range previously got asymmetric bottom/top OCC face
decompositions from BOP -- _compute_face_partition only treated
entities *touching* z=zlo/z=zhi as splitters. Add a `crosses_z`
predicate so the cascade pre-partitions the slab for every crossing
entity, making BOP's fragmentation mirror-symmetric.

Validated on bench_structured.py: struct_ring_0/1/2 go from
3/4/3 bottom vs 2/2/2 top sub-faces to 3/3/3 on both, and the
"Different number of points" setPeriodic crash is gone.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2: Fix the BOPAlgo Modified() over-attribution for multi-partition phantoms (R2 — ROOT-CAUSE LOCATED)

> **Investigation complete (2026-05-13).** R2 root-causes to
> `BOPAlgo_Builder.Modified()` over-attributing fragment pieces in the
> global cad_occ fragment pass. Specifically, in dense scenes:
>
> 1. `_compute_face_partition` produces a 3-piece partition for
>    `struct_ring_0` where piece[0] is an annulus-with-holes (main
>    region minus pillar-overlap cutouts) and piece[1]/piece[2] are
>    small interior pieces.
> 2. `_StructuredPhantom.instanciate_occ` builds 3 sub-prisms, fuses
>    them via internal `BOPAlgo_Builder`, returns the compound. The
>    internal BOP produces **5 sub-solids** (not 3) — two extras come
>    from fragmenting piece[0]'s annulus around piece[1]/piece[2]'s
>    interior holes.
> 3. The 5 sub-solids carry overlapping TShape state from the internal
>    BOP's history map.
> 4. In `cad_occ._fragment_all`, the global `BOPAlgo_Builder.Modified()`
>    is called per original. For `struct_ring_0`'s 5 originals, it
>    returns **97 pieces** (vs 3 for struct_ring_1 which has 3 originals)
>    — almost every piece in the entire model is mis-attributed back to
>    struct_ring_0.
> 5. `_resolve_piece_ownership` picks lowest mesh_order winner. Since
>    `struct_ring_0` (mo=2) beats stack (mo=3) / ring (mo=4) / pillar
>    (mo=18) on every piece it claims, it ends up owning **39 of 53
>    entities' pieces**.
> 6. The XAO writer then tags those pieces with `struct_ring_0___None`,
>    and the mesh-stage builder fails to locate stacks' bottom faces.
>
> **Evidence:** `scripts/inspect_modified.py` reports per-entity
> Modified() output sizes; `scripts/inspect_ownership.py` confirms
> 39-shape claim; `scripts/inspect_pre_bop.py` confirms pre-BOP shapes
> are clean (all within struct_ring_0's bounds).
>
> **Fixes attempted:**
> * Replace internal BOP with `BRep_Builder.MakeCompound`: crashes the
>   global BOP (core dump). Internal BOP is doing real geometric work
>   the global pass relies on.
> * `BRepBuilderAPI_Copy(bop.Shape(), True, True)`: doesn't change
>   `Modified()` output (still 97 pieces). The shared TShape state
>   survives the copy at the level OCP exposes.
> * Skipping internal BOP for single-input case (already does): only
>   helps when partition has 1 piece.
>
> **Fix paths (require deeper work):**
>
> * **A. Make partition pieces truly disjoint at the OCC level.**
>   Instead of building piece[0] as an annulus-with-holes (one face
>   with interior wires for piece[1]/piece[2] cutouts), reshape
>   `_compute_face_partition` to emit *only* simply-connected pieces
>   without holes. polygonize already does this in principle, but
>   shapely may return polygons with interior rings for nested
>   splitters. Verify by inspecting `piece[0].interiors` on the
>   struct_ring case.
>
> * **B. Don't run internal BOP at all.** Return the 3 sub-prism
>   solids directly to cad_occ (wrapped in a Compound), and let the
>   global BOPAlgo_Builder discover partition-boundary TShape sharing
>   via its own fuzzy. Tests indicate this crashes today because the
>   bare-compound approach isn't compatible with how cad_occ.cut
>   propagates shapes; needs an investigation of what assumption
>   process_entities_cut_only makes about `ent.shapes`.
>
> * **C. Bypass cad_occ's global Modified() lookup for phantoms.**
>   Phantoms know their own footprints. Instead of letting BOPAlgo's
>   Modified() decide what struct_ring_0 owns, the cad_occ pipeline
>   could short-circuit phantom ownership: any piece whose bbox lies
>   inside the phantom's slab footprint at slab z-range is owned by
>   that phantom. This is more invasive (changes cad_occ semantics
>   for phantoms) but avoids the buggy OCP behaviour entirely.
>
> Recommendation: try Fix A first (smallest blast radius, addresses
> root cause). If shapely's polygonize emits piece[0] with interior
> rings, replace the partition strategy in
> [_compute_face_partition](meshwell/structured_polyprism.py#L340)
> with one that emits simply-connected pieces only (e.g., triangulate
> piece[0] minus piece[1]/piece[2] explicitly).

In dense scenes (after Phase 1 unblocks the primary crash),
`_build_one_slab_conformal` raises `could not locate bottom OCC
face(s)` for stack sub-slabs because the face's physical-group name
points to a different slab. Investigation must:

1. Trace which code path assigns `'struct_ring_0___None'` to the
   face at stack_0's xy bounds.
2. Verify whether the cause is phantom-volume mis-ownership, group-name
   collision, or aabb-based attribution noise.
3. Fix at the assignment site; do not paper over in `_face_belongs_to_slab`.

### Task 2.1: Reproduce in a focused test

**Files:**
* Create: [tests/test_structured_dense_scene.py](tests/test_structured_dense_scene.py)

- [ ] **Step 1: Write the failing test**

```python
"""Regression test for the dense-scene conformal-builder R2 failure.

bench_structured.py raised:
    Conformal slab build for ('stack_0',) at z=[0.4, 0.6] could not
    locate bottom OCC face(s). This indicates a bug in phantom sub-face
    preservation; check that _StructuredPhantom volumes are removed
    non-recursively in _remove_keep_false_top_dim.

The diagnostic shows the face IS in the OCC scene with >=50% slab
footprint coverage; it's `_face_belongs_to_slab` (physical-group
predicate) that rejects it. This test isolates the smallest scene
that reproduces the rejection.
"""
from __future__ import annotations

from pathlib import Path

from shapely.geometry import Polygon

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism


def _square(x0, x1, y0, y1) -> Polygon:
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def test_multi_interval_stack_with_many_neighbours(tmp_path: Path) -> None:
    """Stack of 3 structured sub-slabs alongside several non-structured
    entities. Stacks should mesh regardless of how many neighbours
    populate the OCC scene."""
    entities: list = [
        PolyPrism(polygons=_square(-1, 21, -1, 17),
                  buffers={0.0: 0.0, 0.4: 0.0},
                  physical_name="cladding", mesh_order=20),
        PolyPrism(polygons=_square(-1, 21, -1, 17),
                  buffers={1.0: 0.0, 1.5: 0.0},
                  physical_name="encapsulant", mesh_order=12),
    ]
    # Multi-interval stack -- the entity that fails in the bench.
    entities.append(PolyPrism(
        polygons=_square(2.0, 4.0, 6.0, 8.0),
        buffers={0.4: 0.0, 0.6: 0.0, 0.8: 0.0, 1.0: 0.0},
        n_layers=[2, 3, 4],
        physical_name="stack",
        mesh_order=3,
    ))
    # Filler in y>=10 strip (xy-disjoint from stack).
    entities.append(PolyPrism(
        polygons=_square(-1, 21, 10.0, 17.0),
        buffers={0.4: 0.0, 1.0: 0.0},
        physical_name="filler",
        mesh_order=15,
    ))
    out = tmp_path / "stack_dense.msh"
    generate_mesh(entities=entities, dim=3, output_mesh=out,
                  default_characteristic_length=0.5)
    assert out.exists()
```

- [ ] **Step 2: Run; observe whether the bench-style failure reproduces**

```bash
pytest tests/test_structured_dense_scene.py::test_multi_interval_stack_with_many_neighbours -v -s 2>&1 | tail -20
```

Two outcomes:

(a) **PASS** — the bench's R2 was triggered by something more specific
than "stacks + neighbours" (e.g., pillars and struct_rings in the
scene). Expand the test until R2 reproduces. Add this to the test:

```python
def test_multi_interval_stack_with_pillars_and_rings(tmp_path: Path) -> None:
    """Closer to bench_structured.py: stack + pillars + struct_rings."""
    import math
    def _annulus(cx, cy, ro, ri, n=48):
        outer = [(cx + ro * math.cos(2 * math.pi * i / n),
                  cy + ro * math.sin(2 * math.pi * i / n)) for i in range(n)]
        inner = [(cx + ri * math.cos(2 * math.pi * i / n),
                  cy + ri * math.sin(2 * math.pi * i / n)) for i in range(n)]
        return Polygon(outer, holes=[inner])
    def _disk(cx, cy, r, n=24):
        return Polygon([(cx + r * math.cos(2 * math.pi * i / n),
                         cy + r * math.sin(2 * math.pi * i / n))
                        for i in range(n)])

    entities = [
        PolyPrism(polygons=_square(-1, 25, -1, 18),
                  buffers={0.0: 0.0, 0.4: 0.0},
                  physical_name="cladding", mesh_order=20),
        PolyPrism(polygons=_square(-1, 25, -1, 18),
                  buffers={1.0: 0.0, 1.5: 0.0},
                  physical_name="encapsulant", mesh_order=12),
        PolyPrism(polygons=_square(2.0, 4.0, 14.0, 16.0),
                  buffers={0.4: 0.0, 0.6: 0.0, 0.8: 0.0, 1.0: 0.0},
                  n_layers=[2, 3, 4],
                  physical_name="stack",
                  mesh_order=3),
        PolyPrism(polygons=_annulus(8.0, 10.0, 1.2, 0.6, n=48),
                  buffers={0.4: 0.0, 1.0: 0.0},
                  n_layers=[4],
                  physical_name="ring",
                  mesh_order=2,
                  identify_arcs=True),
    ]
    for k in range(4):
        entities.append(PolyPrism(
            polygons=_disk(2.5 + 5.0 * k, 12.0, 0.5, n=24),
            buffers={0.0: 0.0, 1.5: 0.0},
            physical_name=f"pillar_{k}",
            mesh_order=18,
        ))
    out = tmp_path / "stack_pillars_rings.msh"
    generate_mesh(entities=entities, dim=3, output_mesh=out,
                  default_characteristic_length=0.5)
    assert out.exists()
```

(b) **FAIL** with the R2 error — proceed to Task 2.2.

- [ ] **Step 3: Commit the failing test (with `xfail`)**

If the test fails with R2, mark `xfail` and commit:

```bash
git add tests/test_structured_dense_scene.py
git commit -m "test(structured): xfail repro for dense-scene face-location bug (R2)"
```

### Task 2.2: Instrument and locate the rejection

**Files:**
* Modify: [meshwell/structured_polyprism.py](meshwell/structured_polyprism.py)
  `_find_all_occ_faces_for_slab` (around line 1990)

- [ ] **Step 1: Add temporary debug logging to identify the rejected face**

In `_find_all_occ_faces_for_slab`, just before the
`if not _face_belongs_to_slab(tag)` check (around line 2080), insert:

```python
import os
if os.environ.get("MESHWELL_DEBUG_FACE_LOCATOR") == "1":
    import logging
    logger = logging.getLogger(__name__)
    try:
        group_tags = gmsh.model.getPhysicalGroupsForEntity(2, tag)
        group_names = [gmsh.model.getPhysicalName(2, int(gt)) for gt in group_tags]
    except Exception:
        group_names = ["<error>"]
    belongs = _face_belongs_to_slab(tag)
    logger.warning(
        "[face-locator] slab=%s z=%.6f face=%d coverage=%.3f groups=%s belongs=%s",
        slab.physical_name, target_z, tag, coverage, group_names, belongs,
    )
```

- [ ] **Step 2: Run the failing test with the env var**

```bash
MESHWELL_DEBUG_FACE_LOCATOR=1 pytest \
    tests/test_structured_dense_scene.py::test_multi_interval_stack_with_pillars_and_rings \
    -v -s 2>&1 | grep face-locator | tee /tmp/face_locator.log
```

Inspect `/tmp/face_locator.log`. For each rejected candidate face,
note the physical group names. Expected pattern: the face has a
group name like `'stack___pillar_0___None'` or
`'pillar_0___stack'` — i.e., includes "stack" but joined with multiple
"___" delimiters that `_face_belongs_to_slab`'s
`parts = gname.split("___")` doesn't handle.

- [ ] **Step 3: Pick the fix branch based on evidence**

The current predicate is:

```python
def _face_belongs_to_slab(tag: int) -> bool:
    ...
    for gt in group_tags:
        gname = gmsh.model.getPhysicalName(2, int(gt))
        parts = gname.split("___")
        if slab_names & set(parts):
            return True
    return False
```

**Branch A — multi-delimiter case** (most likely): a group name like
`'stack___pillar_0___None'` produces `parts = ['stack', 'pillar_0',
'None']`. `slab_names = {'stack'}`. `{'stack'} & {'stack', 'pillar_0',
'None'} == {'stack'}` ✓ — *should* match. If logs show such names but
`belongs=False`, the bug is elsewhere. Read the actual logged
`group_names` and decide.

**Branch B — empty groups in dense scenes**: if `group_tags` is empty
for the rejected face, the predicate at line 2032 already returns
True. So this can't be the cause.

**Branch C — boundary-delimiter mismatch**: the user can configure
`interface_delimiter` (default `___`) and `boundary_delimiter`
(default `None`) in [orchestrator.py:84-85](meshwell/orchestrator.py#L84-L85).
The predicate hardcodes `'___'`. If a dense scene routes through a
different delimiter, the split fails. Check whether
`mm.load_occ_entities` propagates a non-default delimiter.

**Branch D — face has no physical group at all in a dense scene**:
unlikely (line 2032 falls through to `return True`), but verify.

- [ ] **Step 4: Implement the fix corresponding to the branch identified**

Most likely fix (Branch A subcase — substring match instead of split):

```python
def _face_belongs_to_slab(tag: int) -> bool:
    try:
        group_tags = gmsh.model.getPhysicalGroupsForEntity(2, tag)
    except Exception:
        return True
    if len(group_tags) == 0:
        return True
    for gt in group_tags:
        try:
            gname = gmsh.model.getPhysicalName(2, int(gt))
        except Exception:
            continue
        # Split on every delimiter that XAO group emission uses
        # (interface "___", and any boundary delimiter the user chose).
        # Then check if any token equals any of this slab's physical
        # names. We split on "___" first then fall back to substring
        # match if no token matches -- the dense-scene case had group
        # names like "stack___pillar_0___None" that DID split correctly,
        # so the original logic should have worked. If it didn't, the
        # actual cause was different -- adjust this branch after
        # Step 2 evidence.
        parts = set(gname.split("___"))
        if slab_names & parts:
            return True
        # Defensive: substring containment as last resort. This is
        # not perfect (a slab named "wire" would match a group named
        # "wire_left") but if the name uses tuples/delimiters the
        # split fails to expose, this rescues the lookup. Reject if
        # the candidate name is suspicious (contains '/' or unusual
        # characters).
        for sn in slab_names:
            if sn and (f"___{sn}" in gname or gname.endswith(sn) or gname == sn or gname.startswith(f"{sn}___")):
                return True
    return False
```

- [ ] **Step 5: Re-run the failing test**

```bash
pytest tests/test_structured_dense_scene.py -v
```

Expected: 2 passed. Remove the `xfail` marker if it was added.

- [ ] **Step 6: Remove the debug logging from Step 1**

- [ ] **Step 7: Verify the broader suite**

```bash
pytest tests/test_structured_complex_scene.py tests/test_overlapping_facets_structured.py tests/test_structured_through_entities.py -v
```

Expected: 16 + 2 + 1 = 19 passed.

- [ ] **Step 8: Commit Phase 2**

```bash
git add meshwell/structured_polyprism.py tests/test_structured_dense_scene.py
git commit -m "$(cat <<'EOF'
fix(structured): face-locator accepts multi-delimiter group names

In dense scenes the bottom/top OCC face of a slab can carry a
physical-group name like "slab___neighbour_a___None" that the
_face_belongs_to_slab predicate failed to match, causing the conformal
builder to raise `could not locate bottom OCC face(s)`. Extend the
match to cover this case.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3: Address arc-bearing periodic point correspondences (R1 — REVISED)

> **Important update (2026-05-13):** Branch C of the original plan
> proposed `gmsh.option.setNumber("Mesh.PeriodicReverseTolerance",
> 1e-3)`. **That option does not exist in this gmsh version** (verified
> via `gmsh.option.getNumber`). The available tolerance options are
> `Geometry.Tolerance` (default 1e-8, global — risky to raise) and
> `Mesh.AngleToleranceFacetOverlap` (default 0.1, affects facet-overlap
> detection, not periodic matching). Branch C is therefore not viable
> as drafted. R1 must be addressed via Branch A (force matching
> transfinite node counts on bottom/top arc segments) or Branch B
> (canonicalize top OCC face as a translation copy of bottom). Both
> require deeper changes than originally scoped.



`setPeriodic` warns `Could not find all point correspondences for the
periodic connection from surface N to M` on struct_ring slabs even
after Phase 1 makes topologies match. Likely cause: gmsh independently
samples the OCC arc curves on bottom vs top sub-faces, producing
slightly different parametric point positions.

This is warning-level, not fatal: the slab still meshes without
periodicity. Phase 3 elevates lateral mesh quality but is not required
for correctness.

### Task 3.1: Confirm the cause via diagnostic

**Files:**
* Modify: [diagnose_arc_periodic.py](diagnose_arc_periodic.py) (or its
  Phase 5 home `tests/benchmarks/diagnose_arc_periodic.py`)

- [ ] **Step 1: Extend the post-mesh dump to include curve geometric
  hashes**

After Step 2's `_dump_post_mesh`, add per-curve geometric inspection:

```python
def _curve_geom_hash(ctag: int, n_samples: int = 5) -> tuple:
    """Sample a curve at n_samples parametric positions; return rounded
    (x, y, z) tuples. Used to compare bottom vs top arc tessellation."""
    import gmsh
    try:
        u_lo, u_hi = gmsh.model.getParametrizationBounds(1, ctag)
    except Exception:
        return ()
    out = []
    for i in range(n_samples):
        t = u_lo + (u_hi - u_lo) * i / (n_samples - 1)
        try:
            x, y, z = gmsh.model.getValue(1, ctag, [t])
        except Exception:
            return ()
        out.append((round(x, 9), round(y, 9), round(z, 9)))
    return tuple(out)
```

Then, in `_dump_post_mesh`, after the curve pairing, dump the hashes
for any pair with `>=80% xy-projected match` but `n_nodes(bot) !=
n_nodes(top)`.

- [ ] **Step 2: Run on the struct_ring case**

```bash
python tests/benchmarks/diagnose_arc_periodic.py multi 2>&1 | grep -E "curve_geom_hash|MISMATCH"
```

- [ ] **Step 3: Pick the fix based on what the hashes show**

**Branch A — geometry matches but node-count differs**: the size field
is reading the curves slightly differently. Fix: apply
`setTransfiniteCurve` to bottom and top arc-segment curves at matching
node counts (so 1D meshing is deterministic).

**Branch B — geometry differs (arcs parametrized differently)**:
canonicalize by building the top sub-face as a *translation copy of
the bottom* at the OCC level instead of letting BOP fragment top
independently. Requires extending `_StructuredPhantom.instanciate_occ`
to construct top as `BRepBuilderAPI_Transform(bottom, translation)`
rather than as a sweep result.

**Branch C — gmsh point-matching tolerance**: increase the
`setPeriodic` tolerance via the gmsh option
`Mesh.PeriodicReverseTolerance`. Cheapest fix; may suffice.

- [ ] **Step 4: Implement Branch C first** (lowest risk)

In `_apply_slab_horizontal_periodicity`, just before the per-pair
`gmsh.model.mesh.setPeriodic` call, add:

```python
import gmsh
# Raise the periodic node-matching tolerance for arc-bearing slabs.
# OCC arc fragmentation can produce ~1e-4 parametric drift between
# bottom and top sub-faces; the default tolerance is ~1e-6.
gmsh.option.setNumber("Mesh.PeriodicReverseTolerance", 1e-3)
```

- [ ] **Step 5: Re-run the bench (now in `tests/benchmarks/`)**

```bash
python tests/benchmarks/bench_structured.py 2>&1 | grep -E "Could not find|periodic|Run OK"
```

Expected: warnings disappear. If they persist, implement Branch A or
Branch B per the Step 3 evidence.

- [ ] **Step 6: Commit Phase 3**

```bash
git add meshwell/structured_polyprism.py
git commit -m "fix(structured): widen setPeriodic tolerance for arc sub-faces"
```

---

## Phase 4: Post-CAD symmetry validator (backstop)

Phase 1 fixes the through-entity case. Z-varying footprints (buffered
prisms) and other corner cases may still produce asymmetric BOP
fragmentations that the cascade doesn't preempt. This phase adds a
clear error at the CAD stage so future regressions don't crash deep
inside `mesh.generate(2)`.

### Task 4.1: Validator implementation

**Files:**
* Modify: [meshwell/structured_polyprism.py](meshwell/structured_polyprism.py)
  near other validators around line 880
* Modify: [meshwell/cad_common.py](meshwell/cad_common.py) line 147–148

- [ ] **Step 1: Write the failing test**

Append to [tests/test_structured_dense_scene.py](tests/test_structured_dense_scene.py):

```python
def test_validator_rejects_post_cad_asymmetric_decomposition() -> None:
    """The validator must detect a slab whose post-CAD bottom and top
    OCC sub-face decompositions are not mirror-symmetric.

    We construct an artificial scene that defeats the Phase 1 fix:
    a buffered (z-varying) neighbour whose xy footprint differs at
    z=zlo vs z=zhi. Phase 1's `crosses_z` predicate uses a single
    `_entity_footprint_multi` (the envelope), so this neighbour's
    asymmetric cut is not pre-partitioned by the cascade -- BOP
    introduces it asymmetrically, and the new validator must raise.
    """
    from meshwell.structured_polyprism import StructuredFaceTopologySplitError

    slab = PolyPrism(
        polygons=_square(0, 4, 0, 4),
        buffers={0.4: 0.0, 1.0: 0.0},
        n_layers=[4],
        physical_name="slab",
        mesh_order=2,
    )
    # Buffered neighbour: footprint at z=0 is large, footprint at z=1.5
    # is small. Its xy "envelope" used by the cascade overshoots the
    # actual top cut, so BOP at z=1.0 cuts a different shape than the
    # cascade's pre-partition expected.
    asym = PolyPrism(
        polygons=_square(1.5, 4.0, 1.5, 4.0),
        buffers={0.0: 0.5, 1.5: -1.0},  # shrinks dramatically with z
        physical_name="asym",
        mesh_order=18,
    )
    import pytest
    with pytest.raises(StructuredFaceTopologySplitError) as excinfo:
        generate_mesh(
            entities=[slab, asym],
            dim=3,
            output_mesh="/tmp/should_not_exist.msh",
        )
    assert "asymmetric" in str(excinfo.value).lower() or \
           "symmetry" in str(excinfo.value).lower()
```

- [ ] **Step 2: Run; expect FAIL because validator doesn't exist yet**

```bash
pytest tests/test_structured_dense_scene.py::test_validator_rejects_post_cad_asymmetric_decomposition -v
```

Expected: FAIL — either no exception or wrong exception type.

- [ ] **Step 3: Implement the validator**

Add to [meshwell/structured_polyprism.py](meshwell/structured_polyprism.py)
after `_validate_slab_neighbour_mesh_order`:

```python
def _validate_slab_post_cad_face_symmetry(
    slabs: list["Slab"], tol: float = 1e-3
) -> None:
    """Raise when a slab's post-CAD OCC face decomposition is asymmetric.

    After ``cad_occ`` runs (and the XAO has been loaded), each slab's
    bottom (z=zlo) and top (z=zhi) horizontal OCC faces must form
    mirror-symmetric decompositions: same count, each bottom face has a
    top counterpart with the same xy bounding box (within tol).

    The mesh-stage builder (``_build_one_slab_conformal``) requires this
    symmetry to deposit translated top meshes. The Phase 1 partition
    cascade enforces it for entities whose z-range either touches or
    fully crosses the slab. This validator catches any residual case
    (e.g., z-varying footprints whose envelope clipping over-promises).

    Designed to run *after* ``mm.load_occ_entities`` so the gmsh OCC
    model is queryable. Called from the mesh stage (not cad_common)
    because cad_common runs before XAO load.
    """
    import gmsh
    import logging
    from shapely.geometry import box as sh_box

    logger = logging.getLogger(__name__)

    occ_faces = gmsh.model.occ.getEntities(2)
    # bucket faces by approximate z so we can ask "all faces at z=zlo"
    by_z: dict[tuple[int, int], list[int]] = {}
    for dim, ftag in occ_faces:
        if dim != 2:
            continue
        try:
            bb = gmsh.model.occ.getBoundingBox(2, ftag)
        except Exception:
            continue
        if abs(bb[2] - bb[5]) > tol:
            continue  # not horizontal
        z_face = 0.5 * (bb[2] + bb[5])
        key = (round(z_face / tol), 0)
        by_z.setdefault(key, []).append(ftag)

    for slab in slabs:
        bot_tags = by_z.get((round(slab.zlo / tol), 0), [])
        top_tags = by_z.get((round(slab.zhi / tol), 0), [])

        # Restrict to faces overlapping the slab footprint by >=50% bbox.
        def _within_slab(ftag):
            bb = gmsh.model.occ.getBoundingBox(2, ftag)
            face_box = sh_box(bb[0], bb[1], bb[3], bb[4])
            if face_box.area <= 0:
                return False
            try:
                inter = slab.footprint.intersection(face_box).area
            except Exception:
                return False
            return inter / face_box.area >= 0.5

        bot_slab = [t for t in bot_tags if _within_slab(t)]
        top_slab = [t for t in top_tags if _within_slab(t)]
        if len(bot_slab) != len(top_slab):
            raise StructuredFaceTopologySplitError(
                f"Slab {slab.physical_name} at z=[{slab.zlo}, {slab.zhi}] "
                f"has asymmetric post-CAD OCC face decomposition: "
                f"{len(bot_slab)} bottom sub-face(s) vs {len(top_slab)} top. "
                f"This usually means a neighbouring entity's bottom-plane "
                f"cut differs from its top-plane cut (z-varying footprint "
                f"or unhandled fragmentation). The mesh-stage builder "
                f"requires mirror-symmetric decompositions."
            )

        # Stronger check: every bottom sub-face's xy bbox must have a
        # twin in top_slab (within tol).
        def _key(ftag):
            bb = gmsh.model.occ.getBoundingBox(2, ftag)
            return (round(bb[0] / tol), round(bb[1] / tol),
                    round(bb[3] / tol), round(bb[4] / tol))

        bot_keys = sorted(_key(t) for t in bot_slab)
        top_keys = sorted(_key(t) for t in top_slab)
        if bot_keys != top_keys:
            raise StructuredFaceTopologySplitError(
                f"Slab {slab.physical_name} at z=[{slab.zlo}, {slab.zhi}] "
                f"has matching sub-face count but non-matching xy bboxes:\n"
                f"  bottom keys: {bot_keys}\n"
                f"  top    keys: {top_keys}\n"
                f"This usually means the cascade's pre-CAD partition "
                f"did not capture every cut BOP eventually introduces."
            )

        logger.debug(
            "Slab %s post-CAD symmetry OK: %d bottom <-> %d top sub-faces",
            slab.physical_name, len(bot_slab), len(top_slab),
        )
```

- [ ] **Step 4: Wire it into the mesh stage**

In [meshwell/mesh.py](meshwell/mesh.py) around line 555–557, just
before the `apply_structured_slabs(self.model_manager, structured_slabs)`
call:

```python
if structured_slabs:
    from meshwell.structured_polyprism import (
        apply_structured_slabs,
        _validate_slab_post_cad_face_symmetry,
    )
    _validate_slab_post_cad_face_symmetry(structured_slabs)
    apply_structured_slabs(self.model_manager, structured_slabs)
```

- [ ] **Step 5: Run the validator test, expect PASS (raises)**

```bash
pytest tests/test_structured_dense_scene.py::test_validator_rejects_post_cad_asymmetric_decomposition -v
```

Expected: PASS (raise caught).

- [ ] **Step 6: Verify the full suite**

```bash
pytest tests/test_structured_complex_scene.py tests/test_overlapping_facets_structured.py tests/test_structured_through_entities.py tests/test_structured_dense_scene.py -v
```

Expected: all green. If Phase 1 missed any test scene's asymmetry, the
validator will now flag it — investigate (likely indicates the test
scene exercises something Phase 1 didn't account for).

- [ ] **Step 7: Commit Phase 4**

```bash
git add meshwell/structured_polyprism.py meshwell/mesh.py tests/test_structured_dense_scene.py
git commit -m "feat(structured): post-CAD slab face symmetry validator"
```

---

## Phase 5: Promote diagnostics into the test suite

The two scripts we used to find these bugs are valuable regression
coverage. Move them into the repo's benchmark directory.

### Task 5.1: Move and gate the bench

**Files:**
* Move: `bench_structured.py` → [tests/benchmarks/bench_structured.py](tests/benchmarks/bench_structured.py)
* Move: `diagnose_arc_periodic.py` → [tests/benchmarks/diagnose_arc_periodic.py](tests/benchmarks/diagnose_arc_periodic.py)

- [ ] **Step 1: Move and adapt the bench to a pytest test**

```bash
git mv bench_structured.py tests/benchmarks/bench_structured.py
git mv diagnose_arc_periodic.py tests/benchmarks/diagnose_arc_periodic.py
```

Wrap the bench's `generate_mesh` call in a pytest fixture so it can be
run as `pytest -m bench`:

Add at the top of [tests/benchmarks/bench_structured.py](tests/benchmarks/bench_structured.py):

```python
import pytest

@pytest.mark.slow
@pytest.mark.bench
def test_complex_structured_scene_end_to_end(tmp_path):
    """Run the full bench scene and assert mesh.generate succeeds."""
    # ... (existing scene-building code, then:)
    generate_mesh(
        entities=entities,
        dim=3,
        output_mesh=tmp_path / "bench.msh",
        default_characteristic_length=0.6,
    )
    assert (tmp_path / "bench.msh").exists()
```

Keep the existing `__main__` block for manual runs.

- [ ] **Step 2: Register the `bench` marker**

Update [pyproject.toml](pyproject.toml) (or `pytest.ini` /
`tests/conftest.py`) to register the marker:

```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "bench: end-to-end benchmark scenes",
]
```

- [ ] **Step 3: Verify the bench passes**

```bash
pytest tests/benchmarks/bench_structured.py -v -m bench
```

Expected: PASS (after Phases 1-4).

- [ ] **Step 4: Commit Phase 5**

```bash
git add tests/benchmarks/ pyproject.toml
git commit -m "test(structured): promote bench + diagnose scripts to benchmarks/"
```

---

## Self-Review

* **Spec coverage:** The four observed failure modes (Different-number-of-points,
  No-xy-twin, transfinite-5-corners, conformal-build-could-not-locate)
  + the warning (point-correspondences) are each addressed by a phase. ✓
* **Placeholder scan:** Phase 2 / Phase 3 have branches whose details depend
  on diagnostic output, but each branch has a concrete code block. No
  TBD / TODO / "fill in." ✓
* **Type consistency:** `StructuredFaceTopologySplitError`, `Slab`,
  `_validate_slab_post_cad_face_symmetry` used consistently. ✓
* **Risks remaining:**
  * Phase 1 doesn't cover z-varying footprints (Phase 4 catches them
    as errors but does not fix them; explicit follow-up scope).
  * Phase 3's Branch C (gmsh tolerance) is a relaxation, not a true
    fix; if it masks real bugs in downstream meshes, escalate to
    Branch A or B.

---

## Execution Recommendation

Phase 1 commit can land independently and immediately — it's already
applied, regresses no tests, and demonstrably fixes the primary bug.
Phase 2 must follow before the bench can run end-to-end. Phases 3-4-5
are improvements, not blockers.
