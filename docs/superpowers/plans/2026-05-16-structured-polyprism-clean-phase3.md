# Structured PolyPrism Clean Rewrite — Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the minimum end-to-end structured-polyprism pipeline. From `PolyPrism(structured=True)` user input through `generate_mesh(...)` to a valid `.msh` file containing wedge (or hex) prism elements in the slab volume. Single-slab, single-piece (no neighbour cuts) only — multi-piece, mid-height cuts, arc provenance, performance optimization all defer to Phase 4.

**Architecture:** Phase 3 adds the **mesh-stage** counterpart to Phase 2's CAD-stage phantom: `meshwell/structured/builder.py` derives the slab's top OCC face mesh by translating the bottom mesh (with boundary nodes mapped via Layer B for fuzzy-tolerant matching), stamps it onto the top OCC faces, then creates one discrete 3D entity per slab with wedge or hex elements bridging bottom-to-top. The orchestrator detects structured entities, runs the full pipeline. `cad_occ.py` gets a minimal hook to accept extra OCP shapes (the phantom sub-prisms) into the global BOP and to expose the `BOPAlgo_Builder` for post-Perform history extraction.

**Tech Stack:** Python 3.12, OCP, gmsh (mesh stage), shapely, pytest. We use gmsh's existing pattern of `mesh.clear([(2, face_tag)])` + `addNodes` + `addElements` to override a 2D mesh (proven in `feat/structured`'s `structured_polyprism.py:2376-2434` and our Phase-0 spike).

**Spec reference:** `docs/superpowers/specs/2026-05-15-structured-polyprism-clean-design.md`, Layer C ("Mesh stage owns the top mesh").

**Phase 2 reference:** `docs/superpowers/plans/2026-05-16-structured-polyprism-clean-phase2.md` provides `build_phantom_shapes`, `extract_phantom_map`, `PhantomMap`.

---

## File Structure

**Create:**
- `meshwell/structured/builder.py` — mesh-stage entry points: `resolve_mesh_plan(plan, entities)`, `apply_structured_mesh(plan, mesh_plan, phantom_map, model_manager)`. (~250-400 LOC at completion.)
- `tests/structured/test_mesh_plan.py` — unit tests for `StructuredMeshPlan` resolver (~80 LOC).
- `tests/structured/test_builder_unit.py` — unit tests for builder helpers using direct gmsh fixtures (~150 LOC).
- `tests/structured/test_end_to_end_minimal.py` — single-slab end-to-end test (~80 LOC).

**Modify:**
- `meshwell/structured/spec.py` — add `StructuredMeshPlan` dataclass + `StructuredMeshOverlapError`.
- `meshwell/structured/__init__.py` — re-export `StructuredMeshPlan`, `resolve_mesh_plan`, `apply_structured_mesh`.
- `meshwell/cad_occ.py` — accept optional `extra_occ_shapes: list[Any]` kwarg in the relevant entry; expose the `BOPAlgo_Builder` via a return value or callback (one new optional `cad_occ_callback` parameter). This is a narrow surgical change — do NOT refactor the rest of cad_occ.
- `meshwell/orchestrator.py` — detect structured entities, run the structured pipeline alongside the existing cad_occ + mesh calls.

**Untouched in Phase 3:**
- `meshwell/structured/plan.py` — Phase 1 is complete; we only consume.
- `meshwell/structured/phantom.py` — Phase 2 is complete; we only consume.
- `meshwell/polyprism.py` — no further changes.
- `meshwell/mesh.py` — no structural changes (we inject from orchestrator side, not from inside mesh).

---

## Tech notes (read before coding)

### Layer C overview (from the spec)

For each slab → for each piece *k*:

1. Mark the piece's output top OCC faces as "we'll handle these — don't let gmsh's auto-mesh interfere". In practice: let gmsh mesh them normally, then `gmsh.model.mesh.clear([(2, top_face)])` and replace.
2. After `gmsh.model.mesh.generate(2)`, read the bottom face's 2D mesh (nodes + triangles/quads).
3. Derive the top mesh by transferring connectivity + computing top node positions:
   - **Boundary nodes** that sit on a piece corner: look up the matching top OCC vertex (via Layer B's `output_vertices[(slab, "top", k, corner)]`), use its actual coordinates (BOP-displaced).
   - **Boundary nodes** that sit on a piece edge interior: walk the top OCC edge at the corresponding parameter as on the bottom edge (more sophisticated; for Phase 3 minimum we approximate as `bottom_xy + (0, 0, h)`, valid when edges are straight vertical extrusions, which is the only case in Phase 3's single-piece-no-neighbour scope).
   - **Interior nodes**: `bottom_xy + (0, 0, h)`.
4. Stamp the derived mesh onto the corresponding output top face(s).
5. Build the slab's volume as a single `gmsh.model.addDiscreteEntity(3, -1, [])` with wedge elements (gmsh type 6) or hex elements (type 5) bridging bottom layer → top layer node by node.

### Element orientation (from Phase-0 spike P3 finding)

A wedge prism is gmsh element type 6 with 6 node tags: `[bot_a, bot_b, bot_c, top_a, top_b, top_c]`. Positive Jacobian requires the bottom triangle to be CCW when viewed from above (= CW from below = outward-bottom-normal-down) AND the top to be CCW when viewed from above (= outward-top-normal-up). Bot and top triangles must have **matching connectivity** (same order of `(a, b, c)` indices).

shapely's `orient(poly, sign=1.0)` from Phase 2 produces CCW exterior. The OCC bottom face inherits this orientation. The OCC top face (built by `BRepPrimAPI_MakePrism`) also inherits CCW-from-above. gmsh's 2D mesher produces triangles consistent with the face's orientation — so as long as we trust gmsh's triangulation orientation and translate it directly to the top, the prism orientation is correct.

### n_layers > 1 handling

For `n_layers > 1` we need intermediate node layers. Layer 0 = bottom (existing OCC node tags). Layer n = top (existing OCC node tags via Layer B). Layers 1..n-1 = interior, get fresh node tags at `bottom_xy + (0, 0, i * height/n)`. Wedge elements are built between each consecutive layer pair, so 1 piece × 2 triangles × n_layers prisms.

### removeDuplicateNodes tolerance

Per spec: `tolerance = 2 × max(slab.fragment_fuzzy_value)`. If `fragment_fuzzy_value` is None on all slabs (Phase 3 minimum case), default to `1e-6`.

### Hooking cad_occ

Don't rewrite cad_occ. Add **one new optional parameter** `extra_occ_shapes: list[Any] | None = None` and **one optional return shape**: cad_occ optionally returns a tuple `(occ_entities, bopalgo_builder)` when called with `return_builder=True`. Default behaviour unchanged. The orchestrator passes `extra_occ_shapes=[p.solid for p in phantom_result.shapes]` and `return_builder=True` when structured entities are present.

If this proves too invasive in practice (e.g. cad_occ's internal flow doesn't easily expose the builder), fall back: have cad_occ accept a `cad_occ_callback: Callable[[BOPAlgo_Builder], None] | None` that gets called with the builder right after `Perform()`. Either way the impact on cad_occ is small.

### Phantom volume non-removal

Per the integration explore: **phantom solids don't need explicit removal**. They get fragmented along with everything else; the OCC writer skips bodies for `mesh_bool=False` entities. For Phase 3 we tag the phantom shapes as `mesh_bool=False` via a thin wrapper class (or extend `OCCLabeledEntity` to accept `is_phantom=True`).

---

## Task 1: `StructuredMeshPlan` dataclass + `resolve_mesh_plan` helper

**Files:**
- Modify: `meshwell/structured/spec.py`
- Create: `tests/structured/test_mesh_plan.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/structured/test_mesh_plan.py`:

```python
"""Tests for StructuredMeshPlan resolution from StructuredPlan + entities."""
from __future__ import annotations

import pytest
from shapely.geometry import Polygon


def _square(x=0, y=0, w=1, h=1) -> Polygon:
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def _structured(polygon, buffers, n_layers, name, recombine=False, mesh_order=1.0):
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    return PolyPrism(
        polygons=polygon,
        buffers=buffers,
        structured=True,
        resolutions=[
            StructuredExtrusionResolutionSpec(n_layers=n_layers, recombine=recombine),
        ],
        physical_name=name,
        mesh_order=mesh_order,
    )


def test_resolve_mesh_plan_single_slab():
    from meshwell.structured import build_plan
    from meshwell.structured.builder import resolve_mesh_plan

    s = _structured(_square(), {0.0: 0.0, 1.0: 0.0}, [3], "s")
    plan = build_plan([s])
    mp = resolve_mesh_plan(plan, [s])
    assert len(mp.slabs) == 1
    assert mp.n_layers == (3,)
    assert mp.recombine == (False,)


def test_resolve_mesh_plan_multi_z_intervals():
    from meshwell.structured import build_plan
    from meshwell.structured.builder import resolve_mesh_plan

    s = _structured(_square(), {0.0: 0.0, 1.0: 0.0, 2.5: 0.0}, [2, 5], "s")
    plan = build_plan([s])
    mp = resolve_mesh_plan(plan, [s])
    # Two slabs; n_layers parallel by slab order.
    assert mp.n_layers == (2, 5)


def test_resolve_mesh_plan_recombine_true():
    from meshwell.structured import build_plan
    from meshwell.structured.builder import resolve_mesh_plan

    s = _structured(_square(), {0.0: 0.0, 1.0: 0.0}, [2], "s", recombine=True)
    plan = build_plan([s])
    mp = resolve_mesh_plan(plan, [s])
    assert mp.recombine == (True,)


def test_resolve_mesh_plan_overlap_mismatch_raises():
    """Two overlapping slabs whose owning specs have different n_layers raise."""
    from meshwell.structured import build_plan
    from meshwell.structured.builder import resolve_mesh_plan
    from meshwell.structured.spec import StructuredMeshOverlapError

    # Two structured prisms covering the same xy with same z but different n_layers
    # would be caught by Phase-1 Policy B at plan time already. So this test
    # focuses on the OverlapPair cross-check: synthesize an OverlapPair where
    # the loser's n_layers != the winner's, by directly constructing the plan.
    from meshwell.structured.spec import OverlapPair, Slab, StructuredPlan

    winner = Slab(
        footprint=_square(), zlo=0.0, zhi=1.0,
        physical_name=("a",), source_index=0, z_interval_index=0,
        mesh_order=1.0, face_partition=[_square()],
    )
    # Loser spec would have had n_layers=5 (winner spec: n_layers=3)
    plan = StructuredPlan(
        slabs=(winner,),
        z_planes=(0.0, 1.0),
        overlaps=(
            OverlapPair(
                winner_slab_index=0,
                loser_source_index=1,
                loser_z_interval_index=0,
                z_extent=(0.0, 1.0),
            ),
        ),
    )

    # Construct entities such that:
    #   entities[0] is the winner with n_layers=[3]
    #   entities[1] is the (would-be) loser with n_layers=[5]
    e_winner = _structured(_square(), {0.0: 0.0, 1.0: 0.0}, [3], "a")
    e_loser = _structured(_square(), {0.0: 0.0, 1.0: 0.0}, [5], "b")
    with pytest.raises(StructuredMeshOverlapError, match="n_layers"):
        resolve_mesh_plan(plan, [e_winner, e_loser])
```

- [ ] **Step 2: Run; expect ImportError on StructuredMeshOverlapError, resolve_mesh_plan, StructuredMeshPlan.**

`.venv/bin/python -m pytest tests/structured/test_mesh_plan.py -v`

- [ ] **Step 3: Add `StructuredMeshPlan` + `StructuredMeshOverlapError` to `meshwell/structured/spec.py`**

Append (after `PhantomMap`):

```python
class StructuredMeshOverlapError(ValueError):
    """Raised when an overlap-pair winner and loser have different n_layers.

    Plan-stage Policy B catches direct overlap mismatches at slab
    construction. Mesh-stage catches the case where an OverlapPair
    records a winner/loser whose spec n_layers actually disagree —
    paranoid double-check before we commit to the loser-was-dominated
    decision.
    """


@dataclass(frozen=True)
class StructuredMeshPlan:
    """Output of ``resolve_mesh_plan(plan, entities)``.

    Carries the mesh-stage parameters resolved from each slab's owning
    ``StructuredExtrusionResolutionSpec``. Parallel arrays: index i in
    ``n_layers`` / ``recombine`` corresponds to ``plan.slabs[i]``.
    """
    slabs: tuple["Slab", ...]
    n_layers: tuple[int, ...]
    recombine: tuple[bool, ...]
```

(The forward-quoted `Slab` is needed because Slab is defined above; if it's already imported in scope, the quotes are optional.)

- [ ] **Step 4: Create `meshwell/structured/builder.py`** with `resolve_mesh_plan`:

```python
"""Phase-3: mesh-stage builder (Layer C).

Public entry points (added incrementally):

- :func:`resolve_mesh_plan` — second-pass over the spec list to attach
  ``n_layers`` and ``recombine`` to each slab; also cross-checks
  Phase-1 OverlapPairs.
- :func:`apply_structured_mesh` — full mesh-stage execution: stamp
  derived top meshes, build discrete 3D entities per slab, run global
  removeDuplicateNodes.
"""
from __future__ import annotations

from typing import Any

from meshwell.structured.spec import (
    StructuredExtrusionResolutionSpec,
    StructuredMeshOverlapError,
    StructuredMeshPlan,
    StructuredPlan,
)


def _spec_of(entity: Any) -> StructuredExtrusionResolutionSpec | None:
    for r in getattr(entity, "resolutions", None) or []:
        if isinstance(r, StructuredExtrusionResolutionSpec):
            return r
    return None


def resolve_mesh_plan(
    plan: StructuredPlan, entities: list[Any]
) -> StructuredMeshPlan:
    """Look up (n_layers, recombine) for each slab via its owning spec.

    Cross-checks every ``OverlapPair`` in the plan: if the loser slab's
    spec n_layers != the winner's, raises
    ``StructuredMeshOverlapError``. This is a paranoid double-check;
    Phase-1's Policy B already catches direct mismatches at plan time.
    """
    n_layers_list: list[int] = []
    recombine_list: list[bool] = []
    for slab in plan.slabs:
        owner = entities[slab.source_index]
        spec = _spec_of(owner)
        if spec is None:
            raise StructuredMeshOverlapError(
                f"Slab {slab.physical_name} source entity has no "
                f"StructuredExtrusionResolutionSpec attached."
            )
        n_layers_list.append(int(spec.n_layers[slab.z_interval_index]))
        recombine_list.append(bool(spec.recombine))

    # OverlapPair cross-check.
    for op in plan.overlaps:
        winner = plan.slabs[op.winner_slab_index]
        winner_spec = _spec_of(entities[winner.source_index])
        loser_owner = entities[op.loser_source_index]
        loser_spec = _spec_of(loser_owner)
        if winner_spec is None or loser_spec is None:
            continue
        winner_n = winner_spec.n_layers[winner.z_interval_index]
        loser_n = loser_spec.n_layers[op.loser_z_interval_index]
        if winner_n != loser_n:
            raise StructuredMeshOverlapError(
                f"OverlapPair winner {winner.physical_name} "
                f"(n_layers={winner_n}) and loser source_index="
                f"{op.loser_source_index} (n_layers={loser_n}) "
                f"at z={op.z_extent}: n_layers must match for the "
                f"overlap to be valid."
            )

    return StructuredMeshPlan(
        slabs=plan.slabs,
        n_layers=tuple(n_layers_list),
        recombine=tuple(recombine_list),
    )
```

- [ ] **Step 5: Run; expect 4 PASSES**

`.venv/bin/python -m pytest tests/structured/test_mesh_plan.py -v`

- [ ] **Step 6: Commit**

```bash
git add meshwell/structured/spec.py meshwell/structured/builder.py tests/structured/test_mesh_plan.py
git commit -m "$(cat <<'EOF'
feat(structured): StructuredMeshPlan + resolve_mesh_plan

Mesh-stage second pass: walks the StructuredPlan slabs, looks up
each slab's owning entity, retrieves (n_layers, recombine) from its
StructuredExtrusionResolutionSpec, and packages them as parallel
arrays into a frozen StructuredMeshPlan.

Cross-checks every OverlapPair: if the would-be loser's spec n_layers
differs from the winner's, raises StructuredMeshOverlapError. Phase-1
Policy B already catches direct mismatches; this is a defence in
depth.
EOF
)"
```

---

## Task 2: cad_occ hook — accept extra OCP shapes + expose builder

**Files:**
- Modify: `meshwell/cad_occ.py`
- Test: `tests/structured/test_cad_occ_phantom_hook.py`

Implementer: first **read `meshwell/cad_occ.py` lines 270-370** to confirm the exact lines to modify. The plan describes intent; the implementer adapts to the actual code shape.

**Intent:**
1. Add an optional `extra_occ_shapes: list[Any] | None = None` parameter to `cad_occ()`. These shapes get added as additional arguments to the BOPAlgo_Builder before `Perform()`.
2. Add an optional `cad_occ_callback: Callable[[Any], None] | None = None` parameter. When non-None, it's called with the BOPAlgo_Builder object **right after Perform() and before history extraction**. This lets the caller capture the builder for downstream history walking.

These two parameters are independent and default to None. When both are None, cad_occ behaviour is identical to before.

- [ ] **Step 1: Inspect the current cad_occ signature and BOP location**

```bash
sed -n '110,150p' meshwell/cad_occ.py
sed -n '290,360p' meshwell/cad_occ.py
```

Identify:
- The `cad_occ()` function definition line.
- The `BOPAlgo_Builder()` instantiation line (around 310).
- The `.Perform()` call line (around 331).
- The history-extraction loop (around 350).

- [ ] **Step 2: Write the failing test**

Create `tests/structured/test_cad_occ_phantom_hook.py`:

```python
"""Tests for the cad_occ hook that lets the structured pipeline:
1. Push extra OCP shapes into the global BOP.
2. Receive the BOPAlgo_Builder after Perform() for history extraction.
"""
from __future__ import annotations

import pytest
from shapely.geometry import Polygon


def _square() -> Polygon:
    return Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


def test_cad_occ_accepts_extra_occ_shapes_kwarg():
    """Smoke test: passing extra_occ_shapes=[] is a no-op vs no kwarg."""
    from meshwell.cad_occ import cad_occ
    from meshwell.polyprism import PolyPrism

    p = PolyPrism(
        polygons=_square(),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="p",
    )
    a = cad_occ([p])
    b = cad_occ([p], extra_occ_shapes=[])
    assert len(a) == len(b)


def test_cad_occ_callback_invoked_with_builder():
    """When passed, callback receives the BOPAlgo_Builder post-Perform."""
    from meshwell.cad_occ import cad_occ
    from meshwell.polyprism import PolyPrism

    captured: list = []

    def cb(builder):
        captured.append(builder)

    p = PolyPrism(
        polygons=_square(),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="p",
    )
    cad_occ([p], cad_occ_callback=cb)
    assert len(captured) == 1
    # The captured object should expose Modified() — the BOP history API.
    assert hasattr(captured[0], "Modified")
    assert hasattr(captured[0], "Generated")


def test_cad_occ_extra_shapes_participate_in_fragmentation():
    """An extra phantom shape should get fragmented against the entities."""
    from meshwell.cad_occ import cad_occ
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import build_plan
    from meshwell.structured.phantom import build_phantom_shapes
    from meshwell.structured import StructuredExtrusionResolutionSpec

    p = PolyPrism(
        polygons=_square(),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="p",
    )
    plan = build_plan([p])
    phantom_result = build_phantom_shapes(plan)
    extra = [s.solid for s in phantom_result.shapes]

    captured: list = []
    cad_occ(
        [p],
        extra_occ_shapes=extra,
        cad_occ_callback=lambda b: captured.append(b),
    )
    # Callback was invoked; builder exposes Modified() for any of the extras.
    assert len(captured) == 1
    for s in extra:
        # Modified() returns a TopTools_ListOfShape; we just check the call
        # doesn't raise and the input was tracked.
        modified = captured[0].Modified(s)
        assert modified is not None
```

- [ ] **Step 3: Run; expect TypeError on unexpected kwarg.**

`.venv/bin/python -m pytest tests/structured/test_cad_occ_phantom_hook.py -v`

- [ ] **Step 4: Implement the cad_occ hook**

Edit `meshwell/cad_occ.py`:

a. Add `extra_occ_shapes: list[Any] | None = None` and `cad_occ_callback` to the `cad_occ()` function signature. Default both to None.

b. In `_fragment_all()` (the method that calls `BOPAlgo_Builder.Perform()`), after the existing loop that adds entity shapes to the builder, append a loop that adds `self._extra_occ_shapes` (or however you thread it through):

```python
for s in (extra_occ_shapes or []):
    builder.AddArgument(s)
```

c. Immediately after `builder.Perform()`, invoke the callback if non-None:

```python
if cad_occ_callback is not None:
    cad_occ_callback(builder)
```

How to thread the parameters from the top-level `cad_occ()` into `_fragment_all()` depends on how the class is structured. The implementer may need to pass them as instance attributes (`self._extra_occ_shapes = extra_occ_shapes`) and read in `_fragment_all()`. Whatever pattern matches the existing code.

- [ ] **Step 5: Run; expect 3 PASSES**

`.venv/bin/python -m pytest tests/structured/test_cad_occ_phantom_hook.py -v`

- [ ] **Step 6: Regression check — existing tests still pass**

`.venv/bin/python -m pytest tests/test_cad_occ.py tests/test_cad.py -x -q --ignore=tests/test_structured_complex_scene.py --ignore=tests/test_overlapping_facets_structured.py 2>&1 | tail -10`

Expected: same green baseline as before (the hook is additive; default behaviour is unchanged).

- [ ] **Step 7: Commit**

```bash
git add meshwell/cad_occ.py tests/structured/test_cad_occ_phantom_hook.py
git commit -m "$(cat <<'EOF'
feat(structured): cad_occ hook for phantom shapes + BOP builder access

Adds two optional parameters to cad_occ():

- extra_occ_shapes: list[TopoDS_*] | None — extra OCP solids added as
  BOPAlgo_Builder arguments alongside entity shapes. Used by the
  structured pipeline to inject phantom sub-prisms into the global
  fragment.
- cad_occ_callback: Callable[[BOPAlgo_Builder], None] | None — called
  with the builder immediately after Perform(), before history
  extraction. Lets the structured pipeline walk Modified() / Generated()
  to build the PhantomMap.

Both default to None; existing call sites unchanged. Surgical addition,
no refactor of the cad_occ flow.
EOF
)"
```

---

## Task 3: `builder._stamp_top_face_mesh` — derive top mesh from bottom

**Files:**
- Modify: `meshwell/structured/builder.py`
- Create: `tests/structured/test_builder_unit.py`

The smallest building block. Given a bottom OCC face (with its 2D gmsh mesh already generated), a top OCC face (same face partition piece, possibly displaced by ~fuzzy), and the Layer B vertex map for boundary correspondence, derive the top mesh and stamp it.

For Phase 3 minimum, single-piece case: bottom and top faces are 1 face each. Bottom mesh exists from gmsh.model.mesh.generate(2). Top mesh is cleared via gmsh.model.mesh.clear, then re-stamped with bottom triangulation translated to top vertex positions.

- [ ] **Step 1: Write the failing test**

Create `tests/structured/test_builder_unit.py`:

```python
"""Unit tests for builder helpers using direct gmsh fixtures."""
from __future__ import annotations

import gmsh
import pytest
from shapely.geometry import Polygon


def _square(x=0, y=0, w=1, h=1) -> Polygon:
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


@pytest.fixture
def gmsh_session():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("test")
    try:
        yield
    finally:
        gmsh.finalize()


def _make_box_in_gmsh_and_mesh_2d(z_lo: float, z_hi: float):
    """Create a unit box in gmsh.model.occ, sync, mesh 2D. Returns (bot_face_tag, top_face_tag)."""
    gmsh.model.occ.addBox(0, 0, z_lo, 1, 1, z_hi - z_lo)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.3)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.3)
    gmsh.model.mesh.generate(2)
    # Identify horizontal faces by bbox z.
    bot_tag = top_tag = None
    for dim, tag in gmsh.model.getEntities(2):
        bb = gmsh.model.getBoundingBox(dim, tag)
        if abs(bb[2] - z_lo) < 1e-6 and abs(bb[5] - z_lo) < 1e-6:
            bot_tag = tag
        elif abs(bb[2] - z_hi) < 1e-6 and abs(bb[5] - z_hi) < 1e-6:
            top_tag = tag
    assert bot_tag is not None and top_tag is not None
    return bot_tag, top_tag


def test_stamp_top_face_mesh_replaces_top_with_translated_bottom(gmsh_session):
    """Bottom mesh on bot_face -> derived top mesh on top_face with same connectivity."""
    from meshwell.structured.builder import _stamp_top_face_mesh

    bot_tag, top_tag = _make_box_in_gmsh_and_mesh_2d(z_lo=0.0, z_hi=1.0)
    # Bottom mesh exists; read it.
    bot_node_tags_before, bot_coords_before, _ = gmsh.model.mesh.getNodes(
        2, bot_tag, includeBoundary=True
    )
    # Apply the stamp: clear top, deposit translated copy of bottom.
    _stamp_top_face_mesh(
        bottom_face_tag=bot_tag,
        top_face_tag=top_tag,
        zlo=0.0,
        zhi=1.0,
    )
    # Top now has the same number of nodes as bottom (we translated each).
    top_node_tags, top_coords, _ = gmsh.model.mesh.getNodes(
        2, top_tag, includeBoundary=True
    )
    assert len(top_node_tags) == len(bot_node_tags_before)
    # Z of every top node should be ~zhi.
    import numpy as np
    top_z = np.asarray(top_coords, dtype=float).reshape(-1, 3)[:, 2]
    assert (abs(top_z - 1.0) < 1e-6).all()


def test_stamp_top_face_mesh_produces_matching_triangle_count(gmsh_session):
    from meshwell.structured.builder import _stamp_top_face_mesh

    bot_tag, top_tag = _make_box_in_gmsh_and_mesh_2d(z_lo=0.0, z_hi=1.0)
    bot_types_before, bot_tags_before, _ = gmsh.model.mesh.getElements(2, bot_tag)
    n_bot_tris = sum(len(t) for et, t in zip(bot_types_before, bot_tags_before) if et == 2)

    _stamp_top_face_mesh(
        bottom_face_tag=bot_tag,
        top_face_tag=top_tag,
        zlo=0.0,
        zhi=1.0,
    )
    top_types, top_tags_, _ = gmsh.model.mesh.getElements(2, top_tag)
    n_top_tris = sum(len(t) for et, t in zip(top_types, top_tags_) if et == 2)
    assert n_top_tris == n_bot_tris
```

- [ ] **Step 2: Run; expect ImportError on _stamp_top_face_mesh.**

`.venv/bin/python -m pytest tests/structured/test_builder_unit.py -v`

- [ ] **Step 3: Append `_stamp_top_face_mesh` to `meshwell/structured/builder.py`**

```python
import numpy as np


def _stamp_top_face_mesh(
    bottom_face_tag: int,
    top_face_tag: int,
    zlo: float,
    zhi: float,
) -> dict[int, int]:
    """Replace the top OCC face's 2D mesh with a translated copy of the bottom's.

    Phase 3 minimum: pure translation. Boundary node positions for the
    top are computed as ``bottom_xy + (0, 0, zhi - zlo)``. The full Layer
    B vertex-map lookup (for BOP-displaced boundary nodes) lands in
    Phase 4.

    Returns a dict mapping bottom node tag -> newly-allocated top node
    tag (so the volume builder can use it for prism construction).
    """
    import gmsh

    height = zhi - zlo
    bot_node_tags_arr, bot_coords_flat, _ = gmsh.model.mesh.getNodes(
        2, bottom_face_tag, includeBoundary=True
    )
    bot_node_tags = np.asarray(bot_node_tags_arr, dtype=np.int64)
    bot_coords = np.asarray(bot_coords_flat, dtype=float).reshape(-1, 3)

    # Read bottom element triangles (we'll re-stamp with the same connectivity).
    elem_types, elem_tags_per_type, elem_nodes_per_type = gmsh.model.mesh.getElements(
        2, bottom_face_tag
    )
    # Collect triangles only (element type 2 in gmsh).
    bot_tri_nodes: list[np.ndarray] = []
    for et, en in zip(elem_types, elem_nodes_per_type):
        if et == 2:
            bot_tri_nodes.append(np.asarray(en, dtype=np.int64).reshape(-1, 3))
    if not bot_tri_nodes:
        return {}
    bot_triangles = np.concatenate(bot_tri_nodes, axis=0)

    # Clear the existing top-face 2D mesh.
    try:
        gmsh.model.mesh.clear([(2, top_face_tag)])
    except Exception:
        pass

    # Allocate fresh node tags for the top mesh.
    next_tag = int(gmsh.model.mesh.getMaxNodeTag()) + 1
    bot_to_top_tag: dict[int, int] = {}
    top_coords_flat: list[float] = []
    top_node_tags: list[int] = []
    for i, bt in enumerate(bot_node_tags):
        new_tag = next_tag + i
        bot_to_top_tag[int(bt)] = new_tag
        top_node_tags.append(new_tag)
        top_coords_flat.extend([bot_coords[i, 0], bot_coords[i, 1], zlo + height])

    gmsh.model.mesh.addNodes(2, top_face_tag, top_node_tags, top_coords_flat)

    # Stamp triangles with the bottom->top tag mapping.
    top_tri_nodes_flat: list[int] = []
    for tri in bot_triangles:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        top_tri_nodes_flat.extend([bot_to_top_tag[a], bot_to_top_tag[b], bot_to_top_tag[c]])

    next_elem_tag = int(gmsh.model.mesh.getMaxElementTag()) + 1
    elem_tags = list(range(next_elem_tag, next_elem_tag + bot_triangles.shape[0]))
    gmsh.model.mesh.addElements(
        2, top_face_tag, [2], [elem_tags], [top_tri_nodes_flat]
    )

    return bot_to_top_tag
```

- [ ] **Step 4: Run; expect 2 PASSES**

`.venv/bin/python -m pytest tests/structured/test_builder_unit.py -v`

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/builder.py tests/structured/test_builder_unit.py
git commit -m "$(cat <<'EOF'
feat(structured): builder._stamp_top_face_mesh (translate bottom -> top)

Reads the bottom OCC face's 2D mesh, clears the top face's mesh,
deposits a translated copy with fresh node tags. Returns the
bottom-node-tag -> top-node-tag map for the volume builder.

Phase 3 minimum: pure translation by (0, 0, height). Boundary nodes
get displaced top OCC vertex coords via the Layer B map in Phase 4.
EOF
)"
```

---

## Task 4: `builder._build_slab_volume` — discrete 3D entity with wedge prisms

**Files:**
- Modify: `meshwell/structured/builder.py`
- Modify: `tests/structured/test_builder_unit.py`

Given a bottom face's mesh + the top-tag map from Task 3 + `n_layers`, build the slab's volume as a discrete 3D entity with wedge or hex elements.

For `n_layers > 1` we need to create `(n_layers - 1)` intermediate node layers between bottom and top, with linear z interpolation.

- [ ] **Step 1: Append the failing tests** to `tests/structured/test_builder_unit.py`:

```python
def test_build_slab_volume_single_layer_produces_wedges(gmsh_session):
    """Single-layer slab: 2 triangles in bottom -> 2 wedge prisms."""
    from meshwell.structured.builder import _build_slab_volume, _stamp_top_face_mesh

    bot_tag, top_tag = _make_box_in_gmsh_and_mesh_2d(z_lo=0.0, z_hi=1.0)
    bot_to_top = _stamp_top_face_mesh(bot_tag, top_tag, zlo=0.0, zhi=1.0)

    vol_tag = _build_slab_volume(
        bottom_face_tag=bot_tag,
        bot_to_top_layer_tags=[bot_to_top],
        n_layers=1,
        recombine=False,
    )
    # Discrete 3D entity created.
    assert vol_tag > 0
    # 2 triangle pairs in bottom -> 2 wedges per layer.
    etypes, etags, _ = gmsh.model.mesh.getElements(3, vol_tag)
    n_wedges = sum(len(t) for et, t in zip(etypes, etags) if et == 6)
    n_bot_tris_initial = 2  # 0.3 mesh size on unit square produces ~2 triangles
    assert n_wedges >= 2  # at least the triangle count


def test_build_slab_volume_multi_layer_produces_n_times_wedges(gmsh_session):
    """3 layers -> 3 x (bottom triangle count) wedges."""
    from meshwell.structured.builder import _build_slab_volume, _stamp_top_face_mesh

    bot_tag, top_tag = _make_box_in_gmsh_and_mesh_2d(z_lo=0.0, z_hi=1.0)
    bot_to_top = _stamp_top_face_mesh(bot_tag, top_tag, zlo=0.0, zhi=1.0)
    # For n_layers > 1 we'd need intermediate node maps too. For this
    # unit test we test the n_layers=1 case only. Multi-layer wiring is
    # exercised in the end-to-end test (Task 6).
    pytest.skip("multi-layer requires intermediate node-tag maps; covered in end-to-end test")
```

(Note: Task 4 ships only the n_layers=1 wedge case as a unit test. Multi-layer + hex come through the end-to-end test in Task 6 because constructing intermediate node maps in isolation is fiddly.)

- [ ] **Step 2: Run; expect ImportError on _build_slab_volume.**

- [ ] **Step 3: Append `_build_slab_volume` to `meshwell/structured/builder.py`**

```python
def _build_slab_volume(
    bottom_face_tag: int,
    bot_to_top_layer_tags: list[dict[int, int]],
    n_layers: int,
    recombine: bool,
) -> int:
    """Create one discrete 3D entity with wedge or hex elements.

    Args:
        bottom_face_tag: OCC face tag whose 2D mesh is the source
            triangulation (or quad mesh, if recombine).
        bot_to_top_layer_tags: list of length n_layers, each a dict
            mapping bottom node tag -> the node tag in that layer.
            Layer 0 is implicit (= bottom node tags themselves);
            bot_to_top_layer_tags[i] maps to layer i+1.
        n_layers: number of element layers in z. Must equal
            len(bot_to_top_layer_tags).
        recombine: if True, build hex elements (type 5) instead of
            wedges (type 6). Bottom must have quads in that case.

    Returns:
        The discrete 3D entity's tag.
    """
    import gmsh

    assert len(bot_to_top_layer_tags) == n_layers, (
        f"Expected {n_layers} layer maps, got {len(bot_to_top_layer_tags)}"
    )

    # Read bottom triangle/quad connectivity.
    elem_types, _, elem_nodes_per_type = gmsh.model.mesh.getElements(
        2, bottom_face_tag
    )
    target_type = 3 if recombine else 2  # gmsh 2D element type code
    elem_3d_type = 5 if recombine else 6  # 5 = hex, 6 = wedge
    cells_per_face = 4 if recombine else 3
    bot_cells: list[np.ndarray] = []
    for et, en in zip(elem_types, elem_nodes_per_type):
        if et == target_type:
            bot_cells.append(np.asarray(en, dtype=np.int64).reshape(-1, cells_per_face))
    if not bot_cells:
        raise RuntimeError(
            f"Bottom OCC face {bottom_face_tag} has no element type "
            f"{target_type} (need {'quads' if recombine else 'triangles'})"
        )
    bot_cells_flat = np.concatenate(bot_cells, axis=0)
    n_cells = bot_cells_flat.shape[0]

    # Allocate the discrete 3D entity.
    vol_tag = gmsh.model.addDiscreteEntity(3, -1, [])

    # Per spec: do NOT add the volume's interior layer nodes to the volume
    # entity if n_layers == 1 (no interior). For n_layers >= 2, the layer
    # tags 1..n-1 need to be added to vol_tag with their coordinates.
    # For Phase 3 minimum we ship n_layers=1; multi-layer support gets
    # exercised in the end-to-end test which provides intermediate maps.
    # The implementer can extend this loop when n_layers > 1 inputs arrive.

    # Build the 3D element node lists: for each cell (a, b, c[, d]) and
    # each layer i in 0..n_layers-1, the volume element nodes are
    # [layer_i[a], ..., layer_{i+1}[a], ...].
    # Layer 0 = bottom node tags themselves; layer i+1 = bot_to_top_layer_tags[i].
    layer_maps_with_zero: list[dict[int, int] | None] = [None] + list(
        bot_to_top_layer_tags
    )

    def _layer_tag(layer_idx: int, bot_node_tag: int) -> int:
        if layer_idx == 0:
            return bot_node_tag
        return layer_maps_with_zero[layer_idx][bot_node_tag]

    all_volume_nodes: list[int] = []
    for cell in bot_cells_flat:
        for layer_i in range(n_layers):
            for c in cell:
                all_volume_nodes.append(_layer_tag(layer_i, int(c)))
            for c in cell:
                all_volume_nodes.append(_layer_tag(layer_i + 1, int(c)))

    next_elem_tag = int(gmsh.model.mesh.getMaxElementTag()) + 1
    n_3d = n_cells * n_layers
    elem_tags = list(range(next_elem_tag, next_elem_tag + n_3d))
    gmsh.model.mesh.addElements(3, vol_tag, [elem_3d_type], [elem_tags], [all_volume_nodes])
    return vol_tag
```

- [ ] **Step 4: Run; expect 1 PASS and 1 SKIP**

`.venv/bin/python -m pytest tests/structured/test_builder_unit.py -v`

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/builder.py tests/structured/test_builder_unit.py
git commit -m "$(cat <<'EOF'
feat(structured): builder._build_slab_volume (wedge/hex discrete 3D entity)

Creates one discrete 3D entity per slab via addDiscreteEntity(3),
populates it with wedge (type 6) or hex (type 5) elements bridging
the bottom mesh to the top layer nodes. Multi-layer support via the
bot_to_top_layer_tags list (length n_layers); layer 0 is implicit
(= bottom node tags themselves).

Phase 3 minimum unit-tests the single-layer wedge path; multi-layer +
hex exercised in the end-to-end test.
EOF
)"
```

---

## Task 5: `builder.apply_structured_mesh` — orchestrator-facing entry point

**Files:**
- Modify: `meshwell/structured/builder.py`
- (Tests for this come via the end-to-end test in Task 6.)

This is the single entry point the orchestrator calls between `mesh.generate(2)` and `mesh.generate(3)`. It walks the plan + phantom map, finds the bottom/top OCC face gmsh tags for each piece, calls `_stamp_top_face_mesh`, then `_build_slab_volume`, then runs `removeDuplicateNodes`.

The tricky part: mapping `output_faces[FaceKey]` (which gives OCP `TopoDS_Face` objects from cad_occ's BOP) to gmsh face tags. Because cad_occ loads the OCP shapes into gmsh via the XAO writer, there's a correspondence between OCP TopoDS_Face and gmsh face tag — but it's not direct. The implementer needs to figure out the lookup mechanism (probably by matching bounding boxes, or by leveraging the gmsh-side OCC integration if cad_occ uses `gmsh.model.occ.importShapes`).

**Implementer: investigate first.** Read `meshwell/cad_occ.py` and `meshwell/model.py` and `meshwell/occ_xao_writer.py` to find out how cad_occ's OCP shapes end up with gmsh face tags. Common patterns:
- After XAO round-trip, gmsh has its own face numbering — match by bbox.
- Or: the XAO writer writes OCP tags into XAO attributes; gmsh's read sees them as physical groups.

For Phase 3 minimum (single piece, no neighbour cuts), the slab's bottom and top OCC faces have well-separated bboxes (z=zlo and z=zhi). A simple bbox match suffices: look for gmsh faces whose z-min and z-max both equal zlo (or zhi) within fragment_fuzzy_value (or 1e-6 default).

- [ ] **Step 1: Implement** `apply_structured_mesh` in `meshwell/structured/builder.py`:

```python
def _find_horizontal_face_at_z(z: float, tol: float = 1e-6) -> int | None:
    """Return the gmsh face tag whose z-bbox is at z (within tol), else None.

    Phase 3 minimum assumption: single piece, no neighbour cuts -> exactly
    one bottom face at z=zlo and one top face at z=zhi for the slab.
    """
    import gmsh

    for dim, tag in gmsh.model.getEntities(2):
        bb = gmsh.model.getBoundingBox(dim, tag)
        if abs(bb[2] - z) < tol and abs(bb[5] - z) < tol:
            return tag
    return None


def apply_structured_mesh(
    plan: StructuredPlan,
    mesh_plan: StructuredMeshPlan,
    phantom_map: Any,  # PhantomMap — Any to avoid circular type imports
    fuzzy_tol: float = 1e-6,
) -> list[int]:
    """Run the mesh-stage Layer C: derive top meshes + build discrete 3D volumes.

    Returns a list of (slab-index-parallel) discrete-3D entity tags so the
    caller can assert/inspect.

    Phase 3 minimum: assumes single-piece partition per slab and no
    neighbour cuts of horizontal faces. Multi-piece + neighbour-cut
    handling lands in Phase 4.
    """
    import gmsh

    vol_tags: list[int] = []
    for slab_idx, slab in enumerate(plan.slabs):
        n_layers = mesh_plan.n_layers[slab_idx]
        recombine = mesh_plan.recombine[slab_idx]

        bot_tag = _find_horizontal_face_at_z(slab.zlo, tol=fuzzy_tol)
        top_tag = _find_horizontal_face_at_z(slab.zhi, tol=fuzzy_tol)
        if bot_tag is None or top_tag is None:
            raise RuntimeError(
                f"Slab {slab.physical_name}: could not find bottom face "
                f"at z={slab.zlo} or top face at z={slab.zhi} in gmsh model"
            )

        # Layer maps: index 0 = bottom layer (implicit); index 1..n_layers map
        # bottom node tag -> the node tag at that layer.
        layer_maps: list[dict[int, int]] = []
        height = slab.zhi - slab.zlo
        # For n_layers == 1, the only layer map is bottom -> top.
        if n_layers == 1:
            layer_maps = [_stamp_top_face_mesh(bot_tag, top_tag, slab.zlo, slab.zhi)]
        else:
            # For n_layers > 1 we need intermediate node layers.
            # Allocate fresh tags per layer; layer n is the top face mesh.
            import numpy as np

            bot_node_tags_arr, bot_coords_flat, _ = gmsh.model.mesh.getNodes(
                2, bot_tag, includeBoundary=True
            )
            bot_node_tags = np.asarray(bot_node_tags_arr, dtype=np.int64)
            bot_coords = np.asarray(bot_coords_flat, dtype=float).reshape(-1, 3)
            n_nodes = len(bot_node_tags)
            next_tag = int(gmsh.model.mesh.getMaxNodeTag()) + 1
            # Build interior layers 1..n_layers-1 first, add to vol_tag later.
            # We need the vol_tag before adding interior nodes, but
            # _build_slab_volume creates it. To keep this simple, we
            # pre-allocate tags and stage the addNodes call into the
            # discrete volume via gmsh after _build_slab_volume.
            interior_layer_maps: list[dict[int, int]] = []
            interior_layer_coords: list[list[float]] = []
            for i_layer in range(1, n_layers):
                m = {}
                coords = []
                z_i = slab.zlo + height * (i_layer / n_layers)
                for j, bt in enumerate(bot_node_tags):
                    new_tag = next_tag
                    next_tag += 1
                    m[int(bt)] = new_tag
                    coords.extend([bot_coords[j, 0], bot_coords[j, 1], z_i])
                interior_layer_maps.append(m)
                interior_layer_coords.append(coords)
            # Top layer map.
            top_map = _stamp_top_face_mesh(bot_tag, top_tag, slab.zlo, slab.zhi)
            layer_maps = interior_layer_maps + [top_map]

        vol_tag = _build_slab_volume(
            bottom_face_tag=bot_tag,
            bot_to_top_layer_tags=layer_maps,
            n_layers=n_layers,
            recombine=recombine,
        )

        # For n_layers > 1, the interior layer nodes were tag-allocated but
        # not yet added to the gmsh model. Add them to vol_tag now.
        if n_layers > 1:
            all_interior_tags: list[int] = []
            all_interior_coords: list[float] = []
            for m, coords in zip(interior_layer_maps, interior_layer_coords):
                all_interior_tags.extend(m.values())
                all_interior_coords.extend(coords)
            if all_interior_tags:
                gmsh.model.mesh.addNodes(
                    3, vol_tag, all_interior_tags, all_interior_coords
                )

        vol_tags.append(vol_tag)

    # Global cleanup: merge ~coincident nodes.
    fuzzy = max(
        (slab.fragment_fuzzy_value for slab in plan.slabs if slab.fragment_fuzzy_value is not None),
        default=fuzzy_tol,
    )
    try:
        gmsh.model.mesh.removeDuplicateNodes(tag=[], tol=2 * fuzzy)
    except TypeError:
        # Older gmsh API: no `tol` kwarg.
        gmsh.model.mesh.removeDuplicateNodes()

    return vol_tags
```

Note: this function is not unit-tested directly; the end-to-end test in Task 6 exercises it.

- [ ] **Step 2: Commit** (no test added yet — Task 6 covers it):

```bash
git add meshwell/structured/builder.py
git commit -m "$(cat <<'EOF'
feat(structured): builder.apply_structured_mesh (orchestrator entry point)

Walks plan + mesh_plan: for each slab, finds the gmsh face tags at
z=zlo / z=zhi, derives the top mesh from bottom via _stamp_top_face_mesh,
then builds the slab volume as a discrete 3D entity via
_build_slab_volume. Final step: gmsh.model.mesh.removeDuplicateNodes
with tol = 2 * max(fragment_fuzzy).

Phase 3 minimum assumptions: single-piece partition per slab, no
neighbour cuts of horizontal faces (lookups by z-bbox only). Multi-
piece + neighbour-cut handling deferred to Phase 4.
EOF
)"
```

---

## Task 6: Orchestrator wiring + end-to-end test

**Files:**
- Modify: `meshwell/orchestrator.py`
- Create: `tests/structured/test_end_to_end_minimal.py`

The orchestrator detects structured entities, runs the full pipeline.

- [ ] **Step 1: Write the failing end-to-end test**

Create `tests/structured/test_end_to_end_minimal.py`:

```python
"""End-to-end test: PolyPrism(structured=True) -> mesh -> wedge elements."""
from __future__ import annotations

import pytest
from shapely.geometry import Polygon


def _square(x=0, y=0, w=1, h=1) -> Polygon:
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def test_single_structured_slab_produces_wedge_mesh(tmp_path):
    """A single structured PolyPrism with n_layers=2 produces wedges in the mesh."""
    import meshio
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    p = PolyPrism(
        polygons=_square(0, 0, 2, 2),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="block",
    )

    out_msh = tmp_path / "structured.msh"
    generate_mesh([p], dim=3, output_mesh=out_msh)

    m = meshio.read(out_msh)
    # Wedge cells should be present.
    cell_types = {cb.type for cb in m.cells}
    assert "wedge" in cell_types, f"Expected wedge cells, got {cell_types}"
    # Physical "block" should be present.
    assert "block" in m.field_data


def test_single_structured_slab_default_characteristic_length(tmp_path):
    """Smoke: structured pipeline runs without exceptions."""
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    p = PolyPrism(
        polygons=_square(),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="b",
    )
    out_msh = tmp_path / "smoke.msh"
    generate_mesh([p], dim=3, output_mesh=out_msh)
    assert out_msh.exists()
```

- [ ] **Step 2: Run; expect failure** (orchestrator doesn't yet detect structured entities).

`.venv/bin/python -m pytest tests/structured/test_end_to_end_minimal.py -v`

- [ ] **Step 3: Wire the orchestrator**

Read `meshwell/orchestrator.py` (~116 LOC). Find where cad_occ is called and where mesh() is called. Insert the structured pipeline:

a. Detect: `has_structured = any(getattr(e, "structured", False) for e in entities)`.

b. If `has_structured`:
   - `plan = build_plan(entities)`
   - `phantom_result = build_phantom_shapes(plan)`
   - `extra = [s.solid for s in phantom_result.shapes]`
   - Capture the BOP builder via callback:
     ```python
     captured_builder: list = []
     occ_entities = cad_occ(
         entities,
         extra_occ_shapes=extra,
         cad_occ_callback=lambda b: captured_builder.append(b),
         **cad_kwargs,
     )
     phantom_map = extract_phantom_map(phantom_result, captured_builder[0])
     mesh_plan_obj = resolve_mesh_plan(plan, entities)
     ```
   - Load the OCC entities into gmsh as usual (`mm.load_occ_entities(...)`).
   - **After** the existing `mesh()` call's `mesh.generate(2)` but **before** `mesh.generate(3)`, run `apply_structured_mesh(plan, mesh_plan_obj, phantom_map)`. The cleanest way to interject is to call mesh() with `dim=2` first, then `apply_structured_mesh`, then a second mesh() call with `dim=3` and `mesh_only_empty=True` (so gmsh doesn't re-mesh the discrete entity).

This may not be the cleanest possible architecture, but it works without restructuring `meshwell/mesh.py`. The implementer may discover a simpler approach.

c. **Alternative orchestration if the two-pass-mesh proves too messy:** add a hook parameter to `meshwell/mesh.py`'s `mesh()` function: `pre_3d_hook: Callable[[], None] | None = None`. The structured orchestrator passes `lambda: apply_structured_mesh(...)`. The hook runs between mesh.generate(2) and mesh.generate(3) automatically.

The implementer picks whichever approach is simpler. Document the choice in the commit message.

d. If `has_structured` is False, run the existing cad_occ + mesh flow unchanged.

- [ ] **Step 4: Set `Mesh.MeshOnlyEmpty=1` for structured runs**

To prevent gmsh from re-meshing the discrete 3D entities we built, set this option before mesh.generate(3):

```python
gmsh.option.setNumber("Mesh.MeshOnlyEmpty", 1)
```

If `mesh()` already calls `setNumber("Mesh.MeshOnlyEmpty", 0)` somewhere, override it for structured runs.

- [ ] **Step 5: Run end-to-end tests; expect 2 PASSES**

`.venv/bin/python -m pytest tests/structured/test_end_to_end_minimal.py -v`

If this fails, iterate:
- If the mesh doesn't have wedges: check that `apply_structured_mesh` is actually being called, and that the discrete 3D entity was created.
- If the mesh has wedges but the test asserts wrong cell type name: `meshio` may call them "wedge" or "prism" or "wedge6" — adjust the assertion accordingly.
- If gmsh complains about double-meshing: ensure `MeshOnlyEmpty=1`.

- [ ] **Step 6: Run the full structured suite for regression**

`.venv/bin/python -m pytest tests/structured/ -v`

Expected: all PASS (~75 tests now: Phase 1 30 + Phase 2 33 + Phase 3 12 = 75-ish).

- [ ] **Step 7: Commit**

```bash
git add meshwell/orchestrator.py tests/structured/test_end_to_end_minimal.py
git commit -m "$(cat <<'EOF'
feat(structured): orchestrator wires Phase-3 pipeline + end-to-end test

generate_mesh() detects structured=True entities and runs the full
clean pipeline: build_plan -> build_phantom_shapes -> cad_occ (with
phantom shapes added to BOP, builder captured via callback) ->
extract_phantom_map -> resolve_mesh_plan -> mesh.generate(2) ->
apply_structured_mesh (stamps top mesh + builds discrete 3D entities) ->
mesh.generate(3) (Mesh.MeshOnlyEmpty=1 so gmsh skips the
already-meshed slab volumes).

Single end-to-end test passes: a PolyPrism(structured=True,
n_layers=[2]) produces wedge cells visible to meshio. The full chain
from user API to a valid .msh file works for the minimum case
(single-piece partition, no neighbour cuts).

Multi-piece + neighbour cuts + arc provenance are Phase 4.
EOF
)"
```

---

## Task 7: Re-export Phase 3 public surface + docs

**Files:**
- Modify: `meshwell/structured/__init__.py`
- Modify: `tests/structured/test_package_smoke.py`

- [ ] **Step 1: Append the failing assertion**

To `tests/structured/test_package_smoke.py`:

```python
def test_phase3_public_exports():
    from meshwell.structured import (
        StructuredMeshPlan,
        apply_structured_mesh,
        resolve_mesh_plan,
    )

    assert StructuredMeshPlan is not None
    assert apply_structured_mesh is not None
    assert resolve_mesh_plan is not None
```

- [ ] **Step 2: Run; expect ImportError.**

- [ ] **Step 3: Update `meshwell/structured/__init__.py`**:

```python
"""Clean structured-polyprism pipeline.

Public surface:

- :class:`StructuredExtrusionResolutionSpec` — attach to a
  ``PolyPrism(structured=True)`` to specify per-z-interval layer counts.
- :func:`build_plan` — planner entry point.
- :func:`build_phantom_shapes` — CAD-stage Layer A.
- :func:`extract_phantom_map` — CAD-stage Layer B.
- :class:`PhantomMap` — post-BOP correspondence map.
- :func:`resolve_mesh_plan` — mesh-stage parameter resolver.
- :func:`apply_structured_mesh` — mesh-stage Layer C entry point.
- :class:`StructuredMeshPlan` — output of resolve_mesh_plan.
"""
from __future__ import annotations

from meshwell.structured.builder import apply_structured_mesh, resolve_mesh_plan
from meshwell.structured.phantom import build_phantom_shapes, extract_phantom_map
from meshwell.structured.plan import build_plan
from meshwell.structured.spec import (
    PhantomMap,
    StructuredExtrusionResolutionSpec,
    StructuredMeshPlan,
)

__all__ = [
    "PhantomMap",
    "StructuredExtrusionResolutionSpec",
    "StructuredMeshPlan",
    "apply_structured_mesh",
    "build_phantom_shapes",
    "build_plan",
    "extract_phantom_map",
    "resolve_mesh_plan",
]
```

- [ ] **Step 4: Run smoke tests + full suite**

`.venv/bin/python -m pytest tests/structured/ -v`

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/__init__.py tests/structured/test_package_smoke.py
git commit -m "$(cat <<'EOF'
feat(structured): re-export Phase-3 builder API + StructuredMeshPlan

Completes the public surface. End of Phase 3:

  meshwell.structured exports
    StructuredExtrusionResolutionSpec  (Phase 1)
    build_plan                          (Phase 1)
    PhantomMap                          (Phase 2)
    build_phantom_shapes                (Phase 2)
    extract_phantom_map                 (Phase 2)
    StructuredMeshPlan                  (Phase 3)
    resolve_mesh_plan                   (Phase 3)
    apply_structured_mesh               (Phase 3)
EOF
)"
```

---

## Self-Review Checklist

**1. Spec coverage:**

| Spec section | Phase 3 task | Status |
|---|---|---|
| Layer C — top mesh from bottom (translation) | Task 3 | ✓ |
| Layer C — discrete 3D entity with wedges/hexes | Task 4 | ✓ |
| Layer C — entry point for mesh stage | Task 5 | ✓ |
| StructuredMeshPlan + n_layers resolution | Task 1 | ✓ |
| OverlapPair cross-check | Task 1 | ✓ |
| cad_occ integration (phantom shapes + builder access) | Task 2 | ✓ |
| Orchestrator wiring | Task 6 | ✓ |
| End-to-end (single slab, single piece) | Task 6 | ✓ |
| removeDuplicateNodes global cleanup | Task 5 | ✓ |
| **Layer B vertex-map lookup for top boundary nodes** | **Phase 4** | deferred |
| Multi-piece partition routing | **Phase 4** | deferred |
| Mid-height cut → unstructured lateral fallback | **Phase 4** | deferred |
| Arc-provenance migration | **Phase 4** | deferred |
| Performance instrumentation | **Phase 4** | deferred |

**2. Placeholder scan:** none. Every step has runnable code or a clear investigation directive (Task 2 step 1, Task 6 step 3).

**3. Type consistency:**
- `StructuredMeshPlan.n_layers: tuple[int, ...]` consistent across Tasks 1, 5. ✓
- `resolve_mesh_plan(plan, entities) -> StructuredMeshPlan` signature consistent across Tasks 1, 5, 7. ✓
- `_stamp_top_face_mesh(bot, top, zlo, zhi) -> dict[int, int]` consistent between Tasks 3, 4, 5. ✓
- `_build_slab_volume(bottom_face_tag, bot_to_top_layer_tags, n_layers, recombine) -> int` consistent between Tasks 4, 5. ✓

**4. Ambiguity check:**
- Task 2's `cad_occ` modification depends on the existing function shape (class vs free function, where _fragment_all lives). The implementer reads the code first and adapts.
- Task 5's `_find_horizontal_face_at_z` assumes well-separated horizontal faces. Documented as a Phase 3 simplification; Phase 4 replaces it with the Layer B map lookup.
- Task 6's "two-pass-mesh" orchestration vs `pre_3d_hook` parameter: the implementer picks whichever is simpler and documents.

---

## Out of scope for Phase 3

- **Layer B vertex map for top boundary nodes**: Phase 3 uses pure translation (`bot_xy + (0, 0, h)`). Phase 4 wires the Layer B `output_vertices` map to use BOP-displaced top OCC vertex positions.
- **Multi-piece partition**: Phase 3 assumes 1 piece per slab. Phase 4 handles N pieces with per-piece bottom/top face lookup.
- **Mid-height cut handling**: the `lateral_has_midheight_cut` map exists from Phase 2 but isn't consumed yet. Phase 4 uses it to skip transfinite hints on cut lateral faces.
- **Arc provenance migration**: Phase 4 ports the arc-provenance design.
- **`logging.py`**: deferred. Phase 4 adds per-phase timing instrumentation.
- **`scripts/bench_structured.py`**: deferred.
- **Mixed-type slabs (some hex, some wedge)**: not supported; all pieces of a slab use the slab's `recombine` setting.
- **Tapered structured prisms**: rejected at construction (Phase 1).
