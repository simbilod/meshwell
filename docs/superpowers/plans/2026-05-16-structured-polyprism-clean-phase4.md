# Structured PolyPrism Clean Rewrite — Phase 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Phase 3's single-piece `_find_horizontal_face_at_z` bbox lookup with proper PhantomMap-driven routing. End state: a structured polyprism whose `face_partition` has N pieces — because a non-structured neighbour shares its top z-plane — meshes correctly end-to-end. This is what makes Phase 2's Layer B (the OCC vertex/face map) actually load-bearing in the pipeline.

**Architecture:** Add `_map_phantom_faces_to_gmsh(phantom_map, fuzzy_tol)` that bbox-matches each `FaceKey` entry in `phantom_map.output_faces` (which holds OCP `TopoDS_Face` objects) to the corresponding gmsh face tag(s) loaded into the model post-XAO. Refactor `apply_structured_mesh` to iterate per `(slab, piece)` using the FaceKey map instead of the current single-face-per-z bbox lookup. Each piece is stamped + volume-built independently. The slab's physical_name groups all per-piece volumes logically.

**Tech Stack:** Same as Phase 3 — OCP, gmsh, shapely, pytest. No new dependencies.

**Spec reference:** `docs/superpowers/specs/2026-05-15-structured-polyprism-clean-design.md`, Layers A+B+C — Phase 4 closes the loop between B (the map) and C (the mesh stage) for the multi-piece case.

**Phase 3 reference:** `docs/superpowers/plans/2026-05-16-structured-polyprism-clean-phase3.md`.

---

## Out of scope for Phase 4 (defer to Phase 5+)

- **Neighbour cuts that split a piece's top into multiple output faces** (e.g. a small box that lands on a piece's interior, requiring per-cell xy-routing of stamped mesh to multiple output top sub-faces). Phase 4 handles `len(phantom_map.output_faces[FaceKey]) == 1` per piece; assert otherwise.
- **Mid-height-cut → unstructured lateral fallback** (uses `phantom_map.lateral_has_midheight_cut` — Phase 5).
- **BOP-displaced top boundary node positions via `output_vertices`**. Phase 3's `_stamp_top_face_mesh` already reuses surviving gmsh boundary node tags (which inherit the BOP-displaced positions from gmsh's OCC import), so this works for the typical case without needing Layer B's `output_vertices` directly. Phase 5 may revisit if a failure mode emerges.
- **Arc-provenance migration** (Phase 5+).
- **`logging.py` + `scripts/bench_structured.py`** (Phase 5+).

---

## File Structure

**Modify:**
- `meshwell/structured/builder.py` — add `_map_phantom_faces_to_gmsh` helper; refactor `apply_structured_mesh` to consume `phantom_map` (currently flagged `noqa: ARG001`). The two-pass (per-piece) volume-stamping replaces the single-pass `_find_volume_with_faces` routine.

**Create:**
- `tests/structured/test_phantom_to_gmsh_map.py` — unit tests for the new `_map_phantom_faces_to_gmsh` helper (~120 LOC).
- `tests/structured/test_end_to_end_multipiece.py` — end-to-end tests for the multi-piece scenario (~150 LOC).

**Untouched in Phase 4:**
- `meshwell/structured/spec.py` — no new dataclasses needed; Phase 2's `PhantomMap` is sufficient.
- `meshwell/structured/plan.py`, `phantom.py` — unchanged.
- `meshwell/cad_occ.py` — Phase 3's hook is sufficient.
- `meshwell/orchestrator.py` — Phase 3's wiring already passes `phantom_map` through `apply_structured_mesh`.
- `meshwell/mesh.py` — `pre_3d_hook` is already in place.

---

## Tech notes (read before coding)

### OCP-face → gmsh-face matching by bbox

After cad_occ writes the post-BOP OCC scene to XAO and gmsh loads it, every OCP `TopoDS_Face` in `phantom_map.output_faces[FaceKey]` corresponds to exactly one gmsh face tag (or to zero, if BOP deleted it, which shouldn't happen for phantom-side faces). The match is by axis-aligned bounding box.

OCP bbox of a TopoDS_Face:

```python
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib

def _occ_face_bbox(face):
    b = Bnd_Box()
    BRepBndLib.Add_s(face, b)
    xmin, ymin, zmin, xmax, ymax, zmax = b.Get()
    return (xmin, ymin, zmin, xmax, ymax, zmax)
```

gmsh face bbox:

```python
bb = gmsh.model.getBoundingBox(2, gmsh_face_tag)  # → (xmin, ymin, zmin, xmax, ymax, zmax)
```

Match if all 6 coordinates are within `tol` (default `1e-6`, increase to `fragment_fuzzy_value` if available). For Phase 4 minimum, `1e-6` is sufficient.

To make lookup O(n_faces) instead of O(n_pieces × n_faces): build a dict keyed by a rounded 6-tuple of the gmsh bbox, then look up each OCP face's bbox in that dict.

### Per-piece volume stamping

Phase 3's `_find_volume_with_faces` walks `getEntities(3)` to find the volume whose boundary contains a given (bot, top) pair. For multi-piece, EACH piece's sub-prism becomes its own gmsh volume after XAO load (cad_occ doesn't fuse them per Phase 2's decision). So for each piece:

1. Find the piece's bottom + top gmsh face tags via the map.
2. Find the gmsh volume whose boundary contains both — that's this piece's slab-volume container.
3. Stamp wedges into THAT volume (using Phase 3's approach).

If `_find_volume_with_faces` returns None (e.g. the piece's volume was eliminated by a higher-priority overlap winner — shouldn't happen in Phase 4 minimum), raise with a clear error.

### Routing the slab's physical_name

Currently the slab → physical_name mapping is one-to-one. With N pieces per slab, all N piece-volumes share the slab's `physical_name`. gmsh's physical-group machinery handles this naturally (multiple volume tags can join the same physical group). Confirm the existing tagging code (in `meshwell/model.py` or `meshwell/occ_xao_writer.py`) groups by physical name, not by volume tag, before assuming it just works.

---

## Task 1: `_map_phantom_faces_to_gmsh` helper

**Files:**
- Modify: `meshwell/structured/builder.py`
- Create: `tests/structured/test_phantom_to_gmsh_map.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/structured/test_phantom_to_gmsh_map.py`:

```python
"""Tests for builder._map_phantom_faces_to_gmsh."""
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


def test_map_phantom_faces_to_gmsh_single_piece(gmsh_session):
    """Single-piece slab: one bottom + one top face match cleanly."""
    from meshwell.structured.builder import _map_phantom_faces_to_gmsh
    from meshwell.structured.phantom import build_phantom_shapes, extract_phantom_map
    from meshwell.structured.spec import (
        FaceKey,
        Slab,
        StructuredPlan,
    )
    from OCP.BOPAlgo import BOPAlgo_Builder

    # Build a phantom for a single slab with one piece.
    slab = Slab(
        footprint=_square(0, 0, 2, 2),
        zlo=0.0,
        zhi=1.0,
        physical_name=("s",),
        source_index=0,
        z_interval_index=0,
        mesh_order=1.0,
        face_partition=[_square(0, 0, 2, 2)],
    )
    plan = StructuredPlan(slabs=(slab,), z_planes=(0.0, 1.0), overlaps=())
    phantom_result = build_phantom_shapes(plan)

    # Run BOP on the phantom alone (no neighbours).
    builder = BOPAlgo_Builder()
    for s in phantom_result.shapes:
        builder.AddArgument(s.solid)
    builder.Perform()
    phantom_map = extract_phantom_map(phantom_result, builder)

    # Load the phantom solid into gmsh via the OCC kernel.
    gmsh.model.occ.importShapesNativePointer(int(phantom_result.shapes[0].solid.this))
    gmsh.model.occ.synchronize()

    # Map should resolve each FaceKey to exactly one gmsh face tag.
    fmap = _map_phantom_faces_to_gmsh(phantom_map)
    assert FaceKey(0, "bot", 0) in fmap
    assert FaceKey(0, "top", 0) in fmap
    assert len(fmap[FaceKey(0, "bot", 0)]) == 1
    assert len(fmap[FaceKey(0, "top", 0)]) == 1
```

NOTE: `importShapesNativePointer` may or may not work depending on the OCP+gmsh build. If it doesn't, the implementer must find another way to push the phantom solid into gmsh for this unit test (e.g. via a temporary BREP file: dump OCP shape to BREP, then `gmsh.model.occ.importShapes(brep_path)`). Document the choice in the commit.

If the unit test can't easily round-trip OCP→gmsh in isolation, the implementer may skip this unit test and rely on the end-to-end test in Task 3 to exercise the helper.

- [ ] **Step 2: Run; expect ImportError on `_map_phantom_faces_to_gmsh`.**

- [ ] **Step 3: Implement the helper in `meshwell/structured/builder.py`**

```python
def _occ_face_bbox(face: Any) -> tuple[float, float, float, float, float, float]:
    """Return AABB (xmin, ymin, zmin, xmax, ymax, zmax) of an OCP TopoDS_Face."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    b = Bnd_Box()
    BRepBndLib.Add_s(face, b)
    xmin, ymin, zmin, xmax, ymax, zmax = b.Get()
    return (xmin, ymin, zmin, xmax, ymax, zmax)


def _map_phantom_faces_to_gmsh(
    phantom_map: Any,  # PhantomMap
    tol: float = 1e-6,
) -> dict[Any, list[int]]:  # dict[FaceKey, list[int]]
    """Match each PhantomMap.output_faces entry (OCP TopoDS_Face) to a gmsh face tag.

    Implementation: build a dict of gmsh face tags keyed by rounded
    bbox, then look up each OCP face's bbox.

    Returns a dict[FaceKey, list[int]] — the same shape as
    phantom_map.output_faces but with gmsh tags as values. Faces with no
    match raise RuntimeError (this should not happen if cad_occ loaded
    the phantom solid into gmsh).
    """
    import gmsh

    def _round_bbox(bb: tuple[float, ...], tol: float) -> tuple[int, ...]:
        scale = 1.0 / tol
        return tuple(round(v * scale) for v in bb)

    gmsh_by_bbox: dict[tuple[int, ...], list[int]] = {}
    for dim, tag in gmsh.model.getEntities(2):
        bb = gmsh.model.getBoundingBox(dim, tag)
        gmsh_by_bbox.setdefault(_round_bbox(bb, tol), []).append(tag)

    out: dict[Any, list[int]] = {}
    for face_key, occ_faces in phantom_map.output_faces.items():
        gmsh_tags: list[int] = []
        for occ_face in occ_faces:
            bb = _occ_face_bbox(occ_face)
            key = _round_bbox(bb, tol)
            matches = gmsh_by_bbox.get(key, [])
            if not matches:
                # Loosen by checking nearby rounded keys (±1 in each dimension).
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            for dx2 in (-1, 0, 1):
                                for dy2 in (-1, 0, 1):
                                    for dz2 in (-1, 0, 1):
                                        nearby = (
                                            key[0] + dx, key[1] + dy, key[2] + dz,
                                            key[3] + dx2, key[4] + dy2, key[5] + dz2,
                                        )
                                        nm = gmsh_by_bbox.get(nearby, [])
                                        if nm:
                                            matches = nm
                                            break
            gmsh_tags.extend(matches)
        if not gmsh_tags:
            raise RuntimeError(
                f"PhantomMap face {face_key} has no matching gmsh face "
                f"(OCP bbox {_occ_face_bbox(occ_faces[0])} not found in gmsh model)."
            )
        out[face_key] = gmsh_tags
    return out
```

The 6-nested-loop "nearby key" expansion is ugly; if the simple direct match works (which it should for axis-aligned boxes with clean coordinates), the nearby fallback is dead code. The implementer can simplify to just the direct match and add fallback only if the test fails.

- [ ] **Step 4: Run; expect 1 PASS (or report if `importShapesNativePointer` doesn't work; in that case skip the unit test and proceed)**

`.venv/bin/python -m pytest tests/structured/test_phantom_to_gmsh_map.py -v`

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/builder.py tests/structured/test_phantom_to_gmsh_map.py
git commit -m "$(cat <<'EOF'
feat(structured): builder._map_phantom_faces_to_gmsh helper

Bbox-based mapping from PhantomMap's OCP TopoDS_Face entries to the
corresponding gmsh face tags loaded into the model post-XAO. Builds a
rounded-bbox dict for O(n) lookup, falls back to nearby keys if the
direct match misses (fuzzy-tolerance defensive).

The helper is the bridge between Phase 2's Layer B (the OCP-side
post-BOP map) and Phase 4's per-piece mesh routing.
EOF
)"
```

---

## Task 2: Refactor `apply_structured_mesh` to use the FaceKey map

**Files:**
- Modify: `meshwell/structured/builder.py`

Replace the current single-piece `_find_horizontal_face_at_z` lookup with the FaceKey-based map. Iterate per `(slab, piece)`, find the bottom + top gmsh face tags via the map, stamp + build volume.

- [ ] **Step 1: Implement the refactor**

In `meshwell/structured/builder.py`, replace the `apply_structured_mesh` function body. Key changes:

a. Call `_map_phantom_faces_to_gmsh(phantom_map)` once at the start.
b. For each slab, iterate `range(len(slab.face_partition))` for piece_index. For each piece:
   - Look up `bot_tags = face_map[FaceKey(slab_idx, "bot", piece_idx)]`
   - Look up `top_tags = face_map[FaceKey(slab_idx, "top", piece_idx)]`
   - For Phase 4 minimum: assert `len(bot_tags) == 1 and len(top_tags) == 1` (defer multi-output-face per piece to Phase 5).
   - Pass `bot_tags[0]` and `top_tags[0]` into the existing `_stamp_top_face_mesh` + `_build_slab_volume` flow.
c. Keep the `removeDuplicateNodes` global cleanup at the end.

Pseudocode:

```python
def apply_structured_mesh(
    plan: StructuredPlan,
    mesh_plan: StructuredMeshPlan,
    phantom_map: Any,
    fuzzy_tol: float = 1e-6,
) -> list[int]:
    import gmsh

    face_map = _map_phantom_faces_to_gmsh(phantom_map, tol=fuzzy_tol)

    vol_tags: list[int] = []
    for slab_idx, slab in enumerate(plan.slabs):
        n_layers = mesh_plan.n_layers[slab_idx]
        recombine = mesh_plan.recombine[slab_idx]
        for piece_idx in range(len(slab.face_partition)):
            from meshwell.structured.spec import FaceKey

            bot_key = FaceKey(slab_idx, "bot", piece_idx)
            top_key = FaceKey(slab_idx, "top", piece_idx)
            bot_tags = face_map.get(bot_key, [])
            top_tags = face_map.get(top_key, [])
            if len(bot_tags) != 1 or len(top_tags) != 1:
                raise RuntimeError(
                    f"Slab {slab.physical_name} piece {piece_idx}: "
                    f"expected exactly one bottom + one top gmsh face "
                    f"(Phase 4 minimum); got bottom={bot_tags}, top={top_tags}. "
                    f"Multi-output-face support is Phase 5+."
                )
            bot_tag = bot_tags[0]
            top_tag = top_tags[0]

            # ... existing per-piece logic from Phase 3's apply_structured_mesh,
            # but using bot_tag/top_tag from the map instead of
            # _find_horizontal_face_at_z. Existing logic:
            # - if n_layers == 1: layer_maps = [_stamp_top_face_mesh(...)]
            # - else: build intermediate layer maps + stamp top
            # - _build_slab_volume + _find_volume_with_faces stamp
            # - add interior node coords if n_layers > 1
            # All of that gets called per-piece in the inner loop now.

            # Per-piece vol_tag accumulates into vol_tags.

    # removeDuplicateNodes (unchanged).
    return vol_tags
```

The existing per-piece logic (stamp, build, register interior nodes) doesn't change — it just runs N times per slab instead of once. The `_find_horizontal_face_at_z` helper can be deleted (unused) or kept as dead code with a note.

- [ ] **Step 2: Confirm existing single-piece tests still pass**

`.venv/bin/python -m pytest tests/structured/ -v`

Expected: all PASS (Phase 3's single-piece end-to-end test still works because for single-piece slabs, the map gives 1 bottom + 1 top per piece).

- [ ] **Step 3: Commit**

```bash
git add meshwell/structured/builder.py
git commit -m "$(cat <<'EOF'
feat(structured): apply_structured_mesh routes by FaceKey via PhantomMap

Replaces the Phase-3 z-bbox lookup (_find_horizontal_face_at_z, which
only handled single-piece slabs) with PhantomMap-driven routing per
(slab, piece). Each piece's bottom/top gmsh face is looked up by the
FaceKey -> gmsh-tag map; the existing stamp + build-volume flow runs
once per piece.

Phase 4 minimum: requires exactly one output face per (slab, side,
piece) — raises if BOP split a piece's top into multiple sub-faces.
Multi-output-face routing (e.g. neighbour cuts piece interior) is
Phase 5+.

Single-piece end-to-end tests still pass; the multi-piece end-to-end
test in Task 3 exercises the new code path.
EOF
)"
```

---

## Task 3: End-to-end multi-piece test

**Files:**
- Create: `tests/structured/test_end_to_end_multipiece.py`

- [ ] **Step 1: Write the test**

Create `tests/structured/test_end_to_end_multipiece.py`:

```python
"""End-to-end test: structured polyprism whose face_partition has 2+ pieces.

The partition is induced by a non-structured neighbour sharing the slab's
top z-plane.
"""
from __future__ import annotations

import pytest
from shapely.geometry import Polygon


def _square(x=0, y=0, w=1, h=1) -> Polygon:
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def test_structured_slab_with_top_neighbour_produces_multi_piece_wedges(tmp_path):
    """A structured slab whose top is partially covered by a non-structured
    neighbour produces a 2-piece face_partition; both pieces mesh correctly."""
    import meshio

    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    # Structured slab: 4x4 footprint, z=[0, 1], 2 layers.
    s = PolyPrism(
        polygons=_square(0, 0, 4, 4),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="slab",
    )
    # Non-structured neighbour: 2x4, sits on top of s's top half.
    n = PolyPrism(
        polygons=_square(0, 0, 2, 4),
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="cap",
    )

    out_msh = tmp_path / "multipiece.msh"
    generate_mesh([s, n], dim=3, output_mesh=out_msh)

    m = meshio.read(out_msh)
    cell_types = {cb.type for cb in m.cells}
    # The slab produces wedges (one set per piece).
    wedge_like = any(ct in cell_types for ct in ("wedge", "wedge6"))
    assert wedge_like, f"Expected wedge cells, got {cell_types}"
    # Both physicals are present.
    assert "slab" in m.field_data
    assert "cap" in m.field_data


def test_structured_slab_with_top_neighbour_face_partition_has_two_pieces():
    """Inspect the plan directly to confirm the 2-piece partition is built."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan

    s = PolyPrism(
        polygons=_square(0, 0, 4, 4),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="slab",
    )
    n = PolyPrism(
        polygons=_square(0, 0, 2, 4),
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="cap",
    )
    plan = build_plan([s, n])
    assert len(plan.slabs) == 1
    assert len(plan.slabs[0].face_partition) == 2, (
        f"Expected 2-piece partition; got "
        f"{len(plan.slabs[0].face_partition)}"
    )
```

- [ ] **Step 2: Run; expect failures**

`.venv/bin/python -m pytest tests/structured/test_end_to_end_multipiece.py -v`

Likely failure modes:
- The face_partition test (no-mesh path) should pass immediately — it just validates Phase 1's planner.
- The mesh test will hit issues:
  - If the cad_occ + phantom + map flow works but BOP splits a piece's top into multiple output faces → "expected exactly one bottom + one top" assertion fires. This is the Phase-5 case (defer + skip).
  - If the bbox map misses → "no matching gmsh face" error. Investigate; loosen tol if needed.

- [ ] **Step 3: Iterate**

Iterate on the multi-piece end-to-end test until it passes. Diagnostic info to surface in case of failures:
- Print the PhantomMap's `output_faces` lengths to see if BOP split a piece.
- Print the gmsh face tags + bboxes after `mesh.generate(2)` to compare against the OCP faces.
- If the per-piece volume can't be located (`_find_volume_with_faces` returns None), check that XAO load created per-piece gmsh volumes.

Budget ~5 attempts before reporting BLOCKED with a detailed diagnostic.

- [ ] **Step 4: Run full structured suite for regression**

`.venv/bin/python -m pytest tests/structured/ -v`

Expected: all PASS (existing single-piece tests + new multi-piece tests).

- [ ] **Step 5: Commit**

```bash
git add tests/structured/test_end_to_end_multipiece.py
git commit -m "$(cat <<'EOF'
test(structured): end-to-end multi-piece slab with top neighbour

A structured slab whose top is partially covered by a non-structured
neighbour produces a 2-piece face_partition. Both pieces should mesh
correctly via the Phase-4 PhantomMap-driven routing.

This exercises the full Layers A/B/C clean architecture: plan stage
produces the 2-piece partition, phantom builds 2 sub-prisms, BOP
fragments them against the neighbour, PhantomMap records the
output-face correspondence per piece, builder routes mesh stage by
FaceKey instead of bbox-by-z lookup.
EOF
)"
```

---

## Self-Review Checklist

**1. Spec coverage:**

| Spec section | Phase 4 task | Status |
|---|---|---|
| Layer B map consumption (mesh stage) | Tasks 1, 2 | ✓ |
| Per-piece routing | Task 2 | ✓ |
| Multi-piece end-to-end | Task 3 | ✓ |
| **Mid-height-cut → unstructured lateral fallback** | **Phase 5** | deferred |
| **Neighbour cuts splitting a piece's top into N outputs** | **Phase 5** | deferred |
| **Layer B output_vertices for fresh boundary node positions** | **Phase 5** | deferred (Phase 3's surviving-node-tag approach handles the common case) |
| **Arc-provenance migration** | **Phase 5+** | deferred |
| **logging.py + bench_structured.py** | **Phase 5+** | deferred |

**2. Placeholder scan:** none. Every step has runnable code or clear iterate-budget directive (Task 3 Step 3).

**3. Type consistency:**
- `_map_phantom_faces_to_gmsh(phantom_map, tol) -> dict[FaceKey, list[int]]` consistent across Tasks 1, 2. ✓
- `FaceKey` from `meshwell.structured.spec` consistent with Phase 2's definition. ✓

**4. Ambiguity check:**
- Task 1's `importShapesNativePointer` may or may not work — the implementer has explicit license to skip the unit test and rely on the end-to-end test if so.
- Task 2's "existing per-piece logic doesn't change" is true for the inner-loop body of apply_structured_mesh — the implementer copy-pastes from Phase 3's body into the new piece-loop. There's no design change inside the per-piece work.
- Task 3 may hit the BOP-split-per-piece case (a neighbour cuts a piece's top into multiple output faces). If so, the assertion fires loudly. Defer to Phase 5 + skip the test with `pytest.skip` documenting the deferral.

---

## Out of scope for Phase 4

- Mid-height-cut → unstructured lateral fallback (Phase 5).
- Neighbour cuts that split a piece's top mid-piece (Phase 5).
- Layer B output_vertices for fresh boundary node positions (Phase 5+; not strictly needed because Phase 3's surviving-node-tag approach already works for the common cases).
- Arc-provenance migration (Phase 5+).
- logging.py + bench_structured.py (Phase 5+).
- Performance optimization of `_map_phantom_faces_to_gmsh` (Phase 5+; current implementation is O(n_faces) per phantom face which is fine for moderate scenes).
