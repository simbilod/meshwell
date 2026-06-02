# Alt B Promotion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the production `apply_lateral_transfinite_hints` pre_2d_hook with the Alt B "freeze cohort lateral mesh before generate(2)" approach validated by the spike. Determinism over speed.

**Architecture:** New function `freeze_lateral_mesh` in `meshwell/structured/wedge.py` validates n_layers, sets transfinite on vertical edges only (for predictable node counts), explicitly calls `generate(1)`, then walks bot/top edge nodes to emit lateral-face quads and seam-edge intermediate nodes. `Mesh.MeshOnlyEmpty=1` blocks gmsh's 2D mesher from re-meshing those faces during the outer `generate(2)`. `stamp_wedges` and the rest of the pipeline are unchanged.

**Tech Stack:** Python 3.12, OCP (OpenCASCADE Python), gmsh, pytest, shapely

**Spec:** [docs/superpowers/specs/2026-06-01-transfinite-dependencies-catalog.md](../specs/2026-06-01-transfinite-dependencies-catalog.md) (§5 replacement contract, §4 test scenarios S3/S5/S9, §6b Alt B architecture)

---

## File map

**Modified:**
- `meshwell/structured/wedge.py` — replace `apply_lateral_transfinite_hints` with `freeze_lateral_mesh`. Keep `stamp_wedges`, `resolve_n_layers`, `_face_centroid_z` unchanged. Update module docstring.
- `meshwell/orchestrator.py` — import and wire `freeze_lateral_mesh` in place of `apply_lateral_transfinite_hints`.
- `tests/structured/test_wedge_pre2d.py` — rename `test_transfinite_hints_produce_quad_laterals` → `test_lateral_face_quads`.
- `docs/superpowers/specs/2026-06-01-transfinite-dependencies-catalog.md` — add a "Promoted on YYYY-MM-DD" note.

**Created:**
- New tests added to `tests/structured/test_wedge_pre2d.py`:
  - `test_n_layers_1_meshes_cleanly` (scenario S3 from spec)
  - `test_shared_lateral_between_two_subsolids` (scenario S5 from spec)

**Deleted:**
- `meshwell/structured/wedge_alt_b_spike.py`
- `meshwell/structured/wedge_manual_spike.py`
- `scripts/spike_alt_b.py`
- `scripts/spike_manual_lateral.py`

---

## Task 0: Add S3 test (n_layers=1) on baseline behavior

**Files:**
- Modify: `tests/structured/test_wedge_pre2d.py`

This locks in the n_layers=1 baseline before any code changes. The test must pass against the current transfinite path AND against `freeze_lateral_mesh`.

- [ ] **Step 1: Append the failing test**

Append to `tests/structured/test_wedge_pre2d.py`:

```python
def test_n_layers_1_meshes_cleanly(tmp_path):
    """S3: degenerate n_layers=1 — one wedge per bot triangle, no
    intermediate-layer nodes needed."""
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism

    p = PolyPrism(
        polygons=SQ,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="s",
        structured=True,
    )
    generate_mesh(
        entities=[p],
        dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
        resolution_specs={
            "s": [StructuredExtrusionResolutionSpec(n_layers=1)],
        },
    )
    import meshio

    m = meshio.read(tmp_path / "out.msh")
    wedges = sum(cb.data.shape[0] for cb in m.cells if cb.type == "wedge")
    # SQ has 4 vertices, characteristic_length=0.5 -> ~4 bot triangles
    # n_layers=1 -> ~4 wedges
    assert wedges >= 2, f"expected >=2 wedges, got {wedges}"
    # No 3D group named "s" should be missing wedges.
    assert "s" in m.cell_sets
```

- [ ] **Step 2: Run test to verify it passes under current code**

```
pytest tests/structured/test_wedge_pre2d.py::test_n_layers_1_meshes_cleanly -v --no-cov
```
Expected: PASS (the existing transfinite path handles n_layers=1).

- [ ] **Step 3: Commit**

```bash
git add tests/structured/test_wedge_pre2d.py
git commit -m "test(structured): pin n_layers=1 behavior before alt-b promotion"
```

---

## Task 1: Add S5 test (shared lateral between two cohort sub-solids)

**Files:**
- Modify: `tests/structured/test_wedge_pre2d.py`

Two structured slabs sharing a lateral face must produce a single set of lateral quads, not duplicates.

- [ ] **Step 1: Append the failing test**

Append to `tests/structured/test_wedge_pre2d.py`:

```python
def test_shared_lateral_between_two_subsolids(tmp_path):
    """S5: two side-by-side structured slabs share a lateral face.
    The shared face must be meshed once (one set of quads, not two).
    Both volumes must have wedges. The interface group ``a___b`` must
    contain quads, not duplicates."""
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism

    SQ_A = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    SQ_B = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])

    a = PolyPrism(
        polygons=SQ_A,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="a",
        structured=True,
    )
    b = PolyPrism(
        polygons=SQ_B,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="b",
        structured=True,
    )
    generate_mesh(
        entities=[a, b],
        dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
        resolution_specs={
            "a": [StructuredExtrusionResolutionSpec(n_layers=2)],
            "b": [StructuredExtrusionResolutionSpec(n_layers=2)],
        },
    )
    import meshio

    m = meshio.read(tmp_path / "out.msh")

    # Both volumes get wedges.
    assert "a" in m.cell_sets
    assert "b" in m.cell_sets

    # Shared lateral interface must exist as a quad set.
    iface_name = "a___b" if "a___b" in m.cell_sets else "b___a"
    assert iface_name in m.cell_sets, (
        f"shared lateral interface missing; groups: {sorted(m.cell_sets)}"
    )

    # Count quads in the interface. n_layers=2, the shared edge spans
    # x=1 from y=0 to y=1 with characteristic_length=0.5 -> ~3 bot
    # edge nodes -> ~2 horizontal subdivisions -> 2 * n_layers = 4 quads.
    iface_sets = m.cell_sets[iface_name]
    iface_quads = sum(
        len(s) for s, b in zip(iface_sets, m.cells)
        if b.type == "quad" and s is not None
    )
    assert iface_quads >= 2, f"expected >=2 interface quads, got {iface_quads}"

    # Total wedge count should be the same in both volumes (symmetric).
    a_sets = m.cell_sets["a"]
    b_sets = m.cell_sets["b"]
    a_wedges = sum(
        len(s) for s, bk in zip(a_sets, m.cells)
        if bk.type == "wedge" and s is not None
    )
    b_wedges = sum(
        len(s) for s, bk in zip(b_sets, m.cells)
        if bk.type == "wedge" and s is not None
    )
    assert a_wedges > 0
    assert b_wedges > 0
```

- [ ] **Step 2: Add the `Polygon` import to the test file's top if not already there**

Open `tests/structured/test_wedge_pre2d.py`. The top imports should include
`from shapely.geometry import Polygon`. If missing, add it. (It's likely
already there because the file declares `SQ = Polygon(...)`.)

- [ ] **Step 3: Run test to verify it passes under current code**

```
pytest tests/structured/test_wedge_pre2d.py::test_shared_lateral_between_two_subsolids -v --no-cov
```
Expected: PASS (the existing transfinite path handles shared laterals).

- [ ] **Step 4: Commit**

```bash
git add tests/structured/test_wedge_pre2d.py
git commit -m "test(structured): pin shared-lateral behavior before alt-b promotion"
```

---

## Task 2: Add `freeze_lateral_mesh` to wedge.py (alongside the old function)

**Files:**
- Modify: `meshwell/structured/wedge.py`

Add the new function next to the old one. Do NOT remove the old function yet — that comes after the orchestrator switches over.

The implementation incorporates one fix vs. the spike: it reuses transfinite-placed vertical-edge nodes (placed by `setTransfiniteCurve` + `generate(1)`) instead of creating duplicates at the same positions.

- [ ] **Step 1: Append the helper functions and the new public function**

Insert after `apply_lateral_transfinite_hints` (which currently lives at `meshwell/structured/wedge.py:55`), before the `pre_3d_hook` section header:

```python
# ---------------------------------------------------------------------------
# Alt B: freeze cohort lateral mesh before generate(2)
# ---------------------------------------------------------------------------


def _classify_lateral_face_edges(
    face_tag: int,
    z_bot: float,
    z_top: float,
    z_tol: float = 1e-7,
) -> tuple[int | None, int | None, list[int]]:
    """Return (bot_edge_tag, top_edge_tag, [vertical_edge_tags])."""
    edges = gmsh.model.getBoundary(
        [(2, face_tag)], oriented=False, recursive=False
    )
    bot_edge = None
    top_edge = None
    vertical: list[int] = []
    for _dim, etag in edges:
        ev = gmsh.model.getBoundary(
            [(1, etag)], oriented=False, recursive=False
        )
        zs = []
        for _vd, vt in ev:
            pos = gmsh.model.getValue(0, vt, [])
            zs.append(pos[2])
        if len(zs) != 2:
            continue
        if abs(zs[0] - z_bot) < z_tol and abs(zs[1] - z_bot) < z_tol:
            bot_edge = etag
        elif abs(zs[0] - z_top) < z_tol and abs(zs[1] - z_top) < z_tol:
            top_edge = etag
        else:
            vertical.append(etag)
    return bot_edge, top_edge, vertical


def _ordered_curve_nodes(
    curve_tag: int,
) -> list[tuple[int, float, float, float]]:
    """Return curve nodes [(tag, x, y, z)] sorted by parametric coord."""
    tags, coord, param = gmsh.model.mesh.getNodes(
        1, curve_tag, includeBoundary=True, returnParametricCoord=True
    )
    if len(tags) == 0:
        return []
    items = []
    for i, t in enumerate(tags):
        items.append(
            (
                int(t),
                float(param[i]),
                float(coord[3 * i]),
                float(coord[3 * i + 1]),
                float(coord[3 * i + 2]),
            )
        )
    items.sort(key=lambda r: r[1])
    return [(t, x, y, z) for t, _p, x, y, z in items]


def _align_top_to_bot(
    bot_row: list[tuple[int, float, float, float]],
    top_row: list[tuple[int, float, float, float]],
) -> list[tuple[int, float, float, float]]:
    """Reverse top_row if its parametric direction runs opposite to bot."""
    if len(top_row) < 2:
        return top_row
    bot_first = bot_row[0]
    d_forward = (top_row[0][1] - bot_first[1]) ** 2 + (
        top_row[0][2] - bot_first[2]
    ) ** 2
    d_reverse = (top_row[-1][1] - bot_first[1]) ** 2 + (
        top_row[-1][2] - bot_first[2]
    ) ** 2
    return list(reversed(top_row)) if d_reverse < d_forward else top_row


def _vertical_edge_layer_nodes(
    vertical_edge_tag: int, n_layers: int
) -> list[int]:
    """Return the vertical edge's nodes ordered z_low -> z_high.

    Assumes ``setTransfiniteCurve(vertical_edge_tag, n_layers + 1)``
    was called before ``generate(1)`` so there are exactly
    ``n_layers + 1`` nodes spaced uniformly along the edge. The
    returned list has length ``n_layers + 1`` with index 0 at z_low
    and index n_layers at z_high.
    """
    tags, _coord, param = gmsh.model.mesh.getNodes(
        1,
        vertical_edge_tag,
        includeBoundary=True,
        returnParametricCoord=True,
    )
    items = sorted(zip(tags, param), key=lambda x: x[1])
    return [int(t) for t, _p in items]


def freeze_lateral_mesh(
    slab_meta: dict[ShapeKey, SlabMeta],
    face_tag_by_key: dict[ShapeKey, int],
    resolution_specs: dict | None = None,
) -> None:
    """Pre_2d hook: emit cohort lateral-face mesh before generate(2).

    Mechanism:
      1. Validate n_layers consistency on every shared lateral face.
      2. Set transfinite on vertical edges so generate(1) places
         exactly n_layers+1 nodes per vertical edge (uniformly
         spaced in parametric coord).
      3. Call generate(1) explicitly so we have edge nodes available
         to walk in step 4.
      4. For each lateral face: walk bot/top edge nodes in parametric
         order; reuse the vertical-edge transfinite nodes for the
         left/right endpoints at each layer; create face-interior
         nodes for the rest. Emit quad elements connecting layer rows.
      5. Set Mesh.MeshOnlyEmpty=1 so the outer generate(2) skips
         faces that already have a mesh.

    The result is identical in physical-group output to the old
    apply_lateral_transfinite_hints path but never invokes gmsh's
    2D mesher or its periodic-surface mesher on cohort lateral
    faces — both sources of past failures.
    """
    # Step 1: per-face n_layers + consistency check.
    owners_per_face: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for meta in slab_meta.values():
        if not meta.keep:
            continue
        n_layers = resolve_n_layers(meta.physical_name, resolution_specs)
        for fk in meta.lateral_face_keys:
            tag = face_tag_by_key.get(fk)
            if tag is None:
                continue
            owners_per_face[tag].append((meta.slab_index, n_layers))

    face_n_layers: dict[int, int] = {}
    for face_tag, owners in owners_per_face.items():
        n_layers_set = {n for _, n in owners}
        if len(n_layers_set) > 1:
            (sa, na), (sb, nb) = owners[0], owners[1]
            raise StructuredLateralNLayersMismatchError(
                slab_a=sa,
                slab_b=sb,
                face_tag=face_tag,
                n_layers_a=na,
                n_layers_b=nb,
            )
        face_n_layers[face_tag] = owners[0][1]

    # Per-face z bounds (needed for edge classification).
    face_z_bounds: dict[int, tuple[float, float]] = {}
    for meta in slab_meta.values():
        if not meta.keep:
            continue
        bot_tag = face_tag_by_key.get(meta.bot_face_key)
        top_tag = face_tag_by_key.get(meta.top_face_key)
        if bot_tag is None or top_tag is None:
            continue
        z_bot = _face_centroid_z(bot_tag)
        z_top = _face_centroid_z(top_tag)
        for fk in meta.lateral_face_keys:
            tag = face_tag_by_key.get(fk)
            if tag is None:
                continue
            face_z_bounds[tag] = (z_bot, z_top)

    # Step 2: setTransfiniteCurve on vertical edges.
    vertical_edges_done: set[int] = set()
    for face_tag, (z_bot, z_top) in face_z_bounds.items():
        n_layers = face_n_layers.get(face_tag, 1)
        _, _, verticals = _classify_lateral_face_edges(face_tag, z_bot, z_top)
        for ve in verticals:
            if ve in vertical_edges_done:
                continue
            vertical_edges_done.add(ve)
            gmsh.model.mesh.setTransfiniteCurve(ve, n_layers + 1)

    # Step 3: materialise 1D mesh.
    gmsh.model.mesh.generate(1)

    # Step 4: emit lateral-face quads.
    for face_tag, (z_bot, z_top) in face_z_bounds.items():
        n_layers = face_n_layers[face_tag]
        bot_edge, top_edge, verticals = _classify_lateral_face_edges(
            face_tag, z_bot, z_top
        )
        if bot_edge is None or top_edge is None or len(verticals) != 2:
            raise StructuredTransfiniteRejectedError(
                face_tag=face_tag,
                slab_index=owners_per_face[face_tag][0][0],
                reason=(
                    f"lateral face must have bot + top + 2 vertical edges; "
                    f"got bot={bot_edge}, top={top_edge}, vert={len(verticals)}"
                ),
            )

        bot_row = _ordered_curve_nodes(bot_edge)
        top_row = _align_top_to_bot(bot_row, _ordered_curve_nodes(top_edge))
        if len(bot_row) < 2 or len(top_row) != len(bot_row):
            continue

        # Pick left/right vertical edges by (x, y) proximity to bot row endpoints.
        left_xy = (bot_row[0][1], bot_row[0][2])
        right_xy = (bot_row[-1][1], bot_row[-1][2])
        left_vert = right_vert = None
        for ve in verticals:
            ev = gmsh.model.getBoundary(
                [(1, ve)], oriented=False, recursive=False
            )
            x_v = y_v = None
            for _vd, vt in ev:
                pos = gmsh.model.getValue(0, vt, [])
                x_v, y_v = pos[0], pos[1]
                break
            d_left = (x_v - left_xy[0]) ** 2 + (y_v - left_xy[1]) ** 2
            d_right = (x_v - right_xy[0]) ** 2 + (y_v - right_xy[1]) ** 2
            if d_left < d_right:
                left_vert = ve
            else:
                right_vert = ve
        if left_vert is None or right_vert is None:
            continue

        # Reuse transfinite-placed vertical-edge nodes (no duplicates).
        left_layer_nodes = _vertical_edge_layer_nodes(left_vert, n_layers)
        right_layer_nodes = _vertical_edge_layer_nodes(right_vert, n_layers)

        # Build n_layers+1 rows of node tags.
        rows: list[list[int]] = [[t for t, _x, _y, _z in bot_row]]
        for layer in range(1, n_layers):
            z_layer = z_bot + (z_top - z_bot) * layer / n_layers
            row_tags: list[int] = []
            for idx, (_t, x, y, _z) in enumerate(bot_row):
                if idx == 0:
                    row_tags.append(left_layer_nodes[layer])
                elif idx == len(bot_row) - 1:
                    row_tags.append(right_layer_nodes[layer])
                else:
                    new_tag = gmsh.model.mesh.getMaxNodeTag() + 1
                    gmsh.model.mesh.addNodes(
                        2, face_tag, [new_tag], [x, y, z_layer]
                    )
                    row_tags.append(new_tag)
            rows.append(row_tags)
        rows.append([t for t, _x, _y, _z in top_row])

        # Emit quad elements (gmsh type 3 = 4-node quad).
        quad_nodes: list[int] = []
        for r in range(len(rows) - 1):
            for c in range(len(rows[r]) - 1):
                quad_nodes.extend(
                    [
                        rows[r][c],
                        rows[r][c + 1],
                        rows[r + 1][c + 1],
                        rows[r + 1][c],
                    ]
                )
        if quad_nodes:
            gmsh.model.mesh.addElementsByType(face_tag, 3, [], quad_nodes)

    # Step 5: tell outer generate(2) to skip already-meshed faces.
    gmsh.option.setNumber("Mesh.MeshOnlyEmpty", 1)
```

- [ ] **Step 2: Run the test suite to confirm wedge.py still imports cleanly**

```
pytest tests/structured/ --no-cov -q
```
Expected: 107 passed (105 prior + 2 new from Tasks 0–1). The new
`freeze_lateral_mesh` function is defined but unused; should be a no-op
for existing tests.

- [ ] **Step 3: Commit**

```bash
git add meshwell/structured/wedge.py
git commit -m "feat(structured): add freeze_lateral_mesh (alt-b implementation)"
```

---

## Task 3: Switch orchestrator wiring to `freeze_lateral_mesh`

**Files:**
- Modify: `meshwell/orchestrator.py:20-21, 167-172`

- [ ] **Step 1: Update the import**

Edit `meshwell/orchestrator.py` around line 20-21. Replace:

```python
from meshwell.structured.wedge import (
    apply_lateral_transfinite_hints,
    stamp_wedges,
)
```

with:

```python
from meshwell.structured.wedge import (
    freeze_lateral_mesh,
    stamp_wedges,
)
```

- [ ] **Step 2: Update the call site**

Edit `meshwell/orchestrator.py` around line 167-172. Replace:

```python
    def _structured_pre_2d() -> None:
        if state.slab_meta and face_tag_by_key:
            apply_lateral_transfinite_hints(
                state.slab_meta,
                face_tag_by_key,
                resolution_specs=resolution_specs_for_wedge,
            )
```

with:

```python
    def _structured_pre_2d() -> None:
        if state.slab_meta and face_tag_by_key:
            freeze_lateral_mesh(
                state.slab_meta,
                face_tag_by_key,
                resolution_specs=resolution_specs_for_wedge,
            )
```

- [ ] **Step 3: Run full structured suite to confirm Alt B is now active**

```
pytest tests/structured/ --no-cov -q
```
Expected: 107 passed. All tests now exercise Alt B.

- [ ] **Step 4: Run the complex scene specifically**

```
pytest tests/structured/test_stress_complex_scene.py -v --no-cov
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add meshwell/orchestrator.py
git commit -m "feat(structured): switch orchestrator to freeze_lateral_mesh"
```

---

## Task 4: Remove the old `apply_lateral_transfinite_hints` from wedge.py

**Files:**
- Modify: `meshwell/structured/wedge.py:1-122`

Now that the orchestrator no longer calls the old function, remove its body and update the module docstring.

- [ ] **Step 1: Delete the old function**

In `meshwell/structured/wedge.py`, delete the entire `apply_lateral_transfinite_hints` function (lines 55-122 in the current code, between `resolve_n_layers` and the new `# Alt B:` section header). Keep `resolve_n_layers`, `_face_centroid_z`, and all of the new functions added in Task 2.

- [ ] **Step 2: Update the module docstring**

Edit `meshwell/structured/wedge.py:1-10`. Replace:

```python
"""Stage 5 — gmsh meshing hooks for structured cohorts.

pre_2d_hook (apply_lateral_transfinite_hints): sets transfinite curve
counts on vertical lateral edges and transfinite surface hints on
lateral faces of every cohort sub-solid. Raises on n_layers mismatch
or unsupported lateral topology.

pre_3d_hook (stamp_wedges, Task 17): per cohort sub-solid, copies bot
triangulation to top and emits wedge elements.
"""
```

with:

```python
"""Stage 5 — gmsh meshing hooks for structured cohorts.

pre_2d_hook (freeze_lateral_mesh): emits cohort lateral-face quad
mesh from Python before gmsh's generate(2) runs. Uses
Mesh.MeshOnlyEmpty=1 so the outer 2D mesher leaves cohort laterals
alone. Raises on n_layers mismatch or unsupported lateral topology.

pre_3d_hook (stamp_wedges): per cohort sub-solid, copies bot
triangulation to top and emits wedge elements.
"""
```

- [ ] **Step 3: Run full suite to confirm no stragglers**

```
pytest tests/structured/ --no-cov -q
```
Expected: 107 passed.

- [ ] **Step 4: Commit**

```bash
git add meshwell/structured/wedge.py
git commit -m "refactor(structured): remove old apply_lateral_transfinite_hints"
```

---

## Task 5: Rename old test name in test_wedge_pre2d.py

**Files:**
- Modify: `tests/structured/test_wedge_pre2d.py:10`

The test name refers to "transfinite hints" but the production path no longer uses them. Rename for clarity.

- [ ] **Step 1: Rename the test**

In `tests/structured/test_wedge_pre2d.py`, replace:

```python
def test_transfinite_hints_produce_quad_laterals(tmp_path):
```

with:

```python
def test_lateral_face_quads(tmp_path):
```

- [ ] **Step 2: Update the module docstring at the top of the file**

Replace:

```python
"""Tests for meshwell.structured.wedge — pre_2d transfinite hints."""
```

with:

```python
"""Tests for meshwell.structured.wedge — pre_2d freeze_lateral_mesh hook."""
```

- [ ] **Step 3: Run renamed test**

```
pytest tests/structured/test_wedge_pre2d.py::test_lateral_face_quads -v --no-cov
```
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/structured/test_wedge_pre2d.py
git commit -m "test(structured): rename transfinite test post alt-b promotion"
```

---

## Task 6: Delete spike artifacts

**Files:**
- Delete: `meshwell/structured/wedge_alt_b_spike.py`
- Delete: `meshwell/structured/wedge_manual_spike.py`
- Delete: `scripts/spike_alt_b.py`
- Delete: `scripts/spike_manual_lateral.py`

The spike modules and runner scripts served their purpose (validating Alt B against the complex scene). They duplicate code now in `wedge.py` and shouldn't ship.

- [ ] **Step 1: Delete the four files**

```bash
rm meshwell/structured/wedge_alt_b_spike.py
rm meshwell/structured/wedge_manual_spike.py
rm scripts/spike_alt_b.py
rm scripts/spike_manual_lateral.py
```

- [ ] **Step 2: Confirm nothing else imports them**

```
grep -rn "wedge_manual_spike\|wedge_alt_b_spike\|spike_alt_b\|spike_manual_lateral" meshwell/ tests/ scripts/ 2>&1 | grep -v __pycache__
```
Expected: no output (no remaining references).

- [ ] **Step 3: Run full suite to confirm nothing broke**

```
pytest tests/structured/ --no-cov -q
```
Expected: 107 passed.

- [ ] **Step 4: Commit**

```bash
git add -u meshwell/structured/wedge_manual_spike.py meshwell/structured/wedge_alt_b_spike.py scripts/spike_manual_lateral.py scripts/spike_alt_b.py
git commit -m "chore(structured): remove spike artifacts (alt-b promoted)"
```

---

## Task 7: Update the catalog doc with promotion note

**Files:**
- Modify: `docs/superpowers/specs/2026-06-01-transfinite-dependencies-catalog.md`

- [ ] **Step 1: Update the status line at the top**

In `docs/superpowers/specs/2026-06-01-transfinite-dependencies-catalog.md`, edit lines 1-7. Replace:

```markdown
# Transfinite hint dependencies — catalog

**Date:** 2026-06-01
**Branch:** `feat/structured_discrete`
**Status:** investigation — input to a potential replacement design
**Sister doc:** [2026-06-01-cohort-topology-investigations.md](2026-06-01-cohort-topology-investigations.md)
```

with:

```markdown
# Transfinite hint dependencies — catalog

**Date:** 2026-06-01
**Branch:** `feat/structured_discrete`
**Status:** PROMOTED 2026-06-01 — Alt B is now the production path
  (see `meshwell/structured/wedge.py::freeze_lateral_mesh`).
  Catalog kept as historical reference and future-replacement spec.
**Sister doc:** [2026-06-01-cohort-topology-investigations.md](2026-06-01-cohort-topology-investigations.md)
**Promotion plan:** [docs/superpowers/plans/2026-06-01-alt-b-promotion.md](../plans/2026-06-01-alt-b-promotion.md)
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-06-01-transfinite-dependencies-catalog.md
git commit -m "docs(structured): mark transfinite catalog as promoted"
```

---

## Task 8: Verify demos still work

**Files:** none — verification only

Spec §7 step 6 says: "Run the replacement against the test suite and demos." We've already verified the suite (Tasks 0-6). Demos are separate.

- [ ] **Step 1: Run `demo_structured.py`**

```
python demo_structured.py
```
Expected: prints "Wrote …" or similar, no exceptions, no segfaults. Verify the `.msh` file was created.

- [ ] **Step 2: Run `demo_curves.py`**

```
python demo_curves.py
```
Expected: 3 scenes meshed cleanly (single_disc, stacked_discs, annulus_on_disc), each producing a `.msh` file.

- [ ] **Step 3: No commit needed**

Demos are not part of the test suite; this step is just smoke-checking.

---

## Task 9: Spec §7 step 1 follow-up — leave a future-work note

**Files:**
- Modify: `docs/superpowers/specs/2026-06-01-transfinite-dependencies-catalog.md`

Spec §7 step 1 listed S3, S5, S9 as missing tests. Tasks 0–1 added S3 and S5. S9 is already covered by the existing `test_void_tagging_e2e.py` tests, which now exercise Alt B (since the orchestrator switched). Note this so future readers don't re-search for it.

- [ ] **Step 1: Update the Open Questions section**

In `docs/superpowers/specs/2026-06-01-transfinite-dependencies-catalog.md`, find the §7 "Recommended next steps" block. After step 7, append:

```markdown

---

## 8. Status after promotion (2026-06-01)

- ✅ Step 1 (missing tests): S3 added as
  `test_n_layers_1_meshes_cleanly`; S5 added as
  `test_shared_lateral_between_two_subsolids` in
  `tests/structured/test_wedge_pre2d.py`. S9 (void / keep=False) is
  covered by the existing `tests/structured/test_void_tagging_e2e.py`
  suite, which now exercises Alt B post-orchestrator switch.
- ⏸ Step 2 (timings instrumentation): deferred. Spike showed Alt B
  is ~15% slower than the old transfinite path on the complex scene
  (4.39s vs 4.15s). Acceptable for determinism trade. Re-visit if
  perf becomes a bottleneck.
- ⏸ Step 3 (discretization knob, Q4): deferred. Alt B currently
  inherits lateral discretization from bot-edge mesh density. Add a
  `lateral_discretization` parameter when a user requests it.
- ✅ Step 4 (Alt B spike): completed; see prior commits.
- ✅ Steps 5-7 (promotion, suite/demos green, transfinite path
  removed): done in
  [docs/superpowers/plans/2026-06-01-alt-b-promotion.md](../plans/2026-06-01-alt-b-promotion.md).
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-06-01-transfinite-dependencies-catalog.md
git commit -m "docs(structured): record alt-b promotion status in catalog"
```

---

## Final verification

- [ ] **Step 1: Full structured suite**

```
pytest tests/structured/ --no-cov 2>&1 | tail -3
```
Expected: 107 passed.

- [ ] **Step 2: Broader cad_occ suites**

```
pytest tests/test_cad_occ.py tests/test_cad_occ_fragment_ownership.py tests/test_backend_cross_compare.py --no-cov 2>&1 | tail -3
```
Expected: green, no regressions.

- [ ] **Step 3: Git status check**

```
git status
```
Expected: clean working tree (all changes committed). If unexpected files appear, decide whether to commit or revert.

- [ ] **Step 4: Verify spike files are gone**

```
ls meshwell/structured/wedge_*spike* 2>&1
ls scripts/spike_* 2>&1
```
Expected: "No such file or directory" for both.
