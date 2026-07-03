# WP1 — Silent Wrong Results: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix every bug from the 2026-07-03 review where meshwell silently produces wrong results or a documented feature does nothing (spec: `docs/superpowers/specs/2026-07-03-code-improvements-design.md`, WP1).

**Architecture:** Ten independent bug fixes on the `code_improvement` branch, each with a failing test first, one commit per task. No new modules; all changes are in existing files.

**Tech Stack:** Python 3.13, pytest, gmsh, OCP (OpenCascade), shapely, gdstk, meshio.

## Global Constraints

- Test baseline: 3 stable + 1 flaky pre-existing failures (recorded in Task 0). A task is done when the suite shows **no new failures** vs. baseline — not when it is all green.
- Run tests from repo root `/home/simbil/Github/meshwell_structured_manual/meshwell` with `uv run pytest ...` (env is uv-managed, Python 3.13).
- Pre-commit hooks run black/ruff/codespell — if a commit is rejected, fix the reported issue and re-commit; do not `--no-verify`.
- **Model dispatch (per user requirement):** each task header carries a `Model:` line — dispatch the implementing subagent with that model (`haiku` = mechanical, `sonnet` = standard, `opus` = topology-critical).
- Do not fix unrelated issues you notice; they are covered by WP2–WP7.

---

### Task 0: Record test baseline

**Model:** haiku

**Files:**
- Create: `docs/superpowers/plans/2026-07-03-wp1-baseline-failures.txt`

**Interfaces:**
- Produces: the baseline failure list every later task compares against.

- [ ] **Step 1: Run the full suite and capture failures** (no `-x` — the full failure list is the deliverable)

```bash
uv run pytest meshwell/tests -q --no-header | tail -20
uv run pytest meshwell/tests -q --no-header 2>&1 | grep -E "^(FAILED|ERROR)" > docs/superpowers/plans/2026-07-03-wp1-baseline-failures.txt
cat docs/superpowers/plans/2026-07-03-wp1-baseline-failures.txt
```

Expected: ~3-4 FAILED lines (3 stable + 1 flaky per project memory).

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/plans/2026-07-03-wp1-baseline-failures.txt
git commit -m "test: record WP1 baseline failures"
```

---

### Task 1: Remove dead `periodic_entities` plumbing

**Model:** sonnet

**Files:**
- Modify: `meshwell/mesh.py:109-142` (delete `_apply_periodic_boundaries`, `_set_periodic_pair`), `:506` (param), `:529` (docstring line), `:619` / `:648` / `:690` (public `mesh()` param, docstring line, forwarding kwarg)
- Test: `meshwell/tests/test_removed_apis.py` (create)

**Interfaces:**
- Produces: `Mesh.process_geometry` and `mesh()` no longer accept `periodic_entities`. Task 2 appends to the same test file.

- [ ] **Step 1: Write the failing test**

```python
"""APIs removed in the 2026-07-03 code-improvement effort stay removed."""
import inspect

from meshwell.mesh import Mesh, mesh


def test_periodic_entities_removed_from_mesh_wrapper():
    assert "periodic_entities" not in inspect.signature(mesh).parameters


def test_periodic_entities_removed_from_process_geometry():
    assert "periodic_entities" not in inspect.signature(Mesh.process_geometry).parameters


def test_periodic_helper_methods_deleted():
    assert not hasattr(Mesh, "_apply_periodic_boundaries")
    assert not hasattr(Mesh, "_set_periodic_pair")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest meshwell/tests/test_removed_apis.py -v`
Expected: 3 FAILED (parameter/attributes still present).

- [ ] **Step 3: Implement**

In `meshwell/mesh.py`:
1. Delete the entire `_apply_periodic_boundaries` method (lines 109-122) and `_set_periodic_pair` method (lines 124-142).
2. In `process_geometry`, delete the line `periodic_entities: list[tuple[str, str]] | None = None,  # noqa: ARG002` and the docstring line `periodic_entities: List of periodic boundary pairs`.
3. In the module-level `mesh()` function, delete the `periodic_entities: list[tuple[str, str]] | None = None,` parameter, its docstring line, and the `periodic_entities=periodic_entities,` forwarding line in the `process_geometry(...)` call.
4. Grep to confirm no remaining references: `grep -rn periodic meshwell/ --include="*.py" | grep -v tests` must return nothing.

- [ ] **Step 4: Run tests**

Run: `uv run pytest meshwell/tests/test_removed_apis.py meshwell/tests/test_mesh_in_memory.py -v`
Expected: new tests PASS; no new failures vs. baseline elsewhere (`uv run pytest meshwell/tests -q`).

- [ ] **Step 5: Commit**

```bash
git add meshwell/mesh.py meshwell/tests/test_removed_apis.py
git commit -m "feat!: remove no-op periodic_entities parameter

_apply_periodic_boundaries was never called; users requesting periodic
boundaries silently got a non-periodic mesh."
```

---

### Task 2: `additive=True` raises `NotImplementedError`

**Model:** sonnet

**Files:**
- Modify: `meshwell/occ_entity.py:37`, `meshwell/polyprism.py:106,115`, `meshwell/polysurface.py:77`, `meshwell/polyline.py:81`
- Test: `meshwell/tests/test_removed_apis.py` (append)

**Interfaces:**
- Consumes: test file from Task 1.
- Produces: all four entity classes raise on `additive=True` at construction; `additive=False` still round-trips through `to_dict`/`from_dict` unchanged.

- [ ] **Step 1: Write the failing test** (append to `test_removed_apis.py`)

```python
import pytest
from shapely.geometry import LineString, Polygon

from meshwell.occ_entity import OCC_entity
from meshwell.polyline import PolyLine
from meshwell.polyprism import PolyPrism
from meshwell.polysurface import PolySurface

_SQUARE = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


@pytest.mark.parametrize(
    "build",
    [
        lambda: PolyPrism(polygons=_SQUARE, buffers={0.0: 0.0, 1.0: 0.0},
                          physical_name="x", additive=True),
        lambda: PolySurface(polygons=_SQUARE, physical_name="x", additive=True),
        lambda: PolyLine(linestrings=LineString([(0, 0), (1, 1)]),
                         physical_name="x", additive=True),
        lambda: OCC_entity(occ_function=lambda: None, physical_name="x",
                           additive=True),
    ],
    ids=["polyprism", "polysurface", "polyline", "occ_entity"],
)
def test_additive_true_raises(build):
    with pytest.raises(NotImplementedError, match="additive"):
        build()


def test_additive_false_round_trips():
    ps = PolySurface(polygons=_SQUARE, physical_name="x", additive=False)
    assert ps.to_dict()["additive"] is False
    assert PolySurface.from_dict(ps.to_dict()).additive is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest meshwell/tests/test_removed_apis.py -v -k additive`
Expected: 4 FAILED (no error raised), round-trip test PASSES already.

- [ ] **Step 3: Implement**

Add to each of the four `__init__` methods, immediately after the signature (before any other logic), then keep the existing `self.additive = additive` assignment:

```python
if additive:
    raise NotImplementedError(
        "additive=True is not implemented: entities are always cut by "
        "higher-priority (lower mesh_order) entities. Remove the argument."
    )
```

In `meshwell/polyprism.py` also delete the duplicate `self.additive = additive` at line 115 (keep the one at line 106). In `meshwell/occ_entity.py` update the docstring line for `additive` (line 20) to say it is accepted only as `False` and reserved.

- [ ] **Step 4: Run tests**

Run: `uv run pytest meshwell/tests/test_removed_apis.py -v && uv run pytest meshwell/tests -q`
Expected: all new tests PASS; no new failures vs. baseline.

- [ ] **Step 5: Commit**

```bash
git add meshwell/occ_entity.py meshwell/polyprism.py meshwell/polysurface.py meshwell/polyline.py meshwell/tests/test_removed_apis.py
git commit -m "feat!: additive=True now raises NotImplementedError

The flag was stored and serialized but never read by either CAD
backend, so callers silently got cut semantics."
```

---

### Task 3: Unify closed-arc splitting (conformality bug)

**Model:** opus

**Files:**
- Modify: `meshwell/structured/build.py:200-250` (`_emit_edges_for_segments`), `:430-488` (`_flatten_decomposition_to_polyline_segments`), new module-level helpers near line 360
- Test: `meshwell/tests/test_structured_arc_split.py` (create)

**Interfaces:**
- Produces: `closed_arc_split_indices(n_pts) -> tuple[int, int, int]` and `ring_is_closed(start_xy, end_xy, point_tolerance) -> bool`, module-level in `meshwell/structured/build.py`, used by both paths.

**Background (why this matters):** `EdgeRegistry.arc_xy`'s cache key includes the quantized *mid* point (`build.py:139`). The horizontal-face path splits a closed circle at `q3 = (3*len)//4` while the lateral-face path uses `q3 = (mid_idx + len - 1)//2` — off by one for most point counts (n=8: 6 vs 5). Different mids ⇒ different TShapes for the same physical half-circle ⇒ duplicate coincident OCC edges ⇒ non-conformal structured meshes on discs/annuli. The two paths also test closedness differently (quantized-key equality vs per-axis `abs < tol`).

- [ ] **Step 1: Write the failing test**

```python
"""Both closed-arc split paths must produce identical (start, mid, end) triples."""
import math

import pytest

from meshwell.structured.build import (
    _flatten_decomposition_to_polyline_segments,
    closed_arc_split_indices,
    ring_is_closed,
)
from meshwell.geometry_entity import decompose_vertices_2d

TOL = 1e-3


def _circle_coords(n):
    # closed ring: first point repeated at the end
    pts = [
        (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]
    return pts + [pts[0]]


@pytest.mark.parametrize("n", [8, 11, 16, 33])
def test_split_indices_match_emit_path_convention(n):
    # the emit path uses n//4, n//2, (3*n)//4 — the helper must pin that
    n_pts = n + 1
    q1, mid, q3 = closed_arc_split_indices(n_pts)
    assert (q1, mid, q3) == (n_pts // 4, n_pts // 2, (n_pts * 3) // 4)


@pytest.mark.parametrize("n", [8, 11, 16, 33])
def test_lateral_flatten_uses_same_arc_midpoints(n):
    coords = _circle_coords(n)
    raw = decompose_vertices_2d(
        coords, z=0.0, point_tolerance=TOL, identify_arcs=True,
        min_arc_points=5, arc_tolerance=TOL,
    )
    closed_arcs = [s for s in raw if s.is_arc]
    assert closed_arcs, "decomposition should identify the circle as an arc"

    flat = _flatten_decomposition_to_polyline_segments(raw, TOL)
    arc_segs = [s for s in flat if s.kind == "arc"]
    assert len(arc_segs) == 2, "closed circle must split into two half-arcs"

    pts = closed_arcs[0].points
    q1, mid, q3 = closed_arc_split_indices(len(pts))

    def key(xy):
        return (round(xy[0] / TOL), round(xy[1] / TOL))

    # first half-arc: start -> q1 -> mid; second: mid -> q3 -> end
    assert key(arc_segs[0].mid) == key((pts[q1][0], pts[q1][1]))
    assert key(arc_segs[1].mid) == key((pts[q3][0], pts[q3][1]))


def test_ring_is_closed_matches_vertex_registry_quantization():
    assert ring_is_closed((0.0, 0.0), (0.0004, -0.0004), 1e-3)
    assert not ring_is_closed((0.0, 0.0), (0.002, 0.0), 1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest meshwell/tests/test_structured_arc_split.py -v`
Expected: ImportError (`closed_arc_split_indices` does not exist yet).

- [ ] **Step 3: Implement**

Add module-level helpers in `meshwell/structured/build.py` (place above `_PolylineSegment`, ~line 360):

```python
def closed_arc_split_indices(n_pts: int) -> tuple[int, int, int]:
    """Return (q1, mid, q3) point indices splitting a closed arc run in two.

    The horizontal-face emission path and the lateral-face flattening path
    MUST both use these indices: EdgeRegistry.arc_xy's cache key includes
    the quantized mid point, so divergent midpoints create duplicate
    coincident OCC edges for the same physical half-circle.
    """
    return n_pts // 4, n_pts // 2, (n_pts * 3) // 4


def ring_is_closed(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    point_tolerance: float,
) -> bool:
    """True if start/end quantize to the same key (VertexRegistry._key convention)."""
    s = point_tolerance
    return (
        round(start_xy[0] / s) == round(end_xy[0] / s)
        and round(start_xy[1] / s) == round(end_xy[1] / s)
    )
```

In `_emit_edges_for_segments` (build.py:209-234), replace the closed-arc branch internals:

```python
            if seg.is_arc:
                start = pts[0]
                end = pts[-1]
                is_closed = ring_is_closed(
                    (start[0], start[1]), (end[0], end[1]), self.point_tolerance
                )
                if is_closed:
                    quarter_idx, mid_idx, three_quarter_idx = closed_arc_split_indices(
                        len(pts)
                    )
                    # ... existing two arc_xy calls, unchanged ...
```

(The old `vertices._key` comparison and inline `mid_idx`/`quarter_idx`/`three_quarter_idx` computations are deleted; `round(x/s)` in `ring_is_closed` is exactly `VertexRegistry._key`'s quantization, so behavior on this path is unchanged.)

In `_flatten_decomposition_to_polyline_segments` (build.py:439-469), replace the closedness test and split indices:

```python
        if seg.is_arc:
            start_xy = (pts[0][0], pts[0][1])
            end_xy = (pts[-1][0], pts[-1][1])
            if ring_is_closed(start_xy, end_xy, point_tolerance):
                q1_idx, mid_idx, q3_idx = closed_arc_split_indices(len(pts))
                mid_xy = (pts[mid_idx][0], pts[mid_idx][1])
                # ... existing two _PolylineSegment appends, now using the
                #     shared q1_idx / q3_idx ...
            else:
                mid_idx = len(pts) // 2
                # ... existing single-arc append, unchanged ...
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest meshwell/tests/test_structured_arc_split.py meshwell/tests/test_arc_identification.py meshwell/tests/test_arc_extrusion.py meshwell/tests/test_arc_fusion.py -v`
Expected: all PASS. Then full suite: `uv run pytest meshwell/tests -q` — no new failures vs. baseline. If a structured reference mesh diverges, inspect: the *new* topology (shared TShape) is correct; regenerate that reference via `uv run python meshwell/tests/generate_references.py` only for affected cases and say so in the commit message.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/build.py meshwell/tests/test_structured_arc_split.py
git commit -m "fix(structured): unify closed-arc splitting across face paths

Horizontal and lateral paths split full circles at different q3 indices
and used different closedness tests, producing duplicate coincident OCC
arc edges (non-conformal meshes) on discs/annuli."
```

---

### Task 4: Remesh loads tetra (not surface triangles) from meshio meshes

**Model:** sonnet

**Files:**
- Modify: `meshwell/remesh.py:185-194` (`_load_mesh_data` meshio branch)
- Test: `meshwell/tests/test_remesh_load.py` (create)

**Interfaces:**
- Produces: for a meshio input, `self.triangles` holds tetra connectivity when tetra cells exist, else triangles; `self.vtags`/`self.triangles_tags` stay `None` (documented) in this branch.

- [ ] **Step 1: Write the failing test**

```python
"""_load_mesh_data must prefer volume cells for 3D meshio inputs."""
import meshio
import numpy as np

from meshwell.remesh import Remesher


def _mesh_3d():
    # one tetra with its four boundary triangles — mimics any real 3D mesh
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
    )
    tetra = np.array([[0, 1, 2, 3]])
    tris = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    return meshio.Mesh(points, [("triangle", tris), ("tetra", tetra)])


def test_meshio_input_prefers_tetra():
    r = Remesher()
    r._load_mesh_data(_mesh_3d())
    assert r.triangles.shape == (1, 4), "must load tetra, not boundary triangles"


def test_meshio_input_falls_back_to_triangles():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    m = meshio.Mesh(points, [("triangle", np.array([[0, 1, 2]]))])
    r = Remesher()
    r._load_mesh_data(m)
    assert r.triangles.shape == (1, 3)
```

(`Remesher` is defined at `meshwell/remesh.py:117` with an all-defaults constructor.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest meshwell/tests/test_remesh_load.py -v`
Expected: `test_meshio_input_prefers_tetra` FAILS with shape (4, 3) instead of (1, 4).

- [ ] **Step 3: Implement**

Replace the meshio branch in `_load_mesh_data` (remesh.py:185-194):

```python
        elif isinstance(input_mesh, meshio.Mesh):
            self.vxyz = input_mesh.points
            # Prefer volume cells: a 3D mesh always also contains its
            # boundary triangles, which must not drive the size field.
            if "tetra" in input_mesh.cells_dict:
                self.triangles = input_mesh.cells_dict["tetra"]
            elif "triangle" in input_mesh.cells_dict:
                self.triangles = input_mesh.cells_dict["triangle"]
            else:
                self.triangles = None
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest meshwell/tests/test_remesh_load.py -v && uv run pytest meshwell/tests -q -k remesh`
Expected: PASS; no new failures vs. baseline.

- [ ] **Step 5: Commit**

```bash
git add meshwell/remesh.py meshwell/tests/test_remesh_load.py
git commit -m "fix(remesh): load tetra cells before triangles from meshio input

Boundary triangles were driving the 3D size field."
```

---

### Task 5: Check `BRepAlgoAPI_Cut` success in the OCC backend

**Model:** sonnet

**Files:**
- Modify: `meshwell/cad_occ.py:527-546` (cut cascade), add `import warnings` to imports
- Test: `meshwell/tests/test_cad_occ_cut_failure.py` (create)

**Interfaces:**
- Produces: a failed/null cut keeps the pre-cut shape for that tool and emits a `UserWarning`; never feeds a null shape to `_unwrap_shape`.

- [ ] **Step 1: Write the failing test**

```python
"""A failed BRepAlgoAPI_Cut must warn and keep the uncut shape, not crash."""
import pytest
from shapely.geometry import Polygon

import meshwell.cad_occ as cad_occ_mod
from meshwell.polysurface import PolySurface


class _FailingCut:
    def __init__(self, *args):
        pass

    def SetFuzzyValue(self, v):
        pass

    def Build(self):
        pass

    def IsDone(self):
        return False

    def Shape(self):  # pragma: no cover — must not be reached
        raise AssertionError("Shape() must not be called when IsDone() is False")


def test_failed_cut_warns_and_keeps_shape(monkeypatch):
    monkeypatch.setattr(cad_occ_mod, "BRepAlgoAPI_Cut", _FailingCut)
    # two overlapping same-dim entities with different mesh_order → a cut is attempted
    a = PolySurface(polygons=Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                    physical_name="a", mesh_order=1)
    b = PolySurface(polygons=Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                    physical_name="b", mesh_order=2)
    proc = cad_occ_mod.CAD_OCC()
    with pytest.warns(UserWarning, match="Cut"):
        labeled = proc.process_entities_cut_only([a, b])
    # entity b keeps its (uncut) shape rather than losing it
    assert all(le.shapes for le in labeled)
```

Note: if `CAD_OCC()` requires constructor arguments or `process_entities_cut_only` needs extra parameters, mirror the minimal invocation used in `meshwell/tests/test_cad_occ.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest meshwell/tests/test_cad_occ_cut_failure.py -v`
Expected: FAIL — either `AssertionError: Shape() must not be called` or no warning raised.

- [ ] **Step 3: Implement**

Replace the inner cut loop (cad_occ.py:527-546) with:

```python
                new_shapes: list[TopoDS_Shape] = []
                for s in labeled.shapes:
                    result = s
                    for ts in tool_shapes:
                        cut_op = BRepAlgoAPI_Cut(result, ts)
                        cut_op.SetFuzzyValue(self.cut_fuzzy_value)
                        cut_op.Build()
                        if not cut_op.IsDone():
                            warnings.warn(
                                f"BRepAlgoAPI_Cut failed (not done) for entity "
                                f"{orig_idx} ({labeled.physical_name}); keeping "
                                f"uncut shape for this tool.",
                                stacklevel=2,
                            )
                            continue
                        shape = cut_op.Shape()
                        if shape is None or shape.IsNull():
                            warnings.warn(
                                f"BRepAlgoAPI_Cut returned a null shape for entity "
                                f"{orig_idx} ({labeled.physical_name}); keeping "
                                f"uncut shape for this tool.",
                                stacklevel=2,
                            )
                            continue
                        result = shape
                    # Flatten compound wrapper so BOPAlgo_Builder.Modified()
                    # in the final fragment pass tracks sub-shape provenance.
                    new_shapes.extend(self._unwrap_shape(result, labeled.dim))
                labeled.shapes = new_shapes
```

Delete the old `try/except Exception` wrapper and the `if result is not None` guard (a C++-level exception from `Build()` on valid inputs is a genuine bug we want loud). Add `import warnings` to the imports block.

- [ ] **Step 4: Run tests**

Run: `uv run pytest meshwell/tests/test_cad_occ_cut_failure.py meshwell/tests/test_cad_occ.py meshwell/tests/test_cad_occ_fragment_ownership.py -v`
Expected: PASS; then `uv run pytest meshwell/tests -q` — no new failures vs. baseline.

- [ ] **Step 5: Commit**

```bash
git add meshwell/cad_occ.py meshwell/tests/test_cad_occ_cut_failure.py
git commit -m "fix(cad_occ): check IsDone/IsNull after BRepAlgoAPI_Cut

A failed BOP returned a null TopoDS_Shape that passed the None guard
and reached _unwrap_shape."
```

---

### Task 6: Always warn on gmsh cut failure

**Model:** haiku

**Files:**
- Modify: `meshwell/cad_gmsh.py:441-453`, add `import warnings`
- Test: `meshwell/tests/test_cad_gmsh_cut_failure.py` (create)

**Interfaces:**
- Produces: cut failures emit `UserWarning` regardless of `progress_bars`.

- [ ] **Step 1: Write the failing test**

```python
"""gmsh cut failures must warn even with progress_bars=False (the default)."""
import gmsh
import pytest
from shapely.geometry import Polygon

from meshwell.cad_gmsh import CAD_GMSH
from meshwell.polysurface import PolySurface


def test_cut_failure_warns_without_progress_bars(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("synthetic cut failure")

    a = PolySurface(polygons=Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                    physical_name="a", mesh_order=1)
    b = PolySurface(polygons=Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                    physical_name="b", mesh_order=2)
    proc = CAD_GMSH()
    monkeypatch.setattr(gmsh.model.occ, "cut", boom)
    with pytest.warns(UserWarning, match="Cut failed"):
        proc.process_entities([a, b], progress_bars=False)
```

Note: mirror the minimal `CAD_GMSH()` construction/teardown used in `meshwell/tests/test_cad_gmsh.py` (it may need a `ModelManager` or a finalize step) — apply the monkeypatch only after construction so model setup still works.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest meshwell/tests/test_cad_gmsh_cut_failure.py -v`
Expected: FAIL — `DID NOT WARN`.

- [ ] **Step 3: Implement**

In `cad_gmsh.py:451-453` replace:

```python
                    except Exception as e:
                        if progress_bars:
                            print(f"Warning: Cut failed for entity {orig_idx}: {e}")
```

with:

```python
                    except Exception as e:
                        warnings.warn(
                            f"Cut failed for entity {orig_idx} "
                            f"({labeled_ent.physical_name}): {e}; proceeding with "
                            f"un-cut dimtags.",
                            stacklevel=2,
                        )
```

Add `import warnings` to the imports.

- [ ] **Step 4: Run tests**

Run: `uv run pytest meshwell/tests/test_cad_gmsh_cut_failure.py meshwell/tests/test_cad_gmsh.py -v && uv run pytest meshwell/tests -q`
Expected: PASS; no new failures vs. baseline.

- [ ] **Step 5: Commit**

```bash
git add meshwell/cad_gmsh.py meshwell/tests/test_cad_gmsh_cut_failure.py
git commit -m "fix(cad_gmsh): warn on cut failure regardless of progress_bars"
```

---

### Task 7: Warn (with entity name) when `addThruSections` fails

**Model:** haiku

**Files:**
- Modify: `meshwell/polyprism.py:206-216`, add `import warnings`
- Test: `meshwell/tests/test_polyprism_loft_failure.py` (create)

**Interfaces:**
- Produces: a failed loft still returns 0 (dropped volume) but emits `UserWarning` naming the entity.

- [ ] **Step 1: Write the failing test**

```python
"""A failed thru-sections loft must warn with the entity's physical_name."""
import gmsh
import pytest
from shapely.geometry import Polygon

from meshwell.polyprism import PolyPrism


def test_thrusections_failure_warns(monkeypatch):
    prism = PolyPrism(
        polygons=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="my_prism",
    )

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic loft failure")

    monkeypatch.setattr(gmsh.model.occ, "addThruSections", boom)
    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.model.add("loft_failure_test")
    try:
        with pytest.warns(UserWarning, match="my_prism"):
            tag = prism._create_volume_directly(prism.buffered_polygons[0])
        assert tag == 0
    finally:
        gmsh.model.remove()
```

Note: `_create_volume_directly` needs curve loops built first; if the direct call errors before reaching `addThruSections` (e.g. needs `occ` synchronize setup), follow the setup pattern from `meshwell/tests/test_prism.py` for constructing prisms against a live gmsh model.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest meshwell/tests/test_polyprism_loft_failure.py -v`
Expected: FAIL — `DID NOT WARN`.

- [ ] **Step 3: Implement**

In `polyprism.py:206-216` replace:

```python
        try:
            volume_dimtags = gmsh.model.occ.addThruSections(
                curve_loops, makeSolid=True, makeRuled=True
            )
            gmsh.model.occ.synchronize()
            if volume_dimtags and volume_dimtags[0][0] == 3:
                return volume_dimtags[0][1]
        except Exception:
            return 0

        return 0
```

with:

```python
        try:
            volume_dimtags = gmsh.model.occ.addThruSections(
                curve_loops, makeSolid=True, makeRuled=True
            )
            gmsh.model.occ.synchronize()
            if volume_dimtags and volume_dimtags[0][0] == 3:
                return volume_dimtags[0][1]
        except Exception as e:
            warnings.warn(
                f"addThruSections failed for PolyPrism {self.physical_name}: {e}; "
                f"this volume is DROPPED from the model.",
                stacklevel=2,
            )
            return 0

        warnings.warn(
            f"addThruSections produced no volume for PolyPrism "
            f"{self.physical_name}; this volume is DROPPED from the model.",
            stacklevel=2,
        )
        return 0
```

Add `import warnings` to imports.

- [ ] **Step 4: Run tests**

Run: `uv run pytest meshwell/tests/test_polyprism_loft_failure.py meshwell/tests/test_prism.py meshwell/tests/test_buffers_prism.py -v && uv run pytest meshwell/tests -q`
Expected: PASS; no new failures vs. baseline.

- [ ] **Step 5: Commit**

```bash
git add meshwell/polyprism.py meshwell/tests/test_polyprism_loft_failure.py
git commit -m "fix(polyprism): warn when a thru-sections loft drops a volume"
```

---

### Task 8: Hierarchical GDS import (subcells + paths)

**Model:** sonnet

**Files:**
- Modify: `meshwell/import_gds.py` (whole file is 65 lines)
- Test: `meshwell/tests/test_from_gds.py` (append)

**Interfaces:**
- Produces: `gdstk_to_shapely(cell, layer_tuple)` and `read_gds_layers(gds_file, cell_name=None, layers=None)` keep their signatures but see referenced-subcell polygons and path geometry (gdstk's `get_polygons(depth=None, include_paths=True)` resolves both).

- [ ] **Step 1: Write the failing test** (append to `meshwell/tests/test_from_gds.py`)

```python
import gdstk
import pytest

from meshwell.import_gds import read_gds_layers


def _write_hierarchical_gds(path):
    lib = gdstk.Library()
    child = lib.new_cell("CHILD")
    child.add(gdstk.rectangle((0, 0), (1, 1), layer=1, datatype=0))
    top = lib.new_cell("TOP")
    top.add(gdstk.rectangle((2, 0), (3, 1), layer=1, datatype=0))
    top.add(gdstk.Reference(child, origin=(5, 5)))
    top.add(gdstk.FlexPath([(0, 3), (4, 3)], 0.5, layer=2, datatype=0))
    lib.write_gds(path)
    return path


def test_subcell_polygons_are_imported(tmp_path):
    gds = _write_hierarchical_gds(tmp_path / "hier.gds")
    layers = read_gds_layers(gds, cell_name="TOP")
    geom = layers[(1, 0)]
    # own rectangle (1.0) + referenced child rectangle (1.0)
    assert geom.area == pytest.approx(2.0, rel=1e-6)


def test_path_layers_are_not_empty(tmp_path):
    gds = _write_hierarchical_gds(tmp_path / "hier.gds")
    layers = read_gds_layers(gds, cell_name="TOP")
    assert (2, 0) in layers
    assert layers[(2, 0)].area == pytest.approx(4 * 0.5, rel=1e-2)


def test_missing_top_cell_raises_value_error(tmp_path):
    lib = gdstk.Library()
    empty = tmp_path / "empty.gds"
    lib.write_gds(empty)
    with pytest.raises(ValueError, match="top-level"):
        read_gds_layers(empty)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest meshwell/tests/test_from_gds.py -v -k "subcell or path_layers or missing_top"`
Expected: subcell test FAILS (area 1.0 not 2.0); path test FAILS (empty geometry); missing-top test FAILS (`IndexError` not `ValueError`).

- [ ] **Step 3: Implement**

Rewrite the two functions in `meshwell/import_gds.py`:

```python
def gdstk_to_shapely(cell, layer_tuple):
    """Convert GDSTK polygons (hierarchy- and path-resolved) to Shapely geometries."""
    polygons = []
    layer, datatype = layer_tuple

    # depth=None descends through all referenced cells; include_paths=True
    # (gdstk default) converts FlexPath/RobustPath to polygons.
    for polygon in cell.get_polygons(depth=None, layer=layer, datatype=datatype):
        points = [(float(x), float(y)) for x, y in polygon.points]

        if len(points) >= 3:
            poly = sg.Polygon(points)
            if not poly.is_valid:
                fixed = poly.buffer(0)
                if isinstance(fixed, sg.Polygon):
                    polygons.append(fixed)
                elif isinstance(fixed, sg.MultiPolygon):
                    polygons.extend(list(fixed.geoms))
            else:
                polygons.append(poly)

    if polygons:
        return unary_union(polygons)
    return sg.MultiPolygon([])
```

and in `read_gds_layers`:

```python
    else:
        top_cells = library.top_level()
        if not top_cells:
            raise ValueError(f"No top-level cell found in GDS file '{gds_file}'")
        cell = top_cells[0]

    # Get all layers if none specified — from the fully resolved hierarchy,
    # so subcell-only and path-only layers are detected too.
    if layers is None:
        layers = {
            (polygon.layer, polygon.datatype)
            for polygon in cell.get_polygons(depth=None)
        }
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest meshwell/tests/test_from_gds.py -v && uv run pytest meshwell/tests -q`
Expected: PASS (old flat-GDS tests must still pass — `get_polygons` on a flat cell returns the same set); no new failures vs. baseline.

- [ ] **Step 5: Commit**

```bash
git add meshwell/import_gds.py meshwell/tests/test_from_gds.py
git commit -m "fix(import_gds): resolve cell hierarchy and paths

cell.polygons only held the cell's own polygons; referenced subcells
were silently dropped and path layers imported as empty."
```

---

### Task 9: Fix `filter_tags_by_target_dimension` (points case + crash)

**Model:** opus

**Files:**
- Modify: `meshwell/_mesh_entity.py:174-216`
- Test: `meshwell/tests/test_mesh_entity_filters.py` (create)

**Interfaces:**
- Consumes: `_MeshEntity` with `self.tags` (list[int]), `self.boundaries` (list[int] attribute), `self.dim`, `self.model` (gmsh model object).
- Produces: correct tags for every `(self.dim, target_dimension)` pair: same dim → tags; one below → boundaries; points (target 0, dim ≥ 2) → recursive-boundary points; curves from volumes (dim 3, target 1) → curve-loop curves; impossible (target > dim) → warn + `[]`. Never `UnboundLocalError`; never returns curves when points were requested. `mesh.py:324-327` iterates `range(entity.dim + 1)`, so every case is hit in production.

- [ ] **Step 1: Write the failing test**

```python
"""filter_tags_by_target_dimension must return the right dimension's tags."""
import gmsh
import pytest

from meshwell._mesh_entity import _MeshEntity


@pytest.fixture
def box_entity():
    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.model.add("filter_dim_test")
    tag = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    ent = _MeshEntity.__new__(_MeshEntity)  # bypass full __init__ plumbing
    ent.model = gmsh.model
    ent.dim = 3
    ent.tags = [tag]
    ent.boundaries = [
        abs(t) for d, t in gmsh.model.getBoundary([(3, tag)], oriented=False)
    ]
    yield ent
    gmsh.model.remove()


def test_same_dim_returns_tags(box_entity):
    assert box_entity.filter_tags_by_target_dimension(3) == box_entity.tags


def test_one_below_returns_boundaries(box_entity):
    assert sorted(box_entity.filter_tags_by_target_dimension(2)) == sorted(
        box_entity.boundaries
    )


def test_curves_from_volume(box_entity):
    curves = box_entity.filter_tags_by_target_dimension(1)
    assert len(set(curves)) == 12  # a box has 12 edges


def test_points_from_volume_returns_points_not_curves(box_entity):
    points = box_entity.filter_tags_by_target_dimension(0)
    expected = {
        t for d, t in gmsh.model.getBoundary(
            [(3, box_entity.tags[0])], combined=False, oriented=False, recursive=True
        ) if d == 0
    }
    assert set(points) == expected
    assert len(expected) == 8  # a box has 8 corners


def test_points_from_surface_returns_points(box_entity):
    surf = box_entity.boundaries[0]
    ent = _MeshEntity.__new__(_MeshEntity)
    ent.model = gmsh.model
    ent.dim = 2
    ent.tags = [surf]
    ent.boundaries = [
        abs(t) for d, t in gmsh.model.getBoundary([(2, surf)], oriented=False)
    ]
    points = ent.filter_tags_by_target_dimension(0)
    assert len(set(points)) == 4  # a face has 4 corners, not its 4 curves


def test_target_above_dim_warns_and_returns_empty(box_entity):
    ent = _MeshEntity.__new__(_MeshEntity)
    ent.model = gmsh.model
    ent.dim = 1
    ent.tags = [1]
    ent.boundaries = []
    with pytest.warns(UserWarning):
        assert ent.filter_tags_by_target_dimension(3) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest meshwell/tests/test_mesh_entity_filters.py -v`
Expected: `points_from_volume` and `points_from_surface` FAIL (curves returned); `target_above_dim` FAILS with `UnboundLocalError` for dim 1 → target 3.

- [ ] **Step 3: Implement**

Replace the body of `filter_tags_by_target_dimension` (keep signature and docstring, update the docstring's Returns section):

```python
        diff = self.dim - target_dimension

        if diff < 0:
            warnings.warn(
                f"Target dimension {target_dimension} exceeds entity dimension "
                f"{self.dim}; skipping resolution assignment.",
                stacklevel=2,
            )
            return []
        if diff == 0:
            return list(self.tags)
        if diff == 1:
            return list(self.boundaries)
        if target_dimension == 0:
            # Points from a surface (diff 2) or volume (diff 3): recursive
            # boundary walk straight to dimension 0.
            dimtags = self.model.getBoundary(
                [(self.dim, tag) for tag in self.tags],
                combined=False,
                oriented=False,
                recursive=True,
            )
            return [tag for dim, tag in dimtags if dim == 0]

        # Remaining case: dim 3, target 1 — curves via each boundary
        # surface's curve loops.
        tags: list[int] = []
        for b in self.boundaries:
            try:
                for cs in self.model.occ.getCurveLoops(b)[1]:
                    tags.extend(cs)
            except Exception as e:
                # Surface may be unknown to OCC (e.g. discrete); skip it.
                import logging

                logging.getLogger(__name__).debug(
                    f"Failed to get curve loops for {b}: {e}"
                )
        return tags
```

(`warnings` is already imported in `_mesh_entity.py` — verify, else add.)

- [ ] **Step 4: Run tests**

Run: `uv run pytest meshwell/tests/test_mesh_entity_filters.py meshwell/tests/test_resolution.py meshwell/tests/test_lines_circles_resolution.py -v && uv run pytest meshwell/tests -q`
Expected: PASS; no new failures vs. baseline. If a resolution test changes behavior, it is because point-targeted specs previously matched curve tags — inspect and confirm the new behavior is the documented one before adjusting any test.

- [ ] **Step 5: Commit**

```bash
git add meshwell/_mesh_entity.py meshwell/tests/test_mesh_entity_filters.py
git commit -m "fix(_mesh_entity): return points (not curves) for target dim 0

Also removes the UnboundLocalError path for target dims more than one
above the entity's."
```

---

### Task 10: Exact-name matching in the legacy sharing fallback

**Model:** sonnet

**Files:**
- Modify: `meshwell/_mesh_entity.py:360, 405-423`
- Test: `meshwell/tests/test_mesh_entity_filters.py` (append)

**Interfaces:**
- Consumes: Task 9's test file.
- Produces: module-level helper `entity_name_set(physical_name) -> set[str]` in `meshwell/_mesh_entity.py`; the legacy fallback matches names exactly (no substring / per-character matching); the indexed path is used whenever `tag_to_entity_names is not None` (empty dict included).

- [ ] **Step 1: Write the failing test** (append)

```python
from meshwell._mesh_entity import entity_name_set


def test_entity_name_set_from_str():
    assert entity_name_set("metal") == {"metal"}


def test_entity_name_set_from_tuple():
    assert entity_name_set(("metal", "conductor")) == {"metal", "conductor"}


def test_no_substring_matching():
    # "metal" must NOT be treated as matching "metal2"
    assert "metal2" not in entity_name_set("metal")
    assert not entity_name_set("metal") & entity_name_set("metal2")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest meshwell/tests/test_mesh_entity_filters.py -v -k name_set`
Expected: ImportError (`entity_name_set` does not exist).

- [ ] **Step 3: Implement**

Add near the top of `meshwell/_mesh_entity.py` (module level, after imports):

```python
def entity_name_set(physical_name: str | tuple[str, ...]) -> set[str]:
    """Normalize a physical_name (str or tuple of str) to a set of exact names.

    Matching between entities must always be by exact name — substring or
    per-character containment makes "metal" match "metal2".
    """
    if isinstance(physical_name, str):
        return {physical_name}
    return set(physical_name)
```

Then in the sharing-resolution method:

1. Line 360: change `if tag_to_entity_names:` to `if tag_to_entity_names is not None:` (an empty index means "indexed path, nothing shared", not "fall back to legacy scanning").
2. Lines 362-365: replace the inline str/tuple normalization with `self_names = entity_name_set(self.physical_name)`.
3. Line 409: replace `if all(item in other_name for item in self.physical_name):` with `if other_name in entity_name_set(self.physical_name):`.
4. Line 423: replace `if any(item in other_name for item in superset):` with `if other_name in superset:`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest meshwell/tests/test_mesh_entity_filters.py meshwell/tests/test_resolution.py meshwell/tests/test_interface_tag.py -v && uv run pytest meshwell/tests -q`
Expected: PASS; no new failures vs. baseline.

- [ ] **Step 5: Commit**

```bash
git add meshwell/_mesh_entity.py meshwell/tests/test_mesh_entity_filters.py
git commit -m "fix(_mesh_entity): exact-name matching in legacy sharing fallback

Per-character/substring matching made 'metal' match 'metal2'; the
fallback also now only triggers when no reverse index was built at all."
```

---

## Final verification (orchestrator, after Task 10)

- [ ] `uv run pytest meshwell/tests -q` — compare against `2026-07-03-wp1-baseline-failures.txt`; require no new failures.
- [ ] `git log --oneline` shows one commit per task.
- [ ] Run superpowers:requesting-code-review on the WP1 diff before starting WP2.
