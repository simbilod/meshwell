# Phase C: Cross-backend test parametrization

**Status:** ready for implementation
**Date:** 2026-04-26
**Depends on:** Phase A+B (cad_occ port) — committed at HEAD `eb25299`

## Problem

`cad_gmsh` and `cad_occ` now share the shapely pre-pass and the same overall
pipeline (buffer → resolve InterfaceTags → instantiate + sequential cut → final
fragment), but they're tested independently. Divergence between backends —
either accidental drift or behavioural differences hidden by separate test
suites — won't be caught until a downstream user runs into it.

Goal: run a representative set of CAD+mesh scenarios through both backends in
the same test session and assert the meshes are equivalent at the level of
physical groups, element types, and per-group masses (within a tolerance that
accommodates the small `perturbation` differences in floating-point geometry
ops).

## Solution

Two complementary mechanisms:

1. **A `cad_pipeline` pytest fixture** that abstracts the backend difference.
   Tests requesting it get a callable that, given an entity list, runs the full
   pipeline (CAD + mesh) through one backend and returns the resulting
   `meshio.Mesh`. pytest's `params=["gmsh", "occ"]` makes each consuming test
   run twice — once per backend.

2. **A new `tests/test_backend_cross_compare.py`** with explicit side-by-side
   tests that run BOTH backends on the same input and assert structural
   equivalence (set of physical names, mass per group within tolerance).
   Catches divergence directly rather than relying on independent assertions
   matching by coincidence.

Existing backend-specific tests (`test_cad_gmsh.py`, `test_cad_occ.py`) stay
backend-specific — they test backend-internal state (`OCCLabeledEntity.shapes`,
gmsh model physical groups before write) that doesn't translate cleanly. Only
NEW tests written from scratch use the cross-backend pattern.

## Architecture

### `cad_pipeline` fixture (in `tests/conftest.py`)

```python
@pytest.fixture(params=["gmsh", "occ"])
def cad_pipeline(request, tmp_path):
    """Run the full CAD+mesh pipeline through a backend.

    Tests requesting this fixture run twice -- once per backend -- and
    receive a callable ``run(entities, dim=3, **mesh_kwargs) -> meshio.Mesh``.
    The callable handles the backend-specific glue (in-memory model for
    cad_gmsh; xao round-trip for cad_occ) so the test body can focus on
    asserting properties of the final mesh.

    The current backend name is exposed as ``run.backend`` for tests that
    need to skip a known-incompatible scene per backend.
    """
    backend = request.param
    msh_path = tmp_path / "out.msh"

    if backend == "gmsh":
        from meshwell.cad_gmsh import cad_gmsh
        from meshwell.mesh import mesh

        def run(entities, dim=3, **mesh_kwargs):
            _, mm = cad_gmsh(entities)
            return mesh(
                model=mm,
                output_file=str(msh_path),
                dim=dim,
                n_threads=1,
                **mesh_kwargs,
            )
    else:
        from meshwell.cad_occ import cad_occ
        from meshwell.mesh import mesh
        from meshwell.occ_xao_writer import write_xao

        def run(entities, dim=3, **mesh_kwargs):
            labeled = cad_occ(entities)
            xao_path = tmp_path / "out.xao"
            write_xao(labeled, str(xao_path))
            return mesh(
                input_file=str(xao_path),
                output_file=str(msh_path),
                dim=dim,
                n_threads=1,
                **mesh_kwargs,
            )

    run.backend = backend
    return run
```

Notes:
- `n_threads=1` is forced. Multi-threaded meshing is non-deterministic (node
  ordering varies), and we want bit-stable per-backend output for the
  side-by-side comparator below.
- Tests that need backend-specific behaviour can branch on `run.backend`.
- `mm.finalize()` is intentionally NOT called by the fixture — the meshio
  output is what's compared, not the gmsh model state. tmp_path teardown
  cleans the .msh and .xao.

### Side-by-side comparator (used by `test_backend_cross_compare.py`)

A helper in the new test file:

```python
def _mesh_summary(m: meshio.Mesh) -> dict[str, dict[str, tuple[int, float]]]:
    """Per-(physical_group, element_type) -> (count, total_mass).

    Mass = sum of cell volumes / areas / lengths over the cells in that
    (group, type) bucket. Element types limited to tetra/triangle/line.
    """
    summary: dict[str, dict[str, tuple[int, float]]] = {}
    for name, cell_arrays in m.cell_sets_dict.items():
        for cell_type, indices in cell_arrays.items():
            if cell_type not in ("tetra", "triangle", "line"):
                continue
            block = next(c for c in m.cells if c.type == cell_type)
            count = len(indices)
            mass = _cells_mass(m.points, block.data[indices], cell_type)
            summary.setdefault(name, {})[cell_type] = (count, mass)
    return summary
```

`_cells_mass` is a small geometric helper: tetra volume via signed-volume
formula, triangle area via cross product, line length via norm.

The comparator asserts:
- Both summaries have the same set of `(group, type)` keys.
- Counts MAY differ (mesher non-determinism), but masses must agree within
  a relative tolerance (`1e-3` is realistic given perturbation effects).

## Test scenes (`tests/test_backend_cross_compare.py`)

Six representative scenes, each as a side-by-side test:

1. **`test_two_abutting_prisms`** — two unit-square prisms sharing an interface.
   Compare: `{A, B, A___B, A___None, B___None}` present on both; volumes match.

2. **`test_three_abutting_prisms`** — A/B/C in a row. Compare: shared interfaces
   tagged correctly on both backends.

3. **`test_overlapping_polysurfaces`** — 2D scene with `mesh_order` cascade.
   Compare: winner's area vs loser's area sum to the union area on both.

4. **`test_donut_with_inner_prism`** — outer prism with a hole, inner prism in
   the hole. Compare: inner volume on both, hole interface tagged.

5. **`test_polyprism_with_interface_tag`** — tests that `InterfaceTag` produces
   equivalent output on both backends. Compare: `iface` group exists with
   matching face area.

6. **`test_keep_false_helper`** — a helper prism (`mesh_bool=False`) carving an
   interface in a kept neighbour. Compare: helper not in mesh on either, but
   interface still tagged.

For each scene, also expose a parametrized version that uses the
`cad_pipeline` fixture so it runs through both backends and asserts the
expected scene-specific properties (independent of backend).

## Out of scope

- Refactoring existing `test_cad_gmsh.py` / `test_cad_occ.py` tests — they
  verify backend-internal state and stay backend-specific.
- Refactoring existing `test_interface_tag.py` e2e tests — they're written
  against the gmsh in-memory API; converting them to use `cad_pipeline` is
  invasive and adds little value beyond what the new cross-compare tests
  already provide.
- Mesh-level assertions on node coordinates or element ordering — non-portable
  across backends and across pytest runs.

---

# Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute this plan task-by-task.

**Goal:** Add cross-backend test parametrization so divergence between cad_gmsh and cad_occ is caught in CI.

**Architecture:** A `cad_pipeline` pytest fixture abstracts the backend difference; a new `tests/test_backend_cross_compare.py` file contains both backend-parametrized tests AND explicit side-by-side comparators.

**Tech Stack:** pytest (parametrize, tmp_path), meshio, numpy, shapely.

---

## Task 1: Add `cad_pipeline` fixture to `tests/conftest.py`

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1.1: Append the fixture**

Append to `tests/conftest.py`:

```python
@pytest.fixture(params=["gmsh", "occ"])
def cad_pipeline(request, tmp_path):
    """Run the full CAD+mesh pipeline through one backend.

    Tests requesting this fixture run twice (once per backend) and receive
    a callable ``run(entities, dim=3, **mesh_kwargs) -> meshio.Mesh``. The
    callable abstracts the in-memory (cad_gmsh) vs xao-roundtrip (cad_occ)
    glue. ``run.backend`` is the active backend name ("gmsh" or "occ").
    """
    backend = request.param
    msh_path = tmp_path / "out.msh"

    if backend == "gmsh":
        from meshwell.cad_gmsh import cad_gmsh
        from meshwell.mesh import mesh

        def run(entities, dim=3, **mesh_kwargs):
            _, mm = cad_gmsh(entities)
            return mesh(
                model=mm,
                output_file=str(msh_path),
                dim=dim,
                n_threads=1,
                **mesh_kwargs,
            )
    else:
        from meshwell.cad_occ import cad_occ
        from meshwell.mesh import mesh
        from meshwell.occ_xao_writer import write_xao

        def run(entities, dim=3, **mesh_kwargs):
            labeled = cad_occ(entities)
            xao_path = tmp_path / "out.xao"
            write_xao(labeled, str(xao_path))
            return mesh(
                input_file=str(xao_path),
                output_file=str(msh_path),
                dim=dim,
                n_threads=1,
                **mesh_kwargs,
            )

    run.backend = backend
    return run
```

- [ ] **Step 1.2: Verify the fixture loads**

```bash
pytest --fixtures tests/ 2>&1 | grep -A 3 cad_pipeline
```
Expected: shows `cad_pipeline` parameterized with gmsh and occ.

- [ ] **Step 1.3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add cad_pipeline fixture parametrized over gmsh/occ backends"
```

---

## Task 2: Add the side-by-side comparator helpers

**Files:**
- Create: `tests/test_backend_cross_compare.py`

- [ ] **Step 2.1: Create the file with helpers and one smoke test**

Create `tests/test_backend_cross_compare.py`:

```python
"""Cross-backend equivalence tests.

Each test in this file exercises both ``cad_gmsh`` and ``cad_occ`` on the
same input and asserts the output meshes are equivalent at the level of
physical groups and per-group geometric mass.

Two patterns are used:

1. Parametrized tests via the ``cad_pipeline`` fixture (in conftest.py):
   pytest runs the test twice -- once per backend -- and the same
   assertions hit both. Useful for tests where the assertion is
   self-contained and backend-agnostic.

2. Side-by-side tests that call BOTH backends explicitly in one test
   body and compare summaries. Useful for catching subtle divergence
   that wouldn't trigger an isolated assertion failure.
"""
from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np
import shapely
from shapely.geometry import LineString

from meshwell.cad_gmsh import cad_gmsh
from meshwell.cad_occ import cad_occ
from meshwell.interface_tag import InterfaceTag
from meshwell.mesh import mesh
from meshwell.occ_xao_writer import write_xao
from meshwell.polyprism import PolyPrism
from meshwell.polysurface import PolySurface

# ----- Geometric mass helpers -----------------------------------------------


def _tet_volume(p0, p1, p2, p3) -> float:
    """Signed tetrahedron volume."""
    return abs(np.dot(p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0


def _tri_area(p0, p1, p2) -> float:
    """Triangle area via cross product."""
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))


def _line_length(p0, p1) -> float:
    return float(np.linalg.norm(p1 - p0))


def _cells_mass(points: np.ndarray, cells: np.ndarray, cell_type: str) -> float:
    """Sum geometric mass (volume / area / length) over a block of cells."""
    total = 0.0
    if cell_type == "tetra":
        for c in cells:
            p0, p1, p2, p3 = points[c]
            total += _tet_volume(p0, p1, p2, p3)
    elif cell_type == "triangle":
        for c in cells:
            p0, p1, p2 = points[c]
            total += _tri_area(p0, p1, p2)
    elif cell_type == "line":
        for c in cells:
            p0, p1 = points[c]
            total += _line_length(p0, p1)
    return total


# ----- Side-by-side comparator ----------------------------------------------


def _mesh_summary(m: meshio.Mesh) -> dict[str, dict[str, tuple[int, float]]]:
    """Per-(physical_group, element_type) -> (count, total_mass).

    Mass is geometric: tetra volume, triangle area, line length.
    """
    summary: dict[str, dict[str, tuple[int, float]]] = {}
    for name, cell_arrays in m.cell_sets_dict.items():
        for cell_type, indices in cell_arrays.items():
            if cell_type not in ("tetra", "triangle", "line"):
                continue
            block = next(b for b in m.cells if b.type == cell_type)
            cells = block.data[indices]
            count = len(indices)
            mass = _cells_mass(m.points, cells, cell_type)
            summary.setdefault(name, {})[cell_type] = (count, mass)
    return summary


def _assert_summaries_equivalent(
    s_gmsh: dict[str, dict[str, tuple[int, float]]],
    s_occ: dict[str, dict[str, tuple[int, float]]],
    rel_tol: float = 1e-3,
    ignore_groups: set[str] = frozenset(),
) -> None:
    """Two summaries are equivalent if they have the same (group, type)
    keys and per-key masses within ``rel_tol`` (relative).

    Element COUNTS are NOT required to match (mesher non-determinism is
    fine -- only the integrated mass per group matters).
    """
    g_keys = {(g, t) for g, types in s_gmsh.items() if g not in ignore_groups
              for t in types}
    o_keys = {(g, t) for g, types in s_occ.items() if g not in ignore_groups
              for t in types}
    assert g_keys == o_keys, (
        f"Group/type sets differ.\n"
        f"  gmsh-only: {g_keys - o_keys}\n"
        f"  occ-only:  {o_keys - g_keys}"
    )
    for g, t in g_keys:
        gc, gm = s_gmsh[g][t]
        oc, om = s_occ[g][t]
        if gm == 0.0 and om == 0.0:
            continue
        denom = max(abs(gm), abs(om))
        rel = abs(gm - om) / denom if denom > 0 else 0.0
        assert rel < rel_tol, (
            f"Mass mismatch on ({g!r}, {t!r}): gmsh={gm:.6g} (n={gc}) "
            f"vs occ={om:.6g} (n={oc}); rel={rel:.3e} > rel_tol={rel_tol:.3e}"
        )


def _run_both(entities_factory, tmp_path: Path, dim: int = 3) -> tuple[meshio.Mesh, meshio.Mesh]:
    """Run the same scene through gmsh and occ backends.

    ``entities_factory`` is a zero-arg callable returning a fresh entities
    list -- needed because the pre-pass mutates entities in place.
    """
    # gmsh path
    gmsh_msh = tmp_path / "gmsh.msh"
    _, mm = cad_gmsh(entities_factory())
    m_gmsh = mesh(model=mm, output_file=str(gmsh_msh), dim=dim, n_threads=1)
    mm.finalize()

    # occ path
    occ_msh = tmp_path / "occ.msh"
    occ_xao = tmp_path / "occ.xao"
    labeled = cad_occ(entities_factory())
    write_xao(labeled, str(occ_xao))
    m_occ = mesh(
        input_file=str(occ_xao),
        output_file=str(occ_msh),
        dim=dim,
        n_threads=1,
    )
    return m_gmsh, m_occ


# ----- Smoke test -----------------------------------------------------------


def test_smoke_two_unit_prisms_match(tmp_path):
    """Sanity check that the comparator works on a trivial scene."""
    def make():
        A = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        B = shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)])
        buffers = {0.0: 0.0, 2.0: 0.0}
        return [
            PolyPrism(polygons=A, buffers=buffers, physical_name="A", mesh_order=1),
            PolyPrism(polygons=B, buffers=buffers, physical_name="B", mesh_order=2),
        ]

    m_gmsh, m_occ = _run_both(make, tmp_path)
    s_gmsh = _mesh_summary(m_gmsh)
    s_occ = _mesh_summary(m_occ)
    _assert_summaries_equivalent(s_gmsh, s_occ)
```

- [ ] **Step 2.2: Run the smoke test**

```bash
pytest tests/test_backend_cross_compare.py::test_smoke_two_unit_prisms_match -v --no-cov
```
Expected: PASS (single test).

If it fails, the failure message will pinpoint either a missing physical group on one backend or a mass discrepancy. STOP and report — that's a real backend divergence to investigate before adding more tests.

- [ ] **Step 2.3: Commit**

```bash
git add tests/test_backend_cross_compare.py
git commit -m "test(cross_compare): add side-by-side comparator and smoke test"
```

---

## Task 3: Three abutting prisms (side-by-side)

**Files:**
- Modify: `tests/test_backend_cross_compare.py` (append)

- [ ] **Step 3.1: Append the test**

```python
def test_three_abutting_prisms_match(tmp_path):
    """Three prisms in a row. Both backends must tag both interfaces
    (A___B and B___C) and produce equivalent volumes per region."""
    def make():
        A = shapely.Polygon([(0, 0), (2, 0), (2, 5), (0, 5)])
        B = shapely.Polygon([(2, 0), (5, 0), (5, 5), (2, 5)])
        C = shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)])
        buffers = {0.0: 0.0, 1.0: 0.0}
        return [
            PolyPrism(polygons=A, buffers=buffers, physical_name="A", mesh_order=1),
            PolyPrism(polygons=B, buffers=buffers, physical_name="B", mesh_order=2),
            PolyPrism(polygons=C, buffers=buffers, physical_name="C", mesh_order=3),
        ]

    m_gmsh, m_occ = _run_both(make, tmp_path)
    s_gmsh = _mesh_summary(m_gmsh)
    s_occ = _mesh_summary(m_occ)

    # Both backends must produce both interfaces.
    for iface in ("A___B", "B___C"):
        assert iface in s_gmsh, (s_gmsh.keys(), "gmsh missing", iface)
        assert iface in s_occ, (s_occ.keys(), "occ missing", iface)

    _assert_summaries_equivalent(s_gmsh, s_occ)
```

- [ ] **Step 3.2: Run + commit**

```bash
pytest tests/test_backend_cross_compare.py::test_three_abutting_prisms_match -v --no-cov
git add tests/test_backend_cross_compare.py
git commit -m "test(cross_compare): three abutting prisms"
```

---

## Task 4: Overlapping polysurfaces (2D, mesh_order cascade)

```python
def test_overlapping_polysurfaces_match(tmp_path):
    """2D scene with overlapping polysurfaces. Winner's area + loser's
    surviving area must equal the union area, on both backends."""
    def make():
        A = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        B = shapely.Polygon([(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)])
        return [
            PolySurface(polygons=A, physical_name="A", mesh_order=1),
            PolySurface(polygons=B, physical_name="B", mesh_order=2),
        ]

    m_gmsh, m_occ = _run_both(make, tmp_path, dim=2)
    s_gmsh = _mesh_summary(m_gmsh)
    s_occ = _mesh_summary(m_occ)
    _assert_summaries_equivalent(s_gmsh, s_occ)
```

Run + commit message: `test(cross_compare): overlapping polysurfaces`

---

## Task 5: Donut with inner prism

```python
def test_donut_with_inner_prism_match(tmp_path):
    """Outer prism with a hole, inner prism filling it. Both backends
    must tag the hole interface and produce matching inner volumes."""
    def make():
        outer = shapely.Polygon(
            shell=[(0, 0), (10, 0), (10, 10), (0, 10)],
            holes=[[(4, 4), (6, 4), (6, 6), (4, 6)]],
        )
        inner = shapely.Polygon([(4, 4), (6, 4), (6, 6), (4, 6)])
        buffers = {0.0: 0.0, 1.0: 0.0}
        return [
            PolyPrism(polygons=outer, buffers=buffers, physical_name="O", mesh_order=2),
            PolyPrism(polygons=inner, buffers=buffers, physical_name="I", mesh_order=1),
        ]

    m_gmsh, m_occ = _run_both(make, tmp_path)
    s_gmsh = _mesh_summary(m_gmsh)
    s_occ = _mesh_summary(m_occ)
    _assert_summaries_equivalent(s_gmsh, s_occ)
```

Run + commit message: `test(cross_compare): donut with inner prism`

---

## Task 6: PolyPrism with InterfaceTag

```python
def test_polyprism_with_interface_tag_match(tmp_path):
    """Two abutting prisms + one InterfaceTag at their shared face.
    Both backends must produce an `iface` physical group with matching
    face area."""
    def make():
        A = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        B = shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)])
        buffers = {0.0: 0.0, 1.0: 0.0}
        return [
            PolyPrism(polygons=A, buffers=buffers, physical_name="A", mesh_order=1),
            PolyPrism(polygons=B, buffers=buffers, physical_name="B", mesh_order=2),
            InterfaceTag(
                linestrings=LineString([(5, 0), (5, 5)]),
                zmin=0.0, zmax=1.0, physical_name="iface", mesh_order=3,
            ),
        ]

    m_gmsh, m_occ = _run_both(make, tmp_path)
    s_gmsh = _mesh_summary(m_gmsh)
    s_occ = _mesh_summary(m_occ)
    assert "iface" in s_gmsh and "iface" in s_occ
    _assert_summaries_equivalent(s_gmsh, s_occ)
```

Run + commit message: `test(cross_compare): polyprism with interface tag`

---

## Task 7: Keep-false helper

```python
def test_keep_false_helper_match(tmp_path):
    """A helper prism (mesh_bool=False) carves an interface in a kept
    neighbour. Both backends must omit the helper from the mesh but
    still tag the kept___helper interface."""
    def make():
        A = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        B = shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)])
        buffers = {0.0: 0.0, 1.0: 0.0}
        return [
            PolyPrism(polygons=A, buffers=buffers, physical_name="A", mesh_order=1),
            PolyPrism(
                polygons=B, buffers=buffers,
                physical_name="helper", mesh_order=2, mesh_bool=False,
            ),
        ]

    m_gmsh, m_occ = _run_both(make, tmp_path)
    s_gmsh = _mesh_summary(m_gmsh)
    s_occ = _mesh_summary(m_occ)
    # helper has no body in the mesh on either side
    assert "helper" not in s_gmsh or "tetra" not in s_gmsh.get("helper", {})
    assert "helper" not in s_occ or "tetra" not in s_occ.get("helper", {})
    # Compare ignoring the helper itself; A___helper interface must match.
    _assert_summaries_equivalent(s_gmsh, s_occ, ignore_groups={"helper"})
```

Run + commit message: `test(cross_compare): keep_false helper across backends`

---

## Task 8: Add a parametrized smoke test using the `cad_pipeline` fixture

The point: prove the fixture works for users who want to write tests once and have them run on both backends.

```python
import pytest


@pytest.mark.parametrize(
    "scene_factory",
    [
        pytest.param(
            lambda: [
                PolyPrism(
                    polygons=shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),
                    buffers={0.0: 0.0, 2.0: 0.0},
                    physical_name="A", mesh_order=1,
                ),
            ],
            id="single_prism",
        ),
        pytest.param(
            lambda: [
                PolySurface(
                    polygons=shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    physical_name="A", mesh_order=1,
                ),
            ],
            id="single_polysurface_2d",
        ),
    ],
)
def test_pipeline_runs_on_both_backends(cad_pipeline, scene_factory):
    """Demonstrate the `cad_pipeline` fixture: pytest runs this 2x2 = 4
    times, once per (scene, backend) combination. Each run produces a
    valid mesh with the expected physical group."""
    entities = scene_factory()
    dim = 2 if any(type(e).__name__ == "PolySurface" for e in entities) else 3
    m = cad_pipeline(entities, dim=dim)
    assert "A" in m.cell_sets_dict
```

Run:
```bash
pytest tests/test_backend_cross_compare.py::test_pipeline_runs_on_both_backends -v --no-cov
```
Expected: 4 PASS (2 scenes × 2 backends).

Commit:
```bash
git add tests/test_backend_cross_compare.py
git commit -m "test(cross_compare): demonstrate cad_pipeline parametrized fixture"
```

---

## Task 9: Final regression sweep

```bash
pytest -m "not slow" --no-cov 2>&1 | tail -10
```

Expected: previous baseline (127 passed, 2 known failures) plus the new tests
(should be ~134 passed, 2 known failures unchanged).

If anything previously passing now fails, STOP and report.

## Out-of-scope follow-ups (not for this plan)

- Refactoring the existing `test_interface_tag.py` e2e tests onto `cad_pipeline`.
- Backporting the cross-compare pattern to `test_cad_gmsh.py` and `test_cad_occ.py`.
- A `pytest.ini` marker (`@pytest.mark.cross_compare`) for filtering.
