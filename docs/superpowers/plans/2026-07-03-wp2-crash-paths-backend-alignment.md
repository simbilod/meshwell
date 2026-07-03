# WP2 — Crash Paths + Backend Alignment: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the crash-path bugs and align the gmsh CAD backend's tie-cut and same-material-interface policies to the OCC backend (spec: `docs/superpowers/specs/2026-07-03-code-improvements-design.md`, WP2; user confirmed: skip `A___A` groups in both backends).

**Architecture:** Seven independent fixes on the `code_improvement` branch, TDD, one commit per task. Task 6 (backend alignment) is the only behavior-visible change and may require golden-reference regeneration.

**Tech Stack:** Python 3.13, pytest, gmsh, OCP, shapely, meshio.

## Global Constraints

- Test baseline at start of WP2: **375 passed / 7 skipped, fully green** (commit `ea4e21e`). Any new failure is a regression; each task updates this count by its own new tests.
- Run tests from repo root `/home/simbil/Github/meshwell_structured_manual/meshwell` with `uv run pytest tests ...` (tests live in `tests/`, NOT `meshwell/tests/`).
- Pre-commit hooks run black/ruff/codespell — fix and re-commit if rejected; never `--no-verify`.
- **Model dispatch:** each task header carries a `Model:` line — dispatch the implementer with that model.
- Do not fix unrelated issues; WP3–WP7 cover them.

---

### Task 1: `resolution_specs` default `None`, normalized to `{}`

**Model:** haiku

**Files:**
- Modify: `meshwell/mesh.py:472` (in `process_geometry` signature)
- Test: `tests/test_removed_apis.py` (append — it already collects API-contract tests)

**Interfaces:**
- Produces: `Mesh.process_geometry(resolution_specs=None)` works; the tuple default that crashed `.get()` calls is gone.

- [ ] **Step 1: Write the failing test** (append to `tests/test_removed_apis.py`)

```python
def test_resolution_specs_default_is_none():
    import inspect

    from meshwell.mesh import Mesh

    default = inspect.signature(Mesh.process_geometry).parameters[
        "resolution_specs"
    ].default
    assert default is None, "tuple default () crashes .get() calls downstream"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_removed_apis.py -v -k resolution_specs`
Expected: FAIL (default is `()`).

- [ ] **Step 3: Implement**

In `meshwell/mesh.py`, change the `process_geometry` parameter `resolution_specs: dict = ()` to `resolution_specs: dict | None = None`, and at the top of the method body (right after `self._initialize_model()`) add:

```python
        if resolution_specs is None:
            resolution_specs = {}
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_removed_apis.py -v && uv run pytest tests -q`
Expected: PASS; no regressions (376 passed / 7 skipped).

- [ ] **Step 5: Commit**

```bash
git add meshwell/mesh.py tests/test_removed_apis.py
git commit -m "fix(mesh): resolution_specs default None instead of tuple

A () default reached resolution_specs.get() and raised AttributeError."
```

---

### Task 2: Normalize `PolyPrism.polygons` to a flat list + clear empty-buffers error

**Model:** sonnet

**Files:**
- Modify: `meshwell/polyprism.py:72-89` (`__init__` polygon handling + buffers validation), `:150-170` (`_get_buffered_polygons`), `:612-650` (`to_dict` — collapse to list-only branch), `:652-685` (`from_dict` — pass flat list)
- Test: `tests/test_polyprism_inputs.py` (create)

**Interfaces:**
- Produces: `self.polygons` is ALWAYS `list[Polygon]` after `__init__` (MultiPolygons exploded, single Polygon wrapped); `PolyPrism(buffers={})` raises `ValueError` mentioning "buffers".

- [ ] **Step 1: Write the failing test**

```python
"""PolyPrism input normalization."""
import pytest
from shapely.geometry import MultiPolygon, Polygon

from meshwell.polyprism import PolyPrism

_SQ1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
_SQ2 = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])


def test_polygons_normalized_to_flat_list():
    for inp in [_SQ1, [_SQ1, _SQ2], MultiPolygon([_SQ1, _SQ2]), [MultiPolygon([_SQ1, _SQ2])]]:
        prism = PolyPrism(polygons=inp, buffers={0.0: 0.0, 1.0: 0.0}, physical_name="x")
        assert isinstance(prism.polygons, list)
        assert all(isinstance(p, Polygon) for p in prism.polygons)


def test_list_input_with_nonzero_buffers_builds_buffered_polygons():
    # crashed before: list has no .geoms, so .buffer() was called on a list
    prism = PolyPrism(polygons=[_SQ1, _SQ2], buffers={0.0: 0.0, 1.0: 0.1},
                      physical_name="x")
    assert len(prism.buffered_polygons) == 2
    assert all(len(entry) == 2 for entry in prism.buffered_polygons)


def test_serialization_round_trip_two_polygons():
    prism = PolyPrism(polygons=[_SQ1, _SQ2], buffers={0.0: 0.0, 1.0: 0.1},
                      physical_name="x")
    clone = PolyPrism.from_dict(prism.to_dict())
    assert len(clone.polygons) == 2
    assert clone.to_dict() == prism.to_dict()


def test_empty_buffers_raises_value_error():
    with pytest.raises(ValueError, match="buffers"):
        PolyPrism(polygons=_SQ1, buffers={}, physical_name="x")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_polyprism_inputs.py -v`
Expected: flat-list, nonzero-buffers-list, and empty-buffers tests FAIL (`AttributeError`/`ValueError: min() arg is an empty sequence`); round-trip may fail on `MultiPolygon` reconstruction.

- [ ] **Step 3: Implement**

1. In `__init__` (before the `point_tolerance > 0` snap), normalize exactly the way `PolySurface.__init__` does (`meshwell/polysurface.py:57-67`):

```python
        # Normalize to a flat list of Polygons (MultiPolygons exploded).
        if isinstance(polygons, (Polygon, MultiPolygon)):
            polygons = list(polygons.geoms if hasattr(polygons, "geoms") else [polygons])
        else:
            flat: list[Polygon] = []
            for entry in polygons:
                flat.extend(entry.geoms if hasattr(entry, "geoms") else [entry])
            polygons = flat
```

then keep the existing snap logic but only the list branch (the non-list branch is now dead — delete it).

2. Validate buffers right before the `all(buffer == 0 ...)` check:

```python
        if not buffers:
            raise ValueError(
                "buffers must contain at least one {z: buffer} entry; got an empty dict."
            )
```

3. `_get_buffered_polygons`: iterate the flat list directly — replace `for polygon in polygons.geoms if hasattr(polygons, "geoms") else [polygons]:` with `for polygon in polygons:`.

4. `to_dict`: `self.polygons` is now always a list — keep only the list branch, delete the `MultiPolygon` and scalar branches.

5. `from_dict`: replace the `MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]` re-packing with passing the flat list straight through (`polygons=polygons`).

6. Grep for other consumers of `.polygons` on PolyPrism (`grep -n "\.polygons" meshwell/*.py meshwell/structured/*.py`) and confirm each handles a flat list (PolySurface already guarantees one, so shared helpers do). Name what you checked in your report; if any consumer genuinely requires the old scalar/MultiPolygon form, STOP and report BLOCKED.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_polyprism_inputs.py tests/test_prism.py tests/test_buffers_prism.py -v && uv run pytest tests -q`
Expected: PASS; no regressions.

- [ ] **Step 5: Commit**

```bash
git add meshwell/polyprism.py tests/test_polyprism_inputs.py
git commit -m "fix(polyprism): normalize polygons to flat list; reject empty buffers

list[Polygon] input crashed _get_buffered_polygons (buffer() called on
the list); from_dict re-packed lists into MultiPolygon inconsistently;
empty buffers raised a bare min() ValueError."
```

---

### Task 3: PolyLine zero-wire filter + dead serialization branches

**Model:** sonnet

**Files:**
- Modify: `meshwell/polyline.py:150-164` (`instanciate`), `:194-226` (`to_dict`), `meshwell/polysurface.py:208-243` (`to_dict`)
- Test: `tests/test_polyline.py` (append)

**Interfaces:**
- Produces: `PolyLine.instanciate` never returns `(1, 0)` dimtags; `to_dict` in polyline/polysurface has only the list branch (both classes normalize in `__init__`).

- [ ] **Step 1: Write the failing test** (append to `tests/test_polyline.py`)

```python
def test_instanciate_filters_zero_wire_tags(monkeypatch):
    """A degenerate linestring must not leak a (1, 0) dimtag."""
    import gmsh
    from shapely.geometry import LineString

    from meshwell.polyline import PolyLine

    pl = PolyLine(linestrings=[LineString([(0, 0), (1, 1)])], physical_name="pl")
    monkeypatch.setattr(pl, "_create_wire_from_linestring", lambda ls: 0)
    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.model.add("zero_wire_test")
    try:
        dimtags = pl.instanciate()
        assert (1, 0) not in dimtags
        assert dimtags == []
    finally:
        gmsh.model.remove()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_polyline.py -v -k zero_wire`
Expected: FAIL — `(1, 0)` present.

- [ ] **Step 3: Implement**

1. In `instanciate` (`polyline.py:155-158`), filter like PolySurface does:

```python
        wires = []
        for linestring in self.linestrings:
            wire_id = self._create_wire_from_linestring(linestring)
            if wire_id != 0:
                wires.append(wire_id)
```

2. First VERIFY `PolyLine.__init__` normalizes `linestrings` to a flat list (check the code around `polyline.py:60-90`); if it does, delete the `MultiLineString` and scalar branches in `to_dict` (keep the list branch). Do the same for `PolySurface.to_dict` (its `__init__` provably normalizes at `polysurface.py:57-67`). If PolyLine does NOT normalize, leave its `to_dict` alone and say so in your report — do not add normalization (that is WP3's dedup work).

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_polyline.py tests/test_polysurface.py -v && uv run pytest tests -q`
Expected: PASS; no regressions.

- [ ] **Step 5: Commit**

```bash
git add meshwell/polyline.py meshwell/polysurface.py tests/test_polyline.py
git commit -m "fix(polyline): drop zero wire tags from instanciate dimtags

Also delete dead non-list to_dict branches (inputs are normalized to
lists in __init__)."
```

---

### Task 4: Guard empty hole-cut result in PolySurface

**Model:** sonnet

**Files:**
- Modify: `meshwell/polysurface.py:87-128` (`_create_surface_with_holes`)
- Test: `tests/test_polysurface.py` (append)

**Interfaces:**
- Produces: a hole cut that annihilates the exterior returns 0 (skipped surface) with a `UserWarning`, instead of `IndexError`; a zero exterior short-circuits to 0 before any cutting.

- [ ] **Step 1: Write the failing test** (append to `tests/test_polysurface.py`)

```python
def test_hole_covering_exterior_warns_and_skips():
    """A hole >= the exterior annihilates the surface; must warn, not IndexError."""
    import gmsh
    import pytest
    from shapely.geometry import Polygon

    from meshwell.polysurface import PolySurface

    # shell == hole ring: zero-area surface; gmsh cut returns empty outDimTags
    degenerate = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        holes=[[(0, 0), (10, 0), (10, 10), (0, 10)]],
    )
    ps = PolySurface(polygons=degenerate, physical_name="void")
    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.model.add("hole_annihilation_test")
    try:
        with pytest.warns(UserWarning, match="void"):
            dimtags = ps.instanciate()
        assert (2, 0) not in dimtags
    finally:
        gmsh.model.remove()
```

Note: shapely may normalize/reject the fully-degenerate ring; if `set_precision` collapses it before gmsh is reached, use a hole slightly LARGER than needed to cover the exterior after the snap (e.g. shell 10×10, hole -1..11) — the assertion contract stays the same. If gmsh's cut raises instead of returning empty for your geometry, catch that variant too (see Step 3).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_polysurface.py -v -k annihilation`
Expected: FAIL with `IndexError` (or an unraised-warning failure).

- [ ] **Step 3: Implement**

In `_create_surface_with_holes`:

```python
        # A degenerate exterior cannot host holes.
        if exterior == 0:
            return 0

        # Cut holes from exterior surface
        for interior_surface in interior_surfaces:
            cut_result = gmsh.model.occ.cut(
                [(2, exterior)],
                [(2, interior_surface)],
                removeObject=True,
                removeTool=True,
            )
            gmsh.model.occ.synchronize()
            if not cut_result[0]:
                warnings.warn(
                    f"Hole cut annihilated surface for PolySurface "
                    f"{self.physical_name}; this surface is DROPPED.",
                    stacklevel=2,
                )
                self._clear_caches()
                return 0
            exterior = cut_result[0][0][1]
            self._clear_caches()

        return exterior
```

Add `import warnings` to the module imports. (Mirror of `polyprism.py`'s guarded hole handling; keep the existing comment lines where they still apply.)

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_polysurface.py -v && uv run pytest tests -q`
Expected: PASS; no regressions.

- [ ] **Step 5: Commit**

```bash
git add meshwell/polysurface.py tests/test_polysurface.py
git commit -m "fix(polysurface): guard hole cuts that annihilate the exterior

Empty cut results raised IndexError; zero exteriors reached the cut."
```

---

### Task 5: Visualization guards (line-only meshes, unknown group ids)

**Model:** sonnet

**Files:**
- Modify: `meshwell/visualization.py:146-230` (`plot2D` triangle and line loops)
- Test: `tests/test_visualization_guards.py` (create)

**Interfaces:**
- Produces: `plot2D` renders line-only meshes (no `KeyError: 'triangle'`), triangle-only meshes with `ignore_lines=False` (no `KeyError: 'line'`), and meshes whose physical ids are missing from `field_data`.

- [ ] **Step 1: Write the failing test**

```python
"""plot2D must not crash on meshes missing a cell block or field_data entry."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import meshio
import numpy as np

from meshwell.visualization import plot2D


def _line_only_mesh():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    return meshio.Mesh(points, [("line", np.array([[0, 1], [1, 2]]))])


def _triangle_only_mesh():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return meshio.Mesh(points, [("triangle", np.array([[0, 1, 2]]))])


def test_plot2d_line_only_mesh_does_not_crash():
    plot2D(_line_only_mesh())
    plt.close("all")


def test_plot2d_triangle_only_mesh_with_lines_enabled():
    plot2D(_triangle_only_mesh(), ignore_lines=False)
    plt.close("all")


def test_plot2d_physicals_filter_with_missing_field_data():
    m = _triangle_only_mesh()
    m.cell_data = {"gmsh:physical": [np.array([7])]}  # id 7 absent from field_data
    plot2D(m, physicals=["anything"])
    plt.close("all")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_visualization_guards.py -v`
Expected: line-only FAILS with `KeyError: 'triangle'`; the physicals-filter test FAILS with `KeyError: 7`.

- [ ] **Step 3: Implement**

In `plot2D`:

1. Guard the whole triangle loop: wrap the `for i, group in enumerate(physical_groups_2D):` block in `if "triangle" in mesh.cells_dict:` (line-only meshes skip it entirely); symmetrically guard the line loop with `if "line" in mesh.cells_dict:` inside the existing `if not ignore_lines:`.
2. Replace both unguarded `id_to_name[group]` lookups in the physicals-filter conditions (visualization.py:152 and :204) with `id_to_name.get(group, [])`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_visualization_guards.py -v && uv run pytest tests -q`
Expected: PASS; no regressions.

- [ ] **Step 5: Commit**

```bash
git add meshwell/visualization.py tests/test_visualization_guards.py
git commit -m "fix(visualization): guard plot2D against missing cell blocks and ids"
```

---

### Task 6: Align gmsh backend — skip tie cuts, stop emitting `A___A` groups

**Model:** opus

**Files:**
- Modify: `meshwell/cad_gmsh.py:429-455` (cut-tool selection), `:329-336` (same-material pair emission)
- Test: `tests/test_backend_tie_policy.py` (create)

**Interfaces:**
- Produces: gmsh backend matches OCC policy — (a) mesh-order ties are NOT cut pre-fragment (fragment resolves ownership; earlier-declared entity wins the overlap); (b) no `A___A` interface physical group between identically-named neighbors (`same_material_interfaces` is still tracked for exterior-boundary subtraction at cad_gmsh.py:357).

**Background:** In OCC (`cad_occ.py:479`), `p_ord >= l_ord` skips the pre-fragment cut and `_fragment_all` awards the shared piece to the first candidate in insertion order — final ownership identical, fewer boolean ops, convergent topology. The user confirmed both policies (2026-07-03).

- [ ] **Step 1: Write the failing test**

```python
"""gmsh backend must match OCC: no tie cuts, no A___A groups."""
import gmsh
from shapely.geometry import Polygon

from meshwell.cad_gmsh import CAD_GMSH
from meshwell.polysurface import PolySurface


def _overlapping_pair(mesh_order_b):
    a = PolySurface(polygons=Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                    physical_name="a", mesh_order=1)
    b = PolySurface(polygons=Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                    physical_name="b", mesh_order=mesh_order_b)
    return [a, b]


def _run(entities):
    proc = CAD_GMSH()
    try:
        labeled = proc.process_entities(entities)
        groups = {
            gmsh.model.getPhysicalName(dim, tag)
            for dim, tag in gmsh.model.getPhysicalGroups()
        }
        owner_areas = {
            tuple(ent.physical_name): sum(
                gmsh.model.occ.getMass(dim, tag) for dim, tag in ent.dimtags
            )
            for ent in labeled
            if ent.dimtags
        }
        return groups, owner_areas
    finally:
        proc.model_manager.finalize()


def test_tie_overlap_earlier_entity_wins():
    groups, areas = _run(_overlapping_pair(mesh_order_b=1))  # tie
    # earlier-declared entity keeps the 1x1 overlap: a=4.0, b=4.0-1.0=3.0
    assert abs(areas[("a",)] - 4.0) < 1e-6
    assert abs(areas[("b",)] - 3.0) < 1e-6


def test_tie_and_nontie_ownership_agree():
    _, tie_areas = _run(_overlapping_pair(mesh_order_b=1))
    _, cut_areas = _run(_overlapping_pair(mesh_order_b=2))
    for k in tie_areas:
        assert abs(tie_areas[k] - cut_areas[k]) < 1e-6


def test_no_same_material_interface_group():
    a = PolySurface(polygons=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    physical_name="m", mesh_order=1)
    b = PolySurface(polygons=Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                    physical_name="m", mesh_order=1)
    groups, _ = _run([a, b])
    assert "m___m" not in groups
    # the shared edge must also not appear in m's exterior boundary
    assert "m___None" in groups
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_backend_tie_policy.py -v`
Expected: `test_no_same_material_interface_group` FAILS (`m___m` emitted). The tie-ownership tests may already pass (fragment resolution gives the same final areas) — that is fine and expected; they pin the invariant. Verify at least one test is RED before implementing. Confirm the exterior-boundary assertion (`m___None` present) matches actual group naming before relying on it; adjust the expected name to the repo's convention (`interface_delimiter="___"`, `boundary_delimiter="None"`) if needed.

- [ ] **Step 3: Implement**

1. **Tie skip** — in `process_entities`' cut loop (cad_gmsh.py:434-439), select only strictly-lower-mesh_order tools, mirroring `cad_occ.py:472-480`:

```python
            # Cut only against previously instantiated same-dim entities of
            # STRICTLY lower mesh_order — ties are not cut (matches cad_occ;
            # the final fragment resolves tie ownership by insertion order).
            if labeled_ent.dimtags:
                l_ord = (
                    labeled_ent.mesh_order
                    if labeled_ent.mesh_order is not None
                    else float("inf")
                )
                all_tool_dimtags = []
                for prev_ent in instantiated_entities:
                    if prev_ent.dim != labeled_ent.dim or not prev_ent.dimtags:
                        continue
                    p_ord = (
                        prev_ent.mesh_order
                        if prev_ent.mesh_order is not None
                        else float("inf")
                    )
                    if p_ord >= l_ord:
                        continue
                    all_tool_dimtags.extend(prev_ent.dimtags)
```

(`GMSHLabeledEntity.mesh_order` exists — dataclass field at cad_gmsh.py:61, populated at :166.)

2. **A___A suppression** — in the pair-interface loop (cad_gmsh.py:329-336), keep tracking `same_material_interfaces` (needed at :357) but stop emitting the group:

```python
                for ni in ei.physical_name:
                    for nj in ej.physical_name:
                        if ni == nj:
                            # Same-material contact: not a physical interface.
                            # Track for exterior-boundary subtraction but emit
                            # no A___A group (matches the OCC XAO writer).
                            same_material_interfaces.update(shared)
                            continue
                        key = tuple(sorted((ni, nj)))
                        pair_dimtags[key].update(shared)
```

3. **Module docstring** — update `cad_gmsh.py`'s header lines that describe cut semantics (around lines 1-27) to state: ties are not cut; same-material interfaces are not emitted; ownership matches `cad_occ`.

4. **Reference meshes:** run the full suite. Failures in `tests/test_backend_*`, `tests/test_cad*`, or golden-reference comparisons must be inspected individually: a diff caused by tie-overlap scenes (topology where two same-order entities overlap) or by disappeared `A___A` groups is the intended change — regenerate only those references via `uv run python tests/generate_references.py` (read how it selects cases first) and list each regenerated file with its cause in the commit message. Any other failure: STOP, report BLOCKED with the failing test output.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_backend_tie_policy.py tests/test_cad_gmsh.py tests/test_backend_equivalence.py tests/test_backend_cross_compare.py -v` then `uv run pytest tests -q`
Expected: PASS (after any justified reference regeneration); no other regressions.

- [ ] **Step 5: Commit**

```bash
git add meshwell/cad_gmsh.py tests/test_backend_tie_policy.py
git commit -m "feat!(cad_gmsh): align tie-cut and A___A policy with OCC backend

Mesh-order ties are no longer cut pre-fragment (fragment resolves
ownership identically; earlier entity wins overlaps), and same-material
A___A interface groups are no longer emitted (matches the OCC XAO
writer). User-confirmed policy decisions, 2026-07-03."
```

---

### Task 7: `prepared` flag symmetry + contract docstrings

**Model:** sonnet

**Files:**
- Modify: `meshwell/cad_gmsh.py:390-417` (`process_entities` signature + prepare call), `meshwell/cad_occ.py:1-28` (module docstring contract wording)
- Test: `tests/test_cad_gmsh.py` (append)

**Interfaces:**
- Produces: `CAD_GMSH.process_entities(..., prepared: bool = False)` — when True, `prepare_entities` is skipped (caller already buffered), mirroring `CAD_OCC.process_entities(..., prepared)` at `cad_occ.py:559`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_cad_gmsh.py`)

```python
def test_prepared_flag_skips_double_buffering(monkeypatch):
    """prepared=True must not re-run prepare_entities (double buffer compounds)."""
    from shapely.geometry import Polygon

    import meshwell.cad_gmsh as cad_gmsh_mod
    from meshwell.cad_gmsh import CAD_GMSH
    from meshwell.polysurface import PolySurface

    calls = []
    real = cad_gmsh_mod.prepare_entities
    monkeypatch.setattr(
        cad_gmsh_mod, "prepare_entities",
        lambda *a, **k: (calls.append(1), real(*a, **k))[1],
    )
    ent = [PolySurface(polygons=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                       physical_name="a")]
    proc = CAD_GMSH()
    try:
        proc.process_entities(ent, prepared=True)
    finally:
        proc.model_manager.finalize()
    assert calls == [], "prepare_entities ran despite prepared=True"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cad_gmsh.py -v -k prepared`
Expected: FAIL — `TypeError: unexpected keyword argument 'prepared'`.

- [ ] **Step 3: Implement**

1. Add `prepared: bool = False` to `CAD_GMSH.process_entities` (after `boundary_delimiter`), document it in the docstring the way `cad_occ.py:559+` does, and guard the pre-pass:

```python
        if not prepared:
            prepare_entities(
                entities_list,
                perturbation=self.perturbation,
                resolve_snap=max(self.perturbation, self.point_tolerance),
            )
```

2. In `cad_occ.py`'s module docstring (line 27 area), revise "Ownership semantics match :func:`meshwell.cad_gmsh._resolve_piece_ownership` exactly" to also state the now-shared tie/`A___A` policy (one or two sentences; keep it accurate to what Task 6 implemented).

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_cad_gmsh.py -v && uv run pytest tests -q`
Expected: PASS; no regressions.

- [ ] **Step 5: Commit**

```bash
git add meshwell/cad_gmsh.py meshwell/cad_occ.py tests/test_cad_gmsh.py
git commit -m "feat(cad_gmsh): prepared flag to skip double entity preparation

Mirrors CAD_OCC's prepared parameter; also documents the now-aligned
backend contract."
```

---

## Final verification (orchestrator, after Task 7)

- [ ] `uv run pytest tests -q` — fully green (baseline 375 + new tests).
- [ ] One commit per task; final whole-branch review of the WP2 range before starting WP3.
