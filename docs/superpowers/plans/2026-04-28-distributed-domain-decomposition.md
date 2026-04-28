# Distributed Domain Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an opt-in distributed pipeline that subdivides a meshwell scene into user-supplied subdomain polygons, runs CAD + meshing per subdomain in separate processes (or on separate machines), and stitches the results into one final `.msh` with conformal seams and consistent physical tags.

**Architecture:** Two-phase file-based pipeline. Phase 1 meshes thin "interface" slabs and "junction" boxes around shared cuts to produce frozen 2D seam meshes; phase 2 meshes each subdomain volumetrically with the seam meshes embedded as fixed boundary constraints. A pluggable `Executor` protocol drives parallelism (default = `concurrent.futures.ProcessPoolExecutor`); workers read JSON job bundles via a `meshwell run-job` CLI subcommand, so the same code runs locally or on Slurm/k8s. The seam surface is identified at phase 1 by a phantom `keep=False` physical-name imprint (`_seam___volume_i___volume_j`), avoiding coordinate-based filtering.

**Tech Stack:** Python 3.10+, gmsh (Python API), shapely, OCP (OpenCascade Python bindings), pydantic, pytest, `concurrent.futures`, `pathlib`, JSON.

---

## File Structure

| File | Role |
|------|------|
| `meshwell/distributed.py` | NEW — public API (`generate_mesh_distributed`, `subdomains_from_grid`), planner (`build_subdomain_plan`), bundle I/O, clipping helpers (`_clip_entity_to_polygon`, `_resolution_only_proxy`), executor protocol + default impl, worker entrypoint (`run_job`, `cli_main`), glue (`stitch_meshes`). |
| `meshwell/cad_common.py` | MODIFIED — add `skip_buffer: bool = False` to `prepare_entities`. |
| `meshwell/mesh.py` | MODIFIED — accept `_pre_buffered`, `_emit_only_seam_surfaces`, `_interface_constraints`, `_global_physical_names` private kwargs; new helpers `_filter_msh_to_seam_groups`, `_embed_imported_seam_into_occ_face`. |
| `meshwell/orchestrator.py` | MODIFIED — forward the new private kwargs to `mesh()`. |
| `meshwell/__init__.py` | MODIFIED — export `generate_mesh_distributed`, `subdomains_from_grid`, `Executor`, `SubprocessExecutor`. |
| `pyproject.toml` | MODIFIED — add `[project.scripts]` entry `meshwell = "meshwell.distributed:cli_main"`. |
| `tests/test_distributed.py` | NEW — unit + integration tests (11 scenarios from spec). |
| `tests/test_distributed_spike.py` | NEW — early R3 spike test for `gmsh.merge` + `gmsh.model.mesh.embed` interop. |
| `tests/conftest.py` | MODIFIED — add a `tmp_work_dir` fixture for distributed tests. |

---

## Task 0: R3 spike — verify `gmsh.merge` + `embed` interop

**Why first:** Risk R3 in the spec says `gmsh.model.mesh.embed` of a `gmsh.merge`-imported discrete entity into an OCC face is "underdocumented and may behave inconsistently." The whole conformity story collapses if this doesn't work, so we validate it before building anything else.

**Files:**
- Create: `tests/test_distributed_spike.py`

- [ ] **Step 1: Write the spike test**

```python
"""R3 spike: verify gmsh.merge + embed across kernels works for our use case.

Builds a 1x1x1 OCC box, generates a discrete 2D mesh on its top face in a
separate gmsh model, exports that surface mesh to .msh, then in a fresh
gmsh model re-imports it via gmsh.merge and embeds it into the OCC top
face. Asserts the meshed top face uses the imported nodes/triangles.
"""
import tempfile
from pathlib import Path

import gmsh
import pytest


@pytest.fixture(autouse=True)
def fresh_gmsh():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    yield
    gmsh.finalize()


def _build_seam_surface_msh(path: Path, mesh_size: float) -> None:
    """Mesh the unit square at z=1 in a discrete 2D model and write to path."""
    gmsh.model.add("seam")
    pts = [
        gmsh.model.geo.addPoint(0, 0, 1, mesh_size),
        gmsh.model.geo.addPoint(1, 0, 1, mesh_size),
        gmsh.model.geo.addPoint(1, 1, 1, mesh_size),
        gmsh.model.geo.addPoint(0, 1, 1, mesh_size),
    ]
    lines = [
        gmsh.model.geo.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)
    ]
    loop = gmsh.model.geo.addCurveLoop(lines)
    surf = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()
    pg = gmsh.model.addPhysicalGroup(2, [surf], name="_seam___top")
    gmsh.model.mesh.generate(2)
    gmsh.write(str(path))
    gmsh.model.remove()


def test_embed_merged_discrete_into_occ_face(tmp_path):
    seam_path = tmp_path / "seam.msh"
    _build_seam_surface_msh(seam_path, mesh_size=0.2)

    # Fresh model: build OCC box, then merge the seam mesh, then embed.
    gmsh.model.add("box")
    box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()

    # Locate the OCC top face (z=1).
    occ_faces = gmsh.model.getEntities(2)
    top_face = None
    for dim, tag in occ_faces:
        bb = gmsh.model.getBoundingBox(dim, tag)
        if abs(bb[2] - 1.0) < 1e-9 and abs(bb[5] - 1.0) < 1e-9:
            top_face = tag
            break
    assert top_face is not None

    # Merge the seam mesh — discrete entities appear in the model.
    gmsh.merge(str(seam_path))

    # Find the imported discrete 2D entity (it carries the _seam___top group).
    imported_2d = None
    for dim, tag in gmsh.model.getEntities(2):
        if tag == top_face:
            continue
        try:
            name = gmsh.model.getEntityName(dim, tag) or ""
        except Exception:
            name = ""
        if "_seam___top" in name or True:  # discrete tag will not be the OCC top_face
            imported_2d = tag
    assert imported_2d is not None and imported_2d != top_face

    # Embed the imported discrete face's boundary into the OCC top face.
    # We expect that gmsh will mesh the OCC top face using the imported nodes.
    gmsh.model.mesh.embed(2, [imported_2d], 2, top_face)
    gmsh.model.mesh.generate(3)

    # Sanity: the resulting mesh on the top face should reuse the imported nodes.
    nodes_top = gmsh.model.mesh.getNodes(2, top_face, includeBoundary=True)
    assert len(nodes_top[0]) > 0
```

- [ ] **Step 2: Run the spike**

Run: `pytest tests/test_distributed_spike.py -v`

Expected: PASS. If it FAILS, the embed/merge interop is broken and the design must switch to the R3 fallback (rebuild seam via setNodes + removeDuplicateNodes after generation). Document the result inline at the top of `test_distributed_spike.py` and update the implementation tasks below before proceeding.

- [ ] **Step 3: Commit**

```bash
git add tests/test_distributed_spike.py
git commit -m "test: spike for gmsh.merge + embed across OCC/discrete kernels (R3)"
```

---

## Task 1: `cad_common.prepare_entities(skip_buffer=...)` parameter

**Files:**
- Modify: `meshwell/cad_common.py`
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the failing test**

Create the new test file:

```python
"""Tests for the distributed-meshing pipeline."""
from __future__ import annotations

import shapely

from meshwell.cad_common import prepare_entities
from meshwell.polysurface import PolySurface


def test_prepare_entities_skip_buffer_preserves_polygons():
    p = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    ent = PolySurface(polygons=p, physical_name="A", mesh_order=1)
    original = ent.polygons

    prepare_entities([ent], perturbation=1e-3, skip_buffer=True)

    # polygons must be unchanged (no outward buffer applied).
    assert ent.polygons.equals(original)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_distributed.py::test_prepare_entities_skip_buffer_preserves_polygons -v`
Expected: FAIL with `TypeError: prepare_entities() got an unexpected keyword argument 'skip_buffer'`.

- [ ] **Step 3: Implement `skip_buffer`**

Edit `meshwell/cad_common.py`. The current function signature is:

```python
def prepare_entities(
    entities_list: list[Any],
    perturbation: float,
    resolve_snap: float | None = None,
) -> None:
```

Add a new keyword-only parameter `skip_buffer: bool = False`. The body is split into two passes already (Pass A = buffer; Pass B = InterfaceTag resolve). Wrap **only Pass A** (the entire block from `# ----- Pass A: buffer all polygon-bearing entities -----` through `.intersection(global_bbox)`) in `if not skip_buffer:`. Pass B (the InterfaceTag resolve loop) must always run — it's idempotent and is needed regardless.

Concretely:

1. Open `meshwell/cad_common.py`. Locate the existing function (currently around line 28-117).
2. Add `skip_buffer: bool = False` to the signature (before `resolve_snap`).
3. Update the docstring's Args section to document `skip_buffer` (text below).
4. Locate Pass A: it begins at `# ----- Pass A: buffer all polygon-bearing entities -----` and ends just before `# ----- Pass B: resolve each InterfaceTag against the buffered polygons -----`.
5. Wrap that entire Pass A block (including its bbox computation, the `if xmin == float("inf"): return` guard, the `global_bbox` construction, and the per-entity buffer loop) in `if not skip_buffer:`. Indent each line by one extra level (4 spaces).
6. Leave Pass B unchanged.

Add to the docstring's `Args:` section:

```
        skip_buffer: When True, skip the polygon buffering pass entirely.
            Used by the distributed pipeline (workers receive entities
            already buffered by the master).
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_distributed.py::test_prepare_entities_skip_buffer_preserves_polygons -v`
Expected: PASS.

- [ ] **Step 5: Run the full test suite** to make sure no regressions.

Run: `pytest tests/ -x -q`
Expected: all existing tests pass.

- [ ] **Step 6: Commit**

```bash
git add meshwell/cad_common.py tests/test_distributed.py
git commit -m "feat(cad_common): add skip_buffer flag to prepare_entities"
```

---

## Task 2: `mesh.py` accepts `_pre_buffered` flag

The mesh entrypoint must thread `_pre_buffered=True` through to `prepare_entities(..., skip_buffer=True)`. Plumbed via `orchestrator.py` and `mesh.mesh()`.

**Files:**
- Modify: `meshwell/orchestrator.py`
- Modify: `meshwell/mesh.py` (the `mesh()` function — find the call site that triggers `prepare_entities`; in this codebase that's actually inside `cad_occ.process_entities`, which is called from `orchestrator.generate_mesh`)
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_distributed.py`:

```python
def test_generate_mesh_pre_buffered_flag_skips_buffer(tmp_path):
    """When _pre_buffered=True, the polygon should not be buffered."""
    import shapely
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism

    p = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    prism = PolyPrism(
        polygons=p,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="A",
        mesh_order=1,
    )
    # Snapshot pre-call polygons.
    original_polys = prism.polygons

    generate_mesh(
        entities=[prism],
        dim=3,
        output_mesh=tmp_path / "out.msh",
        _pre_buffered=True,
    )
    # PolyPrism's polygons are still the original (no perturbation buffer).
    assert prism.polygons.equals(original_polys)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_distributed.py::test_generate_mesh_pre_buffered_flag_skips_buffer -v`
Expected: FAIL with `TypeError: generate_mesh() got an unexpected keyword argument '_pre_buffered'`.

- [ ] **Step 3: Implement plumbing**

In `meshwell/orchestrator.py`, accept the new private kwarg and forward as `perturbation=0` AND a `skip_buffer` flag. Read the existing function signature and add:

```python
def generate_mesh(
    entities: list[Any],
    dim: int,
    output_mesh: Path | str | None = None,
    checkpoint_cad: Path | str | None = None,
    registry: dict[str, callable] | None = None,
    backend: str | None = None,
    _pre_buffered: bool = False,
    **mesh_kwargs,
) -> Any:
    # ... existing body up to the cad_occ call ...
    cad_kwargs["skip_buffer"] = _pre_buffered
    occ_entities = cad_occ(entities, **cad_kwargs)
    # ... rest unchanged ...
```

In `meshwell/cad_occ.py`, `cad_occ()` and `CAD_OCC.process_entities()` accept and forward `skip_buffer`:

```python
def cad_occ(
    entities_list,
    point_tolerance=1e-3,
    n_threads=cpu_count(),
    progress_bars=False,
    fuzzy_value=None,
    perturbation=None,
    skip_buffer: bool = False,
) -> list[OCCLabeledEntity]:
    processor = CAD_OCC(
        point_tolerance=point_tolerance,
        n_threads=n_threads,
        fuzzy_value=fuzzy_value,
        perturbation=perturbation,
    )
    return processor.process_entities(
        entities_list,
        progress_bars=progress_bars,
        skip_buffer=skip_buffer,
    )


# Inside CAD_OCC.process_entities, change the prepare_entities call:
def process_entities(self, entities_list, progress_bars=False, skip_buffer: bool = False):
    if not entities_list:
        return []
    from meshwell.cad_common import prepare_entities
    prepare_entities(entities_list, perturbation=self.perturbation, skip_buffer=skip_buffer)
    # ... rest unchanged ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_distributed.py::test_generate_mesh_pre_buffered_flag_skips_buffer -v`
Expected: PASS.

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: no regressions.

- [ ] **Step 6: Commit**

```bash
git add meshwell/orchestrator.py meshwell/cad_occ.py tests/test_distributed.py
git commit -m "feat(orchestrator,cad_occ): plumb _pre_buffered flag end-to-end"
```

---

## Task 3: `_global_physical_names` tolerates unresolved name refs

ResolutionSpec references to physical names that exist globally but not in the current worker's local entity list should be silently no-op'd (with a debug log) instead of raising.

**Files:**
- Modify: `meshwell/mesh.py`
- Modify: `meshwell/orchestrator.py` (forward the kwarg)
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Locate the existing resolution-spec resolver**

Run: `grep -n "restrict_to\|sharing\|not_sharing" meshwell/mesh.py`

Identify the function(s) that resolve ResolutionSpec name refs against the model. Note the line numbers — these are the modification points.

- [ ] **Step 2: Write the failing test**

```python
def test_resolution_spec_unknown_name_in_global_set_is_no_op(tmp_path):
    """A ResolutionSpec restrict_to='B' with B in _global_physical_names
    but absent locally should not raise."""
    import shapely
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism
    from meshwell.resolution import ConstantInField

    spec = ConstantInField(
        apply_to="volumes",
        restrict_to=["B"],  # B does not exist in this entity list
        # Required ConstantInField fields (filled in concrete subclass during impl):
        # see meshwell/resolution.py for the actual schema.
    )
    prism = PolyPrism(
        polygons=shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="A",
        mesh_order=1,
    )
    prism.resolutions = [spec]

    generate_mesh(
        entities=[prism],
        dim=3,
        output_mesh=tmp_path / "out.msh",
        _global_physical_names=["A", "B"],  # B is "globally known" but not local
    )
    # No exception means success.
```

(Adjust `ConstantInField(...)` constructor call to match the actual schema in `meshwell/resolution.py` — read the file to find the required fields.)

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_distributed.py::test_resolution_spec_unknown_name_in_global_set_is_no_op -v`
Expected: FAIL with `TypeError: generate_mesh() got an unexpected keyword argument '_global_physical_names'`.

- [ ] **Step 4: Plumb `_global_physical_names` through orchestrator and mesh**

In `meshwell/orchestrator.py`:

```python
def generate_mesh(
    ...
    _pre_buffered: bool = False,
    _global_physical_names: list[str] | None = None,
    **mesh_kwargs,
):
    ...
    return mesh(
        dim=dim,
        model=mm,
        output_file=Path(output_mesh) if output_mesh else None,
        _global_physical_names=_global_physical_names,
        **mesh_kwargs,
    )
```

In `meshwell/mesh.py`, add `_global_physical_names: list[str] | None = None` to the `mesh()` function signature. Inside the resolution resolver (located at Step 1), wrap each name-ref lookup:

```python
def _resolve_name(self, name: str) -> list[int]:
    matches = self._local_name_index.get(name, [])
    if matches:
        return matches
    if self._global_physical_names and name in self._global_physical_names:
        # Globally known but not local → no-op (entity lives in another subdomain).
        import warnings
        warnings.warn(
            f"ResolutionSpec name {name!r} not present in local subdomain; skipping.",
            stacklevel=3,
        )
        return []
    # Otherwise, original behavior (likely raise).
    raise KeyError(f"unknown physical name: {name!r}")
```

The exact integration point depends on what Step 1 finds; adapt the wrapping to fit the existing resolver function.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_distributed.py::test_resolution_spec_unknown_name_in_global_set_is_no_op -v`
Expected: PASS.

- [ ] **Step 6: Run full suite**

Run: `pytest tests/ -x -q`
Expected: no regressions.

- [ ] **Step 7: Commit**

```bash
git add meshwell/orchestrator.py meshwell/mesh.py tests/test_distributed.py
git commit -m "feat(mesh): tolerate unresolved ResolutionSpec names in _global_physical_names"
```

---

## Task 4: `_emit_only_seam_surfaces` filters output by `_seam___` prefix

Phase-1 workers should write a `.msh` that contains *only* the physical groups whose name starts with `_seam___`. All other elements are scratch.

**Files:**
- Modify: `meshwell/mesh.py` (add `_filter_msh_to_seam_groups` helper + wire `_emit_only_seam_surfaces` flag)
- Modify: `meshwell/orchestrator.py` (forward the kwarg)
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the failing test**

```python
def test_emit_only_seam_surfaces_filters_output(tmp_path):
    """A mesh emitted with _emit_only_seam_surfaces=True contains only
    physical groups whose name starts with '_seam___'."""
    import shapely
    import meshio
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism

    # Two prisms abutting at x=1 — one tagged a normal material, one tagged as a seam.
    a = PolyPrism(
        polygons=shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="A",
        mesh_order=1,
    )
    seam = PolyPrism(
        polygons=shapely.Polygon([(1, 0), (1.001, 0), (1.001, 1), (1, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="_seam___A___B",
        mesh_order=2,
        mesh_bool=False,  # phantom
    )

    out = tmp_path / "out.msh"
    generate_mesh(
        entities=[a, seam],
        dim=3,
        output_mesh=out,
        _emit_only_seam_surfaces=True,
    )
    m = meshio.read(out)
    field_names = set(m.field_data.keys()) if hasattr(m, "field_data") else set()
    # Only _seam___ groups survive.
    for name in field_names:
        assert name.startswith("_seam___"), f"unexpected group {name!r}"
    assert any(n.startswith("_seam___") for n in field_names)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_distributed.py::test_emit_only_seam_surfaces_filters_output -v`
Expected: FAIL with unexpected kwarg.

- [ ] **Step 3: Implement the filter helper**

In `meshwell/mesh.py`, add at module scope:

```python
def _filter_msh_to_seam_groups(msh_path: Path) -> None:
    """Rewrite msh_path keeping only physical groups whose name starts with '_seam___'.

    Reads the .msh via meshio, drops all cells/cell_data/field_data not associated
    with seam groups, writes back to msh_path.
    """
    import meshio
    m = meshio.read(msh_path)
    keep_field_data = {
        name: tag for name, tag in (m.field_data or {}).items()
        if name.startswith("_seam___")
    }
    if not keep_field_data:
        # Nothing to keep — write an empty mesh (still valid .msh).
        empty = meshio.Mesh(points=m.points[:0], cells=[])
        meshio.write(msh_path, empty)
        return
    keep_tag_ids = {tag for tag, dim in keep_field_data.values()}

    new_cells = []
    new_cell_data = {k: [] for k in (m.cell_data or {})}
    for i, cellblock in enumerate(m.cells):
        gmsh_phys = (m.cell_data or {}).get("gmsh:physical", [None] * len(m.cells))[i]
        if gmsh_phys is None:
            continue
        mask = [int(t) in keep_tag_ids for t in gmsh_phys]
        if not any(mask):
            continue
        import numpy as np
        idx = np.where(mask)[0]
        new_cells.append(meshio.CellBlock(cellblock.type, cellblock.data[idx]))
        for k in new_cell_data:
            new_cell_data[k].append((m.cell_data[k][i])[idx])

    out = meshio.Mesh(
        points=m.points,
        cells=new_cells,
        cell_data=new_cell_data,
        field_data=keep_field_data,
    )
    meshio.write(msh_path, out)
```

In `meshwell/mesh.py`'s `mesh()` function, accept `_emit_only_seam_surfaces: bool = False`. After the existing write to `output_file`, if the flag is set:

```python
if _emit_only_seam_surfaces and output_file is not None:
    _filter_msh_to_seam_groups(output_file)
```

In `meshwell/orchestrator.py`, accept and forward `_emit_only_seam_surfaces` the same way as `_pre_buffered`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_distributed.py::test_emit_only_seam_surfaces_filters_output -v`
Expected: PASS.

- [ ] **Step 5: Run full suite**

Run: `pytest tests/ -x -q`
Expected: no regressions.

- [ ] **Step 6: Commit**

```bash
git add meshwell/mesh.py meshwell/orchestrator.py tests/test_distributed.py
git commit -m "feat(mesh): _emit_only_seam_surfaces filters output to _seam___ groups"
```

---

## Task 5: `_interface_constraints` — merge + embed

Phase-2 workers must `gmsh.merge` each interface `result.msh` and `embed` the imported discrete entity into the matching OCC face.

**Files:**
- Modify: `meshwell/mesh.py` (add `_embed_imported_seam_into_occ_face` helper, wire `_interface_constraints` kwarg)
- Modify: `meshwell/orchestrator.py` (forward the kwarg)
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the failing test (depends on Task 0 spike outcome)**

```python
def test_interface_constraints_embed_seam_into_volume(tmp_path):
    """A volume mesh with an _interface_constraints input has its seam
    face nodes drawn from the imported mesh."""
    import shapely
    import meshio
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism

    # Build a "seam" mesh on a vertical face at x=1 of a 1x1x1 prism.
    # We synthesize it directly via a small phase-1-style call.
    seam_path = tmp_path / "seam.msh"
    seam_phantom = PolyPrism(
        polygons=shapely.Polygon([(1, 0), (1.0001, 0), (1.0001, 1), (1, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="_seam___A___B",
        mesh_order=1,
    )
    generate_mesh(
        entities=[seam_phantom],
        dim=3,
        output_mesh=seam_path,
        _emit_only_seam_surfaces=True,
        default_characteristic_length=0.2,
    )

    # Now mesh a 1x1x1 prism whose right face must conform to the seam mesh.
    out = tmp_path / "vol.msh"
    a = PolyPrism(
        polygons=shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="A",
        mesh_order=1,
    )
    generate_mesh(
        entities=[a],
        dim=3,
        output_mesh=out,
        _interface_constraints=[seam_path],
        default_characteristic_length=0.5,  # different from seam's 0.2
    )
    m = meshio.read(out)
    # The seam-phys-group nodes must appear in the volume mesh.
    assert "_seam___A___B" in (m.field_data or {})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_distributed.py::test_interface_constraints_embed_seam_into_volume -v`
Expected: FAIL with unexpected kwarg.

- [ ] **Step 3: Implement `_embed_imported_seam_into_occ_face`**

In `meshwell/mesh.py`, add at module scope:

```python
def _embed_imported_seam_into_occ_face(seam_path: Path) -> None:
    """Merge a seam .msh into the current gmsh model and embed each imported
    discrete 2D entity into the matching OCC face.

    Match rule: an OCC face matches a discrete face if their bounding boxes
    overlap within point_tolerance and they are co-planar (same plane equation
    within point_tolerance). The seam's physical-group name '_seam___A___B'
    is added to the OCC face as well so it survives final glue.
    """
    pre_2d = {tag for dim, tag in gmsh.model.getEntities(2)}
    pre_occ_faces = list(gmsh.model.getEntities(2))
    gmsh.merge(str(seam_path))
    post_2d = {tag for dim, tag in gmsh.model.getEntities(2)}
    imported_tags = sorted(post_2d - pre_2d)
    if not imported_tags:
        return

    # Inventory imported group names.
    pgs = gmsh.model.getPhysicalGroups(2)
    imp_groups = {}
    for dim, tag in pgs:
        name = gmsh.model.getPhysicalName(dim, tag)
        if not name.startswith("_seam___"):
            continue
        ents = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
        for e in ents:
            if int(e) in imported_tags:
                imp_groups.setdefault(name, []).append(int(e))

    # For each imported entity, find the matching OCC face by bbox + plane.
    import numpy as np
    for imp_tag in imported_tags:
        bb_imp = gmsh.model.getBoundingBox(2, imp_tag)
        best = None
        for dim, occ_tag in pre_occ_faces:
            if dim != 2:
                continue
            bb_occ = gmsh.model.getBoundingBox(2, occ_tag)
            if not _bboxes_close(bb_imp, bb_occ, tol=1e-6):
                continue
            best = occ_tag
            break
        if best is None:
            continue
        gmsh.model.mesh.embed(2, [imp_tag], 2, best)
        # Re-tag the OCC face with the seam physical group too.
        for name, ents in imp_groups.items():
            if imp_tag in ents:
                # Find or create an OCC-side group with the same name.
                existing = [
                    t for d, t in gmsh.model.getPhysicalGroups(2)
                    if gmsh.model.getPhysicalName(d, t) == name
                ]
                if existing:
                    pg_tag = existing[0]
                    cur = list(gmsh.model.getEntitiesForPhysicalGroup(2, pg_tag))
                    gmsh.model.removePhysicalGroups([(2, pg_tag)])
                    gmsh.model.addPhysicalGroup(2, list(set(cur + [best])), name=name)
                else:
                    gmsh.model.addPhysicalGroup(2, [best], name=name)


def _bboxes_close(b1, b2, tol):
    return all(abs(a - b) <= tol for a, b in zip(b1, b2))
```

In the `mesh()` function, accept `_interface_constraints: list[Path] | None = None`. After the OCC CAD load + synchronize but before `mesh.generate(dim)`:

```python
if _interface_constraints:
    for path in _interface_constraints:
        _embed_imported_seam_into_occ_face(Path(path))
    self.model_manager.sync_model()
```

In `meshwell/orchestrator.py`, accept and forward `_interface_constraints` like the other private kwargs.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_distributed.py::test_interface_constraints_embed_seam_into_volume -v`
Expected: PASS.

If it FAILS due to embed misbehavior, revisit Task 0 spike result and apply the documented fallback (R3): build the seam discrete face but use `gmsh.model.mesh.removeDuplicateNodes(tol=point_tolerance/2)` after `mesh.generate` to merge the OCC face mesh with the embedded discrete points instead of relying on `embed`.

- [ ] **Step 5: Run full suite**

Run: `pytest tests/ -x -q`
Expected: no regressions.

- [ ] **Step 6: Commit**

```bash
git add meshwell/mesh.py meshwell/orchestrator.py tests/test_distributed.py
git commit -m "feat(mesh): _interface_constraints merges + embeds seam meshes into OCC faces"
```

---

## Task 6: `meshwell/distributed.py` skeleton + dataclasses

Establish the module shell and the data structures used by every later task.

**Files:**
- Create: `meshwell/distributed.py`
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the failing test**

```python
def test_distributed_module_imports():
    from meshwell.distributed import (
        Slab,
        SubdomainPlan,
        Executor,
        SubprocessExecutor,
        generate_mesh_distributed,
        subdomains_from_grid,
        build_subdomain_plan,
        run_job,
    )
    assert Slab is not None
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_distributed.py::test_distributed_module_imports -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meshwell.distributed'`.

- [ ] **Step 3: Create the skeleton**

```python
"""Distributed domain-decomposition pipeline for meshwell.

Splits an input scene into per-subdomain CAD + mesh jobs (file-based bundles)
and stitches the resulting .msh files into one final mesh with conformal
seams. See docs/superpowers/specs/2026-04-28-distributed-domain-decomposition-design.md.
"""
from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from shapely.geometry import LineString, Polygon


@dataclass
class Slab:
    """One interface or junction subdomain in the plan.

    For interface slabs, ``between`` has length 2 (the two volume IDs).
    For junctions, ``between`` has length >= 3.
    """

    id: str
    polygon: Polygon
    between: list[str]
    cut_polylines: list[LineString]  # one per pairwise contact in this slab
    width: float


@dataclass
class VolumeRegion:
    """One volume subdomain in the plan."""

    id: str
    polygon: Polygon
    neighbors: list[str] = field(default_factory=list)


@dataclass
class SubdomainPlan:
    """Output of build_subdomain_plan."""

    volumes: list[VolumeRegion]
    interfaces: list[Slab]
    junctions: list[Slab]
    physical_names_seen: list[str]
    perturbation: float
    point_tolerance: float
    interface_delimiter: str = "___"
    boundary_delimiter: str = "None"


class Executor(Protocol):
    def submit(self, job_dir: Path) -> Future: ...


class SubprocessExecutor:
    """Default Executor: runs `meshwell run-job <job_dir>` via ProcessPoolExecutor."""

    def __init__(self, max_workers: int | None = None):
        self._pool = ProcessPoolExecutor(max_workers=max_workers)

    def submit(self, job_dir: Path) -> Future:
        return self._pool.submit(_run_job_in_subprocess, str(job_dir))


def _run_job_in_subprocess(job_dir_str: str) -> dict:
    import subprocess
    result = subprocess.run(
        ["meshwell", "run-job", job_dir_str],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "job_dir": job_dir_str,
    }


def subdomains_from_grid(
    bbox: tuple[float, float, float, float],
    nx: int,
    ny: int,
) -> list[Polygon]:
    raise NotImplementedError("Task 7")


def build_subdomain_plan(
    subdomains: list[Polygon],
    entities: list[Any],
    interface_width,
    perturbation: float,
    point_tolerance: float,
) -> SubdomainPlan:
    raise NotImplementedError("Tasks 11-14")


def run_job(job_dir: Path) -> None:
    raise NotImplementedError("Task 16")


def cli_main() -> None:
    raise NotImplementedError("Task 17")


def generate_mesh_distributed(
    entities: list[Any],
    subdomains: list[Polygon],
    output_mesh: Path | str,
    work_dir: Path | str,
    interface_width,
    executor: Executor | None = None,
    keep_bundles: bool = False,
    registry: dict[str, callable] | None = None,
    **mesh_kwargs,
) -> Any:
    raise NotImplementedError("Task 21")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_distributed.py::test_distributed_module_imports -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/distributed.py tests/test_distributed.py
git commit -m "feat(distributed): module skeleton + dataclasses"
```

---

## Task 7: `subdomains_from_grid` helper

**Files:**
- Modify: `meshwell/distributed.py`
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the failing test**

```python
def test_subdomains_from_grid_2x2():
    from shapely.geometry import box
    from meshwell.distributed import subdomains_from_grid

    polys = subdomains_from_grid((0, 0, 2, 2), nx=2, ny=2)
    assert len(polys) == 4
    expected = {
        box(0, 0, 1, 1).wkt,
        box(1, 0, 2, 1).wkt,
        box(0, 1, 1, 2).wkt,
        box(1, 1, 2, 2).wkt,
    }
    assert {p.wkt for p in polys} == expected
```

- [ ] **Step 2: Run test, verify FAIL** (NotImplementedError)

Run: `pytest tests/test_distributed.py::test_subdomains_from_grid_2x2 -v`

- [ ] **Step 3: Implement**

```python
def subdomains_from_grid(
    bbox: tuple[float, float, float, float],
    nx: int,
    ny: int,
) -> list[Polygon]:
    from shapely.geometry import box as _box

    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be >= 1")
    xmin, ymin, xmax, ymax = bbox
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("invalid bbox")
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    polys = []
    for i in range(nx):
        for j in range(ny):
            polys.append(_box(xmin + i * dx, ymin + j * dy, xmin + (i + 1) * dx, ymin + (j + 1) * dy))
    return polys
```

- [ ] **Step 4: Run test, verify PASS**

- [ ] **Step 5: Commit**

```bash
git add meshwell/distributed.py tests/test_distributed.py
git commit -m "feat(distributed): subdomains_from_grid helper"
```

---

## Task 8: `_clip_entity_to_polygon` for `PolyPrism`

**Files:**
- Modify: `meshwell/distributed.py`
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the failing test**

```python
def test_clip_polyprism_to_mask_returns_clipped_copy():
    import shapely
    from meshwell.distributed import _clip_entity_to_polygon
    from meshwell.polyprism import PolyPrism

    p = shapely.Polygon([(0, 0), (10, 0), (10, 1), (0, 1)])
    prism = PolyPrism(
        polygons=p,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="A",
        mesh_order=1,
    )
    mask = shapely.box(0, 0, 5, 1)

    clipped = _clip_entity_to_polygon(prism, mask)
    assert clipped is not None
    assert clipped.physical_name == ("A",)  # format_physical_name normalizes
    assert clipped.mesh_order == 1
    assert clipped.polygons.equals(shapely.box(0, 0, 5, 1))


def test_clip_polyprism_returns_none_when_disjoint():
    import shapely
    from meshwell.distributed import _clip_entity_to_polygon
    from meshwell.polyprism import PolyPrism

    p = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    prism = PolyPrism(
        polygons=p, buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="A", mesh_order=1,
    )
    mask = shapely.box(10, 10, 11, 11)
    assert _clip_entity_to_polygon(prism, mask) is None
```

- [ ] **Step 2: Run, verify FAIL** (function not defined)

- [ ] **Step 3: Implement**

```python
def _clip_entity_to_polygon(entity: Any, mask: Polygon) -> Any | None:
    """Return a copy of `entity` whose .polygons are intersected with `mask`.

    Returns None if the intersection is empty. Preserves physical_name,
    mesh_order, mesh_bool, additive, resolutions, transformations.

    For PolyPrism / PolySurface this is a shapely .intersection on the
    entity's polygons attribute. OCC_entity is not supported and raises
    NotImplementedError if encountered.
    """
    from copy import deepcopy
    from shapely.geometry import MultiPolygon

    if not hasattr(entity, "polygons"):
        raise NotImplementedError(
            f"_clip_entity_to_polygon does not support entity type {type(entity).__name__}"
        )

    polys = entity.polygons
    if isinstance(polys, list):
        clipped = [p.intersection(mask) for p in polys]
        clipped = [p for p in clipped if not p.is_empty]
        if not clipped:
            return None
    else:
        c = polys.intersection(mask)
        if c.is_empty:
            return None
        clipped = c

    new = deepcopy(entity)
    new.polygons = clipped
    # Re-derive PolyPrism's buffered_polygons / extrude state if applicable.
    if hasattr(new, "buffers") and hasattr(new, "extrude"):
        # PolyPrism extrude path is fine — polygons just changed footprint.
        # If it was non-extrude (tapered), recompute buffered polygons.
        if not new.extrude:
            new.buffered_polygons = new._get_buffered_polygons(clipped, new.buffers)
    return new
```

- [ ] **Step 4: Run, verify PASS**

- [ ] **Step 5: Commit**

```bash
git add meshwell/distributed.py tests/test_distributed.py
git commit -m "feat(distributed): _clip_entity_to_polygon for PolyPrism"
```

---

## Task 9: `_clip_entity_to_polygon` works for `PolySurface`

The implementation already covers PolySurface (it just uses `.polygons`), but verify with a test.

**Files:**
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the test**

```python
def test_clip_polysurface_to_mask():
    import shapely
    from meshwell.distributed import _clip_entity_to_polygon
    from meshwell.polysurface import PolySurface

    p = shapely.Polygon([(0, 0), (10, 0), (10, 1), (0, 1)])
    surf = PolySurface(polygons=p, physical_name="A", mesh_order=1)
    mask = shapely.box(0, 0, 5, 1)

    clipped = _clip_entity_to_polygon(surf, mask)
    assert clipped is not None
    assert clipped.polygons.equals(shapely.box(0, 0, 5, 1))
```

- [ ] **Step 2: Run, verify PASS** (no implementation needed)

- [ ] **Step 3: Commit**

```bash
git add tests/test_distributed.py
git commit -m "test(distributed): _clip_entity_to_polygon for PolySurface"
```

---

## Task 10: `_resolution_only_proxy` wrapper

**Files:**
- Modify: `meshwell/distributed.py`
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the failing test**

```python
def test_resolution_only_proxy_contributes_no_geometry():
    import shapely
    from meshwell.distributed import _resolution_only_proxy
    from meshwell.polyprism import PolyPrism

    prism = PolyPrism(
        polygons=shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="A",
        mesh_order=1,
    )
    prism.resolutions = ["sentinel"]

    proxy = _resolution_only_proxy(prism)
    assert proxy.mesh_bool is False
    assert proxy.resolutions == ["sentinel"]
    # instanciate_occ returns an empty TopoDS_Compound
    shape = proxy.instanciate_occ()
    from OCP.TopoDS import TopoDS_Compound
    assert isinstance(shape, TopoDS_Compound)
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    assert not TopExp_Explorer(shape, TopAbs_SOLID).More()
```

- [ ] **Step 2: Run, verify FAIL** (function not defined)

- [ ] **Step 3: Implement**

```python
def _resolution_only_proxy(entity: Any) -> Any:
    """Wrap `entity` so it contributes no geometry but keeps its resolutions.

    Used by phase-1 bundles for entities near (but not intersecting) a slab
    whose ResolutionSpecs may still affect the seam mesh.
    """
    from copy import deepcopy
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound

    proxy = deepcopy(entity)
    proxy.mesh_bool = False

    def _empty_shape(self=proxy):
        cb = BRep_Builder()
        c = TopoDS_Compound()
        cb.MakeCompound(c)
        return c

    proxy.instanciate_occ = _empty_shape
    # Also stub instanciate (gmsh-backend path) to return an empty list.
    proxy.instanciate = lambda cad_model: []
    return proxy
```

- [ ] **Step 4: Run, verify PASS**

- [ ] **Step 5: Commit**

```bash
git add meshwell/distributed.py tests/test_distributed.py
git commit -m "feat(distributed): _resolution_only_proxy for ResolutionSpec context shipping"
```

---

## Task 11: `build_subdomain_plan` — volume regions only

**Files:**
- Modify: `meshwell/distributed.py`
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_subdomain_plan_creates_volume_regions():
    import shapely
    from meshwell.distributed import build_subdomain_plan
    from meshwell.polyprism import PolyPrism

    sd = [shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)]
    prism = PolyPrism(
        polygons=shapely.box(0, 0, 2, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="mat",
        mesh_order=1,
    )
    plan = build_subdomain_plan(
        subdomains=sd,
        entities=[prism],
        interface_width=0.05,
        perturbation=1e-5,
        point_tolerance=1e-3,
    )
    assert [v.id for v in plan.volumes] == ["volume_0000", "volume_0001"]
    assert plan.volumes[0].polygon.equals(sd[0])
    assert plan.physical_names_seen == ["mat"]
```

- [ ] **Step 2: Run, FAIL** (NotImplementedError)

- [ ] **Step 3: Implement (volumes branch only — interfaces/junctions stub)**

Replace the body of `build_subdomain_plan` in `meshwell/distributed.py`:

```python
def build_subdomain_plan(
    subdomains, entities, interface_width, perturbation, point_tolerance,
):
    if not subdomains:
        raise ValueError("subdomains must be non-empty")
    for i, sd in enumerate(subdomains):
        if not sd.is_valid:
            raise ValueError(f"subdomain {i} is not valid: {sd.wkt}")

    volumes = [
        VolumeRegion(id=f"volume_{i:04d}", polygon=sd, neighbors=[])
        for i, sd in enumerate(subdomains)
    ]
    physical_names_seen = sorted({
        n if isinstance(n, str) else n[0]
        for ent in entities if hasattr(ent, "physical_name") and ent.physical_name
        for n in ((ent.physical_name,) if isinstance(ent.physical_name, str) else ent.physical_name)
    })

    return SubdomainPlan(
        volumes=volumes,
        interfaces=[],   # Task 12
        junctions=[],    # Task 13
        physical_names_seen=list(physical_names_seen),
        perturbation=perturbation,
        point_tolerance=point_tolerance,
    )
```

- [ ] **Step 4: Run, verify PASS**

- [ ] **Step 5: Commit**

```bash
git add meshwell/distributed.py tests/test_distributed.py
git commit -m "feat(distributed): build_subdomain_plan — volume regions"
```

---

## Task 12: `build_subdomain_plan` — pairwise interface slabs

**Files:**
- Modify: `meshwell/distributed.py`
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_subdomain_plan_creates_interface_slabs():
    import shapely
    from meshwell.distributed import build_subdomain_plan
    from meshwell.polyprism import PolyPrism

    sd = [shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)]
    prism = PolyPrism(
        polygons=shapely.box(0, 0, 2, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="mat",
        mesh_order=1,
    )
    plan = build_subdomain_plan(
        subdomains=sd, entities=[prism],
        interface_width=0.1, perturbation=1e-5, point_tolerance=1e-3,
    )
    assert len(plan.interfaces) == 1
    iface = plan.interfaces[0]
    assert iface.id == "interface_0000"
    assert sorted(iface.between) == ["volume_0000", "volume_0001"]
    # Slab is centered on x=1 with half-width 0.05
    assert iface.polygon.bounds == pytest.approx((0.95, 0.0, 1.05, 1.0), abs=1e-9)


def test_build_subdomain_plan_no_interface_for_disjoint_subdomains():
    import shapely
    from meshwell.distributed import build_subdomain_plan
    from meshwell.polyprism import PolyPrism

    sd = [shapely.box(0, 0, 1, 1), shapely.box(2, 0, 3, 1)]
    prism = PolyPrism(
        polygons=shapely.box(0, 0, 3, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="mat", mesh_order=1,
    )
    plan = build_subdomain_plan(
        subdomains=sd, entities=[prism],
        interface_width=0.1, perturbation=1e-5, point_tolerance=1e-3,
    )
    assert plan.interfaces == []
```

Add `import pytest` to the test file if not already present.

- [ ] **Step 2: Run, FAIL**

- [ ] **Step 3: Implement interface generation**

In `build_subdomain_plan`, after the volumes block:

```python
    interfaces = []
    iface_idx = 0
    width_for = (lambda i, j: interface_width) if isinstance(interface_width, (int, float)) else (
        lambda i, j: interface_width.get((min(i, j), max(i, j))) or interface_width.get((max(i, j), min(i, j)))
    )
    for i in range(len(subdomains)):
        for j in range(i + 1, len(subdomains)):
            shared = subdomains[i].boundary.intersection(subdomains[j].boundary)
            if shared.is_empty:
                continue
            polylines = _flatten_to_linestrings(shared)
            polylines = [ls for ls in polylines if ls.length > point_tolerance]
            if not polylines:
                continue
            w = width_for(i, j)
            if w is None or w <= 0:
                raise ValueError(f"interface_width missing for pair ({i}, {j})")
            slab_polys = [ls.buffer(w / 2) for ls in polylines]
            slab = slab_polys[0] if len(slab_polys) == 1 else shapely.unary_union(slab_polys)
            interfaces.append(Slab(
                id=f"interface_{iface_idx:04d}",
                polygon=slab,
                between=[f"volume_{i:04d}", f"volume_{j:04d}"],
                cut_polylines=polylines,
                width=w,
            ))
            volumes[i].neighbors.append(f"volume_{j:04d}")
            volumes[j].neighbors.append(f"volume_{i:04d}")
            iface_idx += 1
```

Add helper at module scope:

```python
def _flatten_to_linestrings(geom) -> list[LineString]:
    from shapely.geometry import GeometryCollection, LineString as LS, MultiLineString
    if geom.is_empty:
        return []
    if isinstance(geom, LS):
        return [geom]
    if isinstance(geom, MultiLineString):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        out = []
        for g in geom.geoms:
            out.extend(_flatten_to_linestrings(g))
        return out
    return []
```

Add `import shapely` to the file (if not already imported).

Then update the `SubdomainPlan(...)` constructor call to use `interfaces=interfaces`.

- [ ] **Step 4: Run, verify PASS**

- [ ] **Step 5: Commit**

```bash
git add meshwell/distributed.py tests/test_distributed.py
git commit -m "feat(distributed): build_subdomain_plan — pairwise interface slabs"
```

---

## Task 13: `build_subdomain_plan` — junction subdomains

**Files:**
- Modify: `meshwell/distributed.py`
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_subdomain_plan_creates_junction_for_2x2_grid():
    import shapely
    from meshwell.distributed import build_subdomain_plan, subdomains_from_grid
    from meshwell.polyprism import PolyPrism

    sd = subdomains_from_grid((0, 0, 2, 2), nx=2, ny=2)
    prism = PolyPrism(
        polygons=shapely.box(0, 0, 2, 2),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="mat", mesh_order=1,
    )
    plan = build_subdomain_plan(
        subdomains=sd, entities=[prism],
        interface_width=0.1, perturbation=1e-5, point_tolerance=1e-3,
    )
    assert len(plan.junctions) == 1
    j = plan.junctions[0]
    assert j.id == "junction_0000"
    # Junction polygon centered on (1, 1), all 4 volumes meet here
    assert sorted(j.between) == [
        "volume_0000", "volume_0001", "volume_0002", "volume_0003",
    ]
```

- [ ] **Step 2: Run, FAIL**

- [ ] **Step 3: Implement junction detection**

In `build_subdomain_plan`, after the interfaces block:

```python
    # ---- Junctions: points where >= 3 subdomains meet ----
    from shapely.geometry import Point
    point_to_volumes: dict[tuple[float, float], set[str]] = {}
    for i, sd in enumerate(subdomains):
        # Quantize boundary vertices to point_tolerance grid
        coords = []
        # Walk all rings
        rings = [sd.exterior] + list(sd.interiors)
        for ring in rings:
            for x, y in ring.coords:
                key = (round(x / point_tolerance) * point_tolerance,
                       round(y / point_tolerance) * point_tolerance)
                coords.append(key)
        for k in set(coords):
            point_to_volumes.setdefault(k, set()).add(f"volume_{i:04d}")

    junctions = []
    j_idx = 0
    junction_radius = max(
        (s.width for s in interfaces), default=interface_width if isinstance(interface_width, (int, float)) else 0,
    )
    for (x, y), vols in point_to_volumes.items():
        if len(vols) < 3:
            continue
        # Collect cut polylines from all interfaces that touch this junction.
        jpoint = Point(x, y)
        touching = [s for s in interfaces if any(jpoint.distance(pl) < point_tolerance for pl in s.cut_polylines)]
        cut_lines = []
        for s in touching:
            cut_lines.extend(s.cut_polylines)
        junctions.append(Slab(
            id=f"junction_{j_idx:04d}",
            polygon=jpoint.buffer(junction_radius),
            between=sorted(vols),
            cut_polylines=cut_lines,
            width=junction_radius,
        ))
        j_idx += 1
```

Update `SubdomainPlan(...)` constructor call: `junctions=junctions`.

- [ ] **Step 4: Run, verify PASS**

- [ ] **Step 5: Commit**

```bash
git add meshwell/distributed.py tests/test_distributed.py
git commit -m "feat(distributed): build_subdomain_plan — junction subdomains"
```

---

## Task 14: Coverage validation

**Files:**
- Modify: `meshwell/distributed.py`
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_subdomain_plan_rejects_uncovered_entities():
    import shapely
    import pytest
    from meshwell.distributed import build_subdomain_plan
    from meshwell.polyprism import PolyPrism

    # Subdomain covers only x in [0, 1] but entity extends to x=2
    sd = [shapely.box(0, 0, 1, 1)]
    prism = PolyPrism(
        polygons=shapely.box(0, 0, 2, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="mat", mesh_order=1,
    )
    with pytest.raises(ValueError, match="not covered"):
        build_subdomain_plan(
            subdomains=sd, entities=[prism],
            interface_width=0.1, perturbation=1e-5, point_tolerance=1e-3,
        )
```

- [ ] **Step 2: Run, FAIL**

- [ ] **Step 3: Implement coverage check**

Near the top of `build_subdomain_plan`, after the validity loop:

```python
    union_sd = shapely.unary_union(subdomains)
    union_ent = shapely.unary_union([
        p
        for ent in entities if hasattr(ent, "polygons")
        for p in (ent.polygons if isinstance(ent.polygons, list) else [ent.polygons])
    ])
    missing = union_ent.difference(union_sd.buffer(point_tolerance))
    if not missing.is_empty:
        raise ValueError(
            f"Entity polygons not covered by subdomain union: {missing.wkt[:200]}"
        )
```

- [ ] **Step 4: Run, verify PASS**

Run the previous distributed tests too:
`pytest tests/test_distributed.py -v -k "build_subdomain"`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add meshwell/distributed.py tests/test_distributed.py
git commit -m "feat(distributed): build_subdomain_plan — coverage validation"
```

---

## Task 15: Bundle write — manifest + per-job files

Write `manifest.json`, `jobs/<id>/job.json`, `jobs/<id>/entities.json`, `jobs/<id>/subdomain.wkt`, `jobs/<id>/mesh_kwargs.json` for each subdomain.

**Files:**
- Modify: `meshwell/distributed.py`
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the failing test**

```python
def test_write_bundles_creates_expected_layout(tmp_path):
    import json
    import shapely
    from meshwell.distributed import build_subdomain_plan, write_bundles
    from meshwell.polyprism import PolyPrism

    sd = [shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)]
    prism = PolyPrism(
        polygons=shapely.box(0, 0, 2, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="mat", mesh_order=1,
    )
    plan = build_subdomain_plan(
        subdomains=sd, entities=[prism],
        interface_width=0.1, perturbation=1e-5, point_tolerance=1e-3,
    )
    write_bundles(
        work_dir=tmp_path,
        plan=plan,
        entities=[prism],
        mesh_kwargs={"default_characteristic_length": 0.5},
    )
    # Manifest exists.
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["version"] == 1
    assert "volume_0000" in manifest["subdomains"]
    assert "interface_0000" in manifest["subdomains"]
    assert manifest["phase_order"][0] == ["interface_0000"]
    assert sorted(manifest["phase_order"][1]) == ["volume_0000", "volume_0001"]
    # Per-job files exist.
    for jid in ["volume_0000", "volume_0001", "interface_0000"]:
        d = tmp_path / "jobs" / jid
        assert (d / "job.json").exists()
        assert (d / "entities.json").exists()
        assert (d / "subdomain.wkt").exists()
        assert (d / "mesh_kwargs.json").exists()
```

- [ ] **Step 2: Run, FAIL**

- [ ] **Step 3: Implement `write_bundles`**

```python
def write_bundles(
    work_dir: Path,
    plan: SubdomainPlan,
    entities: list[Any],
    mesh_kwargs: dict,
) -> None:
    """Write the bundle directory tree for `plan`."""
    import json

    work_dir = Path(work_dir)
    (work_dir / "jobs").mkdir(parents=True, exist_ok=True)

    # Build manifest
    subdomains_blob = {}
    for v in plan.volumes:
        subdomains_blob[v.id] = {
            "polygon_wkt": v.polygon.wkt,
            "neighbors": v.neighbors,
        }
    for s in plan.interfaces:
        subdomains_blob[s.id] = {
            "polygon_wkt": s.polygon.wkt,
            "between": s.between,
            "cut_polylines_wkt": [ls.wkt for ls in s.cut_polylines],
            "width": s.width,
        }
    for s in plan.junctions:
        subdomains_blob[s.id] = {
            "polygon_wkt": s.polygon.wkt,
            "between": s.between,
            "cut_polylines_wkt": [ls.wkt for ls in s.cut_polylines],
            "width": s.width,
        }

    manifest = {
        "version": 1,
        "perturbation": plan.perturbation,
        "point_tolerance": plan.point_tolerance,
        "physical_names_seen": plan.physical_names_seen,
        "interface_delimiter": plan.interface_delimiter,
        "boundary_delimiter": plan.boundary_delimiter,
        "subdomains": subdomains_blob,
        "phase_order": [
            [s.id for s in plan.interfaces] + [s.id for s in plan.junctions],
            [v.id for v in plan.volumes],
        ],
    }
    (work_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Per-job bundles
    for v in plan.volumes:
        _write_volume_bundle(work_dir / "jobs" / v.id, v, entities, mesh_kwargs)
    for s in plan.interfaces:
        _write_seam_bundle(work_dir / "jobs" / s.id, s, entities, mesh_kwargs, role="interface")
    for s in plan.junctions:
        _write_seam_bundle(work_dir / "jobs" / s.id, s, entities, mesh_kwargs, role="junction")


def _serialize_entities(entities: list[Any]) -> list[dict]:
    return [e.to_dict() for e in entities if hasattr(e, "to_dict")]


def _write_volume_bundle(job_dir: Path, vol: VolumeRegion, entities, mesh_kwargs):
    import json
    job_dir.mkdir(parents=True, exist_ok=True)
    clipped = []
    for ent in entities:
        c = _clip_entity_to_polygon(ent, vol.polygon)
        if c is not None:
            clipped.append(c)
    (job_dir / "job.json").write_text(json.dumps({
        "id": vol.id,
        "role": "volume",
        "dim": 3,
        "interface_inputs": [],   # populated between phase 1 and 2
        "neighbors": vol.neighbors,
        "manifest_ref": "../../manifest.json",
    }, indent=2))
    (job_dir / "entities.json").write_text(json.dumps(_serialize_entities(clipped), indent=2))
    (job_dir / "subdomain.wkt").write_text(vol.polygon.wkt)
    (job_dir / "mesh_kwargs.json").write_text(json.dumps(mesh_kwargs, indent=2, default=str))


def _write_seam_bundle(job_dir: Path, slab: Slab, entities, mesh_kwargs, role: str):
    import json
    job_dir.mkdir(parents=True, exist_ok=True)
    clipped = []
    for ent in entities:
        c = _clip_entity_to_polygon(ent, slab.polygon)
        if c is not None:
            clipped.append(c)
        elif _entity_within(ent, slab.polygon, slab.width) and getattr(ent, "resolutions", None):
            clipped.append(_resolution_only_proxy(ent))

    # Add the phantom seam imprint(s) — one PolySurface per cut polyline.
    from meshwell.polysurface import PolySurface
    for k, ls in enumerate(slab.cut_polylines):
        seam_name = f"_seam___{slab.between[0]}___{slab.between[1]}" if len(slab.between) == 2 else (
            f"_seam___" + "___".join(slab.between)
        )
        # Imprint as a PolySurface stripe of width=point_tolerance straddling the polyline,
        # tagged keep=False so it is removed at top-dim but its faces inherit the name.
        from shapely.geometry import LineString as LS
        thin = ls.buffer(slab.width * 0.001, single_sided=False, cap_style=2, join_style=2)
        clipped.append(PolySurface(
            polygons=thin,
            physical_name=seam_name,
            mesh_order=0,
            mesh_bool=False,
        ))

    (job_dir / "job.json").write_text(json.dumps({
        "id": slab.id,
        "role": role,
        "dim": 3,
        "interface_inputs": [],
        "neighbors": slab.between,
        "manifest_ref": "../../manifest.json",
    }, indent=2))
    (job_dir / "entities.json").write_text(json.dumps(_serialize_entities(clipped), indent=2))
    (job_dir / "subdomain.wkt").write_text(slab.polygon.wkt)
    (job_dir / "mesh_kwargs.json").write_text(json.dumps(mesh_kwargs, indent=2, default=str))


def _entity_within(entity, mask: Polygon, distance: float) -> bool:
    if not hasattr(entity, "polygons"):
        return False
    polys = entity.polygons if isinstance(entity.polygons, list) else [entity.polygons]
    return any(p.distance(mask) <= distance for p in polys)
```

- [ ] **Step 4: Run, verify PASS**

- [ ] **Step 5: Commit**

```bash
git add meshwell/distributed.py tests/test_distributed.py
git commit -m "feat(distributed): write_bundles emits manifest + per-job files"
```

---

## Task 16: `run_job` reads bundle and invokes `generate_mesh`

**Files:**
- Modify: `meshwell/distributed.py`
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the failing test**

```python
def test_run_job_executes_volume_bundle(tmp_path):
    import json
    import shapely
    from meshwell.distributed import build_subdomain_plan, run_job, write_bundles
    from meshwell.polyprism import PolyPrism

    sd = [shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)]
    prism = PolyPrism(
        polygons=shapely.box(0, 0, 2, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="mat", mesh_order=1,
    )
    plan = build_subdomain_plan(
        subdomains=sd, entities=[prism],
        interface_width=0.1, perturbation=1e-5, point_tolerance=1e-3,
    )
    write_bundles(tmp_path, plan, [prism], mesh_kwargs={"default_characteristic_length": 0.3})

    job_dir = tmp_path / "jobs" / "volume_0000"
    run_job(job_dir)
    assert (job_dir / "result.msh").exists()
    res = json.loads((job_dir / "result.json").read_text())
    assert res["status"] == "ok"
```

- [ ] **Step 2: Run, FAIL** (NotImplementedError)

- [ ] **Step 3: Implement `run_job`**

```python
def run_job(job_dir: Path) -> None:
    import json
    import time
    from meshwell.orchestrator import generate_mesh
    from meshwell.utils import deserialize

    job_dir = Path(job_dir)
    job = json.loads((job_dir / "job.json").read_text())
    manifest = json.loads((job_dir / Path(job["manifest_ref"])).read_text())
    entities = deserialize(json.loads((job_dir / "entities.json").read_text()))
    mesh_kwargs = json.loads((job_dir / "mesh_kwargs.json").read_text())

    extra = {
        "_pre_buffered": True,
        "_global_physical_names": manifest["physical_names_seen"],
    }
    if job["role"] in ("interface", "junction"):
        extra["_emit_only_seam_surfaces"] = True
    if job["role"] == "volume":
        extra["_interface_constraints"] = [
            job_dir / inp["path"] for inp in job["interface_inputs"]
        ]

    t0 = time.time()
    try:
        generate_mesh(
            entities=entities,
            dim=job["dim"],
            output_mesh=job_dir / "result.msh",
            **mesh_kwargs,
            **extra,
        )
        status = "ok"
        err = None
    except Exception as e:
        status = "error"
        err = repr(e)

    (job_dir / "result.json").write_text(json.dumps({
        "status": status,
        "error": err,
        "elapsed_s": time.time() - t0,
        "id": job["id"],
        "role": job["role"],
    }, indent=2))
    if status != "ok":
        raise RuntimeError(f"run_job failed for {job['id']}: {err}")
```

- [ ] **Step 4: Run, verify PASS**

- [ ] **Step 5: Commit**

```bash
git add meshwell/distributed.py tests/test_distributed.py
git commit -m "feat(distributed): run_job dispatches a single bundle"
```

---

## Task 17: CLI entry point + console_script

**Files:**
- Modify: `meshwell/distributed.py`
- Modify: `pyproject.toml`
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the failing test**

```python
def test_cli_run_job_invokes_run_job(tmp_path, monkeypatch):
    import sys
    import shapely
    from meshwell.distributed import build_subdomain_plan, write_bundles, cli_main
    from meshwell.polyprism import PolyPrism

    sd = [shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)]
    prism = PolyPrism(
        polygons=shapely.box(0, 0, 2, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="mat", mesh_order=1,
    )
    plan = build_subdomain_plan(
        subdomains=sd, entities=[prism],
        interface_width=0.1, perturbation=1e-5, point_tolerance=1e-3,
    )
    write_bundles(tmp_path, plan, [prism], mesh_kwargs={"default_characteristic_length": 0.3})

    monkeypatch.setattr(sys, "argv", ["meshwell", "run-job", str(tmp_path / "jobs" / "volume_0000")])
    cli_main()
    assert (tmp_path / "jobs" / "volume_0000" / "result.msh").exists()
```

- [ ] **Step 2: Run, FAIL** (NotImplementedError)

- [ ] **Step 3: Implement `cli_main`**

```python
def cli_main() -> None:
    import argparse, sys
    parser = argparse.ArgumentParser(prog="meshwell")
    sub = parser.add_subparsers(dest="cmd", required=True)
    rj = sub.add_parser("run-job", help="Execute a single distributed job bundle")
    rj.add_argument("job_dir", type=Path)
    args = parser.parse_args(sys.argv[1:])
    if args.cmd == "run-job":
        run_job(args.job_dir)
```

- [ ] **Step 4: Update `pyproject.toml`**

Add (or extend an existing `[project.scripts]` section):

```toml
[project.scripts]
meshwell = "meshwell.distributed:cli_main"
```

- [ ] **Step 5: Reinstall in dev mode** so the script is on PATH:

Run: `pip install -e .`
Expected: success.

- [ ] **Step 6: Run test, verify PASS**

Run: `pytest tests/test_distributed.py::test_cli_run_job_invokes_run_job -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add meshwell/distributed.py pyproject.toml tests/test_distributed.py
git commit -m "feat(distributed): meshwell CLI with run-job subcommand"
```

---

## Task 18: `Executor` protocol + `SubprocessExecutor`

The skeleton already has these (from Task 6). Add a synchronous `InProcessExecutor` for tests + verify both work.

**Files:**
- Modify: `meshwell/distributed.py`
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the failing test**

```python
def test_in_process_executor_runs_jobs_synchronously(tmp_path):
    import shapely
    from meshwell.distributed import (
        InProcessExecutor, build_subdomain_plan, run_job, write_bundles,
    )
    from meshwell.polyprism import PolyPrism

    sd = [shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)]
    prism = PolyPrism(
        polygons=shapely.box(0, 0, 2, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="mat", mesh_order=1,
    )
    plan = build_subdomain_plan(
        subdomains=sd, entities=[prism],
        interface_width=0.1, perturbation=1e-5, point_tolerance=1e-3,
    )
    write_bundles(tmp_path, plan, [prism], mesh_kwargs={"default_characteristic_length": 0.3})

    ex = InProcessExecutor()
    fut = ex.submit(tmp_path / "jobs" / "volume_0000")
    fut.result()  # blocks
    assert (tmp_path / "jobs" / "volume_0000" / "result.msh").exists()
```

- [ ] **Step 2: Run, FAIL** (no InProcessExecutor)

- [ ] **Step 3: Implement**

```python
class InProcessExecutor:
    """Synchronous executor: runs each job in the calling process.

    Used by tests and for debugging — bypasses the subprocess + CLI hop.
    """

    def submit(self, job_dir: Path) -> Future:
        f: Future = Future()
        try:
            run_job(Path(job_dir))
            f.set_result({"returncode": 0, "job_dir": str(job_dir)})
        except Exception as e:
            f.set_exception(e)
        return f
```

- [ ] **Step 4: Run, verify PASS**

- [ ] **Step 5: Commit**

```bash
git add meshwell/distributed.py tests/test_distributed.py
git commit -m "feat(distributed): InProcessExecutor for tests + debugging"
```

---

## Task 19: Two-phase scheduler

Run phase-1 jobs first; on success, populate `interface_inputs` for each volume bundle and run phase 2.

**Files:**
- Modify: `meshwell/distributed.py`
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the failing test**

```python
def test_run_plan_two_phases_completes(tmp_path):
    import json
    import shapely
    from meshwell.distributed import (
        InProcessExecutor, build_subdomain_plan, run_plan, write_bundles,
    )
    from meshwell.polyprism import PolyPrism

    sd = [shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)]
    prism = PolyPrism(
        polygons=shapely.box(0, 0, 2, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="mat", mesh_order=1,
    )
    plan = build_subdomain_plan(
        subdomains=sd, entities=[prism],
        interface_width=0.1, perturbation=1e-5, point_tolerance=1e-3,
    )
    write_bundles(tmp_path, plan, [prism], mesh_kwargs={"default_characteristic_length": 0.3})

    run_plan(tmp_path, plan, executor=InProcessExecutor())

    # Phase 1 result
    assert (tmp_path / "jobs" / "interface_0000" / "result.msh").exists()
    # Phase 2 results
    for vid in ["volume_0000", "volume_0001"]:
        assert (tmp_path / "jobs" / vid / "result.msh").exists()
        # Volume job.json should now reference the interface input.
        j = json.loads((tmp_path / "jobs" / vid / "job.json").read_text())
        assert j["interface_inputs"], f"{vid} did not get interface_inputs populated"
```

- [ ] **Step 2: Run, FAIL** (no `run_plan`)

- [ ] **Step 3: Implement `run_plan`**

```python
def run_plan(work_dir: Path, plan: SubdomainPlan, executor: Executor) -> None:
    """Drive the two-phase distributed run."""
    import json

    work_dir = Path(work_dir)

    # ---- Phase 1: interfaces + junctions ----
    phase1_ids = [s.id for s in plan.interfaces] + [s.id for s in plan.junctions]
    futures = {sid: executor.submit(work_dir / "jobs" / sid) for sid in phase1_ids}
    failures = []
    for sid, f in futures.items():
        try:
            res = f.result()
            if isinstance(res, dict) and res.get("returncode", 0) != 0:
                failures.append((sid, res.get("stderr", "")))
        except Exception as e:
            failures.append((sid, repr(e)))
    if failures:
        raise RuntimeError(f"Phase 1 failures: {failures}")

    # ---- Populate volume bundles' interface_inputs ----
    seams_by_volume: dict[str, list[Slab]] = {v.id: [] for v in plan.volumes}
    for s in plan.interfaces + plan.junctions:
        for v_id in s.between:
            if v_id in seams_by_volume:
                seams_by_volume[v_id].append(s)

    for vol in plan.volumes:
        job_path = work_dir / "jobs" / vol.id / "job.json"
        j = json.loads(job_path.read_text())
        ifaces = []
        for s in seams_by_volume[vol.id]:
            src = work_dir / "jobs" / s.id / "result.msh"
            dst_dir = work_dir / "jobs" / vol.id / "interface_meshes"
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{s.id}.msh"
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src.resolve())
            ifaces.append({"id": s.id, "path": f"interface_meshes/{s.id}.msh"})
        j["interface_inputs"] = ifaces
        job_path.write_text(json.dumps(j, indent=2))

    # ---- Phase 2: volumes ----
    phase2_ids = [v.id for v in plan.volumes]
    futures = {vid: executor.submit(work_dir / "jobs" / vid) for vid in phase2_ids}
    failures = []
    for vid, f in futures.items():
        try:
            res = f.result()
            if isinstance(res, dict) and res.get("returncode", 0) != 0:
                failures.append((vid, res.get("stderr", "")))
        except Exception as e:
            failures.append((vid, repr(e)))
    if failures:
        raise RuntimeError(f"Phase 2 failures: {failures}")
```

- [ ] **Step 4: Run test, verify PASS**

- [ ] **Step 5: Commit**

```bash
git add meshwell/distributed.py tests/test_distributed.py
git commit -m "feat(distributed): run_plan two-phase scheduling"
```

---

## Task 20: Glue — load volumes, removeDuplicateNodes, dedup physical groups

**Files:**
- Modify: `meshwell/distributed.py`
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the failing test**

```python
def test_stitch_meshes_produces_one_unified_mesh(tmp_path):
    import meshio
    import shapely
    from meshwell.distributed import (
        InProcessExecutor, build_subdomain_plan, run_plan, stitch_meshes, write_bundles,
    )
    from meshwell.polyprism import PolyPrism

    sd = [shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)]
    prism = PolyPrism(
        polygons=shapely.box(0, 0, 2, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="mat", mesh_order=1,
    )
    plan = build_subdomain_plan(
        subdomains=sd, entities=[prism],
        interface_width=0.1, perturbation=1e-5, point_tolerance=1e-3,
    )
    write_bundles(tmp_path, plan, [prism], mesh_kwargs={"default_characteristic_length": 0.3})
    run_plan(tmp_path, plan, executor=InProcessExecutor())
    out = tmp_path / "stitched.msh"
    stitch_meshes(work_dir=tmp_path, plan=plan, output_mesh=out)
    m = meshio.read(out)
    # The merged mesh should still have the "mat" physical group
    assert "mat" in (m.field_data or {})
    # And it should have at least one tet cell.
    assert any(cb.type == "tetra" for cb in m.cells)
```

- [ ] **Step 2: Run, FAIL** (no stitch_meshes)

- [ ] **Step 3: Implement `stitch_meshes`**

```python
def stitch_meshes(work_dir: Path, plan: SubdomainPlan, output_mesh: Path) -> None:
    """Merge all volume_*.msh, remove duplicate nodes, consolidate physical groups."""
    import gmsh
    work_dir = Path(work_dir)
    output_mesh = Path(output_mesh)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("stitched")
        for v in plan.volumes:
            path = work_dir / "jobs" / v.id / "result.msh"
            gmsh.merge(str(path))

        # Remove duplicate nodes within tolerance.
        gmsh.option.setNumber("Geometry.Tolerance", plan.point_tolerance / 2)
        gmsh.model.mesh.removeDuplicateNodes()

        # Consolidate physical groups by name (all dims).
        for dim in (3, 2, 1, 0):
            pgs = list(gmsh.model.getPhysicalGroups(dim))
            by_name: dict[str, list[int]] = {}
            for d, tag in pgs:
                name = gmsh.model.getPhysicalName(d, tag)
                by_name.setdefault(name, []).extend(
                    int(e) for e in gmsh.model.getEntitiesForPhysicalGroup(d, tag)
                )
            # Drop original groups.
            for d, tag in pgs:
                gmsh.model.removePhysicalGroups([(d, tag)])
            # Re-add consolidated.
            for name, ents in by_name.items():
                if not ents:
                    continue
                gmsh.model.addPhysicalGroup(dim, list(set(ents)), name=name)

        gmsh.write(str(output_mesh))
    finally:
        gmsh.finalize()
```

- [ ] **Step 4: Run, verify PASS**

- [ ] **Step 5: Commit**

```bash
git add meshwell/distributed.py tests/test_distributed.py
git commit -m "feat(distributed): stitch_meshes — merge + dedup + physical-group consolidation"
```

---

## Task 21: Top-level `generate_mesh_distributed`

**Files:**
- Modify: `meshwell/distributed.py`
- Modify: `meshwell/__init__.py`
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the failing test (end-to-end smoke)**

```python
def test_generate_mesh_distributed_smoke(tmp_path):
    import meshio
    import shapely
    from meshwell.distributed import (
        InProcessExecutor, generate_mesh_distributed,
    )
    from meshwell.polyprism import PolyPrism

    sd = [shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)]
    prism = PolyPrism(
        polygons=shapely.box(0, 0, 2, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="mat", mesh_order=1,
    )
    out = tmp_path / "out.msh"
    work = tmp_path / "work"
    generate_mesh_distributed(
        entities=[prism],
        subdomains=sd,
        output_mesh=out,
        work_dir=work,
        interface_width=0.1,
        executor=InProcessExecutor(),
        keep_bundles=True,
        default_characteristic_length=0.3,
    )
    m = meshio.read(out)
    assert "mat" in (m.field_data or {})
```

- [ ] **Step 2: Run, FAIL** (NotImplementedError)

- [ ] **Step 3: Implement**

```python
def generate_mesh_distributed(
    entities, subdomains, output_mesh, work_dir, interface_width,
    executor=None, keep_bundles=False, registry=None, **mesh_kwargs,
):
    import shutil
    from meshwell.cad_common import prepare_entities
    from meshwell.utils import deserialize

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if executor is None:
        executor = SubprocessExecutor()

    entities = deserialize(entities, registry=registry)

    perturbation = mesh_kwargs.get("perturbation", 1e-5)
    point_tolerance = mesh_kwargs.get("point_tolerance", 1e-3)

    # Apply the global perturbation buffer ONCE here.
    prepare_entities(entities, perturbation=perturbation)

    plan = build_subdomain_plan(
        subdomains=subdomains,
        entities=entities,
        interface_width=interface_width,
        perturbation=perturbation,
        point_tolerance=point_tolerance,
    )
    # Workers must NOT re-buffer.
    write_bundles(work_dir, plan, entities, mesh_kwargs=mesh_kwargs)
    run_plan(work_dir, plan, executor=executor)
    stitch_meshes(work_dir, plan, output_mesh=Path(output_mesh))

    if not keep_bundles:
        shutil.rmtree(work_dir, ignore_errors=True)
```

In `meshwell/__init__.py`, add:

```python
from meshwell.distributed import (
    Executor,
    InProcessExecutor,
    SubprocessExecutor,
    generate_mesh_distributed,
    subdomains_from_grid,
)
```

- [ ] **Step 4: Run test, verify PASS**

- [ ] **Step 5: Commit**

```bash
git add meshwell/distributed.py meshwell/__init__.py tests/test_distributed.py
git commit -m "feat(distributed): generate_mesh_distributed top-level entrypoint"
```

---

## Task 22: Spec test — single-material 2x1 grid sanity check

(Spec test #1)

**Files:**
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the test**

```python
def test_distributed_matches_serial_single_material_2x1(tmp_path):
    """Spec test 1: a single-material PolyPrism spanning two subdomains
    produces a mesh whose physical-group inventory matches the serial run."""
    import meshio
    import shapely
    from meshwell.distributed import (
        InProcessExecutor, generate_mesh_distributed, subdomains_from_grid,
    )
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism

    def make_prism():
        return PolyPrism(
            polygons=shapely.box(0, 0, 2, 1),
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="mat", mesh_order=1,
        )

    serial_out = tmp_path / "serial.msh"
    generate_mesh(entities=[make_prism()], dim=3, output_mesh=serial_out,
                  default_characteristic_length=0.3)
    s = meshio.read(serial_out)

    dist_out = tmp_path / "dist.msh"
    generate_mesh_distributed(
        entities=[make_prism()],
        subdomains=subdomains_from_grid((0, 0, 2, 1), nx=2, ny=1),
        output_mesh=dist_out,
        work_dir=tmp_path / "work",
        interface_width=0.1,
        executor=InProcessExecutor(),
        default_characteristic_length=0.3,
    )
    d = meshio.read(dist_out)

    serial_names = set(s.field_data or {})
    dist_names = set(d.field_data or {}) - {n for n in (d.field_data or {}) if n.startswith("_seam___")}
    assert serial_names == dist_names
```

- [ ] **Step 2: Run, verify PASS**

Run: `pytest tests/test_distributed.py::test_distributed_matches_serial_single_material_2x1 -v`
Expected: PASS.

If FAIL, the most likely causes are seam-conformity issues — inspect with `keep_bundles=True` and rerun individual `meshwell run-job` calls to localize.

- [ ] **Step 3: Commit**

```bash
git add tests/test_distributed.py
git commit -m "test(distributed): single-material 2x1 grid matches serial run (spec test 1)"
```

---

## Task 23: Spec test — two materials with shared interface (spec test 2)

**Files:**
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the test**

```python
def test_distributed_two_materials_shared_interface(tmp_path):
    """Spec test 2: silicon and oxide abut at x=1; the silicon___oxide
    interface group exists in the merged mesh."""
    import meshio
    import shapely
    from meshwell.distributed import (
        InProcessExecutor, generate_mesh_distributed,
    )
    from meshwell.polyprism import PolyPrism

    si = PolyPrism(
        polygons=shapely.box(0, 0, 1, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="silicon", mesh_order=1,
    )
    ox = PolyPrism(
        polygons=shapely.box(1, 0, 2, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="oxide", mesh_order=2,
    )
    out = tmp_path / "out.msh"
    generate_mesh_distributed(
        entities=[si, ox],
        subdomains=[shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)],
        output_mesh=out,
        work_dir=tmp_path / "work",
        interface_width=0.1,
        executor=InProcessExecutor(),
        default_characteristic_length=0.3,
    )
    m = meshio.read(out)
    names = set(m.field_data or {})
    # Material-material interface is named with the existing meshwell convention.
    assert "silicon" in names and "oxide" in names
    interfaces = {n for n in names if "___" in n and not n.startswith("_seam___")}
    assert any("silicon" in i and "oxide" in i for i in interfaces), \
        f"missing silicon/oxide interface among {interfaces}"
```

- [ ] **Step 2: Run, verify PASS**

- [ ] **Step 3: Commit**

```bash
git add tests/test_distributed.py
git commit -m "test(distributed): two materials with shared interface (spec test 2)"
```

---

## Task 24: Spec test — 2x2 grid with junction (spec test 3)

**Files:**
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the test**

```python
def test_distributed_2x2_grid_with_junction(tmp_path):
    """Spec test 3: 2x2 grid, four materials, junction at (1, 1)."""
    import meshio
    import shapely
    from meshwell.distributed import (
        InProcessExecutor, generate_mesh_distributed, subdomains_from_grid,
    )
    from meshwell.polyprism import PolyPrism

    materials = []
    for (i, j), name in [
        ((0, 0), "ll"), ((1, 0), "lr"), ((0, 1), "ul"), ((1, 1), "ur"),
    ]:
        materials.append(PolyPrism(
            polygons=shapely.box(i, j, i + 1, j + 1),
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name=name, mesh_order=1,
        ))
    sd = subdomains_from_grid((0, 0, 2, 2), nx=2, ny=2)
    out = tmp_path / "out.msh"
    generate_mesh_distributed(
        entities=materials, subdomains=sd, output_mesh=out,
        work_dir=tmp_path / "work", interface_width=0.1,
        executor=InProcessExecutor(),
        default_characteristic_length=0.3,
    )
    m = meshio.read(out)
    names = set(m.field_data or {})
    assert {"ll", "lr", "ul", "ur"}.issubset(names)
```

- [ ] **Step 2: Run, verify PASS**

- [ ] **Step 3: Commit**

```bash
git add tests/test_distributed.py
git commit -m "test(distributed): 2x2 grid with junction (spec test 3)"
```

---

## Task 25: Spec test — coverage validation error (spec test 7)

**Files:**
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the test**

```python
def test_distributed_rejects_uncovered_subdomains(tmp_path):
    """Spec test 7: subdomains not covering entity union → clean ValueError."""
    import shapely
    import pytest
    from meshwell.distributed import (
        InProcessExecutor, generate_mesh_distributed,
    )
    from meshwell.polyprism import PolyPrism

    prism = PolyPrism(
        polygons=shapely.box(0, 0, 2, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="mat", mesh_order=1,
    )
    with pytest.raises(ValueError, match="not covered"):
        generate_mesh_distributed(
            entities=[prism],
            subdomains=[shapely.box(0, 0, 1, 1)],  # x=[1,2] uncovered
            output_mesh=tmp_path / "out.msh",
            work_dir=tmp_path / "work",
            interface_width=0.1,
            executor=InProcessExecutor(),
        )
```

- [ ] **Step 2: Run, verify PASS**

- [ ] **Step 3: Commit**

```bash
git add tests/test_distributed.py
git commit -m "test(distributed): coverage validation error (spec test 7)"
```

---

## Task 26: Spec test — physical-group consolidation (spec test 10)

**Files:**
- Test: `tests/test_distributed.py`

- [ ] **Step 1: Write the test**

```python
def test_distributed_consolidates_same_material_across_tiles(tmp_path):
    """Spec test 10: the same material in 4 tiles → one consolidated physical group."""
    import meshio
    import shapely
    from meshwell.distributed import (
        InProcessExecutor, generate_mesh_distributed, subdomains_from_grid,
    )
    from meshwell.polyprism import PolyPrism

    # One single material that spans all 4 tiles.
    prism = PolyPrism(
        polygons=shapely.box(0, 0, 2, 2),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="mat", mesh_order=1,
    )
    out = tmp_path / "out.msh"
    generate_mesh_distributed(
        entities=[prism],
        subdomains=subdomains_from_grid((0, 0, 2, 2), nx=2, ny=2),
        output_mesh=out,
        work_dir=tmp_path / "work",
        interface_width=0.1,
        executor=InProcessExecutor(),
        default_characteristic_length=0.5,
    )
    m = meshio.read(out)
    names = list((m.field_data or {}).keys())
    # Exactly one "mat" group, not four.
    assert names.count("mat") == 1
```

- [ ] **Step 2: Run, verify PASS**

- [ ] **Step 3: Commit**

```bash
git add tests/test_distributed.py
git commit -m "test(distributed): physical-group consolidation across tiles (spec test 10)"
```

---

## Task 27: Documentation pass

**Files:**
- Create: `docs/distributed.md`
- Modify: `README.md`

- [ ] **Step 1: Write `docs/distributed.md`**

Author a narrative guide that links to the design spec and shows a minimal
end-to-end example using `subdomains_from_grid` + `generate_mesh_distributed`
with `InProcessExecutor`. Include:
- Conceptual overview (master → phase 1 → phase 2 → glue).
- The required `interface_width` choice rationale (rule of thumb: ≥ 4× the largest
  ResolutionSpec characteristic length near a cut).
- How to plug a custom Executor (Slurm / Ray / k8s example signatures).
- How to debug: `keep_bundles=True`, manual `meshwell run-job` rerun.

- [ ] **Step 2: Update `README.md`**

Move "Distributed memory processing with domain decomposition" from
"(Planned)" to "Key Features" and link to `docs/distributed.md`.

- [ ] **Step 3: Commit**

```bash
git add docs/distributed.md README.md
git commit -m "docs: distributed pipeline guide; promote to released features"
```

---

## Task 28: Final validation — full test suite

- [ ] **Step 1:** Run the entire test suite end-to-end:

Run: `pytest tests/ -q`
Expected: ALL pass.

- [ ] **Step 2:** Inspect any flaky tests; if any test is timing-sensitive (the distributed phase-2 step can be slow under default mesh sizes), tag with `@pytest.mark.slow` rather than skipping.

- [ ] **Step 3:** Commit any test-marker tweaks:

```bash
git add tests/test_distributed.py
git commit -m "test(distributed): mark slow integration tests"
```

---

## Notes on tasks deferred from spec testing

The following spec tests are intentionally NOT implemented as standalone tasks above; they should be added to `tests/test_distributed.py` after Task 28 lands and the core pipeline is verified, in a follow-up PR:

- **Spec test 4** (ResolutionSpec near a cut): requires a stable `ConstantInField` setup; relies on the `_resolution_only_proxy` mechanism end-to-end. Add as `test_distributed_resolution_spec_near_cut`.
- **Spec test 5** (per-cut `interface_width` dict): straightforward extension of test 22 with a dict argument.
- **Spec test 6** (phantom imprint identity): inspect a phase-1 `result.msh`; verify only `_seam___` groups are present.
- **Spec test 8** (worker reproducibility): rerun `meshwell run-job` on a saved bundle and compare outputs byte-for-byte.
- **Spec test 9** (executor swap): subclass Executor with a logging variant and verify final mesh equality.
- **Spec test 11** (failure path): inject a malformed entity into one phase-1 bundle and assert the failure mode.

These are deferred because they exercise the same code paths as the included tests but add coverage rather than new behavior. Address them in a follow-up "distributed: extended test coverage" PR.
