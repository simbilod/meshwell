# BoundaryLayerResolutionSpec Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `ResolutionSpec` that, attached to a 1D physical name, configures a gmsh **BoundaryLayer** mesh field on those curves (anisotropic graded layer, optional quads).

**Architecture:** A new `ResolutionSpec` subclass whose `apply()` builds a gmsh `BoundaryLayer` field, registers it via `setAsBoundaryLayer`, and returns `None` so it never enters the `Min`/background size field. It reuses the existing per-name dispatch in `_mesh_entity.add_refinement_fields_to_model` (which already resolves a physical name to its dim-1 tags and already filters `None` returns). Two small hardening changes: a `None` guard on the global-spec apply path, and a clear error when boundary layers and StructuredSweep are requested on the same model (the sweep kernel sets `Mesh.MeshOnlyEmpty=1`, which suppresses BL generation).

**Tech Stack:** Python, pydantic (v1-style `class Config` where needed), gmsh Python API, pytest via `uv run pytest`.

## Global Constraints

- Repo: `meshwell` (`/home/simbil/Github/laser/meshwell`). Branch: `feat/boundary-layer-resolution` (stacked on `feat/structured-sweeps`). All paths relative to repo root.
- Read the spec first: `docs/superpowers/specs/2026-09-06-boundary-layer-resolution-design.md`.
- Fan points are OUT OF SCOPE for this plan (they need named 0D point support; separate follow-up PR). Do NOT add `fan_points`.
- gmsh BoundaryLayer option names (verified against the installed gmsh): `Size`, `Thickness`, `Ratio`, `Quads`, `SizeFar`, `NbLayers`, `IntersectMetrics`, `AnisoMax`, `Beta`, `CurvesList`. Register the field with `model.mesh.field.setAsBoundaryLayer(field_tag)`.
- Optional params (`size_far`, `nb_layers`, `aniso_max`, `beta`) are pushed to gmsh ONLY when not `None`, so gmsh defaults hold otherwise.
- `apply()` must return `None` (the BL field is registered via `setAsBoundaryLayer`, not combined into the `Min` field).
- Follow existing code style: `from __future__ import annotations` is already at the top of `resolution.py`; Google-style docstrings; pydantic `Field(...)` for constraints as the existing specs do.
- pytest auto-injects coverage flags in this repo; do NOT pass `-p no:cov`.
- Commit after every green test cycle. Message style: `feat(bl): …`, `test(bl): …`, `docs(bl): …`. End each commit message with the two trailer lines:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01SbBeTxf5wuJfSiu6KHiXHx`
- Run the full suite `uv run pytest -q` before the final commit of Tasks 3 and 4 (they touch shared `mesh.py`); Tasks 1–2 run their own test file plus `uv run pytest -q` once before commit.

---

### Task 1: `BoundaryLayerResolutionSpec` class and `apply()`

**Files:**
- Modify: `meshwell/resolution.py` (append a new class near the other spec classes)
- Test: `tests/test_boundary_layer.py` (create)

**Interfaces:**
- Consumes: the existing `ResolutionSpec` base in `meshwell/resolution.py`; the per-name apply dispatch in `meshwell/_mesh_entity.py:add_refinement_fields_to_model`, which calls `spec.apply(model=<gmsh model>, entities_mass_dict=<{tag: mass}>, restrict_to_str=..., restrict_to_tags=...)` and appends the return value only when it is not `None` (`_mesh_entity.py:481`).
- Produces (used by Tasks 3–4):
  - `BoundaryLayerResolutionSpec(size: float, thickness: float, ratio: float = 1.0, quads: bool = False, size_far: float | None = None, nb_layers: int | None = None, intersect_metrics: bool = False, aniso_max: float | None = None, beta: float | None = None)` with `apply_to: Literal["curves"] = "curves"`.
  - `.apply(model, entities_mass_dict, **kwargs) -> None` — creates a gmsh `BoundaryLayer` field on `list(entities_mass_dict.keys())` and registers it via `setAsBoundaryLayer`; returns `None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_boundary_layer.py
import shapely

from meshwell.orchestrator import generate_mesh
from meshwell.polysurface import PolySurface
from meshwell.resolution import BoundaryLayerResolutionSpec


def _sheet():
    return [PolySurface(polygons=shapely.box(0, 0, 1, 1), physical_name="sheet", mesh_order=1)]


def _first_offwall_offset(mesh, xlo=0.2, xhi=0.8):
    """Smallest positive y among interior-x nodes = first boundary-layer row off y=0."""
    pts = mesh.points[:, :2]
    interior = pts[(pts[:, 0] > xlo) & (pts[:, 0] < xhi) & (pts[:, 1] > 1e-9)]
    return float(interior[:, 1].min())


def test_boundary_layer_grades_and_quads(tmp_path):
    mesh = generate_mesh(
        entities=_sheet(),
        dim=2,
        output_mesh=str(tmp_path / "bl.msh"),
        default_characteristic_length=0.1,
        resolution_specs={
            "sheet___None": [
                BoundaryLayerResolutionSpec(
                    size=0.01, thickness=0.08, ratio=1.3, quads=True
                )
            ],
        },
    )
    # quad cells are present in the layer
    n_quad = sum(cb.data.shape[0] for cb in mesh.cells if cb.type == "quad")
    assert n_quad > 0
    # first off-wall node sits ~size (0.01) away -- far below the default CL (0.1)
    first = _first_offwall_offset(mesh)
    assert 0.004 < first < 0.02


def test_boundary_layer_triangles(tmp_path):
    mesh = generate_mesh(
        entities=_sheet(),
        dim=2,
        output_mesh=str(tmp_path / "bl_tri.msh"),
        default_characteristic_length=0.1,
        resolution_specs={
            "sheet___None": [
                BoundaryLayerResolutionSpec(
                    size=0.01, thickness=0.08, ratio=1.3, quads=False
                )
            ],
        },
    )
    n_quad = sum(cb.data.shape[0] for cb in mesh.cells if cb.type == "quad")
    assert n_quad == 0
    first = _first_offwall_offset(mesh)
    assert 0.004 < first < 0.02


def test_two_boundary_layers_distinct_params(tmp_path):
    entities = [
        PolySurface(polygons=shapely.box(0, 0, 4, 1), physical_name="lower", mesh_order=2),
        PolySurface(polygons=shapely.box(0, 1, 4, 2), physical_name="upper", mesh_order=1),
    ]
    mesh = generate_mesh(
        entities=entities,
        dim=2,
        output_mesh=str(tmp_path / "bl2.msh"),
        default_characteristic_length=0.2,
        resolution_specs={
            "lower___None": [BoundaryLayerResolutionSpec(size=0.005, thickness=0.05)],
            "upper___None": [BoundaryLayerResolutionSpec(size=0.02, thickness=0.1)],
        },
    )
    pts = mesh.points[:, :2]
    # near the bottom wall (y=0), first row ~0.005
    bot = pts[(pts[:, 0] > 1) & (pts[:, 0] < 3) & (pts[:, 1] > 1e-9)]
    assert bot[:, 1].min() < 0.012
    # near the top wall (y=2), first row ~0.02
    top = pts[(pts[:, 0] > 1) & (pts[:, 0] < 3) & (pts[:, 1] < 2 - 1e-9)]
    assert (2.0 - top[:, 1].max()) < 0.04


def test_boundary_layer_validation():
    import pytest

    with pytest.raises(Exception):
        BoundaryLayerResolutionSpec(size=-1.0, thickness=0.05)
    with pytest.raises(Exception):
        BoundaryLayerResolutionSpec(size=0.01, thickness=0.0)
    with pytest.raises(Exception):
        BoundaryLayerResolutionSpec(size=0.01, thickness=0.05, ratio=0.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_boundary_layer.py -q`
Expected: FAIL with `ImportError: cannot import name 'BoundaryLayerResolutionSpec'`.

- [ ] **Step 3: Implement the class**

In `meshwell/resolution.py`, append after `StructuredExtrusionResolutionSpec` (the file already imports `Field`, `Literal`, and defines `ResolutionSpec`):

```python
class BoundaryLayerResolutionSpec(ResolutionSpec):
    """Configure a gmsh BoundaryLayer field on a 1D physical name.

    Attach (via ``resolution_specs``) to any dim-1 physical group -- a
    PolyLine name, an interface ``a___b``, or a boundary ``a___None``. Grows
    an anisotropic graded layer off those curves. Registered with gmsh via
    ``setAsBoundaryLayer`` (NOT combined into the Min background size field),
    hence ``apply`` returns ``None``. Full gmsh passthrough; optional params
    are only pushed when set. Fan points are a separate follow-up (they need
    named 0D point support).
    """

    apply_to: Literal["curves"] = "curves"
    size: float = Field(gt=0)  # gmsh Size: first-layer normal size
    thickness: float = Field(gt=0)  # gmsh Thickness: total layer thickness
    ratio: float = Field(default=1.0, ge=1.0)  # gmsh Ratio: geometric growth
    quads: bool = False  # gmsh Quads (0/1)
    size_far: float | None = None  # gmsh SizeFar
    nb_layers: int | None = None  # gmsh NbLayers
    intersect_metrics: bool = False  # gmsh IntersectMetrics
    aniso_max: float | None = None  # gmsh AnisoMax
    beta: float | None = None  # gmsh Beta

    def apply(self, model: Any, entities_mass_dict, **_kwargs) -> None:
        """Create a gmsh BoundaryLayer field on the given curves.

        Registered via setAsBoundaryLayer; returns None so it is not merged
        into the Min background size field.
        """
        if not entities_mass_dict:
            return None
        f = model.mesh.field.add("BoundaryLayer")
        model.mesh.field.setNumbers(f, "CurvesList", list(entities_mass_dict.keys()))
        model.mesh.field.setNumber(f, "Size", self.size)
        model.mesh.field.setNumber(f, "Thickness", self.thickness)
        model.mesh.field.setNumber(f, "Ratio", self.ratio)
        model.mesh.field.setNumber(f, "Quads", 1 if self.quads else 0)
        if self.size_far is not None:
            model.mesh.field.setNumber(f, "SizeFar", self.size_far)
        if self.nb_layers is not None:
            model.mesh.field.setNumber(f, "NbLayers", self.nb_layers)
        model.mesh.field.setNumber(
            f, "IntersectMetrics", 1 if self.intersect_metrics else 0
        )
        if self.aniso_max is not None:
            model.mesh.field.setNumber(f, "AnisoMax", self.aniso_max)
        if self.beta is not None:
            model.mesh.field.setNumber(f, "Beta", self.beta)
        model.mesh.field.setAsBoundaryLayer(f)
        return None
```

`Any` is already imported in `resolution.py` (used by `ConstantInField.apply`). If a type-checker complains, confirm the import at the top of the file.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_boundary_layer.py -q` — Expected: PASS (4 tests).
Then `uv run pytest -q` — Expected: no regressions.

- [ ] **Step 5: Commit**

```bash
git add meshwell/resolution.py tests/test_boundary_layer.py
git commit -m "feat(bl): BoundaryLayerResolutionSpec sets a gmsh BoundaryLayer field on 1D names"
```

---

### Task 2: `None`-guard on the global-spec apply path

**Files:**
- Modify: `meshwell/mesh.py` (the global-spec loop inside the resolution-application method — currently appends `apply()` returns without a `None` check)
- Test: `tests/test_boundary_layer.py` (append)

**Interfaces:**
- Consumes: `BoundaryLayerResolutionSpec` (Task 1), whose `apply()` returns `None` (and, in the global path, is called with an empty `entities_mass_dict`, so it returns `None`).
- Produces: no new API; makes a `None`-returning spec safe to place under the global (`None`) key.

**Background:** the per-entity path already guards `if field_idx is not None` (`_mesh_entity.py:481`), but the global-spec loop does not:

```python
        # Handle Global Specs (key is None)
        if None in resolution_specs:
            for spec in resolution_specs[None]:
                field_index = spec.apply(
                    self.model_manager.model, {}, restrict_to_tags=None
                )
                refinement_field_indices.append(field_index)
```

Appending `None` here would later break the `Min` field (`setNumbers(..., "FieldsList", [..., None])`).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_boundary_layer.py
def test_global_none_returning_spec_does_not_crash(tmp_path):
    """A None-returning spec under the global (None) key must not corrupt the Min field."""
    mesh = generate_mesh(
        entities=_sheet(),
        dim=2,
        output_mesh=str(tmp_path / "glob.msh"),
        default_characteristic_length=0.1,
        resolution_specs={
            None: [BoundaryLayerResolutionSpec(size=0.01, thickness=0.05)],
        },
    )
    assert mesh is not None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_boundary_layer.py::test_global_none_returning_spec_does_not_crash -q`
Expected: FAIL (a gmsh error from the `Min` field receiving `None` in its `FieldsList`).

- [ ] **Step 3: Add the guard**

In `meshwell/mesh.py`, in the global-spec loop shown above, replace the unconditional append with a guarded one:

```python
        # Handle Global Specs (key is None)
        if None in resolution_specs:
            for spec in resolution_specs[None]:
                field_index = spec.apply(
                    self.model_manager.model, {}, restrict_to_tags=None
                )
                if field_index is not None:
                    refinement_field_indices.append(field_index)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_boundary_layer.py -q` — Expected: PASS (5 tests).
Then `uv run pytest -q` — Expected: no regressions.

- [ ] **Step 5: Commit**

```bash
git add meshwell/mesh.py tests/test_boundary_layer.py
git commit -m "fix(bl): skip None-returning specs in the global-spec apply path"
```

---

### Task 3: Clear error when boundary layers and StructuredSweep share a model

**Files:**
- Modify: `meshwell/mesh.py` (`process_geometry`, at the sweep-discovery block)
- Test: `tests/test_boundary_layer.py` (append)

**Interfaces:**
- Consumes: `discover_sweeps()` result (`discovered`, already computed in `process_geometry`), `resolution_specs`, and `BoundaryLayerResolutionSpec` (Task 1).
- Produces: a `ValueError` (message contains "boundary layer") when a model has both discovered sweep groups and at least one `BoundaryLayerResolutionSpec`. Rationale: the sweep stamping kernel sets `Mesh.MeshOnlyEmpty=1`, which suppresses gmsh boundary-layer generation, so combining them silently yields a bad mesh.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_boundary_layer.py
import pytest

from meshwell.resolution import StructuredSweepResolutionSpec
from meshwell.structured.sweep import StructuredSweep


def _two_boxes():
    return [
        PolySurface(polygons=shapely.box(0, 0, 4, 1), physical_name="lower", mesh_order=2),
        PolySurface(polygons=shapely.box(0, 1, 4, 2), physical_name="upper", mesh_order=1),
    ]


def test_boundary_layer_with_sweep_raises(tmp_path):
    with pytest.raises(ValueError, match="boundary layer"):
        generate_mesh(
            entities=_two_boxes(),
            sweeps=[StructuredSweep(name="qw", on="lower___upper", thickness={"upper": 0.4})],
            dim=2,
            output_mesh=str(tmp_path / "conflict.msh"),
            default_characteristic_length=0.5,
            resolution_specs={
                "qw": [StructuredSweepResolutionSpec(tangential=1.0, normal={"upper": 2})],
                "lower___None": [BoundaryLayerResolutionSpec(size=0.01, thickness=0.05)],
            },
        )
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_boundary_layer.py::test_boundary_layer_with_sweep_raises -q`
Expected: FAIL — no `ValueError` raised (either it meshes, or fails with a different error).

- [ ] **Step 3: Add the conflict check**

In `meshwell/mesh.py`, in `process_geometry`, immediately after `discovered = discover_sweeps()` and before `if discovered:`, insert the check:

```python
        discovered = discover_sweeps()
        if discovered:
            from meshwell.resolution import BoundaryLayerResolutionSpec

            has_bl = any(
                isinstance(spec, BoundaryLayerResolutionSpec)
                for specs in (resolution_specs or {}).values()
                for spec in specs
            )
            if has_bl:
                raise ValueError(
                    "Cannot combine a StructuredSweep with a boundary layer "
                    "(BoundaryLayerResolutionSpec) in the same model: the sweep "
                    "stamping kernel sets Mesh.MeshOnlyEmpty=1, which suppresses "
                    "gmsh boundary-layer generation. Mesh them in separate models."
                )
            specs_by_name = validate_sweep_pairing(discovered, resolution_specs)
            # ... (existing hook-composition code unchanged) ...
```

Keep all existing lines under `if discovered:` intact; only add the `has_bl` check at the top of that block. Note `resolution_specs` values are lists of specs (see `resolution_specs.get(physical_name, [])` usage), so the double comprehension is correct.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_boundary_layer.py -q` — Expected: PASS (6 tests).
Then the full suite `uv run pytest -q` — Expected: no regressions (in particular the `tests/test_sweep_*.py` sweep tests still pass, since they carry no BL spec).

- [ ] **Step 5: Commit**

```bash
git add meshwell/mesh.py tests/test_boundary_layer.py
git commit -m "feat(bl): raise a clear error when a boundary layer and StructuredSweep share a model"
```

---

### Task 4: Docs notebook

**Files:**
- Create: `docs/25_boundary_layer.py` (percent-format notebook like `docs/23_structured.py` / `docs/24_structured_sweeps.py`)

**Interfaces:** none produced; documentation only.

- [ ] **Step 1: Write `docs/25_boundary_layer.py`**

Read `docs/24_structured_sweeps.py` first to match its percent-format conventions (cell markers, imports, `plot2D` usage, local output filenames). Write a notebook with these cells:

```python
# %% [markdown]
# # Boundary-layer meshing with BoundaryLayerResolutionSpec
#
# `BoundaryLayerResolutionSpec` attaches a gmsh **BoundaryLayer** field to any
# 1D physical name -- a PolyLine, an interface `a___b`, or a boundary
# `a___None` -- growing an anisotropic, geometrically graded layer off those
# curves. It complements `StructuredSweep` (notebook 24): the sweep stamps an
# exact tensor grid at straight interfaces, while this delegates to gmsh's
# boundary-layer mesher (curved walls, quads, 3D-capable), trading exactness
# for generality.

# %%
import shapely

from meshwell.orchestrator import generate_mesh
from meshwell.polysurface import PolySurface
from meshwell.resolution import BoundaryLayerResolutionSpec
from meshwell.visualization import plot2D

# %% [markdown]
# ## A graded quad boundary layer on a region's outer boundary

# %%
sheet = PolySurface(polygons=shapely.box(0, 0, 1, 1), physical_name="sheet", mesh_order=1)
bl_mesh = generate_mesh(
    entities=[sheet],
    dim=2,
    output_mesh="boundary_layer_quad.msh",
    default_characteristic_length=0.1,
    resolution_specs={
        "sheet___None": [
            BoundaryLayerResolutionSpec(size=0.01, thickness=0.08, ratio=1.3, quads=True)
        ],
    },
)
n_quad = sum(cb.data.shape[0] for cb in bl_mesh.cells if cb.type == "quad")
print(f"quad cells in the layer: {n_quad}")
plot2D(bl_mesh, title="Quad boundary layer on sheet___None", wireframe=True)

# %% [markdown]
# ## Triangle layer (quads=False)

# %%
tri_mesh = generate_mesh(
    entities=[PolySurface(polygons=shapely.box(0, 0, 1, 1), physical_name="sheet", mesh_order=1)],
    dim=2,
    output_mesh="boundary_layer_tri.msh",
    default_characteristic_length=0.1,
    resolution_specs={
        "sheet___None": [
            BoundaryLayerResolutionSpec(size=0.01, thickness=0.08, ratio=1.3, quads=False)
        ],
    },
)
plot2D(tri_mesh, title="Triangle boundary layer", wireframe=True)

# %% [markdown]
# ## Two independent boundary layers with different parameters
#
# Each BoundaryLayerResolutionSpec becomes its own gmsh boundary-layer field,
# so different curves can carry different first-layer sizes / thicknesses.

# %%
two = generate_mesh(
    entities=[
        PolySurface(polygons=shapely.box(0, 0, 4, 1), physical_name="lower", mesh_order=2),
        PolySurface(polygons=shapely.box(0, 1, 4, 2), physical_name="upper", mesh_order=1),
    ],
    dim=2,
    output_mesh="boundary_layer_two.msh",
    default_characteristic_length=0.2,
    resolution_specs={
        "lower___None": [BoundaryLayerResolutionSpec(size=0.005, thickness=0.05)],
        "upper___None": [BoundaryLayerResolutionSpec(size=0.02, thickness=0.1)],
    },
)
plot2D(two, title="Two boundary layers, different sizes", wireframe=True)

# %% [markdown]
# ## Notes and limitations
#
# - Full gmsh passthrough: `size`, `thickness`, `ratio`, `quads`, and the
#   optional `size_far`, `nb_layers`, `intersect_metrics`, `aniso_max`, `beta`.
# - `quads=True` emits quadrilateral layers; to split them into triangles
#   downstream, call `gmsh.model.mesh.splitQuadrangles()`.
# - A boundary layer and a `StructuredSweep` cannot share a model (the sweep
#   sets `Mesh.MeshOnlyEmpty=1`, which suppresses boundary-layer generation) --
#   meshwell raises a clear error if both are requested.
# - Fan points (wrapping a layer around a sharp convex vertex) are a planned
#   follow-up; they need named 0D point support first.
```

- [ ] **Step 2: Verify the notebook executes**

Run: `MPLBACKEND=Agg uv run python docs/25_boundary_layer.py`
Expected: exits 0, prints the quad-cell count, writes its example `.msh` files. (The `FigureCanvasAgg is non-interactive` warning from `plt.show()` under Agg is expected and harmless.)

- [ ] **Step 3: Commit**

```bash
git add docs/25_boundary_layer.py
git commit -m "docs(bl): boundary-layer resolution notebook"
```

---

## Self-review notes (already applied)

- **Spec coverage:** the class + full passthrough params (Task 1), `apply()` self-registration returning `None` (Task 1), the global-path `None` guard edge case (Task 2), the BL+StructuredSweep conflict edge case (Task 3), and the notebook (Task 4) each map to the spec's sections. Fan points are explicitly deferred per the spec and are NOT in this plan.
- **Type consistency:** `BoundaryLayerResolutionSpec` field names (`size`, `thickness`, `ratio`, `quads`, `size_far`, `nb_layers`, `intersect_metrics`, `aniso_max`, `beta`) are identical across Task 1's definition, Task 3's `isinstance` check, and Task 4's usage. `apply()` returns `None` everywhere it is described.
- **Discriminating assertions:** the near-wall offset assertions (`0.004 < first < 0.02`) work because the boundary-layer first-row size (0.01) is 10x smaller than the default characteristic length (0.1); without the field applied, the nearest node would be ~0.1 away and the assertion would fail — so the test actually proves the field took effect, not just that a mesh was produced.
- **Known soft spot:** exact near-wall node offsets depend on the gmsh version's boundary-layer node placement. If the offset-window assertions prove brittle on this gmsh build, widen the window (e.g. `first < 0.03`) while keeping it strictly below the default CL (0.1) so the test still discriminates "field applied" from "not applied" — do not remove the discriminator.
```
