# Structured PolyPrism Clean Rewrite — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the data-only foundation of the clean structured-polyprism rewrite — spec dataclasses, plan-stage validator, and the `PolyPrism(structured=True)` API hook — on a fresh branch off `main`. No CAD / mesh integration; those are Phase 2 / Phase 3.

**Architecture:** Pure-Python data layer. `meshwell/structured/spec.py` defines `StructuredExtrusionResolutionSpec`, `Slab`, `OverlapPair`, `StructuredPlan`. `meshwell/structured/plan.py` walks a user entity list, expands `structured=True` polyprisms into slabs, validates the overlap rule (Policy B: volumetric overlap allowed iff z-extents match exactly AND `n_layers` agrees), computes a per-slab `face_partition`, and returns a frozen `StructuredPlan`. `meshwell/polyprism.py` gains a `structured: bool = False` keyword that activates the new pipeline.

**Tech Stack:** Python 3.12, dataclasses (frozen for value types), shapely 2.x (polygonize, unary_union), pydantic 2.x (for the spec, matching the existing `ResolutionSpec` base), pytest.

**Spec reference:** `docs/superpowers/specs/2026-05-15-structured-polyprism-clean-design.md`

---

## File Structure

**Create:**
- `meshwell/structured/__init__.py` — public re-exports.
- `meshwell/structured/spec.py` — `StructuredExtrusionResolutionSpec`, `Slab`, `OverlapPair`, `StructuredPlan`, `StructuredOverlapError`, `StructuredBufferTaperError`.
- `meshwell/structured/plan.py` — `build_plan(entities) -> StructuredPlan` plus its private helpers.
- `tests/structured/__init__.py` — empty pytest package marker.
- `tests/structured/test_spec.py` — dataclass instantiation + pydantic validation tests.
- `tests/structured/test_plan.py` — planner tests with handcrafted entity fixtures.
- `tests/structured/test_polyprism_structured_flag.py` — API contract tests for `PolyPrism(structured=True)`.

**Modify:**
- `meshwell/polyprism.py` — add `structured: bool = False` kw, validate it against the resolutions list, store it on the instance. Leave the existing `__new__` dispatcher and `n_layers=` parameter untouched (the new branch starts from `main`, where they don't exist — see Task 1).

**Untouched in Phase 1:**
- `meshwell/orchestrator.py`, `meshwell/mesh.py`, `meshwell/cad_occ.py`, `meshwell/cad_gmsh.py`.
- `meshwell/resolution.py` (the new spec subclasses `pydantic.BaseModel` directly, not `ResolutionSpec`, because it doesn't share `apply_to` semantics — see Task 3).

---

## Task 1: Cut the new branch from main; cherry-pick spec + spike artifacts

**Files:**
- New branch: `feat/structured-clean` off `origin/main`.
- Cherry-pick three commits from `feat/structured`:
  - `2d7ddf7` docs(structured): clean-rewrite design + Phase-0 spike
  - `a1c4cc9` docs(structured): swap BOP fuse for piece-by-piece sewing in the spec
  - `27022a6` docs(structured): drop the per-phantom fuse step entirely

- [ ] **Step 1: Confirm working tree state is safe**

Run: `git status --short`

Expected: only the untracked items already present (`.claude/`, `OCCT/`, `docs/distributed_example_work/`, `repro.structured_slabs.json`, `scripts/debug_*.py`, `tests/repro.structured_slabs.json`, `tests/test_overlapping_facets_structured.py`, `tests/test_structured_complex_scene.py`) and the modified `tests/test_backend_cross_compare.py`. No new staged changes.

- [ ] **Step 2: Stash modified + untracked Python files (preserves them across the switch)**

Run:
```bash
git stash push --include-untracked -m "phase1-prep: stash before branch switch" -- \
  tests/test_backend_cross_compare.py
```

Expected: `Saved working directory and index state On feat/structured: phase1-prep: stash before branch switch`.

Note: the OCCT/, gmsh/, docs/distributed_example_work/ directories are gitignored or already-untracked-elsewhere — they survive the switch without explicit stashing. Only the tracked-modified file needs stashing.

- [ ] **Step 3: Cut the new branch from origin/main**

Run:
```bash
git fetch origin main
git switch -c feat/structured-clean origin/main
```

Expected: `Switched to a new branch 'feat/structured-clean'` and branch is at `43abfaa` (or whatever `origin/main` HEAD is).

- [ ] **Step 4: Cherry-pick the three spec/spike commits**

Run:
```bash
git cherry-pick 2d7ddf7 a1c4cc9 27022a6
```

Expected: three commits applied cleanly with `[feat/structured-clean <new-hash>]` lines. No conflicts (the docs/superpowers/ paths don't exist on main).

- [ ] **Step 5: Verify the artifacts landed**

Run:
```bash
git log --oneline -4
ls docs/superpowers/specs/ docs/superpowers/spikes/
```

Expected: three cherry-pick hashes on top of `origin/main` HEAD, plus presence of both `2026-05-15-structured-polyprism-clean-design.md` and `discrete_entity_displaced_vertex.py`.

- [ ] **Step 6: Pop the stash so test_backend_cross_compare.py modifications come back if needed**

Run: `git stash pop`

Expected: `tests/test_backend_cross_compare.py` shows as modified again. (If we later decide we don't want those mods on the new branch, drop them; for now keep them available.)

---

## Task 2: Create the structured package skeleton

**Files:**
- Create: `meshwell/structured/__init__.py`
- Create: `tests/structured/__init__.py`

- [ ] **Step 1: Write the failing test that imports the new package**

Create `tests/structured/__init__.py` as an empty file.

Create `tests/structured/test_package_smoke.py`:

```python
"""Smoke test: the structured package exists and is importable."""
from __future__ import annotations


def test_structured_package_importable():
    import meshwell.structured  # noqa: F401


def test_structured_public_exports_present():
    """__init__ re-exports the public spec dataclass."""
    from meshwell.structured import StructuredExtrusionResolutionSpec

    assert StructuredExtrusionResolutionSpec is not None
```

- [ ] **Step 2: Run the smoke tests; expect ImportError**

Run: `pytest tests/structured/test_package_smoke.py -v`

Expected: both tests fail with `ModuleNotFoundError: No module named 'meshwell.structured'`.

- [ ] **Step 3: Create the package init**

Create `meshwell/structured/__init__.py`:

```python
"""Clean structured-polyprism pipeline.

Public surface:

- :class:`StructuredExtrusionResolutionSpec` -- attach to a
  ``PolyPrism(structured=True)`` to specify per-z-interval layer counts.

CAD-stage and mesh-stage internals (``Slab``, ``StructuredPlan``,
``OverlapPair``, the planner) live in submodules and are loaded on demand
by the orchestrator. They are not part of the public surface for end
users in Phase 1.
"""
from __future__ import annotations

from meshwell.structured.spec import StructuredExtrusionResolutionSpec

__all__ = ["StructuredExtrusionResolutionSpec"]
```

- [ ] **Step 4: Run smoke tests; expect ImportError on spec**

Run: `pytest tests/structured/test_package_smoke.py -v`

Expected: `test_structured_package_importable` PASSES; `test_structured_public_exports_present` fails with `ImportError: cannot import name 'StructuredExtrusionResolutionSpec'` (spec.py doesn't exist yet). This is the expected handoff to Task 3.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/__init__.py tests/structured/__init__.py tests/structured/test_package_smoke.py
git commit -m "$(cat <<'EOF'
feat(structured): create empty package skeleton

Lands the meshwell.structured package with an __init__ that will
re-export StructuredExtrusionResolutionSpec once spec.py is added in
the next task. Adds tests/structured/ as the test package root.
EOF
)"
```

---

## Task 3: Implement `StructuredExtrusionResolutionSpec`

**Files:**
- Create: `meshwell/structured/spec.py`
- Test: `tests/structured/test_spec.py`

The spec is a pydantic `BaseModel` for validation (matching the existing `meshwell/resolution.py:18` style), not a `dataclass`. It does NOT subclass `ResolutionSpec` because it doesn't have the `apply_to` / `min_mass` / sharing semantics; it's a distinct concept and forcing inheritance would couple unrelated APIs.

- [ ] **Step 1: Write the failing tests**

Create `tests/structured/test_spec.py`:

```python
"""Tests for meshwell.structured.spec dataclasses + validators."""
from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_spec_minimal_valid():
    from meshwell.structured.spec import StructuredExtrusionResolutionSpec

    spec = StructuredExtrusionResolutionSpec(n_layers=[3, 5])
    assert spec.n_layers == [3, 5]
    assert spec.recombine is False


def test_spec_recombine_true():
    from meshwell.structured.spec import StructuredExtrusionResolutionSpec

    spec = StructuredExtrusionResolutionSpec(n_layers=[2], recombine=True)
    assert spec.recombine is True


def test_spec_rejects_empty_n_layers():
    from meshwell.structured.spec import StructuredExtrusionResolutionSpec

    with pytest.raises(ValidationError, match="n_layers"):
        StructuredExtrusionResolutionSpec(n_layers=[])


def test_spec_rejects_non_positive_layer_count():
    from meshwell.structured.spec import StructuredExtrusionResolutionSpec

    with pytest.raises(ValidationError, match="positive"):
        StructuredExtrusionResolutionSpec(n_layers=[3, 0, 4])

    with pytest.raises(ValidationError, match="positive"):
        StructuredExtrusionResolutionSpec(n_layers=[-1])


def test_spec_is_hashable():
    """Frozen pydantic models are hashable; we use them as dict keys downstream."""
    from meshwell.structured.spec import StructuredExtrusionResolutionSpec

    a = StructuredExtrusionResolutionSpec(n_layers=[2])
    b = StructuredExtrusionResolutionSpec(n_layers=[2])
    # Pydantic equality on identical fields.
    assert a == b
```

- [ ] **Step 2: Run tests; expect ImportError**

Run: `pytest tests/structured/test_spec.py -v`

Expected: every test fails with `ModuleNotFoundError: No module named 'meshwell.structured.spec'`.

- [ ] **Step 3: Implement the spec**

Create `meshwell/structured/spec.py`:

```python
"""Data model for the clean structured-polyprism pipeline.

Phase 1 ships only ``StructuredExtrusionResolutionSpec`` and the CAD-stage
``Slab`` / ``OverlapPair`` / ``StructuredPlan`` dataclasses. The PhantomMap
(Layer B) and StructuredMeshPlan (mesh-stage) land in Phase 2 / Phase 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from shapely.geometry import MultiPolygon, Polygon


class StructuredExtrusionResolutionSpec(BaseModel):
    """Per-z-interval layer counts for a structured ``PolyPrism``.

    Attached via ``PolyPrism(..., structured=True, resolutions=[spec])``.

    Attributes:
        n_layers: One positive integer per z-interval of the owning
            ``PolyPrism``. Length must equal ``len(buffers) - 1`` where
            ``buffers`` is the prism's z-keys dict. Enforced at planner
            time (not here, because the spec has no reference to the
            owning entity).
        recombine: When True, the slab volume is meshed with hex
            elements (gmsh element type 5) instead of wedges
            (element type 6).
    """

    model_config = ConfigDict(frozen=True)

    n_layers: list[int] = Field(
        ..., min_length=1, description="positive layer counts, one per z-interval"
    )
    recombine: bool = False

    @field_validator("n_layers")
    @classmethod
    def _all_positive(cls, v: list[int]) -> list[int]:
        for i, n in enumerate(v):
            if n <= 0:
                raise ValueError(
                    f"n_layers[{i}] = {n}: layer counts must be positive integers"
                )
        return v


class StructuredOverlapError(ValueError):
    """Raised when two structured slabs volumetrically overlap with
    mismatched z-extents (Policy B rejects all overlap unless z-extents
    match exactly within tolerance).
    """


class StructuredBufferTaperError(ValueError):
    """Raised when ``PolyPrism(structured=True)`` is used with non-zero
    buffers (tapered geometry). Structured mode requires uniform extrusion.
    """


@dataclass
class Slab:
    """One structured-polyprism z-interval, CAD-stage data only.

    Mesh parameters (``n_layers``, ``recombine``) are NOT stored here -
    they live on the resolution spec and are resolved in a second pass
    at mesh time. This keeps Slab a pure geometry+identity record.
    """

    footprint: "Polygon | MultiPolygon"
    zlo: float
    zhi: float
    physical_name: tuple[str, ...]
    source_index: int
    z_interval_index: int
    mesh_order: float
    identify_arcs: bool = False
    min_arc_points: int = 4
    arc_tolerance: float = 1e-3
    fragment_fuzzy_value: float | None = None
    # Populated by compute_face_partition (default: one piece = the whole footprint).
    face_partition: list["Polygon"] = field(default_factory=list)


@dataclass(frozen=True)
class OverlapPair:
    """Record of a Policy-B-resolved volumetric overlap.

    The winner slab is in ``StructuredPlan.slabs``; the loser was
    dropped during planning. Mesh stage uses this to verify the loser
    spec's n_layers agreed with the winner's.
    """

    winner_slab_index: int
    loser_source_index: int
    loser_z_interval_index: int
    z_extent: tuple[float, float]


@dataclass(frozen=True)
class StructuredPlan:
    """Frozen output of the planner; consumed by phantom + builder stages."""

    slabs: list[Slab]
    z_planes: list[float]
    overlaps: list[OverlapPair]
```

- [ ] **Step 4: Run spec tests**

Run: `pytest tests/structured/test_spec.py -v`

Expected: all 5 tests PASS.

- [ ] **Step 5: Run smoke tests to confirm public export works**

Run: `pytest tests/structured/test_package_smoke.py -v`

Expected: both smoke tests now PASS.

- [ ] **Step 6: Commit**

```bash
git add meshwell/structured/spec.py tests/structured/test_spec.py
git commit -m "$(cat <<'EOF'
feat(structured): StructuredExtrusionResolutionSpec + CAD-stage dataclasses

Adds the spec module with pydantic-validated
StructuredExtrusionResolutionSpec (n_layers: list[int], recombine: bool)
and the CAD-only Slab / OverlapPair / StructuredPlan dataclasses. The
spec does not subclass ResolutionSpec because it has no apply_to /
min_mass semantics; coupling them would mix unrelated concepts.

Slab carries no mesh parameters (n_layers, recombine) by design - those
are resolved from the owning entity's spec in a second pass at mesh
time. Phase 1 ends here for the data layer.
EOF
)"
```

---

## Task 4: Add `structured: bool` keyword to `PolyPrism`

**Files:**
- Modify: `meshwell/polyprism.py:44-128` (the `__init__` signature + body)
- Test: `tests/structured/test_polyprism_structured_flag.py`

On the new branch (cut from main), `PolyPrism` does NOT have the `__new__` dispatcher or the `n_layers=` parameter — that machinery only existed on `feat/structured`. So we're adding a fresh keyword to a clean `__init__`.

- [ ] **Step 1: Confirm the starting state**

Run: `grep -n "def __new__\|n_layers" meshwell/polyprism.py`

Expected: NO matches (we're on `feat/structured-clean`, which inherits `main`'s polyprism).

- [ ] **Step 2: Write the failing tests**

Create `tests/structured/test_polyprism_structured_flag.py`:

```python
"""API contract tests for ``PolyPrism(structured=True)``."""
from __future__ import annotations

import pytest
from shapely.geometry import Polygon


def _square() -> Polygon:
    return Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


def test_structured_flag_defaults_false():
    from meshwell.polyprism import PolyPrism

    p = PolyPrism(polygons=_square(), buffers={0.0: 0.0, 1.0: 0.0})
    assert p.structured is False


def test_structured_true_requires_resolution_spec():
    """structured=True without a StructuredExtrusionResolutionSpec raises."""
    from meshwell.polyprism import PolyPrism

    with pytest.raises(ValueError, match="StructuredExtrusionResolutionSpec"):
        PolyPrism(
            polygons=_square(),
            buffers={0.0: 0.0, 1.0: 0.0},
            structured=True,
        )


def test_structured_true_with_spec_succeeds():
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    spec = StructuredExtrusionResolutionSpec(n_layers=[3])
    p = PolyPrism(
        polygons=_square(),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[spec],
    )
    assert p.structured is True
    # Spec is preserved on the entity for the planner to retrieve.
    assert any(
        isinstance(r, StructuredExtrusionResolutionSpec) for r in p.resolutions
    )


def test_structured_true_rejects_tapered_buffers():
    """Non-uniform buffers raise StructuredBufferTaperError."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec
    from meshwell.structured.spec import StructuredBufferTaperError

    spec = StructuredExtrusionResolutionSpec(n_layers=[2])
    with pytest.raises(StructuredBufferTaperError, match="buffers"):
        PolyPrism(
            polygons=_square(),
            # Non-zero buffer at z=1.0 -> taper.
            buffers={0.0: 0.0, 1.0: 0.1},
            structured=True,
            resolutions=[spec],
        )


def test_structured_true_n_layers_length_matches_z_intervals():
    """spec.n_layers length must equal len(buffers) - 1."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    # Two z-intervals (3 z-keys), but spec provides 1 entry.
    spec = StructuredExtrusionResolutionSpec(n_layers=[3])
    with pytest.raises(ValueError, match="n_layers length"):
        PolyPrism(
            polygons=_square(),
            buffers={0.0: 0.0, 1.0: 0.0, 2.0: 0.0},
            structured=True,
            resolutions=[spec],
        )


def test_spec_on_non_structured_entity_warns():
    """Attaching the spec without structured=True emits a warning and ignores it."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    spec = StructuredExtrusionResolutionSpec(n_layers=[3])
    with pytest.warns(UserWarning, match="structured=True"):
        p = PolyPrism(
            polygons=_square(),
            buffers={0.0: 0.0, 1.0: 0.0},
            structured=False,
            resolutions=[spec],
        )
    assert p.structured is False
```

- [ ] **Step 3: Inspect the current PolyPrism signature**

Run: `grep -n "def __init__\|resolutions" meshwell/polyprism.py | head -10`

Expected: shows `__init__` starting around line 44 (or wherever it is on main). Note whether `resolutions=` is already a keyword (it should be, from the existing `ResolutionSpec` system).

If `resolutions=` is NOT present on PolyPrism's `__init__`, check the parent `GeometryEntity` — it's likely there. The tests above use `p.resolutions`; we need to ensure that attribute is reachable from PolyPrism.

Run: `grep -n "resolutions" meshwell/geometry_entity.py meshwell/polyprism.py | head -20`

Expected: at least one of them stores `self.resolutions = ...`. If not, add the kw to `PolyPrism.__init__` as part of this task (it's a small omission either way).

- [ ] **Step 4: Run the tests; expect TypeError (unknown kw `structured`)**

Run: `pytest tests/structured/test_polyprism_structured_flag.py -v`

Expected: every test fails with `TypeError: __init__() got an unexpected keyword argument 'structured'` (or similar).

- [ ] **Step 5: Add the `structured` kw + validation to `PolyPrism.__init__`**

Modify `meshwell/polyprism.py`. After the existing positional + keyword parameters in `__init__`, add `structured: bool = False`. After the parent `super().__init__(...)` call and the existing attribute assignments (around the block that sets `self.identify_arcs`, etc.), add the validation block:

```python
# Phase 1: structured-mode flag and validation.
self.structured = structured

# Find any StructuredExtrusionResolutionSpec instances on the entity.
# resolutions is set by the parent / kw plumbing; if it doesn't exist
# yet on this entity, treat as empty for the check.
from meshwell.structured.spec import (
    StructuredBufferTaperError,
    StructuredExtrusionResolutionSpec,
)

_resolutions = getattr(self, "resolutions", None) or []
_structured_specs = [
    r for r in _resolutions if isinstance(r, StructuredExtrusionResolutionSpec)
]

if self.structured:
    if not _structured_specs:
        raise ValueError(
            "PolyPrism(structured=True) requires a "
            "StructuredExtrusionResolutionSpec in resolutions=. None found."
        )
    if len(_structured_specs) > 1:
        raise ValueError(
            "PolyPrism(structured=True) accepts at most one "
            "StructuredExtrusionResolutionSpec; "
            f"{len(_structured_specs)} were attached."
        )
    spec = _structured_specs[0]
    n_intervals = len(buffers) - 1
    if len(spec.n_layers) != n_intervals:
        raise ValueError(
            f"StructuredExtrusionResolutionSpec.n_layers length "
            f"({len(spec.n_layers)}) must equal len(buffers) - 1 "
            f"({n_intervals}) for the owning PolyPrism."
        )
    # Buffers must be uniform (no taper) in structured mode.
    if not all(b == 0 for b in buffers.values()):
        raise StructuredBufferTaperError(
            "PolyPrism(structured=True) requires zero buffers "
            f"(uniform extrusion). Got non-zero entries: "
            f"{ {z: b for z, b in buffers.items() if b != 0} }"
        )
elif _structured_specs:
    import warnings

    warnings.warn(
        "StructuredExtrusionResolutionSpec attached to a PolyPrism with "
        "structured=False; the spec will be ignored. Pass "
        "structured=True to activate the structured pipeline.",
        UserWarning,
        stacklevel=2,
    )
```

If `resolutions=` is NOT already a keyword on `PolyPrism.__init__` (per Step 3's grep), add it to the signature with default `None`, then before the validation block insert:

```python
self.resolutions = list(resolutions) if resolutions else []
```

(Skip this if `super().__init__` already does it.)

- [ ] **Step 6: Run the tests; expect all to pass**

Run: `pytest tests/structured/test_polyprism_structured_flag.py -v`

Expected: all 6 tests PASS.

- [ ] **Step 7: Run the existing PolyPrism test suite to confirm no regressions**

Run: `pytest tests/test_buffers_prism.py tests/test_polyprism*.py 2>/dev/null tests/ -k "polyprism" -x -q`

Expected: green. If anything regresses, the new validation block is too aggressive — narrow it to only fire when `structured=True` or when the spec is actually present.

- [ ] **Step 8: Commit**

```bash
git add meshwell/polyprism.py tests/structured/test_polyprism_structured_flag.py
git commit -m "$(cat <<'EOF'
feat(structured): PolyPrism(structured=True) keyword + validation

Adds the public opt-in for the clean structured pipeline:
PolyPrism(..., structured=True, resolutions=[StructuredExtrusionResolutionSpec(...)]).

Validation:
- structured=True requires exactly one StructuredExtrusionResolutionSpec
  in resolutions; raises ValueError otherwise.
- spec.n_layers length must equal len(buffers) - 1.
- Tapered buffers (non-zero values) raise StructuredBufferTaperError.
- Attaching the spec to a non-structured prism warns and ignores it.

No CAD/mesh integration yet - structured=True is currently a no-op
beyond surfacing the flag for the planner (Task 5-8) to consume.
EOF
)"
```

---

## Task 5: Plan-stage helper — `gather_structured_entities`

**Files:**
- Modify: `meshwell/structured/plan.py` (new file)
- Test: `tests/structured/test_plan.py`

- [ ] **Step 1: Write the failing test**

Create `tests/structured/test_plan.py`:

```python
"""Tests for meshwell.structured.plan."""
from __future__ import annotations

import pytest
from shapely.geometry import Polygon


def _square(x=0, y=0, w=1, h=1) -> Polygon:
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def _structured(polygon, buffers, n_layers, name, mesh_order=1.0):
    """Test helper: build a structured PolyPrism with a spec attached."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    return PolyPrism(
        polygons=polygon,
        buffers=buffers,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=n_layers)],
        physical_name=name,
        mesh_order=mesh_order,
    )


def test_gather_filters_structured_entities():
    from meshwell.polyprism import PolyPrism
    from meshwell.structured.plan import gather_structured_entities

    s = _structured(_square(), {0.0: 0.0, 1.0: 0.0}, [3], "s")
    u = PolyPrism(
        polygons=_square(2, 2),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="u",
    )

    pairs = gather_structured_entities([u, s])
    assert len(pairs) == 1
    entity, spec, source_idx = pairs[0]
    assert entity is s
    assert spec.n_layers == [3]
    assert source_idx == 1  # s was at index 1 in the input list


def test_gather_returns_empty_when_no_structured_entities():
    from meshwell.polyprism import PolyPrism
    from meshwell.structured.plan import gather_structured_entities

    u = PolyPrism(
        polygons=_square(),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="u",
    )
    assert gather_structured_entities([u]) == []


def test_gather_preserves_source_indices_across_mixed_list():
    from meshwell.polyprism import PolyPrism
    from meshwell.structured.plan import gather_structured_entities

    u1 = PolyPrism(polygons=_square(), buffers={0.0: 0.0, 1.0: 0.0}, physical_name="u1")
    s1 = _structured(_square(2, 0), {0.0: 0.0, 1.0: 0.0}, [3], "s1")
    u2 = PolyPrism(polygons=_square(4, 0), buffers={0.0: 0.0, 1.0: 0.0}, physical_name="u2")
    s2 = _structured(_square(6, 0), {0.0: 0.0, 2.0: 0.0}, [4], "s2")

    pairs = gather_structured_entities([u1, s1, u2, s2])
    assert [p[2] for p in pairs] == [1, 3]
    assert [p[0].physical_name for p in pairs] == [("s1",), ("s2",)]
```

- [ ] **Step 2: Run; expect ImportError**

Run: `pytest tests/structured/test_plan.py::test_gather_filters_structured_entities -v`

Expected: `ModuleNotFoundError: No module named 'meshwell.structured.plan'`.

- [ ] **Step 3: Create plan.py with the helper**

Create `meshwell/structured/plan.py`:

```python
"""Plan stage for the clean structured-polyprism pipeline.

Public surface: ``build_plan(entities) -> StructuredPlan``. Private
helpers handle the pipeline steps (gather, expand, validate, partition).
"""
from __future__ import annotations

from typing import Any

from meshwell.structured.spec import StructuredExtrusionResolutionSpec


def gather_structured_entities(
    entities: list[Any],
) -> list[tuple[Any, StructuredExtrusionResolutionSpec, int]]:
    """Return ``(entity, spec, source_index)`` for every structured prism.

    A structured entity is one with ``structured=True`` AND exactly one
    ``StructuredExtrusionResolutionSpec`` in its ``resolutions``. The
    validation at construction time guarantees this when both conditions
    hold, so we just retrieve here.
    """
    out: list[tuple[Any, StructuredExtrusionResolutionSpec, int]] = []
    for idx, ent in enumerate(entities):
        if not getattr(ent, "structured", False):
            continue
        resolutions = getattr(ent, "resolutions", None) or []
        specs = [
            r for r in resolutions if isinstance(r, StructuredExtrusionResolutionSpec)
        ]
        if len(specs) != 1:
            # PolyPrism construction enforces this; defensive check.
            continue
        out.append((ent, specs[0], idx))
    return out
```

- [ ] **Step 4: Run the gather tests; expect pass**

Run: `pytest tests/structured/test_plan.py::test_gather_filters_structured_entities tests/structured/test_plan.py::test_gather_returns_empty_when_no_structured_entities tests/structured/test_plan.py::test_gather_preserves_source_indices_across_mixed_list -v`

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/plan.py tests/structured/test_plan.py
git commit -m "$(cat <<'EOF'
feat(structured): plan.gather_structured_entities helper

Walks the user entity list, returns (entity, spec, source_index) for
every PolyPrism with structured=True. Source indices preserved across
the input list so downstream slabs can reference their owning entity
for the mesh-stage parameter lookup.
EOF
)"
```

---

## Task 6: Plan-stage helper — `expand_to_slabs`

**Files:**
- Modify: `meshwell/structured/plan.py` (add new function)
- Modify: `tests/structured/test_plan.py` (add tests)

- [ ] **Step 1: Add the failing tests**

Append to `tests/structured/test_plan.py`:

```python
def test_expand_single_interval():
    from meshwell.structured.plan import expand_to_slabs, gather_structured_entities

    s = _structured(_square(), {0.0: 0.0, 1.5: 0.0}, [4], "s", mesh_order=2.0)
    pairs = gather_structured_entities([s])
    slabs = expand_to_slabs(pairs)
    assert len(slabs) == 1
    slab = slabs[0]
    assert slab.zlo == 0.0
    assert slab.zhi == 1.5
    assert slab.physical_name == ("s",)
    assert slab.source_index == 0
    assert slab.z_interval_index == 0
    assert slab.mesh_order == 2.0


def test_expand_multi_interval():
    from meshwell.structured.plan import expand_to_slabs, gather_structured_entities

    s = _structured(_square(), {0.0: 0.0, 1.0: 0.0, 3.0: 0.0}, [2, 5], "s")
    pairs = gather_structured_entities([s])
    slabs = expand_to_slabs(pairs)
    assert len(slabs) == 2
    assert (slabs[0].zlo, slabs[0].zhi, slabs[0].z_interval_index) == (0.0, 1.0, 0)
    assert (slabs[1].zlo, slabs[1].zhi, slabs[1].z_interval_index) == (1.0, 3.0, 1)
    # Both refer to the same owning entity / source_index.
    assert slabs[0].source_index == slabs[1].source_index == 0


def test_expand_propagates_arc_settings():
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec
    from meshwell.structured.plan import expand_to_slabs, gather_structured_entities

    s = PolyPrism(
        polygons=_square(),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[3])],
        identify_arcs=True,
        min_arc_points=8,
        arc_tolerance=5e-4,
        physical_name="s",
    )
    pairs = gather_structured_entities([s])
    slabs = expand_to_slabs(pairs)
    assert len(slabs) == 1
    assert slabs[0].identify_arcs is True
    assert slabs[0].min_arc_points == 8
    assert slabs[0].arc_tolerance == 5e-4
```

- [ ] **Step 2: Run; expect ImportError on expand_to_slabs**

Run: `pytest tests/structured/test_plan.py -v -k expand`

Expected: 3 tests fail with `ImportError: cannot import name 'expand_to_slabs'`.

- [ ] **Step 3: Implement expand_to_slabs**

Append to `meshwell/structured/plan.py`:

```python
from itertools import pairwise

from meshwell.structured.spec import Slab


def expand_to_slabs(
    pairs: list[tuple[Any, StructuredExtrusionResolutionSpec, int]],
) -> list[Slab]:
    """One slab per (entity, z-interval) pair.

    n_layers / recombine are NOT stored on the slab - they live on the
    spec and are resolved at mesh time via (source_index, z_interval_index).
    """
    slabs: list[Slab] = []
    for entity, _spec, source_index in pairs:
        z_keys = list(entity.buffers.keys())
        mesh_order = entity.mesh_order if entity.mesh_order is not None else float("inf")
        # Footprint: keep as-is; planner doesn't normalize multipolygon here.
        # PolyPrism stores `polygons` as a single Polygon or list; coerce in build_plan
        # if necessary.
        footprint = entity.polygons
        for z_idx, (zlo, zhi) in enumerate(pairwise(z_keys)):
            slabs.append(
                Slab(
                    footprint=footprint,
                    zlo=float(zlo),
                    zhi=float(zhi),
                    physical_name=entity.physical_name,
                    source_index=source_index,
                    z_interval_index=z_idx,
                    mesh_order=mesh_order,
                    identify_arcs=getattr(entity, "identify_arcs", False),
                    min_arc_points=getattr(entity, "min_arc_points", 4),
                    arc_tolerance=getattr(entity, "arc_tolerance", 1e-3),
                )
            )
    return slabs
```

- [ ] **Step 4: Run; expect pass**

Run: `pytest tests/structured/test_plan.py -v -k expand`

Expected: 3 expand tests PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/plan.py tests/structured/test_plan.py
git commit -m "$(cat <<'EOF'
feat(structured): plan.expand_to_slabs - one slab per z-interval

Walks (entity, spec) pairs from gather_structured_entities and emits
one Slab per pairwise z-key interval, preserving source_index and
z_interval_index for later mesh-stage parameter lookup. Arc settings
(identify_arcs, min_arc_points, arc_tolerance) are propagated from the
owning entity.
EOF
)"
```

---

## Task 7: Plan-stage validator — `validate_and_resolve_overlap` (Policy B)

**Files:**
- Modify: `meshwell/structured/plan.py`
- Modify: `tests/structured/test_plan.py`

Policy B: volumetric overlap allowed iff z-extents match exactly within tolerance AND n_layers agrees in the overlap region. Lower mesh_order wins; tie-break by source_index. Mismatch raises.

- [ ] **Step 1: Add the failing tests**

Append to `tests/structured/test_plan.py`:

```python
def test_no_overlap_keeps_all_slabs():
    from meshwell.structured.plan import (
        expand_to_slabs,
        gather_structured_entities,
        validate_and_resolve_overlap,
    )

    s1 = _structured(_square(0, 0), {0.0: 0.0, 1.0: 0.0}, [3], "s1")
    s2 = _structured(_square(2, 0), {0.0: 0.0, 1.0: 0.0}, [3], "s2")  # disjoint xy
    slabs = expand_to_slabs(gather_structured_entities([s1, s2]))
    kept, overlaps = validate_and_resolve_overlap(slabs, entities=[s1, s2])
    assert len(kept) == 2
    assert overlaps == []


def test_valid_overlap_drops_loser_records_pair():
    """Same z-extent, same n_layers, footprints overlap: lower mesh_order wins."""
    from meshwell.structured.plan import (
        expand_to_slabs,
        gather_structured_entities,
        validate_and_resolve_overlap,
    )

    # Lower mesh_order (1.0) wins; higher (2.0) is dropped.
    s_lo = _structured(_square(0, 0, 4, 4), {0.0: 0.0, 1.0: 0.0}, [3], "lo", mesh_order=2.0)
    s_hi = _structured(_square(1, 1, 2, 2), {0.0: 0.0, 1.0: 0.0}, [3], "hi", mesh_order=1.0)
    slabs = expand_to_slabs(gather_structured_entities([s_lo, s_hi]))
    kept, overlaps = validate_and_resolve_overlap(slabs, entities=[s_lo, s_hi])
    # Only the winner (hi) survives.
    assert len(kept) == 1
    assert kept[0].physical_name == ("hi",)
    # The loser pair was recorded.
    assert len(overlaps) == 1
    op = overlaps[0]
    assert op.loser_source_index == 0  # s_lo's index in entities=
    assert op.z_extent == (0.0, 1.0)


def test_overlap_with_mismatched_z_extent_raises():
    """Footprints overlap but z-extents differ: Policy B rejects."""
    from meshwell.structured.plan import (
        expand_to_slabs,
        gather_structured_entities,
        validate_and_resolve_overlap,
    )
    from meshwell.structured.spec import StructuredOverlapError

    s_lo = _structured(_square(0, 0, 4, 4), {0.0: 0.0, 1.0: 0.0}, [3], "lo")
    s_hi = _structured(_square(1, 1, 2, 2), {0.5: 0.0, 1.5: 0.0}, [3], "hi")
    slabs = expand_to_slabs(gather_structured_entities([s_lo, s_hi]))
    with pytest.raises(StructuredOverlapError, match="z-extent"):
        validate_and_resolve_overlap(slabs, entities=[s_lo, s_hi])


def test_overlap_with_matching_z_but_mismatched_n_layers_raises():
    """Same z-extent, different n_layers: Policy B rejects."""
    from meshwell.structured.plan import (
        expand_to_slabs,
        gather_structured_entities,
        validate_and_resolve_overlap,
    )
    from meshwell.structured.spec import StructuredOverlapError

    s_a = _structured(_square(0, 0, 4, 4), {0.0: 0.0, 1.0: 0.0}, [3], "a")
    s_b = _structured(_square(1, 1, 2, 2), {0.0: 0.0, 1.0: 0.0}, [5], "b")  # 5 vs 3
    slabs = expand_to_slabs(gather_structured_entities([s_a, s_b]))
    with pytest.raises(StructuredOverlapError, match="n_layers"):
        validate_and_resolve_overlap(slabs, entities=[s_a, s_b])
```

- [ ] **Step 2: Run; expect ImportError**

Run: `pytest tests/structured/test_plan.py -v -k overlap`

Expected: 4 tests fail with `ImportError: cannot import name 'validate_and_resolve_overlap'`.

- [ ] **Step 3: Implement validate_and_resolve_overlap**

Append to `meshwell/structured/plan.py`:

```python
from meshwell.structured.spec import (
    OverlapPair,
    StructuredOverlapError,
)

_Z_TOL = 1e-9  # exact-match tolerance for z-extent comparison


def _z_extent_matches(a: Slab, b: Slab) -> bool:
    return abs(a.zlo - b.zlo) < _Z_TOL and abs(a.zhi - b.zhi) < _Z_TOL


def _footprints_overlap(a: Slab, b: Slab) -> bool:
    """True iff the two slab footprints share positive xy area."""
    inter = a.footprint.intersection(b.footprint)
    return (not inter.is_empty) and inter.area > 0


def _n_layers_of_slab(
    slab: Slab, entities: list[Any]
) -> int:
    """Look up n_layers for slab via (source_index, z_interval_index)."""
    ent = entities[slab.source_index]
    specs = [
        r
        for r in (getattr(ent, "resolutions", None) or [])
        if isinstance(r, StructuredExtrusionResolutionSpec)
    ]
    return specs[0].n_layers[slab.z_interval_index]


def validate_and_resolve_overlap(
    slabs: list[Slab],
    entities: list[Any],
) -> tuple[list[Slab], list[OverlapPair]]:
    """Apply Policy B: drop volumetric overlap losers, fail on mismatch.

    Returns (kept_slabs, overlap_pairs). Lower mesh_order wins; tie-break
    by source_index then z_interval_index for determinism.
    """
    # Sort by (mesh_order, source_index, z_interval_index): winners first.
    order = sorted(
        range(len(slabs)),
        key=lambda i: (slabs[i].mesh_order, slabs[i].source_index, slabs[i].z_interval_index),
    )

    kept_indices: list[int] = []
    overlaps: list[OverlapPair] = []
    # `kept_xy` indexed parallel to kept_indices: each kept slab keeps its winner status.
    for idx in order:
        slab = slabs[idx]
        dominated = False
        for k_idx in kept_indices:
            kept = slabs[k_idx]
            if not _footprints_overlap(kept, slab):
                continue
            # Footprints overlap; Policy B requires matching z-extent + n_layers.
            if not _z_extent_matches(kept, slab):
                raise StructuredOverlapError(
                    f"Volumetric overlap of structured prisms "
                    f"{kept.physical_name} and {slab.physical_name} "
                    f"requires matching z-extents (got "
                    f"[{kept.zlo}, {kept.zhi}] vs "
                    f"[{slab.zlo}, {slab.zhi}]). "
                    f"Adjust the prisms so z-extents match exactly or so "
                    f"footprints do not overlap."
                )
            kept_n = _n_layers_of_slab(kept, entities)
            slab_n = _n_layers_of_slab(slab, entities)
            if kept_n != slab_n:
                raise StructuredOverlapError(
                    f"Volumetric overlap of structured prisms "
                    f"{kept.physical_name} (n_layers={kept_n}) and "
                    f"{slab.physical_name} (n_layers={slab_n}) at "
                    f"z=[{kept.zlo}, {kept.zhi}]: n_layers must agree."
                )
            # Valid overlap: this slab is dominated; record the pair.
            overlaps.append(
                OverlapPair(
                    winner_slab_index=kept_indices.index(k_idx),  # index into final kept list
                    loser_source_index=slab.source_index,
                    loser_z_interval_index=slab.z_interval_index,
                    z_extent=(slab.zlo, slab.zhi),
                )
            )
            dominated = True
            break
        if not dominated:
            kept_indices.append(idx)

    kept_slabs = [slabs[i] for i in kept_indices]
    return kept_slabs, overlaps
```

- [ ] **Step 4: Run; expect pass**

Run: `pytest tests/structured/test_plan.py -v -k overlap`

Expected: 4 overlap tests PASS.

- [ ] **Step 5: Run the full structured test suite for a regression check**

Run: `pytest tests/structured/ -v`

Expected: all tests PASS (no Phase 1 tests have regressed).

- [ ] **Step 6: Commit**

```bash
git add meshwell/structured/plan.py tests/structured/test_plan.py
git commit -m "$(cat <<'EOF'
feat(structured): plan.validate_and_resolve_overlap (Policy B)

Implements Policy B: volumetric overlap of structured prisms is allowed
only when z-extents match exactly within tolerance AND n_layers agrees
in the overlap region. Lower mesh_order wins; ties broken by
source_index then z_interval_index for determinism.

Mismatched z-extent or n_layers raises StructuredOverlapError with a
message naming both prisms and the offending values. No proportional
n_layers distribution; no cascade.

OverlapPair records the (winner_slab_index, loser_source_index,
loser_z_interval_index, z_extent) tuple for mesh-stage cross-check
(Phase 3).
EOF
)"
```

---

## Task 8: Plan-stage helper — `compute_face_partition`

**Files:**
- Modify: `meshwell/structured/plan.py`
- Modify: `tests/structured/test_plan.py`

Phase 1 ships a minimal `compute_face_partition` that handles the no-neighbour case (single piece = the slab's footprint). The full neighbour-aware partition with arc provenance ports in Phase 2 once the CAD-stage primitives exist.

- [ ] **Step 1: Add the failing tests**

Append to `tests/structured/test_plan.py`:

```python
def test_face_partition_no_neighbours_single_piece():
    """No other entities touching the slab's z-planes: partition is one piece."""
    from meshwell.structured.plan import (
        compute_face_partition,
        expand_to_slabs,
        gather_structured_entities,
    )

    s = _structured(_square(0, 0, 4, 4), {0.0: 0.0, 1.0: 0.0}, [3], "s")
    slabs = expand_to_slabs(gather_structured_entities([s]))
    compute_face_partition(slabs, entities=[s])
    assert len(slabs[0].face_partition) == 1
    # Single piece equals the footprint (within shapely's equality tolerance).
    assert slabs[0].face_partition[0].equals(slabs[0].footprint)


def test_face_partition_with_neighbour_on_top_plane():
    """A non-structured prism touching the slab's top z-plane partitions the top."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured.plan import (
        compute_face_partition,
        expand_to_slabs,
        gather_structured_entities,
    )

    s = _structured(_square(0, 0, 4, 4), {0.0: 0.0, 1.0: 0.0}, [3], "s")
    # Non-structured neighbour sits on top of s (its zmin == s.zhi == 1.0),
    # covering the left half of s's footprint.
    n = PolyPrism(
        polygons=_square(0, 0, 2, 4),
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="n",
    )
    slabs = expand_to_slabs(gather_structured_entities([s, n]))
    compute_face_partition(slabs, entities=[s, n])
    # Slab partition has 2 pieces: covered (xy left half) and uncovered (xy right half).
    assert len(slabs[0].face_partition) == 2
    areas = sorted(p.area for p in slabs[0].face_partition)
    # Each piece has area 8 (2 wide x 4 tall).
    assert areas == pytest.approx([8.0, 8.0])
```

- [ ] **Step 2: Run; expect ImportError on compute_face_partition**

Run: `pytest tests/structured/test_plan.py -v -k face_partition`

Expected: tests fail with `ImportError`.

- [ ] **Step 3: Implement compute_face_partition**

Append to `meshwell/structured/plan.py`:

```python
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import polygonize, unary_union


def _entity_z_range(ent: Any) -> tuple[float, float] | None:
    """Return (zmin, zmax) for an entity that has a buffers/z-range, else None."""
    buffers = getattr(ent, "buffers", None)
    if not buffers:
        return None
    return (min(buffers.keys()), max(buffers.keys()))


def _entity_footprint(ent: Any) -> Polygon | MultiPolygon | None:
    polys = getattr(ent, "polygons", None)
    if polys is None:
        return None
    if isinstance(polys, list):
        flat: list[Polygon] = []
        for p in polys:
            if isinstance(p, MultiPolygon):
                flat.extend(p.geoms)
            elif isinstance(p, Polygon):
                flat.append(p)
        if not flat:
            return None
        return flat[0] if len(flat) == 1 else MultiPolygon(flat)
    return polys


def _neighbours_touching_z(
    z: float, entities: list[Any], skip_indices: set[int], tol: float = 1e-9
) -> list[Polygon | MultiPolygon]:
    """Footprints of entities whose buffers include z (within tol)."""
    out: list[Polygon | MultiPolygon] = []
    for i, ent in enumerate(entities):
        if i in skip_indices:
            continue
        rng = _entity_z_range(ent)
        if rng is None:
            continue
        zmin, zmax = rng
        if abs(zmin - z) < tol or abs(zmax - z) < tol:
            fp = _entity_footprint(ent)
            if fp is not None:
                out.append(fp)
    return out


def compute_face_partition(slabs: list[Slab], entities: list[Any]) -> None:
    """Compute slab.face_partition in place.

    For each slab, decompose its footprint into pairwise-disjoint pieces
    based on the union of any neighbouring entity footprints touching
    z=zlo or z=zhi. No-neighbour case: one piece = the whole footprint.
    """
    # Build a source_index -> slab indices map so we don't include a slab's
    # own owning entity as its own neighbour.
    own_indices_by_slab = {id(s): {s.source_index} for s in slabs}

    for slab in slabs:
        skip = own_indices_by_slab[id(slab)]
        neighbours_lo = _neighbours_touching_z(slab.zlo, entities, skip)
        neighbours_hi = _neighbours_touching_z(slab.zhi, entities, skip)
        all_neighbour_polys = neighbours_lo + neighbours_hi
        if not all_neighbour_polys:
            slab.face_partition = [slab.footprint]
            continue
        # Clip neighbour union to the slab footprint, then split.
        neighbour_union = unary_union(all_neighbour_polys)
        # The partition pieces = (slab.footprint ∩ neighbour_union) +
        # (slab.footprint − neighbour_union), each split by neighbour
        # boundaries via polygonize.
        seam_lines = neighbour_union.boundary
        boundary = slab.footprint.boundary
        combined = unary_union([boundary, seam_lines.intersection(slab.footprint)])
        raw = list(polygonize(combined))
        # Filter to pieces wholly inside the slab footprint (point-in-polygon
        # on the representative point).
        pieces = [
            piece for piece in raw if slab.footprint.contains(piece.representative_point())
        ]
        slab.face_partition = pieces if pieces else [slab.footprint]
```

- [ ] **Step 4: Run face_partition tests; expect pass**

Run: `pytest tests/structured/test_plan.py -v -k face_partition`

Expected: 2 face-partition tests PASS.

- [ ] **Step 5: Run the full plan suite for regression**

Run: `pytest tests/structured/ -v`

Expected: all tests in tests/structured/ PASS.

- [ ] **Step 6: Commit**

```bash
git add meshwell/structured/plan.py tests/structured/test_plan.py
git commit -m "$(cat <<'EOF'
feat(structured): plan.compute_face_partition (no-neighbour + simple cut)

Per-slab xy partition of the footprint induced by neighbouring entities
whose z-range touches z=zlo or z=zhi. Implementation uses shapely's
polygonize on (footprint boundary) U (neighbour boundary intersect
footprint).

Phase 1 ships the basic case (no-neighbour single piece + simple
single-neighbour cut). Arc-provenance preservation (porting the
2026-05-13-arc-provenance-face-partition design) and the multi-arc /
stacked-slab matrix port in Phase 2 once the CAD-stage primitives that
consume the provenance exist.
EOF
)"
```

---

## Task 9: Top-level `build_plan` orchestrator

**Files:**
- Modify: `meshwell/structured/plan.py`
- Modify: `meshwell/structured/__init__.py` (export `build_plan`)
- Modify: `tests/structured/test_plan.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/structured/test_plan.py`:

```python
def test_build_plan_empty_entities():
    from meshwell.structured.plan import build_plan

    plan = build_plan([])
    assert plan.slabs == []
    assert plan.z_planes == []
    assert plan.overlaps == []


def test_build_plan_no_structured_entities():
    from meshwell.polyprism import PolyPrism
    from meshwell.structured.plan import build_plan

    u = PolyPrism(polygons=_square(), buffers={0.0: 0.0, 1.0: 0.0}, physical_name="u")
    plan = build_plan([u])
    assert plan.slabs == []
    assert plan.z_planes == []
    assert plan.overlaps == []


def test_build_plan_simple_structured_only():
    from meshwell.structured.plan import build_plan

    s = _structured(_square(), {0.0: 0.0, 1.0: 0.0, 2.5: 0.0}, [3, 4], "s")
    plan = build_plan([s])
    assert len(plan.slabs) == 2
    assert plan.z_planes == [0.0, 1.0, 2.5]
    assert plan.overlaps == []
    # Both slabs get a single-piece partition (no neighbours).
    assert all(len(slab.face_partition) == 1 for slab in plan.slabs)


def test_build_plan_returns_frozen():
    """StructuredPlan is frozen; tuple lists are still mutable lists, but the
    StructuredPlan instance itself can't have fields reassigned."""
    from meshwell.structured.plan import build_plan

    plan = build_plan([])
    with pytest.raises((AttributeError, TypeError)):
        plan.slabs = []  # frozen dataclass rejects reassignment
```

- [ ] **Step 2: Run; expect ImportError**

Run: `pytest tests/structured/test_plan.py -v -k build_plan`

Expected: 4 tests fail.

- [ ] **Step 3: Implement build_plan**

Append to `meshwell/structured/plan.py`:

```python
from meshwell.structured.spec import StructuredPlan


def build_plan(entities: list[Any]) -> StructuredPlan:
    """Top-level planner: entities -> validated, partitioned StructuredPlan.

    Pipeline:

    1. ``gather_structured_entities`` filters and pairs entities with specs.
    2. ``expand_to_slabs`` produces one raw Slab per (entity, z-interval).
    3. ``validate_and_resolve_overlap`` applies Policy B; drops losers,
       records OverlapPairs. Raises ``StructuredOverlapError`` on mismatch.
    4. ``compute_face_partition`` decorates each surviving slab with its
       xy partition based on touching neighbour entities.

    The returned StructuredPlan is frozen and ready for the phantom +
    builder stages (Phase 2 / Phase 3).
    """
    pairs = gather_structured_entities(entities)
    if not pairs:
        return StructuredPlan(slabs=[], z_planes=[], overlaps=[])
    raw_slabs = expand_to_slabs(pairs)
    kept_slabs, overlaps = validate_and_resolve_overlap(raw_slabs, entities)
    compute_face_partition(kept_slabs, entities)
    # z_planes: sorted unique union of (zlo, zhi) across kept slabs.
    z_set: set[float] = set()
    for s in kept_slabs:
        z_set.add(s.zlo)
        z_set.add(s.zhi)
    z_planes = sorted(z_set)
    return StructuredPlan(slabs=kept_slabs, z_planes=z_planes, overlaps=overlaps)
```

- [ ] **Step 4: Re-export build_plan from the package**

Edit `meshwell/structured/__init__.py`:

```python
"""Clean structured-polyprism pipeline.

Public surface:

- :class:`StructuredExtrusionResolutionSpec` -- attach to a
  ``PolyPrism(structured=True)`` to specify per-z-interval layer counts.
- :func:`build_plan` -- orchestrator-facing entry point: validates
  structured entities, returns a frozen ``StructuredPlan`` for the
  CAD + mesh stages.
"""
from __future__ import annotations

from meshwell.structured.plan import build_plan
from meshwell.structured.spec import StructuredExtrusionResolutionSpec

__all__ = ["StructuredExtrusionResolutionSpec", "build_plan"]
```

- [ ] **Step 5: Run build_plan tests**

Run: `pytest tests/structured/test_plan.py -v -k build_plan`

Expected: 4 tests PASS.

- [ ] **Step 6: Run the full structured suite**

Run: `pytest tests/structured/ -v`

Expected: all tests PASS.

- [ ] **Step 7: Run the broader meshwell test suite for regression**

Run: `pytest tests/ -x -q --ignore=tests/structured 2>&1 | tail -20`

Expected: same baseline as `main` (some tests may already be broken/xfail; just ensure Phase 1 didn't introduce new failures).

- [ ] **Step 8: Commit**

```bash
git add meshwell/structured/plan.py meshwell/structured/__init__.py tests/structured/test_plan.py
git commit -m "$(cat <<'EOF'
feat(structured): build_plan orchestrator + public re-export

The planner pipeline runs end-to-end:
  gather -> expand -> validate_and_resolve_overlap -> compute_face_partition
yielding a frozen StructuredPlan with sorted unique z_planes and
recorded overlap pairs.

Phase 1 ships the data layer with no CAD/mesh integration. Phase 2
will add the phantom module (CAD stage with BOP history tracking via
OCP) and Phase 3 will add the builder module (mesh stage discrete-3D
entity construction).
EOF
)"
```

---

## Self-Review Checklist

**1. Spec coverage** (from `2026-05-15-structured-polyprism-clean-design.md`):

| Spec section | Phase 1 task | Status |
|---|---|---|
| `StructuredExtrusionResolutionSpec` | Task 3 | ✓ |
| `Slab` (CAD-only) | Task 3 | ✓ |
| `OverlapPair` | Task 3 | ✓ |
| `StructuredPlan` | Task 3 | ✓ |
| Policy B (overlap rule) | Task 7 | ✓ |
| `PolyPrism(structured=True)` API | Task 4 | ✓ |
| Buffers-uniform-or-raise | Task 4 | ✓ |
| `n_layers` length validation | Task 4 | ✓ |
| `compute_face_partition` (basic) | Task 8 | ✓ |
| `build_plan` orchestrator | Task 9 | ✓ |
| Layer A (mirror-symmetric topology) | **Phase 2** | deferred |
| Layer B (OCC vertex map via BOP history) | **Phase 2** | deferred |
| Layer C (mesh stage owns top mesh) | **Phase 3** | deferred |
| `StructuredMeshPlan` | **Phase 3** | deferred |
| `logging.py` (per-phase timing) | **Phase 3** | deferred |
| Orchestrator wiring | **Phase 3** | deferred |
| Arc-provenance partition (full) | **Phase 2** | deferred |
| `removeDuplicateNodes` global cleanup | **Phase 3** | deferred |

**2. Placeholder scan:** none. Every step has the actual code.

**3. Type consistency:**
- `Slab.face_partition: list[Polygon]` (Task 3) consistent with Task 8's `slab.face_partition = pieces` (list[Polygon]). ✓
- `OverlapPair(winner_slab_index, loser_source_index, loser_z_interval_index, z_extent)` (Task 3) matches construction in Task 7. ✓
- `StructuredPlan(slabs, z_planes, overlaps)` field order matches construction in Task 9. ✓
- `gather_structured_entities` returns `list[tuple[Any, StructuredExtrusionResolutionSpec, int]]`; Task 6 unpacks as `(entity, _spec, source_index)`. ✓

**4. Ambiguity check:**
- Task 4's `super().__init__` call may or may not already accept `resolutions=`; Step 3 explicitly grep-checks this and Step 5 adds a fallback assignment if needed.
- Task 7's `_Z_TOL = 1e-9` is a module-level constant; tests use exact float literals that match within this tolerance.
- Task 8's `compute_face_partition` does not handle stacked-slab matching across z-planes (deferred to Phase 2 when neighbour-aware partition runs against sibling slabs too).

---

## Out of scope for Phase 1

- `meshwell/structured/phantom.py` — Phase 2.
- `meshwell/structured/builder.py` — Phase 3.
- `meshwell/structured/logging.py` — Phase 3.
- Orchestrator integration; `structured=True` is currently a no-op past planning.
- Arc-provenance face-partition migration from `feat/structured` — Phase 2.
- Cross-stage tests that exercise CAD or gmsh — Phase 3.
- Performance benchmarks — Phase 3 (after the mesh stage exists).
- Migration of `feat/structured`'s test suite — done piecewise in Phases 2/3 as the consumed primitives land.
