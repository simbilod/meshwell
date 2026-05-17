# Structured-mesh conformality validator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a live-gmsh-session validator that detects non-conformality between the structured wedge/hex slabs and surrounding tet regions produced by `meshwell.structured`.

**Architecture:** One new module `meshwell/structured/validator.py` exposing `validate_structured_mesh(plan, mesh_plan, phantom_map, occ_entities, vol_tags, ...) -> ValidationResult`. Eight conformality checks, each its own private function, called in order with results accumulated into a severity-tagged result object. Uses gmsh's Python API primitives (`createFaces`, `getElementFaceNodes`, `getDuplicateNodes`, `getElementQualities`, `getElements`, `getNodes`); no meshio dependency. No changes to `StructuredMeshPlan` or `PhantomMap` — the validator consumes the in-memory outputs of `resolve_mesh_plan` / `apply_structured_mesh`.

**Tech Stack:** Python 3.12, pytest, gmsh (live session via `import gmsh`), numpy, existing `meshwell.structured.spec` dataclasses.

**Spec:** [`docs/superpowers/specs/2026-05-17-structured-mesh-validator-design.md`](../specs/2026-05-17-structured-mesh-validator-design.md)

---

## File Structure

- **Create:** [`meshwell/structured/validator.py`](../../../meshwell/structured/validator.py) — the validator module (Issue / ValidationResult / `validate_structured_mesh` + eight private check helpers).
- **Create:** [`tests/structured/test_validator_dataclasses.py`](../../../tests/structured/test_validator_dataclasses.py) — Issue / ValidationResult unit tests.
- **Create:** [`tests/structured/test_validator_watertight.py`](../../../tests/structured/test_validator_watertight.py) — check 1 unit tests.
- **Create:** [`tests/structured/test_validator_duplicates.py`](../../../tests/structured/test_validator_duplicates.py) — check 5 unit tests.
- **Create:** [`tests/structured/test_validator_quality.py`](../../../tests/structured/test_validator_quality.py) — check 7 unit tests.
- **Create:** [`tests/structured/test_validator_plan_consistency.py`](../../../tests/structured/test_validator_plan_consistency.py) — check 4 unit tests.
- **Create:** [`tests/structured/test_validator_seams.py`](../../../tests/structured/test_validator_seams.py) — check 3 unit tests.
- **Create:** [`tests/structured/test_validator_interface.py`](../../../tests/structured/test_validator_interface.py) — checks 2 + 6 unit tests.
- **Create:** [`tests/structured/test_validator_top_bottom_symmetry.py`](../../../tests/structured/test_validator_top_bottom_symmetry.py) — check 8 unit tests.
- **Modify:** [`tests/structured/test_end_to_end_minimal.py`](../../../tests/structured/test_end_to_end_minimal.py) — append validator call after mesh generation.
- **Modify:** [`tests/structured/test_end_to_end_multipiece.py`](../../../tests/structured/test_end_to_end_multipiece.py) — append validator call after mesh generation.

Each check gets its own test file so failures localize to a specific check, and so the unit tests can synthesize minimal gmsh state per check rather than running a full pipeline.

---

## Task 1: Scaffold module + dataclasses

**Files:**
- Create: `meshwell/structured/validator.py`
- Create: `tests/structured/test_validator_dataclasses.py`

This task delivers the `Issue` and `ValidationResult` types and an empty `validate_structured_mesh` function that returns a passing result for any input. Subsequent tasks add real checks one at a time.

- [ ] **Step 1: Write failing tests for `Issue` and `ValidationResult`**

Create `tests/structured/test_validator_dataclasses.py`:

```python
"""Unit tests for Issue / ValidationResult dataclasses."""
from meshwell.structured.validator import Issue, ValidationResult


def test_issue_is_frozen_dataclass():
    issue = Issue(
        severity="error",
        check="watertight",
        message="hole in volume",
        entities=(("face", 42),),
    )
    assert issue.severity == "error"
    assert issue.check == "watertight"
    assert issue.message == "hole in volume"
    assert issue.entities == (("face", 42),)


def test_validation_result_truthy_when_no_errors():
    result = ValidationResult(errors=(), warnings=())
    assert bool(result) is True

    only_warnings = ValidationResult(
        errors=(),
        warnings=(Issue("warning", "near_duplicates", "1 pair", ()),),
    )
    assert bool(only_warnings) is True


def test_validation_result_falsy_when_errors_present():
    result = ValidationResult(
        errors=(Issue("error", "watertight", "hole", ()),),
        warnings=(),
    )
    assert bool(result) is False


def test_format_report_groups_by_check():
    result = ValidationResult(
        errors=(
            Issue("error", "watertight", "hole at face 42", (("face", 42),)),
            Issue("error", "watertight", "hole at face 51", (("face", 51),)),
            Issue("error", "interface", "T-junction", (("face", 99),)),
        ),
        warnings=(
            Issue("warning", "near_duplicates", "1 pair", ()),
        ),
    )
    report = result.format_report()
    # Errors before warnings, grouped by check name.
    assert "watertight" in report
    assert "interface" in report
    assert "near_duplicates" in report
    error_idx = report.index("ERRORS")
    warning_idx = report.index("WARNINGS")
    assert error_idx < warning_idx


def test_format_report_empty_result_is_clean():
    result = ValidationResult(errors=(), warnings=())
    report = result.format_report()
    assert "no issues" in report.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/structured/test_validator_dataclasses.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meshwell.structured.validator'`

- [ ] **Step 3: Create `meshwell/structured/validator.py` with scaffold**

```python
"""Conformality validator for structured-polyprism meshes.

Runs in the live gmsh session immediately after
``apply_structured_mesh``. Reports topological and geometric
conformality failures between the structured wedge/hex slabs and the
surrounding tet regions.

Public API:

- :class:`Issue` — one validation finding (severity + check name + message + entities).
- :class:`ValidationResult` — collected errors + warnings + report formatter.
- :func:`validate_structured_mesh` — entry point. See its docstring.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from meshwell.structured.spec import (
        PhantomMap,
        StructuredMeshPlan,
        StructuredPlan,
    )


Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Issue:
    """One validation finding.

    ``entities`` is a tuple of (kind, tag) or (kind, key) tuples that
    localize the issue. Examples:
    - ``(("face", 42),)`` — a single gmsh face tag.
    - ``(("node", 17), ("node", 19))`` — a pair of node tags.
    - ``(("slab_piece", (1, 0)),)`` — a (slab_index, piece_index) key.
    """

    severity: Severity
    check: str
    message: str
    entities: tuple = ()


@dataclass(frozen=True)
class ValidationResult:
    """Collected validator output.

    ``__bool__`` is True iff ``errors`` is empty (warnings are not
    failures). Use ``assert result, result.format_report()`` in tests.
    """

    errors: tuple[Issue, ...]
    warnings: tuple[Issue, ...]

    def __bool__(self) -> bool:
        return not self.errors

    def format_report(self) -> str:
        if not self.errors and not self.warnings:
            return "Structured-mesh validation: no issues."
        lines: list[str] = ["Structured-mesh validation report"]
        if self.errors:
            lines.append("")
            lines.append(f"ERRORS ({len(self.errors)})")
            for check, group in _group_by_check(self.errors).items():
                lines.append(f"  [{check}]")
                for issue in group:
                    lines.append(f"    - {issue.message}")
                    if issue.entities:
                        lines.append(f"      entities: {issue.entities}")
        if self.warnings:
            lines.append("")
            lines.append(f"WARNINGS ({len(self.warnings)})")
            for check, group in _group_by_check(self.warnings).items():
                lines.append(f"  [{check}]")
                for issue in group:
                    lines.append(f"    - {issue.message}")
                    if issue.entities:
                        lines.append(f"      entities: {issue.entities}")
        return "\n".join(lines)


def _group_by_check(issues: tuple[Issue, ...]) -> dict[str, list[Issue]]:
    out: dict[str, list[Issue]] = defaultdict(list)
    for issue in issues:
        out[issue.check].append(issue)
    return dict(out)


def validate_structured_mesh(
    plan: "StructuredPlan",
    mesh_plan: "StructuredMeshPlan",
    phantom_map: "PhantomMap",
    occ_entities: list[Any],
    vol_tags: list[int],
    *,
    tol: float | None = None,
    include_quality: bool = False,
) -> ValidationResult:
    """Validate the live-gmsh-session mesh against the builder's plan.

    Must be called while a gmsh model is initialized and meshed (i.e.
    after ``meshwell.structured.builder.apply_structured_mesh``). Reads
    from gmsh's in-memory model; writes nothing.

    Args:
        plan: the StructuredPlan from the planner.
        mesh_plan: the StructuredMeshPlan from ``resolve_mesh_plan``.
        phantom_map: the PhantomMap built by Phase-2 phantom stage.
        occ_entities: the OCC entity list used by the builder
            (needed for face/edge gmsh-tag lookup via the existing
            ``_map_phantom_*`` helpers).
        vol_tags: the list of per-piece 3D entity tags returned by
            ``apply_structured_mesh``.
        tol: absolute coordinate tolerance for geometric checks. If
            None, derived from the minimum edge length reported by
            ``gmsh.model.mesh.getElementQualities(..., "minEdge")``.
        include_quality: if True, additionally run a quality check via
            ``gmsh.model.mesh.getElementQualities``. Off by default;
            the validator's focus is conformality, not quality.

    Returns:
        ValidationResult with collected errors / warnings. ``bool(result)``
        is True iff ``errors`` is empty.

    Raises:
        RuntimeError: if no gmsh model is initialized.
    """
    errors: list[Issue] = []
    warnings: list[Issue] = []
    # Checks are added one at a time in subsequent tasks.
    return ValidationResult(errors=tuple(errors), warnings=tuple(warnings))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/structured/test_validator_dataclasses.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/validator.py tests/structured/test_validator_dataclasses.py
git commit -m "feat(validator): scaffold Issue / ValidationResult and empty entry point"
```

---

## Task 2: Tolerance helper

**Files:**
- Modify: `meshwell/structured/validator.py`
- Create: `tests/structured/test_validator_tolerance.py`

The default tolerance is `min_edge_length × 1e-6` via `gmsh.model.mesh.getElementQualities(..., "minEdge")`. We isolate this in `_resolve_tol(tol)` so later checks can use it.

- [ ] **Step 1: Write failing test**

Create `tests/structured/test_validator_tolerance.py`:

```python
"""Unit tests for _resolve_tol helper."""
import gmsh
import pytest

from meshwell.structured.validator import _resolve_tol


@pytest.fixture
def gmsh_unit_cube_meshed():
    """A meshed unit cube — gives us real elements to query getElementQualities on."""
    gmsh.initialize()
    gmsh.model.add("tol_test")
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.5)
    gmsh.model.mesh.generate(3)
    yield
    gmsh.finalize()


def test_resolve_tol_returns_explicit_value(gmsh_unit_cube_meshed):
    assert _resolve_tol(1e-8) == 1e-8


def test_resolve_tol_derives_from_mesh(gmsh_unit_cube_meshed):
    tol = _resolve_tol(None)
    # Cube edge ~0.5; tol should be ~5e-7, definitely between 1e-9 and 1e-3.
    assert 1e-12 < tol < 1e-3


def test_resolve_tol_zero_falls_back_to_safe_default():
    # No gmsh session — guard against AttributeError or NaN.
    gmsh.initialize()
    gmsh.model.add("empty")
    tol = _resolve_tol(None)
    assert tol > 0
    gmsh.finalize()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/structured/test_validator_tolerance.py -v`
Expected: FAIL with `ImportError: cannot import name '_resolve_tol'`.

- [ ] **Step 3: Add `_resolve_tol` to `meshwell/structured/validator.py`**

Add this function near the top of the module (after `_group_by_check`):

```python
_DEFAULT_TOL_FALLBACK = 1e-9


def _resolve_tol(tol: float | None) -> float:
    """Pick a coordinate tolerance for geometric checks.

    If ``tol`` is given, returns it unchanged. Otherwise derives from
    the minimum edge length of the current gmsh mesh:
    ``min_edge_length * 1e-6``. Falls back to ``1e-9`` when no elements
    exist in the model yet.
    """
    if tol is not None:
        return float(tol)

    import gmsh

    try:
        elem_types, elem_tags, _ = gmsh.model.mesh.getElements(3)
        if not elem_types:
            return _DEFAULT_TOL_FALLBACK
        # Concatenate all 3D element tags across types.
        all_tags = [t for tags in elem_tags for t in tags]
        if not all_tags:
            return _DEFAULT_TOL_FALLBACK
        min_edges = gmsh.model.mesh.getElementQualities(all_tags, "minEdge")
        if len(min_edges) == 0:
            return _DEFAULT_TOL_FALLBACK
        min_edge = float(min(min_edges))
        if min_edge <= 0:
            return _DEFAULT_TOL_FALLBACK
        return min_edge * 1e-6
    except Exception:
        return _DEFAULT_TOL_FALLBACK
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/structured/test_validator_tolerance.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/validator.py tests/structured/test_validator_tolerance.py
git commit -m "feat(validator): _resolve_tol derives from getElementQualities minEdge"
```

---

## Task 3: Check 5 — near-duplicate nodes

**Files:**
- Modify: `meshwell/structured/validator.py`
- Create: `tests/structured/test_validator_duplicates.py`

This check runs first in the implementation order because it's the simplest (single gmsh primitive call) and lets us validate the call shape of `validate_structured_mesh` before more complex checks land.

- [ ] **Step 1: Write failing tests**

Create `tests/structured/test_validator_duplicates.py`:

```python
"""Unit tests for check 5: near-duplicate nodes."""
import gmsh
import pytest

from meshwell.structured.spec import (
    PhantomMap,
    StructuredMeshPlan,
    StructuredPlan,
)
from meshwell.structured.validator import validate_structured_mesh


@pytest.fixture
def empty_inputs():
    """Empty plan / mesh_plan / phantom_map — exercises only mesh-level checks."""
    plan = StructuredPlan(slabs=(), z_planes=(), overlaps=())
    mesh_plan = StructuredMeshPlan(slabs=(), n_layers=(), recombine=())
    phantom_map = PhantomMap()
    return plan, mesh_plan, phantom_map


def _add_lone_node(x: float, y: float, z: float) -> int:
    """Helper: add a node to a fresh discrete 3D entity, return its tag."""
    ent = gmsh.model.addDiscreteEntity(3, -1, [])
    max_tag = gmsh.model.mesh.getMaxNodeTag()
    new_tag = int(max_tag) + 1
    gmsh.model.mesh.addNodes(3, ent, [new_tag], [x, y, z])
    return new_tag


def test_unique_nodes_no_issue(empty_inputs):
    plan, mesh_plan, phantom_map = empty_inputs
    gmsh.initialize()
    gmsh.model.add("unique")
    _add_lone_node(0.0, 0.0, 0.0)
    _add_lone_node(1.0, 0.0, 0.0)
    _add_lone_node(0.0, 1.0, 0.0)

    result = validate_structured_mesh(
        plan, mesh_plan, phantom_map, occ_entities=[], vol_tags=[], tol=1e-6
    )
    duplicate_issues = [i for i in result.errors + result.warnings
                        if i.check == "near_duplicate_nodes"]
    assert duplicate_issues == []
    gmsh.finalize()


def test_exact_duplicate_nodes_reported_as_error(empty_inputs):
    plan, mesh_plan, phantom_map = empty_inputs
    gmsh.initialize()
    gmsh.model.add("exact_dup")
    _add_lone_node(0.5, 0.5, 0.5)
    _add_lone_node(0.5, 0.5, 0.5)  # Exact duplicate.

    result = validate_structured_mesh(
        plan, mesh_plan, phantom_map, occ_entities=[], vol_tags=[], tol=1e-6
    )
    exact_errors = [i for i in result.errors if i.check == "near_duplicate_nodes"]
    assert len(exact_errors) >= 1
    gmsh.finalize()


def test_near_duplicate_nodes_reported_as_warning(empty_inputs):
    plan, mesh_plan, phantom_map = empty_inputs
    gmsh.initialize()
    gmsh.model.add("near_dup")
    _add_lone_node(0.5, 0.5, 0.5)
    _add_lone_node(0.5 + 1e-9, 0.5, 0.5)  # 1 nm offset.

    result = validate_structured_mesh(
        plan, mesh_plan, phantom_map, occ_entities=[], vol_tags=[], tol=1e-7
    )
    near_warnings = [i for i in result.warnings if i.check == "near_duplicate_nodes"]
    assert len(near_warnings) >= 1
    gmsh.finalize()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/structured/test_validator_duplicates.py -v`
Expected: `test_unique_nodes_no_issue` passes (validator currently returns empty), but `test_exact_duplicate_nodes_reported_as_error` and `test_near_duplicate_nodes_reported_as_warning` FAIL because no detection is wired up.

- [ ] **Step 3: Implement check in `meshwell/structured/validator.py`**

Add this helper function (between `_resolve_tol` and `validate_structured_mesh`):

```python
def _check_near_duplicate_nodes(tol: float) -> tuple[list[Issue], list[Issue]]:
    """Detect exact + near-duplicate node coordinates.

    Returns ``(errors, warnings)``. Exact duplicates are reported as
    errors (indicates ``removeDuplicateNodes`` was skipped or too tight).
    Near-duplicates within ``tol`` but with non-zero offset are warnings.
    """
    import gmsh
    import numpy as np

    node_tags_arr, node_coords_flat, _ = gmsh.model.mesh.getNodes()
    if len(node_tags_arr) == 0:
        return [], []

    node_tags = np.asarray(node_tags_arr, dtype=np.int64)
    coords = np.asarray(node_coords_flat, dtype=float).reshape(-1, 3)
    n = coords.shape[0]
    if n < 2:
        return [], []

    # Spatial-hash bin = tol. Pairs within sqrt(3)*tol may straddle bins,
    # so check the 27 neighbouring bins around each.
    bin_size = max(tol, 1e-15)
    bins = np.floor(coords / bin_size).astype(np.int64)

    bucket: dict[tuple[int, int, int], list[int]] = {}
    for i in range(n):
        key = (int(bins[i, 0]), int(bins[i, 1]), int(bins[i, 2]))
        bucket.setdefault(key, []).append(i)

    errors: list[Issue] = []
    warnings: list[Issue] = []
    reported_pairs: set[tuple[int, int]] = set()

    for i in range(n):
        bi = (int(bins[i, 0]), int(bins[i, 1]), int(bins[i, 2]))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    key = (bi[0] + dx, bi[1] + dy, bi[2] + dz)
                    nbrs = bucket.get(key)
                    if not nbrs:
                        continue
                    for j in nbrs:
                        if j <= i:
                            continue
                        d2 = float(np.sum((coords[i] - coords[j]) ** 2))
                        if d2 > tol * tol:
                            continue
                        pair = (int(node_tags[i]), int(node_tags[j]))
                        if pair in reported_pairs:
                            continue
                        reported_pairs.add(pair)
                        if d2 == 0.0:
                            errors.append(Issue(
                                severity="error",
                                check="near_duplicate_nodes",
                                message=(
                                    f"Exact duplicate node coords at "
                                    f"({coords[i, 0]:.6g}, {coords[i, 1]:.6g}, "
                                    f"{coords[i, 2]:.6g}); removeDuplicateNodes "
                                    f"may have been skipped."
                                ),
                                entities=(("node", pair[0]), ("node", pair[1])),
                            ))
                        else:
                            warnings.append(Issue(
                                severity="warning",
                                check="near_duplicate_nodes",
                                message=(
                                    f"Near-duplicate node pair: distance "
                                    f"{d2 ** 0.5:.3e} < tol {tol:.3e}."
                                ),
                                entities=(("node", pair[0]), ("node", pair[1])),
                            ))

    return errors, warnings
```

Then modify `validate_structured_mesh` to call it. Replace the function body with:

```python
def validate_structured_mesh(
    plan: "StructuredPlan",
    mesh_plan: "StructuredMeshPlan",
    phantom_map: "PhantomMap",
    occ_entities: list[Any],
    vol_tags: list[int],
    *,
    tol: float | None = None,
    include_quality: bool = False,
) -> ValidationResult:
    """Validate the live-gmsh-session mesh against the builder's plan.

    [docstring unchanged from Task 1]
    """
    resolved_tol = _resolve_tol(tol)

    errors: list[Issue] = []
    warnings: list[Issue] = []

    dup_errors, dup_warnings = _check_near_duplicate_nodes(resolved_tol)
    errors.extend(dup_errors)
    warnings.extend(dup_warnings)

    return ValidationResult(errors=tuple(errors), warnings=tuple(warnings))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/structured/test_validator_duplicates.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/validator.py tests/structured/test_validator_duplicates.py
git commit -m "feat(validator): check 5 — near-duplicate nodes (error/warning split)"
```

---

## Task 4: Check 7 — opt-in element quality

**Files:**
- Modify: `meshwell/structured/validator.py`
- Create: `tests/structured/test_validator_quality.py`

Wraps `gmsh.model.mesh.getElementQualities(..., "minSICN")` and flags negative-Jacobian elements as errors, very-small-positive values as warnings.

- [ ] **Step 1: Write failing tests**

Create `tests/structured/test_validator_quality.py`:

```python
"""Unit tests for check 7: element quality (opt-in)."""
import gmsh
import pytest

from meshwell.structured.spec import (
    PhantomMap,
    StructuredMeshPlan,
    StructuredPlan,
)
from meshwell.structured.validator import validate_structured_mesh


@pytest.fixture
def empty_plan_inputs():
    plan = StructuredPlan(slabs=(), z_planes=(), overlaps=())
    mesh_plan = StructuredMeshPlan(slabs=(), n_layers=(), recombine=())
    phantom_map = PhantomMap()
    return plan, mesh_plan, phantom_map


@pytest.fixture
def meshed_cube():
    gmsh.initialize()
    gmsh.model.add("quality_cube")
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.4)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.4)
    gmsh.model.mesh.generate(3)
    yield
    gmsh.finalize()


def test_quality_check_off_by_default(meshed_cube, empty_plan_inputs):
    plan, mesh_plan, phantom_map = empty_plan_inputs
    result = validate_structured_mesh(
        plan, mesh_plan, phantom_map, occ_entities=[], vol_tags=[], tol=1e-6,
    )
    quality_issues = [i for i in result.errors + result.warnings
                      if i.check == "element_quality"]
    assert quality_issues == []


def test_good_quality_cube_passes(meshed_cube, empty_plan_inputs):
    plan, mesh_plan, phantom_map = empty_plan_inputs
    result = validate_structured_mesh(
        plan, mesh_plan, phantom_map, occ_entities=[], vol_tags=[],
        tol=1e-6, include_quality=True,
    )
    # A cube meshed at uniform CL=0.4 should have no quality errors.
    quality_errors = [i for i in result.errors if i.check == "element_quality"]
    assert quality_errors == []


def test_negative_jacobian_reported_as_error(empty_plan_inputs):
    """A wedge with reversed top-bottom orientation has negative minSICN."""
    plan, mesh_plan, phantom_map = empty_plan_inputs
    gmsh.initialize()
    gmsh.model.add("bad_wedge")
    ent = gmsh.model.addDiscreteEntity(3, -1, [])
    # Wedge node order [bot0, bot1, bot2, top0, top1, top2] with top
    # winding reversed → negative Jacobian.
    coords = [
        0.0, 0.0, 0.0,    # 1: bot0
        1.0, 0.0, 0.0,    # 2: bot1
        0.0, 1.0, 0.0,    # 3: bot2
        0.0, 1.0, 1.0,    # 4: top0  (reversed)
        1.0, 0.0, 1.0,    # 5: top1
        0.0, 0.0, 1.0,    # 6: top2
    ]
    gmsh.model.mesh.addNodes(3, ent, [1, 2, 3, 4, 5, 6], coords)
    # Element type 6 = 6-node prism.
    gmsh.model.mesh.addElements(3, ent, [6], [[100]], [[1, 2, 3, 4, 5, 6]])

    result = validate_structured_mesh(
        plan, mesh_plan, phantom_map, occ_entities=[], vol_tags=[],
        tol=1e-6, include_quality=True,
    )
    quality_errors = [i for i in result.errors if i.check == "element_quality"]
    assert len(quality_errors) >= 1
    gmsh.finalize()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/structured/test_validator_quality.py -v`
Expected: `test_quality_check_off_by_default` passes; `test_good_quality_cube_passes` passes; `test_negative_jacobian_reported_as_error` FAILS (no quality check yet).

- [ ] **Step 3: Implement check in `meshwell/structured/validator.py`**

Add helper:

```python
def _check_element_quality() -> tuple[list[Issue], list[Issue]]:
    """Flag negative-Jacobian and near-degenerate elements.

    Uses ``gmsh.model.mesh.getElementQualities(..., "minSICN")``. The
    SICN (Scaled Inverse Condition Number) metric is in [0, 1] for
    valid elements and negative for tangled elements.

    - minSICN <= 0   → error (tangled / inverted element).
    - minSICN < 1e-3 → warning (near-degenerate).
    """
    import gmsh

    errors: list[Issue] = []
    warnings: list[Issue] = []

    elem_types, elem_tags_per_type, _ = gmsh.model.mesh.getElements(3)
    if not elem_types:
        return errors, warnings

    all_tags: list[int] = []
    for tags in elem_tags_per_type:
        all_tags.extend(int(t) for t in tags)
    if not all_tags:
        return errors, warnings

    sicn = gmsh.model.mesh.getElementQualities(all_tags, "minSICN")
    for tag, q in zip(all_tags, sicn):
        if q <= 0:
            errors.append(Issue(
                severity="error",
                check="element_quality",
                message=f"Element {tag} has minSICN={q:.3e} (tangled/inverted).",
                entities=(("element", int(tag)),),
            ))
        elif q < 1e-3:
            warnings.append(Issue(
                severity="warning",
                check="element_quality",
                message=f"Element {tag} has minSICN={q:.3e} (near-degenerate).",
                entities=(("element", int(tag)),),
            ))
    return errors, warnings
```

Wire it in `validate_structured_mesh`, after the duplicates block:

```python
    if include_quality:
        q_errors, q_warnings = _check_element_quality()
        errors.extend(q_errors)
        warnings.extend(q_warnings)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/structured/test_validator_quality.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/validator.py tests/structured/test_validator_quality.py
git commit -m "feat(validator): check 7 — opt-in element quality via getElementQualities"
```

---

## Task 5: Check 4 — plan↔mesh consistency

**Files:**
- Modify: `meshwell/structured/validator.py`
- Create: `tests/structured/test_validator_plan_consistency.py`

For each `(slab, piece)` in the plan, count wedge or hex elements in the corresponding 3D entity and compare to `n_layers × triangle_count`. We don't have a per-piece-bottom-triangle-count stored on `Slab`, so we infer it: the bottom triangulation count = the volume's element count / n_layers. Therefore this check is actually structured as "for each piece volume, its element count must be a positive multiple of `n_layers`", plus "the number of vol_tags must equal `sum(len(slab.face_partition) for slab in plan.slabs)`".

- [ ] **Step 1: Write failing tests**

Create `tests/structured/test_validator_plan_consistency.py`:

```python
"""Unit tests for check 4: plan ↔ mesh consistency."""
import gmsh
import pytest

from meshwell.structured.spec import (
    PhantomMap,
    Slab,
    StructuredMeshPlan,
    StructuredPlan,
)
from meshwell.structured.validator import validate_structured_mesh


def _empty_slab(z_interval_index: int, n_pieces: int) -> Slab:
    """Make a Slab with `n_pieces` empty footprint entries (validator
    doesn't need real geometry for this check)."""
    from shapely.geometry import Polygon

    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    slab = Slab(
        footprint=poly,
        zlo=0.0,
        zhi=1.0,
        physical_name=("test",),
        source_index=0,
        z_interval_index=z_interval_index,
        mesh_order=1.0,
        face_partition=[poly] * n_pieces,
    )
    return slab


def _make_wedges(vol_tag: int, count: int, start_node: int = 1) -> int:
    """Add `count` disjoint wedge elements to `vol_tag`. Returns next free node."""
    nodes: list[int] = []
    coords: list[float] = []
    elem_node_lists: list[int] = []
    next_node = start_node
    for k in range(count):
        base_x = float(k) * 2.0  # offset so wedges don't overlap.
        n_ids = [next_node + i for i in range(6)]
        next_node += 6
        nodes.extend(n_ids)
        coords.extend([
            base_x + 0.0, 0.0, 0.0,
            base_x + 1.0, 0.0, 0.0,
            base_x + 0.0, 1.0, 0.0,
            base_x + 0.0, 0.0, 1.0,
            base_x + 1.0, 0.0, 1.0,
            base_x + 0.0, 1.0, 1.0,
        ])
        elem_node_lists.extend(n_ids)
    gmsh.model.mesh.addNodes(3, vol_tag, nodes, coords)
    elem_tag_start = int(gmsh.model.mesh.getMaxElementTag()) + 1
    gmsh.model.mesh.addElements(
        3, vol_tag, [6],
        [list(range(elem_tag_start, elem_tag_start + count))],
        [elem_node_lists],
    )
    return next_node


@pytest.fixture
def gmsh_session():
    gmsh.initialize()
    gmsh.model.add("plan_consistency")
    yield
    gmsh.finalize()


def test_one_slab_one_piece_correct_count_passes(gmsh_session):
    slab = _empty_slab(0, n_pieces=1)
    plan = StructuredPlan(slabs=(slab,), z_planes=(0.0, 1.0), overlaps=())
    mesh_plan = StructuredMeshPlan(slabs=(slab,), n_layers=(2,), recombine=(False,))

    vol = gmsh.model.addDiscreteEntity(3, -1, [])
    _make_wedges(vol, count=4)  # 2 layers × 2 triangles = 4.

    result = validate_structured_mesh(
        plan, mesh_plan, PhantomMap(), occ_entities=[],
        vol_tags=[vol], tol=1e-6,
    )
    plan_errors = [i for i in result.errors if i.check == "plan_mesh_consistency"]
    assert plan_errors == []


def test_vol_tag_count_mismatch_reported(gmsh_session):
    """Plan says 2 pieces but only 1 vol_tag — error."""
    slab = _empty_slab(0, n_pieces=2)
    plan = StructuredPlan(slabs=(slab,), z_planes=(0.0, 1.0), overlaps=())
    mesh_plan = StructuredMeshPlan(slabs=(slab,), n_layers=(2,), recombine=(False,))

    vol = gmsh.model.addDiscreteEntity(3, -1, [])
    _make_wedges(vol, count=4)

    result = validate_structured_mesh(
        plan, mesh_plan, PhantomMap(), occ_entities=[],
        vol_tags=[vol], tol=1e-6,
    )
    plan_errors = [i for i in result.errors if i.check == "plan_mesh_consistency"]
    assert any("vol_tag count" in i.message for i in plan_errors)


def test_element_count_not_multiple_of_n_layers_reported(gmsh_session):
    slab = _empty_slab(0, n_pieces=1)
    plan = StructuredPlan(slabs=(slab,), z_planes=(0.0, 1.0), overlaps=())
    mesh_plan = StructuredMeshPlan(slabs=(slab,), n_layers=(3,), recombine=(False,))

    vol = gmsh.model.addDiscreteEntity(3, -1, [])
    _make_wedges(vol, count=5)  # 5 is not a multiple of 3.

    result = validate_structured_mesh(
        plan, mesh_plan, PhantomMap(), occ_entities=[],
        vol_tags=[vol], tol=1e-6,
    )
    plan_errors = [i for i in result.errors if i.check == "plan_mesh_consistency"]
    assert any("not a multiple" in i.message for i in plan_errors)


def test_empty_volume_reported(gmsh_session):
    slab = _empty_slab(0, n_pieces=1)
    plan = StructuredPlan(slabs=(slab,), z_planes=(0.0, 1.0), overlaps=())
    mesh_plan = StructuredMeshPlan(slabs=(slab,), n_layers=(2,), recombine=(False,))

    vol = gmsh.model.addDiscreteEntity(3, -1, [])  # No elements added.

    result = validate_structured_mesh(
        plan, mesh_plan, PhantomMap(), occ_entities=[],
        vol_tags=[vol], tol=1e-6,
    )
    plan_errors = [i for i in result.errors if i.check == "plan_mesh_consistency"]
    assert any("zero elements" in i.message for i in plan_errors)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/structured/test_validator_plan_consistency.py -v`
Expected: All but the first FAIL (no check yet).

- [ ] **Step 3: Implement check in `meshwell/structured/validator.py`**

Add helper:

```python
def _check_plan_mesh_consistency(
    plan: "StructuredPlan",
    mesh_plan: "StructuredMeshPlan",
    vol_tags: list[int],
) -> list[Issue]:
    """Each plan piece has a vol_tag; each vol_tag holds n_layers × N elements."""
    import gmsh

    issues: list[Issue] = []

    expected_pieces = sum(len(s.face_partition) for s in plan.slabs)
    if len(vol_tags) != expected_pieces:
        issues.append(Issue(
            severity="error",
            check="plan_mesh_consistency",
            message=(
                f"vol_tag count {len(vol_tags)} does not match expected "
                f"piece count {expected_pieces} from the plan."
            ),
            entities=(("vol_tags", tuple(vol_tags)),),
        ))
        return issues

    cursor = 0
    for slab_idx, slab in enumerate(plan.slabs):
        n_layers = int(mesh_plan.n_layers[slab_idx])
        for piece_idx in range(len(slab.face_partition)):
            vol = vol_tags[cursor]
            cursor += 1

            elem_types, elem_tags_per_type, _ = gmsh.model.mesh.getElements(3, vol)
            total = sum(len(t) for t in elem_tags_per_type)

            if total == 0:
                issues.append(Issue(
                    severity="error",
                    check="plan_mesh_consistency",
                    message=(
                        f"Slab {slab_idx} piece {piece_idx} (vol_tag {vol}) "
                        f"has zero elements; expected multiple of n_layers={n_layers}."
                    ),
                    entities=(("slab_piece", (slab_idx, piece_idx)),
                              ("vol_tag", vol)),
                ))
                continue

            if total % n_layers != 0:
                issues.append(Issue(
                    severity="error",
                    check="plan_mesh_consistency",
                    message=(
                        f"Slab {slab_idx} piece {piece_idx} (vol_tag {vol}) "
                        f"has {total} elements, not a multiple of n_layers={n_layers}."
                    ),
                    entities=(("slab_piece", (slab_idx, piece_idx)),
                              ("vol_tag", vol)),
                ))

    return issues
```

Wire it in `validate_structured_mesh`, after the quality block:

```python
    errors.extend(
        _check_plan_mesh_consistency(plan, mesh_plan, vol_tags)
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/structured/test_validator_plan_consistency.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/validator.py tests/structured/test_validator_plan_consistency.py
git commit -m "feat(validator): check 4 — plan ↔ mesh element count consistency"
```

---

## Task 6: Check 1 — watertight volume boundary

**Files:**
- Modify: `meshwell/structured/validator.py`
- Create: `tests/structured/test_validator_watertight.py`

Per `vol_tag`, every face from every 3D element must appear in either 1 element (boundary face of that volume) or 2 elements of the same volume (internal face). 3+ occurrences inside a single volume = geometric overlap.

We do NOT check boundary-face matching across volumes here — that's check 2's job. This check is single-volume integrity only.

- [ ] **Step 1: Write failing tests**

Create `tests/structured/test_validator_watertight.py`:

```python
"""Unit tests for check 1: watertight (per-volume face occurrence)."""
import gmsh
import pytest

from meshwell.structured.spec import (
    PhantomMap,
    StructuredMeshPlan,
    StructuredPlan,
)
from meshwell.structured.validator import validate_structured_mesh


@pytest.fixture
def empty_inputs():
    return (
        StructuredPlan(slabs=(), z_planes=(), overlaps=()),
        StructuredMeshPlan(slabs=(), n_layers=(), recombine=()),
        PhantomMap(),
    )


@pytest.fixture
def gmsh_session():
    gmsh.initialize()
    gmsh.model.add("watertight")
    yield
    gmsh.finalize()


def _add_single_tet(vol_tag: int):
    """Add a unit tet to vol_tag."""
    gmsh.model.mesh.addNodes(3, vol_tag,
        [1, 2, 3, 4],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    )
    gmsh.model.mesh.addElements(3, vol_tag, [4], [[10]], [[1, 2, 3, 4]])


def test_single_tet_passes_watertight(gmsh_session, empty_inputs):
    plan, mesh_plan, phantom_map = empty_inputs
    vol = gmsh.model.addDiscreteEntity(3, -1, [])
    _add_single_tet(vol)
    result = validate_structured_mesh(
        plan, mesh_plan, phantom_map, occ_entities=[], vol_tags=[vol], tol=1e-6,
    )
    wt_errors = [i for i in result.errors if i.check == "watertight"]
    assert wt_errors == []


def test_three_tets_sharing_a_face_reported_as_error(gmsh_session, empty_inputs):
    """Build 3 tets that share a single face — same volume, face count = 3."""
    plan, mesh_plan, phantom_map = empty_inputs
    vol = gmsh.model.addDiscreteEntity(3, -1, [])

    # 5 nodes: 4 form a "shared face" + apex; we add 3 distinct apex nodes
    # so 3 tets each contain the same base face.
    gmsh.model.mesh.addNodes(3, vol,
        [1, 2, 3, 4, 5, 6],
        [
            0, 0, 0,     # 1 (face)
            1, 0, 0,     # 2 (face)
            0, 1, 0,     # 3 (face)
            0, 0, 1,     # 4 apex a
            0, 0, -1,    # 5 apex b
            0, 0, 2,     # 6 apex c
        ],
    )
    gmsh.model.mesh.addElements(3, vol, [4],
        [[100, 101, 102]],
        [[1, 2, 3, 4,
          1, 2, 3, 5,
          1, 2, 3, 6]],
    )

    result = validate_structured_mesh(
        plan, mesh_plan, phantom_map, occ_entities=[], vol_tags=[vol], tol=1e-6,
    )
    wt_errors = [i for i in result.errors if i.check == "watertight"]
    assert len(wt_errors) >= 1
    assert any("3 elements" in i.message or "3 occurrences" in i.message
               for i in wt_errors)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/structured/test_validator_watertight.py -v`
Expected: first passes; second FAILS.

- [ ] **Step 3: Implement check in `meshwell/structured/validator.py`**

Add helper:

```python
# (element type, face type, nodes per face): from gmsh element type reference.
# - 4 = 4-node tet → 4 triangular faces (type 3, 3 nodes each)
# - 5 = 8-node hex → 6 quad faces (type 4, 4 nodes each)
# - 6 = 6-node prism → 2 triangular faces + 3 quad faces
# We use gmsh.model.mesh.getElementFaceNodes which handles this automatically.

def _check_watertight(vol_tags: list[int]) -> list[Issue]:
    """For each volume, every face appears in 1 or 2 elements of that volume."""
    import gmsh
    import numpy as np

    issues: list[Issue] = []

    for vol in vol_tags:
        elem_types, elem_tags_per_type, _ = gmsh.model.mesh.getElements(3, vol)
        if not elem_types:
            continue

        face_count: dict[frozenset[int], int] = {}
        for et in elem_types:
            # Triangular faces (face_type=3 in gmsh's terminology).
            try:
                tri_nodes = gmsh.model.mesh.getElementFaceNodes(et, 3, vol)
                tri_arr = np.asarray(tri_nodes, dtype=np.int64).reshape(-1, 3)
                for row in tri_arr:
                    key = frozenset(int(x) for x in row)
                    face_count[key] = face_count.get(key, 0) + 1
            except Exception:
                pass
            # Quad faces (face_type=4).
            try:
                quad_nodes = gmsh.model.mesh.getElementFaceNodes(et, 4, vol)
                quad_arr = np.asarray(quad_nodes, dtype=np.int64).reshape(-1, 4)
                for row in quad_arr:
                    key = frozenset(int(x) for x in row)
                    face_count[key] = face_count.get(key, 0) + 1
            except Exception:
                pass

        for face_key, count in face_count.items():
            if count > 2:
                issues.append(Issue(
                    severity="error",
                    check="watertight",
                    message=(
                        f"vol_tag {vol}: face shared by {count} elements "
                        f"(should be 1 boundary or 2 internal)."
                    ),
                    entities=(("vol_tag", vol),
                              ("face_nodes", tuple(sorted(face_key)))),
                ))

    return issues
```

Wire it in `validate_structured_mesh`, after `_check_plan_mesh_consistency`:

```python
    errors.extend(_check_watertight(vol_tags))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/structured/test_validator_watertight.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/validator.py tests/structured/test_validator_watertight.py
git commit -m "feat(validator): check 1 — watertight per-volume face occurrence"
```

---

## Task 7: Check 3 — internal seams unmeshed

**Files:**
- Modify: `meshwell/structured/validator.py`
- Create: `tests/structured/test_validator_seams.py`

The builder clears 2D mesh on faces that are internal seams between two pieces of the same slab (see `apply_structured_mesh` lines 408-430). The validator re-derives the same set from `phantom_map.output_laterals` and confirms each such face has zero 2D elements.

- [ ] **Step 1: Write failing tests**

Create `tests/structured/test_validator_seams.py`:

```python
"""Unit tests for check 3: internal-seam faces unmeshed."""
import gmsh
import pytest

from meshwell.structured.spec import (
    LateralKey,
    PhantomMap,
    StructuredMeshPlan,
    StructuredPlan,
)
from meshwell.structured.validator import validate_structured_mesh


@pytest.fixture
def empty_pm():
    return (
        StructuredPlan(slabs=(), z_planes=(), overlaps=()),
        StructuredMeshPlan(slabs=(), n_layers=(), recombine=()),
    )


@pytest.fixture
def gmsh_session():
    gmsh.initialize()
    gmsh.model.add("seams")
    yield
    gmsh.finalize()


class _FakeOccEntity:
    """Minimal stand-in for OCCLabeledEntity for the _map_phantom_laterals_to_gmsh lookup."""
    def __init__(self, dim, shapes):
        self.dim = dim
        self.shapes = shapes
        self.keep = True


def test_no_internal_seams_passes(gmsh_session, empty_pm):
    plan, mesh_plan = empty_pm
    pm = PhantomMap()  # No laterals.
    result = validate_structured_mesh(
        plan, mesh_plan, pm, occ_entities=[], vol_tags=[], tol=1e-6,
    )
    seam_issues = [i for i in result.errors if i.check == "internal_seam_unmeshed"]
    assert seam_issues == []


def test_internal_seam_with_2d_elements_reported(gmsh_session, empty_pm):
    """Construct a PhantomMap with a fake lateral key that maps to a face
    carrying 2D elements — the validator must report it."""
    plan, mesh_plan = empty_pm

    # Create a real surface entity with a triangle on it.
    face_tag = gmsh.model.addDiscreteEntity(2, -1, [])
    gmsh.model.mesh.addNodes(2, face_tag,
        [1, 2, 3],
        [0, 0, 0,  1, 0, 0,  0, 1, 0],
    )
    gmsh.model.mesh.addElements(2, face_tag, [2], [[10]], [[1, 2, 3]])

    # PhantomMap: two LateralKeys (same slab, different pieces) both
    # mapping to the same face_tag → that face IS an internal seam.
    pm = PhantomMap()
    key_a = LateralKey(slab_index=0, piece_index=0, outer_edge_index=0)
    key_b = LateralKey(slab_index=0, piece_index=1, outer_edge_index=0)
    # The validator uses `phantom_map.output_laterals[key] -> list[Any]`.
    # We need to give it something the lookup translates to face_tag.
    # Simplest: bypass the OCC→gmsh mapping by injecting the face_tag
    # directly. The validator's seam check should accept either OCC
    # TopoDS_Face values (real path) or int gmsh tags (test convenience)
    # — we use the int-passthrough path so the test doesn't need OCP.
    pm.output_laterals[key_a] = [face_tag]
    pm.output_laterals[key_b] = [face_tag]

    result = validate_structured_mesh(
        plan, mesh_plan, pm, occ_entities=[], vol_tags=[], tol=1e-6,
    )
    seam_errors = [i for i in result.errors if i.check == "internal_seam_unmeshed"]
    assert len(seam_errors) >= 1


def test_internal_seam_without_2d_elements_passes(gmsh_session, empty_pm):
    plan, mesh_plan = empty_pm
    face_tag = gmsh.model.addDiscreteEntity(2, -1, [])  # No elements.

    pm = PhantomMap()
    pm.output_laterals[LateralKey(0, 0, 0)] = [face_tag]
    pm.output_laterals[LateralKey(0, 1, 0)] = [face_tag]

    result = validate_structured_mesh(
        plan, mesh_plan, pm, occ_entities=[], vol_tags=[], tol=1e-6,
    )
    seam_errors = [i for i in result.errors if i.check == "internal_seam_unmeshed"]
    assert seam_errors == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/structured/test_validator_seams.py -v`
Expected: tests 1 and 3 pass (no check yet, but no error reported); test 2 FAILS.

- [ ] **Step 3: Implement check in `meshwell/structured/validator.py`**

Add helper:

```python
def _check_internal_seams_unmeshed(phantom_map: "PhantomMap") -> list[Issue]:
    """Faces shared between two pieces of the same slab must carry no 2D elements.

    Mirrors the detection logic in
    ``meshwell.structured.builder.apply_structured_mesh`` (the block
    that calls ``gmsh.model.mesh.clear([(2, face_tag)])`` for interior
    seam faces). Reports any face that the builder should have cleared
    but didn't (still has 2D elements).

    Accepts both real-pipeline values (TopoDS_Face mapped via
    occ_entities) and int gmsh tags (tests bypass the OCC layer).
    Real pipeline lookup uses ``_map_phantom_laterals_to_gmsh``; tests
    pass ints directly.
    """
    import gmsh

    issues: list[Issue] = []

    # Group LateralKey occurrences by underlying value (gmsh face tag or
    # OCC face). The builder clears any face that appears under >=2
    # keys of the same slab with different piece indices.
    value_to_keys: dict[Any, list[Any]] = {}
    for key, values in phantom_map.output_laterals.items():
        for v in values:
            value_to_keys.setdefault(v, []).append(key)

    for value, keys in value_to_keys.items():
        if len(keys) < 2:
            continue
        slabs = {k.slab_index for k in keys}
        pieces = {k.piece_index for k in keys}
        if len(slabs) != 1 or len(pieces) < 2:
            continue
        # This value identifies an internal seam face. Find its gmsh tag.
        face_tag = _resolve_face_tag(value)
        if face_tag is None:
            continue
        # Check it has no 2D elements.
        elem_types, elem_tags_per_type, _ = gmsh.model.mesh.getElements(2, face_tag)
        total_2d = sum(len(t) for t in elem_tags_per_type)
        if total_2d > 0:
            issues.append(Issue(
                severity="error",
                check="internal_seam_unmeshed",
                message=(
                    f"Internal seam face {face_tag} carries {total_2d} "
                    f"2D elements; should have been cleared by the builder."
                ),
                entities=(("face", face_tag),
                          ("seam_keys", tuple(sorted(
                              (k.slab_index, k.piece_index, k.outer_edge_index)
                              for k in keys
                          )))),
            ))

    return issues


def _resolve_face_tag(value: Any) -> int | None:
    """Return a gmsh face tag from a phantom_map.output_laterals value.

    Accepts an int (already a gmsh tag — test path) or an OCC
    TopoDS_Face (real pipeline path). Returns None if the value can't
    be resolved.
    """
    if isinstance(value, int):
        return value
    # Real pipeline: TopoDS_Face. We'd normally look this up via
    # _map_phantom_laterals_to_gmsh. To keep the validator self-contained
    # and avoid re-implementing that helper, we delegate when needed:
    try:
        from meshwell.structured.builder import _map_phantom_laterals_to_gmsh
        # Caller would normally do this once and cache. For Phase-1 keep
        # the call site simple; if profiling shows hot, cache later.
        # For the int-passthrough path we never reach here.
    except ImportError:
        return None
    return None  # Real-pipeline resolution wired in Task 8.
```

Wire it in `validate_structured_mesh`, after watertight:

```python
    errors.extend(_check_internal_seams_unmeshed(phantom_map))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/structured/test_validator_seams.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/validator.py tests/structured/test_validator_seams.py
git commit -m "feat(validator): check 3 — internal-seam faces carry no 2D elements"
```

---

## Task 8: Wire real-pipeline lateral resolution

**Files:**
- Modify: `meshwell/structured/validator.py`
- Modify: `tests/structured/test_validator_seams.py`

Task 7's `_resolve_face_tag` returns `None` for TopoDS_Face inputs. Real pipeline runs need to look up via `_map_phantom_laterals_to_gmsh(phantom_map, occ_entities)`. We compute that lookup once at the top of `validate_structured_mesh` and pass it through.

- [ ] **Step 1: Add failing test for real-pipeline path**

Append to `tests/structured/test_validator_seams.py`:

```python
def test_internal_seam_via_lateral_gmsh_map(gmsh_session, empty_pm, monkeypatch):
    """Pipeline path: PhantomMap values are TopoDS_Face; the validator
    resolves them to gmsh face tags via _map_phantom_laterals_to_gmsh.
    We monkeypatch that helper to return a known mapping."""
    plan, mesh_plan = empty_pm

    face_tag = gmsh.model.addDiscreteEntity(2, -1, [])
    gmsh.model.mesh.addNodes(2, face_tag, [1, 2, 3],
                              [0, 0, 0, 1, 0, 0, 0, 1, 0])
    gmsh.model.mesh.addElements(2, face_tag, [2], [[20]], [[1, 2, 3]])

    pm = PhantomMap()
    key_a = LateralKey(0, 0, 0)
    key_b = LateralKey(0, 1, 0)
    # Use sentinel objects to stand in for TopoDS_Face values.
    sentinel_face = object()
    pm.output_laterals[key_a] = [sentinel_face]
    pm.output_laterals[key_b] = [sentinel_face]

    def fake_map(phantom_map, occ_entities):
        return {key_a: [face_tag], key_b: [face_tag]}

    monkeypatch.setattr(
        "meshwell.structured.validator._map_phantom_laterals_to_gmsh",
        fake_map,
    )

    result = validate_structured_mesh(
        plan, mesh_plan, pm, occ_entities=[], vol_tags=[], tol=1e-6,
    )
    seam_errors = [i for i in result.errors if i.check == "internal_seam_unmeshed"]
    assert len(seam_errors) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/structured/test_validator_seams.py::test_internal_seam_via_lateral_gmsh_map -v`
Expected: FAIL (validator returns no seam_errors because `_resolve_face_tag` can't resolve the sentinel).

- [ ] **Step 3: Refactor `_check_internal_seams_unmeshed` to use the lateral map**

Replace `_check_internal_seams_unmeshed` and `_resolve_face_tag` with:

```python
# Import at module level (after the existing imports):
from meshwell.structured.builder import (
    _map_phantom_laterals_to_gmsh,  # noqa: F401 — re-exported for monkeypatching
)


def _check_internal_seams_unmeshed(
    phantom_map: "PhantomMap",
    occ_entities: list[Any],
) -> list[Issue]:
    """Faces shared between two pieces of the same slab must carry no 2D elements."""
    import gmsh

    issues: list[Issue] = []

    # Resolve lateral keys → gmsh face tags once.
    # Two code paths:
    # (a) Tests pass int gmsh tags directly in output_laterals values.
    # (b) Real pipeline passes TopoDS_Face values; use the lateral map.
    lateral_to_gmsh: dict[Any, list[int]] = {}
    use_direct_int_path = any(
        isinstance(v, int)
        for vals in phantom_map.output_laterals.values()
        for v in vals
    )
    if use_direct_int_path:
        lateral_to_gmsh = {
            k: [int(v) for v in vals if isinstance(v, int)]
            for k, vals in phantom_map.output_laterals.items()
        }
    elif phantom_map.output_laterals:
        lateral_to_gmsh = _map_phantom_laterals_to_gmsh(phantom_map, occ_entities)

    face_tag_to_keys: dict[int, list[Any]] = {}
    for key, tags in lateral_to_gmsh.items():
        for tag in tags:
            face_tag_to_keys.setdefault(int(tag), []).append(key)

    for face_tag, keys in face_tag_to_keys.items():
        if len(keys) < 2:
            continue
        slabs = {k.slab_index for k in keys}
        pieces = {k.piece_index for k in keys}
        if len(slabs) != 1 or len(pieces) < 2:
            continue
        elem_types, elem_tags_per_type, _ = gmsh.model.mesh.getElements(2, face_tag)
        total_2d = sum(len(t) for t in elem_tags_per_type)
        if total_2d > 0:
            issues.append(Issue(
                severity="error",
                check="internal_seam_unmeshed",
                message=(
                    f"Internal seam face {face_tag} carries {total_2d} "
                    f"2D elements; should have been cleared by the builder."
                ),
                entities=(("face", face_tag),
                          ("seam_keys", tuple(sorted(
                              (k.slab_index, k.piece_index, k.outer_edge_index)
                              for k in keys
                          )))),
            ))

    return issues
```

Update the call site in `validate_structured_mesh`:

```python
    errors.extend(_check_internal_seams_unmeshed(phantom_map, occ_entities))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/structured/test_validator_seams.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/validator.py tests/structured/test_validator_seams.py
git commit -m "feat(validator): resolve internal-seam laterals via _map_phantom_laterals_to_gmsh"
```

---

## Task 9: Check 2 — prism↔tet interface matching

**Files:**
- Modify: `meshwell/structured/validator.py`
- Create: `tests/structured/test_validator_interface.py`

For each face that's a slab boundary (a face whose 3D neighbours come from two different element types — wedge/hex on one side, tet on the other), the prism face nodes must be matched 1:1 (triangle) or by a 2-triangle split (quad) on the tet side. We don't need the phantom_map for this check — we can infer slab boundaries from element-type adjacency.

- [ ] **Step 1: Write failing tests**

Create `tests/structured/test_validator_interface.py`:

```python
"""Unit tests for check 2: prism ↔ tet interface conformality."""
import gmsh
import pytest

from meshwell.structured.spec import (
    PhantomMap,
    StructuredMeshPlan,
    StructuredPlan,
)
from meshwell.structured.validator import validate_structured_mesh


@pytest.fixture
def empty_pm():
    return (
        StructuredPlan(slabs=(), z_planes=(), overlaps=()),
        StructuredMeshPlan(slabs=(), n_layers=(), recombine=()),
        PhantomMap(),
    )


@pytest.fixture
def gmsh_session():
    gmsh.initialize()
    gmsh.model.add("interface")
    yield
    gmsh.finalize()


def test_conformal_wedge_tet_share_quad_passes(gmsh_session, empty_pm):
    """One wedge sharing its lateral quad face with two coplanar tets,
    same 4 nodes. Should pass."""
    plan, mesh_plan, pm = empty_pm

    # Shared quad nodes: 1, 2, 5, 4 (z=0 plane, x in {0,1})
    coords = [
        0, 0, 0,    # 1
        1, 0, 0,    # 2
        0, 1, 0,    # 3
        0, 0, 1,    # 4 (= node above 1)
        1, 0, 1,    # 5 (= node above 2)
        0, 1, 1,    # 6 (= node above 3)
        2, 0, 0,    # 7
        2, 0, 1,    # 8
    ]
    gmsh.model.mesh.addNodes(3, gmsh.model.addDiscreteEntity(3, -1, []),
        [1, 2, 3, 4, 5, 6, 7, 8], coords,
    )

    # Wedge on the left.
    wedge_vol = gmsh.model.addDiscreteEntity(3, -1, [])
    gmsh.model.mesh.addElements(3, wedge_vol, [6], [[100]],
        [[1, 2, 3, 4, 5, 6]])

    # Two tets on the right sharing the quad (1-2-5-4) split into triangles
    # (1-2-5) and (1-5-4), each completed with apex node 7 or 8.
    tet_vol = gmsh.model.addDiscreteEntity(3, -1, [])
    gmsh.model.mesh.addElements(3, tet_vol, [4], [[200, 201, 202, 203]],
        [[
            1, 2, 5, 7,
            1, 5, 4, 8,
            7, 8, 5, 2,
            7, 1, 5, 8,
        ]])

    result = validate_structured_mesh(
        plan, mesh_plan, pm, occ_entities=[],
        vol_tags=[wedge_vol, tet_vol], tol=1e-6,
    )
    iface_errors = [i for i in result.errors if i.check == "prism_tet_interface"]
    # No T-junction or hanging node — should pass.
    assert iface_errors == []


def test_wedge_quad_with_t_junction_on_tet_side_reported(gmsh_session, empty_pm):
    """Wedge quad face (1-2-5-4) but tet side introduces extra node 99 on
    edge 1-2, splitting into a non-matching triangulation."""
    plan, mesh_plan, pm = empty_pm

    coords = [
        0, 0, 0,    # 1
        1, 0, 0,    # 2
        0, 1, 0,    # 3
        0, 0, 1,    # 4
        1, 0, 1,    # 5
        0, 1, 1,    # 6
        2, 0, 0,    # 7
        0.5, 0, 0,  # 99 — Steiner node on shared edge (T-junction!)
    ]
    gmsh.model.mesh.addNodes(3, gmsh.model.addDiscreteEntity(3, -1, []),
        [1, 2, 3, 4, 5, 6, 7, 99], coords,
    )

    wedge_vol = gmsh.model.addDiscreteEntity(3, -1, [])
    gmsh.model.mesh.addElements(3, wedge_vol, [6], [[100]],
        [[1, 2, 3, 4, 5, 6]])

    tet_vol = gmsh.model.addDiscreteEntity(3, -1, [])
    # 3 tets on the right, with the shared face built using node 99 instead
    # of node 1 or 2 — so the wedge's quad-face {1,2,5,4} is NOT matched by
    # any set of two tet triangles sharing the same 4 nodes.
    gmsh.model.mesh.addElements(3, tet_vol, [4], [[200, 201, 202]],
        [[
            99, 2, 5, 7,
            1, 99, 5, 7,
            1, 5, 4, 7,
        ]])

    result = validate_structured_mesh(
        plan, mesh_plan, pm, occ_entities=[],
        vol_tags=[wedge_vol, tet_vol], tol=1e-6,
    )
    iface_errors = [i for i in result.errors if i.check == "prism_tet_interface"]
    assert len(iface_errors) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/structured/test_validator_interface.py -v`
Expected: first passes (no check yet); second FAILS.

- [ ] **Step 3: Implement check in `meshwell/structured/validator.py`**

Add helper:

```python
# Gmsh element type codes used here.
_ELEM_TET = 4    # 4-node tet
_ELEM_HEX = 5    # 8-node hex
_ELEM_PRISM = 6  # 6-node prism
_STRUCTURED_TYPES = (_ELEM_HEX, _ELEM_PRISM)


def _check_prism_tet_interface(vol_tags: list[int]) -> list[Issue]:
    """Faces shared between a structured volume (wedge/hex) and a tet
    volume must be matched 1:1 (triangle face) or 2 triangles on the
    tet side spanning the same 4 nodes (quad face).
    """
    import gmsh
    import numpy as np

    if not vol_tags:
        return []

    issues: list[Issue] = []

    # Classify each volume by its element types.
    vol_kind: dict[int, str] = {}
    for vol in vol_tags:
        elem_types, _, _ = gmsh.model.mesh.getElements(3, vol)
        types = set(int(t) for t in elem_types)
        if not types:
            continue
        if types & set(_STRUCTURED_TYPES):
            vol_kind[vol] = "structured"
        elif _ELEM_TET in types:
            vol_kind[vol] = "tet"
        else:
            vol_kind[vol] = "other"

    # Collect all faces per volume: structured (tri + quad) vs. tet (tri).
    tri_faces_by_vol: dict[int, dict[frozenset[int], list[int]]] = {}
    quad_faces_by_vol: dict[int, dict[frozenset[int], list[int]]] = {}

    for vol, kind in vol_kind.items():
        elem_types, elem_tags_per_type, _ = gmsh.model.mesh.getElements(3, vol)
        tri_map: dict[frozenset[int], list[int]] = {}
        quad_map: dict[frozenset[int], list[int]] = {}
        for et, tags_for_type in zip(elem_types, elem_tags_per_type):
            try:
                tri = gmsh.model.mesh.getElementFaceNodes(int(et), 3, vol)
                tri_arr = np.asarray(tri, dtype=np.int64).reshape(-1, 3)
                # 1 face per element for tets; 2 faces (top + bottom) per prism.
                per_elem = tri_arr.shape[0] // len(tags_for_type) if len(tags_for_type) else 0
                for k, row in enumerate(tri_arr):
                    key = frozenset(int(x) for x in row)
                    elem_tag = int(tags_for_type[k // max(per_elem, 1)])
                    tri_map.setdefault(key, []).append(elem_tag)
            except Exception:
                pass
            try:
                quad = gmsh.model.mesh.getElementFaceNodes(int(et), 4, vol)
                quad_arr = np.asarray(quad, dtype=np.int64).reshape(-1, 4)
                per_elem = quad_arr.shape[0] // len(tags_for_type) if len(tags_for_type) else 0
                for k, row in enumerate(quad_arr):
                    key = frozenset(int(x) for x in row)
                    elem_tag = int(tags_for_type[k // max(per_elem, 1)])
                    quad_map.setdefault(key, []).append(elem_tag)
            except Exception:
                pass
        tri_faces_by_vol[vol] = tri_map
        quad_faces_by_vol[vol] = quad_map

    # For each structured volume, find faces that bound it (count == 1 in
    # its own face map) — those that don't match a counterpart on a tet
    # volume are non-conformal.
    structured_vols = [v for v, k in vol_kind.items() if k == "structured"]
    tet_vols = [v for v, k in vol_kind.items() if k == "tet"]

    for sv in structured_vols:
        # Triangle faces: must exactly match a triangle face on some tet vol.
        for face_key, owners in tri_faces_by_vol[sv].items():
            if len(owners) != 1:
                continue  # Internal to the structured volume.
            matched = any(face_key in tri_faces_by_vol[tv] for tv in tet_vols)
            # Only flag if there's no neighbouring tet at all and this is
            # genuinely shared (any tet element references >=1 of these nodes).
            if not matched and tet_vols:
                referenced = any(
                    face_key & fk
                    for tv in tet_vols
                    for fk in tri_faces_by_vol[tv]
                )
                if referenced:
                    issues.append(Issue(
                        severity="error",
                        check="prism_tet_interface",
                        message=(
                            f"Structured vol {sv} triangle face has no exact "
                            f"match in any tet volume (possible T-junction)."
                        ),
                        entities=(("vol_tag", sv),
                                  ("face_nodes", tuple(sorted(face_key)))),
                    ))

        # Quad faces: must be split into exactly 2 tet triangles on the
        # other side, both triangles' node sets contained in the quad's 4.
        for face_key, owners in quad_faces_by_vol[sv].items():
            if len(owners) != 1:
                continue
            covering_tris: list[frozenset[int]] = []
            for tv in tet_vols:
                for tri_key in tri_faces_by_vol[tv]:
                    if tri_key.issubset(face_key):
                        covering_tris.append(tri_key)
            if not tet_vols:
                continue
            # We expect exactly 2 covering tris whose union == face_key.
            if len(covering_tris) == 0:
                # No tet face referenced — only an error if the quad is
                # in contact with tet (any node shared).
                touches_tet = any(
                    face_key & fk
                    for tv in tet_vols
                    for fk in tri_faces_by_vol[tv]
                )
                if touches_tet:
                    issues.append(Issue(
                        severity="error",
                        check="prism_tet_interface",
                        message=(
                            f"Structured vol {sv} quad face has 0 covering tet "
                            f"triangles (T-junction or missing element)."
                        ),
                        entities=(("vol_tag", sv),
                                  ("face_nodes", tuple(sorted(face_key)))),
                    ))
            elif len(covering_tris) != 2:
                issues.append(Issue(
                    severity="error",
                    check="prism_tet_interface",
                    message=(
                        f"Structured vol {sv} quad face has "
                        f"{len(covering_tris)} covering tet triangles "
                        f"(expected 2 for clean quad split)."
                    ),
                    entities=(("vol_tag", sv),
                              ("face_nodes", tuple(sorted(face_key)))),
                ))
            else:
                # Two covering tris — verify their union is the quad.
                union = covering_tris[0] | covering_tris[1]
                if union != face_key:
                    issues.append(Issue(
                        severity="error",
                        check="prism_tet_interface",
                        message=(
                            f"Structured vol {sv} quad face: two covering tet "
                            f"triangles don't span the full quad (extra Steiner node)."
                        ),
                        entities=(("vol_tag", sv),
                                  ("face_nodes", tuple(sorted(face_key))),
                                  ("tet_tris", tuple(tuple(sorted(t)) for t in covering_tris))),
                    ))

    return issues
```

Wire it in `validate_structured_mesh`, after internal seams:

```python
    errors.extend(_check_prism_tet_interface(vol_tags))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/structured/test_validator_interface.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/validator.py tests/structured/test_validator_interface.py
git commit -m "feat(validator): check 2 — prism/tet interface face matching"
```

---

## Task 10: Check 6 — geometric localization on check-2 failures

**Files:**
- Modify: `meshwell/structured/validator.py`
- Modify: `tests/structured/test_validator_interface.py`

When check 2 reports a face mismatch, run a geometric pass on that specific face to refine the diagnosis: (a) no candidate at all = hole; (b) candidate exists at matching coords but with different node IDs = duplicate-node bug; (c) candidate exists with coords offset = misalignment.

- [ ] **Step 1: Add failing test**

Append to `tests/structured/test_validator_interface.py`:

```python
def test_check2_failure_is_geometrically_localized(gmsh_session, empty_pm):
    """When check 2 fails, the report should mention the geometric refinement."""
    plan, mesh_plan, pm = empty_pm

    # Two nodes at nearly the same place (different IDs) form a face on each side.
    coords = [
        0, 0, 0,        # 1
        1, 0, 0,        # 2
        0, 1, 0,        # 3
        0, 0, 1,        # 4
        1, 0, 1,        # 5
        0, 1, 1,        # 6
        0, 0, 0.0001,   # 7 — near-duplicate of node 1 (different id)
        2, 0, 0,        # 8
    ]
    gmsh.model.mesh.addNodes(3, gmsh.model.addDiscreteEntity(3, -1, []),
        [1, 2, 3, 4, 5, 6, 7, 8], coords,
    )

    wedge_vol = gmsh.model.addDiscreteEntity(3, -1, [])
    gmsh.model.mesh.addElements(3, wedge_vol, [6], [[100]],
        [[1, 2, 3, 4, 5, 6]])

    tet_vol = gmsh.model.addDiscreteEntity(3, -1, [])
    # Tet uses node 7 (near-duplicate of 1) instead of node 1.
    gmsh.model.mesh.addElements(3, tet_vol, [4], [[200, 201]],
        [[
            7, 2, 5, 8,
            7, 5, 4, 8,
        ]])

    result = validate_structured_mesh(
        plan, mesh_plan, pm, occ_entities=[],
        vol_tags=[wedge_vol, tet_vol], tol=1e-3,  # tol large enough to catch the offset
    )
    iface_errors = [i for i in result.errors if i.check == "prism_tet_interface"]
    assert any("near-duplicate" in i.message.lower()
               or "duplicate" in i.message.lower()
               or "candidate exists at matching coords" in i.message.lower()
               for i in iface_errors)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/structured/test_validator_interface.py::test_check2_failure_is_geometrically_localized -v`
Expected: FAIL — current check 2 message doesn't mention near-duplicate.

- [ ] **Step 3: Add geometric refinement to `_check_prism_tet_interface`**

Inside the helper, after the existing error append for "0 covering tet triangles", call `_localize_face_mismatch(face_key, tet_vols, tol)` and merge its message into the issue. Add the helper:

```python
def _localize_face_mismatch(
    face_key: frozenset[int],
    tet_vols: list[int],
    tol: float,
) -> str:
    """Given a structured face that check 2 couldn't match topologically,
    look for a geometrically-matching tet-side face within ``tol``.
    Returns a short refinement string to append to the error message.
    """
    import gmsh
    import numpy as np

    # Coords of the structured face's nodes.
    struct_coords: list[np.ndarray] = []
    for n in face_key:
        try:
            xyz, _, _, _ = gmsh.model.mesh.getNode(int(n))
            struct_coords.append(np.asarray(xyz, dtype=float))
        except Exception:
            continue
    if not struct_coords:
        return ""
    struct_pts = np.stack(struct_coords)

    # Scan all triangle faces on tet vols whose node coords fall within tol.
    matched_node_ids: list[int] = []
    for tv in tet_vols:
        elem_types, elem_tags_per_type, _ = gmsh.model.mesh.getElements(3, tv)
        for et in elem_types:
            try:
                tri = gmsh.model.mesh.getElementFaceNodes(int(et), 3, tv)
                tri_arr = np.asarray(tri, dtype=np.int64).reshape(-1, 3)
            except Exception:
                continue
            for row in tri_arr:
                for n in row:
                    if int(n) in face_key:
                        continue
                    try:
                        xyz, _, _, _ = gmsh.model.mesh.getNode(int(n))
                        p = np.asarray(xyz, dtype=float)
                    except Exception:
                        continue
                    dists = np.linalg.norm(struct_pts - p, axis=1)
                    if float(dists.min()) <= tol:
                        matched_node_ids.append(int(n))

    if matched_node_ids:
        return (
            f" Localization: candidate exists at matching coords but with "
            f"different node IDs (near-duplicate on tet side: nodes "
            f"{sorted(set(matched_node_ids))[:5]})."
        )
    return " Localization: no candidate within tol — likely hole or misalignment."
```

Modify the existing `len(covering_tris) == 0` case to call `_localize_face_mismatch`:

```python
                if touches_tet:
                    refinement = _localize_face_mismatch(face_key, tet_vols, tol)
                    issues.append(Issue(
                        severity="error",
                        check="prism_tet_interface",
                        message=(
                            f"Structured vol {sv} quad face has 0 covering tet "
                            f"triangles (T-junction or missing element)." + refinement
                        ),
                        entities=(("vol_tag", sv),
                                  ("face_nodes", tuple(sorted(face_key)))),
                    ))
```

And similarly for the triangle case ("Structured vol ... triangle face has no exact match ..."). Pass `tol` into `_check_prism_tet_interface`:

```python
def _check_prism_tet_interface(vol_tags: list[int], tol: float) -> list[Issue]:
```

Update the call site:

```python
    errors.extend(_check_prism_tet_interface(vol_tags, resolved_tol))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/structured/test_validator_interface.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/validator.py tests/structured/test_validator_interface.py
git commit -m "feat(validator): check 6 — geometric localization of check-2 failures"
```

---

## Task 11: Check 8 — top↔bottom symmetry

**Files:**
- Modify: `meshwell/structured/validator.py`
- Create: `tests/structured/test_validator_top_bottom_symmetry.py`

For each structured volume (`vol_kind == "structured"`), the bottom-z slice of nodes and the top-z slice of nodes must have the same (x, y) within `tol`. We don't need the phantom_map — we identify "bottom" and "top" as the planes `z == min(z_in_vol)` and `z == max(z_in_vol)` for each structured vol.

- [ ] **Step 1: Write failing tests**

Create `tests/structured/test_validator_top_bottom_symmetry.py`:

```python
"""Unit tests for check 8: top↔bottom z-translation symmetry."""
import gmsh
import pytest

from meshwell.structured.spec import (
    PhantomMap,
    StructuredMeshPlan,
    StructuredPlan,
)
from meshwell.structured.validator import validate_structured_mesh


@pytest.fixture
def empty_pm():
    return (
        StructuredPlan(slabs=(), z_planes=(), overlaps=()),
        StructuredMeshPlan(slabs=(), n_layers=(), recombine=()),
        PhantomMap(),
    )


@pytest.fixture
def gmsh_session():
    gmsh.initialize()
    gmsh.model.add("symmetry")
    yield
    gmsh.finalize()


def test_symmetric_wedge_passes(gmsh_session, empty_pm):
    plan, mesh_plan, pm = empty_pm
    vol = gmsh.model.addDiscreteEntity(3, -1, [])
    coords = [
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        1, 0, 1,
        0, 1, 1,
    ]
    gmsh.model.mesh.addNodes(3, vol, [1, 2, 3, 4, 5, 6], coords)
    gmsh.model.mesh.addElements(3, vol, [6], [[100]], [[1, 2, 3, 4, 5, 6]])

    result = validate_structured_mesh(
        plan, mesh_plan, pm, occ_entities=[], vol_tags=[vol], tol=1e-6,
    )
    sym_errors = [i for i in result.errors if i.check == "top_bottom_symmetry"]
    assert sym_errors == []


def test_misaligned_top_reported(gmsh_session, empty_pm):
    plan, mesh_plan, pm = empty_pm
    vol = gmsh.model.addDiscreteEntity(3, -1, [])
    coords = [
        0,     0, 0,
        1,     0, 0,
        0,     1, 0,
        0.01,  0, 1,   # x offset by 0.01 — > tol=1e-3
        1.01,  0, 1,
        0.01,  1, 1,
    ]
    gmsh.model.mesh.addNodes(3, vol, [1, 2, 3, 4, 5, 6], coords)
    gmsh.model.mesh.addElements(3, vol, [6], [[100]], [[1, 2, 3, 4, 5, 6]])

    result = validate_structured_mesh(
        plan, mesh_plan, pm, occ_entities=[], vol_tags=[vol], tol=1e-3,
    )
    sym_errors = [i for i in result.errors if i.check == "top_bottom_symmetry"]
    assert len(sym_errors) >= 1


def test_non_structured_volume_skipped(gmsh_session, empty_pm):
    """Tet volumes are not subject to z-translation symmetry."""
    plan, mesh_plan, pm = empty_pm
    vol = gmsh.model.addDiscreteEntity(3, -1, [])
    gmsh.model.mesh.addNodes(3, vol, [1, 2, 3, 4],
                              [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1])
    gmsh.model.mesh.addElements(3, vol, [4], [[100]], [[1, 2, 3, 4]])

    result = validate_structured_mesh(
        plan, mesh_plan, pm, occ_entities=[], vol_tags=[vol], tol=1e-6,
    )
    sym_errors = [i for i in result.errors if i.check == "top_bottom_symmetry"]
    assert sym_errors == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/structured/test_validator_top_bottom_symmetry.py -v`
Expected: 1 and 3 pass; 2 FAILS.

- [ ] **Step 3: Implement check in `meshwell/structured/validator.py`**

Add helper:

```python
def _check_top_bottom_symmetry(vol_tags: list[int], tol: float) -> list[Issue]:
    """Each structured volume's top-z and bottom-z node slices must
    differ only in z (within tol).
    """
    import gmsh
    import numpy as np

    issues: list[Issue] = []

    for vol in vol_tags:
        elem_types, _, _ = gmsh.model.mesh.getElements(3, vol)
        types = set(int(t) for t in elem_types)
        if not (types & set(_STRUCTURED_TYPES)):
            continue

        node_tags_arr, coords_flat, _ = gmsh.model.mesh.getNodes(3, vol, includeBoundary=True)
        node_tags = np.asarray(node_tags_arr, dtype=np.int64)
        coords = np.asarray(coords_flat, dtype=float).reshape(-1, 3)
        if coords.shape[0] < 6:
            continue

        z_min = float(np.min(coords[:, 2]))
        z_max = float(np.max(coords[:, 2]))
        bot_mask = np.abs(coords[:, 2] - z_min) <= tol
        top_mask = np.abs(coords[:, 2] - z_max) <= tol
        bot_xy = coords[bot_mask][:, :2]
        top_xy = coords[top_mask][:, :2]

        if bot_xy.shape[0] != top_xy.shape[0]:
            issues.append(Issue(
                severity="error",
                check="top_bottom_symmetry",
                message=(
                    f"Structured vol {vol}: bottom has {bot_xy.shape[0]} nodes "
                    f"but top has {top_xy.shape[0]} (non-isomorphic slices)."
                ),
                entities=(("vol_tag", vol),),
            ))
            continue

        # For each bot xy, find nearest top xy. Max distance should be <= tol.
        max_resid = 0.0
        for i in range(bot_xy.shape[0]):
            d = np.linalg.norm(top_xy - bot_xy[i:i+1], axis=1)
            max_resid = max(max_resid, float(d.min()))
        if max_resid > tol:
            issues.append(Issue(
                severity="error",
                check="top_bottom_symmetry",
                message=(
                    f"Structured vol {vol}: top↔bottom xy residual "
                    f"{max_resid:.3e} exceeds tol {tol:.3e}."
                ),
                entities=(("vol_tag", vol),),
            ))

    return issues
```

Wire it in `validate_structured_mesh`, after the interface block:

```python
    errors.extend(_check_top_bottom_symmetry(vol_tags, resolved_tol))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/structured/test_validator_top_bottom_symmetry.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/validator.py tests/structured/test_validator_top_bottom_symmetry.py
git commit -m "feat(validator): check 8 — top↔bottom z-translation symmetry"
```

---

## Task 12: End-to-end integration into existing structured tests

**Files:**
- Modify: `tests/structured/test_end_to_end_minimal.py`
- Modify: `tests/structured/test_end_to_end_multipiece.py`

Hook the validator into the two existing structured end-to-end tests. Every structured mesh built in CI now polices its own conformality.

- [ ] **Step 1: Read existing end-to-end tests to find their assertion points**

Read both files to find where `apply_structured_mesh` is called and what return values are available. Locate the position after mesh generation but before file write.

Run: `grep -n "apply_structured_mesh\|generate(3)\|gmsh.write" tests/structured/test_end_to_end_minimal.py tests/structured/test_end_to_end_multipiece.py`

- [ ] **Step 2: Append a validator call to the minimal end-to-end test**

In `tests/structured/test_end_to_end_minimal.py`, just before the test's final assertions (the ones that inspect the produced `.msh` file), insert:

```python
from meshwell.structured.validator import validate_structured_mesh
result = validate_structured_mesh(
    plan, mesh_plan, phantom_map, occ_entities, vol_tags,
)
assert result, result.format_report()
```

(Adapt the variable names to whatever the existing test uses — likely `structured_plan`, `mesh_plan`, `phantom_map`, `occ_entities`, `vol_tags` based on `apply_structured_mesh`'s signature.)

- [ ] **Step 3: Run the minimal end-to-end test to verify it still passes**

Run: `pytest tests/structured/test_end_to_end_minimal.py -v`
Expected: PASS. If it fails, the validator caught a real conformality bug in the pipeline — investigate before proceeding. Treat failure as a finding, not a test bug.

- [ ] **Step 4: Repeat for multipiece end-to-end**

Apply the same pattern to `tests/structured/test_end_to_end_multipiece.py`.

Run: `pytest tests/structured/test_end_to_end_multipiece.py -v`
Expected: PASS.

- [ ] **Step 5: Run the whole structured test suite to confirm no regressions**

Run: `pytest tests/structured/ -v`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add tests/structured/test_end_to_end_minimal.py tests/structured/test_end_to_end_multipiece.py
git commit -m "test(structured): assert validator passes in end-to-end tests"
```

---

## Task 13: Documentation polish

**Files:**
- Modify: `meshwell/structured/validator.py`

One pass to clean up the module's top-level docstring (now that the module is complete) and to ensure `validate_structured_mesh`'s docstring lists all eight checks with a one-liner each.

- [ ] **Step 1: Update the module docstring**

Replace the module docstring in `meshwell/structured/validator.py` with:

```python
"""Conformality validator for structured-polyprism meshes.

Runs in the live gmsh session immediately after
``meshwell.structured.builder.apply_structured_mesh``. Reports
topological and geometric conformality failures between the
structured wedge/hex slabs and the surrounding tet regions.

Public API:

- :class:`Issue` — one validation finding.
- :class:`ValidationResult` — collected errors + warnings + report formatter.
- :func:`validate_structured_mesh` — entry point.

Checks performed (in order):

1. Watertight per-volume face occurrence (each face appears in 1 or 2 elements of the volume).
2. Prism↔tet interface face matching (triangle 1:1, quad split into 2 tet triangles).
3. Internal-seam faces (between two pieces of the same slab) carry no 2D elements.
4. Plan↔mesh element-count consistency (each piece's elements is a multiple of n_layers).
5. Near-duplicate node detection (exact = error, within-tol = warning).
6. Geometric localization of check-2 failures.
7. Element quality via gmsh.model.mesh.getElementQualities (opt-in).
8. Top↔bottom z-translation symmetry (structured volumes only).
"""
```

- [ ] **Step 2: Run the whole test suite one more time**

Run: `pytest tests/structured/ -v`
Expected: all green.

- [ ] **Step 3: Commit**

```bash
git add meshwell/structured/validator.py
git commit -m "docs(validator): list all eight checks in module docstring"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task(s) covering it |
| --- | --- |
| API surface (Issue, ValidationResult, validate_structured_mesh) | Task 1 |
| Tolerance default + override | Task 2 |
| Check 1 (watertight) | Task 6 |
| Check 2 (prism↔tet interface) | Task 9 |
| Check 3 (internal seams unmeshed) | Task 7, Task 8 |
| Check 4 (plan↔mesh consistency) | Task 5 |
| Check 5 (near-duplicate nodes) | Task 3 |
| Check 6 (geometric localization) | Task 10 |
| Check 7 (opt-in quality) | Task 4 |
| Check 8 (top↔bottom symmetry) | Task 11 |
| End-to-end integration | Task 12 |

All eight checks plus the supporting API are covered.

**Placeholder scan:** No "TBD" / "implement later" / "similar to Task N" found.

**Type consistency:** Issue uses `severity: Literal["error", "warning"]` everywhere; `check` is a string constant per task; `entities` is always a tuple of (kind, value) pairs. Function signature for `_check_prism_tet_interface` evolves in Task 10 to add `tol`; the call site update is included in that task.

One thing worth flagging for the executing agent: Task 12 (end-to-end integration) is the highest-risk step — if the validator finds a real conformality bug, the test will fail. That is the **correct** outcome and should be investigated, not worked around. The task instructions call this out explicitly.

---

## Execution choice

**Plan complete and saved to `docs/superpowers/plans/2026-05-17-structured-mesh-validator.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
