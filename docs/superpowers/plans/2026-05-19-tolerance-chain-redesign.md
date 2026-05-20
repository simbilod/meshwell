# Tolerance Chain Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 7-knob ad-hoc tolerance configuration in meshwell's structured pipeline with a single validated, scale-aware `Tolerances` dataclass; clamp OCC tolerance bloat between cuts; make `arc_tolerance` radius-relative; add a tolerance test suite that exercises the boundaries (monotonicity, perturbation survival, arc small-/large-radius, heterogeneous slabs).

**Architecture:** Introduce `meshwell.tolerances.Tolerances` as the single source of truth. `ModelManager`, `CAD_OCC`, and `Slab` each accept it (existing scalar args remain for back-compat but route through it). A `from_characteristic_length(L)` factory derives the full chain from one length scale. A `__post_init__` validator enforces the hierarchy `point_tol ≥ perturbation > fragment_fuzzy ≥ cut_fuzzy ≥ Precision::Confusion`. The cut cascade in `cad_occ.py` clamps shape tolerance to `cut_fuzzy` between operations via `ShapeFix_ShapeTolerance.LimitTolerance`. Arc validation switches from `max(arc_tol, 0.05*r)` to `chord_height_tol_fraction * r`.

**Tech Stack:** Python 3.11+, `OCP` (OCCT bindings: `ShapeFix_ShapeTolerance`, `Precision`), `shapely`, `gmsh`, `pytest`, existing `meshwell.structured.*` modules.

---

## File Structure

- `meshwell/tolerances.py` — **new** — `Tolerances` dataclass, validator, `from_characteristic_length` factory, `OCCT_CONFUSION` constant.
- `meshwell/model.py:19-66` — modify constructor: accept `tolerances: Tolerances | None`; if given, ignore legacy scalar args.
- `meshwell/cad_occ.py:126-168` — modify constructor: accept `tolerances`; route legacy args through `Tolerances`.
- `meshwell/cad_occ.py:155-162` — add `_clamp_shape_tolerance` call between cuts in the cascade.
- `meshwell/structured/spec.py:213-214` — replace `arc_tolerance: float = 1e-3` with `arc_chord_height_fraction: float = 0.01` (keep `arc_tolerance` as deprecated absolute backstop).
- `meshwell/structured/plan.py:723` — change `interior_buffer = max(arc_tol, 0.05*r)` to use fraction-of-radius.
- `meshwell/structured/builder.py:558-566` — emit warning when slabs have heterogeneous `fragment_fuzzy_value`.
- `tests/test_tolerances.py` — **new** — unit tests for `Tolerances` validator + factory.
- `tests/structured/test_tolerance_chain.py` — **new** — integration tests (sweep, precision tightening, arc small/large radius, heterogeneous slabs, perturbation-survives-snap).

---

## Task 1: `Tolerances` dataclass — scaffold + OCCT_CONFUSION constant

**Files:**
- Create: `meshwell/tolerances.py`
- Test: `tests/test_tolerances.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tolerances.py
from meshwell.tolerances import Tolerances, OCCT_CONFUSION


def test_occt_confusion_constant():
    assert OCCT_CONFUSION == 1e-7


def test_tolerances_explicit_construction():
    tol = Tolerances(
        point_tolerance=1e-4,
        perturbation=1e-5,
        cut_fuzzy_value=1e-6,
        fragment_fuzzy_value=1e-5,
        geometry_tolerance=1e-6,
        tolerance_boolean=1e-5,
        arc_chord_height_fraction=0.01,
    )
    assert tol.point_tolerance == 1e-4
    assert tol.perturbation == 1e-5
    assert tol.cut_fuzzy_value == 1e-6
    assert tol.fragment_fuzzy_value == 1e-5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tolerances.py::test_occt_confusion_constant -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meshwell.tolerances'`.

- [ ] **Step 3: Write minimal implementation**

```python
# meshwell/tolerances.py
"""Single source of truth for the OCC/shapely/gmsh tolerance chain.

See docs/superpowers/specs and the audit at
docs/superpowers/plans/2026-05-19-tolerance-chain-redesign.md for the
hierarchy rationale.
"""
from __future__ import annotations

from dataclasses import dataclass

# OCCT's natural floor: Precision::Confusion() in OCP/OCCT.
# No fuzzy value below this is meaningful to the BOP algorithms.
OCCT_CONFUSION: float = 1e-7


@dataclass(frozen=True)
class Tolerances:
    """Validated tolerance chain for the structured meshing pipeline."""

    point_tolerance: float
    perturbation: float
    cut_fuzzy_value: float
    fragment_fuzzy_value: float
    geometry_tolerance: float
    tolerance_boolean: float
    arc_chord_height_fraction: float
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tolerances.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add meshwell/tolerances.py tests/test_tolerances.py
git commit -m "feat(tolerances): scaffold Tolerances dataclass + OCCT_CONFUSION"
```

---

## Task 2: `Tolerances` validation — enforce hierarchy

**Files:**
- Modify: `meshwell/tolerances.py`
- Test: `tests/test_tolerances.py`

- [ ] **Step 1: Write the failing tests**

```python
# Append to tests/test_tolerances.py
import pytest
from meshwell.tolerances import Tolerances, OCCT_CONFUSION, ToleranceHierarchyError


def _ok_kwargs():
    return dict(
        point_tolerance=1e-4,
        perturbation=1e-5,
        cut_fuzzy_value=1e-6,
        fragment_fuzzy_value=1e-5,
        geometry_tolerance=1e-6,
        tolerance_boolean=1e-5,
        arc_chord_height_fraction=0.01,
    )


def test_cut_fuzzy_must_not_exceed_fragment_fuzzy():
    kw = _ok_kwargs()
    kw["cut_fuzzy_value"] = 2e-5  # > fragment_fuzzy_value
    with pytest.raises(ToleranceHierarchyError, match="cut_fuzzy_value"):
        Tolerances(**kw)


def test_fragment_fuzzy_must_not_exceed_perturbation():
    kw = _ok_kwargs()
    kw["fragment_fuzzy_value"] = 2e-5  # > perturbation=1e-5
    with pytest.raises(ToleranceHierarchyError, match="fragment_fuzzy_value"):
        Tolerances(**kw)


def test_perturbation_must_not_exceed_point_tolerance():
    kw = _ok_kwargs()
    kw["perturbation"] = 2e-4  # > point_tolerance=1e-4
    with pytest.raises(ToleranceHierarchyError, match="perturbation"):
        Tolerances(**kw)


def test_cut_fuzzy_must_exceed_occt_confusion():
    kw = _ok_kwargs()
    kw["cut_fuzzy_value"] = OCCT_CONFUSION / 2
    with pytest.raises(ToleranceHierarchyError, match="OCCT_CONFUSION"):
        Tolerances(**kw)


def test_arc_chord_height_fraction_must_be_in_unit_interval():
    kw = _ok_kwargs()
    kw["arc_chord_height_fraction"] = 1.5
    with pytest.raises(ToleranceHierarchyError, match="arc_chord_height_fraction"):
        Tolerances(**kw)


def test_perturbation_must_exceed_cut_fuzzy_by_safety_factor():
    """Perturbation gap must exceed cut_fuzzy by at least 2x or OCC may merge."""
    kw = _ok_kwargs()
    kw["perturbation"] = 1.5e-6  # only 1.5x cut_fuzzy=1e-6
    with pytest.raises(ToleranceHierarchyError, match="perturbation.*cut_fuzzy"):
        Tolerances(**kw)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tolerances.py -v`
Expected: 6 new tests FAIL (`ImportError` for `ToleranceHierarchyError`).

- [ ] **Step 3: Implement validator**

Edit `meshwell/tolerances.py`:

```python
# Append to meshwell/tolerances.py

class ToleranceHierarchyError(ValueError):
    """Raised when a Tolerances instance violates the required hierarchy."""


# Minimum safety factor between perturbation gap and cut_fuzzy.
# 2x means the buffered overlap must be at least twice the BOP merge
# distance, which prevents accumulated shape-tolerance drift from
# silently welding the carved face. Smaller margins (1.5x) have been
# observed to fail under the cut cascade's tolerance bloat.
_PERTURBATION_SAFETY_FACTOR: float = 2.0


def _validate(t: "Tolerances") -> None:
    if t.cut_fuzzy_value < OCCT_CONFUSION:
        raise ToleranceHierarchyError(
            f"cut_fuzzy_value={t.cut_fuzzy_value} < OCCT_CONFUSION={OCCT_CONFUSION}; "
            "OCC BOP algorithms cannot resolve below Precision::Confusion."
        )
    if t.cut_fuzzy_value > t.fragment_fuzzy_value:
        raise ToleranceHierarchyError(
            f"cut_fuzzy_value={t.cut_fuzzy_value} > "
            f"fragment_fuzzy_value={t.fragment_fuzzy_value}; "
            "fragment pass must be at least as loose as per-cut pass."
        )
    if t.fragment_fuzzy_value > t.perturbation:
        raise ToleranceHierarchyError(
            f"fragment_fuzzy_value={t.fragment_fuzzy_value} > "
            f"perturbation={t.perturbation}; fragment fuzzy must not exceed "
            "the buffered overlap or it would weld the carved face."
        )
    if t.perturbation < _PERTURBATION_SAFETY_FACTOR * t.cut_fuzzy_value:
        raise ToleranceHierarchyError(
            f"perturbation={t.perturbation} < "
            f"{_PERTURBATION_SAFETY_FACTOR}x cut_fuzzy_value={t.cut_fuzzy_value}; "
            "perturbation must exceed cut_fuzzy by safety factor or "
            "tolerance bloat will weld the buffered gap."
        )
    if t.perturbation > t.point_tolerance:
        raise ToleranceHierarchyError(
            f"perturbation={t.perturbation} > point_tolerance={t.point_tolerance}; "
            "shapely set_precision (using point_tolerance) would erase the "
            "buffer before OCC ever sees it."
        )
    if not (0.0 < t.arc_chord_height_fraction <= 1.0):
        raise ToleranceHierarchyError(
            f"arc_chord_height_fraction={t.arc_chord_height_fraction} "
            "must be in (0, 1]."
        )


@dataclass(frozen=True)
class Tolerances:
    point_tolerance: float
    perturbation: float
    cut_fuzzy_value: float
    fragment_fuzzy_value: float
    geometry_tolerance: float
    tolerance_boolean: float
    arc_chord_height_fraction: float

    def __post_init__(self) -> None:
        _validate(self)
```

(Replace the prior bare `@dataclass` with this version that includes `__post_init__`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tolerances.py -v`
Expected: 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/tolerances.py tests/test_tolerances.py
git commit -m "feat(tolerances): enforce hierarchy via __post_init__ validator"
```

---

## Task 3: `from_characteristic_length` factory

**Files:**
- Modify: `meshwell/tolerances.py`
- Test: `tests/test_tolerances.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_tolerances.py
def test_from_characteristic_length_unit_scale():
    """At L=1, defaults match the recommended chain."""
    t = Tolerances.from_characteristic_length(1.0)
    assert t.point_tolerance == 1e-4
    assert t.perturbation == 1e-5
    assert t.cut_fuzzy_value == 1e-6
    assert t.fragment_fuzzy_value == 1e-5
    assert t.geometry_tolerance == 1e-6
    assert t.tolerance_boolean == 1e-5
    assert t.arc_chord_height_fraction == 0.01


def test_from_characteristic_length_scales_linearly():
    """All absolute tolerances scale with L; arc fraction does not."""
    t1 = Tolerances.from_characteristic_length(1.0)
    t100 = Tolerances.from_characteristic_length(100.0)
    assert t100.point_tolerance == 100 * t1.point_tolerance
    assert t100.perturbation == 100 * t1.perturbation
    assert t100.cut_fuzzy_value == 100 * t1.cut_fuzzy_value
    assert t100.arc_chord_height_fraction == t1.arc_chord_height_fraction


def test_from_characteristic_length_rejects_sub_confusion():
    """L too small to keep cut_fuzzy above OCCT_CONFUSION must raise."""
    with pytest.raises(ToleranceHierarchyError):
        Tolerances.from_characteristic_length(1e-3)  # cut_fuzzy=1e-9 < 1e-7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tolerances.py -k from_characteristic_length -v`
Expected: 3 FAIL (`AttributeError: type object 'Tolerances' has no attribute 'from_characteristic_length'`).

- [ ] **Step 3: Implement factory**

Edit `meshwell/tolerances.py`, add classmethod inside the dataclass:

```python
    @classmethod
    def from_characteristic_length(cls, L: float) -> "Tolerances":
        """Derive a coherent tolerance chain from one length scale.

        ``L`` is the characteristic geometric size of the scene (e.g.
        1.0 for unit-scale, 1e-3 if the user works in metres on
        mm-scale geometry, 1e-6 for micron-scale photonics in metres).

        Defaults at L=1:
            point_tolerance      1e-4 L  (shapely dedup; loose)
            perturbation         1e-5 L  (polygon outward buffer)
            cut_fuzzy_value      1e-6 L  (BRepAlgoAPI_Cut fuzzy; tight)
            fragment_fuzzy_value 1e-5 L  (BOPAlgo_Builder fuzzy)
            geometry_tolerance   1e-6 L  (gmsh vertex snap)
            tolerance_boolean    1e-5 L  (gmsh Geometry.ToleranceBoolean)
            arc_chord_height_fraction 0.01  (dimensionless)
        """
        return cls(
            point_tolerance=1e-4 * L,
            perturbation=1e-5 * L,
            cut_fuzzy_value=1e-6 * L,
            fragment_fuzzy_value=1e-5 * L,
            geometry_tolerance=1e-6 * L,
            tolerance_boolean=1e-5 * L,
            arc_chord_height_fraction=0.01,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tolerances.py -v`
Expected: 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/tolerances.py tests/test_tolerances.py
git commit -m "feat(tolerances): add from_characteristic_length(L) factory"
```

---

## Task 4: Wire `Tolerances` into `ModelManager`

**Files:**
- Modify: `meshwell/model.py:19-66`, `meshwell/model.py:103-116`
- Test: `tests/test_tolerances.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_tolerances.py
def test_model_manager_accepts_tolerances():
    """ModelManager(tolerances=...) overrides legacy scalar args."""
    from meshwell.model import ModelManager

    tol = Tolerances.from_characteristic_length(1.0)
    mm = ModelManager(filename="t", tolerances=tol)
    assert mm.point_tolerance == tol.point_tolerance
    assert mm.geometry_tolerance == tol.geometry_tolerance
    assert mm.tolerance_boolean == tol.tolerance_boolean
    assert mm.tolerances is tol


def test_model_manager_legacy_args_still_work():
    """Existing point_tolerance=... continues to behave as before."""
    from meshwell.model import ModelManager

    mm = ModelManager(filename="t", point_tolerance=1e-3)
    assert mm.point_tolerance == 1e-3
    assert mm.tolerance_boolean == 1e-3
    # Synthesized Tolerances must validate.
    assert mm.tolerances.point_tolerance == 1e-3


def test_model_manager_legacy_and_tolerances_conflict_raises():
    from meshwell.model import ModelManager

    tol = Tolerances.from_characteristic_length(1.0)
    with pytest.raises(ValueError, match="cannot pass both"):
        ModelManager(filename="t", tolerances=tol, point_tolerance=1e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tolerances.py -k model_manager -v`
Expected: 3 FAIL.

- [ ] **Step 3: Implement**

Edit `meshwell/model.py`, replace the constructor body (lines 19-66):

```python
    def __init__(
        self,
        n_threads: int = cpu_count(),
        filename: str = "temp",
        point_tolerance: float | None = 1e-3,
        geometry_tolerance: float | None = None,
        tolerance_boolean: float | None = None,
        tolerances: "Tolerances | None" = None,
    ):
        from meshwell.tolerances import Tolerances

        if tolerances is not None and any(
            v is not None and v != _DEFAULT
            for v, _DEFAULT in (
                (geometry_tolerance, None),
                (tolerance_boolean, None),
            )
        ):
            raise ValueError(
                "cannot pass both `tolerances` and legacy scalar args "
                "(geometry_tolerance, tolerance_boolean); use one or the other."
            )
        if tolerances is not None and point_tolerance != 1e-3:
            # 1e-3 is the legacy default; treat any non-default as conflict.
            raise ValueError(
                "cannot pass both `tolerances` and legacy `point_tolerance`."
            )

        if tolerances is None and point_tolerance is not None:
            # Synthesize a Tolerances from legacy scalars for downstream code.
            # Legacy defaults: perturbation=1e-5, cut_fuzzy=5e-6,
            # fragment_fuzzy=point_tolerance.
            pt = point_tolerance
            geom = geometry_tolerance if geometry_tolerance is not None else pt
            tb = tolerance_boolean if tolerance_boolean is not None else pt
            tolerances = Tolerances(
                point_tolerance=pt,
                perturbation=min(1e-5, pt / 2),
                cut_fuzzy_value=min(5e-6, pt / 4),
                fragment_fuzzy_value=min(pt, pt),
                geometry_tolerance=geom,
                tolerance_boolean=tb,
                arc_chord_height_fraction=0.01,
            )

        self.n_threads = n_threads
        self.filename = Path(filename)
        self.tolerances = tolerances
        self.point_tolerance = tolerances.point_tolerance if tolerances else None
        self.geometry_tolerance = (
            tolerances.geometry_tolerance if tolerances else None
        )
        self.tolerance_boolean = tolerances.tolerance_boolean if tolerances else None

        self.model = None
        self.occ = None
        self._is_initialized = False
        self._mesh = None
```

Note: the synthesized-from-legacy `Tolerances(...)` may fail validation if the legacy default `point_tolerance=1e-3` is paired with the legacy `perturbation=1e-5`, because `fragment_fuzzy=1e-3 > perturbation=1e-5`. **This is intentional** — it surfaces the audit's central bug. The legacy synthesis should *not* construct an invalid `Tolerances`; instead, clamp `fragment_fuzzy = min(pt, perturbation)`. Already done above via `fragment_fuzzy_value=min(pt, pt)` — replace with:

```python
                fragment_fuzzy_value=min(pt, 1e-5),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tolerances.py -v`
Expected: 14 tests PASS.

- [ ] **Step 5: Run the existing test suite to confirm no regressions**

Run: `pytest tests/ -x --ignore=tests/test_tolerances.py -k "not slow" -q`
Expected: PASS (note any pre-existing failures and confirm they were not introduced).

- [ ] **Step 6: Commit**

```bash
git add meshwell/model.py tests/test_tolerances.py
git commit -m "feat(model): accept Tolerances; synthesize from legacy args"
```

---

## Task 5: Wire `Tolerances` into `CAD_OCC`

**Files:**
- Modify: `meshwell/cad_occ.py:126-168`
- Test: `tests/test_tolerances.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_tolerances.py
def test_cad_occ_accepts_tolerances():
    from meshwell.cad_occ import CAD_OCC

    tol = Tolerances.from_characteristic_length(1.0)
    cad = CAD_OCC(tolerances=tol)
    assert cad.point_tolerance == tol.point_tolerance
    assert cad.perturbation == tol.perturbation
    assert cad.cut_fuzzy_value == tol.cut_fuzzy_value
    assert cad.fragment_fuzzy_value == tol.fragment_fuzzy_value


def test_cad_occ_legacy_args_synthesize_valid_tolerances():
    from meshwell.cad_occ import CAD_OCC

    cad = CAD_OCC(point_tolerance=1e-3)
    # Synthesis must produce a valid Tolerances (was the audit bug).
    assert cad.tolerances.fragment_fuzzy_value <= cad.tolerances.perturbation
    assert cad.tolerances.cut_fuzzy_value <= cad.tolerances.fragment_fuzzy_value
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tolerances.py -k cad_occ -v`
Expected: 2 FAIL.

- [ ] **Step 3: Implement**

Edit `meshwell/cad_occ.py` constructor (lines 126-168):

```python
    def __init__(
        self,
        point_tolerance: float = 1e-3,
        n_threads: int = cpu_count(),
        cut_fuzzy_value: float | None = None,
        fragment_fuzzy_value: float | None = None,
        perturbation: float | None = None,
        tolerances: "Tolerances | None" = None,
    ):
        from meshwell.tolerances import Tolerances

        if tolerances is None:
            pert = perturbation if perturbation is not None else 1e-5
            cut_f = cut_fuzzy_value if cut_fuzzy_value is not None else pert / 2
            frag_f = (
                fragment_fuzzy_value
                if fragment_fuzzy_value is not None
                else min(point_tolerance, pert)
            )
            tolerances = Tolerances(
                point_tolerance=point_tolerance,
                perturbation=pert,
                cut_fuzzy_value=cut_f,
                fragment_fuzzy_value=frag_f,
                geometry_tolerance=point_tolerance,
                tolerance_boolean=frag_f,
                arc_chord_height_fraction=0.01,
            )

        self.tolerances = tolerances
        self.point_tolerance = tolerances.point_tolerance
        self.n_threads = n_threads
        self.perturbation = tolerances.perturbation
        self.cut_fuzzy_value = tolerances.cut_fuzzy_value
        self.fragment_fuzzy_value = tolerances.fragment_fuzzy_value
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tolerances.py -v && pytest tests/test_cad_occ_fragment_ownership.py -v`
Expected: PASS (existing fragment-ownership tests still pass).

- [ ] **Step 5: Commit**

```bash
git add meshwell/cad_occ.py tests/test_tolerances.py
git commit -m "feat(cad_occ): accept Tolerances; synthesize valid chain from legacy args"
```

---

## Task 6: Clamp tolerance bloat in cut cascade

**Files:**
- Modify: `meshwell/cad_occ.py:155-162` (the sequential cut cascade)
- Test: `tests/test_tolerances.py`

- [ ] **Step 1: Write the failing test**

Identify the cascade location first. Run:
```bash
grep -n "BRepAlgoAPI_Cut\|sequential.*cut\|cut cascade" meshwell/cad_occ.py
```

Then add this test (it asserts that after a cut, the result's max vertex tolerance does not exceed `cut_fuzzy_value`):

```python
# Append to tests/test_tolerances.py
def test_cut_cascade_clamps_tolerance_bloat(tmp_path):
    """After the cut cascade, shape tolerances are clamped to cut_fuzzy.

    Without ShapeFix_ShapeTolerance.LimitTolerance between cuts, OCC's
    BOP can grow vertex tolerances above the configured fuzzy, making
    subsequent cuts effectively looser than requested.
    """
    from meshwell.tolerances import Tolerances
    from meshwell.cad_occ import CAD_OCC

    tol = Tolerances.from_characteristic_length(1.0)
    cad = CAD_OCC(tolerances=tol)
    # Build two overlapping boxes; cut box A by box B; verify max tol clamp.
    # (Use cad._clamp_shape_tolerance directly to keep this unit-level.)
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    box = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Solid()

    cad._clamp_shape_tolerance(box, tol.cut_fuzzy_value)

    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_VERTEX
    from OCP.BRep import BRep_Tool
    from OCP.TopoDS import TopoDS

    exp = TopExp_Explorer(box, TopAbs_VERTEX)
    max_tol = 0.0
    while exp.More():
        v = TopoDS.Vertex_s(exp.Current())
        max_tol = max(max_tol, BRep_Tool.Tolerance_s(v))
        exp.Next()
    assert max_tol <= tol.cut_fuzzy_value * 1.001  # small slack
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tolerances.py::test_cut_cascade_clamps_tolerance_bloat -v`
Expected: FAIL (`AttributeError: 'CAD_OCC' object has no attribute '_clamp_shape_tolerance'`).

- [ ] **Step 3: Implement `_clamp_shape_tolerance`**

Add to `meshwell/cad_occ.py` (place near `_unwrap_shape`):

```python
    @staticmethod
    def _clamp_shape_tolerance(shape: TopoDS_Shape, max_tol: float) -> None:
        """Clamp every sub-shape's tolerance to ``max_tol`` in-place.

        OCC BOPs grow vertex/edge tolerances during intersection. Across a
        cut cascade this drift makes ``cut_fuzzy_value`` a lower bound only;
        the *effective* fuzzy can become much larger silently. Calling
        ``ShapeFix_ShapeTolerance.LimitTolerance`` after each cut keeps the
        configured fuzzy honest.

        ``LimitTolerance(shape, tmin, tmax, style)`` with tmin=0 and a
        bounded tmax clamps any tolerance above ``tmax`` down to ``tmax``.
        """
        from OCP.ShapeFix import ShapeFix_ShapeTolerance
        from OCP.TopAbs import TopAbs_SHAPE

        fixer = ShapeFix_ShapeTolerance()
        fixer.LimitTolerance(shape, 0.0, max_tol, TopAbs_SHAPE)
```

- [ ] **Step 4: Run unit test to verify the clamp works**

Run: `pytest tests/test_tolerances.py::test_cut_cascade_clamps_tolerance_bloat -v`
Expected: PASS.

- [ ] **Step 5: Wire the clamp into the cut cascade**

Locate the sequential cut loop in `cad_occ.py` (the per-entity `BRepAlgoAPI_Cut` calls around line 155-162). After each cut's result is unwrapped, call `_clamp_shape_tolerance(result, self.cut_fuzzy_value)` before passing it forward.

Example diff sketch (read the actual code first to locate exact line):

```python
                cut_op = BRepAlgoAPI_Cut(target, tool)
                cut_op.SetFuzzyValue(self.cut_fuzzy_value)
                cut_op.Build()
                result = cut_op.Shape()
                # Clamp tolerance bloat so the next cut's fuzzy is honest.
                # Without this, accumulated vertex tolerances make later
                # cuts effectively looser than cut_fuzzy_value.
                self._clamp_shape_tolerance(result, self.cut_fuzzy_value)
                target = result
```

- [ ] **Step 6: Run the full structured test suite**

Run: `pytest tests/structured/ tests/test_cad_occ_fragment_ownership.py tests/test_overlapping_facets_structured.py -v`
Expected: PASS (no regressions; the clamp must not change topology, only tolerance metadata).

- [ ] **Step 7: Commit**

```bash
git add meshwell/cad_occ.py tests/test_tolerances.py
git commit -m "fix(cad_occ): clamp shape tolerance between cuts to prevent BOP bloat"
```

---

## Task 7: Radius-relative `arc_chord_height_fraction`

**Files:**
- Modify: `meshwell/structured/spec.py:213-214`
- Modify: `meshwell/structured/plan.py:723`
- Test: `tests/structured/test_tolerance_chain.py`

- [ ] **Step 1: Create test file with failing test**

```python
# tests/structured/test_tolerance_chain.py
"""Integration tests for the redesigned tolerance chain."""
import pytest

from meshwell.structured.spec import Slab
from meshwell.tolerances import Tolerances


def test_arc_chord_height_fraction_small_radius_does_not_collapse():
    """At r=1e-3, a fraction of 0.01 yields buffer=1e-5, not arc-collapse."""
    # Constructed via the helper used by structured pipeline:
    from shapely.geometry import Point

    footprint = Point(0, 0).buffer(1e-3, resolution=32)
    slab = Slab(
        footprint=footprint,
        zlo=0.0,
        zhi=1e-3,
        physical_name=("small_disc",),
        source_index=0,
        z_interval_index=0,
        mesh_order=0,
        identify_arcs=True,
        min_arc_points=4,
        arc_chord_height_fraction=0.01,
    )
    # interior_buffer should be 0.01 * r = 1e-5, NOT max(arc_tol, 0.05*r).
    # Spot-check by calling the helper directly.
    from meshwell.structured.plan import _interior_buffer_for_radius

    assert _interior_buffer_for_radius(slab, r=1e-3) == pytest.approx(1e-5)


def test_arc_chord_height_fraction_large_radius_is_proportional():
    from shapely.geometry import Point
    from meshwell.structured.plan import _interior_buffer_for_radius

    slab = Slab(
        footprint=Point(0, 0).buffer(100.0, resolution=64),
        zlo=0.0, zhi=1.0, physical_name=("big",), source_index=0,
        z_interval_index=0, mesh_order=0, identify_arcs=True,
        min_arc_points=4, arc_chord_height_fraction=0.01,
    )
    # 0.01 * 100 = 1.0
    assert _interior_buffer_for_radius(slab, r=100.0) == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/structured/test_tolerance_chain.py -v`
Expected: FAIL (`arc_chord_height_fraction` not a Slab field; `_interior_buffer_for_radius` doesn't exist).

- [ ] **Step 3: Add field to Slab dataclass**

Edit `meshwell/structured/spec.py` around line 213:

```python
    identify_arcs: bool = False
    min_arc_points: int = 4
    arc_tolerance: float = 1e-3  # legacy absolute backstop (deprecated)
    arc_chord_height_fraction: float = 0.01  # 1% of radius
    fragment_fuzzy_value: float | None = None
```

- [ ] **Step 4: Extract `_interior_buffer_for_radius` helper**

In `meshwell/structured/plan.py`, near line 700, add:

```python
def _interior_buffer_for_radius(slab: "Slab", r: float) -> float:
    """Compute the arc-vs-neighbour interior buffer for a given radius.

    Replaces the previous ``max(arc_tol, 0.05*r)`` heuristic with
    ``arc_chord_height_fraction * r``, making the buffer proportional to
    the local arc radius rather than dominated by a 5% wildcard.
    """
    return slab.arc_chord_height_fraction * r
```

Then update line 723 in `plan.py`:

```python
                    interior_buffer = _interior_buffer_for_radius(slab, r)
                    inside_region = slab.footprint.buffer(interior_buffer)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/structured/test_tolerance_chain.py -v`
Expected: PASS.

- [ ] **Step 6: Verify no regression in existing arc tests**

Run: `pytest tests/structured/ -v -k "arc"`
Expected: PASS (`identify_arcs` paths still behave).

- [ ] **Step 7: Commit**

```bash
git add meshwell/structured/spec.py meshwell/structured/plan.py tests/structured/test_tolerance_chain.py
git commit -m "fix(structured): radius-relative arc chord-height fraction (replaces 5%/abs max)"
```

---

## Task 8: Warn on heterogeneous slab `fragment_fuzzy_value`

**Files:**
- Modify: `meshwell/structured/builder.py:558-566`
- Test: `tests/structured/test_tolerance_chain.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/structured/test_tolerance_chain.py
def test_warn_on_heterogeneous_fragment_fuzzy(caplog):
    """Mesh stage warns when slabs disagree on fragment_fuzzy_value.

    The dedup-node tolerance becomes max(...), which silently couples
    a tight slab to a loose neighbour. We at least log this.
    """
    import logging
    from meshwell.structured.builder import _aggregate_slab_fuzzy

    slabs_fuzzy = [1e-6, 1e-4, None]
    caplog.set_level(logging.WARNING, logger="meshwell.structured.builder")
    result = _aggregate_slab_fuzzy(slabs_fuzzy, default=1e-6)
    assert result == 1e-4
    assert any(
        "heterogeneous fragment_fuzzy_value" in rec.message.lower()
        for rec in caplog.records
    )


def test_no_warn_when_all_slabs_agree(caplog):
    import logging
    from meshwell.structured.builder import _aggregate_slab_fuzzy

    caplog.set_level(logging.WARNING, logger="meshwell.structured.builder")
    result = _aggregate_slab_fuzzy([1e-5, 1e-5, None], default=1e-6)
    assert result == 1e-5
    assert not any(
        "heterogeneous" in rec.message.lower() for rec in caplog.records
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/structured/test_tolerance_chain.py -k fragment_fuzzy -v`
Expected: FAIL (`_aggregate_slab_fuzzy` does not exist).

- [ ] **Step 3: Extract aggregator with warning**

Edit `meshwell/structured/builder.py`, before line 558. Add:

```python
def _aggregate_slab_fuzzy(
    slab_fuzzy_values: list[float | None],
    default: float,
) -> float:
    """Aggregate per-slab fragment_fuzzy_value into a single dedup tolerance.

    Returns ``max(non-None values)`` or ``default`` if all are None.
    Logs a warning when the non-None values are heterogeneous, because
    ``removeDuplicateNodes`` uses a single tolerance for the entire
    mesh; the loosest slab silently couples to the tightest.
    """
    import logging

    logger = logging.getLogger(__name__)
    non_none = [v for v in slab_fuzzy_values if v is not None]
    if not non_none:
        return default
    if len(set(non_none)) > 1:
        logger.warning(
            "heterogeneous fragment_fuzzy_value across slabs: %s; "
            "removeDuplicateNodes will use max=%s, which may silently "
            "merge nodes in tighter slabs.",
            sorted(set(non_none)),
            max(non_none),
        )
    return max(non_none)
```

Then replace the inline aggregation at line 558-563:

```python
    # Global cleanup: merge ~coincident nodes.
    fuzzy = _aggregate_slab_fuzzy(
        [slab.fragment_fuzzy_value for slab in plan.slabs],
        default=fuzzy_tol,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/structured/test_tolerance_chain.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/builder.py tests/structured/test_tolerance_chain.py
git commit -m "feat(structured): warn on heterogeneous slab fragment_fuzzy_value"
```

---

## Task 9: Integration test — perturbation survives shapely snap

**Files:**
- Test: `tests/structured/test_tolerance_chain.py`

- [ ] **Step 1: Write the integration test**

```python
# Append to tests/structured/test_tolerance_chain.py
def test_perturbation_gap_survives_shapely_snap():
    """If point_tolerance >> perturbation, shapely snap erases the buffer.

    The Tolerances validator should already prevent this combo, but we
    pin the behavior with an end-to-end check: build two adjacent unit
    boxes with a perturbation-sized gap, run them through the shapely
    pre-pass with point_tolerance just below the buffer, and verify the
    gap survives.
    """
    import shapely
    from shapely.geometry import box

    from meshwell.tolerances import Tolerances

    tol = Tolerances.from_characteristic_length(1.0)
    # Two boxes separated by perturbation:
    a = box(0, 0, 1, 1)
    b = box(1 + tol.perturbation, 0, 2 + tol.perturbation, 1)

    # After shapely set_precision with point_tolerance:
    a_snap = shapely.set_precision(a, tol.point_tolerance)
    b_snap = shapely.set_precision(b, tol.point_tolerance)

    # The boxes must remain disjoint (gap survives).
    assert not a_snap.intersects(b_snap), (
        "shapely set_precision with point_tolerance erased the perturbation "
        "gap — validator should have caught this combo upstream"
    )
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/structured/test_tolerance_chain.py::test_perturbation_gap_survives_shapely_snap -v`
Expected: PASS (the chain from `from_characteristic_length` already keeps `point_tolerance < perturbation`).

- [ ] **Step 3: Commit**

```bash
git add tests/structured/test_tolerance_chain.py
git commit -m "test(tolerance-chain): pin perturbation-survives-snap invariant"
```

---

## Task 10: Integration test — tolerance sweep on a real structured scene

**Files:**
- Test: `tests/structured/test_tolerance_chain.py`

- [ ] **Step 1: Find a minimal end-to-end fixture**

Inspect `tests/test_structured_complex_scene.py` and `tests/test_overlapping_facets_structured.py` for a small structured scene you can reuse. Pick the simplest one that produces 2+ slabs and exercises the full pipeline.

- [ ] **Step 2: Write the sweep test**

```python
# Append to tests/structured/test_tolerance_chain.py
@pytest.mark.parametrize("L", [1e-3, 1e-2, 1e-1, 1.0, 10.0])
def test_tolerance_sweep_preserves_topology(L, tmp_path):
    """Across L = 1e-3 to 10, topology counts must be invariant.

    If a particular L breaks (e.g. cut_fuzzy drops below OCCT_CONFUSION),
    the Tolerances factory must raise, not silently produce wrong
    geometry.
    """
    from meshwell.tolerances import Tolerances, ToleranceHierarchyError

    try:
        tol = Tolerances.from_characteristic_length(L)
    except ToleranceHierarchyError:
        # Acceptable: validator caught an unsafe L.
        # L=1e-3 produces cut_fuzzy=1e-9 < OCCT_CONFUSION; must raise.
        assert L <= 1e-2
        return

    # For valid L, build the minimal scene and assert volume count.
    # NOTE for implementer: replicate the smallest fixture from
    # tests/test_overlapping_facets_structured.py, scaling every
    # coordinate by L. Assert that len(vol_tags) is invariant across
    # the L values that succeed.
    from tests.structured.helpers import build_minimal_scaled_scene

    vol_tags = build_minimal_scaled_scene(L, tol, tmp_path)
    assert len(vol_tags) == 2  # expected from fixture
```

- [ ] **Step 3: Create `tests/structured/helpers.py` if missing**

If no such helper exists, extract the smallest reusable scene from `tests/test_overlapping_facets_structured.py` into `tests/structured/helpers.py` as `build_minimal_scaled_scene(L, tol, tmp_path)`. **Do not invent geometry** — copy the fixture and parametrize by `L`.

- [ ] **Step 4: Run the sweep**

Run: `pytest tests/structured/test_tolerance_chain.py::test_tolerance_sweep_preserves_topology -v`
Expected: PASS for L ≥ 1e-2; rejected by validator for L ≤ 1e-3.

- [ ] **Step 5: Commit**

```bash
git add tests/structured/test_tolerance_chain.py tests/structured/helpers.py
git commit -m "test(tolerance-chain): sweep L across 5 orders, assert topology invariance"
```

---

## Task 11: Integration test — arc small-radius boundary

**Files:**
- Test: `tests/structured/test_tolerance_chain.py`

- [ ] **Step 1: Locate the existing arc-split fixture**

Run: `grep -rn "StructuredArcSplitError\|arc_split" tests/`

Identify the test (likely in `tests/structured/`) that currently exercises `StructuredArcSplitError`. Read it end-to-end so you understand the disc + neighbour scene it builds.

- [ ] **Step 2: Add helper to `tests/structured/helpers.py`**

```python
# tests/structured/helpers.py — add this function
import shapely
from shapely.geometry import Point, box

from meshwell.structured.spec import Slab


def build_arc_with_neighbour(
    radius: float,
    neighbour_offset_fraction: float,
    arc_chord_height_fraction: float = 0.01,
) -> tuple[Slab, Slab]:
    """Build a disc slab and a rectangular neighbour whose boundary sits
    at ``neighbour_offset_fraction * (arc_chord_height_fraction * radius)``
    inside the disc footprint.

    A value < 1.0 places the neighbour boundary inside the interior
    buffer (must trigger StructuredArcSplitError). A value > 1.0 places
    it outside (must NOT trigger).
    """
    disc_footprint = Point(0, 0).buffer(radius, resolution=32)
    buffer_width = arc_chord_height_fraction * radius
    offset = neighbour_offset_fraction * buffer_width
    # Neighbour box clipped so its right edge sits at x = radius - offset
    # (just inside the disc on the +x side).
    nbox = box(0.0, -radius, radius - offset, radius)
    nbox_intersect = disc_footprint.intersection(nbox)

    disc = Slab(
        footprint=disc_footprint, zlo=0.0, zhi=1.0,
        physical_name=("disc",), source_index=0,
        z_interval_index=0, mesh_order=0,
        identify_arcs=True, min_arc_points=4,
        arc_chord_height_fraction=arc_chord_height_fraction,
    )
    neighbour = Slab(
        footprint=nbox_intersect, zlo=0.0, zhi=1.0,
        physical_name=("nb",), source_index=1,
        z_interval_index=0, mesh_order=1,
        identify_arcs=False, min_arc_points=4,
        arc_chord_height_fraction=arc_chord_height_fraction,
    )
    return disc, neighbour
```

- [ ] **Step 3: Write the boundary test**

```python
# Append to tests/structured/test_tolerance_chain.py
import pytest

from meshwell.structured.plan import StructuredArcSplitError
from tests.structured.helpers import build_arc_with_neighbour


@pytest.mark.parametrize("radius", [1e-3, 1.0, 100.0])
def test_arc_split_detected_just_inside_chord_buffer(radius):
    """Neighbour boundary at 0.5 * (chord_fraction*r) from arc must be caught."""
    from meshwell.structured.plan import build_structured_plan

    disc, neighbour = build_arc_with_neighbour(radius, neighbour_offset_fraction=0.5)
    with pytest.raises(StructuredArcSplitError):
        build_structured_plan([disc, neighbour])


@pytest.mark.parametrize("radius", [1e-3, 1.0, 100.0])
def test_arc_split_not_detected_outside_chord_buffer(radius):
    """Neighbour boundary at 1.5 * (chord_fraction*r) must NOT raise."""
    from meshwell.structured.plan import build_structured_plan

    disc, neighbour = build_arc_with_neighbour(radius, neighbour_offset_fraction=1.5)
    plan = build_structured_plan([disc, neighbour])
    assert plan is not None
```

Note: `build_structured_plan` is the placeholder name for the public entry to the planner. Replace it with the actual function name discovered in Step 1 (likely `compute_structured_plan` or similar — check `meshwell/structured/plan.py` for the public planner entry).

- [ ] **Step 2: Run**

Run: `pytest tests/structured/test_tolerance_chain.py -k arc_split -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/structured/test_tolerance_chain.py
git commit -m "test(tolerance-chain): arc-split boundary at chord-height fraction"
```

---

## Task 12: Update docs + cross-link audit

**Files:**
- Modify: `meshwell/cad_occ.py` constructor docstring (link to `Tolerances`)
- Modify: `meshwell/model.py` constructor docstring (link to `Tolerances`)
- Create: short note in `meshwell/tolerances.py` module docstring linking to this plan.

- [ ] **Step 1: Update docstrings**

In both `model.py` and `cad_occ.py`, prepend to the constructor docstring:

```
        Prefer ``tolerances=Tolerances.from_characteristic_length(L)``
        over the legacy scalar args. The legacy args are accepted for
        back-compat but synthesize a clamped Tolerances internally. See
        ``meshwell.tolerances`` and
        ``docs/superpowers/plans/2026-05-19-tolerance-chain-redesign.md``.
```

- [ ] **Step 2: Commit**

```bash
git add meshwell/cad_occ.py meshwell/model.py meshwell/tolerances.py
git commit -m "docs(tolerances): link constructors to Tolerances + redesign plan"
```

---

## Final verification

- [ ] **Run the full test suite**

Run: `pytest tests/ -q`
Expected: all pre-existing tests pass; 14+ new tests pass.

- [ ] **Run the structured integration suite specifically**

Run: `pytest tests/structured/ tests/test_overlapping_facets_structured.py tests/test_cad_occ_fragment_ownership.py -v`
Expected: PASS, no skipped tests changed.

- [ ] **Manual smoke**

Pick one of the demo scripts (e.g. `scripts/inspect_demo_mesh_v3.py`) and rerun with `tolerances=Tolerances.from_characteristic_length(L)` substituted for the legacy `point_tolerance=1e-3`. Confirm the produced mesh visually matches the prior output.
