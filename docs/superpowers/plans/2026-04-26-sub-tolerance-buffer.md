# Sub-tolerance buffer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple `cad_gmsh`'s buffer scale (`perturbation`) and gmsh's vertex-snap distance (`geometry_tolerance`) from `point_tolerance`, with sub-tolerance defaults so polygon-entity distortion drops below user-promised precision.

**Architecture:** Add a single dimensionless `tolerance_scale` parameter (default `100.0`) to both `ModelManager` and `CAD_GMSH`. Defaults: `perturbation = max(point_tolerance / scale, 1e-9)`, `geometry_tolerance = max(point_tolerance / scale**2, 1e-12)`. Both lower levels can be overridden explicitly. `Geometry.ToleranceBoolean` stays at `point_tolerance`.

**Tech Stack:** Python 3.11+, gmsh, shapely, pytest.

**Spec:** [docs/superpowers/specs/2026-04-26-sub-tolerance-buffer-design.md](../specs/2026-04-26-sub-tolerance-buffer-design.md)

---

## File map

- **Modify:** `meshwell/model.py` — add `tolerance_scale` and `geometry_tolerance` parameters to `ModelManager`; use `geometry_tolerance` for `Geometry.Tolerance`.
- **Modify:** `meshwell/cad_gmsh.py` — replace `perturbation` `@property` with constructor parameter; add `tolerance_scale`; forward to `ModelManager` when processor owns it; widen `cad_gmsh()` wrapper signature.
- **Modify:** `tests/test_cad_gmsh.py` — tighten the loose `5e-2` tolerances on the partial-overlap test back down (now that buffer is small); the 3 currently-failing tests should be re-evaluated and their assertions adjusted as needed.
- **Create / append:** `tests/test_cad_gmsh.py` — one new regression test pinning that distortion stays below `point_tolerance`.

`tests/test_interface_tag.py` should be unaffected — InterfaceTag uses `self.perturbation` which now resolves via the new attribute, but the tests assert behavior (segment counts, face placement) not absolute coordinates that would shift with the smaller buffer.

---

## Task 1: Add `tolerance_scale` + `geometry_tolerance` to `ModelManager`

Pure additive change. New params, new attribute, swap one `gmsh.option.setNumber` argument.

**Files:**
- Modify: `meshwell/model.py` (`ModelManager.__init__` and `_initialize`)

- [ ] **Step 1.1: Update `ModelManager.__init__`**

In `meshwell/model.py`, replace the existing `__init__` signature and body with:

```python
    def __init__(
        self,
        n_threads: int = cpu_count(),
        filename: str = "temp",
        point_tolerance: float | None = 1e-3,
        tolerance_scale: float = 100.0,
        geometry_tolerance: float | None = None,
    ):
        """Initialize Model with common settings.

        Args:
            n_threads: Number of threads for GMSH operations
            filename: Base filename for the model
            point_tolerance: User-promised geometric precision. Drives
                ``shapely.set_precision`` snap on inputs and
                ``Geometry.ToleranceBoolean`` (BOP fuzzy merge).
            tolerance_scale: Dimensionless ratio between adjacent
                tolerance levels in the cad_gmsh ladder
                ``geometry_tolerance < perturbation < point_tolerance``.
                Larger values push the lower levels further below
                ``point_tolerance`` (less distortion, less bridging
                margin).
            geometry_tolerance: Optional explicit override for gmsh's
                ``Geometry.Tolerance`` (vertex-snap distance). Defaults
                to ``max(point_tolerance / (tolerance_scale ** 2), 1e-12)``
                — i.e., two scale-steps below ``point_tolerance`` to
                sit below the cad_gmsh ``perturbation``.

        """
        self.n_threads = n_threads
        self.filename = Path(filename)
        self.point_tolerance = point_tolerance
        self.tolerance_scale = tolerance_scale
        if geometry_tolerance is not None:
            self.geometry_tolerance = geometry_tolerance
        elif point_tolerance is None:
            self.geometry_tolerance = None
        else:
            self.geometry_tolerance = max(
                point_tolerance / (tolerance_scale ** 2), 1e-12
            )

        # GMSH objects (initialized in _initialize)
        self.model = None
        self.occ = None

        # Initialization state
        self._is_initialized = False

        # CAD and Mesh instances (created lazily)
        self._mesh = None
```

- [ ] **Step 1.2: Update `_initialize` to use `geometry_tolerance`**

In the same file, locate (around line 79-82):

```python
        # Configure OCC tolerance if provided
        if self.point_tolerance is not None:
            gmsh.option.setNumber("Geometry.Tolerance", self.point_tolerance)
            gmsh.option.setNumber("Geometry.ToleranceBoolean", self.point_tolerance)
```

Replace the `setNumber("Geometry.Tolerance", ...)` line so it uses `self.geometry_tolerance` (with fallback to `self.point_tolerance` if unset):

```python
        # Configure OCC tolerance if provided
        if self.point_tolerance is not None:
            gmsh_tol = (
                self.geometry_tolerance
                if self.geometry_tolerance is not None
                else self.point_tolerance
            )
            gmsh.option.setNumber("Geometry.Tolerance", gmsh_tol)
            gmsh.option.setNumber("Geometry.ToleranceBoolean", self.point_tolerance)
```

- [ ] **Step 1.3: Run the full test suite to confirm no regression yet**

`cad_gmsh.py` still uses the old `2 * point_tolerance` perturbation, so test outcomes should be UNCHANGED from current baseline:
- `tests/test_interface_tag.py`: 15/15 pass
- `tests/test_cad_gmsh.py`: 4 pass + 3 known failures
  (`test_cad_gmsh_mesh_order_lower_wins_in_overlap`,
  `test_cad_gmsh_same_mesh_order_ties_to_insertion_order`,
  `test_cad_gmsh_keep_false_top_dim_removed_but_interface_named`)

Run: `pytest tests/test_interface_tag.py tests/test_cad_gmsh.py`

- [ ] **Step 1.4: Commit**

```bash
git add meshwell/model.py
git commit -m "feat(model): add tolerance_scale + geometry_tolerance params to ModelManager"
```

---

## Task 2: Replace `CAD_GMSH.perturbation` property with constructor parameter

Move from a derived `@property` (currently `2 * self.point_tolerance`) to a constructor parameter with a sub-tolerance default. Forward `tolerance_scale` to the auto-created `ModelManager`.

**Files:**
- Modify: `meshwell/cad_gmsh.py` (`CAD_GMSH.__init__`, remove `perturbation` property, update `cad_gmsh()` wrapper)

- [ ] **Step 2.1: Update `CAD_GMSH.__init__`**

In `meshwell/cad_gmsh.py`, replace the existing `__init__` of `CAD_GMSH` with:

```python
    def __init__(
        self,
        point_tolerance: float = 1e-3,
        n_threads: int = cpu_count(),
        filename: str = "temp",
        model: ModelManager | None = None,
        tolerance_scale: float = 100.0,
        perturbation: float | None = None,
    ):
        """Initialize gmsh CAD processor.

        Args:
            point_tolerance: User-promised geometric precision. Drives
                ``shapely.set_precision`` snap on inputs and
                ``Geometry.ToleranceBoolean``.
            n_threads: Thread count for gmsh meshing / boolean parallelism.
            filename: Base filename for the model (used when ``model``
                is not provided).
            model: Optional :class:`ModelManager` to reuse. When provided
                the processor does not finalize gmsh on exit -- the caller
                owns the lifecycle.
            tolerance_scale: Dimensionless ratio between adjacent levels
                in the cad_gmsh tolerance ladder
                ``geometry_tolerance < perturbation < point_tolerance``.
                Larger values reduce bound distortion but leave less
                margin for bridging almost-coincident faces.
            perturbation: Optional explicit override for the outward
                shapely buffer applied to polygon entities before the
                sequential cut cascade. Defaults to
                ``max(point_tolerance / tolerance_scale, 1e-9)``.
        """
        if model is None:
            self.model_manager = ModelManager(
                n_threads=n_threads,
                filename=filename,
                point_tolerance=point_tolerance,
                tolerance_scale=tolerance_scale,
            )
            self._owns_model = True
        else:
            self.model_manager = model
            self._owns_model = False
        self.point_tolerance = point_tolerance
        self.n_threads = n_threads
        self.tolerance_scale = tolerance_scale
        if perturbation is not None:
            self.perturbation = perturbation
        else:
            self.perturbation = max(point_tolerance / tolerance_scale, 1e-9)
```

- [ ] **Step 2.2: Remove the `perturbation` `@property`**

Locate the existing block immediately after `__init__`:

```python
    @property
    def perturbation(self) -> float:
        """Outward shapely buffer applied to polygon entities before
        sequential cuts. Same value used as the default snap distance
        for :class:`meshwell.interface_tag.InterfaceTag`. Currently
        ``2 * point_tolerance``; do not change without coordinated
        validation on complex scenes."""
        return 2 * self.point_tolerance
```

Delete it entirely. (`self.perturbation` is now a regular attribute set in `__init__`.)

- [ ] **Step 2.3: Update the top-level `cad_gmsh()` wrapper**

Locate the `cad_gmsh()` function near the bottom of `meshwell/cad_gmsh.py`. Update its signature and body to forward the new parameters:

```python
def cad_gmsh(
    entities_list: list[Any],
    point_tolerance: float = 1e-3,
    n_threads: int = cpu_count(),
    progress_bars: bool = False,
    filename: str = "temp",
    model: ModelManager | None = None,
    interface_delimiter: str = "___",
    boundary_delimiter: str = "None",
    tolerance_scale: float = 100.0,
    perturbation: float | None = None,
) -> tuple[list[GMSHLabeledEntity], ModelManager]:
    """Build + fragment + tag ``entities_list`` in a gmsh model.

    Returns ``(labeled_entities, model_manager)``. Pass
    ``model_manager`` on to :func:`meshwell.mesh.mesh` (with
    ``model=model_manager``) to mesh without round-tripping through XAO.
    """
    processor = CAD_GMSH(
        point_tolerance=point_tolerance,
        n_threads=n_threads,
        filename=filename,
        model=model,
        tolerance_scale=tolerance_scale,
        perturbation=perturbation,
    )
    labeled = processor.process_entities(
        entities_list,
        progress_bars=progress_bars,
        interface_delimiter=interface_delimiter,
        boundary_delimiter=boundary_delimiter,
    )
    return labeled, processor.model_manager
```

- [ ] **Step 2.4: Run the full test suite**

`tests/test_interface_tag.py` should still be 15/15 pass — all those tests assert behavioral properties (segment counts, face placements) that the smaller buffer does not change.

`tests/test_cad_gmsh.py` outcomes will change:
- The 3 currently-failing tests may now pass (smaller buffer = less area inflation = winner/loser ratios stay correct).
- The new test added earlier (`test_cad_gmsh_2d_surface_partially_overlaps_3d_face_is_carved_and_tagged`) was loosened to `5e-2`; it still passes with smaller buffer (overshoots tolerance). Will be tightened in Task 3.

Run: `pytest tests/test_cad_gmsh.py tests/test_interface_tag.py -v`

Record exact pass/fail outcome. Expected best case: 19/19 pass (the 3 baseline failures resolve themselves). Expected worst case: some of the 3 still fail with different/tighter assertions.

If any `test_interface_tag.py` test regresses, STOP and report BLOCKED.

If any of the 3 `test_cad_gmsh.py` tests still fails, capture the assertion + actual values and continue to Task 3 (which addresses test calibration).

- [ ] **Step 2.5: Commit**

```bash
git add meshwell/cad_gmsh.py
git commit -m "feat(cad_gmsh): replace perturbation property with sub-tolerance default"
```

---

## Task 3: Calibrate test assertions for the new buffer

The 3 currently-failing tests assert area equalities at `1e-6` tolerance, which was originally written for a no-buffer codebase. Even with the new sub-tolerance buffer (~`1e-5`), residual area inflation may be `~4e-5` — still over the `1e-6` cutoff.

This task examines each test, confirms whether it now passes, and adjusts assertions only where strictly needed. The new test added earlier is also tightened.

**Files:**
- Modify: `tests/test_cad_gmsh.py` (loosen or tighten specific assertions; do NOT change test names or scenes)

- [ ] **Step 3.1: Re-run the 3 baseline-failing tests + the loosened new test**

```bash
pytest tests/test_cad_gmsh.py::test_cad_gmsh_mesh_order_lower_wins_in_overlap tests/test_cad_gmsh.py::test_cad_gmsh_same_mesh_order_ties_to_insertion_order tests/test_cad_gmsh.py::test_cad_gmsh_keep_false_top_dim_removed_but_interface_named tests/test_cad_gmsh.py::test_cad_gmsh_2d_surface_partially_overlaps_3d_face_is_carved_and_tagged -v
```

Capture pass/fail status and (for failures) the actual numeric values from the AssertionError. The expected pattern: failures shifted from `~8e-3` deltas (with old `2e-3` buffer) to `~4e-5` deltas (with new `1e-5` buffer).

- [ ] **Step 3.2: Tighten `test_cad_gmsh_2d_surface_partially_overlaps_3d_face_*`**

In `tests/test_cad_gmsh.py`, locate the three `< 5e-2` tolerance assertions (added when the test was loosened for the old buffer). Tighten each to `< 1e-4`. The relevant assertions:

```python
        assert abs(total_surf_area - 2.0) < 5e-2, total_surf_area
```

becomes:

```python
        assert abs(total_surf_area - 2.0) < 1e-4, total_surf_area
```

Same change for the two embedded/dangling area assertions in the same test (search for `5e-2` within that test). Update the inline comment that references "O(1e-2)" to "O(1e-5)".

Run that single test; expect PASS.

- [ ] **Step 3.3: Adjust the other 3 tests' assertions if needed**

For each of the 3 tests `test_cad_gmsh_mesh_order_lower_wins_in_overlap`, `test_cad_gmsh_same_mesh_order_ties_to_insertion_order`, `test_cad_gmsh_keep_false_top_dim_removed_but_interface_named`:

- If it now PASSES: leave it alone.
- If it still FAILS on an absolute-equality assertion (`abs(x - X) < 1e-6`): widen the tolerance to the smallest power-of-10 above the actual delta plus a 10× margin. E.g., if actual delta is `4e-5`, set tolerance to `5e-4`. Add a one-line comment explaining the residual is from the sub-tolerance perturbation.
- If it still FAILS on a *relative* / inequality assertion (`areas["A"] > areas["B"]`): the test is now correct in semantic intent — the failure points to a real algorithmic bug, not a tolerance issue. STOP and report BLOCKED with the actual numeric values.

After any changes, re-run the test and confirm PASS. Then run the full `tests/test_cad_gmsh.py` suite and confirm no other test regresses.

- [ ] **Step 3.4: Commit**

```bash
git add tests/test_cad_gmsh.py
git commit -m "test(cad_gmsh): calibrate tolerances for sub-tolerance perturbation"
```

---

## Task 4: New regression test pinning the distortion bound

Add a single test that pins the contract: with default `perturbation`, the bounding box of a polygon-entity result is within `point_tolerance` of the input.

**Files:**
- Modify: `tests/test_cad_gmsh.py` (append new test)

- [ ] **Step 4.1: Append the test**

At the end of `tests/test_cad_gmsh.py`, append:

```python
def test_cad_gmsh_perturbation_below_point_tolerance():
    """At default tolerance_scale, the polygon entity's resulting bounding
    box is within point_tolerance of the input bounds. Pins the contract
    that the cut-bridging buffer does not visibly distort geometry at
    user-promised precision."""
    point_tol = 1e-3
    poly = shapely.Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])
    try:
        labeled, mm = cad_gmsh(
            [PolySurface(polygons=poly, physical_name="A", mesh_order=1)],
            point_tolerance=point_tol,
        )
        ent = next(e for e in labeled if e.physical_name[0].startswith("A"))
        assert ent.dimtags
        for dim, tag in ent.dimtags:
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(dim, tag)
            # Each input bound must be within point_tolerance of the result.
            assert abs(xmin - 0.0) < point_tol, xmin
            assert abs(ymin - 0.0) < point_tol, ymin
            assert abs(xmax - 2.0) < point_tol, xmax
            assert abs(ymax - 1.0) < point_tol, ymax
    finally:
        mm.finalize()
```

- [ ] **Step 4.2: Run the new test**

```bash
pytest tests/test_cad_gmsh.py::test_cad_gmsh_perturbation_below_point_tolerance -v
```

Expected: PASS. With default `perturbation = 1e-5` and `point_tolerance = 1e-3`, all bound shifts are `~1e-5`, well within the `< 1e-3` assertion.

- [ ] **Step 4.3: Commit**

```bash
git add tests/test_cad_gmsh.py
git commit -m "test(cad_gmsh): pin sub-tolerance perturbation distortion contract"
```

---

## Task 5: Final regression sweep

**Files:** none — verification only.

- [ ] **Step 5.1: Run the full meshwell test suite**

```bash
pytest tests/test_interface_tag.py tests/test_cad_gmsh.py -v
```

Expected:
- `tests/test_interface_tag.py`: 15/15 pass.
- `tests/test_cad_gmsh.py`: ALL pass (the 3 prior baseline failures plus the existing ones plus the new regression test).

If the original 3 baseline failures still fail after Task 3's calibration, the calibration did not work — reopen Task 3 and report what's still broken.

If any `test_interface_tag.py` test regresses, STOP and report BLOCKED — InterfaceTag's behavior should be unaffected by the perturbation change.

- [ ] **Step 5.2: Confirm new `cad_gmsh` API surface**

The new public parameters are `tolerance_scale` and `perturbation` (and on `ModelManager`, also `geometry_tolerance`). All have safe defaults — no existing call site needs to change.

Confirm by grepping for callers of `cad_gmsh(` and `ModelManager(` and `CAD_GMSH(` outside the test files. Any caller that does NOT pass either of the new params will continue to work unchanged.

```bash
grep -rn "cad_gmsh(\|CAD_GMSH(\|ModelManager(" meshwell/ tests/ | grep -v "def cad_gmsh\|class CAD_GMSH\|class ModelManager"
```

This step is informational; no code changes expected.

- [ ] **Step 5.3: Done**

No commit needed for this task. The implementation is complete.
