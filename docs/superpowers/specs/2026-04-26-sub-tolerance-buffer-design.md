# Sub-tolerance buffer for `cad_gmsh`

**Status:** approved spec, ready for implementation plan
**Date:** 2026-04-26
**Backend scope:** `cad_gmsh` only

## Problem

`cad_gmsh` buffers polygon entities outward by `2 * point_tolerance` so the
sequential `gmsh.model.occ.cut` cascade has overlapping geometry to bite on.
Without the buffer, BOPAlgo intermittently fails to merge almost-coincident
faces on complex scenes (the failures don't reproduce in simple test cases).
The buffer is load-bearing and must stay.

The cost is that input geometry is distorted by `2 * point_tolerance` —
visible at user-promised precision. Downstream effects:

- Three `tests/test_cad_gmsh.py` tests fail (`mesh_order_lower_wins_in_overlap`,
  `same_mesh_order_ties_to_insertion_order`, `keep_false_top_dim_removed_but_interface_named`)
  because area inflation flips winner/loser ratios.
- One newly-added test (`test_cad_gmsh_2d_surface_partially_overlaps_3d_face_is_carved_and_tagged`)
  had to be loosened to a `5e-2` area tolerance.

The InterfaceTag work (committed 2026-04-26) addressed the asymmetric
distortion problem (only polygon entities get buffered, not gmsh-only ones)
for the *interface tagging* use case. We can now revisit the buffer scale
itself without that complication.

## Discovered constraint (during planning)

Initially we proposed `perturbation = max(point_tolerance / 100, 1e-9)`.
Empirically that fails: `PolySurface`, `PolyPrism`, `PolyLine`, and
`InterfaceTag` constructors all call
`shapely.set_precision(p, grid_size=point_tolerance, mode="pointwise")`,
which installs a precision *model* on the geometry. Subsequent
`polygon.buffer(d)` calls with `d < grid_size` then return **empty**
geometry (GEOS rounds the offset to the precision grid). So while the
input is precision-snapped, the buffer cannot go below `point_tolerance`.

A second discovery: the three previously-flagged "baseline failures" in
`tests/test_cad_gmsh.py` are actually a test-helper bug. They store
`areas[ent.physical_name[0]] = total` where `physical_name[0]` carries the
internal `__#index` suffix, then index `areas["A"]` and KeyError. They are
not buffer-induced and need a separate `strip_suffix(...)` fix.

## Solution

Two independent changes:

**(1) Remove `shapely.set_precision` from entity constructors.** Without
the precision model attached to input polygons, shapely buffer can be
arbitrarily small (verified down to `1e-15`). gmsh's `Geometry.Tolerance`
(set to `point_tolerance`) takes over the role of vertex snapping at
shape ingestion. `Geometry.ToleranceBoolean` (also `point_tolerance`)
handles fuzzy BOP merge. The user-visible behavior is preserved — vertex
coincidence within `point_tolerance` still merges, just at the gmsh
ingestion stage instead of the shapely construction stage.

**(2) Decouple three tolerances** that are currently conflated to `point_tolerance`:

1. **`point_tolerance`** (existing): user-promised precision. Drives
   `shapely.set_precision` snap on inputs. Unchanged.
2. **`perturbation`** (new, `CAD_GMSH` constructor parameter): outward buffer
   for cut bridging. Default well below `point_tolerance` so distortion is
   invisible at user precision.
3. **`geometry_tolerance`** (new, `ModelManager` constructor parameter):
   gmsh's vertex-snap distance (`Geometry.Tolerance`). Must be strictly
   smaller than `perturbation` so the buffer survives ingestion.

`Geometry.ToleranceBoolean` stays at `point_tolerance` so genuine
`point_tolerance`-coincident features still fuse during the final fragment.

The ladder: `geometry_tolerance < perturbation ≪ point_tolerance ≤ ToleranceBoolean`.

### Defaults

With `set_precision` removed, shapely no longer constrains the buffer
size. `perturbation` can be as small as we want — practically, just
above floating-point noise.

- **`perturbation`** (new on `CAD_GMSH`, default `1e-9`): outward buffer
  applied to polygon entities before the cut cascade. At `1e-9`, distortion
  is invisible at any user precision. Six orders of magnitude above
  machine epsilon (`2.2e-16`) so floating-point math stays well-behaved.

- **`geometry_tolerance`** (new on `ModelManager`, default `point_tolerance`):
  gmsh's `Geometry.Tolerance` (vertex-snap distance during shape ingestion).
  Default kept at `point_tolerance` to preserve the previous user-vertex-snap
  contract (vertices within `point_tolerance` merge during ingestion).
  Decoupled from `point_tolerance` so users can override (e.g., for
  high-precision OCC operations).

- **`Geometry.ToleranceBoolean`** stays at `point_tolerance` (existing,
  unchanged). BOP fuzzy merge.

The hierarchy: `perturbation ≪ geometry_tolerance = point_tolerance ≤ ToleranceBoolean`.

Note that `perturbation` is now SMALLER than `geometry_tolerance`. That's
fine because gmsh's `Geometry.Tolerance` only snaps vertices that are
*not yet in the model*; once the buffered polygon is ingested, its
vertices don't get re-snapped against each other. The buffered shape
survives ingestion intact.

Both new parameters accept explicit overrides; defaults are derived
without any `tolerance_scale` parameter (the original spec proposed one
to bridge `point_tolerance / scale / scale` — no longer needed since
shapely doesn't constrain the buffer).

### What this does NOT change

- `Geometry.ToleranceBoolean` stays at `point_tolerance`.
- The bbox-clipping step in Pass A stays (it only matters when buffer is
  large enough to push past scene bounds; with a sub-tolerance buffer the
  clip is essentially a no-op but it's not worth removing in this work).
- The cut-cascade and InterfaceTag pipelines are unchanged.

## Code changes

### Phase 1 — Remove `shapely.set_precision` from entity constructors

- `meshwell/polysurface.py`: delete the `if point_tolerance > 0:` block
  that calls `shapely.set_precision`. Keep `point_tolerance` as a
  constructor parameter (still stored on the instance — other code reads
  `self.point_tolerance`).
- `meshwell/polyprism.py`: delete the equivalent `if point_tolerance > 0:`
  block (handles both the list and singleton paths).
- `meshwell/polyline.py`: delete its `if point_tolerance > 0:` block.
- `meshwell/interface_tag.py`: delete its `if point_tolerance > 0:` block.

The `import shapely` statements that were only used for `set_precision`
should also be removed where no longer needed (verify per file).

### Phase 2 — Tolerance parameters

- **`meshwell/model.py` — `ModelManager.__init__`**
  - Add parameter `geometry_tolerance: float | None = None`.
  - Default: `geometry_tolerance = point_tolerance` if not set.
  - Replace `gmsh.option.setNumber("Geometry.Tolerance", self.point_tolerance)`
    with `gmsh.option.setNumber("Geometry.Tolerance", self.geometry_tolerance)`.
  - Store `self.geometry_tolerance` for introspection.
  - `Geometry.ToleranceBoolean` still uses `point_tolerance`.

- **`meshwell/cad_gmsh.py` — `CAD_GMSH.__init__`**
  - Add parameter `perturbation: float | None = None`.
  - Default: `perturbation = 1e-9` if not set.
  - Replace the existing `@property def perturbation(self)` (returns
    `2 * self.point_tolerance`) with a stored attribute set in `__init__`.

- **`cad_gmsh()` top-level wrapper**
  - Add `perturbation: float | None = None` parameter; forward to processor.

### Phase 3 — Decouple InterfaceTag's snap from perturbation

- **`meshwell/cad_gmsh.py` — `process_entities` Pass B**
  - The current call passes `default_snap=self.perturbation`. With
    `perturbation = 1e-9`, the InterfaceTag strip is too narrow to
    overlap the buffered prism boundary at `linestring + perturbation`.
    Change to `default_snap=self.point_tolerance` so the strip is wide
    enough to find interfaces near the user's nominal trace at user
    precision.

### Phase 4 — Test calibration

- **`tests/test_cad_gmsh.py` — fix the 3 KeyError-failing tests**
  - In `test_cad_gmsh_mesh_order_lower_wins_in_overlap`,
    `test_cad_gmsh_same_mesh_order_ties_to_insertion_order`, and
    `test_cad_gmsh_keep_false_top_dim_removed_but_interface_named`,
    change `areas[ent.physical_name[0]] = total` to
    `areas[strip_suffix(ent.physical_name[0])] = total` (and import
    `strip_suffix` from `meshwell.cad_gmsh` if not already).

- **`tests/test_cad_gmsh.py` — tighten the loose tolerances on
  `test_cad_gmsh_2d_surface_partially_overlaps_3d_face_*`**
  - The three `< 5e-2` tolerance assertions can tighten to `< 1e-6`
    (perturbation is now `1e-9`, distortion is essentially zero).

- **New test** in `tests/test_cad_gmsh.py`:
  `test_cad_gmsh_perturbation_below_point_tolerance` — assert the
  resulting bounding box of a polygon entity is within `point_tolerance`
  of the input (now trivially passes with perturbation = `1e-9`).

## Test impact

### Tests fixed by the suffix-strip key fix (Phase 4)

- `test_cad_gmsh_mesh_order_lower_wins_in_overlap`
- `test_cad_gmsh_same_mesh_order_ties_to_insertion_order`
- `test_cad_gmsh_keep_false_top_dim_removed_but_interface_named`

These currently fail with `KeyError: 'A'` (NOT an inequality assertion as
originally claimed). The `physical_name[0]` suffix bug bypasses the
intended assertion entirely. After the suffix strip, the inequality
`areas["A"] > areas["B"]` passes cleanly with the new tiny perturbation
(area inflation ~`4e-9` for unit polygons; well within any reasonable
tolerance). The `abs(areas["A"] - 1.0) < 1e-6` style assertions also pass
— delta is `~4e-9`, far inside `1e-6`.

### Test tightened back

- `test_cad_gmsh_2d_surface_partially_overlaps_3d_face_is_carved_and_tagged`
  was loosened to `< 5e-2` for the old `2*pt` buffer. With the new
  `1e-9` perturbation, tighten to `< 1e-6` (room for FP rounding;
  actual delta is `~4e-9`).

### New regression test

- `test_cad_gmsh_perturbation_below_point_tolerance`: build a single
  PolySurface, run `cad_gmsh`, query the resulting 2D piece's bounding
  box, assert each coordinate is within `point_tolerance` of the input.

### Tests that should be unaffected

- All `tests/test_interface_tag.py` (15 tests). InterfaceTag's
  `default_snap` now resolves to `point_tolerance` (Phase 3), so the
  resolve algorithm continues to find interfaces correctly. Behavioral
  assertions (segment counts, face placements within ±1.0 ranges) are
  insensitive to whether perturbation is `2e-3` or `1e-9`.

## Validation

The user has noted that complex-geometry failures are intermittent and not
reproducible in simple unit tests. We cannot empirically validate that the
new defaults handle complex scenes. Two safeguards:

1. The default `perturbation = 1e-5` is still **10,000× larger** than typical
   FP-drift after OCC operations (~`1e-9`), giving generous bridging headroom.
2. Both `perturbation` and `geometry_tolerance` are constructor parameters,
   so a user who hits a complex-scene failure can override per scene without
   editing meshwell internals.

## Out of scope

- Adaptive per-scene perturbation tuning.
- Inward-buffer or asymmetric per-side buffering.
- Removing the buffer entirely.
- Auto-validation of complex-scene robustness — left to the user.
