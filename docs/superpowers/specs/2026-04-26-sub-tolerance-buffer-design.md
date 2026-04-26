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

## Solution

Decouple three tolerances that are currently conflated to `point_tolerance`:

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

A single dimensionless **`tolerance_scale`** parameter (new, on both
`ModelManager` and `CAD_GMSH`, default `100.0`) sets the gap between
adjacent levels in the ladder:

- `perturbation = max(point_tolerance / tolerance_scale, 1e-9)` →
  `1e-5` at default scale=100 and default `point_tolerance=1e-3`.
- `geometry_tolerance = max(perturbation / tolerance_scale, 1e-12)` →
  `1e-7` at default scale=100.

Both `perturbation` and `geometry_tolerance` can also be passed
explicitly as overrides. Explicit values take precedence over the
scale-derived defaults (the scale is just a default-computation
convenience).

Floors of `1e-9` / `1e-12` keep the values safely above double-precision
machine epsilon (~`2.2e-16`) regardless of the scale chosen.

Rationale for one scale (vs. two independent ratios): `geometry_tolerance`
needs to be strictly smaller than `perturbation` to avoid snapping the
buffer away; the same headroom that keeps perturbation below
`point_tolerance` (avoiding visible distortion) also makes sense between
perturbation and geometry_tolerance (avoiding gmsh snap collapsing the
buffer). One knob is easier to reason about than two.

### What this does NOT change

- `Geometry.ToleranceBoolean` stays at `point_tolerance`.
- The bbox-clipping step in Pass A stays (it only matters when buffer is
  large enough to push past scene bounds; with a sub-tolerance buffer the
  clip is essentially a no-op but it's not worth removing in this work).
- The cut-cascade and InterfaceTag pipelines are unchanged.

## Code changes

- **`meshwell/model.py` — `ModelManager.__init__`**
  - Add parameters `tolerance_scale: float = 100.0` and
    `geometry_tolerance: float | None = None`.
  - Compute `self.geometry_tolerance` as the explicit value if given,
    otherwise `max(point_tolerance / (tolerance_scale ** 2), 1e-12)`.
    (The `**2` is because the geometry_tolerance sits two scale-steps
    below `point_tolerance` in the ladder: one step to perturbation,
    one more to geometry_tolerance.)
  - Replace `gmsh.option.setNumber("Geometry.Tolerance", self.point_tolerance)`
    with `gmsh.option.setNumber("Geometry.Tolerance", self.geometry_tolerance)`.
  - Store `self.tolerance_scale` and `self.geometry_tolerance` for introspection.
  - `Geometry.ToleranceBoolean` still uses `point_tolerance`.

- **`meshwell/cad_gmsh.py` — `CAD_GMSH.__init__`**
  - Add parameters `tolerance_scale: float = 100.0` and
    `perturbation: float | None = None`.
  - Compute `self.perturbation` as the explicit value if given, otherwise
    `max(point_tolerance / tolerance_scale, 1e-9)`.
  - Replace the existing `@property def perturbation(self)` (returns
    `2 * self.point_tolerance`) with a stored attribute set in `__init__`.
  - When the processor owns the model (`model is None`), forward
    `tolerance_scale` to the `ModelManager` it constructs (so the
    derived `geometry_tolerance` matches the processor's perturbation).

- **`cad_gmsh()` top-level wrapper**
  - Add `tolerance_scale: float = 100.0` and `perturbation: float | None = None`
    parameters; forward both to the `CAD_GMSH` processor.

## Test impact

### Tests expected to start passing

- `test_cad_gmsh_mesh_order_lower_wins_in_overlap`
- `test_cad_gmsh_same_mesh_order_ties_to_insertion_order`
- `test_cad_gmsh_keep_false_top_dim_removed_but_interface_named`

These currently fail because `2 * point_tolerance` buffer inflates the
loser entity's area enough to flip `areas["A"] > areas["B"]`. With
perturbation ≈ `1e-5`, area inflation per side is ~`2e-5`; total area
delta is ~`4e-5` for unit polygons. Their assertions use `< 1e-6` for
absolute area equality — that may not be tight enough either. Plan
includes a calibration step: run with the new defaults, tighten or
loosen specific assertions as warranted by the new arithmetic.

### Test to tighten back

- `test_cad_gmsh_2d_surface_partially_overlaps_3d_face_is_carved_and_tagged`
  was loosened to `5e-2`. With the new buffer, tighten back to `< 1e-4`.

### New regression test

- `test_cad_gmsh_perturbation_below_point_tolerance`: build a single
  PolySurface, run `cad_gmsh`, query the resulting 2D piece's bounding
  box, assert each coordinate is within `point_tolerance` of the input.
  Pins the contract that distortion is below user-promised precision.

### Tests that must NOT change behavior

- All `tests/test_interface_tag.py` (15 tests). InterfaceTag uses
  `self.perturbation` for the default snap distance, so the snap will
  shrink along with the buffer. The InterfaceTag-internal arithmetic
  is unaffected by perturbation magnitude as long as `snap_distance >
  perturbation` (which the default arrangement preserves: snap defaults
  to perturbation, and the bridging only needs snap ≥ perturbation).

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
