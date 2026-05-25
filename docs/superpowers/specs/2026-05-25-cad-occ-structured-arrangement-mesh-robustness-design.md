# Structured planner: arrangement robustness for arc-bearing mesh paths

**Date:** 2026-05-25
**Builds on:** `docs/superpowers/specs/2026-05-25-cad-occ-structured-planar-arrangement-design.md`
**Resurrects (partially):** `docs/superpowers/specs/2026-05-25-cad-occ-structured-annular-piece-split-design.md` (originally superseded)

**Scope:** fix the two planner-side gaps that block the planar-arrangement pipeline from producing meshable output for arc-bearing stacked scenes. Both `tests/structured/test_stress_stacked_patterns.py::test_stacked_concentric_arc_discs_mesh_clean` and `test_stacked_overlapping_ring_segments_mesh_clean` (currently strict xfails) flip to passing as a direct consequence.

## Problem

The planar-arrangement refactor (commit `486a3b6`, spec `2026-05-25-cad-occ-structured-planar-arrangement-design.md`) wired a new pipeline that produces canonical edges per connected stack. Existing non-arc tests stayed green; arc xfails remained xfailed because two latent mesh-stage failure modes were not addressed by the architectural change alone:

1. **Annular faces from non-crossing nested boundaries.** Two concentric circles (e.g., L1's R=1 outer disc and L2's R=0.7 disc at z=1) don't cross. `shapely.unary_union` produces a `MultiLineString` of two disjoint closed curves. `polygonize` returns two faces: the inner disc (R<0.7) and the **annulus** (0.7<R<1, with the inner circle as an interior ring of the polygon). gmsh's `setTransfiniteSurface` rejects 5+ corner faces (which an annulus presents as a multi-loop boundary); the structured pipeline falls back to tet meshing, producing non-conformal wedge/tet interfaces and orphan triangles. The original planar-arrangement spec claimed every arrangement face would be topologically a disk by construction — this is true when boundaries cross but false when they're nested.

2. **Floating-point endpoint divergence at periodic wraparound.** Ring-segment polygons constructed via `cos(theta) / sin(theta)` for `theta in [0, 2π]` produce a vertex at `theta=2π` that differs from the `theta=0` vertex by ~2e-16. `shapely.unary_union` treats these as distinct points; the arrangement misses the junction that the geometry intended. Downstream, the phantom builder sees lateral OCC faces with no `IsSame()` partner across the wraparound — `PhantomMap` raises `RuntimeError: PhantomMap lateral … has no IsSame() match in the XAO compound`.

Both fixes are planner-side. The mesh-stage code (`phantom.py`, `builder.py`) needs no changes once the planner emits a clean arrangement.

## Goal

Both arc xfails flip to passing:

- `test_stacked_concentric_arc_discs_mesh_clean` — concentric discs of decreasing radius, identify_arcs=True on each.
- `test_stacked_overlapping_ring_segments_mesh_clean` — three rotated half-annuli (R=0.5–1.0), identify_arcs=True on each.

Plus a new unit test that constructs an annular face and asserts the split produces single-loop sub-pieces.

Other existing tests stay green. The legacy `compute_face_partition` machinery (already dead code per the prior spec) can be deleted in a follow-up cleanup task.

Out of scope:
- Changes to `phantom.py` or `builder.py`.
- Multi-hole annular faces (faces with two or more interior rings). The current planner produces at most one interior ring per face in realistic scenes; defer until a failing test demonstrates otherwise.

## Approach

Two independent fixes in `meshwell/structured/plan.py`. Apply in order; each can land separately.

### Fix 1 — Coordinate-snap input boundaries before `unary_union`

The fp-precision issue manifests at the geometric input level: two vertices that are "the same point" in user intent but differ by ε in floating-point representation cause `unary_union` to miss the junction. Fix is input sanitization: snap every boundary's vertices to a tolerance grid before passing them to the arrangement.

Tolerance: use `min(slab.point_tolerance for slab in stack)` if any stack member has `point_tolerance` (PolyPrism does), else default `1e-9`. The snap is conservative — it should be tighter than any feature size in the scene so it can't accidentally merge distinct vertices.

Implementation: add a helper `_snap_boundary_coords(geom, tol)` that walks every coordinate in a geometry and rounds it to `round(x / tol) * tol`. Apply to each boundary in `_collect_stack_boundaries` before returning. Use shapely's `set_precision(geom, grid_size=tol, mode="pointwise")` if available (shapely 2.x has it).

```python
def _snap_boundary_coords(geom, tol: float):
    """Snap vertices to a tolerance grid so unary_union sees coincident endpoints as equal."""
    import shapely
    return shapely.set_precision(geom, grid_size=tol, mode="pointwise")
```

Call from `_collect_stack_boundaries` after collecting each boundary:

```python
tol = min((getattr(ent, "point_tolerance", 1e-9) for ent in entities), default=1e-9)
boundaries = [_snap_boundary_coords(b, tol) for b in boundaries]
```

This is enough to fix the ring-segment wraparound case. The snap is a no-op when inputs are already on a coarser grid.

### Fix 2 — Annular face splitting

When `polygonize` returns a face with non-empty `interiors`, split it into single-loop sub-pieces via radial cuts.

Algorithm:

1. For each face F with `len(F.polygon.interiors) > 0`:
   a. Use the first interior ring's centroid as the cut origin (deterministic when there's exactly one hole — the common case).
   b. Generate two diametrically opposite cut lines from the centroid, oriented along the +x and −x axes. Both extend beyond the outer ring's bounding box so they fully cross F.
   c. Clip each cut to F's geometry (only the segment inside F survives).
2. Add the clipped cut lines to the stack's boundary set.
3. Re-run `_planar_arrangement(boundaries)` on the augmented input.

The re-run produces a fresh arrangement where the formerly-annular face is now split into two single-loop sub-pieces (each a half-annular region with outer arc + radial + inner arc + radial). Adjacent slabs in the stack see the same cuts (since they share the boundary set), so the split is consistent across z-interfaces.

Determinism: the +x / −x choice is deterministic globally. For real scenes it doesn't matter which axis — what matters is that all slabs in the stack pick the same cuts.

Implementation: this fits as a post-`_planar_arrangement` loop in `build_stack_arrangements`. After the initial arrangement, scan faces; if any has interiors, generate cuts, augment boundaries, re-run. Repeat until no annular faces remain (typically one pass).

```python
def build_stack_arrangements(slabs, entities) -> dict[int, StackArrangement]:
    components = _connected_z_components(slabs)
    arrangements: dict[int, StackArrangement] = {}
    for comp_idx, stack in enumerate(components):
        boundaries = _collect_stack_boundaries(stack, entities)
        if not boundaries:
            arrangements[comp_idx] = StackArrangement(edges=[], faces=[])
            continue
        # Iterate: arrangement -> detect annular faces -> add radial cuts -> repeat.
        for _ in range(_MAX_ANNULAR_SPLIT_PASSES):
            edges, faces = _planar_arrangement(boundaries)
            annular = [f for f in faces if f.polygon.interiors]
            if not annular:
                break
            for face in annular:
                boundaries.extend(_generate_radial_cuts_for_annular_face(face.polygon))
        # Continue with arc fit + coalesce as before...
```

`_MAX_ANNULAR_SPLIT_PASSES = 4` as a safety cap; one pass suffices for typical scenes. If the cap fires without converging, raise a new `StructuredArrangementError`. (One per-spec error class to surface diagnostics without leaking implementation details into `RuntimeError`.)

Helper:

```python
def _generate_radial_cuts_for_annular_face(poly: Polygon) -> list[LineString]:
    """For one annular face, produce two diametrically-opposed radial cut lines.

    Cut origin = centroid of the first interior ring (the hole).
    Cuts are oriented along +x and -x, extending past the face's bounding box,
    clipped to the face's geometry so only the in-face portion is added.
    """
    from shapely.geometry import LineString

    if not poly.interiors:
        return []
    hole_centroid = list(poly.interiors[0].centroid.coords)[0]
    cx, cy = hole_centroid
    minx, miny, maxx, maxy = poly.bounds
    half_extent = max(maxx - cx, cx - minx, maxy - cy, cy - miny) * 2 + 1
    raw_cuts = [
        LineString([(cx, cy), (cx + half_extent, cy)]),
        LineString([(cx, cy), (cx - half_extent, cy)]),
    ]
    clipped = []
    for cut in raw_cuts:
        c = cut.intersection(poly)
        if not c.is_empty:
            clipped.append(c)
    return clipped
```

### Order of fixes

Fix 1 (coord snap) is independent and lands first. Fix 2 (annular split) depends on the arrangement being stable, so coord snap should be applied first to avoid spurious fp-noise faces.

## Tests

### Primary regression flips

`tests/structured/test_stress_stacked_patterns.py`:
- `test_stacked_concentric_arc_discs_mesh_clean` — remove `@pytest.mark.xfail(...)`. Annular-split fix lands this.
- `test_stacked_overlapping_ring_segments_mesh_clean` — remove `@pytest.mark.xfail(...)`. Coord-snap fix lands this.

### New plan-only tests in `tests/structured/test_plan.py`

- `test_snap_boundary_coords_merges_near_coincident_endpoints` — construct a LineString with two near-coincident endpoints (differ by 1e-12); apply `_snap_boundary_coords` with `tol=1e-9`; assert the resulting coordinates have bit-identical endpoints.

- `test_annular_face_split_produces_single_loop_pieces` — construct a Polygon with one exterior (R=1) + one interior (R=0.5) ring, pass through `build_stack_arrangements` via a synthetic stack, assert the resulting `StackArrangement.faces` has no faces with interior rings.

- `test_concentric_arc_planner_no_annular_pieces` — actually-realistic case: two concentric arc-bearing slabs at touching z-levels; assert the planner emits zero annular faces.

### Comprehensive stress test (still pending from prior spec)

After both fixes land and xfails flip, retry Task 16 from the prior plan (`test_complex_disjoint_arc_stacks_with_keep_mix_mesh_clean`). It will exercise both fixes simultaneously.

## Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| Coord snap merges two genuinely distinct vertices within tolerance | The tol is conservative (1e-9 default, or the min `point_tolerance` of any stack entity — typically 1e-3 or smaller). User-authored features rarely reach those scales. |
| Annular split's radial cuts cross another arrangement edge unexpectedly, creating tiny slivers | The cuts are clipped to the annular face only (`.intersection(poly)`); they don't extend outside. Sliver risk is bounded by the cut geometry being short and straight. |
| Multi-hole annular faces produce ambiguous cuts | Out of scope; only the first interior ring is used. If a multi-hole face appears, fewer cuts will be generated and the split loop will iterate (the safety cap `_MAX_ANNULAR_SPLIT_PASSES` bounds runaway). |
| `shapely.set_precision` not available in older shapely | The repo's `pyproject.toml` should already require shapely 2.x (which has `set_precision`). Verify before relying on it; fall back to manual rounding if needed. |
| Annular-split changes arrangement edge IDs, breaking the orchestrator's downstream consumers | The orchestrator re-runs `_planar_arrangement` after augmenting boundaries — produces fresh edges and faces. The face-to-slab assignment in Step F runs on the final post-split arrangement; no cross-pass state to preserve. |

## Out of scope (future work)

- **Mesh-stage robustness for cases where the annular split fundamentally can't be made deterministic across z-interfaces.** The current axis-aligned cuts are deterministic but axis-aligned only — if user geometry demands rotation-invariant cuts (rare), a future spec can introduce angle parameters per stack.
- **Multi-hole faces.** A face with multiple interior rings would need one cut per ring; deferred until needed.
- **Cleanup task: delete legacy `compute_face_partition` machinery.** Already dead code per the prior spec; remove in a follow-up after both xfails flip and the suite is fully green.
