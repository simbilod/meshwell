# Structured planner: all-layer intersection propagation via fixed-point face_partition iteration

**Date:** 2026-05-21
**Scope:** make `tests/structured/test_stress_stacked_patterns.py::test_four_stacked_layers_misaligned_seams_mesh_clean` (currently `xfail strict`) pass by propagating face_partition cuts vertically through the structured slab stack.

## Problem

When multiple structured PolyPrism layers are stacked vertically and each layer has internal lateral seams at different XY positions per layer, gmsh fails with `Surface N is transfinite but has 5 corners`. OCC fragmentation imprints both above- and below-neighbour seams onto interface faces, producing 5-corner faces that the structured pipeline still treats as transfinite (which requires 3 or 4 corners).

Concretely, in the 4-layer xfail scenario (seams at x = 1.0, 1.7, 2.5, 3.2 in layers 1–4 respectively), a slab in layer 1 with footprint `[1.0, 4.0] × [0, 2]` only gets a face_partition cut at x = 1.7 (from layer 2's direct neighbour boundary). It does **not** get cuts at x = 2.5 (from layer 3) or x = 3.2 (from layer 4), even though those seams propagate through the BOP fragmentation chain `L4 → L3 piece boundaries → L2 piece boundaries → L1 top face` and ultimately produce vertices on the lateral faces of the layer-1 piece — yielding 5+ corner lateral faces that break transfinite meshing.

The root cause is in `meshwell/structured/plan.py:compute_face_partition` (lines 570–647): each slab collects cut sources from z-touching **neighbour footprints**, not from neighbour **face_partition pieces**. So a cut introduced one z-step away is never propagated transitively across the stack.

## Goal

Make the planner compute face_partitions that are stable under all-stack BOP fragmentation. Specifically:

- The xfail flips to a regular passing test.
- The three existing passing tests in `test_stress_stacked_patterns.py` remain green.
- All pre-existing structured-pipeline tests remain green; any tests asserting exact piece counts get updated only if the count is genuinely refined by the new propagation.

Out of scope for this design (deferred to future specs):
- Lateral-cut propagation between structured slabs in the **same** z-interval (e.g., L1L and L1R abutting at x = 2 within z = [0, 1]) where their shared edge is *interior* to a neighbour. Today's planner already handles the common case where same-z-interval slabs only contribute exterior boundary edges.
- Arc provenance under transitive propagation (the xfail uses straight-edge slabs only).

## Approach

Replace the single-pass `compute_face_partition` with a **fixed-point iteration** that uses each neighbour's *current* `face_partition` piece boundaries as cut sources (instead of just the neighbour's footprint boundary).

### Algorithm

```
initialize: slab.face_partition = [slab.footprint] for every slab
max_iter = min(stack_depth + 2, 16)

for pass in 1..max_iter:
    changed = False
    for slab in slabs:
        cut_sources = []
        for n_entity in z_touching_unstructured_entities(slab):
            cut_sources.append(n_entity.footprint.boundary)
        for n_slab in z_touching_structured_slabs(slab):
            for piece in n_slab.face_partition:
                cut_sources.append(piece.boundary)

        old_cut_wkb = sorted_wkb(slab.cached_cut_sources)
        new_cut_wkb = sorted_wkb(cut_sources)
        if new_cut_wkb == old_cut_wkb:
            continue                               # slab is stable for this pass

        slab.cached_cut_sources = cut_sources
        combined = unary_union([
            slab.footprint.boundary,
            unary_union(cut_sources).intersection(slab.footprint),
        ])
        slab.face_partition = [
            p for p in polygonize(combined)
            if slab.footprint.contains(p.representative_point())
        ] or [slab.footprint]
        changed = True

    if not changed:
        break
else:
    # Loop exhausted without `break` -> cap hit. `unstable` is the list of
    # slabs whose cached_cut_sources changed during the final pass.
    unstable = [s for s in slabs if s.cut_sources_changed_in_last_pass]
    raise StructuredPartitionConvergenceError(
        f"face_partition did not converge after {max_iter} passes; "
        f"unstable slabs: {[(s.physical_name, s.zlo, s.zhi) for s in unstable]}"
    )

attach_face_partition_provenance(slabs, entities)    # once, on converged partitions
```

### Why iteration 1 reproduces today's behavior

On the first pass, every slab's `face_partition` is `[footprint]`, so each neighbour slab contributes exactly one piece whose boundary equals its footprint boundary. The union of structured-neighbour piece boundaries plus unstructured-neighbour footprint boundaries is therefore identical to today's `unary_union([poly.boundary for poly in all_neighbour_polys])`. Existing tests that pass today continue to pass after iteration 1 produces the same partition; iteration 2+ only fires if a slab gained new cuts from its neighbours during pass 1.

### Convergence

Each pass can only **add** cuts to a slab's partition (the cut-source set grows monotonically because new piece boundaries can only emerge from previous pass refinements). A cut introduced in layer k propagates at most one z-step per pass. Therefore convergence is bounded by the longest face-touching z-chain, K. The cap `min(K + 2, 16)` absorbs rounding noise; the hard ceiling of 16 protects against pathological scenes.

### Termination check

Compare cut-source sets, not polygonized piece sets. Shapely's `polygonize` does not guarantee piece ordering, and Polygon equality across `unary_union`/`polygonize` roundtrips is sensitive to vertex ordering and coincident-boundary fuzz. Cut-source WKB blobs sorted lexicographically are deterministic and cheap.

### Failure surfacing

If `max_iter` is exhausted without convergence, raise `StructuredPartitionConvergenceError` listing the still-changing slabs. This is preferable to silently producing a malformed mesh, and gives a clear signal for future bug reports.

## Code changes

### `meshwell/structured/plan.py`

Refactor `compute_face_partition` (lines 570–647) into:

- `compute_face_partition(slabs, entities)` — orchestrator: initialize, loop, attach provenance once at the end.
- `_partition_pieces_for_slab(slab, slabs, entities, skip_indices) -> list[Polygon]` — single-slab inner kernel (today's per-slab logic).
- `_collect_cut_sources(slab, slabs, entities, skip_indices) -> list[BaseGeometry]` — walks z-touching unstructured entities (footprint boundary) and z-touching structured slabs (piece boundaries from current `face_partition`).
- `_structured_slabs_touching_z(z, slabs, skip_slab_ids, tol) -> list[Slab]` — mirror of the existing `_neighbours_touching_z` but over the slab list.

Existing `_neighbours_touching_z` stays as-is and is used by `_collect_cut_sources` for the unstructured-entity arm. Add a small filter so that entities backed by a slab are *not* counted twice (i.e., unstructured-only).

The provenance block currently embedded around lines 624–632 is moved into a new helper `_attach_face_partition_provenance(slabs, entities)` that runs once after the fixed-point loop converges.

### `meshwell/structured/spec.py`

Add:

```python
class StructuredPartitionConvergenceError(RuntimeError):
    """face_partition fixed-point iteration did not converge within the iteration cap."""
```

next to the other structured-pipeline errors. Export from `meshwell/structured/__init__.py`.

### `meshwell/structured/__init__.py`

Add `StructuredPartitionConvergenceError` to the import list and `__all__`.

## Tests

### Primary regression flip

`tests/structured/test_stress_stacked_patterns.py::test_four_stacked_layers_misaligned_seams_mesh_clean`:
- Remove `@pytest.mark.xfail(...)`.
- Existing assertions stay: wedge cells produced, all 8 physicals present, zero orphan boundary triangles.

### New plan-only unit tests in `tests/structured/test_plan.py`

- `test_partition_propagates_cut_two_steps`: 3-layer stack where only the middle layer has an internal seam. Assert that after planning, the top and bottom layers' face_partitions both contain the middle's cut (propagated one step in each direction).
- `test_partition_misaligned_seams_each_slab_partitioned_by_union`: the 4-layer misaligned scenario, plan-only. Assert each slab's `face_partition` piece count matches the union of seams from all layers whose footprints chain to it.
- `test_partition_converges_within_K_passes`: expose an iteration counter (e.g., via a module-level debug variable or return value extension) and assert convergence ≤ stack_depth + 2 for a 4-layer scene.
- `test_partition_raises_if_not_converged`: synthetic scene that artificially trips the cap. Implementation note: the cap will be a module-level constant in `plan.py` (e.g., `_PARTITION_FIXED_POINT_CAP = 16`) so it can be monkeypatched in the test to a small value (e.g., 1) on a 3-layer scene to force the cap path. Assert `StructuredPartitionConvergenceError`.

### Existing-test fallout audit

After the refactor, run the full `tests/structured/` suite. Any `len(face_partition) == N` assertions that flip get updated only if the new count is correct (more pieces from genuine propagation). Specific watchpoints:

- `tests/structured/test_end_to_end_multipiece.py:34` expects `len == 2`; no transitive cuts in that scene → should stay 2.
- `tests/structured/test_plan.py` overlap and face-partition tests should be unaffected on iteration 1.

If a test's assertion was previously masking under-partition, update the expected count and add a one-line comment in the test explaining the propagation source.

## Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| Iteration over-cuts a slab (cut sources from a far-away neighbour incorrectly intersect the slab footprint) | The per-pass cut-source set is filtered by `intersection(slab.footprint)`. A cut that doesn't intersect contributes nothing. |
| Shapely `polygonize` produces piece-ordering drift across iterations, falsely triggering "changed" | Compare cut-source WKB sets, not piece WKB sets. Cut sources are inputs to polygonize, not outputs, so ordering drift is contained. |
| Convergence cap hit on a legitimate scene | Cap is `min(K + 2, 16)`; K rarely exceeds 5 in real designs. If it triggers in the wild, the error message identifies unstable slabs for follow-up. |
| Performance regression on large scenes | Each pass is O(N · avg_neighbour_pieces); for N≈1000, K≈5, worst case ≈ 5000 polygonize calls, sub-second. No optimization needed in this spec; revisit if production scenes exceed it. |
| Existing tests assert exact piece counts | Audit step in the test plan covers this. Most are unaffected because iteration 1 reproduces today's behavior. |

## Out of scope (future specs)

- **Same-z-interval lateral cut propagation** (brainstorm scope b): if two structured slabs in the same z-interval abut at an XY position that lies *interior* to a vertical neighbour's footprint, the abutting edge should propagate to the neighbour's face_partition. Today's planner doesn't handle this; this spec doesn't either.
- **Void-pattern multi-layer columns** (brainstorm scope c): when layer N has a void filled by layer N+1's material, the column's cross-layer interface logic could need more than vertical cut propagation. Not in scope here.
- **Arc edge provenance under propagation**: provenance is computed once after convergence; arc handling under transitively-introduced piece boundaries is not validated by the xfail scene and is deferred to a separate spec if needed.
