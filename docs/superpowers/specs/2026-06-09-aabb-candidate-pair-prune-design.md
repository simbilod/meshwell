# AABB candidate-pair pruning — design

**Date:** 2026-06-09
**Branch:** `feat/structured-discrete-manual`
**Scope:** `meshwell/occ_xao_writer.py::_compute_physical_groups` outer loop only.

## Goal

Cut the outer-loop pair iteration in `_compute_physical_groups` from O(N²) full enumeration to O(N²)-then-sparse-iteration by pruning pairs whose entity-level AABBs cannot overlap. Output must be bit-identical to the current implementation.

## Motivation

The function is currently 1–3 % of `generate_mesh` wall time on scenes up to ~230 entities. The optimization is preemptive for production scenes (potentially hundreds of entities, heavily fragmented post-BOP). The 2026-06-01 numpy-vectorization (commit `4f7a0d0`) already optimized the inner per-face math; the outer pair iteration is the remaining algorithmic surface.

## Architecture

Add two private helpers in `meshwell/occ_xao_writer.py`, used in `_compute_physical_groups`:

### `_entity_union_aabbs(entity_aabbs)`

**Input:** `list[dict[int, tuple[float, ...]]]` — the per-entity dict of `{tshape_id: face_aabb}` already built in `_compute_physical_groups`.

**Output:** `tuple[np.ndarray, np.ndarray]` —
- `union_aabbs`: shape `(N, 6)`, each row `[xmin, ymin, zmin, xmax, ymax, zmax]` = element-wise (min for first 3, max for last 3) over the entity's face AABBs.
- `valid_mask`: shape `(N,)` boolean. `False` for entities with empty face AABB sets (rows in `union_aabbs` are filled with NaN sentinels but never read directly).

### `_candidate_pair_mask(union_aabbs, valid_mask, tol)`

**Input:**
- `union_aabbs`: `(N, 6)` from above.
- `valid_mask`: `(N,)` from above.
- `tol`: float, the AABB tolerance (`interface_aabb_tolerance`).

**Output:** `np.ndarray` of shape `(M, 2)` — pairs `(i, j)` with `i < j` whose inflated AABBs overlap. Entities with `valid_mask[i] == False` are treated as "intersects everything" (degenerate entities included in all pairs).

**Vectorized intersection check:**
```
xmin_i - tol <= xmax_j  AND  xmin_j - tol <= xmax_i
                       (and same for y, z)
```
Broadcast over (N, 1) vs (1, N) to produce `(N, N)` overlap matrix. Take upper triangle (`i < j`).

### Outer-loop change in `_compute_physical_groups`

Replace:
```python
for (i1, ent1), (i2, ent2) in combinations(enumerate(entities), 2):
    if ent1.dim <= 0 or ent2.dim <= 0:
        continue
    ...
```
With:
```python
union_aabbs, valid_mask = _entity_union_aabbs(entity_aabbs)
candidate_pairs = _candidate_pair_mask(
    union_aabbs, valid_mask, interface_aabb_tolerance
)
for i1, i2 in candidate_pairs:
    ent1, ent2 = entities[i1], entities[i2]
    if ent1.dim <= 0 or ent2.dim <= 0:
        continue
    ...
```

The inner per-face fallback math (the numpy `dists = np.abs(arr2 - b1_arr).max(axis=1)` block) stays unchanged.

## Correctness argument

**Claim:** If pair `(i, j)` produces any interface contribution under the current implementation, it appears in the candidate list.

**Proof:**
- *TShape-identity match:* if entities `i` and `j` share a TShape (some face `f` is in both `entity_boundary[i]` and `entity_boundary[j]`), then `f`'s AABB is in both entity AABB sets, so it's included in both union AABBs. The union AABBs therefore overlap at `f`. The inflated overlap test (with `tol > 0`) passes.
- *AABB-fallback match:* if a face `f1 ∈ entity_i` matches a face `f2 ∈ entity_j` with `L_inf(f1.aabb, f2.aabb) < tol`, then `f1.aabb ⊆ entity_i.union_aabb` and `f2.aabb ⊆ entity_j.union_aabb`. Their L_inf distance ≤ `tol` implies the inflated entity AABBs overlap.
- *Degenerate entities* (entity has no valid face AABBs) are treated as "intersects everything" — no pruning, identical behaviour.

**Iteration order:** `np.argwhere` on an upper-triangular boolean mask yields pairs in `(row, col)` lexicographic order, identical to `combinations(enumerate(entities), 2)`. Determinism preserved.

## Output equivalence

Output must be bit-identical: same `dict` keys, same shape sets per key, same shape iteration order within each set. The candidate-pair filter changes only which pairs are *visited*, not how a visited pair is processed. The unvisited pairs are guaranteed (by the proof above) to produce no output.

## Tests

1. **`test_compute_physical_groups_equivalence_meander`** — snapshot the meander stress scene's `_compute_physical_groups` output, then re-run with the new implementation and assert dict equality. Compare keys, per-key shape counts, per-key shape ordering (by `_HASHER`).

2. **`test_candidate_pair_mask_overlapping`** — two 3D boxes that overlap. Mask should include the pair.

3. **`test_candidate_pair_mask_disjoint_far`** — two 3D boxes far apart (≫ tol). Mask should exclude the pair.

4. **`test_candidate_pair_mask_edge_touching_within_tol`** — two boxes whose edges are within `tol/2`. Mask should include the pair (handles the inflated-overlap semantic).

5. **`test_candidate_pair_mask_lexicographic_order`** — for N=4 entities with various overlap patterns, `argwhere(mask)` order matches `combinations(range(4), 2)` filtered to overlapping pairs.

6. **`test_candidate_pair_mask_degenerate_entity`** — entity 0 has `valid_mask[0] == False`. All pairs `(0, j)` appear regardless of `j`'s union AABB.

7. **`test_entity_union_aabbs_single_face`** — entity with one face's AABB. Union AABB equals the face AABB.

8. **`test_entity_union_aabbs_empty`** — entity with empty face AABB set. `valid_mask[i] == False`.

## Performance characterization

Build/intersection cost: `_entity_union_aabbs` is O(total face count). `_candidate_pair_mask` is O(N²) memory (one (N, N) bool matrix) and O(N²) FLOPs.

For the current scene sizes (N ≤ 250): the (N, N) matrix is ≤ 60k bool entries, well inside numpy's sweet spot. No measurable overhead for setup.

For larger N (production scenes claimed ~bottlenecking): cost is dominated by `np.argwhere` and the upper-triangle mask construction. Both vectorized; expected microseconds.

At very large N (~10k entities), the (N, N) bool matrix is 100M entries (~100 MB). At that point we'd need a true spatial index (`shapely.STRtree` or `rtree`). This design intentionally does not solve that — the threshold is 50–100× larger than current production sizes.

## Files touched

- Modify: `meshwell/occ_xao_writer.py` — add 2 helper functions, modify `_compute_physical_groups` outer loop.
- Create: `tests/test_xao_writer_candidate_pair_prune.py` — tests 2–8 above.
- Modify: `tests/test_xao_writer_aabb_fallback.py` (or add a new dedicated equivalence test file) — test 1.

## Out of scope

- True spatial index (`shapely.STRtree`, `rtree` library). Deferred until measurements show the (N, N) bool matrix is itself a bottleneck.
- Caching `_shape_aabb` results across `_compute_physical_groups` invocations.
- The (separate) wedge.py unstaged bug fix.
- Any change to interface tagging semantics beyond pair filtering.
