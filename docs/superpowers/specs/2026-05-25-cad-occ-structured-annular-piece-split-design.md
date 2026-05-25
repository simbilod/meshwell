# Structured planner: split annular face_partition pieces for transfinite compatibility

**Date:** 2026-05-25
**Follow-up to:** `docs/superpowers/specs/2026-05-21-cad-occ-structured-all-layer-intersections-design.md`
**Scope:** make `tests/structured/test_stress_stacked_patterns.py::test_stacked_concentric_arc_discs_mesh_clean` (currently `xfail strict`) pass by splitting annular face_partition pieces into single-loop sub-pieces before they reach the phantom builder.

## Problem

The 2026-05-21 fixed-point face_partition refactor correctly propagates arc cuts vertically across stacked structured slabs. When the stack consists of arc-bearing discs of *decreasing radius*, e.g.:

- L1: R=1.0 disc at z=[0,1]
- L2: R=0.7 disc at z=[1,2]
- L3: R=0.5 disc at z=[2,3]

The planner correctly partitions L1's footprint into two pieces: an inner disc (R<0.7) and an **annular ring** (0.7<R<1). The annular ring is a Shapely Polygon with one exterior ring (the outer R=1 boundary) and one interior ring (the R=0.7 hole). Topologically, this is an annulus, not a disk.

The structured pipeline downstream assumes each face_partition piece is a topological *disk* with a single closed boundary loop that can be transfinite-meshed (3 or 4 corners). gmsh's `setTransfiniteSurface()` rejects annular faces because the parametric mesh requires single-loop boundary. Failure: `Surface N is transfinite but has K corners` (K = the apparent total vertex count when gmsh tries to interpret the multi-loop face as a single loop, varies with disc vertex count and BOP-introduced cuts).

## Goal

Make the xfail flip to passing by splitting any annular face_partition piece into single-loop sub-pieces before phantom build. Concretely:

- Annular pieces (those with non-empty `piece.interiors`) get cut radially so each output sub-piece is a single closed loop with 4 sides (2 arcs + 2 straight cuts in the typical case).
- The cut direction is deterministic across slabs so adjacent slabs in a vertical stack share the same cut line — guaranteeing conformality at z-interfaces.
- Each resulting sub-piece is a valid transfinite face: corner count is exactly 4 in the canonical case (outer arc start, outer arc end, inner arc start, inner arc end), with 2 arc sides and 2 straight sides.
- Existing single-loop pieces (no interiors) are passed through unchanged.

Out of scope:
- Pieces with multiple interior rings (more than one hole). Real geometry rarely produces these; deferred until a failing test exists.
- Non-circular interior boundaries (e.g., a square hole inside a disc piece). The split algorithm assumes interior rings correspond to arcs from neighbour slabs.

## Approach

Add a helper `_split_annular_piece(piece) -> list[Polygon]` and call it in `_partition_pieces_for_slab` post-polygonize. The helper:

1. Returns `[piece]` immediately when `piece.interiors` is empty.
2. Picks a deterministic radial cut line. Given the annular piece has one exterior ring with M vertices and one interior ring with N vertices, the cut connects:
   - The outer vertex `v_outer` whose angle (relative to the hole's centroid) minimizes some deterministic ordering (e.g., the vertex closest to the +x axis from the hole centroid).
   - The interior vertex `v_inner` at the *same* angle from the hole centroid (the closest existing vertex).
3. Slices the annular piece into two sub-pieces by polygonizing `piece.boundary ∪ LineString(v_outer, v_inner)`.
4. Returns both sub-pieces.

### Deterministic cut direction

The cut must produce identical cuts on adjacent slabs that share the same annular shape (so the z-interface is conformal). Two approaches:

**(A) Geometric canonical angle.** Always cut at angle θ=0 (the +x ray from the hole centroid). Snap each endpoint to the nearest existing vertex on its loop.

**(B) Vertex-based canonical pick.** Choose `v_outer` as the existing outer vertex with maximum x (ties broken by max y). `v_inner` is the existing interior vertex closest in angle to `v_outer`.

(B) is preferred — it deterministically hits *existing* vertices without snapping, so the cut endpoints are bit-exact across slabs that share the same polygon vertex sequences. (A) requires tolerance-based vertex snapping, which may diverge across slabs whose vertex sets differ slightly.

### Why a single radial cut is sufficient

For one outer arc + one inner arc + the cut line + the cut line (traversed in opposite directions on the two sub-pieces), each sub-piece's boundary has exactly four "edges" in topological terms: outer arc (half), inner arc (half), cut line (one direction), cut line (other direction — but on the *other* sub-piece). Both sub-pieces are 4-cornered single loops.

If a future scene produces a piece with multiple interior holes, the algorithm extends naturally: one radial cut per hole, applied in series. Out of scope for this spec but the architecture allows it.

### Where to call the splitter

Inside `_partition_pieces_for_slab` in `meshwell/structured/plan.py`, after the existing polygonize + filter step:

```python
def _partition_pieces_for_slab(slab, cut_sources):
    ...
    pieces = [
        piece
        for piece in raw
        if slab.footprint.contains(piece.representative_point())
    ]
    pieces = pieces if pieces else [slab.footprint]

    # NEW: split any annular pieces so the phantom builder sees only
    # single-loop sub-pieces.
    split: list[Polygon] = []
    for p in pieces:
        split.extend(_split_annular_piece(p))
    return split
```

### Provenance integration

`_classify_piece_boundary` walks each piece's exterior + interior edges and labels them via the arc index. After the annular split, sub-pieces have only an exterior ring; the cut line shows up as two straight `PieceLineEdge` segments. The classifier produces correct provenance with no other changes.

The arc-merge step (`_merge_arc_into_index`) is unaffected — it already runs before the split, and the arcs themselves don't change shape due to splitting.

## Code changes

### `meshwell/structured/plan.py`

- Add `_split_annular_piece(piece: Polygon) -> list[Polygon]`. Returns `[piece]` when `not piece.interiors`. Otherwise: pick `v_outer` (max-x, ties max-y) on `piece.exterior`, pick `v_inner` on the (single, for now) interior ring closest in angle to `v_outer` from the interior's centroid, polygonize `unary_union([piece.boundary, LineString([v_outer, v_inner])])`, filter to pieces with no interiors that lie inside the original `piece`, return them.
- In `_partition_pieces_for_slab`: after the `pieces if pieces else [slab.footprint]` fallback, walk through pieces and replace each annular piece with its split.
- Raise `StructuredPartitionConvergenceError` (or a new error class — TBD during implementation) if a piece has more than one interior ring, since that's deferred.

### No spec.py changes

No new error classes; no new dataclass fields. The split is internal to the planner.

## Tests

### Primary regression flip

`tests/structured/test_stress_stacked_patterns.py::test_stacked_concentric_arc_discs_mesh_clean`:
- Remove `@pytest.mark.xfail(...)`.
- Existing assertions stay (wedge cells present, 3 physicals, 0 orphan triangles).

### New plan-only test in `tests/structured/test_plan.py`

- `test_annular_piece_splits_into_single_loop_subpieces`: construct a Polygon with one exterior ring (R=1 disc) and one interior ring (R=0.5 disc, concentric). Pass it through `_split_annular_piece`. Assert: 2 sub-pieces, each with `len(interiors) == 0`, union covering the original area, and pairwise-disjoint.

- `test_annular_piece_passthrough_when_no_interiors`: pass a simple disc (no holes). Assert: returns `[piece]` unchanged (same object identity is fine).

- `test_annular_split_deterministic_across_repeated_calls`: call `_split_annular_piece` twice on the same input and assert the two outputs are equal (same coordinates, same order). Guards against non-deterministic vertex picking.

### Cross-stack determinism

The xfail flip itself exercises cross-stack determinism implicitly: if L1 and L2 split their annular pieces differently at z=1, the mesh would have non-matching seams across the interface, surfacing as orphan triangles. Zero-orphan assertion in the existing test catches this.

## Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| Cut endpoint doesn't land on an existing vertex (due to floating-point), introducing a tiny new vertex that breaks z-interface alignment | Vertex-based canonical pick (option B above) uses existing polygon vertices — no snapping required. |
| Multi-hole pieces from exotic geometry (e.g., nested ring stacks) | Raise on `len(piece.interiors) > 1` with a clear error message pointing here. Defer multi-hole support to a follow-up spec when the first failing test arrives. |
| Cut direction makes the resulting sub-pieces highly skewed (e.g., one very thin sliver) | Acceptable for now — transfinite handles skewed quads. If mesh quality becomes an issue, add a secondary cut perpendicular to the first. |
| Cut line happens to pass through an existing interior vertex of a different feature | Vertex-based pick from `piece.exterior` / `piece.interiors[0]` only; doesn't introduce new vertices outside the annular piece itself. |
| Polygonize on `piece.boundary ∪ cut_line` returns pieces in non-deterministic order | Sort returned sub-pieces by `representative_point()` lexicographically before returning, so downstream phantom assembly sees a stable order. |

## Out of scope (future specs)

- **Pieces with multiple interior holes.** When the first failing test arrives, extend the algorithm to apply one radial cut per hole.
- **Non-arc interior boundaries.** If a future planner step produces an annular piece whose interior is, e.g., a square hole (from a non-arc neighbour), the split algorithm still works (the cut still lands on existing vertices), but the resulting sub-pieces would have mixed arc + straight + arc + straight boundaries that need careful provenance labeling.
- **Choosing cut direction to minimize mesh skew.** Current heuristic (max-x outer vertex) is simple and deterministic. If real designs produce highly eccentric annular pieces where the canonical cut yields poor mesh quality, revisit the heuristic.
