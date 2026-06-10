# Arrangement-canonical arc fitting for cohort sub-pieces

Status: Draft (design accepted, awaiting plan)
Owner: simbilod
Date: 2026-06-10

## Problem

The greedy arc fitter `decompose_vertices_2d`
([meshwell/geometry_entity.py:110](../../../meshwell/geometry_entity.py#L110))
is called once per **sub-piece ring** during the structured cohort build
([meshwell/structured/build.py](../../../meshwell/structured/build.py)).
Two sub-pieces that share an arc-shaped boundary run will each pass the
fitter a DIFFERENT full ring (different seam picked by
`_find_canonical_seam`, different surrounding context). The greedy walk
can therefore emit a different segmentation along the SHARED coordinate
sequence — e.g. one sub-piece emits `[arc(A,B), line(B,C)]` while the
other emits `[arc(A,C)]` over the same `A→B→C` coordinates. The
`EdgeRegistry.arc_xy` cache keys on `(sorted endpoints, mid)`
([build.py:134](../../../meshwell/structured/build.py#L134)), so the two
sub-pieces produce DIFFERENT TShapes for what should be one shared
curve. Downstream this turns into mismatched faces in the cohort
compound and tiny BOP fragments at curved boundaries.

The fitter is already deterministic for a single closed ring (seam +
direction canonicalisation handles ring rotations of identical input).
The bug surfaces only when DIFFERENT rings share a common boundary
subset.

## Goal

Inside the cohort pre-pass, fit arcs on the cohort's PLANAR ARRANGEMENT
of input boundary lines exactly once per unique arrangement edge, and
route every sub-piece's wire builder through those canonical
segmentations. Sub-pieces that share an arrangement edge will TShape-
match by construction.

## Scope

In-scope:
- `meshwell/structured/decompose.py`: extend the cohort arrangement
  with canonical edge records + arc fitting.
- `meshwell/structured/types.py`: extend `Arrangement` with the new
  fields; add an `ArrangementEdge` dataclass.
- `meshwell/structured/build.py`: route `EdgeRegistry.polyline_xy`,
  `polyline_segments`, and `_build_horizontal_face` through the
  arrangement when one is supplied.
- `meshwell/structured/pipeline.py`: thread the per-cohort
  `Arrangement` from where it is already built into
  `build_cohort_compound`.

Out-of-scope:
- Unstructured `PolyPrism._make_occ_wire_from_vertices` and the GMSH
  per-`GeometryEntity` path — kept as today.
- Cohort↔unstructured TShape matching — already handled by BOP +
  AABB-rescue (`occ_xao_writer._candidate_pair_mask`); not regressed by
  this change.
- `_make_occ_wire_from_vertices(edge_registry=…)` overload on
  `GeometryEntity` ([geometry_entity.py:515](../../../meshwell/geometry_entity.py#L515))
  — left untouched. Its callers don't have an Arrangement in hand.

## Non-goals

- Eliminating BOP entirely on curved scenes.
- Sharing arcs ACROSS cohorts (each cohort builds its own arrangement
  and its own registries today; that stays).

## Architecture

### New dataclass — `ArrangementEdge`

`meshwell/structured/types.py`:

```python
@dataclass(frozen=True)
class ArrangementEdge:
    """Canonical curve between two arrangement nodes.

    ``vertex_keys`` are quantized to the cohort's point_tolerance and
    stored in a canonical direction (see `Arrangement` docstring).
    ``segments`` is the arc/line decomposition fit by
    ``decompose_vertices_2d`` on the canonical vertex sequence — fit
    ONCE per edge, then replayed by every sub-piece that touches it.
    """
    vertex_keys: tuple[VertexKey, ...]
    z: float
    segments: tuple[DecompositionSegment, ...]
    is_closed: bool  # True iff vertex_keys[0] == vertex_keys[-1]
```

### Extended `Arrangement`

```python
@dataclass(frozen=True)
class Arrangement:
    cohort_index: int
    polygons: tuple[Polygon, ...]
    canonical_edges: tuple[ArrangementEdge, ...] = ()
    edge_by_vertex_pair: dict[frozenset[VertexKey], int] = field(
        default_factory=dict
    )  # frozenset({vkey_a, vkey_b}) -> index into canonical_edges
```

Backward-compatible: existing tests that construct an `Arrangement`
with just `(cohort_index, polygons)` keep working; the canonical-edge
fields default to empty, and consumers fall back to per-ring fitting
when they see empty fields.

### Canonicalisation site — `build_cohort_arrangement`

`meshwell/structured/decompose.py:43`. After `merged = unary_union(linework)`
and `pieces = tuple(polygonize(merged))`:

1. **Incidence table.** Walk every component of `merged.geoms`, count
   how many `(component_idx, position)` references each
   `VertexRegistry._key(x, y, z)` has.
2. **Node detection.** A vertex is an arrangement node iff
   `incidence > 2`, OR it is an endpoint of an open component.
3. **Edge splitting.** For each component, split its vertex sequence
   at every node hit. Each run is one `ArrangementEdge`.
4. **Closed-edge case.** A closed component with NO node hits becomes
   one closed `ArrangementEdge`. Pick a deterministic seam via the
   existing `_find_canonical_seam`
   ([geometry_entity.py:51](../../../meshwell/geometry_entity.py#L51))
   and rotate before fitting.
5. **Direction canonicalisation.** Rotate the edge's vertex sequence
   to start at its lex-min vkey; if `keys[1] > keys[-1]`, reverse
   (matches the convention already in `FaceRegistry._canonical_ring`,
   [build.py:248](../../../meshwell/structured/build.py#L248)).
6. **Arc fitting.** Pass the canonical vertex sequence to
   `decompose_vertices_2d(coords, z=…, point_tolerance=…,
   identify_arcs=identify_arcs, …)`. Cache the resulting
   `tuple[DecompositionSegment, ...]` on the `ArrangementEdge`.
7. **Lookup table.** For every consecutive pair `(keys[i], keys[i+1])`
   in the canonical vertex sequence (including the wraparound pair
   `(keys[-1], keys[0])` when `is_closed=True`), register
   `frozenset({keys[i], keys[i+1]}) → edge_idx` in
   `edge_by_vertex_pair`. ASSERT each unordered pair is registered at
   most once across all canonical edges — if a second registration
   attempt collides, `unary_union + polygonize` produced parallel edges
   between the same two arrangement nodes, which violates the planar-
   arrangement assumption (raise `CanonicalArrangementError`).

`identify_arcs` for the cohort canonicaliser is the OR of all
`StructuredSlab.identify_arcs` flags inside the cohort. (Adjacent
unstructured `PolyPrism` entities contribute boundary LINES to the
linework but their `identify_arcs` flag governs only their own
`_make_occ_wire_from_vertices` call, which is not routed through the
arrangement in this design — see "Out-of-scope".) `min_arc_points` /
`arc_tolerance` use the strictest (largest `min_arc_points`, smallest
`arc_tolerance`) across the cohort's slabs so the canonical fit can
never violate any contributor's preference.

**Storage convention:** `ArrangementEdge.vertex_keys` is OPEN for both
open and closed edges — `vertex_keys[0] != vertex_keys[-1]` always.
`is_closed=True` records the topology; the implicit closing pair is
the registered `frozenset({vertex_keys[-1], vertex_keys[0]})`.

### Consumer — `EdgeRegistry.polyline_xy`

New optional parameter `arrangement: Arrangement | None = None`. When
non-None:

1. Quantize each input coord via `self.vertices._key(x, y, z)` to get
   `vkey` per ring vertex.
2. For each consecutive pair `(vk_i, vk_{i+1})`:
   a. Look up `edge_by_vertex_pair[frozenset({vk_i, vk_{i+1}})]` →
      `edge_idx`.
   b. Find the position of `vk_i` in `canonical_edges[edge_idx].vertex_keys`
      and decide direction: forward iff `vk_{i+1}` is the next vkey
      after `vk_i` in the canonical sequence; else reverse.
   c. Walk the canonical sequence in that direction until it returns
      to the ring's next non-spanning vertex — i.e. until the canonical
      edge is exhausted (its other endpoint reached). All ring vertices
      consumed in this walk correspond to interior vertices of the
      canonical edge.
   d. Replay the canonical `segments` covering the consumed positions:
      - line segment → `self.line_xy(...)` (existing cache).
      - arc segment forward → `self.arc_xy(start, mid, end, z)`.
      - arc segment reverse → `self.arc_xy(end, mid, start, z)`; the
        `arc_xy` key is direction-invariant on `(sorted endpoints, mid)`
        so the SAME TShape is returned ([build.py:134](../../../meshwell/structured/build.py#L134)).
3. Advance the ring index past the consumed run and continue.

When `arrangement=None`, the method's body is the current
implementation verbatim (no behaviour change for tests / call sites
that don't have an arrangement).

### Wiring

- `_build_horizontal_face` ([build.py:456](../../../meshwell/structured/build.py#L456))
  grows an `arrangement` parameter and forwards it to
  `ereg.polyline_xy`.
- `FaceRegistry.face_xy` ([build.py:280](../../../meshwell/structured/build.py#L280))
  grows an `arrangement` parameter; key remains
  `(z_q, exterior_canonical, interiors)` — no key change, because two
  sub-pieces sharing a polygon already collide on canonical key today
  and the same face is reused.
- `polyline_segments` ([build.py:328](../../../meshwell/structured/build.py#L328))
  for lateral faces takes the same `arrangement` and consults
  `canonical_edges[*].segments` instead of calling
  `decompose_vertices_2d` itself.
- `build_cohort_compound` ([build.py around line 706](../../../meshwell/structured/build.py#L706))
  accepts `arrangement` and forwards to both face builders.
- `structured_pre_pass` ([pipeline.py:104](../../../meshwell/structured/pipeline.py#L104))
  already builds the per-cohort `Arrangement` — pass it into
  `build_cohort_compound`.

## Algorithm details

### Closed arrangement edges

A closed arrangement edge has `vertex_keys[0] == vertex_keys[-1]` and
no other arrangement-node hits along its length. The lookup map stores
every consecutive pair around the loop, including the closing pair.
When a sub-piece's ring traverses the closed edge, the first vertex-pair
lookup pins a position in the closed sequence; subsequent pairs advance
position by one. The replay handles wraparound naturally.

### Full-circle arc replay

When the canonical `DecompositionSegment` set contains a full-circle
arc (start == end of the segment's `points`), it is split into two
half-arcs via the existing logic in
`EdgeRegistry.polyline_xy` ([build.py:184-206](../../../meshwell/structured/build.py#L184))
when the segments are FIT, not when replayed. So `ArrangementEdge.segments`
already stores TWO half-arcs for a full-circle edge; replay is the same
arc-emission loop with no special case.

### Invariant check (test-only by default)

`build_cohort_arrangement` accepts `validate_subpiece_coverage:
list[Polygon] | None = None`. When supplied:
- For every consecutive vertex pair of every sub-piece exterior /
  interior ring, assert `frozenset({vk_a, vk_b}) in edge_by_vertex_pair`.
- A miss raises `CanonicalArrangementError(coords=(p_a, p_b),
  cohort_index=…)`. New exception lives in
  [meshwell/structured/exceptions.py](../../../meshwell/structured/exceptions.py).

Tests always pass the sub-piece list. Production callers in
`structured_pre_pass` do NOT pass it (hot path stays lean), and rely on
test coverage to catch arrangement-coverage regressions.

### Failure modes

| Scenario | Behaviour |
|----------|-----------|
| Sub-piece introduces a vertex pair not in `edge_by_vertex_pair` (arrangement coverage bug) | Tests fail via `CanonicalArrangementError`. Production silently falls back to today's per-ring fitter (see "Fallback" below). |
| Quantisation collision: two distinct input coords snap to the same vkey | Already handled at the `VertexRegistry` level; arrangement inherits the snap. |
| `decompose_vertices_2d` fits no arcs (e.g., low-curvature polyline + `identify_arcs=False`) | `ArrangementEdge.segments` is a chain of line segments; replay is line-only. Same TShape sharing applies via `line_xy`. |

### Fallback in production

If `edge_by_vertex_pair` lookup misses in production (no
`validate_subpiece_coverage`), `polyline_xy` calls the existing per-ring
`decompose_vertices_2d` for the unmatched run. A `warnings.warn` records
the miss with the cohort index and the offending coords so the user can
add a regression scene. This keeps a bug from being a hard build
failure in the field while still surfacing it.

## Testing strategy

New file — `tests/structured/test_arrangement_canonical_edges.py`:

1. **Two overlapping rectangles.** 3 sub-pieces, 7 canonical edges,
   all interior degree-2 vertices NOT promoted to nodes. Assert
   `len(arr.canonical_edges) == 7`.
2. **Two overlapping discs with `identify_arcs=True`.** Shared lens
   contributes 2 arc edges (one per disc). Assert both discs' sub-piece
   faces SHARE the same `TopoDS_Edge` TShape (TShape identity via
   `BRep.TShape()`) on each lens boundary.
3. **Single disc inside a cohort.** One closed canonical edge; seam
   picked deterministically; input rotation does not change canonical
   form.
4. **CW vs CCW ring traversal.** Same edge consumed in both directions
   yields identical TShape via the direction-invariant `arc_xy` key.
5. **Mixed line+arc sub-piece.** A sub-piece whose ring spans 3
   canonical edges (e.g. line + arc + line) reproduces the canonical
   segmentation in the correct order.

Extended — `tests/structured/test_shared_edge_registry.py` and
`tests/structured/test_arc_unstructured_neighbour.py`:
- Add "two overlapping curved sub-pieces" scene; assert
  `len(ereg._store)` matches the canonical count, not 2×.
- Snapshot `FaceRegistry._store` size — expected to shrink relative to
  pre-change baseline on shared-boundary scenes.

Stress — `tests/structured/test_stress_complex_scene.py`:
- Re-run with the change; assert AABB-rescue count drops on the
  meander and disc-cohort scenes (the metric is already captured).

Invariant guard — every unit test calls `build_cohort_arrangement` with
`validate_subpiece_coverage=` the sub-piece polygons, so coverage
violations fail tests loudly.

## Migration / rollout

Single PR; no flag gate (failure surface is narrow and well-tested).
Backwards compatibility is preserved:
- `Arrangement` extension is additive.
- `EdgeRegistry.polyline_xy(arrangement=None)` default reproduces today's
  behaviour byte-for-byte.

Existing tests must remain green. New tests cover the canonicalisation
path explicitly. The "fallback in production" path provides safety
net: even if a downstream caller adds a new sub-piece source that the
arrangement misses, the build still completes.

## Risk register

- **Wrong canonical direction for closed edges.** Mitigation:
  `_find_canonical_seam` is already proven for the legacy path; the
  closed-edge case here reuses it.
- **Vertex-pair lookup ambiguity** (could two different canonical edges
  share an unordered vertex-pair?). Only possible if two distinct
  canonical edges share BOTH endpoints AND those endpoints are adjacent
  in BOTH edges — i.e. the two edges are parallel between the same two
  arrangement nodes. The cohort arrangement is planar; parallel edges
  between the same nodes are not produced by `unary_union + polygonize`.
  The assert in step 7 of "Canonicalisation site" enforces this and
  surfaces the violation immediately if it ever occurs.
- **Mid-point drift on arc replay.** Arcs use the canonical
  `DecompositionSegment.points[len/2]` as `mid`; replay uses the same.
  No drift risk.
- **Performance.** Per-cohort cost added: one extra
  `decompose_vertices_2d` per arrangement edge (replacing per-sub-piece
  fits). Edge count ≤ sub-piece-edge count, so this is a NET REDUCTION
  in fitter calls on shared-boundary scenes.

## Open questions

None at design time — every choice above was confirmed during the
brainstorm. Open implementation questions will be triaged in the
follow-on plan.
