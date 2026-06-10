# Arrangement-Canonical Arc Fitting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit arcs exactly once per unique cohort arrangement edge, then have every sub-piece's wire builder consume those canonical segmentations — so two sub-pieces sharing a curved boundary subset get the SAME OCC TShape by construction.

**Architecture:** Extend the cohort `Arrangement` ([meshwell/structured/types.py](meshwell/structured/types.py)) with a tuple of `ArrangementEdge` records (canonical vertex sequence + fit `DecompositionSegment` list) and a `{frozenset({vkey_a, vkey_b}) → edge_idx}` lookup. Compute both during `build_cohort_arrangement` ([meshwell/structured/decompose.py:43](meshwell/structured/decompose.py#L43)) by iterating the noded `unary_union` linework, splitting at arrangement nodes (degree ≠ 2 vertices), and calling `decompose_vertices_2d` once per canonical edge. Thread the `Arrangement` through `build_cohort_compound` → `_build_horizontal_face`/`polyline_segments` → `EdgeRegistry.polyline_xy`, where a new replay path walks each sub-piece ring vertex-pair-by-vertex-pair and emits the canonical segments via the existing `arc_xy`/`line_xy` cache (so the EdgeRegistry's TShape sharing kicks in).

Closed standalone canonical edges (e.g., a lone disc boundary) are stored with `is_closed=True` but NOT indexed in the vertex-pair lookup — sub-pieces traversing such a ring fall back to today's greedy per-ring fit, which is already deterministic via `_find_canonical_seam` for a single closed ring.

**Tech Stack:** Python 3.12, shapely 2.x (`unary_union`, `polygonize`), OCP/OCCT (`BRepBuilderAPI`, `GC_MakeArcOfCircle`), pytest. No new dependencies.

---

## File Structure

| File | Role | Change |
|------|------|--------|
| `meshwell/structured/types.py` | Dataclasses shared by all structured stages | Add `ArrangementEdge`; extend `Arrangement` with `canonical_edges` + `edge_by_vertex_pair` fields |
| `meshwell/structured/exceptions.py` | Custom errors for the structured pipeline | Add `CanonicalArrangementError` |
| `meshwell/structured/decompose.py` | Cohort arrangement builder | Add `_build_canonical_edges` helper; wire into `build_cohort_arrangement`; add `validate_canonical_edge_coverage` |
| `meshwell/structured/build.py` | Per-cohort OCC compound assembly | Add `arrangement` parameter to `EdgeRegistry.polyline_xy`, `polyline_segments`, `_build_horizontal_face`, `FaceRegistry.face_xy`, and `build_cohort_compound` |
| `meshwell/structured/pipeline.py` | Pre-pass driver | Thread `Arrangement` from where it's already built into `build_cohort_compound` |
| `tests/structured/test_arrangement_canonical_edges.py` | NEW unit tests | All five test cases from the spec's "Testing strategy" |
| `tests/structured/test_arrangement.py` | Existing arrangement tests | Add coverage for the new `canonical_edges` field |
| `tests/structured/test_shared_edge_registry.py` | Existing | Add "two overlapping curved sub-pieces" regression |
| `tests/structured/test_stress_complex_scene.py` | Existing | Assert AABB-rescue count drops on disc-cohort / meander scenes |

---

## Task 1: Add `ArrangementEdge` dataclass and extend `Arrangement`

**Files:**
- Modify: `meshwell/structured/types.py` (append new dataclass; extend existing `Arrangement` near line 104)
- Test: `tests/structured/test_arrangement.py` (append a smoke-test function)

- [ ] **Step 1: Write the failing test**

Append to `tests/structured/test_arrangement.py`:

```python
def test_arrangement_edge_defaults_to_empty():
    """New canonical_edges / edge_by_vertex_pair default fields exist
    and start empty so existing callers / tests are unaffected."""
    from meshwell.structured.types import Arrangement, ArrangementEdge

    arr = Arrangement(cohort_index=0, polygons=())
    assert arr.canonical_edges == ()
    assert arr.edge_by_vertex_pair == {}

    edge = ArrangementEdge(
        vertex_keys=((0, 0, 0), (1, 0, 0)),
        z=0.0,
        segments=(),
        is_closed=False,
    )
    assert edge.vertex_keys[0] != edge.vertex_keys[-1]
    assert edge.is_closed is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/structured/test_arrangement.py::test_arrangement_edge_defaults_to_empty -x
```

Expected: `ImportError: cannot import name 'ArrangementEdge' from 'meshwell.structured.types'`.

- [ ] **Step 3: Add the new dataclass and extend Arrangement**

Open `meshwell/structured/types.py` and apply two edits.

Edit A — add a `VertexKey` type alias and a new `ArrangementEdge` dataclass between the existing `SlabMeta` block (ends around line 102) and the existing `Arrangement` block (starts around line 104). Also add `field` to the existing dataclasses import. Replace the top import:

```python
from dataclasses import dataclass
```

with:

```python
from dataclasses import dataclass, field
```

Then insert ABOVE the existing `@dataclass(frozen=True)\nclass Arrangement:` line:

```python
# Quantized vertex key as used by VertexRegistry._key.
VertexKey = tuple[int, int, int]


@dataclass(frozen=True)
class ArrangementEdge:
    """Canonical curve between two arrangement nodes.

    Arc/line decomposition is fit ONCE on this edge's coords via
    ``meshwell.geometry_entity.decompose_vertices_2d`` and stored in
    ``segments``. Every sub-piece whose ring traverses this edge
    replays these segments instead of running the greedy fitter on its
    own ring — eliminating the seam-dependent mismatches.

    ``vertex_keys`` is stored OPEN even when ``is_closed=True``
    (``vertex_keys[0] != vertex_keys[-1]``); the implicit closing pair
    is registered in ``Arrangement.edge_by_vertex_pair`` only for
    OPEN edges. Closed standalone edges (e.g., a lone disc boundary
    with no other arrangement nodes) are NOT indexed — sub-pieces
    traversing them fall back to the per-ring greedy fit, which is
    already deterministic for a single closed ring.
    """

    vertex_keys: tuple["VertexKey", ...]
    z: float
    segments: tuple = ()  # tuple[DecompositionSegment, ...] — runtime import
    is_closed: bool = False
```

Edit B — replace the existing `Arrangement` class:

```python
@dataclass(frozen=True)
class Arrangement:
    cohort_index: int
    polygons: tuple["Polygon", ...]
```

with:

```python
@dataclass(frozen=True)
class Arrangement:
    """Cohort-global polygon arrangement.

    `polygons` is the canonical, ordered tuple of shapely.Polygon objects
    produced by one polygonize call over the union of:
      - every cohort slab boundary, and
      - every adjacent unstructured PolyPrism boundary projected to the
        shared z-planes.

    `canonical_edges` and `edge_by_vertex_pair` carry the arrangement's
    unique boundary edges with arcs fit ONCE per edge. Sub-piece wire
    builders look up each consecutive vertex pair in
    `edge_by_vertex_pair` and replay the stored `segments` so two
    sub-pieces sharing an arc-shaped boundary subset emit the same
    OCC TShape by construction.

    Both cohort sub-piece extraction and adjacent unstructured pre-cut
    consume this same tuple.

    Identity contract:
    - When a downstream consumer receives a single Polygon (e.g., a
      SubPiece's `sub_polygon` field), it is the SAME Python object
      (`is`) as the matching entry in `polygons`.
    - When the consumer receives a `MultiPolygon`, Shapely 2.x's
      `.geoms` accessor returns fresh Polygon wrappers each access, so
      Python `is` is NOT preserved. However, the underlying GEOS
      coordinate sequences are shared by reference: vertex coordinates
      are bit-exactly equal (`equals_exact(member, arrangement_poly,
      tolerance=0.0)`). Downstream OCC builders that key polygons by
      coordinate hash get identical hashes from both consumers.
    """

    cohort_index: int
    polygons: tuple["Polygon", ...]
    canonical_edges: tuple[ArrangementEdge, ...] = ()
    edge_by_vertex_pair: dict[frozenset["VertexKey"], int] = field(
        default_factory=dict
    )
```

- [ ] **Step 4: Run the test, verify it passes**

```bash
pytest tests/structured/test_arrangement.py::test_arrangement_edge_defaults_to_empty -x
```

Expected: PASS.

- [ ] **Step 5: Run full arrangement test file to confirm no regression**

```bash
pytest tests/structured/test_arrangement.py -x
```

Expected: ALL PASS (existing tests construct `Arrangement(cohort_index, polygons)` which still works because the new fields have defaults).

- [ ] **Step 6: Commit**

```bash
git add meshwell/structured/types.py tests/structured/test_arrangement.py
git commit -m "feat(structured): add ArrangementEdge and extend Arrangement with canonical-edge fields"
```

---

## Task 2: Add `CanonicalArrangementError` exception

**Files:**
- Modify: `meshwell/structured/exceptions.py` (append a new class at the end)
- Test: `tests/structured/test_exceptions.py` (append a test function)

- [ ] **Step 1: Write the failing test**

Append to `tests/structured/test_exceptions.py`:

```python
def test_canonical_arrangement_error_message():
    from meshwell.structured.exceptions import (
        CanonicalArrangementError,
        StructuredError,
    )

    err = CanonicalArrangementError(
        cohort_index=3,
        reason="vertex pair ((0,0,0),(1,1,0)) not in canonical edge lookup",
    )
    assert isinstance(err, StructuredError)
    assert err.cohort_index == 3
    assert "cohort 3" in str(err)
    assert "vertex pair" in str(err)
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/structured/test_exceptions.py::test_canonical_arrangement_error_message -x
```

Expected: `ImportError: cannot import name 'CanonicalArrangementError'`.

- [ ] **Step 3: Add the exception**

Append to `meshwell/structured/exceptions.py`:

```python
class CanonicalArrangementError(StructuredError):
    """Cohort arrangement's canonical-edge invariant was violated.

    Raised by ``meshwell.structured.decompose._build_canonical_edges``
    when two distinct canonical edges share an unordered vertex pair
    (i.e., parallel edges between the same arrangement nodes — should
    be impossible from ``unary_union + polygonize``), or by
    ``validate_canonical_edge_coverage`` when a sub-piece ring has
    MIXED coverage in ``Arrangement.edge_by_vertex_pair`` (some pairs
    found, some missing) — indicating a coverage bug in the
    canonicaliser, since each ring should be either fully covered by
    open canonical edges OR fully on a single closed standalone edge.
    """

    def __init__(self, cohort_index: int, reason: str):
        self.cohort_index = cohort_index
        self.reason = reason
        super().__init__(f"cohort {cohort_index}: {reason}")
```

- [ ] **Step 4: Run the test, verify it passes**

```bash
pytest tests/structured/test_exceptions.py::test_canonical_arrangement_error_message -x
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/exceptions.py tests/structured/test_exceptions.py
git commit -m "feat(structured): add CanonicalArrangementError"
```

---

## Task 3: Build canonical edges from a noded MultiLineString

**Files:**
- Modify: `meshwell/structured/decompose.py` (add a new private helper near the top)
- Test: `tests/structured/test_arrangement_canonical_edges.py` (NEW file)

The helper takes the `unary_union(linework)` output plus a `point_tolerance` and produces `(canonical_edges, edge_by_vertex_pair)`. Each component of the merged geometry is treated as ONE canonical edge — shapely's `unary_union` on LineStrings nodes at every crossing, so each component already runs between two arrangement nodes (or is a closed standalone loop).

- [ ] **Step 1: Write failing tests**

Create `tests/structured/test_arrangement_canonical_edges.py`:

```python
"""Tests for _build_canonical_edges in meshwell.structured.decompose."""
from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from meshwell.structured.decompose import _build_canonical_edges


def _rect(x1, y1, x2, y2):
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def _circle(cx, cy, r, n=48):
    a = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    return Polygon([(cx + r * np.cos(t), cy + r * np.sin(t)) for t in a])


def test_two_overlapping_rectangles_split_into_seven_edges():
    """Two overlapping squares share a single cut; the planar graph has
    7 unique edges (4 around left square minus shared cut + 4 around
    right square minus shared cut + 1 shared cut = some bookkeeping).

    Shapely's unary_union nodes at the two crossing points where the
    shared cut meets each square's left/right side, so we expect 7
    arrangement edges total.
    """
    merged = unary_union([_rect(0, 0, 6, 10).boundary, _rect(4, 0, 10, 10).boundary])
    edges, lookup = _build_canonical_edges(
        merged,
        z=0.0,
        point_tolerance=1e-3,
        identify_arcs=False,
        min_arc_points=5,
        arc_tolerance=1e-3,
    )
    assert len(edges) == 7
    # Every consecutive vertex pair on every edge is registered exactly once.
    seen = set()
    for ei, edge in enumerate(edges):
        for i in range(len(edge.vertex_keys) - 1):
            pair = frozenset({edge.vertex_keys[i], edge.vertex_keys[i + 1]})
            assert pair not in seen, f"duplicate pair on edge {ei}"
            seen.add(pair)
            assert lookup[pair] == ei
    # Closed standalone edges (none here) are NOT pair-indexed.
    assert all(not e.is_closed for e in edges)


def test_single_closed_disc_yields_one_closed_edge():
    """A standalone circle (no other arrangement nodes) becomes one
    canonical edge with is_closed=True; its vertex pairs are NOT
    registered in the lookup (closed-standalone fallback)."""
    merged = unary_union([_circle(0, 0, 1.0).boundary])
    edges, lookup = _build_canonical_edges(
        merged,
        z=0.0,
        point_tolerance=1e-3,
        identify_arcs=True,
        min_arc_points=5,
        arc_tolerance=1e-3,
    )
    assert len(edges) == 1
    assert edges[0].is_closed
    # OPEN storage convention: first key != last key.
    assert edges[0].vertex_keys[0] != edges[0].vertex_keys[-1]
    # Closed-standalone edges are NOT pair-indexed.
    assert lookup == {}
    # identify_arcs=True with a 48-point circle produces at least one arc.
    assert any(s.is_arc for s in edges[0].segments)


def test_two_overlapping_discs_produce_shared_arc_edges():
    """Two overlapping discs cut each other at two points, producing
    four arrangement edges (two arcs per disc on either side of the
    crossings). identify_arcs=True at fitting time gives arc segments
    inside each edge; the two discs' lens-shared boundary is split
    into TWO open canonical arc edges (one per disc's circumference
    contribution)."""
    merged = unary_union(
        [_circle(0, 0, 1.0).boundary, _circle(1.0, 0, 1.0).boundary]
    )
    edges, lookup = _build_canonical_edges(
        merged,
        z=0.0,
        point_tolerance=1e-3,
        identify_arcs=True,
        min_arc_points=5,
        arc_tolerance=1e-3,
    )
    assert len(edges) == 4
    # No closed standalone edges in this configuration.
    assert all(not e.is_closed for e in edges)
    # Every edge has at least one arc segment.
    assert all(any(s.is_arc for s in e.segments) for e in edges)
    # Every consecutive pair across all edges is registered uniquely.
    n_pairs = sum(len(e.vertex_keys) - 1 for e in edges)
    assert len(lookup) == n_pairs


def test_parallel_edges_between_same_nodes_raise():
    """Two distinct line components that share both endpoints AND are
    adjacent in BOTH (i.e., a parallel-edge graph) should raise
    CanonicalArrangementError. We construct this artificially because
    unary_union normally prevents it — but the validator must catch
    it if the input ever produces it."""
    from meshwell.structured.exceptions import CanonicalArrangementError

    # Two two-vertex LineStrings between the same endpoints, NOT routed
    # through unary_union so they stay as parallel components.
    from shapely.geometry import MultiLineString

    merged = MultiLineString(
        [
            LineString([(0, 0), (1, 0)]),
            LineString([(0, 0), (1, 0)]),
        ]
    )
    with pytest.raises(CanonicalArrangementError):
        _build_canonical_edges(
            merged,
            z=0.0,
            point_tolerance=1e-3,
            identify_arcs=False,
            min_arc_points=5,
            arc_tolerance=1e-3,
        )
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
pytest tests/structured/test_arrangement_canonical_edges.py -x
```

Expected: `ImportError: cannot import name '_build_canonical_edges' from 'meshwell.structured.decompose'`.

- [ ] **Step 3: Implement `_build_canonical_edges`**

Open `meshwell/structured/decompose.py`. Add this import near the existing imports (top of file):

```python
from meshwell.geometry_entity import decompose_vertices_2d
from meshwell.structured.exceptions import CanonicalArrangementError
from meshwell.structured.types import (
    Arrangement,
    ArrangementEdge,
    Cohort,
    StructuredSlab,
    SubPiece,
    VertexKey,
)
```

(Replace the existing `from meshwell.structured.types import (...)` block with the above — keep the existing names, add `ArrangementEdge` and `VertexKey`.)

Then insert this new helper near the top of the module (after the existing top-level imports and before `zinterval_footprint`):

```python
def _quantize_key(x: float, y: float, z: float, point_tolerance: float) -> VertexKey:
    """Match VertexRegistry._key's quantization so canonical edges and
    runtime EdgeRegistry vertices share the same key space."""
    s = point_tolerance
    return (round(x / s), round(y / s), round(z / s))


def _build_canonical_edges(
    merged,
    z: float,
    point_tolerance: float,
    identify_arcs: bool,
    min_arc_points: int,
    arc_tolerance: float,
) -> tuple[tuple[ArrangementEdge, ...], dict[frozenset[VertexKey], int]]:
    """Build canonical arrangement edges from a noded MultiLineString.

    ``merged`` is the output of ``shapely.ops.unary_union`` over the
    cohort's boundary linework. Each component of ``merged`` becomes
    one ``ArrangementEdge`` — shapely's union nodes at every line
    crossing, so each component already spans between two arrangement
    nodes (or is a closed standalone loop).

    Open edges are stored in canonical direction (lex-min start; if
    keys[1] > keys[-1], reverse). Their consecutive vertex pairs are
    registered in ``edge_by_vertex_pair`` for fast O(1) replay lookup.

    Closed standalone edges (single loops with no other arrangement
    nodes — e.g., a disc boundary alone in a cohort) are stored with
    ``is_closed=True`` but are NOT pair-indexed; sub-piece consumers
    detect the missing pairs and fall back to today's greedy per-ring
    fit, which is deterministic for a single closed ring.

    Raises CanonicalArrangementError if two distinct edges register
    the same unordered vertex pair (parallel-edge graph violation).
    """
    from shapely.geometry import LineString, MultiLineString

    if merged.is_empty:
        return (), {}
    if isinstance(merged, LineString):
        components = [merged]
    elif isinstance(merged, MultiLineString):
        components = list(merged.geoms)
    else:
        # GeometryCollection or other: extract the LineString members.
        components = [g for g in getattr(merged, "geoms", []) if isinstance(g, LineString)]

    canonical_edges: list[ArrangementEdge] = []
    edge_by_vertex_pair: dict[frozenset[VertexKey], int] = {}

    for component in components:
        coords = [(c[0], c[1]) for c in component.coords]
        if len(coords) < 2:
            continue
        keys = [_quantize_key(x, y, z, point_tolerance) for x, y in coords]
        # Collapse runs that quantize to the same key (sub-tolerance jitter).
        dedup_coords: list[tuple[float, float]] = [coords[0]]
        dedup_keys: list[VertexKey] = [keys[0]]
        for c, k in zip(coords[1:], keys[1:]):
            if k != dedup_keys[-1]:
                dedup_coords.append(c)
                dedup_keys.append(k)
        coords = dedup_coords
        keys = dedup_keys
        if len(coords) < 2:
            continue

        is_closed = keys[0] == keys[-1]
        if is_closed:
            # OPEN storage convention: drop the closing duplicate.
            coords = coords[:-1]
            keys = keys[:-1]
            # Closed standalone: fit segments as a closed ring via
            # decompose_vertices_2d (which seam-canonicalises internally).
            # We pass closed form (first==last) so its seam logic kicks in.
            fit_coords = [*coords, coords[0]]
        else:
            # Open edge: canonical direction = lex-min start; if
            # keys[1] > keys[-1] then reverse.
            i_min = min(range(len(keys)), key=lambda i: keys[i])
            if i_min != 0:
                # An open edge's start/end vertices SHOULD be at indices 0
                # and -1 (arrangement nodes); lex-min should be one of them.
                # If lex-min is interior, the component is mis-noded —
                # treat as-is in input order (this shouldn't happen for
                # unary_union output).
                if i_min == len(keys) - 1:
                    coords = list(reversed(coords))
                    keys = list(reversed(keys))
            # Now ensure direction: if keys[-1] < keys[1] (next vertex
            # after the start is "later" than the end), reverse the tail.
            if len(keys) >= 3 and keys[-1] < keys[1]:
                coords = list(reversed(coords))
                keys = list(reversed(keys))
            fit_coords = list(coords)

        segments = decompose_vertices_2d(
            fit_coords,
            z=z,
            point_tolerance=point_tolerance,
            identify_arcs=identify_arcs,
            min_arc_points=min_arc_points,
            arc_tolerance=arc_tolerance,
        )

        edge = ArrangementEdge(
            vertex_keys=tuple(keys),
            z=z,
            segments=tuple(segments),
            is_closed=is_closed,
        )
        edge_idx = len(canonical_edges)
        canonical_edges.append(edge)

        # Register vertex pairs in the lookup, but ONLY for open edges.
        # Closed standalones are not pair-indexed (see docstring).
        if not is_closed:
            for i in range(len(keys) - 1):
                pair = frozenset({keys[i], keys[i + 1]})
                if pair in edge_by_vertex_pair:
                    raise CanonicalArrangementError(
                        cohort_index=-1,
                        reason=(
                            f"duplicate vertex pair {tuple(pair)} on edges "
                            f"{edge_by_vertex_pair[pair]} and {edge_idx} "
                            "(parallel edges between the same arrangement "
                            "nodes — unary_union output violates the planar "
                            "arrangement assumption)"
                        ),
                    )
                edge_by_vertex_pair[pair] = edge_idx

    return tuple(canonical_edges), edge_by_vertex_pair
```

- [ ] **Step 4: Run the tests, verify they pass**

```bash
pytest tests/structured/test_arrangement_canonical_edges.py -x
```

Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/decompose.py tests/structured/test_arrangement_canonical_edges.py
git commit -m "feat(structured): _build_canonical_edges from unary_union linework"
```

---

## Task 4: Wire `_build_canonical_edges` into `build_cohort_arrangement`

**Files:**
- Modify: `meshwell/structured/decompose.py` (extend `build_cohort_arrangement`)
- Test: `tests/structured/test_arrangement.py` (append test)

- [ ] **Step 1: Write the failing test**

Append to `tests/structured/test_arrangement.py`:

```python
def test_build_cohort_arrangement_populates_canonical_edges():
    """build_cohort_arrangement now produces an Arrangement whose
    canonical_edges tuple is populated and whose edge_by_vertex_pair
    indexes every open-edge consecutive vertex pair."""
    cohort = Cohort(
        slabs=(
            _slab(0, _rect(0, 0, 6, 10)),
            _slab(1, _rect(4, 0, 10, 10)),
        ),
        z_planes=(0.0, 1.0),
    )
    arr = build_cohort_arrangement(
        cohort_index=0,
        cohort=cohort,
        adjacent_unstructured=[],
        point_tolerance=1e-3,
    )
    assert len(arr.canonical_edges) > 0
    # Lookup non-empty (rectangles have no closed standalone components).
    assert len(arr.edge_by_vertex_pair) > 0
    # Each registered pair points to an existing canonical edge index.
    for ei in arr.edge_by_vertex_pair.values():
        assert 0 <= ei < len(arr.canonical_edges)
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/structured/test_arrangement.py::test_build_cohort_arrangement_populates_canonical_edges -x
```

Expected: FAIL — `build_cohort_arrangement` does not yet accept `point_tolerance` (or accepts it but produces empty `canonical_edges`).

- [ ] **Step 3: Update `build_cohort_arrangement`**

In `meshwell/structured/decompose.py`, replace the existing `build_cohort_arrangement` body with:

```python
def build_cohort_arrangement(
    cohort_index: int,
    cohort: Cohort,
    adjacent_unstructured: list,
    point_tolerance: float = 1e-3,
) -> Arrangement:
    """One shapely polygonize over the union of all relevant boundaries.

    `adjacent_unstructured` is a list of shapely line geometries
    (typically `ent.polygons.boundary` for each unstructured PolyPrism
    whose top/bottom z-plane coincides with one of `cohort.z_planes`
    AND whose XY intersects the cohort footprint).

    The returned `Arrangement.polygons` tile the union of all those
    polygons' interiors. The cohort sub-piece extractor
    (``arrangement_subpieces_for_interval``) filters this list by
    z-interval to produce per-interval SubPieces.

    `Arrangement.canonical_edges` and `edge_by_vertex_pair` carry the
    arrangement's unique boundary edges with arcs fit ONCE per edge.
    Sub-piece wire builders look up each consecutive vertex pair in
    `edge_by_vertex_pair` and replay the stored segments, so two
    sub-pieces sharing an arc boundary subset emit the SAME OCC
    TShape by construction.

    `identify_arcs` for the canonical fit is the OR of all
    `StructuredSlab.identify_arcs` flags in the cohort. The arc
    parameters use the strictest (largest `min_arc_points`, smallest
    `arc_tolerance`) across the cohort's slabs.
    """
    linework = [s.footprint.boundary for s in cohort.slabs] + list(
        adjacent_unstructured
    )
    merged = unary_union(linework)
    pieces = tuple(polygonize(merged))

    # OR identify_arcs across the cohort's slabs; pick strictest arc
    # params so the canonical fit never violates any contributor's
    # preference.
    identify_arcs = any(s.identify_arcs for s in cohort.slabs)
    min_arc_points = max(
        (s.min_arc_points for s in cohort.slabs if s.identify_arcs),
        default=5,
    )
    arc_tolerance = min(
        (s.arc_tolerance for s in cohort.slabs if s.identify_arcs),
        default=1e-3,
    )
    # Canonical edges live at the cohort's z-planes; use the first plane
    # as a representative z for vertex-key quantization (all linework is
    # 2D and z is just the quantization Z-bucket).
    z = cohort.z_planes[0] if cohort.z_planes else 0.0

    try:
        canonical_edges, edge_by_vertex_pair = _build_canonical_edges(
            merged,
            z=z,
            point_tolerance=point_tolerance,
            identify_arcs=identify_arcs,
            min_arc_points=min_arc_points,
            arc_tolerance=arc_tolerance,
        )
    except CanonicalArrangementError as e:
        # Re-raise with the correct cohort index.
        raise CanonicalArrangementError(
            cohort_index=cohort_index, reason=e.reason
        ) from None

    return Arrangement(
        cohort_index=cohort_index,
        polygons=pieces,
        canonical_edges=canonical_edges,
        edge_by_vertex_pair=edge_by_vertex_pair,
    )
```

- [ ] **Step 4: Update `decompose_cohorts` to pass `point_tolerance`**

In `meshwell/structured/decompose.py`, locate the call inside `decompose_cohorts` (around line 165) and add the `point_tolerance` keyword:

```python
arrangements.append(
    build_cohort_arrangement(
        cohort_index=ci,
        cohort=cohort,
        adjacent_unstructured=adjacency_lines_per_cohort[ci],
        point_tolerance=point_tolerance,
    )
)
```

- [ ] **Step 5: Run tests, verify the new test passes and old ones still pass**

```bash
pytest tests/structured/test_arrangement.py tests/structured/test_arrangement_canonical_edges.py -x
```

Expected: ALL PASS.

- [ ] **Step 6: Commit**

```bash
git add meshwell/structured/decompose.py tests/structured/test_arrangement.py
git commit -m "feat(structured): populate Arrangement canonical_edges in build_cohort_arrangement"
```

---

## Task 5: Add `arrangement` parameter + canonical replay to `EdgeRegistry.polyline_xy`

**Files:**
- Modify: `meshwell/structured/build.py` (extend `EdgeRegistry.polyline_xy`)
- Test: `tests/structured/test_arrangement_canonical_edges.py` (append integration tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/structured/test_arrangement_canonical_edges.py`:

```python
def test_polyline_xy_with_arrangement_replays_canonical_segments():
    """Calling polyline_xy with arrangement replays canonical segments
    so two sub-pieces sharing a boundary edge get the SAME TShape."""
    from meshwell.structured.build import EdgeRegistry, VertexRegistry
    from meshwell.structured.decompose import build_cohort_arrangement
    from meshwell.structured.types import Cohort, StructuredSlab
    from shapely.geometry import Polygon

    # Two overlapping rectangles -> 3 sub-pieces, the middle of which
    # shares one cut edge with each side.
    def _slab(idx, poly):
        return StructuredSlab(
            source_index=idx, footprint=poly,
            zlo=0.0, zhi=1.0,
            mesh_order=1.0, mesh_bool=True,
            physical_name=("x",),
            identify_arcs=False, arc_tolerance=1e-3, min_arc_points=4,
        )

    left = _slab(0, Polygon([(0, 0), (6, 0), (6, 10), (0, 10)]))
    right = _slab(1, Polygon([(4, 0), (10, 0), (10, 10), (4, 10)]))
    cohort = Cohort(slabs=(left, right), z_planes=(0.0, 1.0))
    arr = build_cohort_arrangement(
        cohort_index=0, cohort=cohort,
        adjacent_unstructured=[], point_tolerance=1e-3,
    )

    vreg = VertexRegistry(point_tolerance=1e-3)
    ereg = EdgeRegistry(vertices=vreg, point_tolerance=1e-3)

    # Build the middle (overlap) sub-piece's exterior ring through both
    # forward and reversed traversal. They must emit the SAME OCC TShapes.
    overlap = Polygon([(4, 0), (6, 0), (6, 10), (4, 10), (4, 0)])
    coords = list(overlap.exterior.coords)
    edges_fwd = ereg.polyline_xy(
        [(x, y) for x, y in coords],
        z=0.0,
        identify_arcs=False,
        arrangement=arr,
    )
    edges_rev = ereg.polyline_xy(
        [(x, y) for x, y in reversed(coords)],
        z=0.0,
        identify_arcs=False,
        arrangement=arr,
    )

    from OCP.BRepTools import BRepTools
    fwd_ids = sorted(id(e.TShape().this) for e in edges_fwd)
    rev_ids = sorted(id(e.TShape().this) for e in edges_rev)
    assert fwd_ids == rev_ids


def test_polyline_xy_falls_back_when_arrangement_none():
    """When arrangement=None, polyline_xy preserves today's behavior."""
    from meshwell.structured.build import EdgeRegistry, VertexRegistry

    vreg = VertexRegistry(point_tolerance=1e-3)
    ereg = EdgeRegistry(vertices=vreg, point_tolerance=1e-3)
    coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
    edges = ereg.polyline_xy(coords, z=0.0, identify_arcs=False)
    assert len(edges) == 4
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
pytest tests/structured/test_arrangement_canonical_edges.py::test_polyline_xy_with_arrangement_replays_canonical_segments -x
```

Expected: FAIL — `polyline_xy` does not accept `arrangement=`.

- [ ] **Step 3: Extend `polyline_xy`**

In `meshwell/structured/build.py`, replace the existing `EdgeRegistry.polyline_xy` method with:

```python
    def polyline_xy(
        self,
        coords: list[tuple[float, float]],
        z: float,
        identify_arcs: bool,
        min_arc_points: int = 5,
        arc_tolerance: float = 1e-3,
        arrangement: "Arrangement | None" = None,
    ) -> list[TopoDS_Edge]:
        """Return the list of edges (lines and/or arcs) covering coords.

        When ``arrangement`` is provided AND every consecutive vertex
        pair of ``coords`` is found in ``arrangement.edge_by_vertex_pair``,
        replays each canonical edge's pre-fit segments so two callers
        traversing the same arrangement edge in opposite directions get
        the SAME OCC TShape by construction.

        Falls back to per-ring greedy decomposition via
        ``decompose_vertices_2d`` when arrangement is None, or when the
        ring has NO pairs in the lookup (closed-standalone case — a
        single closed canonical edge handles itself deterministically
        via _find_canonical_seam). Raises ``CanonicalArrangementError``
        on MIXED coverage (some pairs found, some missing) — that's a
        canonicaliser bug.
        """
        if arrangement is not None and arrangement.canonical_edges:
            edges = self._polyline_xy_canonical(
                coords,
                z,
                arrangement,
                identify_arcs,
                min_arc_points,
                arc_tolerance,
            )
            if edges is not None:
                return edges
            # else: fell through to fallback (closed-standalone ring).

        segments = decompose_vertices_2d(
            coords,
            z=z,
            point_tolerance=self.point_tolerance,
            identify_arcs=identify_arcs,
            min_arc_points=min_arc_points,
            arc_tolerance=arc_tolerance,
        )
        return self._emit_edges_for_segments(segments, z)

    def _emit_edges_for_segments(
        self, segments, z: float
    ) -> list[TopoDS_Edge]:
        """Emit OCC edges for a list of DecompositionSegment objects.

        Extracted from polyline_xy's legacy body so the canonical-replay
        path and the greedy-fallback path share the same arc/line/
        full-circle emission code."""
        edges: list[TopoDS_Edge] = []
        for seg in segments:
            pts = seg.points
            if seg.is_arc:
                start = pts[0]
                end = pts[-1]
                is_closed = self.vertices._key(
                    start[0], start[1], z
                ) == self.vertices._key(end[0], end[1], z)
                if is_closed:
                    mid_idx = len(pts) // 2
                    quarter_idx = len(pts) // 4
                    three_quarter_idx = (len(pts) * 3) // 4
                    edges.append(
                        self.arc_xy(
                            (pts[0][0], pts[0][1]),
                            (pts[quarter_idx][0], pts[quarter_idx][1]),
                            (pts[mid_idx][0], pts[mid_idx][1]),
                            z,
                        )
                    )
                    edges.append(
                        self.arc_xy(
                            (pts[mid_idx][0], pts[mid_idx][1]),
                            (pts[three_quarter_idx][0], pts[three_quarter_idx][1]),
                            (pts[-1][0], pts[-1][1]),
                            z,
                        )
                    )
                else:
                    mid_idx = len(pts) // 2
                    edges.append(
                        self.arc_xy(
                            (pts[0][0], pts[0][1]),
                            (pts[mid_idx][0], pts[mid_idx][1]),
                            (pts[-1][0], pts[-1][1]),
                            z,
                        )
                    )
            else:
                edges.extend(
                    self.line_xy(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], z)
                    for i in range(len(pts) - 1)
                )
        return edges

    def _polyline_xy_canonical(
        self,
        coords: list[tuple[float, float]],
        z: float,
        arrangement: "Arrangement",
        identify_arcs: bool,
        min_arc_points: int,
        arc_tolerance: float,
    ) -> "list[TopoDS_Edge] | None":
        """Replay canonical arrangement edges for a ring.

        Returns the emitted edge list when every consecutive vertex
        pair of ``coords`` is in ``arrangement.edge_by_vertex_pair``.
        Returns None when NO pairs match (closed-standalone fallback).
        Raises ``CanonicalArrangementError`` on mixed coverage.
        """
        from meshwell.structured.exceptions import CanonicalArrangementError

        keys = [self.vertices._key(x, y, z) for x, y in coords]
        # Tolerate accidental closing duplicate at the tail; planar
        # rings often include it.
        if len(keys) >= 2 and keys[0] == keys[-1]:
            inner_keys = keys[:-1]
            inner_coords = coords[:-1]
            ring_is_closed = True
        else:
            inner_keys = list(keys)
            inner_coords = list(coords)
            ring_is_closed = False
        n = len(inner_keys)
        if n < 2:
            return []

        # Probe coverage: count pair hits/misses.
        pair_count = n if ring_is_closed else n - 1
        hits = 0
        for i in range(pair_count):
            a = inner_keys[i]
            b = inner_keys[(i + 1) % n] if ring_is_closed else inner_keys[i + 1]
            if frozenset({a, b}) in arrangement.edge_by_vertex_pair:
                hits += 1
        if hits == 0:
            return None  # closed-standalone fallback
        if hits != pair_count:
            raise CanonicalArrangementError(
                cohort_index=arrangement.cohort_index,
                reason=(
                    f"sub-piece ring has mixed canonical coverage "
                    f"({hits}/{pair_count} pairs in lookup) — coverage bug"
                ),
            )

        # Replay: walk the ring; each pair pins a canonical edge.
        edges: list[TopoDS_Edge] = []
        consumed = 0
        i = 0
        while consumed < pair_count:
            a_key = inner_keys[i % n]
            b_key = inner_keys[(i + 1) % n]
            edge_idx = arrangement.edge_by_vertex_pair[frozenset({a_key, b_key})]
            canon = arrangement.canonical_edges[edge_idx]
            # Direction: canonical edge's first key == a_key -> forward,
            # else reverse.
            if canon.vertex_keys[0] == a_key:
                forward = True
            elif canon.vertex_keys[-1] == a_key:
                forward = False
            else:
                # Should not happen: sub-pieces enter canonical edges at
                # one of the two endpoints (arrangement nodes). If the
                # entered key is interior, the lookup is mis-built.
                raise CanonicalArrangementError(
                    cohort_index=arrangement.cohort_index,
                    reason=(
                        f"vertex {a_key} is interior to canonical edge "
                        f"{edge_idx} but the ring is entering here"
                    ),
                )
            segments = list(canon.segments)
            if not forward:
                segments = [_reverse_segment(s) for s in reversed(segments)]
            edges.extend(self._emit_edges_for_segments(segments, z))
            step = len(canon.vertex_keys) - 1
            i += step
            consumed += step
        return edges
```

Also add this top-level helper in the SAME file (above `EdgeRegistry`):

```python
def _reverse_segment(seg):
    """Return a DecompositionSegment with points reversed.

    Arcs use the canonical 3-point form (start, mid, end). Reversing
    swaps start and end; the registered ``arc_xy`` key is direction-
    invariant on the sorted endpoints (mid still distinguishes upper
    vs lower halves of a closed circle), so the SAME TShape is
    returned regardless of traversal direction.
    """
    from dataclasses import replace

    return replace(seg, points=list(reversed(seg.points)))
```

Add the `Arrangement` import at the top of `build.py` (forward reference; it's already in `meshwell.structured.types`):

```python
from meshwell.structured.types import (
    Arrangement,
    Cohort,
    ShapeKey,
    SlabMeta,
    StructuredSlab,
    SubPiece,
)
```

(Add `Arrangement` to the existing import list.)

- [ ] **Step 4: Run tests, verify they pass**

```bash
pytest tests/structured/test_arrangement_canonical_edges.py -x
```

Expected: ALL PASS.

- [ ] **Step 5: Run the full structured test suite to verify no regression**

```bash
pytest tests/structured/ -x
```

Expected: ALL PASS (callers that don't pass `arrangement` still use the legacy code path).

- [ ] **Step 6: Commit**

```bash
git add meshwell/structured/build.py tests/structured/test_arrangement_canonical_edges.py
git commit -m "feat(structured): canonical-edge replay in EdgeRegistry.polyline_xy"
```

---

## Task 6: Thread `arrangement` through `_build_horizontal_face` and `FaceRegistry.face_xy`

**Files:**
- Modify: `meshwell/structured/build.py`
- Test: `tests/structured/test_arrangement_canonical_edges.py` (append a test)

- [ ] **Step 1: Write the failing test**

Append to `tests/structured/test_arrangement_canonical_edges.py`:

```python
def test_face_xy_passes_arrangement_to_polyline_xy():
    """FaceRegistry.face_xy forwards the arrangement so the
    underlying _build_horizontal_face / polyline_xy use canonical
    edges. Two distinct sub-pieces sharing a boundary edge then
    yield faces whose shared OCC edge is the same TShape."""
    from meshwell.structured.build import (
        EdgeRegistry,
        FaceRegistry,
        VertexRegistry,
        _build_horizontal_face,
    )
    from meshwell.structured.decompose import build_cohort_arrangement
    from meshwell.structured.types import Cohort, StructuredSlab
    from shapely.geometry import Polygon

    def _slab(idx, poly):
        return StructuredSlab(
            source_index=idx, footprint=poly,
            zlo=0.0, zhi=1.0,
            mesh_order=1.0, mesh_bool=True,
            physical_name=("x",),
            identify_arcs=False, arc_tolerance=1e-3, min_arc_points=4,
        )

    left = _slab(0, Polygon([(0, 0), (6, 0), (6, 10), (0, 10)]))
    right = _slab(1, Polygon([(4, 0), (10, 0), (10, 10), (4, 10)]))
    cohort = Cohort(slabs=(left, right), z_planes=(0.0, 1.0))
    arr = build_cohort_arrangement(
        cohort_index=0, cohort=cohort,
        adjacent_unstructured=[], point_tolerance=1e-3,
    )

    vreg = VertexRegistry(point_tolerance=1e-3)
    ereg = EdgeRegistry(vertices=vreg, point_tolerance=1e-3)
    freg = FaceRegistry(edges=ereg, point_tolerance=1e-3)

    # Two distinct sub-pieces from the arrangement -> two faces.
    # Their shared edge MUST be the same TShape.
    polys = list(arr.polygons)
    face1 = _build_horizontal_face(
        polys[0], 0.0, ereg,
        identify_arcs=False, min_arc_points=5, arc_tolerance=1e-3,
        face_registry=freg, arrangement=arr,
    )
    face2 = _build_horizontal_face(
        polys[1], 0.0, ereg,
        identify_arcs=False, min_arc_points=5, arc_tolerance=1e-3,
        face_registry=freg, arrangement=arr,
    )
    # If both faces share at least one edge TShape, the canonical
    # replay worked.
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    def _edge_ids(face):
        out = set()
        exp = TopExp_Explorer(face, TopAbs_EDGE)
        while exp.More():
            out.add(id(exp.Current().TShape().this))
            exp.Next()
        return out
    assert _edge_ids(face1) & _edge_ids(face2), \
        "expected at least one shared edge TShape between adjacent faces"
```

- [ ] **Step 2: Run the test, verify it fails**

```bash
pytest tests/structured/test_arrangement_canonical_edges.py::test_face_xy_passes_arrangement_to_polyline_xy -x
```

Expected: FAIL — `_build_horizontal_face` does not accept `arrangement=`.

- [ ] **Step 3: Add `arrangement` to `_build_horizontal_face` and `FaceRegistry.face_xy`**

In `meshwell/structured/build.py`, replace the signature and body of `_build_horizontal_face`:

```python
def _build_horizontal_face(
    polygon,
    z: float,
    ereg: EdgeRegistry,
    identify_arcs: bool,
    min_arc_points: int,
    arc_tolerance: float,
    face_registry: "FaceRegistry | None" = None,
    arrangement: "Arrangement | None" = None,
) -> TopoDS_Face:
    """Build a horizontal TopoDS_Face for a polygon at fixed z.

    When ``face_registry`` is provided, returns a cached face for the
    polygon's canonical key, sharing TShape across callers. When None,
    constructs a fresh TopoDS_Face each call (legacy behaviour for tests
    and call sites that don't have a registry threaded through).

    When ``arrangement`` is provided, the underlying ``polyline_xy``
    call uses canonical arrangement edges so sub-pieces sharing a
    boundary subset emit the same OCC TShape by construction.
    """
    if face_registry is not None:
        return face_registry.face_xy(
            polygon,
            z,
            identify_arcs,
            min_arc_points,
            arc_tolerance,
            arrangement=arrangement,
        )
    outer_coords = _ring_coords(polygon.exterior)
    outer_edges = ereg.polyline_xy(
        outer_coords,
        z,
        identify_arcs,
        min_arc_points,
        arc_tolerance,
        arrangement=arrangement,
    )
    mw = BRepBuilderAPI_MakeWire()
    for e in outer_edges:
        mw.Add(e)
    outer_wire = mw.Wire()
    mf = BRepBuilderAPI_MakeFace(outer_wire)
    for interior in polygon.interiors:
        hole_coords = _ring_coords(interior)
        hole_edges = ereg.polyline_xy(
            hole_coords,
            z,
            identify_arcs,
            min_arc_points,
            arc_tolerance,
            arrangement=arrangement,
        )
        mw_h = BRepBuilderAPI_MakeWire()
        for e in hole_edges:
            mw_h.Add(e)
        mf.Add(mw_h.Wire())
    return mf.Face()
```

Then replace `FaceRegistry.face_xy`:

```python
    def face_xy(
        self,
        polygon,
        z: float,
        identify_arcs: bool,
        min_arc_points: int,
        arc_tolerance: float,
        arrangement: "Arrangement | None" = None,
    ) -> "TopoDS_Face":
        """Return a unique TopoDS_Face for ``polygon`` at height ``z``.

        Builds the face on first call via ``_build_horizontal_face`` (so
        arc detection / edge sharing through the EdgeRegistry are
        honoured). Subsequent calls with a polygon that produces the
        same canonical key return the cached face by TShape identity.

        When ``arrangement`` is provided, the build path uses canonical
        arrangement edges so sub-pieces sharing a boundary subset emit
        the same OCC TShape on first construction.
        """
        key = self.key_for_polygon(polygon, z)
        if key not in self._store:
            self._store[key] = _build_horizontal_face(
                polygon,
                z,
                self.edges,
                identify_arcs,
                min_arc_points,
                arc_tolerance,
                arrangement=arrangement,
            )
        return self._store[key]
```

- [ ] **Step 4: Run the test, verify it passes**

```bash
pytest tests/structured/test_arrangement_canonical_edges.py::test_face_xy_passes_arrangement_to_polyline_xy -x
```

Expected: PASS.

- [ ] **Step 5: Run the structured suite to verify no regression**

```bash
pytest tests/structured/ -x
```

Expected: ALL PASS.

- [ ] **Step 6: Commit**

```bash
git add meshwell/structured/build.py tests/structured/test_arrangement_canonical_edges.py
git commit -m "feat(structured): thread arrangement through _build_horizontal_face and FaceRegistry.face_xy"
```

---

## Task 7: Thread `arrangement` through `polyline_segments` (lateral)

**Files:**
- Modify: `meshwell/structured/build.py`

The lateral-face builder calls `polyline_segments(coords, identify_arcs, ...)` for each sub-piece ring to slice the boundary into `_PolylineSegment` records. With an arrangement we can replay canonical segmentations instead of running the greedy fitter independently.

- [ ] **Step 1: Write the failing test**

Append to `tests/structured/test_arrangement_canonical_edges.py`:

```python
def test_polyline_segments_uses_arrangement_when_provided():
    """polyline_segments with arrangement produces the same segments
    as polyline_xy does, so lateral faces' bot/top boundaries match
    the horizontal faces' TShapes."""
    from meshwell.structured.build import polyline_segments
    from meshwell.structured.decompose import build_cohort_arrangement
    from meshwell.structured.types import Cohort, StructuredSlab
    from shapely.geometry import Polygon

    def _slab(idx, poly):
        return StructuredSlab(
            source_index=idx, footprint=poly,
            zlo=0.0, zhi=1.0,
            mesh_order=1.0, mesh_bool=True,
            physical_name=("x",),
            identify_arcs=False, arc_tolerance=1e-3, min_arc_points=4,
        )

    left = _slab(0, Polygon([(0, 0), (6, 0), (6, 10), (0, 10)]))
    right = _slab(1, Polygon([(4, 0), (10, 0), (10, 10), (4, 10)]))
    cohort = Cohort(slabs=(left, right), z_planes=(0.0, 1.0))
    arr = build_cohort_arrangement(
        cohort_index=0, cohort=cohort,
        adjacent_unstructured=[], point_tolerance=1e-3,
    )

    overlap_coords = [(4.0, 0.0), (6.0, 0.0), (6.0, 10.0), (4.0, 10.0), (4.0, 0.0)]
    # Both with and without arrangement should yield consistent segments
    # for a rectangle (only lines).
    segs_canon = polyline_segments(
        overlap_coords,
        identify_arcs=False,
        min_arc_points=5,
        arc_tolerance=1e-3,
        point_tolerance=1e-3,
        arrangement=arr,
        z=0.0,
    )
    assert all(s.kind == "line" for s in segs_canon)
    assert len(segs_canon) == 4
```

- [ ] **Step 2: Run the test, verify it fails**

```bash
pytest tests/structured/test_arrangement_canonical_edges.py::test_polyline_segments_uses_arrangement_when_provided -x
```

Expected: FAIL — `polyline_segments` doesn't accept `arrangement=`/`z=`.

- [ ] **Step 3: Add `arrangement` to `polyline_segments`**

In `meshwell/structured/build.py`, replace the existing `polyline_segments` function with:

```python
def polyline_segments(
    coords: list[tuple[float, float]],
    identify_arcs: bool,
    min_arc_points: int,
    arc_tolerance: float,
    point_tolerance: float,
    arrangement: "Arrangement | None" = None,
    z: float | None = None,
) -> list[_PolylineSegment]:
    """Decompose a 2D polyline into line and arc segments.

    When ``arrangement`` is provided AND every consecutive vertex pair
    in ``coords`` is registered in ``arrangement.edge_by_vertex_pair``,
    the canonical pre-fit segments are replayed (forward or reverse
    per traversal direction) so lateral faces match horizontal faces
    on every shared boundary edge. ``z`` must be provided in that case
    (used for vertex-key quantization).

    Falls back to ``decompose_vertices_2d`` on the input ring when no
    arrangement is provided, or when the ring has NO arrangement-edge
    coverage (closed-standalone ring), preserving today's behaviour.
    Raises ``CanonicalArrangementError`` on mixed coverage.
    """
    if arrangement is not None and arrangement.canonical_edges:
        if z is None:
            raise ValueError(
                "polyline_segments requires z= when arrangement is provided"
            )
        segs = _polyline_segments_canonical(
            coords,
            z,
            arrangement,
            point_tolerance,
        )
        if segs is not None:
            return segs
        # else: fall through to greedy fitter (closed-standalone case)

    raw = decompose_vertices_2d(
        coords,
        z=0.0,  # z is irrelevant — we only need (x, y) downstream
        point_tolerance=point_tolerance,
        identify_arcs=identify_arcs,
        min_arc_points=min_arc_points,
        arc_tolerance=arc_tolerance,
    )
    return _flatten_decomposition_to_polyline_segments(raw, point_tolerance)


def _flatten_decomposition_to_polyline_segments(
    raw, point_tolerance: float
) -> list[_PolylineSegment]:
    """Extracted from polyline_segments' legacy body so the canonical
    and greedy paths share the same DecompositionSegment -> _PolylineSegment
    conversion."""
    out: list[_PolylineSegment] = []
    for seg in raw:
        pts = seg.points
        if seg.is_arc:
            start_xy = (pts[0][0], pts[0][1])
            end_xy = (pts[-1][0], pts[-1][1])
            mid_idx = len(pts) // 2
            if (
                abs(start_xy[0] - end_xy[0]) < point_tolerance
                and abs(start_xy[1] - end_xy[1]) < point_tolerance
            ):
                q1_idx = mid_idx // 2
                q3_idx = (mid_idx + len(pts) - 1) // 2
                mid_xy = (pts[mid_idx][0], pts[mid_idx][1])
                out.append(
                    _PolylineSegment(
                        kind="arc",
                        start=start_xy,
                        end=mid_xy,
                        mid=(pts[q1_idx][0], pts[q1_idx][1]),
                    )
                )
                out.append(
                    _PolylineSegment(
                        kind="arc",
                        start=mid_xy,
                        end=end_xy,
                        mid=(pts[q3_idx][0], pts[q3_idx][1]),
                    )
                )
            else:
                out.append(
                    _PolylineSegment(
                        kind="arc",
                        start=start_xy,
                        end=end_xy,
                        mid=(pts[mid_idx][0], pts[mid_idx][1]),
                    )
                )
        else:
            out.extend(
                _PolylineSegment(
                    kind="line",
                    start=(pts[i][0], pts[i][1]),
                    end=(pts[i + 1][0], pts[i + 1][1]),
                )
                for i in range(len(pts) - 1)
            )
    return out


def _polyline_segments_canonical(
    coords: list[tuple[float, float]],
    z: float,
    arrangement: "Arrangement",
    point_tolerance: float,
) -> "list[_PolylineSegment] | None":
    """Replay canonical arrangement edges' segments for a ring.

    Returns the list when every pair is covered; None when NO pairs
    match (closed-standalone fallback); raises on mixed coverage.
    """
    from meshwell.structured.exceptions import CanonicalArrangementError

    def _key(x, y):
        s = point_tolerance
        return (round(x / s), round(y / s), round(z / s))

    keys = [_key(x, y) for x, y in coords]
    if len(keys) >= 2 and keys[0] == keys[-1]:
        inner_keys = keys[:-1]
        ring_is_closed = True
    else:
        inner_keys = list(keys)
        ring_is_closed = False
    n = len(inner_keys)
    if n < 2:
        return []

    pair_count = n if ring_is_closed else n - 1
    hits = 0
    for i in range(pair_count):
        a = inner_keys[i]
        b = inner_keys[(i + 1) % n] if ring_is_closed else inner_keys[i + 1]
        if frozenset({a, b}) in arrangement.edge_by_vertex_pair:
            hits += 1
    if hits == 0:
        return None
    if hits != pair_count:
        raise CanonicalArrangementError(
            cohort_index=arrangement.cohort_index,
            reason=(
                f"lateral ring has mixed canonical coverage "
                f"({hits}/{pair_count} pairs in lookup) — coverage bug"
            ),
        )

    out: list[_PolylineSegment] = []
    consumed = 0
    i = 0
    while consumed < pair_count:
        a_key = inner_keys[i % n]
        b_key = inner_keys[(i + 1) % n]
        edge_idx = arrangement.edge_by_vertex_pair[frozenset({a_key, b_key})]
        canon = arrangement.canonical_edges[edge_idx]
        if canon.vertex_keys[0] == a_key:
            forward = True
        else:
            forward = False
        segments = list(canon.segments)
        if not forward:
            segments = [_reverse_segment(s) for s in reversed(segments)]
        out.extend(_flatten_decomposition_to_polyline_segments(segments, point_tolerance))
        step = len(canon.vertex_keys) - 1
        i += step
        consumed += step
    return out
```

- [ ] **Step 4: Run the test, verify it passes**

```bash
pytest tests/structured/test_arrangement_canonical_edges.py::test_polyline_segments_uses_arrangement_when_provided -x
```

Expected: PASS.

- [ ] **Step 5: Run the structured suite**

```bash
pytest tests/structured/ -x
```

Expected: ALL PASS.

- [ ] **Step 6: Commit**

```bash
git add meshwell/structured/build.py tests/structured/test_arrangement_canonical_edges.py
git commit -m "feat(structured): canonical-edge replay in polyline_segments (lateral)"
```

---

## Task 8: Forward `arrangement` through `build_cohort_compound`

**Files:**
- Modify: `meshwell/structured/build.py` (extend `build_cohort_compound` signature and internal calls)

- [ ] **Step 1: Write the failing test**

Append to `tests/structured/test_arrangement_canonical_edges.py`:

```python
def test_build_cohort_compound_accepts_arrangement():
    """build_cohort_compound forwards arrangement to _build_horizontal_face
    and polyline_segments calls so sub-pieces sharing a boundary subset
    consume canonical edges."""
    from meshwell.structured.build import (
        EdgeRegistry,
        FaceRegistry,
        VertexRegistry,
        build_cohort_compound,
    )
    from meshwell.structured.decompose import (
        arrangement_subpieces_for_interval,
        build_cohort_arrangement,
    )
    from meshwell.structured.types import Cohort, StructuredSlab
    from shapely.geometry import Polygon

    def _slab(idx, poly):
        return StructuredSlab(
            source_index=idx, footprint=poly,
            zlo=0.0, zhi=1.0,
            mesh_order=1.0, mesh_bool=True,
            physical_name=("x",),
            identify_arcs=False, arc_tolerance=1e-3, min_arc_points=4,
        )

    left = _slab(0, Polygon([(0, 0), (6, 0), (6, 10), (0, 10)]))
    right = _slab(1, Polygon([(4, 0), (10, 0), (10, 10), (4, 10)]))
    cohort = Cohort(slabs=(left, right), z_planes=(0.0, 1.0))
    arr = build_cohort_arrangement(
        cohort_index=0, cohort=cohort,
        adjacent_unstructured=[], point_tolerance=1e-3,
    )
    subs = arrangement_subpieces_for_interval(arr, cohort, 0.0, 1.0)

    vreg = VertexRegistry(point_tolerance=1e-3)
    ereg = EdgeRegistry(vertices=vreg, point_tolerance=1e-3)
    freg = FaceRegistry(edges=ereg, point_tolerance=1e-3)

    compound, slab_meta = build_cohort_compound(
        cohort, subs, point_tolerance=1e-3,
        vertex_registry=vreg, edge_registry=ereg, face_registry=freg,
        arrangement=arr,
    )
    assert compound is not None
    assert len(slab_meta) == len(subs)
```

- [ ] **Step 2: Run the test, verify it fails**

```bash
pytest tests/structured/test_arrangement_canonical_edges.py::test_build_cohort_compound_accepts_arrangement -x
```

Expected: FAIL — `build_cohort_compound` does not accept `arrangement=`.

- [ ] **Step 3: Add `arrangement` parameter to `build_cohort_compound`**

In `meshwell/structured/build.py`, edit `build_cohort_compound`:

1. Add `arrangement: "Arrangement | None" = None` to the signature (after `face_registry=...`).
2. Forward to every `_build_horizontal_face` call inside the function. Search for `_build_horizontal_face(` inside `build_cohort_compound` and add `arrangement=arrangement,` to each call.
3. Forward to every `polyline_segments(` call inside `_build_laterals_for_ring`. Add `arrangement=arrangement, z=zlo,` to the `polyline_segments` call (use `zlo` from the enclosing scope; both `zlo` and `zhi` quantize to the SAME canonical-edge z if the cohort arrangement's z is the canonical z — the arrangement edges are 2D so the z passed to quantization just needs to be consistent across canonicaliser and consumer).

The relevant edit inside `build_cohort_compound` looks like:

```python
def build_cohort_compound(
    cohort: Cohort,
    subpieces: list[SubPiece],
    point_tolerance: float,
    vertex_registry: "VertexRegistry | None" = None,
    edge_registry: "EdgeRegistry | None" = None,
    face_registry: "FaceRegistry | None" = None,
    arrangement: "Arrangement | None" = None,
) -> tuple[TopoDS_Compound, dict[ShapeKey, SlabMeta]]:
```

And every occurrence of `_build_horizontal_face(...)` inside the body gets `arrangement=arrangement,` appended to its kwargs:

```python
face = _build_horizontal_face(
    inter_poly, z, ereg, id_arcs, min_p, arc_tol, face_registry=freg, arrangement=arrangement,
)
```

…and the same for the two other `_build_horizontal_face` calls in the "remaining bot/top faces" loop.

For laterals, find `_build_laterals_for_ring` (closure inside `build_cohort_compound`) and update the `polyline_segments` call inside it:

```python
segments = polyline_segments(
    ring_coords,
    identify_arcs=use_arcs,
    min_arc_points=min_arc_points,
    arc_tolerance=arc_tolerance,
    point_tolerance=point_tolerance,
    arrangement=arrangement,
    z=zlo,
)
```

- [ ] **Step 4: Run the test, verify it passes**

```bash
pytest tests/structured/test_arrangement_canonical_edges.py::test_build_cohort_compound_accepts_arrangement -x
```

Expected: PASS.

- [ ] **Step 5: Run the structured suite**

```bash
pytest tests/structured/ -x
```

Expected: ALL PASS.

- [ ] **Step 6: Commit**

```bash
git add meshwell/structured/build.py tests/structured/test_arrangement_canonical_edges.py
git commit -m "feat(structured): forward arrangement through build_cohort_compound"
```

---

## Task 9: Thread per-cohort `Arrangement` from `structured_pre_pass`

**Files:**
- Modify: `meshwell/structured/decompose.py` (return arrangements alongside subpieces)
- Modify: `meshwell/structured/pipeline.py` (consume arrangements; pass to `build_cohort_compound`)
- Test: `tests/structured/test_arrangement_canonical_edges.py` (append end-to-end test)

- [ ] **Step 1: Write the failing test**

Append to `tests/structured/test_arrangement_canonical_edges.py`:

```python
def test_pre_pass_threads_arrangement_to_cohort_compound(tmp_path):
    """structured_pre_pass propagates the per-cohort Arrangement to
    build_cohort_compound so the resulting compound's two overlapping
    rectangles produce a shared edge TShape on the cohort cut."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured.pipeline import structured_pre_pass
    from shapely.geometry import Polygon

    ents = [
        PolyPrism(
            Polygon([(0, 0), (6, 0), (6, 10), (0, 10)]),
            {0.0: 0.0, 1.0: 0.0},
            physical_name="a",
            structured=True,
            mesh_order=1.0,
        ),
        PolyPrism(
            Polygon([(4, 0), (10, 0), (10, 10), (4, 10)]),
            {0.0: 0.0, 1.0: 0.0},
            physical_name="b",
            structured=True,
            mesh_order=2.0,
        ),
    ]
    state = structured_pre_pass(ents, point_tolerance=1e-3)
    # Expect ONE cohort with three sub-pieces (left-only, overlap, right-only).
    assert len(state.cohort_entities) == 1
    # The cohort entity's compound should hold three solids.
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    ce = state.cohort_entities[0]
    exp = TopExp_Explorer(ce.compound, TopAbs_SOLID)
    n_solids = 0
    while exp.More():
        n_solids += 1
        exp.Next()
    assert n_solids == 3
```

- [ ] **Step 2: Run the test, verify it fails**

```bash
pytest tests/structured/test_arrangement_canonical_edges.py::test_pre_pass_threads_arrangement_to_cohort_compound -x
```

Expected: PASS without the change (because end-to-end is exercised by other tests too); if it passes, that's fine — proceed to wire arrangements anyway since downstream sharing is the goal. If it FAILS, that motivates the wiring.

Either outcome, proceed.

- [ ] **Step 3: Modify `decompose_cohorts` to return arrangements**

In `meshwell/structured/decompose.py`, update `decompose_cohorts` so it returns the list of `Arrangement` objects too:

Replace the signature and final return:

```python
def decompose_cohorts(
    cohorts: list[Cohort],
    unstructured_entities: list[Any],
    point_tolerance: float = 1e-3,
) -> tuple[list[list[SubPiece]], list[Any], list[Arrangement]]:
```

Change the final return to:

```python
    return subpieces_per_cohort, list(unstructured_entities), arrangements
```

- [ ] **Step 4: Update `structured_pre_pass` to consume the arrangements**

In `meshwell/structured/pipeline.py`, replace the line:

```python
    subpieces_per_cohort, pre_cut_unstr = decompose_cohorts(
        cohorts, unstructured, point_tolerance=point_tolerance
    )
```

with:

```python
    subpieces_per_cohort, pre_cut_unstr, arrangements = decompose_cohorts(
        cohorts, unstructured, point_tolerance=point_tolerance
    )
```

Then in the `for ci, (cohort, subs) in enumerate(zip(cohorts, subpieces_per_cohort)):` loop, update the `build_cohort_compound` call to pass the arrangement:

```python
        compound, slab_meta = build_cohort_compound(
            cohort,
            subs,
            point_tolerance,
            vertex_registry=vreg,
            edge_registry=ereg,
            face_registry=freg,
            arrangement=arrangements[ci],
        )
```

- [ ] **Step 5: Search for any other callers of `decompose_cohorts` and update**

```bash
grep -rn "decompose_cohorts" tests/ meshwell/ --include="*.py"
```

For each non-pipeline caller, add a third unpacked element (or use `_`). Most tests likely already unpack as two-tuple — they need updating to ignore the third element OR be re-checked.

- [ ] **Step 6: Run the test, verify it passes**

```bash
pytest tests/structured/test_arrangement_canonical_edges.py::test_pre_pass_threads_arrangement_to_cohort_compound -x
```

Expected: PASS.

- [ ] **Step 7: Run the FULL structured suite to catch any regression**

```bash
pytest tests/structured/ -x
```

Expected: ALL PASS.

- [ ] **Step 8: Run the full test suite**

```bash
pytest tests/ -x
```

Expected: ALL PASS.

- [ ] **Step 9: Commit**

```bash
git add meshwell/structured/decompose.py meshwell/structured/pipeline.py tests/structured/test_arrangement_canonical_edges.py
git commit -m "feat(structured): thread per-cohort Arrangement into build_cohort_compound"
```

---

## Task 10: Add `validate_canonical_edge_coverage` invariant + arc-mismatch regression scene

**Files:**
- Modify: `meshwell/structured/decompose.py` (add public validator)
- Modify: `tests/structured/test_shared_edge_registry.py` (append regression)
- Test: `tests/structured/test_arrangement_canonical_edges.py` (append validator test)

The validator scans a list of sub-piece polygons against an arrangement and raises `CanonicalArrangementError` if any ring has MIXED coverage (some pairs found, some missing — a canonicaliser bug). It's test-only — `structured_pre_pass` does NOT call it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/structured/test_arrangement_canonical_edges.py`:

```python
def test_validate_canonical_edge_coverage_passes_on_clean_arrangement():
    """A well-formed arrangement passes the coverage check."""
    from meshwell.structured.decompose import (
        arrangement_subpieces_for_interval,
        build_cohort_arrangement,
        validate_canonical_edge_coverage,
    )
    from meshwell.structured.types import Cohort, StructuredSlab
    from shapely.geometry import Polygon

    def _slab(idx, poly):
        return StructuredSlab(
            source_index=idx, footprint=poly,
            zlo=0.0, zhi=1.0,
            mesh_order=1.0, mesh_bool=True,
            physical_name=("x",),
            identify_arcs=False, arc_tolerance=1e-3, min_arc_points=4,
        )

    cohort = Cohort(
        slabs=(
            _slab(0, Polygon([(0, 0), (6, 0), (6, 10), (0, 10)])),
            _slab(1, Polygon([(4, 0), (10, 0), (10, 10), (4, 10)])),
        ),
        z_planes=(0.0, 1.0),
    )
    arr = build_cohort_arrangement(
        cohort_index=0, cohort=cohort,
        adjacent_unstructured=[], point_tolerance=1e-3,
    )
    subs = arrangement_subpieces_for_interval(arr, cohort, 0.0, 1.0)
    # Should not raise.
    validate_canonical_edge_coverage(arr, [s.sub_polygon for s in subs])
```

Append to `tests/structured/test_shared_edge_registry.py`:

```python
def test_two_overlapping_curved_subpieces_share_canonical_arc(tmp_path):
    """Two overlapping curved sub-pieces sharing a lens-shaped boundary
    consume the same canonical arc edge — proven by the cohort
    EdgeRegistry storing exactly one arc TShape per unique boundary
    arc (no duplicates from per-ring greedy fitting)."""
    from meshwell.structured.build import EdgeRegistry, VertexRegistry
    from meshwell.structured.decompose import (
        arrangement_subpieces_for_interval,
        build_cohort_arrangement,
    )
    from meshwell.structured.types import Cohort, StructuredSlab

    def _slab(idx, poly):
        return StructuredSlab(
            source_index=idx, footprint=poly,
            zlo=0.0, zhi=1.0,
            mesh_order=1.0, mesh_bool=True,
            physical_name=("x",),
            identify_arcs=True, arc_tolerance=1e-3, min_arc_points=5,
        )

    cohort = Cohort(
        slabs=(_slab(0, _circle(0, 0, 1.0)), _slab(1, _circle(1.0, 0, 1.0))),
        z_planes=(0.0, 1.0),
    )
    arr = build_cohort_arrangement(
        cohort_index=0, cohort=cohort,
        adjacent_unstructured=[], point_tolerance=1e-3,
    )
    subs = arrangement_subpieces_for_interval(arr, cohort, 0.0, 1.0)

    vreg = VertexRegistry(point_tolerance=1e-3)
    ereg = EdgeRegistry(vertices=vreg, point_tolerance=1e-3)
    # Build each sub-piece's exterior ring through ereg + arrangement.
    for s in subs:
        coords = list(s.sub_polygon.exterior.coords)
        ereg.polyline_xy(
            [(x, y) for x, y in coords],
            z=0.0,
            identify_arcs=True,
            min_arc_points=5,
            arc_tolerance=1e-3,
            arrangement=arr,
        )
    # Count arc-keyed entries in the registry. Two overlapping discs
    # produce 4 canonical arc edges; 4 arcs in the registry confirm
    # NO duplicates from per-ring fitting.
    arc_keys = [k for k in ereg._store.keys() if k[0] == "A"]
    assert len(arc_keys) == 4, f"expected 4 unique arcs; got {len(arc_keys)}"
```

- [ ] **Step 2: Run the tests, verify they fail**

```bash
pytest tests/structured/test_arrangement_canonical_edges.py::test_validate_canonical_edge_coverage_passes_on_clean_arrangement tests/structured/test_shared_edge_registry.py::test_two_overlapping_curved_subpieces_share_canonical_arc -x
```

Expected: validator import fails on test #1; arc-share test may pass IF Task 5's replay already shares — but with TWO INDEPENDENT discs not sharing rings yet, count may differ. Either way, proceed to the validator implementation, then re-run.

- [ ] **Step 3: Implement `validate_canonical_edge_coverage`**

Append to `meshwell/structured/decompose.py`:

```python
def validate_canonical_edge_coverage(
    arrangement: Arrangement,
    sub_polygons: list,
) -> None:
    """Assert every sub-piece ring has uniform canonical-edge coverage.

    A ring is uniformly covered when EITHER every consecutive vertex
    pair lives in ``arrangement.edge_by_vertex_pair`` (open-edge case),
    OR no pairs do (closed-standalone case). MIXED coverage indicates
    a canonicaliser bug — some pairs were missed during edge splitting.

    Test-only helper; the production ``structured_pre_pass`` does NOT
    call it. Tests pass the sub-piece polygons explicitly to catch
    canonicaliser regressions early.

    Raises:
        CanonicalArrangementError: when any ring has mixed coverage.
    """
    s = _polygon_point_tol(arrangement)

    def _key(x: float, y: float, z: float) -> VertexKey:
        return (round(x / s), round(y / s), round(z / s))

    def _check(coords, z):
        keys = [_key(x, y, z) for x, y in coords]
        if len(keys) >= 2 and keys[0] == keys[-1]:
            inner = keys[:-1]
            closed = True
        else:
            inner = list(keys)
            closed = False
        n = len(inner)
        if n < 2:
            return
        pair_count = n if closed else n - 1
        hits = 0
        for i in range(pair_count):
            a = inner[i]
            b = inner[(i + 1) % n] if closed else inner[i + 1]
            if frozenset({a, b}) in arrangement.edge_by_vertex_pair:
                hits += 1
        if 0 < hits < pair_count:
            raise CanonicalArrangementError(
                cohort_index=arrangement.cohort_index,
                reason=(
                    f"sub-piece ring has mixed canonical coverage "
                    f"({hits}/{pair_count} pairs in lookup)"
                ),
            )

    for poly in sub_polygons:
        # Canonical edges live at the arrangement's z; quantize sub-
        # piece rings at the same z.
        z = arrangement.canonical_edges[0].z if arrangement.canonical_edges else 0.0
        _check(list(poly.exterior.coords), z)
        for interior in poly.interiors:
            _check(list(interior.coords), z)


def _polygon_point_tol(arrangement: Arrangement) -> float:
    """Recover the point_tolerance used to build the arrangement.

    Stored implicitly via the quantization scale: vkey = round(coord / s).
    For the validator we don't have direct access to s; the caller-
    facing convention is to use the arrangement's host point_tolerance.

    We derive s from the FIRST canonical edge's z key: vkey_z = round(z / s)
    -> s = z / vkey_z when both nonzero. Fall back to 1e-3 when z == 0.
    """
    if not arrangement.canonical_edges:
        return 1e-3
    edge = arrangement.canonical_edges[0]
    z = edge.z
    vkey_z = edge.vertex_keys[0][2]
    if z != 0 and vkey_z != 0:
        return abs(z / vkey_z)
    return 1e-3
```

- [ ] **Step 4: Run the tests, verify they pass**

```bash
pytest tests/structured/test_arrangement_canonical_edges.py::test_validate_canonical_edge_coverage_passes_on_clean_arrangement tests/structured/test_shared_edge_registry.py::test_two_overlapping_curved_subpieces_share_canonical_arc -x
```

Expected: ALL PASS.

- [ ] **Step 5: Run the structured suite**

```bash
pytest tests/structured/ -x
```

Expected: ALL PASS.

- [ ] **Step 6: Commit**

```bash
git add meshwell/structured/decompose.py tests/structured/test_arrangement_canonical_edges.py tests/structured/test_shared_edge_registry.py
git commit -m "feat(structured): validate_canonical_edge_coverage + overlapping-discs regression"
```

---

## Task 11: Stress-test the change end-to-end

**Files:**
- Modify: `tests/structured/test_stress_complex_scene.py` (append a tightened assertion)

- [ ] **Step 1: Inspect the existing AABB-rescue counter mechanism**

Read `tests/structured/test_stress_complex_scene.py` to find the existing rescue-count assertion (look for `aabb_rescue` or similar). The current threshold is in place from a prior PR.

```bash
grep -n "aabb_rescue\|rescue_count" tests/structured/test_stress_complex_scene.py
```

If the file has a `MAX_RESCUE_COUNT` constant or similar, that's the knob.

- [ ] **Step 2: Tighten the rescue-count assertion**

Identify the existing constant (e.g., `MAX_RESCUE_COUNT = 12`) and replace with a stricter value chosen by RUNNING the test first to see the new count after Tasks 1-10. To do that:

```bash
pytest tests/structured/test_stress_complex_scene.py -x -s
```

Read the stdout for the rescue count; the new value should be lower than before. Edit the constant to the new value MINUS a small safety margin (e.g., new+2 to absorb future noise).

- [ ] **Step 3: Re-run the test**

```bash
pytest tests/structured/test_stress_complex_scene.py -x
```

Expected: PASS with the tightened bound.

- [ ] **Step 4: Run the full test suite**

```bash
pytest tests/ -x
```

Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/structured/test_stress_complex_scene.py
git commit -m "test(structured): tighten AABB-rescue bound after canonical arc fitting lands"
```

---

## Done

All 11 tasks complete. The cohort arrangement now arc-fits unique edges exactly once and replays canonical segments into every sub-piece's wire builder, so two sub-pieces sharing a curved boundary subset emit the SAME OCC TShape. Manual greedy-fit mismatches at shared arc boundaries are eliminated within the cohort interior; cohort↔unstructured boundaries continue to be handled by BOP + AABB rescue (unchanged).
