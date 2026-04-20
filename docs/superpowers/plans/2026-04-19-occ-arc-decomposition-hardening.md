# OCC Arc-Decomposition Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `OCCGeometryCache` arc sharing robust against the two failure modes surfaced by `shapely.geometry.Polygon.difference` output:
  1. The cache keys arcs on `(start, mid, end, center, radius)`. Two samplings of the *same* arc whose midpoints land at different offsets produce different keys, so the cache fails to share. Fix: key on the sweep direction instead of the midpoint.
  2. `decompose_vertices` walks left-to-right and never detects arc-runs that wrap across the input seam. A rotated coordinate sequence (as shapely's diff produces) then emits what should be one arc as a few line segments plus an arc stump. Fix: rotate the input to a canonical seam before decomposing.

After this plan, two PolyPrisms that share an arc should share that arc's TShape even when one's polygon was produced by a shapely boolean (rotated start index, possible duplicate seam vertex).

**Architecture:**
- `OCCGeometryCache.get_arc_edge` switches its key from `(start, mid, end, center, radius)` to `(start, end, center, radius, sweep_sign)`. `sweep_sign ∈ {+1, -1}` is the sign of the z-component of `(start - center) × (mid - center)`. Same-arc samples collapse to the same key; short-vs-long arcs keep different keys. For near-π arcs where the cross product is ~0, fall back to a canonical tiebreaker (lex-quantized midpoint).
- `GeometryEntity._make_occ_wire_from_vertices` (or a new helper it delegates to) de-duplicates consecutive coincident vertices before decomposing, eliminating the `shapely.difference` seam-duplicate artifact that currently splits entities into slivers.
- `GeometryEntity.decompose_vertices` gains a preprocessing step: for closed polylines (first==last), rotate the sequence so it starts at a "hard seam" — a vertex whose incoming/outgoing tangents turn sharply — before running the greedy arc detector. Arcs therefore never straddle the input seam.

**Tech Stack:** Python 3.11+, `numpy`, `OCP` (`gp_Pnt`, `GC_MakeArcOfCircle`, `BRepBuilderAPI_MakeEdge`), `pytest`, `shapely`.

---

## File Structure

- **Modify:** `meshwell/occ_geometry_cache.py` — `get_arc_edge` re-keys on sweep direction.
- **Modify:** `meshwell/geometry_entity.py` — `decompose_vertices` preprocesses with seam rotation; `_make_occ_wire_from_vertices` (non-arc and arc branches) strips consecutive duplicates before handing off.
- **Modify:** `tests/test_occ_geometry_cache.py` — add unit tests for direction-based keying.
- **Modify:** `tests/test_cad_occ.py` — add end-to-end test for `shapely.difference` outer + inner rounded rect producing single-piece inner and fully-shared cylindrical interfaces.
- **Test (existing, must still pass):** `tests/test_cad_occ.py`, `tests/test_cad_occ_fragment_ownership.py`, `tests/test_multidimensional_cad_occ.py`, `tests/test_occ_geometry_cache.py`.

---

## Task 1: Sweep-sign key in `get_arc_edge`

**Files:**
- Modify: `meshwell/occ_geometry_cache.py`
- Modify: `tests/test_occ_geometry_cache.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_occ_geometry_cache.py`:

```python
def test_arc_edge_reused_with_different_midpoints_same_arc():
    """Two samples of the same geometric arc must collide in the cache."""
    import math

    cache = OCCGeometryCache(point_tolerance=1e-3)
    center = gp_Pnt(0.0, 0.0, 0.0)
    radius = 1.0
    p_start = gp_Pnt(1.0, 0.0, 0.0)
    p_end = gp_Pnt(0.0, 1.0, 0.0)

    # Two distinct midpoints, both on the short (counterclockwise) arc.
    mid_a = gp_Pnt(math.cos(math.pi / 5), math.sin(math.pi / 5), 0.0)
    mid_b = gp_Pnt(math.cos(math.pi / 3), math.sin(math.pi / 3), 0.0)

    e_a = cache.get_arc_edge(p_start, mid_a, p_end, center, radius)
    e_b = cache.get_arc_edge(p_start, mid_b, p_end, center, radius)
    assert _HASHER(e_a) == _HASHER(e_b)


def test_arc_edge_near_pi_uses_midpoint_tiebreaker():
    """For a ~π-arc the sweep sign is ambiguous; key must still be stable."""
    import math

    cache = OCCGeometryCache(point_tolerance=1e-3)
    center = gp_Pnt(0.0, 0.0, 0.0)
    radius = 1.0
    p_start = gp_Pnt(1.0, 0.0, 0.0)
    p_end = gp_Pnt(-1.0, 0.0, 0.0)
    p_mid_top = gp_Pnt(0.0, 1.0, 0.0)  # upper semicircle
    p_mid_bot = gp_Pnt(0.0, -1.0, 0.0)  # lower semicircle

    e_top = cache.get_arc_edge(p_start, p_mid_top, p_end, center, radius)
    e_bot = cache.get_arc_edge(p_start, p_mid_bot, p_end, center, radius)
    # The two halves are different arcs: they must not collapse.
    assert _HASHER(e_top) != _HASHER(e_bot)

    # A second call with the same orientation must still collide.
    e_top2 = cache.get_arc_edge(p_start, p_mid_top, p_end, center, radius)
    assert _HASHER(e_top) == _HASHER(e_top2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_occ_geometry_cache.py::test_arc_edge_reused_with_different_midpoints_same_arc -v`
Expected: FAIL — the current key includes the exact midpoint, so different midpoints produce different edges.

- [ ] **Step 3: Replace the arc key with sweep-direction encoding**

In `meshwell/occ_geometry_cache.py`, change `get_arc_edge` to compute `sweep_sign` from the 2D cross product of `(start - center) × (mid - center)`. When `|cross|` is below a numerical threshold (near-π arc), fall back to quantizing the midpoint's lex-minimal direction as a tiebreaker.

```python
def _sweep_key(
    p_start: gp_Pnt,
    p_mid: gp_Pnt,
    center: gp_Pnt,
    ndigits: int,
) -> tuple:
    """Return a key fragment that identifies which of the two possible arcs
    between the fixed (start, end, center, radius) we're on.

    For well-defined arcs (not near π), this is just ``sign(cross)``. Near π,
    we fall back to a quantized direction vector so the two half-circles are
    distinguished deterministically.
    """
    sx, sy = p_start.X() - center.X(), p_start.Y() - center.Y()
    mx, my = p_mid.X() - center.X(), p_mid.Y() - center.Y()
    cross_z = sx * my - sy * mx
    # Normalize the cross to the unit circle so the threshold is scale-free.
    norm = (sx * sx + sy * sy) ** 0.5
    if norm == 0.0:
        norm = 1.0
    cross_norm = cross_z / (norm * norm)

    if cross_norm > 1e-6:
        return ("sgn", 1)
    if cross_norm < -1e-6:
        return ("sgn", -1)
    # Near-π arc: key on quantized midpoint direction from center.
    dx = round(mx, ndigits)
    dy = round(my, ndigits)
    return ("dir", dx, dy)
```

Then in `get_arc_edge`:

```python
def get_arc_edge(self, p_start, p_mid, p_end, center, radius) -> TopoDS_Edge:
    k_start = self.vertex_key(p_start)
    k_end = self.vertex_key(p_end)
    k_center = self.vertex_key(center)
    r_q = round(float(radius), self._ndigits)
    sweep = _sweep_key(p_start, p_mid, center, self._ndigits)
    key = (k_start, k_end, k_center, r_q, sweep)

    v_start = self.get_vertex(p_start)
    v_end = self.get_vertex(p_end)
    with self._lock:
        edge = self._arc_edges.get(key)
        if edge is None:
            arc_geom = GC_MakeArcOfCircle(p_start, p_mid, p_end).Value()
            edge = BRepBuilderAPI_MakeEdge(arc_geom, v_start, v_end).Edge()
            self._arc_edges[key] = edge
        return edge
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_occ_geometry_cache.py -v`
Expected: PASS — existing six tests still pass; the two new tests pass.

- [ ] **Step 5: Run the OCC regression suite**

Run: `uv run pytest tests/test_cad_occ.py tests/test_cad_occ_fragment_ownership.py tests/test_multidimensional_cad_occ.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add meshwell/occ_geometry_cache.py tests/test_occ_geometry_cache.py
git commit -m "feat(occ_geometry_cache): key arcs on sweep direction instead of midpoint

Two different sampled midpoints of the same geometric arc now collapse
to the same cache entry, so polyline samplings that disagree on which
specific vertex falls near the middle of an arc still share topology.
Near-pi arcs where sign(cross) is ambiguous fall back to a quantized
midpoint-direction tiebreaker."
```

---

## Task 2: Strip consecutive duplicate vertices before decomposing

**Files:**
- Modify: `meshwell/geometry_entity.py` (`decompose_vertices` or a preprocessing helper used by it and `_make_occ_wire_from_vertices`).
- Modify: `tests/test_occ_geometry_cache.py` — add unit test for duplicate-vertex stripping.

Context: `shapely.geometry.Polygon(...).difference(...)` can emit a hole boundary whose last vertex duplicates the first (N+1 points for N distinct ones, beyond the usual closing-point convention). This trips `BRepBuilderAPI_MakeEdge` into a zero-length edge — which BOPAlgo treats as a sliver, splitting neighbouring entities into spurious pieces (the `inner: 2 pieces` artifact we observed).

- [ ] **Step 1: Write the failing test**

```python
def test_strip_consecutive_duplicates_in_decompose():
    """Consecutive coincident vertices must not emit zero-length edges."""
    from meshwell.geometry_entity import GeometryEntity

    entity = GeometryEntity(point_tolerance=1e-3)
    # Square with a duplicate at index 2.
    verts = [(0, 0, 0), (1, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0)]
    segs = entity.decompose_vertices(verts, identify_arcs=False)
    # Should collapse to 4 line segments (square), not 5.
    assert len(segs) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_occ_geometry_cache.py::test_strip_consecutive_duplicates_in_decompose -v`
Expected: FAIL — current `decompose_vertices` emits 5 segments, one of them zero-length.

- [ ] **Step 3: Add `_strip_duplicates` preprocessor**

In `meshwell/geometry_entity.py`, add a module-level helper and invoke it at the top of `decompose_vertices` (before the early-return paths):

```python
def _strip_consecutive_duplicates(
    vertices: list[tuple[float, float, float]],
    tolerance: float,
) -> list[tuple[float, float, float]]:
    """Collapse adjacent vertices coincident within ``tolerance``.

    Preserves the closing vertex (equal to the opening vertex) when the
    input is a closed polyline so downstream consumers still see a closed
    ring.
    """
    if len(vertices) < 2:
        return list(vertices)
    tol_sq = tolerance * tolerance
    out = [vertices[0]]
    for v in vertices[1:]:
        prev = out[-1]
        dx = v[0] - prev[0]
        dy = v[1] - prev[1]
        dz = v[2] - prev[2]
        if dx * dx + dy * dy + dz * dz > tol_sq:
            out.append(v)
    # Restore closing vertex if input was closed but stripping removed it.
    if (
        len(vertices) >= 2
        and vertices[0] == vertices[-1]
        and (len(out) < 2 or out[0] != out[-1])
    ):
        out.append(out[0])
    return out
```

In `decompose_vertices`:

```python
def decompose_vertices(self, vertices, identify_arcs=False, min_arc_points=4, arc_tolerance=1e-3):
    vertices = _strip_consecutive_duplicates(list(vertices), self.point_tolerance)
    if not vertices or len(vertices) < 2:
        return []
    ...  # existing logic below
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_occ_geometry_cache.py tests/test_cad_occ.py tests/test_cad_occ_fragment_ownership.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/geometry_entity.py tests/test_occ_geometry_cache.py
git commit -m "feat(geometry_entity): strip consecutive duplicate vertices in decompose

Shapely boolean operations can emit rings whose last vertex duplicates
the first beyond the usual closing convention. The zero-length edge
derived from such a pair was producing sliver entities during
fragmentation. Collapse adjacent duplicates up front."
```

---

## Task 3: Canonicalize the seam before arc decomposition

**Files:**
- Modify: `meshwell/geometry_entity.py` (`decompose_vertices`).
- Modify: `tests/test_occ_geometry_cache.py` — add unit tests for rotation-invariant decomposition.

Context: `decompose_vertices` greedily extends arcs starting from index 0. For a rounded rectangle whose coordinate sequence happens to start mid-arc (as shapely's diff produces), the leading samples of the first arc get misclassified as line segments, and the trailing samples of the last arc likewise. Two entities whose polygons are the "same" but rotated therefore disagree on the arc partitioning — and then the arc cache cannot share them.

Fix: for closed polylines, rotate the input to start at the sharpest turn before decomposing. On a rounded rectangle, the sharpest turns are at the transitions between straight edges and rounded corners — never in the middle of an arc — so every rotation of the input lands on the same canonical starting index.

- [ ] **Step 1: Write the failing tests**

```python
def test_decompose_rotation_invariant_on_rounded_rect():
    """Rotated coord sequences of a rounded rectangle produce identical arc partitions."""
    import numpy as np
    from meshwell.geometry_entity import GeometryEntity

    def rounded(n_arc=8):
        hw, hh, r = 2.0, 1.5, 0.6
        specs = [
            ((hw - r, hh - r), 0.0),
            ((-hw + r, hh - r), np.pi / 2),
            ((-hw + r, -hh + r), np.pi),
            ((hw - r, -hh + r), 3 * np.pi / 2),
        ]
        coords = []
        for (cx, cy), a0 in specs:
            for a in np.linspace(a0, a0 + np.pi / 2, n_arc + 1):
                coords.append((cx + r * np.cos(a), cy + r * np.sin(a), 0.0))
        coords.append(coords[0])  # close
        return coords

    entity = GeometryEntity(point_tolerance=1e-3)
    base = rounded()
    # Rotate so the sequence starts mid-arc (index 4 of the first arc run).
    rotated = base[4:-1] + base[:4]
    rotated.append(rotated[0])

    segs_base = entity.decompose_vertices(base, identify_arcs=True, min_arc_points=4)
    segs_rot = entity.decompose_vertices(rotated, identify_arcs=True, min_arc_points=4)

    n_arcs_base = sum(1 for s in segs_base if s.is_arc)
    n_arcs_rot = sum(1 for s in segs_rot if s.is_arc)
    assert n_arcs_base == n_arcs_rot == 4

    # Arc centers + radii must match as a multiset.
    def sig(seg):
        return (
            tuple(round(c, 3) for c in seg.center),
            round(seg.radius, 3),
        )

    sigs_base = sorted(sig(s) for s in segs_base if s.is_arc)
    sigs_rot = sorted(sig(s) for s in segs_rot if s.is_arc)
    assert sigs_base == sigs_rot
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_occ_geometry_cache.py::test_decompose_rotation_invariant_on_rounded_rect -v`
Expected: FAIL — current implementation detects fewer arcs (2 or 3) on the rotated input because the mid-arc start splits the first/last arc run.

- [ ] **Step 3: Add seam rotation**

Add a helper in `meshwell/geometry_entity.py`:

```python
def _find_canonical_seam(
    vertices: list[tuple[float, float, float]],
    sharp_cos_threshold: float = 0.5,
) -> int:
    """Return an index at which to start a closed polyline so arc runs don't
    straddle the seam.

    Heuristic: pick the vertex whose incoming/outgoing edges turn the most
    sharply (smallest cos(angle)). Ties broken by lexicographic order so the
    result is deterministic across equivalent polygons.
    """
    import numpy as np

    n = len(vertices) - 1  # last equals first in a closed ring
    if n < 3:
        return 0

    best_idx = 0
    best_cos = 2.0  # larger than any valid cos
    best_key = None
    for i in range(n):
        prev = vertices[(i - 1) % n]
        cur = vertices[i]
        nxt = vertices[(i + 1) % n]
        v1 = np.array(cur) - np.array(prev)
        v2 = np.array(nxt) - np.array(cur)
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-12 or n2 < 1e-12:
            continue
        cos_a = float(np.dot(v1, v2) / (n1 * n2))
        key = (cos_a, tuple(round(c, 9) for c in cur))
        if cos_a < best_cos - 1e-9 or (
            abs(cos_a - best_cos) < 1e-9 and (best_key is None or key < best_key)
        ):
            best_cos = cos_a
            best_idx = i
            best_key = key

    # If the sharpest turn is still smooth (e.g., a pure circle), the seam
    # doesn't matter — any deterministic choice works. Use lexmin vertex.
    if best_cos > sharp_cos_threshold:
        best_idx = min(range(n), key=lambda i: vertices[i])
    return best_idx


def _rotate_closed(
    vertices: list[tuple[float, float, float]], start_idx: int
) -> list[tuple[float, float, float]]:
    if start_idx == 0 or len(vertices) < 2:
        return list(vertices)
    closed = vertices[0] == vertices[-1]
    core = vertices[:-1] if closed else list(vertices)
    rotated = core[start_idx:] + core[:start_idx]
    if closed:
        rotated.append(rotated[0])
    return rotated
```

In `decompose_vertices` (after `_strip_consecutive_duplicates`):

```python
vertices = _strip_consecutive_duplicates(list(vertices), self.point_tolerance)
if not vertices or len(vertices) < 2:
    return []
if identify_arcs and len(vertices) >= max(min_arc_points + 1, 4) and vertices[0] == vertices[-1]:
    seam = _find_canonical_seam(vertices)
    vertices = _rotate_closed(vertices, seam)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_occ_geometry_cache.py tests/test_cad_occ.py tests/test_cad_occ_fragment_ownership.py tests/test_multidimensional_cad_occ.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/geometry_entity.py tests/test_occ_geometry_cache.py
git commit -m "feat(geometry_entity): rotate closed polylines to a canonical seam

Greedy arc detection starting at an arbitrary index splits arcs that
straddle the input seam into (partial-arc, line, line, partial-arc). For
closed rings we now rotate the vertex sequence to start at the sharpest
corner before decomposing, so arc detection is invariant under the
caller's choice of starting index."
```

---

## Task 4: End-to-end regression for shapely.difference sharing

**Files:**
- Modify: `tests/test_cad_occ.py`.

- [ ] **Step 1: Write the failing test**

```python
def test_occ_shapely_difference_rounded_rect_shares_all_arcs():
    """Outer polygon produced by shapely.difference must share all arcs
    and surfaces with the matching inner rounded-rect entity."""
    import numpy as np
    from shapely.geometry import Polygon
    from meshwell.model import ModelManager

    def rounded_rect_coords(w, h, r, n_arc=8):
        hw, hh = w / 2, h / 2
        specs = [
            ((hw - r, hh - r), 0.0),
            ((-hw + r, hh - r), np.pi / 2),
            ((-hw + r, -hh + r), np.pi),
            ((hw - r, -hh + r), 3 * np.pi / 2),
        ]
        out = []
        for (cx, cy), a0 in specs:
            for a in np.linspace(a0, a0 + np.pi / 2, n_arc + 1):
                out.append((cx + r * np.cos(a), cy + r * np.sin(a)))
        return out

    inner_coords = rounded_rect_coords(4.0, 3.0, 0.6, n_arc=8)
    inner_poly = Polygon(inner_coords)
    bigger = Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])
    outer_diffed = bigger.difference(inner_poly)

    outer_prism = PolyPrism(
        polygons=outer_diffed,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="outer",
        mesh_order=1,
        identify_arcs=True,
    )
    inner_prism = PolyPrism(
        polygons=inner_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="inner",
        mesh_order=2,
        identify_arcs=True,
    )
    occ_ents = cad_occ([outer_prism, inner_prism])
    by_name = {e.physical_name[0]: e for e in occ_ents}

    # Inner should be a single piece — the duplicate-seam sliver is gone.
    assert len(by_name["inner"].shapes) == 1

    mm = ModelManager(filename="test_shapely_diff_sharing")
    try:
        inject_occ_entities_into_gmsh(occ_ents, mm)
        groups = gmsh.model.getPhysicalGroups(2)
        named = {gmsh.model.getPhysicalName(d, t): t for d, t in groups}
        assert "outer___inner" in named or "inner___outer" in named

        tag = named.get("outer___inner") or named["inner___outer"]
        faces = gmsh.model.getEntitiesForPhysicalGroup(2, tag)
        # Full lateral wall of the inner prism: 4 cylinders + 4 planes.
        types = sorted(gmsh.model.getType(2, f) for f in faces)
        assert types.count("Cylinder") == 4
        assert types.count("Plane") == 4
    finally:
        mm.finalize()
```

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/test_cad_occ.py::test_occ_shapely_difference_rounded_rect_shares_all_arcs -v`
Expected: PASS (this is the regression gate — it was effectively covered by the ad-hoc probe; Tasks 1–3 make it hold formally).

- [ ] **Step 3: Commit**

```bash
git add tests/test_cad_occ.py
git commit -m "test(cad_occ): shapely.difference outer + inner share all arcs cleanly"
```

---

## Self-Review Checklist

- [ ] Arc cache key is now `(start, end, center, radius, sweep_sign)`; two samples of the same arc share a TShape.
- [ ] `decompose_vertices` strips consecutive duplicates before returning any segments.
- [ ] `decompose_vertices` rotates closed polylines to a canonical seam before arc detection, and falls back to lexmin when no sharp corner exists.
- [ ] The shapely-diff end-to-end probe no longer produces a sliver `inner` piece.
- [ ] All regression suites (`test_cad_occ`, `test_cad_occ_fragment_ownership`, `test_multidimensional_cad_occ`, `test_occ_geometry_cache`) still pass.
