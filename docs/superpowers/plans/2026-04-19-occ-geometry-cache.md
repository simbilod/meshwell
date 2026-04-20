# OCC Geometry Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `BOPAlgo_Builder` recognize coincident sub-geometry across entities by giving the same OCC `TopoDS_Vertex` / `TopoDS_Edge` (TShape-identical) to every entity that references it. Today each entity builds its own edges from scratch, and even geometrically-coincident arcs end up with distinct TShapes, so fragmentation cannot merge boundaries between e.g. a rectangle-with-rounded-hole and a rounded inner rectangle.

**Architecture:**
- New `OCCGeometryCache` caches `TopoDS_Vertex` and `TopoDS_Edge` keyed on quantized geometry (rounded coordinates, center, radius, midpoint). All edge/vertex construction inside `GeometryEntity._make_occ_wire_from_vertices` and `_make_occ_face_from_vertices` goes through the cache.
- `CAD_OCC.process_entities` builds one cache per call, serializes the instantiation loop (the cache is mutable, and parallel instantiation gains are dwarfed by fragmentation), and threads the cache through each entity's `instanciate_occ(cache)`.
- `PolyPrism.instanciate_occ` stops cutting holes with `BRepAlgoAPI_Cut` — that operation forges new TShapes and discards the hole wire's identity. Replace with `BRepBuilderAPI_MakeFace(outer_wire).Add(reversed_hole_wire)`, which preserves edge/vertex TShapes from the shared cache.

**Tech Stack:** Python 3.11+, `OCP` (`BRepBuilderAPI_MakeVertex`, `BRepBuilderAPI_MakeEdge`, `BRepBuilderAPI_MakeFace`, `TopoDS_Vertex`, `TopoDS_Edge`, `gp_Pnt`, `gp_Circ`, `GC_MakeArcOfCircle`), `pytest`, `shapely`.

---

## File Structure

- **Create:** `meshwell/occ_geometry_cache.py` — the `OCCGeometryCache` class and canonical keying helpers.
- **Modify:** `meshwell/geometry_entity.py` — `_make_occ_points`, `_make_occ_wire_from_vertices`, `_make_occ_face_from_vertices`, `instanciate_occ` all accept an optional `occ_cache`. Edges built through the cache when provided; fall back to original behaviour otherwise.
- **Modify:** `meshwell/polyprism.py` — `instanciate_occ(self, occ_cache=None)`; drop `BRepAlgoAPI_Cut` for holes in favor of `BRepBuilderAPI_MakeFace.Add(hole_wire)` with proper reversal; thread `occ_cache` into all wire/face helper calls.
- **Modify:** `meshwell/polysurface.py`, `meshwell/polyline.py`, `meshwell/occ_entity.py` — accept `occ_cache=None` parameter (polysurface/polyline pass through to wire builders; `occ_entity` ignores).
- **Modify:** `meshwell/cad_occ.py` — `_instantiate_entity_occ` receives and forwards `occ_cache`; `process_entities` creates one cache per call and serializes the instantiation loop.
- **Create:** `tests/test_occ_geometry_cache.py` — unit tests for cache keying and reuse behavior.
- **Test (existing):** `tests/test_cad_occ.py::test_occ_rounded_rect_inside_rect_with_cutout_shares_arcs` currently fails; must pass after this plan.
- **Test (existing):** `tests/test_cad_occ.py`, `tests/test_cad_occ_fragment_ownership.py`, `tests/test_multidimensional_cad_occ.py`, `tests/test_performance_cad_occ.py` — must pass unchanged as regression.

---

## Task 1: OCCGeometryCache scaffold with vertex caching

**Files:**
- Create: `meshwell/occ_geometry_cache.py`
- Create: `tests/test_occ_geometry_cache.py`

- [ ] **Step 1: Write the failing test**

```python
"""Unit tests for the OCC geometry cache."""
from __future__ import annotations

from OCP.gp import gp_Pnt
from OCP.TopTools import TopTools_ShapeMapHasher

from meshwell.occ_geometry_cache import OCCGeometryCache


_HASHER = TopTools_ShapeMapHasher()


def test_vertex_reused_within_tolerance():
    cache = OCCGeometryCache(point_tolerance=1e-3)
    v1 = cache.get_vertex(gp_Pnt(0.0, 0.0, 0.0))
    v2 = cache.get_vertex(gp_Pnt(0.0001, 0.0, 0.0))  # within tolerance
    assert _HASHER(v1) == _HASHER(v2)


def test_vertex_distinct_outside_tolerance():
    cache = OCCGeometryCache(point_tolerance=1e-3)
    v1 = cache.get_vertex(gp_Pnt(0.0, 0.0, 0.0))
    v2 = cache.get_vertex(gp_Pnt(0.01, 0.0, 0.0))  # outside tolerance
    assert _HASHER(v1) != _HASHER(v2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_occ_geometry_cache.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the cache**

```python
"""Shared geometry cache for OCC entity instantiation.

When multiple PolyPrism (or other GeometryEntity) objects share coincident
points/edges/arcs, BOPAlgo_Builder only recognizes them as shared if they
carry the same TShape identity. Rebuilding geometry entity-by-entity gives
each a fresh TShape, so fragmentation treats geometrically-identical
boundaries as distinct. This cache fixes that by quantizing coordinates
and returning a single TopoDS_Vertex / TopoDS_Edge for every reference to
the same canonical geometry.
"""
from __future__ import annotations

import threading
from math import floor, log10
from typing import TYPE_CHECKING

from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex

if TYPE_CHECKING:
    from OCP.gp import gp_Pnt
    from OCP.TopoDS import TopoDS_Edge, TopoDS_Vertex


def _coord_ndigits(tolerance: float) -> int:
    return max(0, int(-floor(log10(tolerance))))


def _round_pt(x: float, y: float, z: float, ndigits: int) -> tuple[float, float, float]:
    return (round(float(x), ndigits), round(float(y), ndigits), round(float(z), ndigits))


class OCCGeometryCache:
    """Cache TopoDS_Vertex and TopoDS_Edge objects keyed by quantized geometry."""

    def __init__(self, point_tolerance: float = 1e-3):
        self.point_tolerance = point_tolerance
        self._ndigits = _coord_ndigits(point_tolerance)
        self._lock = threading.Lock()
        self._vertices: dict[tuple[float, float, float], TopoDS_Vertex] = {}

    def vertex_key(self, pnt: gp_Pnt) -> tuple[float, float, float]:
        return _round_pt(pnt.X(), pnt.Y(), pnt.Z(), self._ndigits)

    def get_vertex(self, pnt: gp_Pnt) -> TopoDS_Vertex:
        key = self.vertex_key(pnt)
        with self._lock:
            v = self._vertices.get(key)
            if v is None:
                v = BRepBuilderAPI_MakeVertex(pnt).Vertex()
                self._vertices[key] = v
            return v
```

- [ ] **Step 4: Run tests — they pass**

Run: `pytest tests/test_occ_geometry_cache.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/occ_geometry_cache.py tests/test_occ_geometry_cache.py
git commit -m "feat(occ_geometry_cache): add cache with shared vertex construction"
```

---

## Task 2: Extend cache with line-edge reuse

**Files:**
- Modify: `meshwell/occ_geometry_cache.py`
- Modify: `tests/test_occ_geometry_cache.py`

- [ ] **Step 1: Write the failing test**

```python
def test_line_edge_reused_same_endpoints():
    cache = OCCGeometryCache(point_tolerance=1e-3)
    p0 = gp_Pnt(0.0, 0.0, 0.0)
    p1 = gp_Pnt(1.0, 0.0, 0.0)
    e1 = cache.get_line_edge(p0, p1)
    e2 = cache.get_line_edge(p0, p1)
    assert _HASHER(e1) == _HASHER(e2)


def test_line_edge_reused_reverse_endpoints():
    cache = OCCGeometryCache(point_tolerance=1e-3)
    p0 = gp_Pnt(0.0, 0.0, 0.0)
    p1 = gp_Pnt(1.0, 0.0, 0.0)
    e_fwd = cache.get_line_edge(p0, p1)
    e_rev = cache.get_line_edge(p1, p0)
    # Same TShape regardless of direction; orientation flips.
    assert _HASHER(e_fwd) == _HASHER(e_rev)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_occ_geometry_cache.py -v`
Expected: FAIL — `get_line_edge` does not exist.

- [ ] **Step 3: Implement `get_line_edge`**

Add to `OCCGeometryCache`:

```python
    def __init__(self, point_tolerance: float = 1e-3):
        self.point_tolerance = point_tolerance
        self._ndigits = _coord_ndigits(point_tolerance)
        self._lock = threading.Lock()
        self._vertices: dict[tuple[float, float, float], TopoDS_Vertex] = {}
        self._line_edges: dict[
            tuple[tuple[float, float, float], tuple[float, float, float]], TopoDS_Edge
        ] = {}

    def get_line_edge(self, p1: gp_Pnt, p2: gp_Pnt) -> TopoDS_Edge:
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge

        k1 = self.vertex_key(p1)
        k2 = self.vertex_key(p2)
        key = (k1, k2) if k1 <= k2 else (k2, k1)
        with self._lock:
            edge = self._line_edges.get(key)
            if edge is None:
                v1 = self.get_vertex(p1) if k1 <= k2 else self.get_vertex(p2)
                v2 = self.get_vertex(p2) if k1 <= k2 else self.get_vertex(p1)
                edge = BRepBuilderAPI_MakeEdge(v1, v2).Edge()
                self._line_edges[key] = edge
            return edge
```

Note: `get_vertex` is re-entrant so we call it without re-acquiring the lock (`threading.Lock` is not re-entrant). Restructure: build vertices first, then take the lock only for the edges dict.

Final form:

```python
    def get_line_edge(self, p1: gp_Pnt, p2: gp_Pnt) -> TopoDS_Edge:
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge

        k1 = self.vertex_key(p1)
        k2 = self.vertex_key(p2)
        if k1 <= k2:
            key = (k1, k2)
            v1 = self.get_vertex(p1)
            v2 = self.get_vertex(p2)
        else:
            key = (k2, k1)
            v1 = self.get_vertex(p2)
            v2 = self.get_vertex(p1)
        with self._lock:
            edge = self._line_edges.get(key)
            if edge is None:
                edge = BRepBuilderAPI_MakeEdge(v1, v2).Edge()
                self._line_edges[key] = edge
            return edge
```

- [ ] **Step 4: Run tests — they pass**

Run: `pytest tests/test_occ_geometry_cache.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/occ_geometry_cache.py tests/test_occ_geometry_cache.py
git commit -m "feat(occ_geometry_cache): cache line edges by unordered vertex-key pair"
```

---

## Task 3: Extend cache with arc-edge reuse

**Files:**
- Modify: `meshwell/occ_geometry_cache.py`
- Modify: `tests/test_occ_geometry_cache.py`

- [ ] **Step 1: Write the failing test**

```python
def test_arc_edge_reused_same_params():
    import math

    cache = OCCGeometryCache(point_tolerance=1e-3)
    center = gp_Pnt(0.0, 0.0, 0.0)
    radius = 1.0
    p_start = gp_Pnt(1.0, 0.0, 0.0)
    p_mid = gp_Pnt(math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0)
    p_end = gp_Pnt(0.0, 1.0, 0.0)
    e1 = cache.get_arc_edge(p_start, p_mid, p_end, center, radius)
    e2 = cache.get_arc_edge(p_start, p_mid, p_end, center, radius)
    assert _HASHER(e1) == _HASHER(e2)


def test_arc_edge_distinct_for_opposite_arc():
    import math

    cache = OCCGeometryCache(point_tolerance=1e-3)
    center = gp_Pnt(0.0, 0.0, 0.0)
    radius = 1.0
    p_start = gp_Pnt(1.0, 0.0, 0.0)
    p_end = gp_Pnt(0.0, 1.0, 0.0)
    # Short arc vs long arc share endpoints and center but differ in midpoint.
    short_mid = gp_Pnt(math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0)
    long_mid = gp_Pnt(math.cos(5 * math.pi / 4), math.sin(5 * math.pi / 4), 0.0)
    e_short = cache.get_arc_edge(p_start, short_mid, p_end, center, radius)
    e_long = cache.get_arc_edge(p_start, long_mid, p_end, center, radius)
    assert _HASHER(e_short) != _HASHER(e_long)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_occ_geometry_cache.py -v`
Expected: FAIL — `get_arc_edge` does not exist.

- [ ] **Step 3: Implement `get_arc_edge`**

```python
    def __init__(self, point_tolerance: float = 1e-3):
        ...  # existing lines
        self._arc_edges: dict[tuple, TopoDS_Edge] = {}

    def get_arc_edge(
        self,
        p_start: gp_Pnt,
        p_mid: gp_Pnt,
        p_end: gp_Pnt,
        center: gp_Pnt,
        radius: float,
    ) -> TopoDS_Edge:
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
        from OCP.GC import GC_MakeArcOfCircle

        k_start = self.vertex_key(p_start)
        k_end = self.vertex_key(p_end)
        k_mid = self.vertex_key(p_mid)
        k_center = self.vertex_key(center)
        r_q = round(float(radius), self._ndigits)
        key = (k_start, k_mid, k_end, k_center, r_q)

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

- [ ] **Step 4: Run tests — they pass**

Run: `pytest tests/test_occ_geometry_cache.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/occ_geometry_cache.py tests/test_occ_geometry_cache.py
git commit -m "feat(occ_geometry_cache): cache arcs by endpoints + midpoint + center"
```

---

## Task 4: Thread cache into GeometryEntity wire/face builders

**Files:**
- Modify: `meshwell/geometry_entity.py:349-471`

- [ ] **Step 1: Update `_make_occ_wire_from_vertices` to accept and use `occ_cache`**

Add an optional `occ_cache: OCCGeometryCache | None = None` parameter. When provided:
- Replace every `BRepBuilderAPI_MakeEdge(p1, p2).Edge()` (line edges) with `occ_cache.get_line_edge(p1, p2)`.
- Replace every arc edge construction with `occ_cache.get_arc_edge(p_start, p_mid, p_end, center_gp_Pnt, seg.radius)`.
  - For non-full-circle arcs, `p_mid` is the decomposed segment's middle point and `center_gp_Pnt` is built from `seg.center`.
  - For the full-circle fallback, call `get_arc_edge` twice with the quarter/three-quarter midpoints.
- When `occ_cache is None`, preserve the existing behaviour exactly.

- [ ] **Step 2: Update `_make_occ_face_from_vertices` to forward `occ_cache`**

```python
    def _make_occ_face_from_vertices(
        self,
        vertices: list[tuple[float, float, float]],
        identify_arcs: bool = False,
        min_arc_points: int = 4,
        arc_tolerance: float = 1e-3,
        occ_cache: OCCGeometryCache | None = None,
    ) -> TopoDS_Face:
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace

        wire = self._make_occ_wire_from_vertices(
            vertices,
            identify_arcs=identify_arcs,
            min_arc_points=min_arc_points,
            arc_tolerance=arc_tolerance,
            occ_cache=occ_cache,
        )
        return BRepBuilderAPI_MakeFace(wire).Face()
```

- [ ] **Step 3: Update base-class `instanciate_occ` signature**

```python
    def instanciate_occ(
        self, occ_cache: OCCGeometryCache | None = None
    ) -> TopoDS_Shape:
        raise NotImplementedError("Subclasses must implement instanciate_occ method")
```

- [ ] **Step 4: Run regression tests**

Run: `pytest tests/test_cad_occ.py tests/test_cad_occ_fragment_ownership.py -v -x`
Expected: PASS (no behavioural change yet — cache not yet plumbed from callers).

- [ ] **Step 5: Commit**

```bash
git add meshwell/geometry_entity.py
git commit -m "feat(geometry_entity): thread OCCGeometryCache through wire/face builders"
```

---

## Task 5: PolyPrism uses cache + direct face-with-holes (no more BRepAlgoAPI_Cut)

**Files:**
- Modify: `meshwell/polyprism.py:544-604` and `meshwell/polyprism.py:526-542` and `meshwell/polyprism.py:351-446`

- [ ] **Step 1: Replace hole-cutting in extrude branch of `instanciate_occ`**

Before:
```python
face = self._make_occ_face_from_vertices(exterior_vertices, ...)
for interior in poly.interiors:
    hole_face = self._make_occ_face_from_vertices(hole_vertices, ...)
    cut_api = BRepAlgoAPI_Cut(face, hole_face)
    cut_api.Build()
    face = cut_api.Shape()
```

After — build the outer wire, reverse each hole wire, and add them to `BRepBuilderAPI_MakeFace` so the face itself carries the shared edges:

```python
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.TopoDS import TopoDS_Face

outer_wire = self._make_occ_wire_from_vertices(
    exterior_vertices,
    identify_arcs=self.identify_arcs,
    min_arc_points=self.min_arc_points,
    arc_tolerance=self.arc_tolerance,
    occ_cache=occ_cache,
)
mf = BRepBuilderAPI_MakeFace(outer_wire)
for interior in poly.interiors:
    hole_vertices = [(x, y, self.zmin) for x, y in interior.coords]
    hole_wire = self._make_occ_wire_from_vertices(
        hole_vertices,
        identify_arcs=self.identify_arcs,
        min_arc_points=self.min_arc_points,
        arc_tolerance=self.arc_tolerance,
        occ_cache=occ_cache,
    )
    hole_wire.Reverse()
    mf.Add(hole_wire)
face = mf.Face()
```

Signature: `def instanciate_occ(self, occ_cache: OCCGeometryCache | None = None) -> TopoDS_Shape:`

- [ ] **Step 2: Thread `occ_cache` through `_create_occ_volume_with_holes` and `_create_occ_volume`**

The non-extrude branch also calls `BRepAlgoAPI_Cut`. That branch uses `addThruSections`-style volume cutting; we leave it as-is for this iteration (non-zero-buffer case). Document the limitation inline:

```python
# Non-extrude path retains BRepAlgoAPI_Cut for now; shared-topology guarantees
# only hold in the extrude (arc-capable) path until this helper is reworked.
```

Still forward the cache into every `_make_occ_face_from_vertices` call so line-edge reuse works for vertical facets and top/bottom faces.

- [ ] **Step 3: Run the rounded-rect test**

Run: `pytest tests/test_cad_occ.py::test_occ_rounded_rect_inside_rect_with_cutout_shares_arcs -v`
Expected: Still fails — cache not yet created upstream in `CAD_OCC`.

- [ ] **Step 4: Commit**

```bash
git add meshwell/polyprism.py
git commit -m "refactor(polyprism): build faces with holes directly; accept occ_cache"
```

---

## Task 6: Thread cache through remaining entity types

**Files:**
- Modify: `meshwell/polysurface.py:149`, `meshwell/polyline.py:155`, `meshwell/occ_entity.py:40`

- [ ] **Step 1: Add `occ_cache` parameter to each entity's `instanciate_occ`**

Each signature becomes:
```python
def instanciate_occ(
    self, occ_cache: OCCGeometryCache | None = None
) -> TopoDS_Shape:
```

- For `OCC_entity`: parameter is accepted and ignored (pre-built `TopoDS_Shape` has no edges to cache).
- For `PolySurface` and `PolyLine`: forward `occ_cache` into every `_make_occ_wire_from_vertices` / `_make_occ_face_from_vertices` call within the method.

- [ ] **Step 2: Run regression tests**

Run: `pytest tests/test_cad_occ.py tests/test_cad_occ_fragment_ownership.py tests/test_multidimensional_cad_occ.py -v -x`
Expected: PASS (cache still `None` since callers haven't been updated).

- [ ] **Step 3: Commit**

```bash
git add meshwell/polysurface.py meshwell/polyline.py meshwell/occ_entity.py
git commit -m "feat(entities): accept occ_cache parameter in instanciate_occ"
```

---

## Task 7: CAD_OCC creates and threads the cache; serialize instantiation

**Files:**
- Modify: `meshwell/cad_occ.py:117-214`

- [ ] **Step 1: Update `_instantiate_entity_occ` to accept and forward the cache**

```python
def _instantiate_entity_occ(
    self,
    index: int,
    entity_obj: Any,
    occ_cache: OCCGeometryCache,
) -> OCCLabeledEntity:
    shape = entity_obj.instanciate_occ(occ_cache=occ_cache)
    ...
```

- [ ] **Step 2: Serialize instantiation and build the cache in `process_entities`**

```python
def process_entities(
    self,
    entities_list: list[Any],
    _progress_bars: bool = False,
) -> list[OCCLabeledEntity]:
    if not entities_list:
        return []

    occ_cache = OCCGeometryCache(point_tolerance=self.point_tolerance)
    labeled_entities = [
        self._instantiate_entity_occ(i, ent, occ_cache)
        for i, ent in enumerate(entities_list)
    ]
    return self._fragment_all(labeled_entities)
```

Remove the `ThreadPoolExecutor` import if no longer used.

- [ ] **Step 3: Run the rounded-rect test**

Run: `pytest tests/test_cad_occ.py::test_occ_rounded_rect_inside_rect_with_cutout_shares_arcs -v`
Expected: PASS — the rounded hole and the rounded inner rect now share arc and line edges, so BOPAlgo produces an `outer___inner` interface with ≥ 4 surfaces.

- [ ] **Step 4: Run full regression suite**

Run: `pytest tests/test_cad_occ.py tests/test_cad_occ_fragment_ownership.py tests/test_multidimensional_cad_occ.py tests/test_occ_geometry_cache.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/cad_occ.py
git commit -m "feat(cad_occ): build one OCCGeometryCache per call, serialize instantiation"
```

---

## Task 8: End-to-end test for cache-driven sharing across independent entities

**Files:**
- Modify: `tests/test_cad_occ.py`

- [ ] **Step 1: Add a test that two PolyPrisms with a coincident arc boundary produce a shared surface**

Two arc-fitted PolyPrism disks whose boundaries touch along an arc (e.g. two half-disks that together form a circle) — after `cad_occ` + `inject_occ_entities_into_gmsh`, the `left___right` interface must exist as a single surface of GMSH type `Cylinder` or `Circle`.

```python
def test_two_arc_prisms_share_cylindrical_interface():
    import numpy as np
    from shapely.geometry import Polygon

    # Left half-disk (x <= 0) and right half-disk (x >= 0), shared vertical boundary at x=0.
    def half_disk(sign, n=16):
        thetas = np.linspace(-np.pi / 2, np.pi / 2, n + 1) if sign > 0 else np.linspace(np.pi / 2, 3 * np.pi / 2, n + 1)
        coords = [(np.cos(t), np.sin(t)) for t in thetas]
        coords.append((0.0, coords[-1][1]))  # close via vertical chord
        return Polygon(coords)

    left = PolyPrism(
        polygons=half_disk(+1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="left",
        mesh_order=1,
        identify_arcs=True,
    )
    right = PolyPrism(
        polygons=half_disk(-1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="right",
        mesh_order=2,
        identify_arcs=True,
    )
    occ_ents = cad_occ([left, right])
    from meshwell.model import ModelManager
    mm = ModelManager(filename="test_two_arc_prisms_share")
    try:
        inject_occ_entities_into_gmsh(occ_ents, mm)
        groups = gmsh.model.getPhysicalGroups(2)
        interface = next(
            tag for dim, tag in groups
            if gmsh.model.getPhysicalName(dim, tag) in {"left___right", "right___left"}
        )
        faces = gmsh.model.getEntitiesForPhysicalGroup(2, interface)
        assert len(faces) == 1
    finally:
        mm.finalize()
```

- [ ] **Step 2: Run the new test**

Run: `pytest tests/test_cad_occ.py::test_two_arc_prisms_share_cylindrical_interface -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_cad_occ.py
git commit -m "test(cad_occ): two PolyPrisms sharing an arc produce one shared surface"
```

---

## Self-Review Checklist

- [ ] Every `instanciate_occ` caller updated to pass the cache where it has one.
- [ ] `ThreadPoolExecutor` no longer imported in `cad_occ.py` if unused.
- [ ] `BRepAlgoAPI_Cut` removed from the extrude branch of `PolyPrism.instanciate_occ`.
- [ ] Cache quantization uses `point_tolerance` consistently everywhere.
- [ ] No regression in existing `tests/test_cad_occ.py`, `tests/test_cad_occ_fragment_ownership.py`, `tests/test_multidimensional_cad_occ.py`, `tests/test_performance_cad_occ.py`.
- [ ] The previously-failing `test_occ_rounded_rect_inside_rect_with_cutout_shares_arcs` now passes.
