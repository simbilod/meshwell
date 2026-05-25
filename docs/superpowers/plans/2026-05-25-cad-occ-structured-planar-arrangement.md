# Planar-arrangement preprocessing for the structured planner — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-slab face_partition + fixed-point arc inheritance machinery in `meshwell/structured/plan.py` with a planar-arrangement preprocessing phase that builds canonical edges (arcs and lines) once per connected z-touching slab stack and uses them as the single geometric source of truth from face_partition through phantom build.

**Architecture:** Phase-additive refactor. Tasks 1–11 add the new arrangement infrastructure alongside the existing `compute_face_partition` (no behavior change yet, every existing test stays green). Tasks 12–14 migrate `build_plan` to call the new pipeline, update tests that asserted on old behavior, and flip both arc xfails. Tasks 15–17 delete the old machinery and add the comprehensive stress test.

**Tech Stack:** Python 3.12, shapely 2.x (`unary_union`, `polygonize`, `split`), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-25-cad-occ-structured-planar-arrangement-design.md`

---

## File Structure

**Modified files:**
- `meshwell/structured/spec.py` — add 4 new dataclasses; add 2 fields to `Slab`; remove `StructuredPartitionConvergenceError` at the end (Task 15).
- `meshwell/structured/__init__.py` — export the new types; remove convergence error at Task 15.
- `meshwell/structured/plan.py` — substantial: add Step 0 + Steps A–G helpers + two orchestrators (Tasks 2–11); migrate `build_plan` (Task 13); delete old machinery (Task 15).
- `tests/structured/test_plan.py` — add new tests per task; update breaking tests at Tasks 12–13; remove obsolete tests at Task 15.
- `tests/structured/test_stress_stacked_patterns.py` — flip 2 xfails at Task 14; add comprehensive stress test at Task 16.
- `tests/structured/test_structured_arc_polyprism.py` — possibly update assertion content at Task 13.

**New files:** none. All changes land in existing files.

---

## Task 1: Add new dataclasses + Slab fields

**Files:**
- Modify: `meshwell/structured/spec.py`
- Modify: `meshwell/structured/__init__.py`
- Test: `tests/structured/test_spec.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/structured/test_spec.py`:

```python
def test_canonical_circle_is_hashable():
    """CanonicalCircle is a frozen dataclass usable as a dict/set key."""
    from meshwell.structured import CanonicalCircle

    a = CanonicalCircle(center=(1.0, 2.0), radius=3.0)
    b = CanonicalCircle(center=(1.0, 2.0), radius=3.0)
    assert a == b
    assert hash(a) == hash(b)
    assert a in {b}


def test_arrangement_edge_carries_circle_or_none():
    """ArrangementEdge is a line when circle is None, an arc when not."""
    from meshwell.structured import ArrangementEdge, CanonicalCircle

    line = ArrangementEdge(edge_id=0, vertices=((0.0, 0.0), (1.0, 0.0)), circle=None)
    arc = ArrangementEdge(
        edge_id=1,
        vertices=((1.0, 0.0), (0.707, 0.707), (0.0, 1.0)),
        circle=CanonicalCircle(center=(0.0, 0.0), radius=1.0),
    )
    assert line.circle is None
    assert arc.circle is not None
    assert arc.circle.radius == 1.0


def test_arrangement_face_holds_polygon_and_boundary():
    """ArrangementFace carries a Polygon and an ordered edge-id list."""
    from shapely.geometry import Polygon

    from meshwell.structured import ArrangementFace

    f = ArrangementFace(
        face_id=0,
        polygon=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        boundary=[(0, False), (1, False), (2, False), (3, False)],
    )
    assert f.face_id == 0
    assert f.polygon.area == 1.0
    assert len(f.boundary) == 4


def test_stack_arrangement_holds_edges_and_faces():
    """StackArrangement is the per-component output type."""
    from meshwell.structured import StackArrangement

    s = StackArrangement(edges=[], faces=[])
    assert s.edges == []
    assert s.faces == []


def test_slab_has_resolved_footprint_and_face_partition_edges():
    """Slab gains two new optional fields for the arrangement pipeline."""
    from shapely.geometry import Polygon

    from meshwell.structured.spec import Slab

    s = Slab(
        footprint=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        zlo=0.0,
        zhi=1.0,
        physical_name=("X",),
        source_index=0,
        z_interval_index=0,
        mesh_order=1.0,
    )
    # New fields default to safe values.
    assert s.resolved_footprint is None
    assert s.face_partition_edges is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/structured/test_spec.py -v --no-cov -k "canonical_circle or arrangement_edge or arrangement_face or stack_arrangement or has_resolved_footprint"`
Expected: 5 failures with `ImportError: cannot import name ...` or `AttributeError: ...`

- [ ] **Step 3: Add the dataclasses to `meshwell/structured/spec.py`**

Locate the existing `PieceArcEdge` / `PieceLineEdge` / `PieceProvenance` definitions. Add these new dataclasses next to them:

```python
@dataclass(frozen=True)
class CanonicalCircle:
    """Identity of a circular curve shared across arrangement edges.

    Two arrangement edges with CanonicalCircle instances matching on
    (center, radius) within arc_tolerance are sub-arcs of the same
    physical circle. The phantom builder uses (center, radius) plus arc
    endpoints to construct OCC arc geometry; consumers of the same
    circle produce bit-identical TShapes.
    """

    center: tuple[float, float]
    radius: float


@dataclass(frozen=True)
class ArrangementEdge:
    """One non-crossing curve segment in the planar arrangement.

    vertices: ordered XY sample points; >=2 elements. Endpoints are
        vertices[0] and vertices[-1].
    circle: None means a straight line. Not None means a sub-arc of
        the named circle; endpoints lie on it.
    """

    edge_id: int
    vertices: tuple[tuple[float, float], ...]
    circle: "CanonicalCircle | None"


@dataclass
class ArrangementFace:
    """One face of the planar arrangement (a Polygon with no interior holes).

    boundary: ordered list of (edge_id, reversed) tuples describing the
        traversal of the face's outer ring. ``reversed=True`` means the
        edge's vertex sequence is walked in reverse.
    """

    face_id: int
    polygon: "Polygon"
    boundary: list[tuple[int, bool]]


@dataclass
class StackArrangement:
    """Per-z-touching-component planar arrangement; consumed by face-partition assignment."""

    edges: list[ArrangementEdge]
    faces: list[ArrangementFace]
```

Locate the `@dataclass class Slab:` definition. Add two new fields after the existing `face_partition` and `face_partition_provenance` fields:

```python
    # Populated by Step 0 (sub-level mesh-order resolution) of the new
    # planar-arrangement pipeline. Equal to footprint when no carving
    # applies. None until Step 0 runs.
    resolved_footprint: "Polygon | MultiPolygon | None" = None

    # Populated by Step F (assign-faces-to-slabs). Parallel to face_partition;
    # face_partition_edges[i] is the boundary of face_partition[i] expressed
    # as (edge_id, reversed) tuples into the slab's stack arrangement.
    face_partition_edges: "list[list[tuple[int, bool]]] | None" = None
```

- [ ] **Step 4: Export new types from `meshwell/structured/__init__.py`**

Update the import block from `meshwell.structured.spec`:

```python
from meshwell.structured.spec import (
    ArrangementEdge,
    ArrangementFace,
    CanonicalCircle,
    PhantomMap,
    StackArrangement,
    StructuredArcSplitError,
    StructuredExtrusionResolutionSpec,
    StructuredLateralUnstructuredNeighbourError,
    StructuredMeshPlan,
    StructuredMidHeightCutError,
    StructuredOverlapError,
    StructuredPartitionConvergenceError,
)
```

Update `__all__` (alphabetical order, keep `StructuredPartitionConvergenceError` for now — removed at Task 15):

```python
__all__ = [
    "ArrangementEdge",
    "ArrangementFace",
    "CanonicalCircle",
    "PhantomMap",
    "StackArrangement",
    "StructuredArcSplitError",
    "StructuredExtrusionResolutionSpec",
    "StructuredLateralUnstructuredNeighbourError",
    "StructuredMeshPlan",
    "StructuredMidHeightCutError",
    "StructuredOverlapError",
    "StructuredPartitionConvergenceError",
    "apply_structured_mesh",
    "build_phantom_shapes",
    "build_plan",
    "extract_phantom_map",
    "resolve_mesh_plan",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/structured/test_spec.py -v --no-cov -k "canonical_circle or arrangement_edge or arrangement_face or stack_arrangement or has_resolved_footprint"`
Expected: 5 passed.

- [ ] **Step 6: Run full structured suite to ensure no regression**

Run: `.venv/bin/python -m pytest tests/structured/ --no-cov -q 2>&1 | tail -3`
Expected: 168 passed (existing), 2 skipped, 2 xfailed. No new failures.

- [ ] **Step 7: Commit**

```bash
git add meshwell/structured/spec.py meshwell/structured/__init__.py tests/structured/test_spec.py
git commit -m "feat(structured): add planar-arrangement dataclasses + Slab fields"
```

---

## Task 2: `_resolve_sublevel_mesh_order` (Step 0)

Sub-level mesh-order resolution: for each z-interval, carve the loser's footprint by the winner's. `mesh_bool=False` entities carve out of all kept neighbours.

**Files:**
- Modify: `meshwell/structured/plan.py` (add the new helper; do NOT yet call it from `build_plan`)
- Test: `tests/structured/test_plan.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/structured/test_plan.py`:

```python
def test_resolve_sublevel_carves_loser_by_winner():
    """Same-z overlap: lower mesh_order wins; loser's resolved_footprint is carved."""
    from shapely.geometry import Polygon

    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec
    from meshwell.structured.plan import _resolve_sublevel_mesh_order, expand_to_slabs, gather_structured_entities

    a = PolyPrism(
        polygons=Polygon([(0, 0), (4, 0), (4, 2), (0, 2)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="A",
        mesh_order=1,
    )
    b = PolyPrism(
        polygons=Polygon([(2, 0), (6, 0), (6, 2), (2, 2)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="B",
        mesh_order=2,
    )
    entities = [a, b]
    slabs = expand_to_slabs(gather_structured_entities(entities))
    _resolve_sublevel_mesh_order(slabs, entities)

    by_name = {s.physical_name[0]: s for s in slabs}
    # A keeps its full footprint (winner).
    assert by_name["A"].resolved_footprint.area == 8.0  # 4 * 2
    # B keeps only [4,6] x [0,2] (carved by A).
    assert by_name["B"].resolved_footprint.area == 4.0  # 2 * 2


def test_resolve_sublevel_mesh_bool_false_carves_kept_neighbour():
    """A mesh_bool=False entity carves out of overlapping kept slabs."""
    from shapely.geometry import Polygon

    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec
    from meshwell.structured.plan import _resolve_sublevel_mesh_order, expand_to_slabs, gather_structured_entities

    kept = PolyPrism(
        polygons=Polygon([(0, 0), (4, 0), (4, 2), (0, 2)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="kept",
        mesh_order=1,
    )
    void = PolyPrism(
        polygons=Polygon([(1, 0.5), (2, 0.5), (2, 1.5), (1, 1.5)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        mesh_bool=False,
        physical_name="void_tag",
    )
    entities = [kept, void]
    slabs = expand_to_slabs(gather_structured_entities(entities))
    _resolve_sublevel_mesh_order(slabs, entities)

    by_name = {s.physical_name[0]: s for s in slabs}
    # Kept has the 1x1 void carved out.
    assert abs(by_name["kept"].resolved_footprint.area - (8.0 - 1.0)) < 1e-9


def test_resolve_sublevel_disjoint_footprints_unchanged():
    """Slabs at different z-intervals or non-overlapping XY are unaffected."""
    from shapely.geometry import Polygon

    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec
    from meshwell.structured.plan import _resolve_sublevel_mesh_order, expand_to_slabs, gather_structured_entities

    a = PolyPrism(
        polygons=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="A",
        mesh_order=1,
    )
    b_diff_z = PolyPrism(
        polygons=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        buffers={2.0: 0.0, 3.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="B",
        mesh_order=2,
    )
    c_diff_xy = PolyPrism(
        polygons=Polygon([(10, 0), (11, 0), (11, 1), (10, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="C",
        mesh_order=2,
    )
    entities = [a, b_diff_z, c_diff_xy]
    slabs = expand_to_slabs(gather_structured_entities(entities))
    _resolve_sublevel_mesh_order(slabs, entities)

    by_name = {s.physical_name[0]: s for s in slabs}
    assert by_name["A"].resolved_footprint.equals(a.polygons)
    assert by_name["B"].resolved_footprint.equals(b_diff_z.polygons)
    assert by_name["C"].resolved_footprint.equals(c_diff_xy.polygons)
```

- [ ] **Step 2: Run tests to verify failures**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py -v --no-cov -k "resolve_sublevel" 2>&1 | tail -10`
Expected: 3 failures with `ImportError: cannot import name '_resolve_sublevel_mesh_order'`.

- [ ] **Step 3: Implement `_resolve_sublevel_mesh_order`**

In `meshwell/structured/plan.py`, add the helper after `expand_to_slabs`:

```python
def _resolve_sublevel_mesh_order(slabs: list[Slab], entities: list[Any]) -> None:
    """Set ``slab.resolved_footprint`` in place per sub-level mesh-order carving.

    For each z-interval (grouped by (zlo, zhi) keys), sort kept slabs by
    (mesh_order, source_index) ascending. The first (winner) keeps its
    full footprint. Each subsequent slab's resolved_footprint is its
    original footprint minus the union of every prior winner's
    resolved_footprint at the same z-interval.

    mesh_bool=False entities whose z-range covers the sub-level
    additionally carve out of every kept slab's resolved_footprint.
    They do not themselves carry a resolved_footprint (they're not in
    the slab list — only their boundaries propagate to step C).
    """
    # Group slabs by (zlo, zhi).
    by_interval: dict[tuple[float, float], list[Slab]] = {}
    for s in slabs:
        by_interval.setdefault((s.zlo, s.zhi), []).append(s)

    for (zlo, zhi), group in by_interval.items():
        # Collect mesh_bool=False entities whose z-range covers this interval.
        void_footprints: list[Any] = []
        for ent in entities:
            if getattr(ent, "mesh_bool", True):
                continue
            rng = _entity_z_range(ent)
            if rng is None:
                continue
            ent_zmin, ent_zmax = rng
            if ent_zmin <= zlo + _Z_TOL and ent_zmax >= zhi - _Z_TOL:
                fp = _entity_footprint(ent)
                if fp is not None:
                    void_footprints.append(fp)

        # Sort by (mesh_order, source_index): winners first.
        ordered = sorted(group, key=lambda s: (s.mesh_order, s.source_index))
        accumulated_winners: list[Any] = []
        for slab in ordered:
            resolved = slab.footprint
            if accumulated_winners:
                resolved = resolved.difference(unary_union(accumulated_winners))
            if void_footprints:
                resolved = resolved.difference(unary_union(void_footprints))
            slab.resolved_footprint = resolved
            accumulated_winners.append(slab.footprint)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py -v --no-cov -k "resolve_sublevel"`
Expected: 3 passed.

- [ ] **Step 5: Run full structured suite — should still be green (we haven't wired the helper into build_plan yet)**

Run: `.venv/bin/python -m pytest tests/structured/ --no-cov -q 2>&1 | tail -3`
Expected: same as Task 1 — 168 passed, 2 skipped, 2 xfailed.

- [ ] **Step 6: Commit**

```bash
git add meshwell/structured/plan.py tests/structured/test_plan.py
git commit -m "feat(structured): add _resolve_sublevel_mesh_order helper (Step 0)"
```

---

## Task 3: `_connected_z_components` (Step A)

Group slabs into connected components by z-touching.

**Files:**
- Modify: `meshwell/structured/plan.py`
- Test: `tests/structured/test_plan.py`

- [ ] **Step 1: Write failing tests**

```python
def test_connected_z_components_face_touching_chain():
    """Face-touching slabs (a.zhi == b.zlo) end up in the same component."""
    from shapely.geometry import Polygon

    from meshwell.structured.plan import _connected_z_components
    from meshwell.structured.spec import Slab

    fp = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    a = Slab(footprint=fp, zlo=0.0, zhi=1.0, physical_name=("A",), source_index=0, z_interval_index=0, mesh_order=1.0)
    b = Slab(footprint=fp, zlo=1.0, zhi=2.0, physical_name=("B",), source_index=1, z_interval_index=0, mesh_order=1.0)
    c = Slab(footprint=fp, zlo=2.0, zhi=3.0, physical_name=("C",), source_index=2, z_interval_index=0, mesh_order=1.0)

    components = _connected_z_components([a, b, c])
    assert len(components) == 1
    assert {id(s) for s in components[0]} == {id(a), id(b), id(c)}


def test_connected_z_components_disjoint_stacks():
    """Stacks separated by gaps are separate components."""
    from shapely.geometry import Polygon

    from meshwell.structured.plan import _connected_z_components
    from meshwell.structured.spec import Slab

    fp = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    a = Slab(footprint=fp, zlo=0.0, zhi=1.0, physical_name=("A",), source_index=0, z_interval_index=0, mesh_order=1.0)
    b = Slab(footprint=fp, zlo=10.0, zhi=11.0, physical_name=("B",), source_index=1, z_interval_index=0, mesh_order=1.0)

    components = _connected_z_components([a, b])
    assert len(components) == 2
    names = sorted({s.physical_name[0] for c in components for s in c})
    assert names == ["A", "B"]


def test_connected_z_components_same_z_interval_grouped():
    """Slabs sharing the same z-interval are also in the same component (lateral connection)."""
    from shapely.geometry import Polygon

    from meshwell.structured.plan import _connected_z_components
    from meshwell.structured.spec import Slab

    a = Slab(
        footprint=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        zlo=0.0, zhi=1.0, physical_name=("A",), source_index=0, z_interval_index=0, mesh_order=1.0,
    )
    b = Slab(
        footprint=Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        zlo=0.0, zhi=1.0, physical_name=("B",), source_index=1, z_interval_index=0, mesh_order=1.0,
    )
    components = _connected_z_components([a, b])
    assert len(components) == 1
```

- [ ] **Step 2: Run tests to verify failures**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py -v --no-cov -k "connected_z_components"`
Expected: 3 ImportError failures.

- [ ] **Step 3: Implement `_connected_z_components`**

Add in `plan.py` after `_resolve_sublevel_mesh_order`:

```python
def _connected_z_components(slabs: list[Slab]) -> list[list[Slab]]:
    """Group slabs into connected components.

    Two slabs are in the same component iff either:
      - they share a z-face (abs(a.zhi - b.zlo) < _Z_TOL or symmetric), or
      - they share the same z-interval (a.zlo == b.zlo AND a.zhi == b.zhi).

    The two-clause rule ensures that same-z-interval lateral neighbours
    (e.g., two structured slabs at z=[0,1] abutting at x=1) are grouped
    together. Without that, their cuts wouldn't propagate to the
    arrangement at all.

    Implementation: Union-Find on slab indices.
    """
    parent = list(range(len(slabs)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i, a in enumerate(slabs):
        for j in range(i + 1, len(slabs)):
            b = slabs[j]
            face_touching = (
                abs(a.zhi - b.zlo) < _Z_TOL or abs(a.zlo - b.zhi) < _Z_TOL
            )
            same_interval = (
                abs(a.zlo - b.zlo) < _Z_TOL and abs(a.zhi - b.zhi) < _Z_TOL
            )
            if face_touching or same_interval:
                union(i, j)

    components_by_root: dict[int, list[Slab]] = {}
    for i, s in enumerate(slabs):
        components_by_root.setdefault(find(i), []).append(s)
    return list(components_by_root.values())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py -v --no-cov -k "connected_z_components"`
Expected: 3 passed.

- [ ] **Step 5: Run full structured suite**

Run: `.venv/bin/python -m pytest tests/structured/ --no-cov -q 2>&1 | tail -3`
Expected: 171 passed, 2 skipped, 2 xfailed.

- [ ] **Step 6: Commit**

```bash
git add meshwell/structured/plan.py tests/structured/test_plan.py
git commit -m "feat(structured): add _connected_z_components helper (Step A)"
```

---

## Task 4: `_collect_stack_boundaries` (Step B)

Gather the boundaries that feed the arrangement: each member slab's resolved footprint + unstructured z-touching entity footprints.

**Files:**
- Modify: `meshwell/structured/plan.py`
- Test: `tests/structured/test_plan.py`

- [ ] **Step 1: Write failing test**

```python
def test_collect_stack_boundaries_includes_resolved_and_unstructured():
    """Returns resolved footprints of stack members + unstructured z-touching footprints."""
    from shapely.geometry import Polygon

    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec
    from meshwell.structured.plan import _collect_stack_boundaries
    from meshwell.structured.spec import Slab

    structured_ent = PolyPrism(
        polygons=Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="S",
    )
    unstructured_ent = PolyPrism(
        polygons=Polygon([(0, 0), (3, 0), (3, 3), (0, 3)]),
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="U",
    )
    s_slab = Slab(
        footprint=structured_ent.polygons,
        resolved_footprint=structured_ent.polygons,
        zlo=0.0,
        zhi=1.0,
        physical_name=("S",),
        source_index=0,
        z_interval_index=0,
        mesh_order=1.0,
    )
    stack = [s_slab]
    boundaries = _collect_stack_boundaries(stack, [structured_ent, unstructured_ent])
    # Should include 2 LineStrings: s_slab's resolved boundary + unstructured ent's boundary (z-touching at z=1).
    assert len(boundaries) == 2
```

- [ ] **Step 2: Run test to verify failure**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py -v --no-cov -k "collect_stack_boundaries"`
Expected: ImportError failure.

- [ ] **Step 3: Implement**

Add in `plan.py`:

```python
def _collect_stack_boundaries(stack: list[Slab], entities: list[Any]) -> list[Any]:
    """Boundaries to feed the arrangement for this connected component.

    Returns a list of shapely LineString/MultiLineString geometries:
    - Each stack member slab's resolved_footprint.boundary (falling back
      to slab.footprint.boundary when resolved_footprint is None).
    - Each unstructured entity whose z-range touches any z-plane of the
      stack — its boundary contributes line cuts only (no arcs).
    """
    boundaries: list[Any] = []
    seen_source_indices = {s.source_index for s in stack}

    # Stack member resolved footprints.
    for slab in stack:
        fp = slab.resolved_footprint if slab.resolved_footprint is not None else slab.footprint
        if not fp.is_empty:
            boundaries.append(fp.boundary)

    # Stack's z-planes.
    z_planes = set()
    for slab in stack:
        z_planes.add(slab.zlo)
        z_planes.add(slab.zhi)

    # Unstructured z-touching entities.
    for i, ent in enumerate(entities):
        if i in seen_source_indices:
            continue
        if getattr(ent, "structured", False):
            continue
        rng = _entity_z_range(ent)
        if rng is None:
            continue
        zmin, zmax = rng
        if any(abs(zmin - z) < _Z_TOL or abs(zmax - z) < _Z_TOL for z in z_planes):
            fp = _entity_footprint(ent)
            if fp is not None:
                boundaries.append(fp.boundary)

    return boundaries
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py -v --no-cov -k "collect_stack_boundaries"`
Expected: 1 passed.

- [ ] **Step 5: Full suite**

Run: `.venv/bin/python -m pytest tests/structured/ --no-cov -q 2>&1 | tail -3`
Expected: 172 passed, 2 skipped, 2 xfailed.

- [ ] **Step 6: Commit**

```bash
git add meshwell/structured/plan.py tests/structured/test_plan.py
git commit -m "feat(structured): add _collect_stack_boundaries helper (Step B)"
```

---

## Task 5: `_planar_arrangement` (Step C)

Compute the union of stack boundaries, polygonize for arrangement faces, and extract arrangement edges (maximal vertex-runs between arrangement nodes).

**Files:**
- Modify: `meshwell/structured/plan.py`
- Test: `tests/structured/test_plan.py`

- [ ] **Step 1: Write failing tests**

```python
def test_planar_arrangement_single_square_one_face():
    """Single square -> 1 face, 4 edges."""
    from shapely.geometry import Polygon

    from meshwell.structured.plan import _planar_arrangement

    sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    edges, faces = _planar_arrangement([sq.boundary])
    assert len(faces) == 1
    assert abs(faces[0].polygon.area - 1.0) < 1e-9
    # Each edge appears once in the face boundary.
    assert len(faces[0].boundary) == len(edges)


def test_planar_arrangement_two_overlapping_squares_three_faces():
    """Two unit squares overlapping in [0.5,1]x[0,1] -> 3 arrangement faces."""
    from shapely.geometry import Polygon

    from meshwell.structured.plan import _planar_arrangement

    a = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]).boundary
    b = Polygon([(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)]).boundary
    edges, faces = _planar_arrangement([a, b])
    # Expect 3 faces: A-only [0,0.5]x[0,1], A∩B [0.5,1]x[0,1], B-only [1,1.5]x[0,1].
    assert len(faces) == 3
    total_area = sum(f.polygon.area for f in faces)
    assert abs(total_area - 2.0) < 1e-9  # 1 + 1 = 2 total, minus 0.5 overlap counted once
```

- [ ] **Step 2: Run tests to verify failures**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py -v --no-cov -k "planar_arrangement"`
Expected: 2 ImportError failures.

- [ ] **Step 3: Implement**

Add in `plan.py`:

```python
def _planar_arrangement(
    boundaries: list[Any],
) -> tuple[list[ArrangementEdge], list[ArrangementFace]]:
    """Build the planar arrangement from a list of boundary geometries.

    Algorithm:
      1. ``merged = unary_union(boundaries)`` — shapely inserts vertices
         at every curve crossing.
      2. ``polygonize(merged)`` — gives the arrangement faces.
      3. Extract edges: walk each face's exterior ring; consecutive vertex
         pairs are candidate edges. Dedup by canonical (sorted-by-first-
         point, then-by-second-point) coordinate tuples, since each
         internal edge appears on the boundary of exactly two faces.

    Returns (edges, faces) where each face's boundary list references the
    edge_ids in the returned edges list.
    """
    merged = unary_union(boundaries)
    raw_polygons = list(polygonize(merged))

    # Step 1: extract every consecutive (p1, p2) on every face exterior.
    # Use rounded coords to dedup against floating-point noise.
    def _key(p1, p2, ndigits=9):
        a = (round(p1[0], ndigits), round(p1[1], ndigits))
        b = (round(p2[0], ndigits), round(p2[1], ndigits))
        return (a, b) if a <= b else (b, a)

    edge_by_key: dict[tuple, int] = {}
    edges: list[ArrangementEdge] = []
    faces: list[ArrangementFace] = []

    for face_id, poly in enumerate(raw_polygons):
        coords = list(poly.exterior.coords)
        # Polygon exterior is closed: coords[-1] == coords[0].
        boundary_list: list[tuple[int, bool]] = []
        for i in range(len(coords) - 1):
            p1, p2 = coords[i], coords[i + 1]
            key = _key(p1, p2)
            if key not in edge_by_key:
                # New edge — orient by the canonical key order.
                a, b = key
                edge = ArrangementEdge(
                    edge_id=len(edges),
                    vertices=(a, b),
                    circle=None,  # arc fit happens in Step D
                )
                edge_by_key[key] = edge.edge_id
                edges.append(edge)
            edge_id = edge_by_key[key]
            # Determine traversal direction: the edge canonical orientation
            # is key[0] -> key[1]. If face walks p1 -> p2 and that equals
            # key[0] -> key[1], reversed=False; else reversed=True.
            a_round = (round(p1[0], 9), round(p1[1], 9))
            reversed_flag = a_round != edge_by_key  # placeholder; will fix in correct check below
            reversed_flag = a_round != edges[edge_id].vertices[0]
            boundary_list.append((edge_id, reversed_flag))
        faces.append(
            ArrangementFace(
                face_id=face_id,
                polygon=poly,
                boundary=boundary_list,
            )
        )

    return edges, faces
```

> **Implementer note:** the `_planar_arrangement` here treats each consecutive vertex pair as one edge — that's *polygon* edges, not the "maximal vertex-run between arrangement nodes" arrangement edges described in the spec. This is the minimal correct version that satisfies the spec's geometric contract (canonical edge per geometric segment). Coalescing consecutive collinear or arc-fitting edges happens in Step E (Task 7). Don't try to be clever here.

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py -v --no-cov -k "planar_arrangement"`
Expected: 2 passed.

- [ ] **Step 5: Full suite**

Run: `.venv/bin/python -m pytest tests/structured/ --no-cov -q 2>&1 | tail -3`
Expected: 174 passed.

- [ ] **Step 6: Commit**

```bash
git add meshwell/structured/plan.py tests/structured/test_plan.py
git commit -m "feat(structured): add _planar_arrangement helper (Step C)"
```

---

## Task 6: `_fit_arc_to_edge` (Step D)

For each `ArrangementEdge`, try to fit a `CanonicalCircle`. Replaces the edge's `circle=None` with the fitted circle if successful and at least one source slab has `identify_arcs=True`.

**Files:**
- Modify: `meshwell/structured/plan.py`
- Test: `tests/structured/test_plan.py`

- [ ] **Step 1: Write failing tests**

```python
def test_fit_arc_to_edge_line_no_arc():
    """Two-point straight line edge does not fit any circle (returns None)."""
    from meshwell.structured.plan import _fit_arc_to_edge

    vertices = ((0.0, 0.0), (1.0, 0.0))
    result = _fit_arc_to_edge(vertices, arc_tolerance=1e-3)
    assert result is None


def test_fit_arc_to_edge_circle_fits():
    """Four points on a unit circle (within tolerance) fit as an arc."""
    import math

    from meshwell.structured.plan import _fit_arc_to_edge

    # 4 vertices on R=1 at angles 0, pi/4, pi/2, 3pi/4
    vertices = tuple(
        (math.cos(math.pi * i / 4), math.sin(math.pi * i / 4)) for i in range(4)
    )
    result = _fit_arc_to_edge(vertices, arc_tolerance=1e-3)
    assert result is not None
    assert abs(result.radius - 1.0) < 1e-3
    assert abs(result.center[0]) < 1e-3
    assert abs(result.center[1]) < 1e-3


def test_fit_arc_to_edge_too_few_points():
    """Fewer than 3 points cannot define a circle — returns None."""
    from meshwell.structured.plan import _fit_arc_to_edge

    assert _fit_arc_to_edge(((0.0, 0.0),), arc_tolerance=1e-3) is None
    assert _fit_arc_to_edge(((0.0, 0.0), (1.0, 0.0)), arc_tolerance=1e-3) is None
```

- [ ] **Step 2: Run tests to verify failures**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py -v --no-cov -k "fit_arc_to_edge"`
Expected: 3 ImportError failures.

- [ ] **Step 3: Implement**

Add in `plan.py`:

```python
def _fit_arc_to_edge(
    vertices: tuple[tuple[float, float], ...],
    arc_tolerance: float,
) -> "CanonicalCircle | None":
    """Try to fit a circle through the edge's vertices.

    Returns CanonicalCircle if all vertices lie on a common circle
    within ``arc_tolerance``; else None. Requires >=3 vertices (since
    2 points underdetermine a circle).

    Uses the same circle-fitting routine that GeometryEntity uses for
    arc identification today, so the result is consistent with existing
    arc detection.
    """
    if len(vertices) < 3:
        return None

    import numpy as np

    from meshwell.geometry_entity import fit_circle_2d

    pts = np.array(vertices)
    center, radius, residual = fit_circle_2d(pts)
    if residual > arc_tolerance:
        return None
    if radius > 1e6:  # degenerate — colinear points "fit" infinite radius
        return None
    return CanonicalCircle(center=(float(center[0]), float(center[1])), radius=float(radius))
```

Add the import at the top of `plan.py`:

```python
from meshwell.structured.spec import (
    # ... existing imports ...
    ArrangementEdge,
    ArrangementFace,
    CanonicalCircle,
    StackArrangement,
)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py -v --no-cov -k "fit_arc_to_edge"`
Expected: 3 passed.

- [ ] **Step 5: Full suite**

Run: `.venv/bin/python -m pytest tests/structured/ --no-cov -q 2>&1 | tail -3`
Expected: 177 passed.

- [ ] **Step 6: Commit**

```bash
git add meshwell/structured/plan.py tests/structured/test_plan.py
git commit -m "feat(structured): add _fit_arc_to_edge helper (Step D)"
```

---

## Task 7: `_coalesce_adjacent_arcs` (Step E)

Merge `ArrangementEdge`s whose `circle` instances match within tolerance AND that share an endpoint.

**Files:**
- Modify: `meshwell/structured/plan.py`
- Test: `tests/structured/test_plan.py`

- [ ] **Step 1: Write failing tests**

```python
def test_coalesce_adjacent_arcs_merges_shared_circle():
    """Two arc edges on the same circle sharing an endpoint -> one edge."""
    import math

    from meshwell.structured import ArrangementEdge, CanonicalCircle
    from meshwell.structured.plan import _coalesce_adjacent_arcs

    circle = CanonicalCircle(center=(0.0, 0.0), radius=1.0)
    e1 = ArrangementEdge(
        edge_id=0,
        vertices=tuple((math.cos(math.pi * i / 8), math.sin(math.pi * i / 8)) for i in range(5)),  # 0..pi/2
        circle=circle,
    )
    e2 = ArrangementEdge(
        edge_id=1,
        vertices=tuple((math.cos(math.pi * (4 + i) / 8), math.sin(math.pi * (4 + i) / 8)) for i in range(5)),  # pi/2..pi
        circle=circle,
    )
    coalesced = _coalesce_adjacent_arcs([e1, e2], arc_tolerance=1e-3)
    assert len(coalesced) == 1
    # Merged vertex count = 5 + 5 - 1 (shared midpoint at pi/2) = 9
    assert len(coalesced[0].vertices) == 9


def test_coalesce_keeps_non_matching_circles_separate():
    """Two arcs with different radii are not merged."""
    from meshwell.structured import ArrangementEdge, CanonicalCircle
    from meshwell.structured.plan import _coalesce_adjacent_arcs

    e1 = ArrangementEdge(
        edge_id=0,
        vertices=((1.0, 0.0), (0.707, 0.707), (0.0, 1.0)),
        circle=CanonicalCircle(center=(0.0, 0.0), radius=1.0),
    )
    e2 = ArrangementEdge(
        edge_id=1,
        vertices=((0.0, 1.0), (-0.354, 0.354), (-0.5, 0.0)),  # quarter of R=0.5 circle
        circle=CanonicalCircle(center=(0.0, 0.0), radius=0.5),
    )
    coalesced = _coalesce_adjacent_arcs([e1, e2], arc_tolerance=1e-3)
    assert len(coalesced) == 2


def test_coalesce_lines_passthrough():
    """Line edges (circle=None) are returned unchanged."""
    from meshwell.structured import ArrangementEdge
    from meshwell.structured.plan import _coalesce_adjacent_arcs

    e1 = ArrangementEdge(edge_id=0, vertices=((0.0, 0.0), (1.0, 0.0)), circle=None)
    e2 = ArrangementEdge(edge_id=1, vertices=((1.0, 0.0), (2.0, 0.0)), circle=None)
    coalesced = _coalesce_adjacent_arcs([e1, e2], arc_tolerance=1e-3)
    assert len(coalesced) == 2
```

- [ ] **Step 2: Run tests to verify failures**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py -v --no-cov -k "coalesce"`
Expected: 3 ImportError failures.

- [ ] **Step 3: Implement**

Add in `plan.py`:

```python
def _coalesce_adjacent_arcs(
    edges: list[ArrangementEdge],
    arc_tolerance: float,
) -> list[ArrangementEdge]:
    """Merge adjacent ArrangementEdges sharing an endpoint and circle.

    Two edges merge iff:
      - both have non-None circles matching within arc_tolerance on
        (center, radius)
      - they share an endpoint (last vertex of one == first vertex of
        another, or any other endpoint pairing)

    Merging is greedy: scan all pairs, merge the first matching pair,
    repeat until no more merges possible. Line edges (circle=None) are
    not merged here.

    Output edge_ids are re-assigned to be contiguous from 0.
    """

    def _circles_match(c1: "CanonicalCircle", c2: "CanonicalCircle") -> bool:
        return (
            abs(c1.center[0] - c2.center[0]) < arc_tolerance
            and abs(c1.center[1] - c2.center[1]) < arc_tolerance
            and abs(c1.radius - c2.radius) < arc_tolerance
        )

    def _endpoints_match(p1, p2, tol=1e-9):
        return abs(p1[0] - p2[0]) < tol and abs(p1[1] - p2[1]) < tol

    def _try_merge(e1: ArrangementEdge, e2: ArrangementEdge) -> "ArrangementEdge | None":
        if e1.circle is None or e2.circle is None:
            return None
        if not _circles_match(e1.circle, e2.circle):
            return None
        # Try all 4 endpoint pairings.
        v1_start, v1_end = e1.vertices[0], e1.vertices[-1]
        v2_start, v2_end = e2.vertices[0], e2.vertices[-1]
        if _endpoints_match(v1_end, v2_start):
            merged_verts = e1.vertices + e2.vertices[1:]
        elif _endpoints_match(v1_end, v2_end):
            merged_verts = e1.vertices + e2.vertices[-2::-1]
        elif _endpoints_match(v1_start, v2_start):
            merged_verts = e1.vertices[::-1] + e2.vertices[1:]
        elif _endpoints_match(v1_start, v2_end):
            merged_verts = e2.vertices + e1.vertices[1:]
        else:
            return None
        return ArrangementEdge(edge_id=-1, vertices=merged_verts, circle=e1.circle)

    work = list(edges)
    while True:
        merged_any = False
        for i in range(len(work)):
            if merged_any:
                break
            for j in range(i + 1, len(work)):
                m = _try_merge(work[i], work[j])
                if m is not None:
                    work = work[:i] + [m] + work[i + 1:j] + work[j + 1:]
                    merged_any = True
                    break
        if not merged_any:
            break

    return [
        ArrangementEdge(edge_id=i, vertices=e.vertices, circle=e.circle)
        for i, e in enumerate(work)
    ]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py -v --no-cov -k "coalesce"`
Expected: 3 passed.

- [ ] **Step 5: Full suite**

Run: `.venv/bin/python -m pytest tests/structured/ --no-cov -q 2>&1 | tail -3`
Expected: 180 passed.

- [ ] **Step 6: Commit**

```bash
git add meshwell/structured/plan.py tests/structured/test_plan.py
git commit -m "feat(structured): add _coalesce_adjacent_arcs helper (Step E)"
```

---

## Task 8: `_assign_faces_to_slabs` (Step F)

Map each arrangement face to the slab(s) whose `resolved_footprint` contains the face's representative point.

**Files:**
- Modify: `meshwell/structured/plan.py`
- Test: `tests/structured/test_plan.py`

- [ ] **Step 1: Write failing test**

```python
def test_assign_faces_to_slabs_containment():
    """Each face assigns to the slab whose resolved_footprint contains it."""
    from shapely.geometry import Polygon

    from meshwell.structured import ArrangementFace
    from meshwell.structured.plan import _assign_faces_to_slabs
    from meshwell.structured.spec import Slab

    fp_a = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    fp_b = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
    slab_a = Slab(
        footprint=fp_a,
        resolved_footprint=fp_a,
        zlo=0.0, zhi=1.0, physical_name=("A",), source_index=0, z_interval_index=0, mesh_order=1.0,
    )
    slab_b = Slab(
        footprint=fp_b,
        resolved_footprint=fp_b,
        zlo=0.0, zhi=1.0, physical_name=("B",), source_index=1, z_interval_index=0, mesh_order=1.0,
    )

    face_in_a = ArrangementFace(face_id=0, polygon=Polygon([(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)]), boundary=[])
    face_in_b = ArrangementFace(face_id=1, polygon=Polygon([(1.1, 0.1), (1.5, 0.1), (1.5, 0.5), (1.1, 0.5)]), boundary=[])

    _assign_faces_to_slabs([face_in_a, face_in_b], [slab_a, slab_b])
    assert len(slab_a.face_partition) == 1
    assert slab_a.face_partition[0] is face_in_a.polygon
    assert len(slab_b.face_partition) == 1
    assert slab_b.face_partition[0] is face_in_b.polygon
    assert slab_a.face_partition_edges == [[]]
    assert slab_b.face_partition_edges == [[]]
```

- [ ] **Step 2: Run test to verify failure**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py -v --no-cov -k "assign_faces_to_slabs"`
Expected: ImportError failure.

- [ ] **Step 3: Implement**

Add in `plan.py`:

```python
def _assign_faces_to_slabs(
    faces: list[ArrangementFace],
    stack: list[Slab],
) -> None:
    """Set ``face_partition`` and ``face_partition_edges`` in place per containment.

    For each face, find every slab whose resolved_footprint contains
    face.polygon.representative_point(), and append the face's polygon
    to that slab's face_partition (creating the list if necessary).
    Mirrors the assignment to face_partition_edges (the face's boundary
    list of (edge_id, reversed) tuples).

    A face may be contained in zero slabs (e.g., a hole between
    resolved_footprints created by mesh_bool=False carving). Such faces
    are silently dropped.
    """
    # Initialize slab containers if not already.
    for slab in stack:
        if not slab.face_partition:
            slab.face_partition = []
        if slab.face_partition_edges is None:
            slab.face_partition_edges = []

    for face in faces:
        rep = face.polygon.representative_point()
        for slab in stack:
            fp = slab.resolved_footprint if slab.resolved_footprint is not None else slab.footprint
            if fp.contains(rep):
                slab.face_partition.append(face.polygon)
                slab.face_partition_edges.append(face.boundary)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py -v --no-cov -k "assign_faces_to_slabs"`
Expected: 1 passed.

- [ ] **Step 5: Full suite**

Run: `.venv/bin/python -m pytest tests/structured/ --no-cov -q 2>&1 | tail -3`
Expected: 181 passed.

- [ ] **Step 6: Commit**

```bash
git add meshwell/structured/plan.py tests/structured/test_plan.py
git commit -m "feat(structured): add _assign_faces_to_slabs helper (Step F)"
```

---

## Task 9: `_build_provenance_shim` (Step G)

Derive the legacy `face_partition_provenance` from the new `face_partition_edges` + arrangement, so `phantom.py` continues to work unchanged.

**Files:**
- Modify: `meshwell/structured/plan.py`
- Test: `tests/structured/test_plan.py`

- [ ] **Step 1: Write failing test**

```python
def test_build_provenance_shim_arc_and_line_edges():
    """Shim produces PieceArcEdge for arc edges and PieceLineEdge for lines."""
    import math

    from shapely.geometry import Polygon

    from meshwell.structured import ArrangementEdge, ArrangementFace, CanonicalCircle, StackArrangement
    from meshwell.structured.plan import _build_provenance_shim
    from meshwell.structured.spec import PieceArcEdge, PieceLineEdge, Slab

    arc_verts = tuple((math.cos(math.pi * i / 4), math.sin(math.pi * i / 4)) for i in range(5))  # 0..pi
    edges = [
        ArrangementEdge(edge_id=0, vertices=arc_verts, circle=CanonicalCircle(center=(0.0, 0.0), radius=1.0)),
        ArrangementEdge(edge_id=1, vertices=((-1.0, 0.0), (1.0, 0.0)), circle=None),  # diameter line
    ]
    face = ArrangementFace(
        face_id=0,
        polygon=Polygon([(1, 0), (math.cos(math.pi/4), math.sin(math.pi/4)), (-1, 0), (1, 0)]),
        boundary=[(0, False), (1, False)],  # arc then line
    )
    arrangement = StackArrangement(edges=edges, faces=[face])

    slab = Slab(
        footprint=face.polygon,
        zlo=0.0, zhi=1.0, physical_name=("X",), source_index=0, z_interval_index=0, mesh_order=1.0,
        identify_arcs=True,
    )
    slab.face_partition = [face.polygon]
    slab.face_partition_edges = [[(0, False), (1, False)]]

    _build_provenance_shim([slab], arrangement)
    assert slab.face_partition_provenance is not None
    assert len(slab.face_partition_provenance) == 1
    ext = slab.face_partition_provenance[0].exterior_edges
    assert len(ext) == 2
    assert isinstance(ext[0], PieceArcEdge)
    assert ext[0].radius == 1.0
    assert isinstance(ext[1], PieceLineEdge)
```

- [ ] **Step 2: Run test to verify failure**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py -v --no-cov -k "build_provenance_shim"`
Expected: ImportError failure.

- [ ] **Step 3: Implement**

Add in `plan.py`:

```python
def _build_provenance_shim(
    stack: list[Slab],
    arrangement: StackArrangement,
) -> None:
    """Derive face_partition_provenance from face_partition_edges + arrangement.

    Walked once per slab after assignment. The phantom builder consumes
    face_partition_provenance today (PieceArcEdge / PieceLineEdge); this
    shim builds it from the new ArrangementEdge data so phantom.py needs
    no changes.

    Only runs for slabs with identify_arcs=True; others get
    face_partition_provenance=None (matches existing behavior).
    """
    edges_by_id = {e.edge_id: e for e in arrangement.edges}

    for slab in stack:
        if not slab.identify_arcs:
            slab.face_partition_provenance = None
            continue
        if not slab.face_partition_edges or len(slab.face_partition) <= 1:
            slab.face_partition_provenance = None
            continue

        provenances: list[PieceProvenance] = []
        for piece_edges in slab.face_partition_edges:
            ext_edges = []
            for edge_id, reversed_flag in piece_edges:
                arr_edge = edges_by_id[edge_id]
                verts = arr_edge.vertices
                if reversed_flag:
                    verts = verts[::-1]
                if arr_edge.circle is not None:
                    pts_3d = tuple((v[0], v[1], 0.0) for v in verts)
                    ext_edges.append(
                        PieceArcEdge(
                            points=pts_3d,
                            center=(arr_edge.circle.center[0], arr_edge.circle.center[1], 0.0),
                            radius=arr_edge.circle.radius,
                        )
                    )
                else:
                    p1 = (verts[0][0], verts[0][1], 0.0)
                    p2 = (verts[-1][0], verts[-1][1], 0.0)
                    ext_edges.append(PieceLineEdge(points=(p1, p2)))
            provenances.append(PieceProvenance(exterior_edges=ext_edges, interior_edges=[]))
        slab.face_partition_provenance = provenances
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py -v --no-cov -k "build_provenance_shim"`
Expected: 1 passed.

- [ ] **Step 5: Full suite**

Run: `.venv/bin/python -m pytest tests/structured/ --no-cov -q 2>&1 | tail -3`
Expected: 182 passed.

- [ ] **Step 6: Commit**

```bash
git add meshwell/structured/plan.py tests/structured/test_plan.py
git commit -m "feat(structured): add _build_provenance_shim helper (Step G)"
```

---

## Task 10: `build_stack_arrangements` orchestrator

Wires Steps A through E into one function returning `dict[int, StackArrangement]` keyed by component index.

**Files:**
- Modify: `meshwell/structured/plan.py`
- Test: `tests/structured/test_plan.py`

- [ ] **Step 1: Write failing test**

```python
def test_build_stack_arrangements_disjoint_two_stacks():
    """Two non-z-touching slabs -> two independent StackArrangements."""
    from shapely.geometry import Polygon

    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec
    from meshwell.structured.plan import (
        _resolve_sublevel_mesh_order,
        build_stack_arrangements,
        expand_to_slabs,
        gather_structured_entities,
    )

    a = PolyPrism(
        polygons=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="A",
    )
    b = PolyPrism(
        polygons=Polygon([(10, 0), (11, 0), (11, 1), (10, 1)]),
        buffers={5.0: 0.0, 6.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="B",
    )
    entities = [a, b]
    slabs = expand_to_slabs(gather_structured_entities(entities))
    _resolve_sublevel_mesh_order(slabs, entities)
    arrangements = build_stack_arrangements(slabs, entities)
    assert len(arrangements) == 2
    for comp_id, arr in arrangements.items():
        assert len(arr.faces) == 1
```

- [ ] **Step 2: Run test to verify failure**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py -v --no-cov -k "build_stack_arrangements"`
Expected: ImportError failure.

- [ ] **Step 3: Implement**

Add in `plan.py`:

```python
def build_stack_arrangements(
    slabs: list[Slab],
    entities: list[Any],
) -> dict[int, StackArrangement]:
    """Build one StackArrangement per connected z-touching component.

    Assumes ``_resolve_sublevel_mesh_order`` has already populated each
    slab's ``resolved_footprint``.

    Returns dict mapping component_index -> StackArrangement.
    """
    components = _connected_z_components(slabs)
    arrangements: dict[int, StackArrangement] = {}
    for comp_idx, stack in enumerate(components):
        boundaries = _collect_stack_boundaries(stack, entities)
        if not boundaries:
            arrangements[comp_idx] = StackArrangement(edges=[], faces=[])
            continue
        edges, faces = _planar_arrangement(boundaries)
        # Determine effective arc_tolerance for this stack: minimum of
        # all member slabs that have identify_arcs=True.
        arc_tols = [s.arc_tolerance for s in stack if s.identify_arcs]
        tol = min(arc_tols) if arc_tols else 1e-3
        # Step D — try arc fit per edge (only if any stack member identifies arcs).
        if arc_tols:
            edges = [
                ArrangementEdge(
                    edge_id=e.edge_id,
                    vertices=e.vertices,
                    circle=_fit_arc_to_edge(e.vertices, tol),
                )
                for e in edges
            ]
            # Step E — coalesce adjacent arcs.
            edges = _coalesce_adjacent_arcs(edges, tol)
            # Re-emit faces with possibly-stale edge_ids — rebuild face.boundary
            # by mapping each old (edge_id, reversed) to the new edges by vertex match.
            # This is the bookkeeping cost of coalesce.
            edges, faces = _rebuild_face_boundaries(edges, faces)
        arrangements[comp_idx] = StackArrangement(edges=edges, faces=faces)
    return arrangements


def _rebuild_face_boundaries(
    new_edges: list[ArrangementEdge],
    faces: list[ArrangementFace],
) -> tuple[list[ArrangementEdge], list[ArrangementFace]]:
    """After coalesce, faces may reference old edge_ids — re-resolve them.

    For each face, walk its polygon's exterior; for each polygon-edge segment,
    find the new ArrangementEdge whose vertices contain that segment and assign
    the (edge_id, reversed_flag) accordingly.
    """
    def _segment_covered_by_edge(p1, p2, edge_verts, tol=1e-9):
        """Returns reversed flag if p1->p2 traversal lies on edge_verts (canonical) or reversed."""
        # Find p1 in edge_verts.
        for k, v in enumerate(edge_verts):
            if abs(v[0] - p1[0]) < tol and abs(v[1] - p1[1]) < tol:
                # Check the next vertex matches p2 (forward).
                if k + 1 < len(edge_verts):
                    n = edge_verts[k + 1]
                    if abs(n[0] - p2[0]) < tol and abs(n[1] - p2[1]) < tol:
                        return False
                # Check the prev vertex matches p2 (reversed).
                if k > 0:
                    p = edge_verts[k - 1]
                    if abs(p[0] - p2[0]) < tol and abs(p[1] - p2[1]) < tol:
                        return True
        return None

    new_faces: list[ArrangementFace] = []
    for face in faces:
        coords = list(face.polygon.exterior.coords)
        new_boundary: list[tuple[int, bool]] = []
        i = 0
        while i < len(coords) - 1:
            p1, p2 = coords[i], coords[i + 1]
            matched = False
            for edge in new_edges:
                rev = _segment_covered_by_edge(p1, p2, edge.vertices)
                if rev is None:
                    continue
                new_boundary.append((edge.edge_id, rev))
                # Skip ahead past all vertices we just consumed in this edge.
                edge_len = len(edge.vertices) - 1  # number of segments
                # We've consumed 1 segment; find how many more consecutive
                # polygon-segments fit this edge.
                consumed = 1
                while consumed < edge_len and i + consumed + 1 < len(coords):
                    pn, pn1 = coords[i + consumed], coords[i + consumed + 1]
                    rev2 = _segment_covered_by_edge(pn, pn1, edge.vertices)
                    if rev2 is None or rev2 != rev:
                        break
                    consumed += 1
                i += consumed
                matched = True
                break
            if not matched:
                # Shouldn't happen — fall back to a placeholder line edge.
                i += 1
        new_faces.append(
            ArrangementFace(face_id=face.face_id, polygon=face.polygon, boundary=new_boundary)
        )
    return new_edges, new_faces
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py -v --no-cov -k "build_stack_arrangements"`
Expected: 1 passed.

- [ ] **Step 5: Full suite**

Run: `.venv/bin/python -m pytest tests/structured/ --no-cov -q 2>&1 | tail -3`
Expected: 183 passed.

- [ ] **Step 6: Commit**

```bash
git add meshwell/structured/plan.py tests/structured/test_plan.py
git commit -m "feat(structured): add build_stack_arrangements orchestrator"
```

---

## Task 11: `assign_face_partition_from_arrangement` orchestrator

Drive Step F (assign) and Step G (provenance shim) over every stack.

**Files:**
- Modify: `meshwell/structured/plan.py`
- Test: `tests/structured/test_plan.py`

- [ ] **Step 1: Write failing test**

```python
def test_assign_face_partition_from_arrangement_full_pipeline():
    """End-to-end through Steps 0 + A-G on a 2-slab same-z scene."""
    from shapely.geometry import Polygon

    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec
    from meshwell.structured.plan import (
        _resolve_sublevel_mesh_order,
        assign_face_partition_from_arrangement,
        build_stack_arrangements,
        expand_to_slabs,
        gather_structured_entities,
    )

    a = PolyPrism(
        polygons=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="A",
        mesh_order=1,
    )
    b = PolyPrism(
        polygons=Polygon([(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="B",
        mesh_order=2,
    )
    entities = [a, b]
    slabs = expand_to_slabs(gather_structured_entities(entities))
    _resolve_sublevel_mesh_order(slabs, entities)
    arrangements = build_stack_arrangements(slabs, entities)
    assign_face_partition_from_arrangement(slabs, arrangements)

    by_name = {s.physical_name[0]: s for s in slabs}
    # A wins the overlap; A's resolved_footprint is the full [0,1]x[0,1].
    # B's resolved_footprint is the carved [1,1.5]x[0,1].
    # Each slab has exactly 1 face after carving.
    assert len(by_name["A"].face_partition) == 1
    assert len(by_name["B"].face_partition) == 1
```

- [ ] **Step 2: Run test to verify failure**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py -v --no-cov -k "assign_face_partition_from_arrangement_full_pipeline"`
Expected: ImportError failure.

- [ ] **Step 3: Implement**

Add in `plan.py`:

```python
def assign_face_partition_from_arrangement(
    slabs: list[Slab],
    arrangements: dict[int, StackArrangement],
) -> None:
    """Distribute arrangement faces to slabs and build provenance.

    Maps each slab to its containing component (via the same connected-
    components grouping used by build_stack_arrangements), then runs
    Step F (face assignment) and Step G (provenance shim) per stack.
    """
    components = _connected_z_components(slabs)
    for comp_idx, stack in enumerate(components):
        arrangement = arrangements.get(comp_idx)
        if arrangement is None:
            continue
        # Reset face_partition / face_partition_edges (in case slabs were
        # populated by a previous run).
        for slab in stack:
            slab.face_partition = []
            slab.face_partition_edges = []
        _assign_faces_to_slabs(arrangement.faces, stack)
        _build_provenance_shim(stack, arrangement)
        # Slabs whose face_partition is empty (fully dominated by mesh_bool=False
        # carving or by mesh_order overlap) get a one-piece fallback so phantom
        # build doesn't crash. They'll produce no actual mesh content.
        for slab in stack:
            if not slab.face_partition:
                slab.face_partition = [slab.resolved_footprint if slab.resolved_footprint is not None else slab.footprint]
                slab.face_partition_edges = [[]]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py -v --no-cov -k "assign_face_partition_from_arrangement_full_pipeline"`
Expected: 1 passed.

- [ ] **Step 5: Full suite**

Run: `.venv/bin/python -m pytest tests/structured/ --no-cov -q 2>&1 | tail -3`
Expected: 184 passed.

- [ ] **Step 6: Commit**

```bash
git add meshwell/structured/plan.py tests/structured/test_plan.py
git commit -m "feat(structured): add assign_face_partition_from_arrangement orchestrator"
```

---

## Task 12: Migrate `validate_and_resolve_overlap` to non-dropping

Today's Policy B drops the overlap loser; the new model carves it via Step 0. Update `validate_and_resolve_overlap` to record overlap pairs but keep all slabs in the returned list (carving happens at Step 0 in Task 13's wiring).

**Files:**
- Modify: `meshwell/structured/plan.py` lines around the existing `validate_and_resolve_overlap`
- Modify: `tests/structured/test_plan.py` — update `test_valid_overlap_drops_loser_records_pair`

- [ ] **Step 1: Modify `validate_and_resolve_overlap`**

In `meshwell/structured/plan.py`, find `validate_and_resolve_overlap`. Locate the loop that drops dominated slabs. Replace the `dominated` branch so it records the pair but does NOT skip appending:

```python
        if not dominated:
            kept_indices.append(idx)
        else:
            # Old behavior dropped this slab. New behavior keeps all slabs;
            # the loser is carved in Step 0 (_resolve_sublevel_mesh_order).
            kept_indices.append(idx)
```

(The simpler refactor: just remove the `if not dominated:` gate entirely so every slab is kept.)

- [ ] **Step 2: Update breaking test**

In `tests/structured/test_plan.py`, find `test_valid_overlap_drops_loser_records_pair`. Update it:

```python
def test_valid_overlap_records_pair_carve_semantics():
    """Volumetric overlap with matching z/n_layers: both slabs kept, OverlapPair recorded.

    Updated 2026-05-25: old "drop the loser" behavior replaced by "carve the
    loser by the winner" in Step 0. Plan still records the OverlapPair for
    diagnostics. Carving is applied via _resolve_sublevel_mesh_order at
    build_plan time.
    """
    from shapely.geometry import Polygon

    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan

    a = PolyPrism(
        polygons=Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="A",
        mesh_order=1,
    )
    b = PolyPrism(
        polygons=Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="B",
        mesh_order=2,
    )
    plan = build_plan([a, b])
    # Both A and B appear in the plan now (B carved by A).
    names = {s.physical_name[0] for s in plan.slabs}
    assert names == {"A", "B"}
    # OverlapPair still recorded.
    assert len(plan.overlaps) == 1
```

- [ ] **Step 3: Run the updated test**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py::test_valid_overlap_records_pair_carve_semantics -v --no-cov`
Expected: 1 passed.

- [ ] **Step 4: Run full structured suite — expect some failures from other tests asserting drop behavior**

Run: `.venv/bin/python -m pytest tests/structured/ --no-cov -q 2>&1 | tail -15`
Expected: Possibly 0-2 failures in other tests that asserted on drop semantics. Read the failures.

- [ ] **Step 5: For each remaining failure asserting drop behavior, update to carve semantics**

Read the failing test, change `assert len(plan.slabs) == 1` to `== 2` (or similar), update assertions. Keep going until full structured suite is green.

- [ ] **Step 6: Full suite green**

Run: `.venv/bin/python -m pytest tests/structured/ --no-cov -q 2>&1 | tail -3`
Expected: all passed (count may differ from 184 if we updated other tests).

- [ ] **Step 7: Commit**

```bash
git add meshwell/structured/plan.py tests/structured/test_plan.py
git commit -m "feat(structured): replace drop-loser overlap semantics with carve-loser

The new arrangement pipeline (Step 0) carves the loser's resolved_footprint
by the winner's, instead of dropping the slab entirely. Updates Policy B
overlap handling and migrates the drop-loser test to carve semantics."
```

---

## Task 13: Wire new pipeline into `build_plan`

Replace the `compute_face_partition(kept_slabs, entities)` call with the new four-step pipeline.

**Files:**
- Modify: `meshwell/structured/plan.py` — the `build_plan` function

- [ ] **Step 1: Find the existing `build_plan` and locate the call to `compute_face_partition`**

Look in `meshwell/structured/plan.py` for `def build_plan(`. Find the line `compute_face_partition(kept_slabs, entities)`.

- [ ] **Step 2: Replace with the new pipeline**

Replace that single line with:

```python
    _resolve_sublevel_mesh_order(kept_slabs, entities)
    arrangements = build_stack_arrangements(kept_slabs, entities)
    assign_face_partition_from_arrangement(kept_slabs, arrangements)
```

- [ ] **Step 3: Run full structured suite — expect failures from arc-related tests and possibly others**

Run: `.venv/bin/python -m pytest tests/structured/ --no-cov 2>&1 | tail -30`
Expected: Some tests fail because they assert on specific data structures or piece counts that the new pipeline produces differently. The two arc xfails should now potentially pass (or fail with different errors).

- [ ] **Step 4: For each failure, classify**

Read each failure:
- If it's an arc test passing now → flip xfail (Task 14).
- If it's a piece-count assertion that's now valid in a different way → update.
- If it's a true regression (the new pipeline produces incorrect output) → STOP and re-examine the implementation of Tasks 2–11.

Update failing tests inline as you go.

- [ ] **Step 5: Verify both arc xfails now report XPASS(strict)**

Run: `.venv/bin/python -m pytest tests/structured/test_stress_stacked_patterns.py::test_stacked_concentric_arc_discs_mesh_clean tests/structured/test_stress_stacked_patterns.py::test_stacked_overlapping_ring_segments_mesh_clean -v --no-cov 2>&1 | tail -10`
Expected: Both XPASS(strict) (= they now pass, breaking the strict xfail). Task 14 flips them.

- [ ] **Step 6: Full suite — expect only the 2 XPASS(strict) failures**

Run: `.venv/bin/python -m pytest tests/structured/ --no-cov -q 2>&1 | tail -5`
Expected: 2 failed (the xpass-strict markers), all others passed.

- [ ] **Step 7: Commit**

```bash
git add meshwell/structured/plan.py tests/structured/
git commit -m "feat(structured): wire planar-arrangement pipeline into build_plan

Replaces compute_face_partition with the new four-step pipeline:
_resolve_sublevel_mesh_order, build_stack_arrangements, and
assign_face_partition_from_arrangement. Existing tests updated for
any behavior changes. Two arc xfails now report XPASS strict;
Task 14 flips them."
```

---

## Task 14: Flip both arc xfails

**Files:**
- Modify: `tests/structured/test_stress_stacked_patterns.py`

- [ ] **Step 1: Remove the xfail decorator from `test_stacked_concentric_arc_discs_mesh_clean`**

In `tests/structured/test_stress_stacked_patterns.py`, locate the test and remove the entire `@pytest.mark.xfail(...)` block above the function definition.

- [ ] **Step 2: Remove the xfail decorator from `test_stacked_overlapping_ring_segments_mesh_clean`**

Same — locate and remove the decorator block.

- [ ] **Step 3: Run both tests directly**

Run: `.venv/bin/python -m pytest tests/structured/test_stress_stacked_patterns.py::test_stacked_concentric_arc_discs_mesh_clean tests/structured/test_stress_stacked_patterns.py::test_stacked_overlapping_ring_segments_mesh_clean -v --no-cov`
Expected: 2 passed.

- [ ] **Step 4: Full suite**

Run: `.venv/bin/python -m pytest tests/structured/ --no-cov -q 2>&1 | tail -3`
Expected: All passed (no xfails remain).

- [ ] **Step 5: Commit**

```bash
git add tests/structured/test_stress_stacked_patterns.py
git commit -m "test(structured): flip both arc-stress xfails to passing

The planar-arrangement pipeline correctly handles arc-bearing stacked
slabs (concentric discs and overlapping ring segments). Both previously-
xfailed tests now pass cleanly."
```

---

## Task 15: Delete old machinery

Now that the new pipeline is in place and tests are green, delete `compute_face_partition` and the supporting helpers that no longer have any callers.

**Files:**
- Modify: `meshwell/structured/plan.py` — remove unused functions and constants
- Modify: `meshwell/structured/spec.py` — remove `StructuredPartitionConvergenceError`
- Modify: `meshwell/structured/__init__.py` — remove the export
- Modify: `tests/structured/test_plan.py` — remove tests for deleted machinery

- [ ] **Step 1: Identify dead code**

In `meshwell/structured/plan.py`, search for and prepare to delete:

- `compute_face_partition` (the whole function)
- `_partition_pieces_for_slab`
- `_attach_face_partition_provenance`
- `_collect_cut_sources`
- `_collect_inherited_arcs`
- `_merge_arc_into_index`
- `_structured_slabs_touching_z`
- `_classify_piece_boundary`
- `_validate_arc_neighbour_alignment`
- `_interior_buffer_for_radius`
- `_build_arc_index_from_footprint`
- `_ArcIndex`
- `_IndexedArc`
- The constants `_PARTITION_FIXED_POINT_CAP` and `_LAST_PARTITION_ITERATIONS`

- [ ] **Step 2: Search for any remaining callers**

Run: `.venv/bin/python -c "import grep; from pathlib import Path
for n in ['compute_face_partition', '_partition_pieces_for_slab', '_attach_face_partition_provenance', '_collect_cut_sources', '_collect_inherited_arcs', '_merge_arc_into_index', '_structured_slabs_touching_z', '_classify_piece_boundary', '_validate_arc_neighbour_alignment', '_interior_buffer_for_radius', '_build_arc_index_from_footprint', '_ArcIndex', '_IndexedArc', '_PARTITION_FIXED_POINT_CAP', '_LAST_PARTITION_ITERATIONS', 'StructuredPartitionConvergenceError']:
    print(n)"`

Or simpler: use grep:

```bash
grep -rn "compute_face_partition\|_partition_pieces_for_slab\|_attach_face_partition_provenance\|_collect_cut_sources\|_collect_inherited_arcs\|_merge_arc_into_index\|_structured_slabs_touching_z\|_classify_piece_boundary\|_validate_arc_neighbour_alignment\|_interior_buffer_for_radius\|_build_arc_index_from_footprint\|_ArcIndex\|_IndexedArc\|_PARTITION_FIXED_POINT_CAP\|_LAST_PARTITION_ITERATIONS\|StructuredPartitionConvergenceError" meshwell/ tests/
```

Note any references outside `meshwell/structured/plan.py` and tests. If any production code in `meshwell/structured/builder.py` or `meshwell/structured/phantom.py` still references these, STOP and reconsider — those need their own migration.

- [ ] **Step 3: Delete in plan.py**

Edit `meshwell/structured/plan.py` and remove each function / constant / class identified in Step 1. Keep `_neighbours_touching_z` (still used; unstructured-entity helper).

- [ ] **Step 4: Remove `StructuredPartitionConvergenceError`**

In `meshwell/structured/spec.py`, delete the class.

In `meshwell/structured/__init__.py`, remove `StructuredPartitionConvergenceError` from the import block and `__all__`.

- [ ] **Step 5: Remove tests for deleted machinery**

In `tests/structured/test_plan.py`, delete:

- `test_partition_fixed_point_cap_is_module_constant`
- `test_partition_converges_within_K_plus_two_passes`
- `test_partition_raises_if_not_converged`
- `test_structured_partition_convergence_error_is_runtime_error` (in `test_spec.py`)
- `test_partition_propagates_cut_two_steps`
- `test_partition_misaligned_seams_each_slab_partitioned_by_union`
- `test_partition_fixed_point_cap_is_module_constant`
- `test_structured_slabs_touching_z_returns_zlo_zhi_matches`
- `test_merge_arc_into_index_appends_arc_and_indexes_vertices`
- `test_collect_cut_sources_uses_slab_pieces_not_footprints`
- `test_collect_inherited_arcs_pulls_from_neighbour_provenance`
- `test_collect_inherited_arcs_skips_when_identify_arcs_false`

Any other test that directly imports the deleted symbols.

- [ ] **Step 6: Run full suite**

Run: `.venv/bin/python -m pytest tests/structured/ --no-cov -q 2>&1 | tail -5`
Expected: All passed; fewer tests total but no regressions.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor(structured): delete legacy compute_face_partition machinery

After the planar-arrangement pipeline took over, the fixed-point
iteration and its supporting helpers are dead code. Removes:
- compute_face_partition
- _partition_pieces_for_slab, _attach_face_partition_provenance
- _collect_cut_sources, _collect_inherited_arcs
- _merge_arc_into_index, _structured_slabs_touching_z
- _classify_piece_boundary, _validate_arc_neighbour_alignment
- _interior_buffer_for_radius, _build_arc_index_from_footprint
- _ArcIndex, _IndexedArc, the convergence cap + counter constants
- StructuredPartitionConvergenceError (never shipped publicly)
- All tests for the above

Production code in phantom.py and builder.py continues to work via
the face_partition_provenance shim."
```

---

## Task 16: Add the comprehensive stress test

The spec's load-bearing regression target.

**Files:**
- Modify: `tests/structured/test_stress_stacked_patterns.py`

- [ ] **Step 1: Append the new test**

Append to `tests/structured/test_stress_stacked_patterns.py`:

```python
def _disc(cx, cy, r, n=24):
    """Polygon disc on a global angular grid with n vertices."""
    import math
    return Polygon(
        [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]
    )


def test_complex_disjoint_arc_stacks_with_keep_mix_mesh_clean(tmp_path):
    """Disjoint multi-layer stacks, mixed arc geometry, mesh_order + mesh_bool=False mix.

    Stack A at z=[0,3], three sub-levels with rotating half-annuli and discs.
    Stack B at z=[10,12], two sub-levels with nested concentric discs.

    Exercises:
      - Two disjoint connected components (verified by build_stack_arrangements).
      - Multiple structured polyprisms per sub-level resolved by mesh_order.
      - mesh_bool=False entities carving kept neighbours.
      - identify_arcs=True with non-aligned arc patterns between sub-levels.
      - The planar-arrangement pipeline producing a clean wedge mesh
        with zero tetrahedra in the structured regions and zero orphan
        boundary triangles.
    """
    pytest.importorskip("meshio")
    pytest.importorskip("gmsh")
    import math

    import meshio
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    def _ring(theta_start, theta_end, r_in, r_out, zlo, zhi, name, mesh_order=1):
        return PolyPrism(
            polygons=_ring_segment(0, 0, r_in, r_out, theta_start, theta_end),
            buffers={zlo: 0.0, zhi: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
            identify_arcs=True,
            min_arc_points=4,
            arc_tolerance=1e-3,
            physical_name=name,
            mesh_order=mesh_order,
        )

    def _disc_slab(cx, cy, r, zlo, zhi, name, mesh_order=1):
        return PolyPrism(
            polygons=_disc(cx, cy, r),
            buffers={zlo: 0.0, zhi: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
            identify_arcs=True,
            min_arc_points=4,
            arc_tolerance=1e-3,
            physical_name=name,
            mesh_order=mesh_order,
        )

    def _void_box(x0, y0, x1, y1, zlo, zhi, name):
        return PolyPrism(
            polygons=_box(x0, y0, x1, y1),
            buffers={zlo: 0.0, zhi: 0.0},
            mesh_bool=False,
            physical_name=name,
        )

    # Stack A: z in [0,3], three sub-levels.
    stack_a = [
        # Sub-level [0,1]: half-annulus + smaller disc inset (mesh_order=2 winner inside the annulus).
        _ring(0.0, math.pi, 0.5, 1.0, 0.0, 1.0, "A1_main", mesh_order=1),
        _disc_slab(0.5, 0.4, 0.3, 0.0, 1.0, "A1_inset", mesh_order=2),
        _void_box(-0.3, -0.3, -0.1, -0.1, 0.0, 1.0, "A1_void_tag"),

        # Sub-level [1,2]: rotated half-annulus + disc inset at different center.
        _ring(math.pi / 2, 3 * math.pi / 2, 0.5, 1.0, 1.0, 2.0, "A2_main", mesh_order=1),
        _disc_slab(0.4, 0.5, 0.3, 1.0, 2.0, "A2_inset", mesh_order=2),
        _void_box(0.1, -0.3, 0.3, -0.1, 1.0, 2.0, "A2_void_tag"),

        # Sub-level [2,3]: another rotation.
        _ring(math.pi, 2 * math.pi, 0.5, 1.0, 2.0, 3.0, "A3_main", mesh_order=1),
        _void_box(-0.3, 0.1, -0.1, 0.3, 2.0, 3.0, "A3_void_tag"),
    ]

    # Stack B: z in [10,12], two sub-levels with concentric discs.
    stack_b = [
        _disc_slab(0, 0, 0.7, 10.0, 11.0, "B1_outer", mesh_order=2),
        _disc_slab(0, 0, 0.3, 10.0, 11.0, "B1_inner", mesh_order=1),
        _disc_slab(0, 0, 0.5, 11.0, 12.0, "B2_disc", mesh_order=1),
    ]

    entities = stack_a + stack_b

    out = tmp_path / "complex_stress.msh"
    generate_mesh(entities, dim=3, output_mesh=out, default_characteristic_length=0.3)

    m = meshio.read(out)
    cell_types = {cb.type for cb in m.cells}
    # Wedges must be produced; zero tets in the structured regions means
    # the structured pipeline didn't fall back. (Caps tagged via void boxes
    # may produce tets at boundary surfaces — we check zero orphan triangles
    # instead of zero tets, since tet-tagging surfaces are normal.)
    assert any(ct in cell_types for ct in ("wedge", "wedge6", "wedge15")), (
        f"expected wedge cells, got {cell_types}"
    )

    # All meshable physicals (mesh_bool=True) appear.
    for name in (
        "A1_main", "A1_inset",
        "A2_main", "A2_inset",
        "A3_main",
        "B1_outer", "B1_inner",
        "B2_disc",
    ):
        assert name in m.field_data, f"missing physical {name}"

    orphans = _count_orphan_triangles(m)
    assert orphans == 0, (
        f"{orphans} non-conformal boundary triangles in complex stress scene"
    )
```

- [ ] **Step 2: Run the test**

Run: `.venv/bin/python -m pytest tests/structured/test_stress_stacked_patterns.py::test_complex_disjoint_arc_stacks_with_keep_mix_mesh_clean -v --no-cov`
Expected: PASS. If it fails, the planar-arrangement pipeline has an issue with this scene; diagnose and fix.

- [ ] **Step 3: Full suite**

Run: `.venv/bin/python -m pytest tests/structured/ --no-cov -q 2>&1 | tail -3`
Expected: All passed.

- [ ] **Step 4: Commit**

```bash
git add tests/structured/test_stress_stacked_patterns.py
git commit -m "test(structured): comprehensive stress test for planar-arrangement

Two disjoint multi-layer stacks combining:
- mesh_order carving at the sub-level
- mesh_bool=False void tags
- arc-bearing polyprisms (half-annuli, discs, concentric)
- non-aligned arc patterns across sub-levels

Exercises every dimension the new architecture is designed to handle."
```

---

## Task 17: Full-suite regression check

**Files:** none — verification only.

- [ ] **Step 1: Run the full structured directory**

Run: `.venv/bin/python -m pytest tests/structured/ --no-cov -q 2>&1 | tail -8`
Expected: All passed, no xfails, no skips beyond the gmsh/meshio import guards.

- [ ] **Step 2: Run the broader suite**

Run: `.venv/bin/python -m pytest tests/ --no-cov -q 2>&1 | tail -15`
Expected: Pre-existing failures (tolerance hierarchy, etc.) persist unchanged. No NEW failures introduced by this refactor.

- [ ] **Step 3: If any new regression appears, classify and fix**

For each new failure compared to baseline:
- Read the test and traceback.
- Determine whether the new behavior is the legitimate refined output of the planar-arrangement pipeline or a real bug.
- Fix forward (update assertion or fix the implementation) and re-run.

- [ ] **Step 4: Final commit if any fixes were made**

```bash
git add -A
git commit -m "test(structured): post-arrangement regression fixes"
```

---

## Self-Review

**Spec coverage check:**

- New dataclasses (`CanonicalCircle`, `ArrangementEdge`, `ArrangementFace`, `StackArrangement`) + Slab fields → Task 1 ✓
- Step 0 sub-level mesh-order carving → Task 2 ✓
- Step A connected components → Task 3 ✓
- Step B boundary collection → Task 4 ✓
- Step C planar arrangement → Task 5 ✓
- Step D arc fit per edge → Task 6 ✓
- Step E coalesce adjacent arcs → Task 7 ✓
- Step F face assignment → Task 8 ✓
- Step G provenance shim → Task 9 ✓
- Orchestrators → Tasks 10, 11 ✓
- Policy B migration → Task 12 ✓
- build_plan wiring → Task 13 ✓
- Xfail flips → Task 14 ✓
- Old machinery deletion → Task 15 ✓
- Comprehensive stress test → Task 16 ✓
- Final regression → Task 17 ✓

**Placeholder scan:** None. Every step has concrete code.

**Type consistency check:**
- `_resolve_sublevel_mesh_order(slabs, entities)`: defined in Task 2, called in Task 11 (orchestrator) and Task 13 (build_plan wiring). ✓
- `_connected_z_components(slabs)`: defined in Task 3, called in Task 10 (build_stack_arrangements) and Task 11 (assign_face_partition_from_arrangement). ✓
- `_collect_stack_boundaries(stack, entities)`: defined in Task 4, called in Task 10. ✓
- `_planar_arrangement(boundaries)`: defined in Task 5, called in Task 10. ✓
- `_fit_arc_to_edge(vertices, arc_tolerance)`: defined in Task 6, called in Task 10. ✓
- `_coalesce_adjacent_arcs(edges, arc_tolerance)`: defined in Task 7, called in Task 10. ✓
- `_assign_faces_to_slabs(faces, stack)`: defined in Task 8, called in Task 11. ✓
- `_build_provenance_shim(stack, arrangement)`: defined in Task 9, called in Task 11. ✓
- `build_stack_arrangements(slabs, entities)`: defined in Task 10, called in Task 13. ✓
- `assign_face_partition_from_arrangement(slabs, arrangements)`: defined in Task 11, called in Task 13. ✓

All function signatures and types are consistent across tasks.
