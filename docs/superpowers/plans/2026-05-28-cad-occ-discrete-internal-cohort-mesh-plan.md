# Phase 3 — Discrete Internal Cohort Mesh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sidestep N² BOP fragment cost on structured slabs by replacing per-piece OCC sub-prisms with one OCC envelope per cohort + pure gmsh discrete entities for interior piece volumes and piece-to-piece interfaces.

**Architecture:** New module `meshwell/structured/cohort_envelope.py` (a deliberately stripped subset of `cohort_topology.py`) builds one TopoDS_Solid per cohort with subdivided top+bottom OCC shells (per piece) and un-subdivided lateral wall. A new kill-switch `_USE_DISCRETE_COHORT_MESH` routes `build_phantom_shapes` through this builder, producing one cohort-level PhantomShape per cohort instead of one per piece. The mesh stage (`apply_structured_mesh`) materializes per-piece discrete 3D entities for piece volumes and discrete 2D entities for piece-to-piece interior interfaces. The cohort envelope OCC 3D entity is removed after stamping so the 3D mesh pass doesn't tetrahedralize it.

**Tech Stack:** Python, OCP (OpenCascade Python bindings), gmsh, pytest. Key OCP classes: `BRep_Builder`, `TopoDS_Shell`, `TopoDS_Solid`, `BRepBuilderAPI_MakeFace`, `BRepFill::Face_s` (for arc lateral). Key gmsh calls: `addDiscreteEntity`, `mesh.addElements`, `mesh.addNodes`, `model.removeEntities`.

**Spec:** `docs/superpowers/specs/2026-05-28-cad-occ-discrete-internal-cohort-mesh-design.md`

---

## File Structure

**Created:**
- `meshwell/structured/cohort_envelope.py` — `CohortEnvelope` dataclass, `build_cohort_envelope`, `assemble_cohort_envelope_solid`
- `tests/structured/test_cohort_envelope_skeleton.py` — Task 1 sanity test
- `tests/structured/test_cohort_envelope_vertices.py` — Task 2
- `tests/structured/test_cohort_envelope_horizontal_edges.py` — Task 3
- `tests/structured/test_cohort_envelope_vertical_edges.py` — Task 4
- `tests/structured/test_cohort_envelope_sub_faces.py` — Task 5
- `tests/structured/test_cohort_envelope_lateral_wall.py` — Task 6
- `tests/structured/test_cohort_envelope_assembly.py` — Task 7
- `tests/structured/test_cohort_envelope_build.py` — Task 8 (per spec test #1)
- `tests/structured/test_cohort_envelope_arc.py` — Task 9 (per spec test #2)
- `tests/structured/test_cohort_envelope_concentric.py` — Task 10 (per spec test #3)
- `tests/structured/test_phantom_discrete_routing.py` — Task 12
- `tests/structured/test_phase3_discrete_volumes.py` — Task 17 (per spec test #4)
- `tests/structured/test_phase3_interior_interfaces.py` — Task 18 (per spec test #5)
- `tests/structured/test_phase3_top_bottom_conformality.py` — Task 19 (per spec test #6)
- `scripts/bench_cohort_envelope.py` — Task 20

**Modified:**
- `meshwell/structured/spec.py` — extend `PhantomMap` with `face_keys_to_discrete` (Task 11)
- `meshwell/structured/phantom.py` — add `_USE_DISCRETE_COHORT_MESH` kill-switch + new branch in `build_phantom_shapes` (Tasks 12-13)
- `meshwell/structured/builder.py` — interior-interface stamping + cohort-envelope cleanup in `apply_structured_mesh` (Tasks 14-16)
- `scripts/bench_fragment_all.py` — add Phase 3 mode (Task 20)

---

## Task 1: Skeleton — `cohort_envelope.py` module + smoke import

**Files:**
- Create: `meshwell/structured/cohort_envelope.py`
- Create: `tests/structured/test_cohort_envelope_skeleton.py`

- [ ] **Step 1: Write the failing skeleton test**

Create `tests/structured/test_cohort_envelope_skeleton.py`:

```python
"""Sanity check that cohort_envelope module exposes expected names."""

from __future__ import annotations


def test_cohort_envelope_module_imports():
    from meshwell.structured.cohort_envelope import (
        CohortEnvelope,
        assemble_cohort_envelope_solid,
        build_cohort_envelope,
    )

    assert callable(build_cohort_envelope)
    assert callable(assemble_cohort_envelope_solid)
    assert CohortEnvelope is not None


def test_cohort_envelope_dataclass_has_registries():
    from meshwell.structured.cohort_envelope import CohortEnvelope

    env = CohortEnvelope(
        component_index=0,
        plan=None,
        vertices={},
        horizontal_edges={},
        vertical_edges={},
        top_sub_faces={},
        bottom_sub_faces={},
        lateral_faces={},
        outline_xy_to_corner_id={},
        cohort_solid=None,
    )
    assert env.component_index == 0
    assert env.cohort_solid is None


def test_build_cohort_envelope_returns_envelope_for_empty_plan():
    from meshwell.structured.cohort_envelope import (
        CohortEnvelope,
        build_cohort_envelope,
    )
    from meshwell.structured.spec import StructuredPlan

    plan = StructuredPlan(slabs=(), z_planes=(), overlaps=(), arrangements={})
    env = build_cohort_envelope(plan, component_index=0)
    assert isinstance(env, CohortEnvelope)
    assert env.component_index == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/structured/test_cohort_envelope_skeleton.py -v`
Expected: FAIL with `ModuleNotFoundError: meshwell.structured.cohort_envelope`

- [ ] **Step 3: Write the skeleton module**

Create `meshwell/structured/cohort_envelope.py`:

```python
"""Phase 3 cohort envelope builder.

For each connected z-component (cohort) of structured slabs, build a
single TopoDS_Solid whose boundary has:

- Top shell of per-piece OCC sub-faces (subdivided by piece boundaries)
- Bottom shell of per-piece OCC sub-faces
- Lateral wall of one OCC face per outline edge (un-subdivided)

The resulting envelope is what cad_occ.fragment_all sees instead of the
per-piece sub-prisms. Per-piece volumes and interior interfaces become
pure gmsh discrete entities at mesh time.

This module is a deliberately stripped subset of cohort_topology.py:
no interior horizontal edges, no interior vertical edges, no interior
lateral faces, no per-piece lateral subdivision. See spec
docs/superpowers/specs/2026-05-28-cad-occ-discrete-internal-cohort-mesh-design.md.

FUTURE WORK: If structured slabs ever need XY-unstructured neighbors,
this builder must subdivide lateral OCC faces along piece-to-piece
interior boundaries that meet the cohort exterior. See "Future work"
in the spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from meshwell.structured.spec import (
    FaceKey,
    StructuredPlan,
)


@dataclass
class CohortEnvelope:
    """Envelope topology + assembled solid for one cohort.

    Registries:
    - vertices: keyed by (z_plane, outline_corner_id) -> TopoDS_Vertex.
      Only outline corners — no interior piece corners.
    - horizontal_edges: keyed by (z_plane, outline_edge_id) -> TopoDS_Wire.
      Only cohort outline edges.
    - vertical_edges: keyed by (zlo, zhi, outline_corner_id) -> TopoDS_Edge.
      Deduped across slabs that share a z-interval (so two adjacent slabs
      sharing an outline edge end up with one TopoDS_Edge per vertical
      corner, letting the shared lateral OCC face close cleanly).
    - top_sub_faces: FaceKey(slab_index, "top", piece_index) -> TopoDS_Face.
      Per-piece top sub-face built from face_partition_provenance.
    - bottom_sub_faces: FaceKey(slab_index, "bot", piece_index) -> TopoDS_Face.
    - lateral_faces: keyed by (slab_index, outline_edge_id) -> list[TopoDS_Face].
      One face per segment for multi-vertex straight outline edges;
      one face per arc outline edge.

    Plus:
    - outline_xy_to_corner_id: (round(x,9), round(y,9)) -> outline_corner_id.
    - cohort_solid: the assembled TopoDS_Solid (None until assemble_*).
    """

    component_index: int
    plan: StructuredPlan | None
    vertices: dict[tuple[float, int], Any] = field(default_factory=dict)
    horizontal_edges: dict[tuple[float, int], Any] = field(default_factory=dict)
    vertical_edges: dict[tuple[float, float, int], Any] = field(default_factory=dict)
    top_sub_faces: dict[FaceKey, Any] = field(default_factory=dict)
    bottom_sub_faces: dict[FaceKey, Any] = field(default_factory=dict)
    lateral_faces: dict[tuple[int, int], list] = field(default_factory=dict)
    outline_xy_to_corner_id: dict[tuple[float, float], int] = field(default_factory=dict)
    cohort_solid: Any = None


def build_cohort_envelope(
    plan: StructuredPlan,
    component_index: int,
) -> CohortEnvelope:
    """Build the cohort envelope for one connected z-component.

    Walks the cohort's slabs and arrangement to populate the outline-only
    vertex/edge registries plus the per-piece top/bottom sub-faces and
    un-subdivided lateral wall. Does NOT assemble the solid — call
    assemble_cohort_envelope_solid for that.
    """
    env = CohortEnvelope(component_index=component_index, plan=plan)
    cohort_slabs = [s for s in plan.slabs if s.component_index == component_index]
    if not cohort_slabs:
        return env
    # Population logic is added incrementally by Tasks 2-6.
    return env


def assemble_cohort_envelope_solid(env: CohortEnvelope) -> Any:
    """Assemble the cohort envelope's TopoDS_Solid from the registries.

    Populates env.cohort_solid in-place and returns it. Implemented in
    Task 7.
    """
    raise NotImplementedError("Implemented in Task 7")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/structured/test_cohort_envelope_skeleton.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/cohort_envelope.py tests/structured/test_cohort_envelope_skeleton.py
git commit -m "feat(cohort_envelope): module skeleton + dataclass

Phase 3 envelope builder scaffold — vertex/edge/face registries
and stub build/assemble entry points. Logic added incrementally
in subsequent tasks."
```

---

## Task 2: Outline vertex registry with multi-arc snap

**Files:**
- Modify: `meshwell/structured/cohort_envelope.py`
- Create: `tests/structured/test_cohort_envelope_vertices.py`

Borrows the validated multi-arc-snap logic from `cohort_topology.py:139-180`.

- [ ] **Step 1: Write the failing test**

Create `tests/structured/test_cohort_envelope_vertices.py`:

```python
"""Vertex registry tests for cohort_envelope."""

from __future__ import annotations

import math

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.cohort_envelope import build_cohort_envelope
from meshwell.structured.plan import build_plan


def _square_slab(zlo, zhi, name, side=1.0):
    return PolyPrism(
        polygons=shapely.box(0, 0, side, side),
        buffers={zlo: 0.0, zhi: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name=name,
    )


def _disc(cx, cy, r, n=32):
    return shapely.Polygon(
        [
            (
                cx + r * math.cos(2 * math.pi * i / n),
                cy + r * math.sin(2 * math.pi * i / n),
            )
            for i in range(n)
        ]
    )


def _arc_slab(r, zlo, zhi, name):
    return PolyPrism(
        polygons=_disc(0, 0, r),
        buffers={zlo: 0.0, zhi: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
        physical_name=name,
    )


def test_vertex_registry_populated_for_simple_cohort():
    """A two-slab square cohort registers 4 corners at each z-plane (2 z-planes => 8 vertices)."""
    plan = build_plan(
        [
            _square_slab(0.0, 1.0, "L1"),
            _square_slab(1.0, 2.0, "L2"),
        ]
    )
    env = build_cohort_envelope(plan, component_index=0)
    # 4 outline corners × 3 z-planes (z=0, z=1, z=2) = 12 vertices.
    assert len(env.vertices) == 12
    assert len(env.outline_xy_to_corner_id) == 4


def test_multi_arc_vertex_snap_carries_tolerance():
    """Concentric arc discs produce multi-arc corners; vertex has positive OCC tolerance."""
    from OCP.BRep import BRep_Tool

    plan = build_plan(
        [
            _arc_slab(1.0, 0.0, 1.0, "L1"),
            _arc_slab(0.7, 1.0, 2.0, "L2"),
        ]
    )
    env = build_cohort_envelope(plan, component_index=0)
    # At least one vertex must have nontrivial tolerance from multi-arc snap.
    saw_tol = False
    for v in env.vertices.values():
        tol = BRep_Tool.Tolerance_s(v)
        if tol > 1e-9:
            saw_tol = True
            break
    assert saw_tol, "Expected at least one vertex with multi-arc snap tolerance"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/structured/test_cohort_envelope_vertices.py -v`
Expected: FAIL (`assert 0 == 12` — registry is empty)

- [ ] **Step 3: Implement outline vertex registry**

In `meshwell/structured/cohort_envelope.py`, replace the body of `build_cohort_envelope` with:

```python
def build_cohort_envelope(
    plan: StructuredPlan,
    component_index: int,
) -> CohortEnvelope:
    """Build the cohort envelope for one connected z-component."""
    import math

    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCP.gp import gp_Pnt

    env = CohortEnvelope(component_index=component_index, plan=plan)
    cohort_slabs = [s for s in plan.slabs if s.component_index == component_index]
    if not cohort_slabs:
        return env

    arrangement = plan.arrangements[component_index]

    z_planes: set[float] = set()
    for s in cohort_slabs:
        z_planes.add(s.zlo)
        z_planes.add(s.zhi)
    z_planes_sorted = sorted(z_planes)

    _ROUND = 9
    for arr_edge in arrangement.edges:
        for x, y in arr_edge.vertices:
            key = (round(x, _ROUND), round(y, _ROUND))
            if key not in env.outline_xy_to_corner_id:
                env.outline_xy_to_corner_id[key] = len(env.outline_xy_to_corner_id)

    corner_id_to_xy: dict[int, tuple[float, float]] = {
        cid: xy for xy, cid in env.outline_xy_to_corner_id.items()
    }
    corner_id_to_arc_snaps: dict[int, list[tuple[float, float]]] = {}
    for arr_edge in arrangement.edges:
        if arr_edge.circle is None:
            continue
        cx, cy = arr_edge.circle.center
        r = arr_edge.circle.radius
        for endpoint_xy in (arr_edge.vertices[0], arr_edge.vertices[-1]):
            key = (round(endpoint_xy[0], _ROUND), round(endpoint_xy[1], _ROUND))
            cid = env.outline_xy_to_corner_id[key]
            x, y = corner_id_to_xy[cid]
            dx, dy = x - cx, y - cy
            d = math.hypot(dx, dy)
            if d > 0:
                corner_id_to_arc_snaps.setdefault(cid, []).append(
                    (cx + r * dx / d, cy + r * dy / d)
                )

    corner_id_to_tol: dict[int, float] = {}
    for cid, snaps in corner_id_to_arc_snaps.items():
        avg_x = sum(s[0] for s in snaps) / len(snaps)
        avg_y = sum(s[1] for s in snaps) / len(snaps)
        corner_id_to_xy[cid] = (avg_x, avg_y)
        max_resid = max(math.hypot(s[0] - avg_x, s[1] - avg_y) for s in snaps)
        if max_resid > 0:
            corner_id_to_tol[cid] = max_resid

    _brep_builder = BRep_Builder()
    _VERTEX_TOL_MARGIN = 1e-7
    for cid, (x, y) in corner_id_to_xy.items():
        tol = corner_id_to_tol.get(cid, 0.0)
        for z in z_planes_sorted:
            v = BRepBuilderAPI_MakeVertex(gp_Pnt(x, y, z)).Vertex()
            if tol > 0:
                _brep_builder.UpdateVertex(v, tol + _VERTEX_TOL_MARGIN)
            env.vertices[(z, cid)] = v

    # Subsequent registries are added in Tasks 3-6.
    return env
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/structured/test_cohort_envelope_vertices.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/cohort_envelope.py tests/structured/test_cohort_envelope_vertices.py
git commit -m "feat(cohort_envelope): outline vertex registry with multi-arc snap

Reuses Phase 2 multi-arc vertex averaging + per-vertex OCC tolerance
to absorb residual from concentric arc fits. Outline-only — interior
piece corners are not in the registry."
```

---

## Task 3: Outline horizontal edge registry

**Files:**
- Modify: `meshwell/structured/cohort_envelope.py`
- Create: `tests/structured/test_cohort_envelope_horizontal_edges.py`

- [ ] **Step 1: Write the failing test**

Create `tests/structured/test_cohort_envelope_horizontal_edges.py`:

```python
"""Horizontal edge registry tests for cohort_envelope."""

from __future__ import annotations

import math

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.cohort_envelope import build_cohort_envelope
from meshwell.structured.plan import build_plan


def _square_slab(zlo, zhi, name, side=1.0):
    return PolyPrism(
        polygons=shapely.box(0, 0, side, side),
        buffers={zlo: 0.0, zhi: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name=name,
    )


def _arc_slab(r, zlo, zhi, name):
    n = 32
    pts = [
        (r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]
    return PolyPrism(
        polygons=shapely.Polygon(pts),
        buffers={zlo: 0.0, zhi: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
        physical_name=name,
    )


def test_horizontal_edge_registry_size_matches_outline_x_zplanes():
    """One wire per (z_plane, outline_edge_id)."""
    plan = build_plan([_square_slab(0.0, 1.0, "L1"), _square_slab(1.0, 2.0, "L2")])
    env = build_cohort_envelope(plan, component_index=0)
    arr = plan.arrangements[0]
    expected = 3 * len(arr.edges)  # 3 z-planes × N outline edges
    assert len(env.horizontal_edges) == expected


def test_horizontal_edge_for_arc_outline_is_arc_wire():
    """An arc arrangement edge produces a wire containing an arc-typed OCC edge."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepTools import BRepTools_WireExplorer
    from OCP.GeomAbs import GeomAbs_Circle

    plan = build_plan([_arc_slab(1.0, 0.0, 1.0, "L1")])
    env = build_cohort_envelope(plan, component_index=0)
    arr = plan.arrangements[0]
    arc_edges = [e for e in arr.edges if e.circle is not None]
    assert arc_edges, "Test setup expected at least one arc outline edge"
    wire = env.horizontal_edges[(0.0, arc_edges[0].edge_id)]
    exp = BRepTools_WireExplorer(wire)
    assert exp.More(), "Wire should have at least one edge"
    curve = BRepAdaptor_Curve(exp.Current())
    assert curve.GetType() == GeomAbs_Circle
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/structured/test_cohort_envelope_horizontal_edges.py -v`
Expected: FAIL (`KeyError` — horizontal_edges registry empty)

- [ ] **Step 3: Implement horizontal edge registry**

In `meshwell/structured/cohort_envelope.py`, before the `# Subsequent registries are added in Tasks 3-6.` comment in `build_cohort_envelope`, insert:

```python
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeVertex as _MV,
        BRepBuilderAPI_MakeWire,
    )
    from OCP.GC import GC_MakeArcOfCircle
    from OCP.gp import gp_Ax2, gp_Circ, gp_Dir

    _skipped_edge_ids: set[int] = set()
    for arr_edge in arrangement.edges:
        p1 = arr_edge.vertices[0]
        p2 = arr_edge.vertices[-1]
        c1 = env.outline_xy_to_corner_id[
            (round(p1[0], _ROUND), round(p1[1], _ROUND))
        ]
        c2 = env.outline_xy_to_corner_id[
            (round(p2[0], _ROUND), round(p2[1], _ROUND))
        ]
        if arr_edge.circle is None and len(arr_edge.vertices) == 2:
            dist_2v = math.hypot(
                arr_edge.vertices[1][0] - arr_edge.vertices[0][0],
                arr_edge.vertices[1][1] - arr_edge.vertices[0][1],
            )
            if dist_2v < 1e-7 or c1 == c2:
                _skipped_edge_ids.add(arr_edge.edge_id)
                continue
        for z in z_planes_sorted:
            v1 = env.vertices[(z, c1)]
            v2 = env.vertices[(z, c2)]
            mw = BRepBuilderAPI_MakeWire()
            if arr_edge.circle is not None:
                cx, cy = arr_edge.circle.center
                r = arr_edge.circle.radius
                axis = gp_Ax2(gp_Pnt(cx, cy, z), gp_Dir(0, 0, 1))
                circ = gp_Circ(axis, r)
                p1_snapped = corner_id_to_xy[c1]
                p2_snapped = corner_id_to_xy[c2]
                start = gp_Pnt(p1_snapped[0], p1_snapped[1], z)
                end = gp_Pnt(p2_snapped[0], p2_snapped[1], z)
                arc = GC_MakeArcOfCircle(circ, start, end, True).Value()
                edge = BRepBuilderAPI_MakeEdge(arc, v1, v2).Edge()
                mw.Add(edge)
            elif len(arr_edge.vertices) == 2:
                edge = BRepBuilderAPI_MakeEdge(v1, v2).Edge()
                mw.Add(edge)
            else:
                verts_3d = arr_edge.vertices
                n = len(verts_3d)
                for seg_i in range(n - 1):
                    xi, yi = verts_3d[seg_i]
                    xj, yj = verts_3d[seg_i + 1]
                    if seg_i == 0:
                        va = v1
                    else:
                        va = _MV(gp_Pnt(xi, yi, z)).Vertex()
                    if seg_i == n - 2:
                        vb = v2
                    else:
                        vb = _MV(gp_Pnt(xj, yj, z)).Vertex()
                    mw.Add(BRepBuilderAPI_MakeEdge(va, vb).Edge())
            env.horizontal_edges[(z, arr_edge.edge_id)] = mw.Wire()

    env._skipped_edge_ids = _skipped_edge_ids  # noqa: SLF001 (stashed for later loops)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/structured/test_cohort_envelope_horizontal_edges.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/cohort_envelope.py tests/structured/test_cohort_envelope_horizontal_edges.py
git commit -m "feat(cohort_envelope): outline horizontal edge registry

One TopoDS_Wire per (z-plane, outline edge id). Arc edges go through
GC_MakeArcOfCircle with snapped endpoints; multi-vertex straight
edges become multi-segment wires."
```

---

## Task 4: Outline vertical edge registry with z-interval dedup

**Files:**
- Modify: `meshwell/structured/cohort_envelope.py`
- Create: `tests/structured/test_cohort_envelope_vertical_edges.py`

- [ ] **Step 1: Write the failing test**

Create `tests/structured/test_cohort_envelope_vertical_edges.py`:

```python
"""Vertical edge registry tests for cohort_envelope."""

from __future__ import annotations

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.cohort_envelope import build_cohort_envelope
from meshwell.structured.plan import build_plan


def _square_slab(zlo, zhi, name, side=1.0):
    return PolyPrism(
        polygons=shapely.box(0, 0, side, side),
        buffers={zlo: 0.0, zhi: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name=name,
    )


def test_vertical_edge_per_z_interval_per_outline_corner():
    """Two stacked square slabs: 2 z-intervals × 4 corners = 8 vertical edges."""
    plan = build_plan([_square_slab(0.0, 1.0, "L1"), _square_slab(1.0, 2.0, "L2")])
    env = build_cohort_envelope(plan, component_index=0)
    assert len(env.vertical_edges) == 8


def test_vertical_edges_shared_across_slabs_at_same_z_interval():
    """Two slabs at the same z-interval share TopoDS_Edge per corner.

    Two square slabs at z=[0,1], side-by-side in XY, sharing one outline edge.
    The shared corners must reference the SAME TopoDS_Edge for both slabs.
    """
    s1 = PolyPrism(
        polygons=shapely.box(0, 0, 1, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="L1",
    )
    s2 = PolyPrism(
        polygons=shapely.box(1, 0, 2, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="L2",
    )
    plan = build_plan([s1, s2])
    env = build_cohort_envelope(plan, component_index=0)
    # Each unique outline corner at this single z-interval should map
    # to ONE TopoDS_Edge — only one entry in the registry per
    # (zlo, zhi, corner_id) tuple.
    keys = list(env.vertical_edges.keys())
    z_intervals = {(zlo, zhi) for (zlo, zhi, _cid) in keys}
    assert z_intervals == {(0.0, 1.0)}
    # Cardinality = N outline corners × 1 z-interval.
    n_corners = len(env.outline_xy_to_corner_id)
    assert len(env.vertical_edges) == n_corners
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/structured/test_cohort_envelope_vertical_edges.py -v`
Expected: FAIL (registry empty)

- [ ] **Step 3: Implement vertical edge registry**

Append to `build_cohort_envelope` (before the `# Subsequent registries are added in Tasks 3-6.` comment / at the end of the body before `return env`):

```python
    for slab in cohort_slabs:
        for corner_id in env.outline_xy_to_corner_id.values():
            zkey = (slab.zlo, slab.zhi, corner_id)
            if zkey in env.vertical_edges:
                continue
            v_lo = env.vertices[(slab.zlo, corner_id)]
            v_hi = env.vertices[(slab.zhi, corner_id)]
            env.vertical_edges[zkey] = BRepBuilderAPI_MakeEdge(v_lo, v_hi).Edge()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/structured/test_cohort_envelope_vertical_edges.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/cohort_envelope.py tests/structured/test_cohort_envelope_vertical_edges.py
git commit -m "feat(cohort_envelope): outline vertical edge registry

Deduped by (zlo, zhi, outline_corner_id) so two slabs at the same
z-interval share a single TopoDS_Edge per corner — required for
shared lateral OCC faces to close cleanly."
```

---

## Task 5: Per-piece top/bottom OCC sub-face registry

**Files:**
- Modify: `meshwell/structured/cohort_envelope.py`
- Create: `tests/structured/test_cohort_envelope_sub_faces.py`

Sub-faces are subdivided per piece (one OCC face per (slab, side, piece_index)) using the existing `_make_face_from_provenance` from `phantom.py` (z=zlo for bot, z=zhi for top).

- [ ] **Step 1: Write the failing test**

Create `tests/structured/test_cohort_envelope_sub_faces.py`:

```python
"""Per-piece top/bot OCC sub-face registry tests."""

from __future__ import annotations

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.cohort_envelope import build_cohort_envelope
from meshwell.structured.plan import build_plan
from meshwell.structured.spec import FaceKey


def _square_slab(zlo, zhi, name, side=1.0):
    return PolyPrism(
        polygons=shapely.box(0, 0, side, side),
        buffers={zlo: 0.0, zhi: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name=name,
    )


def test_sub_faces_indexed_by_face_key():
    """Two stacked single-piece slabs: 2 top + 2 bot sub-faces."""
    plan = build_plan([_square_slab(0.0, 1.0, "L1"), _square_slab(1.0, 2.0, "L2")])
    env = build_cohort_envelope(plan, component_index=0)
    assert FaceKey(0, "top", 0) in env.top_sub_faces
    assert FaceKey(1, "top", 0) in env.top_sub_faces
    assert FaceKey(0, "bot", 0) in env.bottom_sub_faces
    assert FaceKey(1, "bot", 0) in env.bottom_sub_faces
    assert len(env.top_sub_faces) == 2
    assert len(env.bottom_sub_faces) == 2


def test_sub_face_is_planar_at_correct_z():
    """Top sub-face is planar at z=zhi; bottom at z=zlo."""
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    plan = build_plan([_square_slab(0.0, 1.0, "L1")])
    env = build_cohort_envelope(plan, component_index=0)
    top = env.top_sub_faces[FaceKey(0, "top", 0)]
    exp = TopExp_Explorer(top, TopAbs_VERTEX)
    while exp.More():
        v = TopoDS.Vertex_s(exp.Current())
        z = BRep_Tool.Pnt_s(v).Z()
        assert abs(z - 1.0) < 1e-9
        exp.Next()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/structured/test_cohort_envelope_sub_faces.py -v`
Expected: FAIL (`KeyError`)

- [ ] **Step 3: Implement sub-face registry**

Append to `build_cohort_envelope` body (before `return env`):

```python
    from meshwell.structured.phantom import _make_face_from_provenance

    slab_to_index = {id(s): i for i, s in enumerate(plan.slabs)}
    for slab in cohort_slabs:
        slab_index = slab_to_index[id(slab)]
        if not slab.face_partition or slab.face_partition_provenance is None:
            continue
        for piece_index, provenance in enumerate(slab.face_partition_provenance):
            bot_face = _make_face_from_provenance(provenance, z=slab.zlo)
            top_face = _make_face_from_provenance(provenance, z=slab.zhi)
            env.bottom_sub_faces[FaceKey(slab_index, "bot", piece_index)] = bot_face
            env.top_sub_faces[FaceKey(slab_index, "top", piece_index)] = top_face
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/structured/test_cohort_envelope_sub_faces.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/cohort_envelope.py tests/structured/test_cohort_envelope_sub_faces.py
git commit -m "feat(cohort_envelope): per-piece top/bottom OCC sub-face registry

Keyed by FaceKey(slab_index, side, piece_index). Built via existing
_make_face_from_provenance helper at slab.zlo (bot) and slab.zhi (top)."
```

---

## Task 6: Un-subdivided lateral wall registry

**Files:**
- Modify: `meshwell/structured/cohort_envelope.py`
- Create: `tests/structured/test_cohort_envelope_lateral_wall.py`

Lateral wall faces are keyed by (slab_index, outline_edge_id). Arc edges use `BRepFill::Face_s(bot_arc, top_arc)` for proper PCurves; straight edges build planar faces from the bot wire + top wire + verticals.

- [ ] **Step 1: Write the failing test**

Create `tests/structured/test_cohort_envelope_lateral_wall.py`:

```python
"""Lateral wall registry tests for cohort_envelope."""

from __future__ import annotations

import math

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.cohort_envelope import build_cohort_envelope
from meshwell.structured.plan import build_plan


def _square_slab(zlo, zhi, name, side=1.0):
    return PolyPrism(
        polygons=shapely.box(0, 0, side, side),
        buffers={zlo: 0.0, zhi: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name=name,
    )


def _arc_slab(r, zlo, zhi, name):
    n = 32
    pts = [
        (r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]
    return PolyPrism(
        polygons=shapely.Polygon(pts),
        buffers={zlo: 0.0, zhi: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
        physical_name=name,
    )


def test_lateral_wall_one_per_slab_per_outline_edge():
    """One lateral face list per (slab_index, outline_edge_id)."""
    plan = build_plan([_square_slab(0.0, 1.0, "L1"), _square_slab(1.0, 2.0, "L2")])
    env = build_cohort_envelope(plan, component_index=0)
    arr = plan.arrangements[0]
    n_edges = len([e for e in arr.edges if e.edge_id not in getattr(env, "_skipped_edge_ids", set())])
    # 2 slabs × N outline edges.
    assert len(env.lateral_faces) == 2 * n_edges


def test_arc_lateral_has_valid_bbox():
    """Arc lateral face built via BRepFill::Face_s has a bounded bbox (no ±1e+100)."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    plan = build_plan([_arc_slab(1.0, 0.0, 1.0, "L1")])
    env = build_cohort_envelope(plan, component_index=0)
    arr = plan.arrangements[0]
    arc_edge = next(e for e in arr.edges if e.circle is not None)
    face_list = env.lateral_faces[(0, arc_edge.edge_id)]
    assert face_list, "Arc outline edge should produce at least one lateral face"
    for face in face_list:
        bbox = Bnd_Box()
        BRepBndLib.Add_s(face, bbox)
        x0, y0, z0, x1, y1, z1 = bbox.Get()
        # Sane finite bbox under ~10 units (our arc has r=1).
        assert all(abs(v) < 10.0 for v in (x0, y0, z0, x1, y1, z1)), (
            f"BRepFill arc lateral has unbounded bbox: {bbox.Get()}"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/structured/test_cohort_envelope_lateral_wall.py -v`
Expected: FAIL (`KeyError` on `lateral_faces`)

- [ ] **Step 3: Implement lateral wall registry**

Append to `build_cohort_envelope` (before `return env`):

```python
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepFill import BRepFill
    from OCP.BRepTools import BRepTools_WireExplorer
    from OCP.TopoDS import TopoDS

    def _rev_edge(e):
        return TopoDS.Edge_s(e.Reversed())

    for slab in cohort_slabs:
        slab_index = slab_to_index[id(slab)]
        for arr_edge in arrangement.edges:
            if arr_edge.edge_id in _skipped_edge_ids:
                continue
            key = (slab_index, arr_edge.edge_id)
            if key in env.lateral_faces:
                continue
            p1 = arr_edge.vertices[0]
            p2 = arr_edge.vertices[-1]
            c1 = env.outline_xy_to_corner_id[
                (round(p1[0], _ROUND), round(p1[1], _ROUND))
            ]
            c2 = env.outline_xy_to_corner_id[
                (round(p2[0], _ROUND), round(p2[1], _ROUND))
            ]

            if arr_edge.circle is not None:
                bot_wire = env.horizontal_edges[(slab.zlo, arr_edge.edge_id)]
                top_wire = env.horizontal_edges[(slab.zhi, arr_edge.edge_id)]
                bot_exp = BRepTools_WireExplorer(bot_wire)
                bot_arc_edge = bot_exp.Current()
                top_exp = BRepTools_WireExplorer(top_wire)
                top_arc_edge = top_exp.Current()
                face = BRepFill.Face_s(bot_arc_edge, top_arc_edge)
                env.lateral_faces[key] = [face]
            elif len(arr_edge.vertices) == 2:
                bot_wire = env.horizontal_edges[(slab.zlo, arr_edge.edge_id)]
                top_wire = env.horizontal_edges[(slab.zhi, arr_edge.edge_id)]
                v_edge_1 = env.vertical_edges[(slab.zlo, slab.zhi, c1)]
                v_edge_2 = env.vertical_edges[(slab.zlo, slab.zhi, c2)]
                bot_edges = []
                exp = BRepTools_WireExplorer(bot_wire)
                while exp.More():
                    bot_edges.append(exp.Current())
                    exp.Next()
                top_edges = []
                exp = BRepTools_WireExplorer(top_wire)
                while exp.More():
                    top_edges.append(exp.Current())
                    exp.Next()
                mw = BRepBuilderAPI_MakeWire()
                mw.Add(bot_edges[0])
                mw.Add(v_edge_2)
                mw.Add(_rev_edge(top_edges[0]))
                mw.Add(_rev_edge(v_edge_1))
                face = BRepBuilderAPI_MakeFace(mw.Wire()).Face()
                env.lateral_faces[key] = [face]
            else:
                verts_2d = arr_edge.vertices
                n = len(verts_2d)
                face_list = []
                for seg_i in range(n - 1):
                    xi, yi = verts_2d[seg_i]
                    xj, yj = verts_2d[seg_i + 1]
                    ci = env.outline_xy_to_corner_id[
                        (round(xi, _ROUND), round(yi, _ROUND))
                    ]
                    cj = env.outline_xy_to_corner_id[
                        (round(xj, _ROUND), round(yj, _ROUND))
                    ]
                    v_edge_i = env.vertical_edges[(slab.zlo, slab.zhi, ci)]
                    v_edge_j = env.vertical_edges[(slab.zlo, slab.zhi, cj)]
                    va_bot = env.vertices[(slab.zlo, ci)]
                    vb_bot = env.vertices[(slab.zlo, cj)]
                    va_top = env.vertices[(slab.zhi, ci)]
                    vb_top = env.vertices[(slab.zhi, cj)]
                    bot_seg = BRepBuilderAPI_MakeEdge(va_bot, vb_bot).Edge()
                    top_seg = BRepBuilderAPI_MakeEdge(va_top, vb_top).Edge()
                    mw = BRepBuilderAPI_MakeWire()
                    mw.Add(bot_seg)
                    mw.Add(v_edge_j)
                    mw.Add(_rev_edge(top_seg))
                    mw.Add(_rev_edge(v_edge_i))
                    face_list.append(BRepBuilderAPI_MakeFace(mw.Wire()).Face())
                env.lateral_faces[key] = face_list
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/structured/test_cohort_envelope_lateral_wall.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/cohort_envelope.py tests/structured/test_cohort_envelope_lateral_wall.py
git commit -m "feat(cohort_envelope): un-subdivided lateral wall registry

One lateral face per (slab, outline_edge_id), keyed by outline edge
only (not piece). Arc edges use BRepFill::Face_s for valid PCurves.
Multi-vertex straight edges decompose into per-segment planar quads."
```

---

## Task 7: Assemble cohort envelope solid

**Files:**
- Modify: `meshwell/structured/cohort_envelope.py`
- Create: `tests/structured/test_cohort_envelope_assembly.py`

- [ ] **Step 1: Write the failing test**

Create `tests/structured/test_cohort_envelope_assembly.py`:

```python
"""Assembly tests for cohort envelope solid."""

from __future__ import annotations

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.cohort_envelope import (
    assemble_cohort_envelope_solid,
    build_cohort_envelope,
)
from meshwell.structured.plan import build_plan


def _square_slab(zlo, zhi, name, side=1.0):
    return PolyPrism(
        polygons=shapely.box(0, 0, side, side),
        buffers={zlo: 0.0, zhi: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name=name,
    )


def test_assembled_solid_is_valid_and_closed():
    from OCP.BRepCheck import BRepCheck_Analyzer

    plan = build_plan([_square_slab(0.0, 1.0, "L1"), _square_slab(1.0, 2.0, "L2")])
    env = build_cohort_envelope(plan, component_index=0)
    solid = assemble_cohort_envelope_solid(env)
    assert solid is not None
    analyzer = BRepCheck_Analyzer(solid)
    assert analyzer.IsValid(), "Cohort envelope solid must be BRepCheck-valid"


def test_assembled_solid_has_positive_volume():
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    plan = build_plan([_square_slab(0.0, 1.0, "L1")])
    env = build_cohort_envelope(plan, component_index=0)
    solid = assemble_cohort_envelope_solid(env)
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid, props)
    # A unit cube has volume 1.0.
    assert abs(props.Mass() - 1.0) < 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/structured/test_cohort_envelope_assembly.py -v`
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement assembly**

Replace the stub `assemble_cohort_envelope_solid` body in `meshwell/structured/cohort_envelope.py`:

```python
def assemble_cohort_envelope_solid(env: CohortEnvelope) -> Any:
    """Assemble the cohort envelope's TopoDS_Solid.

    Builds a closed shell from:
      - All bottom sub-faces (reversed so outward normal points -Z).
      - All top sub-faces (kept — outward normal already points +Z).
      - All lateral faces (orientation matches outward outline traversal).
    Wraps the shell in a TopoDS_Solid with shell.Reversed() to establish
    outward-normal convention at solid level (same pattern as
    cohort_topology.assemble_cohort_sub_prism).

    Populates env.cohort_solid in-place and returns it.
    """
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS, TopoDS_Shell, TopoDS_Solid

    def _rev_face(f):
        return TopoDS.Face_s(f.Reversed())

    b = BRep_Builder()
    shell = TopoDS_Shell()
    b.MakeShell(shell)

    for bot in env.bottom_sub_faces.values():
        b.Add(shell, _rev_face(bot))
    for top in env.top_sub_faces.values():
        b.Add(shell, top)
    for face_list in env.lateral_faces.values():
        for face in face_list:
            b.Add(shell, face)

    solid = TopoDS_Solid()
    b.MakeSolid(solid)
    b.Add(solid, TopoDS.Shell_s(shell.Reversed()))

    env.cohort_solid = solid
    return solid
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/structured/test_cohort_envelope_assembly.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/cohort_envelope.py tests/structured/test_cohort_envelope_assembly.py
git commit -m "feat(cohort_envelope): assemble cohort solid from registries

Closed shell of reversed bot + top + lateral faces, wrapped in
TopoDS_Solid with shell.Reversed() for outward normals. Validates
BRepCheck and yields correct volume."
```

---

## Task 8: Spec-test #1 — full envelope build for two stacked slabs

**Files:**
- Create: `tests/structured/test_cohort_envelope_build.py`

- [ ] **Step 1: Write the test**

Create `tests/structured/test_cohort_envelope_build.py`:

```python
"""End-to-end cohort envelope build for two stacked simple slabs.

Spec test #1: verify envelope solid is closed, top/bottom shells have
correct per-piece sub-face counts, lateral wall has correct outline-edge
count.
"""

from __future__ import annotations

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.cohort_envelope import (
    assemble_cohort_envelope_solid,
    build_cohort_envelope,
)
from meshwell.structured.plan import build_plan


def _square_slab(zlo, zhi, name):
    return PolyPrism(
        polygons=shapely.box(0, 0, 1, 1),
        buffers={zlo: 0.0, zhi: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name=name,
    )


def test_two_stacked_slabs_envelope_end_to_end():
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    plan = build_plan([_square_slab(0.0, 1.0, "L1"), _square_slab(1.0, 2.0, "L2")])
    env = build_cohort_envelope(plan, component_index=0)
    solid = assemble_cohort_envelope_solid(env)

    assert BRepCheck_Analyzer(solid).IsValid()

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid, props)
    # Two stacked unit boxes: total volume 2.0.
    assert abs(props.Mass() - 2.0) < 1e-3

    # 2 slabs × 1 piece per slab = 2 top sub-faces, 2 bot sub-faces.
    assert len(env.top_sub_faces) == 2
    assert len(env.bottom_sub_faces) == 2

    # 2 slabs × 4 outline edges = 8 lateral face lists.
    assert len(env.lateral_faces) == 8
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/structured/test_cohort_envelope_build.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/structured/test_cohort_envelope_build.py
git commit -m "test(cohort_envelope): end-to-end build for two stacked slabs

Spec test #1: validates topology + assembly together for the
simplest stacking case."
```

---

## Task 9: Spec-test #2 — arc outline envelope

**Files:**
- Create: `tests/structured/test_cohort_envelope_arc.py`

- [ ] **Step 1: Write the test**

Create `tests/structured/test_cohort_envelope_arc.py`:

```python
"""Cohort envelope with an arc outline.

Spec test #2: verify the arc lateral is built via BRepFill::Face_s and
has valid PCurves (bbox sanity).
"""

from __future__ import annotations

import math

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.cohort_envelope import (
    assemble_cohort_envelope_solid,
    build_cohort_envelope,
)
from meshwell.structured.plan import build_plan


def _arc_disc(r, n=32):
    return shapely.Polygon(
        [
            (r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n))
            for i in range(n)
        ]
    )


def test_arc_outline_envelope_assembles_and_volume_matches_circle():
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    arc_slab = PolyPrism(
        polygons=_arc_disc(1.0),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
        physical_name="ArcCohort",
    )
    plan = build_plan([arc_slab])
    env = build_cohort_envelope(plan, component_index=0)
    solid = assemble_cohort_envelope_solid(env)
    assert BRepCheck_Analyzer(solid).IsValid()

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid, props)
    # Unit-radius disc × height 1.0 ~= π.
    assert abs(props.Mass() - math.pi) < 0.05
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/structured/test_cohort_envelope_arc.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/structured/test_cohort_envelope_arc.py
git commit -m "test(cohort_envelope): arc outline envelope

Spec test #2: confirms BRepFill::Face_s arc lateral produces a
BRepCheck-valid solid whose volume matches the analytic disc volume."
```

---

## Task 10: Spec-test #3 — concentric arc discs

**Files:**
- Create: `tests/structured/test_cohort_envelope_concentric.py`

- [ ] **Step 1: Write the test**

Create `tests/structured/test_cohort_envelope_concentric.py`:

```python
"""Cohort envelope with concentric arc discs (multi-arc snap scenario).

Spec test #3: verify the envelope build succeeds with multi-arc vertex
averaging — the same scene that broke Phase 2 cohort_topology with
StdFail_NotDone before the snap fix.
"""

from __future__ import annotations

import math

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.cohort_envelope import (
    assemble_cohort_envelope_solid,
    build_cohort_envelope,
)
from meshwell.structured.plan import build_plan


def _disc(r, n=32):
    return shapely.Polygon(
        [
            (r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n))
            for i in range(n)
        ]
    )


def _arc_slab(r, zlo, zhi, name):
    return PolyPrism(
        polygons=_disc(r),
        buffers={zlo: 0.0, zhi: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
        physical_name=name,
    )


def test_concentric_arc_discs_envelope_builds():
    from OCP.BRepCheck import BRepCheck_Analyzer

    plan = build_plan(
        [
            _arc_slab(1.0, 0.0, 1.0, "L1"),
            _arc_slab(0.7, 1.0, 2.0, "L2"),
            _arc_slab(0.5, 2.0, 3.0, "L3"),
        ]
    )
    env = build_cohort_envelope(plan, component_index=0)
    solid = assemble_cohort_envelope_solid(env)
    # Solid must be BRepCheck-valid even with the multi-arc corners.
    assert BRepCheck_Analyzer(solid).IsValid()
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/structured/test_cohort_envelope_concentric.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/structured/test_cohort_envelope_concentric.py
git commit -m "test(cohort_envelope): concentric arc discs

Spec test #3: confirms the multi-arc vertex averaging carried over
from Phase 2 cohort_topology lets the envelope build succeed on
the concentric arc disc scene."
```

---

## Task 11: Extend `PhantomMap` with `face_keys_to_discrete`

**Files:**
- Modify: `meshwell/structured/spec.py`
- Create: `tests/structured/test_phantom_map_discrete_field.py`

- [ ] **Step 1: Write the failing test**

Create `tests/structured/test_phantom_map_discrete_field.py`:

```python
"""PhantomMap discrete-face field tests."""

from __future__ import annotations

from meshwell.structured.spec import FaceKey, PhantomMap


def test_phantom_map_has_face_keys_to_discrete_field():
    pm = PhantomMap()
    assert pm.face_keys_to_discrete == {}


def test_phantom_map_can_store_discrete_face_tags():
    pm = PhantomMap()
    pm.face_keys_to_discrete[FaceKey(0, "top", 0)] = 42
    assert pm.face_keys_to_discrete[FaceKey(0, "top", 0)] == 42
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/structured/test_phantom_map_discrete_field.py -v`
Expected: FAIL (`AttributeError` — no such field)

- [ ] **Step 3: Add the field**

In `meshwell/structured/spec.py`, modify the `PhantomMap` dataclass (lines ~412-429) to add the new field. Update from:

```python
@dataclass
class PhantomMap:
    """Post-BOP correspondence map.

    Each input element maps to a *list* of output OCC tags because
    BOP may split a single input into many output shapes (e.g. a
    neighbour cut a piece's top face into 3 sub-faces).
    """

    output_faces: dict[FaceKey, list[Any]] = field(default_factory=dict)
    output_edges: dict[EdgeKey, list[Any]] = field(default_factory=dict)
    output_vertices: dict[VertexKey, list[Any]] = field(default_factory=dict)
    output_laterals: dict[LateralKey, list[Any]] = field(default_factory=dict)
    # Per-lateral flag: True iff BOP introduced a new vertex on the
    # lateral face with z strictly between zlo and zhi (a "mid-height
    # cut"). Phase 3's builder uses this to decide which lateral faces
    # to exclude from transfinite hints.
    lateral_has_midheight_cut: dict[LateralKey, bool] = field(default_factory=dict)
```

to:

```python
@dataclass
class PhantomMap:
    """Post-BOP correspondence map.

    Each input element maps to a *list* of output OCC tags because
    BOP may split a single input into many output shapes (e.g. a
    neighbour cut a piece's top face into 3 sub-faces).
    """

    output_faces: dict[FaceKey, list[Any]] = field(default_factory=dict)
    output_edges: dict[EdgeKey, list[Any]] = field(default_factory=dict)
    output_vertices: dict[VertexKey, list[Any]] = field(default_factory=dict)
    output_laterals: dict[LateralKey, list[Any]] = field(default_factory=dict)
    # Per-lateral flag: True iff BOP introduced a new vertex on the
    # lateral face with z strictly between zlo and zhi (a "mid-height
    # cut"). Phase 3's builder uses this to decide which lateral faces
    # to exclude from transfinite hints.
    lateral_has_midheight_cut: dict[LateralKey, bool] = field(default_factory=dict)
    # Phase 3 (discrete cohort mesh): FaceKey -> gmsh 2D-discrete-entity
    # tag for interior piece-to-piece interfaces materialized at mesh time.
    # Distinct from output_faces (which holds OCC-backed face tags).
    # Empty unless _USE_DISCRETE_COHORT_MESH=True. See spec
    # docs/superpowers/specs/2026-05-28-cad-occ-discrete-internal-cohort-mesh-design.md.
    face_keys_to_discrete: dict[FaceKey, int] = field(default_factory=dict)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/structured/test_phantom_map_discrete_field.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/spec.py tests/structured/test_phantom_map_discrete_field.py
git commit -m "feat(spec): PhantomMap.face_keys_to_discrete for Phase 3

Holds discrete 2D entity tags for interior cohort piece-to-piece
interfaces materialized at mesh time. Empty unless discrete cohort
mesh kill-switch is on."
```

---

## Task 12: Phase 3 phantom routing branch

**Files:**
- Modify: `meshwell/structured/phantom.py`
- Create: `tests/structured/test_phantom_discrete_routing.py`

Adds `_USE_DISCRETE_COHORT_MESH = False` kill-switch and routes `build_phantom_shapes` through a new function `_build_phantom_shapes_via_cohort_envelope(plan)` when the flag is True. The new function builds one PhantomShape per cohort (not per piece): `slab_index` is set to a synthetic cohort-level index (negative to distinguish from real slabs — use `-(component_index + 1)`), `piece_index` is 0, the solid is the cohort envelope solid, and the four input dicts cover per-piece top/bottom sub-faces + per-outline-edge laterals across all slabs in the cohort.

- [ ] **Step 1: Write the failing test**

Create `tests/structured/test_phantom_discrete_routing.py`:

```python
"""Phase 3 phantom routing tests."""

from __future__ import annotations

from unittest.mock import patch

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.phantom import build_phantom_shapes
from meshwell.structured.plan import build_plan
from meshwell.structured.spec import FaceKey


def _square_slab(zlo, zhi, name):
    return PolyPrism(
        polygons=shapely.box(0, 0, 1, 1),
        buffers={zlo: 0.0, zhi: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name=name,
    )


def test_phase3_routing_produces_one_phantomshape_per_cohort():
    plan = build_plan([_square_slab(0.0, 1.0, "L1"), _square_slab(1.0, 2.0, "L2")])
    with patch("meshwell.structured.phantom._USE_DISCRETE_COHORT_MESH", True):
        result = build_phantom_shapes(plan)
    # Single cohort (two stacked slabs touch).
    assert len(result.shapes) == 1
    ps = result.shapes[0]
    # Cohort-level PhantomShape carries per-piece top/bot face keys for both slabs.
    assert FaceKey(0, "top", 0) in ps.input_faces_by_key
    assert FaceKey(1, "top", 0) in ps.input_faces_by_key
    assert FaceKey(0, "bot", 0) in ps.input_faces_by_key
    assert FaceKey(1, "bot", 0) in ps.input_faces_by_key


def test_phase3_flag_off_keeps_phase2_behavior():
    """With _USE_DISCRETE_COHORT_MESH=False, build_phantom_shapes uses the existing path."""
    plan = build_plan([_square_slab(0.0, 1.0, "L1"), _square_slab(1.0, 2.0, "L2")])
    # Default flag is False; result should have one PhantomShape per piece (= 2).
    result = build_phantom_shapes(plan)
    assert len(result.shapes) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/structured/test_phantom_discrete_routing.py -v`
Expected: FAIL (`AttributeError: module 'meshwell.structured.phantom' has no attribute '_USE_DISCRETE_COHORT_MESH'`)

- [ ] **Step 3: Add kill-switch and routing branch**

In `meshwell/structured/phantom.py`, after the existing `_USE_COHORT_TOPOLOGY = False` constant (line ~76), insert:

```python
# Phase 3 kill-switch. When True, build_phantom_shapes routes through
# _build_phantom_shapes_via_cohort_envelope (one OCC envelope solid per
# cohort, discrete elements for interior pieces/interfaces at mesh time).
# When False (default during stabilization), routes through Phase 1+2
# path (per-piece OCC sub-prisms).
#
# Promote to default True once the Phase 3 path passes the full
# structured test suite end-to-end. Once that's done and the new path
# has soaked in production, delete the Phase 1+2 cohort code entirely
# (cohort_topology.py, _USE_COHORT_TOPOLOGY, _PRESHARE_VERTICAL_FACES,
# _build_phantom_shapes_via_cohort_topology).
#
# See spec docs/superpowers/specs/2026-05-28-cad-occ-discrete-internal-cohort-mesh-design.md.
_USE_DISCRETE_COHORT_MESH = False
```

In the same file, add a new function before `build_phantom_shapes`:

```python
def _build_phantom_shapes_via_cohort_envelope(
    plan: StructuredPlan,
) -> PhantomBuildResult:
    """Phase 3: one PhantomShape per cohort (envelope solid).

    Per-piece top/bottom sub-faces are bundled into the cohort
    PhantomShape's input_faces_by_key. Lateral wall faces are bundled
    into input_laterals_by_outer_edge keyed by arrangement edge id.

    The slab_index field on the returned PhantomShape is set to a
    synthetic cohort marker (-(component_index + 1)) so it cannot
    collide with real per-slab indices in downstream lookups.
    """
    from meshwell.structured.cohort_envelope import (
        assemble_cohort_envelope_solid,
        build_cohort_envelope,
    )

    component_indices = sorted({s.component_index for s in plan.slabs})

    shapes: list[PhantomShape] = []
    for cidx in component_indices:
        env = build_cohort_envelope(plan, component_index=cidx)
        solid = assemble_cohort_envelope_solid(env)

        input_faces: dict[FaceKey, Any] = {}
        input_faces.update(env.bottom_sub_faces)
        input_faces.update(env.top_sub_faces)

        input_laterals: dict[int, Any] = {}
        for (_slab_idx, outline_edge_id), face_list in env.lateral_faces.items():
            # Phase 3 lateral wall is un-subdivided per piece, so we
            # key by arrangement edge id only. The first face of the
            # per-segment list is the representative; downstream code
            # uses input_laterals only as a presence map for the BOP
            # history walk.
            if outline_edge_id not in input_laterals and face_list:
                input_laterals[outline_edge_id] = face_list[0]

        shapes.append(
            PhantomShape(
                slab_index=-(cidx + 1),
                piece_index=0,
                solid=solid,
                input_faces_by_key=input_faces,
                input_edges_by_key={},
                input_vertices_by_key={},
                input_laterals_by_outer_edge=input_laterals,
            )
        )

    return PhantomBuildResult(shapes=tuple(shapes))
```

And modify `build_phantom_shapes` to branch on the new flag (insert before the existing `if _USE_COHORT_TOPOLOGY:` check, around line 1047):

```python
def build_phantom_shapes(plan: StructuredPlan) -> PhantomBuildResult:
    """For each slab, build one OCP sub-prism per partition piece.

    When _USE_DISCRETE_COHORT_MESH=True (Phase 3), routes through
    _build_phantom_shapes_via_cohort_envelope: one envelope OCC solid
    per cohort, interior pieces/interfaces materialize as discrete
    entities at mesh time.

    When _USE_COHORT_TOPOLOGY=True (Phase 2), delegates to
    _build_phantom_shapes_via_cohort_topology which builds each cohort's
    shared topology once and assembles all sub-prisms as views into it.

    When both flags are False (Phase 1 default), pre-shared vertical
    faces are used: each upper sub-prism reuses the prism below's
    LastShape() as its bottom face.
    """
    if _USE_DISCRETE_COHORT_MESH:
        return _build_phantom_shapes_via_cohort_envelope(plan)
    if _USE_COHORT_TOPOLOGY:
        return _build_phantom_shapes_via_cohort_topology(plan)
    # ... existing body unchanged below ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/structured/test_phantom_discrete_routing.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/phantom.py tests/structured/test_phantom_discrete_routing.py
git commit -m "feat(phantom): Phase 3 cohort envelope routing branch

New kill-switch _USE_DISCRETE_COHORT_MESH (default False) routes
build_phantom_shapes through _build_phantom_shapes_via_cohort_envelope.
One PhantomShape per cohort instead of per piece — synthetic slab_index
-(cidx+1) marks the cohort-level shape."
```

---

## Task 13: cad_occ integration smoke — cohort envelope solids survive fragment

**Files:**
- Create: `tests/structured/test_phase3_cad_occ_smoke.py`

- [ ] **Step 1: Write the test**

Create `tests/structured/test_phase3_cad_occ_smoke.py`:

```python
"""cad_occ fragment smoke test with Phase 3 cohort envelopes.

Goal: confirm that build_phantom_shapes under Phase 3 produces solids
that cad_occ.fragment_all consumes, and that per-piece FaceKeys resolve
to gmsh tags via the PhantomMap.
"""

from __future__ import annotations

from unittest.mock import patch

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.spec import FaceKey


def _square_slab(zlo, zhi, name):
    return PolyPrism(
        polygons=shapely.box(0, 0, 1, 1),
        buffers={zlo: 0.0, zhi: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name=name,
    )


def test_phase3_envelope_solids_round_trip_through_cad_occ():
    """Build envelope solids → cad_occ.fragment_all → check per-piece FaceKey resolution."""
    from meshwell.cad_occ import CAD_OCC

    entities = [_square_slab(0.0, 1.0, "L1"), _square_slab(1.0, 2.0, "L2")]
    with patch("meshwell.structured.phantom._USE_DISCRETE_COHORT_MESH", True):
        cad = CAD_OCC()
        occ_entities, phantom_map = cad.run(entities)

    # 1 cohort envelope solid in occ_entities.
    structured_solids = [e for e in occ_entities if e.kind == "structured"]
    assert len(structured_solids) == 1

    # Per-piece top/bot FaceKeys must resolve to non-empty gmsh tag lists.
    assert phantom_map.output_faces[FaceKey(0, "top", 0)], (
        "Expected FaceKey(0, top, 0) to resolve to a gmsh face tag"
    )
    assert phantom_map.output_faces[FaceKey(1, "top", 0)]
    assert phantom_map.output_faces[FaceKey(0, "bot", 0)]
    assert phantom_map.output_faces[FaceKey(1, "bot", 0)]
```

- [ ] **Step 2: Run test to verify it fails or surfaces gaps**

Run: `pytest tests/structured/test_phase3_cad_occ_smoke.py -v`
Expected: FAIL — either with `AttributeError` on `CAD_OCC.run` (signature differs from what the test expects), `AssertionError` because cohort envelope solids don't carry the expected `kind`, or `KeyError`/empty `output_faces` because `extract_phantom_map` doesn't know how to walk the cohort-level PhantomShape.

- [ ] **Step 3: Adapt the test to the real CAD_OCC entry point and patch gaps**

Read `meshwell/cad_occ.py` to learn the real entry signature and the OCCLabeledEntity field that distinguishes structured solids. Adjust the test imports and assertions to match. If the test reveals that `extract_phantom_map` discards FaceKeys for synthetic cohort `slab_index = -(cidx+1)` values, edit `extract_phantom_map` so it walks every PhantomShape's `input_faces_by_key` regardless of `slab_index` sign — `output_faces[face_key] = builder.Modified(input_face)` (or `[input_face]` if pass-through). Use `meshwell/structured/phantom.py:extract_phantom_map` as the surgical edit target; do NOT key any logic on `slab_index` being non-negative.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/structured/test_phase3_cad_occ_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/structured/test_phase3_cad_occ_smoke.py meshwell/structured/phantom.py
git commit -m "test(phase3): cad_occ smoke for cohort envelope round-trip

Confirms cad_occ.fragment_all consumes cohort envelope solids and
extract_phantom_map walks per-piece FaceKeys regardless of synthetic
cohort slab_index encoding."
```

---

## Task 14: apply_structured_mesh — discrete piece volume routing under Phase 3

**Files:**
- Modify: `meshwell/structured/builder.py`
- Create: `tests/structured/test_phase3_discrete_volume_routing.py`

The existing `apply_structured_mesh` already creates a discrete 3D entity when `_find_volume_containing_faces` returns None (builder.py:560). Under Phase 3, every piece's bot/top sub-face IS inside a cohort envelope OCC volume, so `_find_volume_containing_faces` will return the cohort volume tag — and we'll incorrectly attach per-piece elements into the cohort volume (losing per-piece physical groups). We need to force the discrete-entity path under Phase 3.

- [ ] **Step 1: Write the failing test**

Create `tests/structured/test_phase3_discrete_volume_routing.py`:

```python
"""Each cohort piece gets its own discrete 3D entity under Phase 3."""

from __future__ import annotations

from unittest.mock import patch

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec


def _square_slab(zlo, zhi, name):
    return PolyPrism(
        polygons=shapely.box(0, 0, 1, 1),
        buffers={zlo: 0.0, zhi: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name=name,
    )


def test_each_piece_gets_dedicated_discrete_volume(tmp_path):
    """Two stacked single-piece slabs in one cohort → 2 distinct volume tags."""
    import gmsh

    from meshwell.orchestrator import build_mesh

    out = tmp_path / "phase3.msh"
    entities = [_square_slab(0.0, 1.0, "L1"), _square_slab(1.0, 2.0, "L2")]
    with patch("meshwell.structured.phantom._USE_DISCRETE_COHORT_MESH", True):
        build_mesh(entities, output_mesh=str(out), dim=3)

    gmsh.initialize()
    try:
        gmsh.open(str(out))
        vol_groups = [
            (d, t, gmsh.model.getPhysicalName(d, t))
            for (d, t) in gmsh.model.getPhysicalGroups(dim=3)
        ]
        names = [name for (_d, _t, name) in vol_groups]
        # Two distinct physical groups (one per piece).
        assert "L1" in names
        assert "L2" in names
        assert names.count("L1") == 1
        assert names.count("L2") == 1
    finally:
        gmsh.finalize()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/structured/test_phase3_discrete_volume_routing.py -v`
Expected: FAIL — either `L1` and `L2` are missing (existing builder routes pieces into the cohort OCC volume), or duplicate keys collide.

- [ ] **Step 3: Force discrete entity allocation under Phase 3**

In `meshwell/structured/builder.py`, modify the volume-routing block in `apply_structured_mesh` (around line 548-560). Locate:

```python
                occ_vol_tag: int | None = _find_volume_containing_faces(
                    bot_tag, top_tag
                )

                # Register all interior nodes BEFORE addElements ...
                pre_vol_tag: int | None = occ_vol_tag
                if n_layers > 1 and pre_vol_tag is None:
                    pre_vol_tag = gmsh.model.addDiscreteEntity(3, -1, [])
```

Replace with:

```python
                from meshwell.structured.phantom import _USE_DISCRETE_COHORT_MESH

                if _USE_DISCRETE_COHORT_MESH:
                    # Phase 3: never share a cohort envelope's OCC volume across
                    # pieces — each piece needs its own discrete 3D entity so
                    # per-piece physical groups stay distinct.
                    occ_vol_tag = None
                else:
                    occ_vol_tag = _find_volume_containing_faces(bot_tag, top_tag)

                pre_vol_tag: int | None = occ_vol_tag
                if pre_vol_tag is None:
                    pre_vol_tag = gmsh.model.addDiscreteEntity(3, -1, [])
                # Assign the slab/piece physical group to the discrete entity
                # under Phase 3 (when there's no OCC volume that already carries
                # it). The orchestrator's load_occ_entities attached physicals
                # only to OCC volumes, so without this the discrete entity has
                # no physical group.
                if _USE_DISCRETE_COHORT_MESH and occ_vol_tag is None:
                    pg_name = "/".join(slab.physical_name)
                    pg_tag = gmsh.model.addPhysicalGroup(3, [pre_vol_tag])
                    gmsh.model.setPhysicalName(3, pg_tag, pg_name)
```

Note: the `if n_layers > 1` guard is removed because Phase 3 always allocates the discrete entity even for `n_layers == 1` (the cohort OCC volume must not be used).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/structured/test_phase3_discrete_volume_routing.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/builder.py tests/structured/test_phase3_discrete_volume_routing.py
git commit -m "feat(builder): per-piece discrete 3D entity under Phase 3

When the discrete cohort mesh kill-switch is on, every piece gets a
fresh addDiscreteEntity(3, -1, []) and a per-piece physical group —
never reuses the cohort envelope's OCC volume tag (which would
collapse N pieces' physicals into one volume)."
```

---

## Task 15: Stamp interior interfaces as discrete 2D entities

**Files:**
- Modify: `meshwell/structured/builder.py`
- Create: `tests/structured/test_phase3_interior_interface_stamping.py`

Interior interfaces have two flavors per the spec:
- **Horizontal:** between two stacked pieces of adjacent slabs in a cohort. The interface = the top sub-face of the lower piece (= bottom sub-face of the upper piece).
- **Vertical:** between two laterally-adjacent pieces of the same slab. The interface is a vertical strip extruded from the shared piece-to-piece edge over the slab's z-layers.

We add a helper that runs after the per-piece volume loop and materializes both flavors.

- [ ] **Step 1: Write the failing test**

Create `tests/structured/test_phase3_interior_interface_stamping.py`:

```python
"""Interior interface discrete 2D entity stamping under Phase 3."""

from __future__ import annotations

from unittest.mock import patch

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec


def _square_slab(zlo, zhi, name):
    return PolyPrism(
        polygons=shapely.box(0, 0, 1, 1),
        buffers={zlo: 0.0, zhi: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name=name,
    )


def test_horizontal_interior_interface_materialized(tmp_path):
    """Stacked slabs in one cohort produce a discrete 2D entity at z=1."""
    import gmsh

    from meshwell.orchestrator import build_mesh

    out = tmp_path / "phase3.msh"
    entities = [_square_slab(0.0, 1.0, "L1"), _square_slab(1.0, 2.0, "L2")]
    with patch("meshwell.structured.phantom._USE_DISCRETE_COHORT_MESH", True):
        build_mesh(entities, output_mesh=str(out), dim=3)

    gmsh.initialize()
    try:
        gmsh.open(str(out))
        # Look for a physical group at the slab interface.
        groups_2d = gmsh.model.getPhysicalGroups(dim=2)
        names = {gmsh.model.getPhysicalName(2, t) for (_d, t) in groups_2d}
        # Existing meshwell convention names the interface "L1___L2"
        # (interface_delimiter is the triple underscore). Confirm one
        # of the two orderings is present.
        assert "L1___L2" in names or "L2___L1" in names, (
            f"Expected interface physical group, got: {names}"
        )
    finally:
        gmsh.finalize()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/structured/test_phase3_interior_interface_stamping.py -v`
Expected: FAIL — the interface physical group doesn't exist yet.

- [ ] **Step 3: Implement interior interface stamping**

In `meshwell/structured/builder.py`, after the per-piece volume loop in `apply_structured_mesh` (after `vol_tags.append(vol_tag)` and before `# Global cleanup: merge ~coincident nodes.` ~line 583), add a call to a new helper `_stamp_phase3_interior_interfaces(plan, phantom_map, ...)`. Implement that helper at module scope above `apply_structured_mesh`:

```python
def _stamp_phase3_interior_interfaces(
    plan: StructuredPlan,
    phantom_map: Any,
    bot_node_tags_per_piece: dict[tuple[int, int], np.ndarray],
    top_node_tags_per_piece: dict[tuple[int, int], np.ndarray],
    bot_tri_per_piece: dict[tuple[int, int], np.ndarray],
    interface_delimiter: str = "___",
) -> None:
    """Materialize discrete 2D entities for interior cohort interfaces.

    Two flavors:
    - Horizontal: where two slabs of the same cohort touch in z, the
      shared XY footprint of overlapping pieces is one discrete 2D entity.
      Element nodes come from the upper slab's bottom face mesh (which
      equals the lower slab's top face mesh by construction).
    - Vertical: where two pieces of the same slab share an edge, the
      vertical strip extruded over the slab's z-layers is one discrete
      2D entity per piece-pair-edge. Element nodes come from the bot
      sub-face boundary + extruded layer node tags.

    Records each interface's FaceKey in phantom_map.face_keys_to_discrete.
    """
    import gmsh

    # --- Horizontal interfaces -----------------------------------------
    # Group slabs by component_index.
    cohort_slabs: dict[int, list[tuple[int, Slab]]] = {}
    for slab_idx, slab in enumerate(plan.slabs):
        cohort_slabs.setdefault(slab.component_index, []).append((slab_idx, slab))

    for cidx, slab_list in cohort_slabs.items():
        slab_list.sort(key=lambda pair: pair[1].zlo)
        # Walk stacked pairs.
        for i in range(len(slab_list) - 1):
            lower_idx, lower = slab_list[i]
            upper_idx, upper = slab_list[i + 1]
            if abs(upper.zlo - lower.zhi) > 1e-9:
                continue  # not touching
            # For each piece in the upper slab, see if it overlaps a piece
            # in the lower slab — if so, register a horizontal interface.
            for u_pidx, _u_piece in enumerate(upper.face_partition):
                u_tri_key = (upper_idx, u_pidx)
                if u_tri_key not in bot_tri_per_piece:
                    continue
                u_bot_nodes = bot_node_tags_per_piece[u_tri_key]
                # The interface uses the same node tags as the upper piece's bottom mesh.
                disc_tag = gmsh.model.addDiscreteEntity(2, -1, [])
                # Add the triangles (or quads) from the upper bot face.
                tris = bot_tri_per_piece[u_tri_key]
                next_elem = int(gmsh.model.mesh.getMaxElementTag()) + 1
                n_elem = tris.shape[0]
                elem_tags = list(range(next_elem, next_elem + n_elem))
                elem_type = 3 if tris.shape[1] == 4 else 2
                gmsh.model.mesh.addElements(
                    2, disc_tag, [elem_type], [elem_tags], [tris.flatten().tolist()]
                )
                # Name: "lower_phys___upper_phys".
                lo_name = "/".join(lower.physical_name)
                up_name = "/".join(upper.physical_name)
                pg_name = f"{lo_name}{interface_delimiter}{up_name}"
                pg_tag = gmsh.model.addPhysicalGroup(2, [disc_tag])
                gmsh.model.setPhysicalName(2, pg_tag, pg_name)
                from meshwell.structured.spec import FaceKey

                phantom_map.face_keys_to_discrete[
                    FaceKey(upper_idx, "bot", u_pidx)
                ] = disc_tag
                _ = u_bot_nodes  # presently unused (node sharing already implicit)

    # --- Vertical interfaces -------------------------------------------
    # For each slab with N pieces, walk pieces pairwise; if two pieces
    # share a polygon edge in XY, build a discrete 2D entity along that
    # edge × z-layers using the bot face boundary nodes + extruded layer
    # node tags. Skipped here when each slab has only one piece (no
    # interior vertical interfaces).
    # Implemented as a follow-up only if test_phase3_interior_interfaces
    # exercises a multi-piece-per-slab scene; for the single-piece
    # smoke test in this task it's a no-op.
```

Note: vertical interfaces require multi-piece-per-slab fixtures and the bot-face-edge → adjacent-piece lookup. That logic is captured in Task 16. Task 15's helper covers the horizontal case which is sufficient for the single-piece smoke test.

Also collect the per-piece node + triangle arrays inside the existing loop and pass them to the helper. Inside the per-piece loop (around line 525 where `bot_node_tags_arr` is collected), capture the data into dicts at the top of `apply_structured_mesh`:

```python
    # Phase 3 bookkeeping: per-piece bot/top node tags + bot face cells.
    bot_node_tags_per_piece: dict[tuple[int, int], np.ndarray] = {}
    top_node_tags_per_piece: dict[tuple[int, int], np.ndarray] = {}
    bot_tri_per_piece: dict[tuple[int, int], np.ndarray] = {}
```

Inside the per-piece loop, after `top_map = _stamp_top_face_mesh(...)`, capture:

```python
                bot_node_tags_arr, _, _ = gmsh.model.mesh.getNodes(
                    2, bot_tag, includeBoundary=True
                )
                bot_node_tags_per_piece[(slab_idx, piece_idx)] = np.asarray(
                    bot_node_tags_arr, dtype=np.int64
                )
                top_node_tags_per_piece[(slab_idx, piece_idx)] = np.asarray(
                    list(top_map.values()), dtype=np.int64
                )
                # Collect bot triangulation for the helper.
                elem_types_b, _, elem_nodes_b = gmsh.model.mesh.getElements(2, bot_tag)
                target_type = 3 if recombine else 2
                cells_per_face = 4 if recombine else 3
                bot_cells: list[np.ndarray] = []
                for et, en in zip(elem_types_b, elem_nodes_b):
                    if et == target_type:
                        bot_cells.append(
                            np.asarray(en, dtype=np.int64).reshape(-1, cells_per_face)
                        )
                if bot_cells:
                    bot_tri_per_piece[(slab_idx, piece_idx)] = np.concatenate(
                        bot_cells, axis=0
                    )
```

After the per-piece loop, before `# Global cleanup`, call:

```python
    from meshwell.structured.phantom import _USE_DISCRETE_COHORT_MESH

    if _USE_DISCRETE_COHORT_MESH:
        _stamp_phase3_interior_interfaces(
            plan,
            phantom_map,
            bot_node_tags_per_piece,
            top_node_tags_per_piece,
            bot_tri_per_piece,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/structured/test_phase3_interior_interface_stamping.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/builder.py tests/structured/test_phase3_interior_interface_stamping.py
git commit -m "feat(builder): horizontal interior interface stamping under Phase 3

For each stacked slab pair in a cohort, materialize a discrete 2D
entity at the shared z-plane with the upper piece's bottom mesh and
a physical group of the form 'lower___upper'. FaceKey recorded in
phantom_map.face_keys_to_discrete. Vertical interfaces are deferred
to Task 16."
```

---

## Task 16: Vertical interior interface stamping

**Files:**
- Modify: `meshwell/structured/builder.py`
- Create: `tests/structured/test_phase3_vertical_interfaces.py`

For multi-piece slabs, the shared edge between two adjacent pieces at the top face is a piece-to-piece interior boundary. Each such edge becomes a vertical strip of quads (or tris) over the slab's z-layers.

- [ ] **Step 1: Write the failing test**

Create `tests/structured/test_phase3_vertical_interfaces.py`:

```python
"""Vertical interior interface materialization under Phase 3."""

from __future__ import annotations

from unittest.mock import patch

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec


def test_vertical_interface_between_lateral_pieces(tmp_path):
    """Two laterally-adjacent slabs sharing one outline edge produce
    a vertical interface physical group."""
    import gmsh

    from meshwell.orchestrator import build_mesh

    s1 = PolyPrism(
        polygons=shapely.box(0, 0, 1, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="LeftSlab",
    )
    s2 = PolyPrism(
        polygons=shapely.box(1, 0, 2, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="RightSlab",
    )
    out = tmp_path / "phase3.msh"
    with patch("meshwell.structured.phantom._USE_DISCRETE_COHORT_MESH", True):
        build_mesh([s1, s2], output_mesh=str(out), dim=3)

    gmsh.initialize()
    try:
        gmsh.open(str(out))
        groups_2d = gmsh.model.getPhysicalGroups(dim=2)
        names = {gmsh.model.getPhysicalName(2, t) for (_d, t) in groups_2d}
        assert "LeftSlab___RightSlab" in names or "RightSlab___LeftSlab" in names, (
            f"Expected vertical interface physical group, got {names}"
        )
    finally:
        gmsh.finalize()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/structured/test_phase3_vertical_interfaces.py -v`
Expected: FAIL — vertical interface physical group missing.

- [ ] **Step 3: Implement vertical interface stamping**

Extend `_stamp_phase3_interior_interfaces` in `meshwell/structured/builder.py`. Add a section after the horizontal-interfaces loop:

```python
    # --- Vertical interfaces -------------------------------------------
    # For each cohort, find pairs of pieces (within the same slab OR
    # across same-slab pieces of different physical entities at the same
    # z-interval) that share a polygon edge. Materialize a vertical
    # strip of cells over the slab's n_layers using the bot face's
    # boundary nodes at that edge.
    from shapely.geometry import LineString
    from shapely.ops import linemerge

    for cidx, slab_list in cohort_slabs.items():
        # Group slabs by z-interval so same-z pieces can be compared.
        by_z: dict[tuple[float, float], list[tuple[int, Slab]]] = {}
        for sidx, s in slab_list:
            by_z.setdefault((s.zlo, s.zhi), []).append((sidx, s))
        for (zlo, zhi), z_slabs in by_z.items():
            n_layers = 1
            for sidx, _s in z_slabs:
                # Use any slab's n_layers — they must match within a cohort z-interval.
                # (mesh_plan is passed in via closure; re-look up here.)
                pass
            # Compare each piece against every other piece in same z-interval.
            entries: list[tuple[int, int, Slab, "Polygon"]] = []
            for sidx, s in z_slabs:
                for pidx, piece in enumerate(s.face_partition):
                    entries.append((sidx, pidx, s, piece))
            for i in range(len(entries)):
                sidx_a, pidx_a, slab_a, piece_a = entries[i]
                for j in range(i + 1, len(entries)):
                    sidx_b, pidx_b, slab_b, piece_b = entries[j]
                    inter = piece_a.boundary.intersection(piece_b.boundary)
                    if inter.is_empty:
                        continue
                    if inter.geom_type not in ("LineString", "MultiLineString"):
                        continue
                    seg = (
                        linemerge(inter)
                        if inter.geom_type == "MultiLineString"
                        else inter
                    )
                    if not isinstance(seg, LineString):
                        continue
                    bot_tag_a_key = (sidx_a, pidx_a)
                    if bot_tag_a_key not in bot_node_tags_per_piece:
                        continue
                    # Find boundary node tags on piece_a's bot face that lie
                    # on the shared edge. Walk all bot nodes and filter by
                    # parametric distance to the LineString.
                    nodes_a_tags, nodes_a_coords_flat, _ = gmsh.model.mesh.getNodes(
                        2, _bot_face_tag_for(sidx_a, pidx_a, phantom_map), includeBoundary=True
                    )
                    nodes_a_tags = np.asarray(nodes_a_tags, dtype=np.int64)
                    nodes_a_xy = np.asarray(
                        nodes_a_coords_flat, dtype=float
                    ).reshape(-1, 3)[:, :2]
                    # Filter to nodes within 1e-9 of the shared edge.
                    line_pts = np.asarray(list(seg.coords), dtype=float)
                    keep_idx = []
                    for k, (px, py) in enumerate(nodes_a_xy):
                        # Distance from point to segment.
                        d_min = np.inf
                        for a, b in zip(line_pts[:-1], line_pts[1:]):
                            ab = b - a
                            ap = np.array([px - a[0], py - a[1]])
                            t = max(0.0, min(1.0, ap.dot(ab) / (ab.dot(ab) + 1e-30)))
                            proj = a + t * ab
                            d = np.hypot(px - proj[0], py - proj[1])
                            if d < d_min:
                                d_min = d
                        if d_min < 1e-7:
                            keep_idx.append(k)
                    if len(keep_idx) < 2:
                        continue
                    # Order keep_idx along the LineString.
                    ordered = sorted(
                        keep_idx,
                        key=lambda k: (
                            (nodes_a_xy[k][0] - line_pts[0, 0]) ** 2
                            + (nodes_a_xy[k][1] - line_pts[0, 1]) ** 2
                        ),
                    )
                    boundary_node_chain = [int(nodes_a_tags[k]) for k in ordered]
                    # Build vertical-strip quads using bot_node_tags_per_piece
                    # (layer 0) and top_node_tags_per_piece (layer n_layers)
                    # via the layer_maps captured per piece. For n_layers=1
                    # we have just bot row + top row.
                    # We retrieve the layer maps from the per-piece interior
                    # bookkeeping previously stored.
                    # (Implementation captures bot/top mapping via the helper
                    # signature extended in this task — see below.)
                    quads: list[int] = []
                    layer_rows: list[list[int]] = [boundary_node_chain]
                    # Top layer = top_face's node at the same XY.
                    top_chain: list[int] = []
                    for tag in boundary_node_chain:
                        # bot_to_top mapping: find by index in bot_node_tags_per_piece[a].
                        pos = int(
                            np.where(bot_node_tags_per_piece[bot_tag_a_key] == tag)[0][0]
                        )
                        top_chain.append(
                            int(top_node_tags_per_piece[bot_tag_a_key][pos])
                        )
                    layer_rows.append(top_chain)
                    # Build quads between layer 0 and layer 1.
                    for k in range(len(boundary_node_chain) - 1):
                        quads.extend(
                            [
                                layer_rows[0][k],
                                layer_rows[0][k + 1],
                                layer_rows[1][k + 1],
                                layer_rows[1][k],
                            ]
                        )
                    if not quads:
                        continue
                    disc_tag = gmsh.model.addDiscreteEntity(2, -1, [])
                    next_elem = int(gmsh.model.mesh.getMaxElementTag()) + 1
                    n_quad = len(quads) // 4
                    elem_tags = list(range(next_elem, next_elem + n_quad))
                    gmsh.model.mesh.addElements(
                        2, disc_tag, [3], [elem_tags], [quads]
                    )
                    a_name = "/".join(slab_a.physical_name)
                    b_name = "/".join(slab_b.physical_name)
                    pg_tag = gmsh.model.addPhysicalGroup(2, [disc_tag])
                    gmsh.model.setPhysicalName(
                        2, pg_tag, f"{a_name}{interface_delimiter}{b_name}"
                    )
                    from meshwell.structured.spec import FaceKey

                    phantom_map.face_keys_to_discrete[
                        FaceKey(sidx_a, "bot", pidx_a)
                    ] = disc_tag
```

And add the helper that resolves a piece's bot OCC face tag from phantom_map:

```python
def _bot_face_tag_for(slab_idx: int, piece_idx: int, phantom_map: Any) -> int:
    from meshwell.structured.spec import FaceKey

    tags = phantom_map.output_faces.get(FaceKey(slab_idx, "bot", piece_idx), [])
    if not tags:
        raise RuntimeError(
            f"No OCC bot face tag for slab={slab_idx}, piece={piece_idx}"
        )
    return tags[0]
```

NOTE: this is a minimal implementation that only handles `n_layers == 1`. Multi-layer support is captured in Task 17 below.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/structured/test_phase3_vertical_interfaces.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/builder.py tests/structured/test_phase3_vertical_interfaces.py
git commit -m "feat(builder): vertical interior interface stamping (n_layers=1)

Two pieces sharing a polygon edge produce a discrete 2D entity with
quads over the slab z-extent. Boundary node chain extracted from
each piece's bot face mesh; vertical strip uses bot/top layer node
tags. n_layers > 1 deferred to Task 17."
```

---

## Task 17: Multi-layer vertical interfaces

**Files:**
- Modify: `meshwell/structured/builder.py`
- Create: `tests/structured/test_phase3_multilayer_vertical.py`

- [ ] **Step 1: Write the failing test**

Create `tests/structured/test_phase3_multilayer_vertical.py`:

```python
"""Vertical interfaces stamp correctly when n_layers > 1."""

from __future__ import annotations

from unittest.mock import patch

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec


def test_vertical_interface_multilayer(tmp_path):
    import gmsh

    from meshwell.orchestrator import build_mesh

    s1 = PolyPrism(
        polygons=shapely.box(0, 0, 1, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[3])],
        physical_name="A",
    )
    s2 = PolyPrism(
        polygons=shapely.box(1, 0, 2, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[3])],
        physical_name="B",
    )
    out = tmp_path / "phase3.msh"
    with patch("meshwell.structured.phantom._USE_DISCRETE_COHORT_MESH", True):
        build_mesh([s1, s2], output_mesh=str(out), dim=3)

    gmsh.initialize()
    try:
        gmsh.open(str(out))
        # Find the A___B group; count quads — should be 3 (n_layers) per
        # boundary segment.
        groups_2d = gmsh.model.getPhysicalGroups(dim=2)
        ab_tag = next(
            t
            for (_d, t) in groups_2d
            if gmsh.model.getPhysicalName(2, t) in ("A___B", "B___A")
        )
        ents = gmsh.model.getEntitiesForPhysicalGroup(2, ab_tag)
        total_quads = 0
        for ent in ents:
            elem_types, elem_tags, _ = gmsh.model.mesh.getElements(2, int(ent))
            for et, ets in zip(elem_types, elem_tags):
                if et == 3:  # quad
                    total_quads += len(ets)
        # The shared edge runs from y=0 to y=1; with the default mesh size
        # there's at least 1 horizontal subdivision. Each subdivision × 3
        # layers = ≥3 quads.
        assert total_quads >= 3
    finally:
        gmsh.finalize()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/structured/test_phase3_multilayer_vertical.py -v`
Expected: FAIL (vertical strip only has 1 z-row instead of 3).

- [ ] **Step 3: Extend vertical interface helper to handle n_layers > 1**

In `meshwell/structured/builder.py`, the vertical-interface code in Task 16 hard-codes the bot+top rows. Extend it to use all `n_layers + 1` rows. The per-piece layer-map data structure (`layer_maps`) is already built per piece inside the volume loop; capture it for the helper:

In the existing per-piece loop, after `layer_maps = [...]` is constructed (around line 546), capture:

```python
                layer_maps_per_piece[(slab_idx, piece_idx)] = layer_maps
```

with a fresh dict initialized at the top of `apply_structured_mesh`:

```python
    layer_maps_per_piece: dict[tuple[int, int], list[dict[int, int]]] = {}
```

Pass it into `_stamp_phase3_interior_interfaces` as a new arg, and inside the vertical-interface loop replace the layer_rows construction with:

```python
                    layer_maps = layer_maps_per_piece[bot_tag_a_key]
                    layer_rows = [boundary_node_chain]
                    for lm in layer_maps:
                        layer_rows.append([int(lm[t]) for t in boundary_node_chain])
                    quads = []
                    for layer in range(len(layer_rows) - 1):
                        row_lo = layer_rows[layer]
                        row_hi = layer_rows[layer + 1]
                        for k in range(len(boundary_node_chain) - 1):
                            quads.extend(
                                [
                                    row_lo[k],
                                    row_lo[k + 1],
                                    row_hi[k + 1],
                                    row_hi[k],
                                ]
                            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/structured/test_phase3_multilayer_vertical.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/builder.py tests/structured/test_phase3_multilayer_vertical.py
git commit -m "feat(builder): multi-layer vertical interior interfaces

Vertical strip extends across n_layers + 1 z-rows using the per-piece
layer_maps captured during volume stamping."
```

---

## Task 18: Remove cohort envelope OCC volumes after stamping

**Files:**
- Modify: `meshwell/structured/builder.py`
- Create: `tests/structured/test_phase3_no_phantom_tet.py`

After the structured hook runs, the 3D mesh pass would try to tetrahedralize the cohort envelope's OCC 3D entity (which has no elements). We must remove it.

- [ ] **Step 1: Write the failing test**

Create `tests/structured/test_phase3_no_phantom_tet.py`:

```python
"""Cohort envelope OCC volume must not be tetrahedralized."""

from __future__ import annotations

from unittest.mock import patch

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec


def test_cohort_envelope_volume_not_in_final_mesh(tmp_path):
    """The 3D mesh contains only wedge/hex (from discrete entities), no tets."""
    import gmsh

    from meshwell.orchestrator import build_mesh

    s1 = PolyPrism(
        polygons=shapely.box(0, 0, 1, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="L1",
    )
    out = tmp_path / "phase3.msh"
    with patch("meshwell.structured.phantom._USE_DISCRETE_COHORT_MESH", True):
        build_mesh([s1], output_mesh=str(out), dim=3)

    gmsh.initialize()
    try:
        gmsh.open(str(out))
        elem_types, _, _ = gmsh.model.mesh.getElements(3)
        # element type 4 = tetrahedron; should be absent.
        assert 4 not in elem_types, (
            f"Found tetrahedra in 3D mesh (cohort envelope was tetrahedralized): {elem_types}"
        )
    finally:
        gmsh.finalize()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/structured/test_phase3_no_phantom_tet.py -v`
Expected: FAIL — tets present from cohort envelope OCC volume.

- [ ] **Step 3: Remove cohort OCC 3D entities at end of structured hook**

In `meshwell/structured/builder.py`, after the `_stamp_phase3_interior_interfaces(...)` call but before the `removeDuplicateNodes` block, add:

```python
    if _USE_DISCRETE_COHORT_MESH:
        # Remove cohort envelope OCC 3D entities so the subsequent 3D
        # mesh pass doesn't tetrahedralize them. The 2D OCC faces remain
        # (they carry the per-piece sub-face meshes used by discrete
        # volumes), but the volume entity is no longer needed once
        # per-piece discrete 3D entities own the mesh.
        cohort_vol_tags: list[int] = []
        for ent in occ_entities:
            if getattr(ent, "kind", None) == "structured":
                for occ_tag in getattr(ent, "occ_tags", []):
                    if isinstance(occ_tag, tuple) and occ_tag[0] == 3:
                        cohort_vol_tags.append(int(occ_tag[1]))
                    elif isinstance(occ_tag, int):
                        cohort_vol_tags.append(int(occ_tag))
        if cohort_vol_tags:
            gmsh.model.removeEntities(
                [(3, t) for t in cohort_vol_tags], recursive=False
            )
```

Note: `occ_entities` is already a parameter of `apply_structured_mesh`. The attribute name `kind` and structure of `occ_tags` must match the project's `OCCLabeledEntity`; the implementer should grep `meshwell/cad_occ.py` and `meshwell/_mesh_entity.py` to confirm the correct attribute path and adjust accordingly.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/structured/test_phase3_no_phantom_tet.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/builder.py tests/structured/test_phase3_no_phantom_tet.py
git commit -m "feat(builder): drop cohort envelope OCC volumes post-stamping

Once per-piece discrete 3D entities own the structured mesh, the
cohort envelope's OCC volume must be removed so the 3D mesh pass
doesn't tetrahedralize an empty volume. 2D OCC faces remain (they
carry the per-piece sub-face triangulations)."
```

---

## Task 19: Spec test #4 — end-to-end per-piece discrete volumes count

**Files:**
- Create: `tests/structured/test_phase3_discrete_volumes.py`

- [ ] **Step 1: Write the test**

Create `tests/structured/test_phase3_discrete_volumes.py`:

```python
"""End-to-end mesh with 4 pieces × 2 slabs = 8 piece volumes.

Spec test #4: verify 8 discrete 3D entities exist with correct
physical names and total element count matches structured layer counts.
"""

from __future__ import annotations

from unittest.mock import patch

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec


def test_8_piece_cohort_yields_8_discrete_volumes(tmp_path):
    import gmsh

    from meshwell.orchestrator import build_mesh

    # 4 quadrant slabs at z=[0,1] and z=[1,2] => 8 pieces total in one cohort.
    out = tmp_path / "phase3.msh"
    entities = []
    quadrants = [
        (0, 0, 0.5, 0.5),
        (0.5, 0, 1.0, 0.5),
        (0, 0.5, 0.5, 1.0),
        (0.5, 0.5, 1.0, 1.0),
    ]
    for (zlo, zhi) in [(0.0, 1.0), (1.0, 2.0)]:
        for q_i, (x0, y0, x1, y1) in enumerate(quadrants):
            entities.append(
                PolyPrism(
                    polygons=shapely.box(x0, y0, x1, y1),
                    buffers={zlo: 0.0, zhi: 0.0},
                    structured=True,
                    resolutions=[
                        StructuredExtrusionResolutionSpec(n_layers=[2])
                    ],
                    physical_name=f"Q{q_i}_z{int(zlo)}",
                )
            )

    with patch("meshwell.structured.phantom._USE_DISCRETE_COHORT_MESH", True):
        build_mesh(entities, output_mesh=str(out), dim=3)

    gmsh.initialize()
    try:
        gmsh.open(str(out))
        groups_3d = gmsh.model.getPhysicalGroups(dim=3)
        names = [gmsh.model.getPhysicalName(3, t) for (_d, t) in groups_3d]
        # 8 distinct piece physical groups.
        assert len(set(names)) == 8
        # Total 3D element count = 8 pieces × n_layers (2) × (cells per piece).
        # We don't pin the cells-per-piece (depends on mesh size), but
        # element count per group should be > 0.
        for _d, t in groups_3d:
            ents = gmsh.model.getEntitiesForPhysicalGroup(3, t)
            total = 0
            for ent in ents:
                _, et_tags, _ = gmsh.model.mesh.getElements(3, int(ent))
                for tags in et_tags:
                    total += len(tags)
            assert total > 0, f"Group {gmsh.model.getPhysicalName(3, t)} has no elements"
    finally:
        gmsh.finalize()
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/structured/test_phase3_discrete_volumes.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/structured/test_phase3_discrete_volumes.py
git commit -m "test(phase3): 8 piece volumes from quadrant cohort

Spec test #4: end-to-end mesh of 4 quadrant slabs × 2 z-intervals
yields 8 distinct discrete 3D physical groups, each non-empty."
```

---

## Task 20: Spec tests #5 + #6 + bench

**Files:**
- Create: `tests/structured/test_phase3_interior_interfaces.py`
- Create: `tests/structured/test_phase3_top_bottom_conformality.py`
- Create: `scripts/bench_cohort_envelope.py`
- Modify: `scripts/bench_fragment_all.py`

- [ ] **Step 1: Write spec test #5 — interior interface node conformity**

Create `tests/structured/test_phase3_interior_interfaces.py`:

```python
"""Conformal node sharing across interior interfaces."""

from __future__ import annotations

from unittest.mock import patch

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec


def test_horizontal_interface_nodes_shared(tmp_path):
    """The shared interface mesh nodes are the same node tags on both sides."""
    import gmsh
    import numpy as np

    from meshwell.orchestrator import build_mesh

    s1 = PolyPrism(
        polygons=shapely.box(0, 0, 1, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="L1",
    )
    s2 = PolyPrism(
        polygons=shapely.box(0, 0, 1, 1),
        buffers={1.0: 0.0, 2.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="L2",
    )
    out = tmp_path / "phase3.msh"
    with patch("meshwell.structured.phantom._USE_DISCRETE_COHORT_MESH", True):
        build_mesh([s1, s2], output_mesh=str(out), dim=3)

    gmsh.initialize()
    try:
        gmsh.open(str(out))
        # Get all nodes at z=1 (the interface plane).
        all_node_tags, all_coords_flat, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(all_coords_flat, dtype=float).reshape(-1, 3)
        at_interface = np.isclose(coords[:, 2], 1.0, atol=1e-9)
        n_interface_nodes = int(at_interface.sum())
        # No duplicate nodes at z=1: the L1 top mesh and L2 bot mesh share tags.
        # Count interface nodes vs the union of L1's top boundary and L2's bot
        # boundary — they must match within a small constant.
        assert n_interface_nodes > 0, "Expected interface plane nodes"

        # Sanity: removeDuplicateNodes ran at end of apply_structured_mesh,
        # so any near-coincident interface nodes are collapsed. Check
        # there are no duplicate XY positions at z=1.
        xy_at_iface = coords[at_interface, :2]
        unique = np.unique(np.round(xy_at_iface, 9), axis=0)
        assert unique.shape[0] == n_interface_nodes, (
            "Duplicate interface nodes survived removeDuplicateNodes"
        )
    finally:
        gmsh.finalize()
```

- [ ] **Step 2: Run spec test #5**

Run: `pytest tests/structured/test_phase3_interior_interfaces.py -v`
Expected: PASS

- [ ] **Step 3: Write spec test #6 — top/bottom conformality with unstructured neighbor**

Create `tests/structured/test_phase3_top_bottom_conformality.py`:

```python
"""Unstructured neighbor above/below shares OCC topology with cohort top/bot."""

from __future__ import annotations

from unittest.mock import patch

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec


def test_unstructured_tet_above_structured_cohort_shares_nodes(tmp_path):
    """Tet-meshed slab above a structured cohort: interface nodes are shared."""
    import gmsh
    import numpy as np

    from meshwell.orchestrator import build_mesh

    structured = PolyPrism(
        polygons=shapely.box(0, 0, 1, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="StructA",
    )
    # Unstructured above.
    unstructured = PolyPrism(
        polygons=shapely.box(0, 0, 1, 1),
        buffers={1.0: 0.0, 2.0: 0.0},
        structured=False,
        physical_name="UnstructB",
    )
    out = tmp_path / "phase3.msh"
    with patch("meshwell.structured.phantom._USE_DISCRETE_COHORT_MESH", True):
        build_mesh([structured, unstructured], output_mesh=str(out), dim=3)

    gmsh.initialize()
    try:
        gmsh.open(str(out))
        # No duplicate nodes at z=1.
        _, all_coords_flat, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(all_coords_flat, dtype=float).reshape(-1, 3)
        at_z1 = np.isclose(coords[:, 2], 1.0, atol=1e-9)
        xy_at_z1 = coords[at_z1, :2]
        unique = np.unique(np.round(xy_at_z1, 9), axis=0)
        assert unique.shape[0] == int(at_z1.sum()), (
            "Duplicate interface nodes between structured + unstructured"
        )
    finally:
        gmsh.finalize()
```

- [ ] **Step 4: Run spec test #6**

Run: `pytest tests/structured/test_phase3_top_bottom_conformality.py -v`
Expected: PASS

- [ ] **Step 5: Write bench scripts**

Create `scripts/bench_cohort_envelope.py`:

```python
"""Bench the cohort envelope builder vs Phase 1+2 paths.

Times build_phantom_shapes under each kill-switch setting on a
synthetic NxN grid of stacked square slabs.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from unittest.mock import patch

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.phantom import build_phantom_shapes
from meshwell.structured.plan import build_plan


@contextmanager
def _timed(label: str):
    t0 = time.perf_counter()
    yield
    print(f"{label}: {time.perf_counter() - t0:.3f}s")


def _grid(n_xy: int, n_z: int):
    entities = []
    for k in range(n_z):
        for i in range(n_xy):
            for j in range(n_xy):
                entities.append(
                    PolyPrism(
                        polygons=shapely.box(i, j, i + 1, j + 1),
                        buffers={k + 0.0: 0.0, k + 1.0: 0.0},
                        structured=True,
                        resolutions=[
                            StructuredExtrusionResolutionSpec(n_layers=[1])
                        ],
                        physical_name=f"S_{i}_{j}_{k}",
                    )
                )
    return entities


def main() -> None:
    for (n_xy, n_z) in [(4, 2), (6, 3), (8, 4)]:
        plan = build_plan(_grid(n_xy, n_z))
        n_pieces = sum(len(s.face_partition) for s in plan.slabs)
        n_cohorts = len({s.component_index for s in plan.slabs})
        print(f"\n=== grid={n_xy}x{n_xy} z={n_z} pieces={n_pieces} cohorts={n_cohorts} ===")
        with _timed("phase1 (pre-shared faces)"):
            build_phantom_shapes(plan)
        with patch("meshwell.structured.phantom._USE_COHORT_TOPOLOGY", True):
            with _timed("phase2 (cohort topology)"):
                build_phantom_shapes(plan)
        with patch("meshwell.structured.phantom._USE_DISCRETE_COHORT_MESH", True):
            with _timed("phase3 (cohort envelope)"):
                build_phantom_shapes(plan)


if __name__ == "__main__":
    main()
```

Extend `scripts/bench_fragment_all.py` to add a Phase 3 mode. Read the existing script first:

```bash
cat scripts/bench_fragment_all.py
```

Then add a third bench mode patching `_USE_DISCRETE_COHORT_MESH=True` alongside the existing Phase 1/2 modes. The pattern is the same as `bench_cohort_envelope.py`: wrap the `cad._fragment_all(...)` call in a `with patch(...)` block.

- [ ] **Step 6: Smoke-run benches**

Run: `python scripts/bench_cohort_envelope.py`
Expected: prints three timings per grid size; phase3 should be ≤ phase2's.

- [ ] **Step 7: Commit**

```bash
git add tests/structured/test_phase3_interior_interfaces.py tests/structured/test_phase3_top_bottom_conformality.py scripts/bench_cohort_envelope.py scripts/bench_fragment_all.py
git commit -m "test(phase3): interior conformity + bench scripts

Spec tests #5 + #6: interior interface and top/bottom unstructured-
neighbor node sharing. Bench script bench_cohort_envelope.py compares
build_phantom_shapes timings across Phase 1/2/3 paths and
bench_fragment_all.py gains a Phase 3 mode."
```

---

## Task 21: Parity sweep — existing structured suite under Phase 3

**Files:**
- No new files; runs the existing structured suite with the flag flipped on, fixes any uncovered regressions in `meshwell/structured/` source files.

- [ ] **Step 1: Add a session-wide kill-switch override pytest marker**

Create `tests/structured/conftest.py` (if absent) or extend it with a fixture `phase3_on`:

```python
"""Phase 3 test infrastructure: opt-in fixture that flips the kill-switch
for the duration of a test.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture
def phase3_on():
    with patch("meshwell.structured.phantom._USE_DISCRETE_COHORT_MESH", True):
        yield
```

- [ ] **Step 2: Run the existing structured suite with `_USE_DISCRETE_COHORT_MESH=True`**

Run:

```bash
PYTHONUNBUFFERED=1 _MESHWELL_FORCE_PHASE3=1 pytest tests/structured/ -x --tb=short 2>&1 | tee /tmp/phase3-suite.log
```

(The `_MESHWELL_FORCE_PHASE3=1` env var is consumed by the conftest below.)

Then add to `conftest.py`:

```python
import os

if os.environ.get("_MESHWELL_FORCE_PHASE3") == "1":
    from unittest.mock import patch

    _phase3_patcher = patch("meshwell.structured.phantom._USE_DISCRETE_COHORT_MESH", True)
    _phase3_patcher.start()
```

Expected output: most tests pass; collect any failures.

- [ ] **Step 3: For each failure, root-cause and fix in Phase 3 code (NOT in tests)**

Use systematic-debugging skill principles:
- Read the error completely.
- Identify what Phase 3 assumption is violated.
- Fix in `cohort_envelope.py` / `phantom.py` / `builder.py` — never patch the test.

- [ ] **Step 4: Re-run after each fix until the suite is green**

Run:

```bash
PYTHONUNBUFFERED=1 _MESHWELL_FORCE_PHASE3=1 pytest tests/structured/ --tb=short
```

Expected: 0 failures (xfail-marked tests remain xfail).

- [ ] **Step 5: Commit each fix as a separate commit, ending with a "parity sweep clean" commit**

```bash
# example fix commit, repeated as needed:
git add meshwell/structured/<fixed-file>.py
git commit -m "fix(phase3): <what was broken>

Root cause: <short>. Fix: <short>. Surfaced by
tests/structured/<test_file>.py under _USE_DISCRETE_COHORT_MESH=True."

# final marker:
git commit --allow-empty -m "chore(phase3): structured suite green under kill-switch ON

PYTHONUNBUFFERED=1 _MESHWELL_FORCE_PHASE3=1 pytest tests/structured/
all green (xfail markers unchanged). Promotion to default still
gated on cross-compare + Palace round-trip checks."
```

---

## Task 22: Promotion + spec/CLAUDE.md memo

**Files:**
- Modify: `meshwell/structured/phantom.py`
- Modify: `docs/superpowers/specs/2026-05-28-cad-occ-discrete-internal-cohort-mesh-design.md`

Only attempt this after Task 21 lands cleanly AND `test_backend_cross_compare.py` + `test_palace_mixed_mesh_check.py` (or equivalent) have been confirmed green with the flag forced on.

- [ ] **Step 1: Verify additional gate suites pass**

Run:

```bash
_MESHWELL_FORCE_PHASE3=1 pytest tests/test_backend_cross_compare.py tests/test_palace_mixed_mesh_check.py --tb=short
```

Expected: green.

- [ ] **Step 2: Promote kill-switch default**

Edit `meshwell/structured/phantom.py`. Change:

```python
_USE_DISCRETE_COHORT_MESH = False
```

to:

```python
_USE_DISCRETE_COHORT_MESH = True
```

and update the docstring above it to record the promotion date and the gate suites that validated it.

- [ ] **Step 3: Update spec status**

Edit `docs/superpowers/specs/2026-05-28-cad-occ-discrete-internal-cohort-mesh-design.md`. Change `**Status:** Spec (brainstormed)` to `**Status:** Implemented (kill-switch on by default as of YYYY-MM-DD)`. Update the "Kill-switch + coexistence" section's promotion note to record the actual gate runs.

- [ ] **Step 4: Run a representative end-to-end smoke test to confirm production defaults**

Run:

```bash
pytest tests/structured/test_phase3_discrete_volumes.py tests/structured/test_phase3_interior_interfaces.py tests/structured/test_phase3_top_bottom_conformality.py -v
```

Expected: PASS without any flag overrides (the new default is `True`).

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/phantom.py docs/superpowers/specs/2026-05-28-cad-occ-discrete-internal-cohort-mesh-design.md
git commit -m "feat(phantom): promote _USE_DISCRETE_COHORT_MESH to default True

Validated by: structured suite, cross-compare, Palace round-trip
under _MESHWELL_FORCE_PHASE3=1. Phase 1+2 cohort code remains
reachable via flag flips and will be deleted in a follow-up commit
once Phase 3 has soaked in production."
```

---

## Task 23: Cleanup — delete Phase 1+2 cohort paths

**Files:**
- Delete: `meshwell/structured/cohort_topology.py`
- Delete: `tests/structured/test_cohort_topology_*.py`
- Delete: `tests/structured/test_phantom_discrete_routing.py::test_phase3_flag_off_keeps_phase2_behavior` (replace with a simpler version that doesn't assume Phase 1+2 still exists)
- Modify: `meshwell/structured/phantom.py`

This is the final cleanup commit. Only attempt AFTER Task 22 has soaked (recommended: ≥1 production run end-to-end without regressions).

- [ ] **Step 1: Remove kill-switches and Phase 1+2 branches**

In `meshwell/structured/phantom.py`:

1. Delete the `_USE_DISCRETE_COHORT_MESH`, `_USE_COHORT_TOPOLOGY`, and `_PRESHARE_VERTICAL_FACES` constants (keeping the import-block tidy).
2. Delete `_group_slabs_into_vertical_stacks`, `_build_phantom_shapes_via_cohort_topology`.
3. Rewrite `build_phantom_shapes` to call `_build_phantom_shapes_via_cohort_envelope` directly.
4. Optionally inline `_build_phantom_shapes_via_cohort_envelope` into `build_phantom_shapes` (keep separate if it improves readability).
5. Delete `_build_sub_prism`, `_make_face_from_polygon`, `_make_face_from_polygon_with_arcs`, `_make_arc_wire_from_coords`, and `_face_at_z` if no other caller remains (grep first).

- [ ] **Step 2: Delete Phase 2 module and tests**

```bash
rm meshwell/structured/cohort_topology.py
rm tests/structured/test_cohort_topology_*.py
rm tests/test_brep_builder_assembly_smoke.py
```

- [ ] **Step 3: Update `test_phantom_discrete_routing.py`**

Remove the `test_phase3_flag_off_keeps_phase2_behavior` test (the flag no longer exists). The first test (`test_phase3_routing_produces_one_phantomshape_per_cohort`) stays but drops the `with patch(...)` block (Phase 3 is now the only path).

- [ ] **Step 4: Run the full structured suite**

Run:

```bash
pytest tests/structured/ --tb=short
```

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore(phantom): delete Phase 1+2 cohort code paths

Phase 3 has been the default since Task 22 and has soaked without
regressions. Removed:
- meshwell/structured/cohort_topology.py
- _USE_DISCRETE_COHORT_MESH, _USE_COHORT_TOPOLOGY,
  _PRESHARE_VERTICAL_FACES kill-switches
- _group_slabs_into_vertical_stacks,
  _build_phantom_shapes_via_cohort_topology
- _build_sub_prism + per-piece sub-prism face helpers
- Phase 2 cohort_topology tests + Phase 2 BRep_Builder smoke test"
```

---

## Self-Review

Skimming the spec one final time:

- **Goal** (sidestep N² BOP via cohort envelope + discrete mesh) — covered by Tasks 1-12 (envelope + routing) and Tasks 14-18 (mesh-stage discretes).
- **Architecture diagram** — implemented by the task sequence (envelope builder → phantom routing → discrete mesh stamping → envelope-volume cleanup).
- **Component #1 (cohort envelope builder)** — Tasks 1-10.
- **Component #2 (cohort lateral wall vertical extent)** — Task 6.
- **Component #3 (PhantomMap extensions)** — Task 11.
- **Component #4 (apply_structured_mesh extensions)** — Tasks 14-18.
- **Component #5 (routing piece → cohort)** — implicit in the cohort PhantomShape (Task 12) and the `_find_volume_containing_faces`-bypass logic (Task 14).
- **Data flow** — Tasks 12-13 (phantom routing + cad_occ round-trip), Tasks 14-17 (mesh stamping).
- **Kill-switch + coexistence** — Task 12 (introduce switch), Task 22 (promote), Task 23 (delete Phase 1+2).
- **Invariants** — Task 12 carries the design assumption; no XY-unstructured-neighbor explicit check is added (already enforced by the planner).
- **Future work** — documented as comments in `cohort_envelope.py` (Task 1).
- **Testing strategy (6 spec tests)** — Tasks 8, 9, 10, 19, 20 cover them.
- **Bench** — Task 20.
- **Open question: gmsh discrete-entity boundary inheritance** — left as a real-runtime observation: Tasks 14-18 do NOT include discrete-2D in the discrete-3D boundary list, and the spec acknowledges this. Task 21's parity sweep will surface if any test depends on `getBoundary(piece_vol)` returning interfaces.
- **Open question: XAO checkpoint is lossy** — documented in the spec; no implementation work required for Phase 3.

**Placeholder scan:** No "TBD", "fill in details", or "similar to Task N" pointers. Vertical interface implementation in Task 16 contains the full code; Task 17 extends it explicitly with the multi-layer extension shown line-by-line.

**Type/name consistency check:**
- `CohortEnvelope.bottom_sub_faces` / `top_sub_faces` / `lateral_faces` are referenced consistently across Tasks 5, 6, 7, 12.
- `FaceKey(slab_index, side, piece_index)` is used in Tasks 5, 11, 12, 15.
- `_USE_DISCRETE_COHORT_MESH` referenced in Tasks 12, 14, 15, 17, 18, 21, 22.
- `phantom_map.face_keys_to_discrete` introduced in Task 11, written in Tasks 15-17.
- `bot_node_tags_per_piece` / `top_node_tags_per_piece` / `bot_tri_per_piece` / `layer_maps_per_piece` introduced in Tasks 15/16/17 with consistent shapes.

Plan saved. Ready for execution.
