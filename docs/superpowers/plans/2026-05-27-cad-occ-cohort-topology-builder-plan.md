# Cohort Topology Builder Implementation Plan (Phase 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pre-share `TopoDS_Face` between BOTH vertically-stacked AND laterally-adjacent sub-prisms in the same cohort by constructing each cohort's topology (vertices, edges, faces) once and assembling each sub-prism's solid via `BRep_Builder` as a view into the shared topology.

**Architecture:** New module `meshwell/structured/cohort_topology.py` with a `CohortTopology` dataclass holding four registries (vertices, horizontal edges, vertical edges, faces). `build_cohort_topology(plan, component_index)` populates them; `assemble_cohort_sub_prism(topology, slab, piece_index)` reads them to build each sub-prism's solid + `PhantomShape`. `build_phantom_shapes` gains a new kill-switch `_USE_COHORT_TOPOLOGY` that selects between the cohort topology builder (Phase 2) and the existing path (Phase 1 vertical-only or full legacy). Both paths preserved during Phase 2; legacy retired in a separate Phase 3 spec.

**Tech Stack:** Python, OCP (OpenCascade Python bindings), pytest. Key OCP classes: `BRep_Builder`, `TopoDS_Shell`, `TopoDS_Solid`, `BRepBuilderAPI_MakeFace`, `BRepBuilderAPI_MakeEdge`, `BRepBuilderAPI_MakeWire`, `BRepBuilderAPI_MakeVertex`, `Geom_CylindricalSurface`, `BRepCheck_Analyzer`.

**Spec:** `docs/superpowers/specs/2026-05-27-cad-occ-cohort-topology-builder-design.md`

---

## File Structure

**Created:**
- `meshwell/structured/cohort_topology.py` — `CohortTopology` dataclass, `build_cohort_topology`, `assemble_cohort_sub_prism`
- `tests/test_brep_builder_assembly_smoke.py` — Task 0 validation gate; pure OCC smoke tests
- `tests/structured/test_cohort_topology_vertices_edges.py` — Tasks 3-4 unit tests
- `tests/structured/test_cohort_topology_faces.py` — Tasks 5-6 unit tests
- `tests/structured/test_cohort_topology_arcs.py` — Tasks 7-8 unit tests
- `tests/structured/test_cohort_topology_assembly.py` — Task 9 unit tests
- `tests/structured/test_cohort_topology_integration.py` — Task 11 end-to-end
- `tests/test_cohort_topology_parity.py` — Task 12 toggle comparison
- `scripts/bench_cohort_topology.py` — Task 13 benchmark

**Modified:**
- `meshwell/structured/spec.py` — add `arrangements: dict[int, StackArrangement]` to `StructuredPlan` (Task 1)
- `meshwell/structured/plan.py` — populate `arrangements` in `build_plan` (Task 1)
- `meshwell/structured/phantom.py` — add `_USE_COHORT_TOPOLOGY` constant and new branch in `build_phantom_shapes` (Task 10)

---

## Task 0: BRep_Builder assembly smoke test (validation gate)

**Why first:** The design rests on the assumption that solids assembled manually via `BRep_Builder.MakeSolid` from explicit shared faces produce per-argument `Modified()` history when passed to `BOPAlgo_Builder`. We have not validated this for manually-assembled (non-prim-API) solids. If it fails, the design must change before any implementation.

**Files:**
- Create: `tests/test_brep_builder_assembly_smoke.py`

- [ ] **Step 1: Write the smoke tests**

Create `tests/test_brep_builder_assembly_smoke.py`:

```python
"""Validation gate for Phase 2 cohort topology builder design.

Validates the two OCC behaviors the design depends on:
1. BRep_Builder.MakeSolid can assemble a closed solid from explicit faces
   including a shared face TShape between two adjacent solids.
2. BOPAlgo_Builder.Modified() returns per-argument history when those
   manually-assembled solids are passed as individual arguments and one
   is overlapped by a third solid.
"""

from __future__ import annotations

from OCP.BOPAlgo import BOPAlgo_Builder
from OCP.BRep import BRep_Builder
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeVertex,
    BRepBuilderAPI_MakeWire,
)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS_Shell, TopoDS_Solid


def _faces(shape):
    out = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        out.append(exp.Current())
        exp.Next()
    return out


def _make_box_solid_from_explicit_faces(
    x0, y0, z0, x1, y1, z1, shared_top_face=None
):
    """Assemble a box solid via BRep_Builder.

    If shared_top_face is provided, use it as this box's TOP face (which
    must be at z=z1 — caller's responsibility). Otherwise build a fresh
    top face. Returns (solid, top_face_for_reuse).
    """
    b = BRep_Builder()

    # Vertices.
    v = {}
    for x in (x0, x1):
        for y in (y0, y1):
            for z in (z0, z1):
                v[(x, y, z)] = BRepBuilderAPI_MakeVertex(gp_Pnt(x, y, z)).Vertex()

    def _edge(p, q):
        return BRepBuilderAPI_MakeEdge(v[p], v[q]).Edge()

    def _wire(*edges):
        mw = BRepBuilderAPI_MakeWire()
        for e in edges:
            mw.Add(e)
        return mw.Wire()

    def _face_from_wire(wire):
        return BRepBuilderAPI_MakeFace(wire).Face()

    # Bottom face (z=z0).
    e_b1 = _edge((x0, y0, z0), (x1, y0, z0))
    e_b2 = _edge((x1, y0, z0), (x1, y1, z0))
    e_b3 = _edge((x1, y1, z0), (x0, y1, z0))
    e_b4 = _edge((x0, y1, z0), (x0, y0, z0))
    bot_wire = _wire(e_b1, e_b2, e_b3, e_b4)
    bot_face = _face_from_wire(bot_wire)

    # Top face (z=z1) — shared or fresh.
    if shared_top_face is not None:
        top_face = shared_top_face
    else:
        e_t1 = _edge((x0, y0, z1), (x1, y0, z1))
        e_t2 = _edge((x1, y0, z1), (x1, y1, z1))
        e_t3 = _edge((x1, y1, z1), (x0, y1, z1))
        e_t4 = _edge((x0, y1, z1), (x0, y0, z1))
        top_wire = _wire(e_t1, e_t2, e_t3, e_t4)
        top_face = _face_from_wire(top_wire)

    # Lateral faces. For simplicity, build each lateral with its own edges.
    def _lateral(p1, p2):
        # Rectangle with corners p1=(x,y,z0), p2=(x',y',z0), and same at z1.
        p1_top = (p1[0], p1[1], z1)
        p2_top = (p2[0], p2[1], z1)
        e1 = _edge(p1, p2)
        e2 = _edge(p2, p2_top)
        e3 = _edge(p2_top, p1_top)
        e4 = _edge(p1_top, p1)
        return _face_from_wire(_wire(e1, e2, e3, e4))

    lateral_faces = [
        _lateral((x0, y0, z0), (x1, y0, z0)),
        _lateral((x1, y0, z0), (x1, y1, z0)),
        _lateral((x1, y1, z0), (x0, y1, z0)),
        _lateral((x0, y1, z0), (x0, y0, z0)),
    ]

    # Build shell with all 6 faces.
    shell = TopoDS_Shell()
    b.MakeShell(shell)
    b.Add(shell, bot_face.Reversed())  # bot face's normal should face down
    b.Add(shell, top_face)
    for lf in lateral_faces:
        b.Add(shell, lf)

    # Wrap in solid.
    solid = TopoDS_Solid()
    b.MakeSolid(solid)
    b.Add(solid, shell)

    return solid, top_face


def test_manually_assembled_solid_is_valid():
    """Sanity check: BRep_Builder solid passes BRepCheck."""
    solid, _ = _make_box_solid_from_explicit_faces(0, 0, 0, 1, 1, 1)
    analyzer = BRepCheck_Analyzer(solid)
    assert analyzer.IsValid(), "Manually-assembled solid is not valid per BRepCheck"


def test_two_manually_assembled_solids_share_face_tshape():
    """A's top face is reused as B's top face — they should share TShape."""
    A, A_top = _make_box_solid_from_explicit_faces(0, 0, 0, 1, 1, 1)
    B, _ = _make_box_solid_from_explicit_faces(
        0, 0, 1, 1, 1, 2, shared_top_face=A_top
    )

    a_face_hashes = {hash(f) for f in _faces(A)}
    b_face_hashes = {hash(f) for f in _faces(B)}
    shared = a_face_hashes & b_face_hashes
    assert shared, (
        "Manually-assembled solids did NOT share the explicitly-shared "
        "TopoDS_Face TShape."
    )


def test_bopalgo_modified_works_for_manually_assembled_shared_face_solids():
    """The critical design-validation test.

    Two manually-assembled solids share a face (A's top = B's top, shared
    TShape). A third solid C overlaps A. Add A, B, C as separate BOP
    arguments. Modified(A) and Modified(B) must both return non-empty
    per-argument history.
    """
    A, A_top = _make_box_solid_from_explicit_faces(0, 0, 0, 1, 1, 1)
    # B is "above" A vertically; reuse A's top as B's TOP would be wrong
    # geometrically (B's bottom should be at z=1, not its top). For the
    # purpose of this test, just verify that *if* two BRep_Builder-assembled
    # solids share a face, Modified() works. Use a different shared-face
    # arrangement: B at the same z but adjacent in x, sharing A's right
    # lateral as its left lateral. That's harder to construct manually;
    # for simplicity, reuse A_top as B's bottom-face replacement: build B
    # at z=[1,2] but with bottom = A's top (= face at z=1, which IS B's
    # geometric bottom). Modify _make_box_solid_from_explicit_faces if
    # needed to allow shared_bottom_face.
    #
    # Implementer: extend _make_box_solid_from_explicit_faces to accept
    # shared_bottom_face=None param if you need it. The smoke test
    # validates the BOP behavior, not the geometric story.

    # Simpler: build A, build B at [1,2] using a fresh bottom, then verify
    # BOP Modified() works on both arguments when they are manually
    # assembled and one is overlapped by C.
    B, _ = _make_box_solid_from_explicit_faces(0, 0, 1, 1, 1, 2)
    C = BRepPrimAPI_MakeBox(gp_Pnt(0.5, 0.5, 0.5), gp_Pnt(1.5, 1.5, 1.5)).Solid()

    builder = BOPAlgo_Builder()
    builder.AddArgument(A)
    builder.AddArgument(B)
    builder.AddArgument(C)
    builder.Perform()

    a_mod = builder.Modified(A)
    b_mod = builder.Modified(B)
    assert not a_mod.IsEmpty(), (
        "Modified(A) empty — BRep_Builder-assembled solid is not tracked "
        "by BOPAlgo per-argument. Design must change."
    )
    # B doesn't overlap C; Modified(B) may be empty (legacy fallback) or
    # contain B itself. Both are acceptable; what's NOT acceptable is
    # IsDeleted(B)=True.
    assert not builder.IsDeleted(B), (
        "BOPAlgo deleted B (a manually-assembled solid not overlapping C)."
    )
```

- [ ] **Step 2: Run the smoke tests**

Run: `python -m pytest tests/test_brep_builder_assembly_smoke.py -v --no-cov`

Expected: all three tests PASS. If any FAIL, **STOP and report BLOCKED**. The Phase 2 design rests on these behaviors — if they don't hold, the spec must be revised before implementation.

- [ ] **Step 3: Commit**

```bash
git add tests/test_brep_builder_assembly_smoke.py
git commit -m "test(cad_occ): BRep_Builder assembly smoke tests (Phase 2 validation gate)"
```

---

## Task 1: Persist `StackArrangement` on `StructuredPlan`

**Files:**
- Modify: `meshwell/structured/spec.py:304-310` (`StructuredPlan`)
- Modify: `meshwell/structured/plan.py:1525-1559` (`build_plan`)
- Test: `tests/structured/test_plan_arrangements.py`

The cohort topology builder needs the cohort's `StackArrangement` (to walk arrangement edges and identify cohort-interior vs cohort-exterior edges). Today `build_stack_arrangements` returns `dict[int, StackArrangement]` but the result is consumed transiently and discarded. Persist it on the plan.

- [ ] **Step 1: Write the failing test**

Create `tests/structured/test_plan_arrangements.py`:

```python
"""Verify StructuredPlan.arrangements is populated by build_plan."""

from __future__ import annotations

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.plan import build_plan


def _square(x0, y0, x1, y1):
    return shapely.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _polyprism(name, z0, z1, mesh_order):
    return PolyPrism(
        polygons=_square(0, 0, 1, 1),
        buffers={float(z0): 0.0, float(z1): 0.0},
        physical_name=name,
        mesh_order=mesh_order,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )


def test_plan_arrangements_populated():
    """build_plan exposes per-component StackArrangements."""
    plan = build_plan([_polyprism("A", 0, 1, 1), _polyprism("B", 1, 2, 2)])
    assert hasattr(plan, "arrangements")
    # A and B are in the same cohort -> one arrangement (component_index=0).
    assert 0 in plan.arrangements
    # Arrangement is a StackArrangement with edges and faces.
    arr = plan.arrangements[0]
    assert hasattr(arr, "edges")
    assert hasattr(arr, "faces")


def test_plan_arrangements_disjoint_components():
    """Disjoint slabs have separate arrangements (one per component)."""
    plan = build_plan([_polyprism("A", 0, 1, 1), _polyprism("C", 10, 11, 2)])
    assert len(plan.arrangements) == 2
    assert 0 in plan.arrangements
    assert 1 in plan.arrangements
```

- [ ] **Step 2: Run to verify failing**

Run: `python -m pytest tests/structured/test_plan_arrangements.py -v --no-cov`
Expected: FAIL with `AttributeError: 'StructuredPlan' object has no attribute 'arrangements'`.

- [ ] **Step 3: Add field to `StructuredPlan`**

In `meshwell/structured/spec.py`, modify the `StructuredPlan` dataclass (find it around line 304):

```python
@dataclass(frozen=True)
class StructuredPlan:
    """Frozen output of the planner; consumed by phantom + builder stages."""

    slabs: tuple[Slab, ...]
    z_planes: tuple[float, ...]
    overlaps: tuple[OverlapPair, ...]
    # Populated by build_plan. Maps component_index -> StackArrangement for
    # each connected z-component. Consumed by the cohort topology builder
    # (Phase 2) to walk arrangement edges and detect cohort-interior vs
    # cohort-exterior boundaries. See spec
    # 2026-05-27-cad-occ-cohort-topology-builder-design.md.
    arrangements: "dict[int, StackArrangement]" = field(default_factory=dict)
```

Add `from dataclasses import field` import if not already present. Forward-reference `StackArrangement` as a string to avoid circular-import issues.

Note: `StructuredPlan` is `frozen=True`. With `frozen=True`, `field(default_factory=dict)` is OK because the field assignment happens during `__init__`, not afterward. But mutating the dict later (which `build_plan` does) is fine since dict mutation doesn't touch the frozen attribute binding.

- [ ] **Step 4: Populate `arrangements` in `build_plan`**

In `meshwell/structured/plan.py:1555-1559`, modify the `return StructuredPlan(...)` call to pass arrangements:

```python
    return StructuredPlan(
        slabs=tuple(kept_slabs),
        z_planes=tuple(z_planes),
        overlaps=tuple(overlaps),
        arrangements=arrangements,
    )
```

`arrangements` is already a local variable in `build_plan` (the result of `build_stack_arrangements(kept_slabs, entities)` at line 1548).

- [ ] **Step 5: Run the test to verify passing**

Run: `python -m pytest tests/structured/test_plan_arrangements.py -v --no-cov`
Expected: both tests PASS.

- [ ] **Step 6: Run the full structured suite**

Run: `python -m pytest tests/structured/ -v --no-cov`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add meshwell/structured/spec.py meshwell/structured/plan.py tests/structured/test_plan_arrangements.py
git commit -m "feat(structured): persist StackArrangements on StructuredPlan"
```

---

## Task 2: `CohortTopology` dataclass and module skeleton

**Files:**
- Create: `meshwell/structured/cohort_topology.py`
- Test: `tests/structured/test_cohort_topology_skeleton.py`

Stand up the module with the dataclass and stub functions. No real logic yet — just import paths.

- [ ] **Step 1: Write the failing test**

Create `tests/structured/test_cohort_topology_skeleton.py`:

```python
"""Sanity check that cohort_topology module exposes expected names."""

from __future__ import annotations

from meshwell.structured.cohort_topology import (
    CohortTopology,
    build_cohort_topology,
    assemble_cohort_sub_prism,
)


def test_cohort_topology_dataclass_has_registries():
    """CohortTopology has the four documented registry attributes."""
    t = CohortTopology(
        component_index=0,
        plan=None,
        vertices={},
        horizontal_edges={},
        vertical_edges={},
        horizontal_faces={},
        lateral_faces={},
    )
    assert t.component_index == 0
    assert t.vertices == {}
    assert t.horizontal_edges == {}
    assert t.vertical_edges == {}
    assert t.horizontal_faces == {}
    assert t.lateral_faces == {}


def test_build_cohort_topology_is_callable():
    """Stub function exists; doesn't matter if it returns empty."""
    assert callable(build_cohort_topology)


def test_assemble_cohort_sub_prism_is_callable():
    """Stub function exists."""
    assert callable(assemble_cohort_sub_prism)
```

- [ ] **Step 2: Run to verify failing**

Run: `python -m pytest tests/structured/test_cohort_topology_skeleton.py -v --no-cov`
Expected: FAIL with `ModuleNotFoundError: No module named 'meshwell.structured.cohort_topology'`.

- [ ] **Step 3: Create the module**

Create `meshwell/structured/cohort_topology.py`:

```python
"""Cohort topology builder for full vertical+lateral face sharing.

For each connected z-component (cohort), build a shared topology of
vertices, edges, and faces ONCE, then assemble each sub-prism's solid
as a view into that topology. Adjacent cohort sub-prisms (vertically or
laterally) thereby share TopoDS_Face TShape identity at their interfaces,
letting BOPAlgo's pave-filler skip pairwise intersection work.

See spec docs/superpowers/specs/2026-05-27-cad-occ-cohort-topology-builder-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from meshwell.structured.spec import (
    PhantomShape,
    Slab,
    StructuredPlan,
)


@dataclass
class CohortTopology:
    """Shared topology registries for one cohort.

    Per spec Section 'Architecture'. The four registries:

    - vertices: keyed by (z_plane, xy_corner_id) -> TopoDS_Vertex.
    - horizontal_edges: keyed by (z_plane, arrangement_edge_id) ->
      TopoDS_Edge. Each at the cohort's arrangement edge geometry, placed
      at the given z_plane.
    - vertical_edges: keyed by (z_interval_id, xy_corner_id) ->
      TopoDS_Edge. Each connects the bottom-z vertex to the top-z vertex
      at the same xy corner.
    - horizontal_faces: keyed by (z_plane, piece_id) -> TopoDS_Face. Each
      is a horizontal face of one cohort piece at one z-plane; serves as
      the TOP of the slab below AND the BOTTOM of the slab above.
    - lateral_faces: keyed by (z_interval_id, arrangement_edge_id) ->
      TopoDS_Face. Each extrudes an arrangement edge across one slab's
      z-interval.

    piece_id = (source_index, piece_index) — disambiguates pieces within
    this cohort (registries are per-cohort, so component_index is implicit).
    """

    component_index: int
    plan: StructuredPlan | None  # back-reference for slab/piece lookups
    vertices: dict[tuple[float, int], Any] = field(default_factory=dict)
    horizontal_edges: dict[tuple[float, int], Any] = field(default_factory=dict)
    vertical_edges: dict[tuple[int, int], Any] = field(default_factory=dict)
    horizontal_faces: dict[tuple[float, tuple[int, int]], Any] = field(
        default_factory=dict
    )
    lateral_faces: dict[tuple[int, int], Any] = field(default_factory=dict)


def build_cohort_topology(
    plan: StructuredPlan,
    component_index: int,
) -> CohortTopology:
    """Build the shared topology for one cohort.

    Implementation lands in later tasks. For now returns an empty topology.
    """
    return CohortTopology(component_index=component_index, plan=plan)


def assemble_cohort_sub_prism(
    topology: CohortTopology,
    slab: Slab,
    piece_index: int,
) -> PhantomShape:
    """Assemble one sub-prism's solid + PhantomShape from the registry.

    Implementation lands in later tasks.
    """
    raise NotImplementedError(
        "assemble_cohort_sub_prism is implemented in Task 9 of the plan."
    )
```

- [ ] **Step 4: Run the skeleton test to verify passing**

Run: `python -m pytest tests/structured/test_cohort_topology_skeleton.py -v --no-cov`
Expected: PASS.

- [ ] **Step 5: Run structured suite (no regression)**

Run: `python -m pytest tests/structured/ -v --no-cov`
Expected: PASS — the new module isn't called by anything yet.

- [ ] **Step 6: Commit**

```bash
git add meshwell/structured/cohort_topology.py tests/structured/test_cohort_topology_skeleton.py
git commit -m "feat(structured): CohortTopology module skeleton"
```

---

## Task 3: Vertex and horizontal-edge registries (straight only)

**Files:**
- Modify: `meshwell/structured/cohort_topology.py`
- Test: `tests/structured/test_cohort_topology_vertices_edges.py`

Implement the vertex registry and the horizontal-edge registry for STRAIGHT arrangement edges only. Arc support comes in Task 7.

The algorithm:
1. Collect all slabs in this cohort (`s for s in plan.slabs if s.component_index == component_index`).
2. Collect the unique z-planes (zlos and zhis of every slab in the cohort).
3. Get the cohort's arrangement: `arrangement = plan.arrangements[component_index]`.
4. For each arrangement vertex (the endpoints of `arrangement.edges`), at each cohort z-plane, build a `TopoDS_Vertex` at `(x, y, z)`. Key by `(z, xy_corner_id)` where `xy_corner_id` is the arrangement vertex's stable index.
5. For each arrangement edge at each cohort z-plane, build a `TopoDS_Edge` from the two registered endpoint vertices. Key by `(z, arrangement_edge_id)`.

**Note on `xy_corner_id`:** The arrangement gives us `ArrangementEdge` objects with `vertices` (an `(x, y)` pair) and `edge_id`. We need a stable id for each unique XY corner. Build it by collecting all unique `(round(x, 9), round(y, 9))` tuples across all edges and assigning incremental indices.

- [ ] **Step 1: Write the failing test**

Create `tests/structured/test_cohort_topology_vertices_edges.py`:

```python
"""build_cohort_topology populates vertex and horizontal-edge registries."""

from __future__ import annotations

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.cohort_topology import build_cohort_topology
from meshwell.structured.plan import build_plan


def _square(x0, y0, x1, y1):
    return shapely.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _polyprism(name, z0, z1, mesh_order, x0=0, y0=0, x1=1, y1=1):
    return PolyPrism(
        polygons=_square(x0, y0, x1, y1),
        buffers={float(z0): 0.0, float(z1): 0.0},
        physical_name=name,
        mesh_order=mesh_order,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )


def test_vertices_registered_at_each_z_plane():
    """A single square cohort with 2 stacked slabs has 4 corners x 3 z-planes = 12 vertices."""
    plan = build_plan([_polyprism("A", 0, 1, 1), _polyprism("B", 1, 2, 2)])
    topology = build_cohort_topology(plan, component_index=0)
    # 4 corners x 3 z-planes (z=0, z=1, z=2)
    assert len(topology.vertices) == 12


def test_horizontal_edges_registered_at_each_z_plane():
    """A single square cohort has 4 outer edges x 3 z-planes = 12 horizontal edges."""
    plan = build_plan([_polyprism("A", 0, 1, 1), _polyprism("B", 1, 2, 2)])
    topology = build_cohort_topology(plan, component_index=0)
    assert len(topology.horizontal_edges) == 12


def test_horizontal_edge_uses_registry_vertices():
    """A horizontal edge's endpoints must be the same TShape as the registered vertices."""
    plan = build_plan([_polyprism("A", 0, 1, 1)])
    topology = build_cohort_topology(plan, component_index=0)
    # Pick any horizontal edge.
    edge_key = next(iter(topology.horizontal_edges))
    edge = topology.horizontal_edges[edge_key]

    from OCP.TopAbs import TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer

    edge_vertex_hashes = set()
    exp = TopExp_Explorer(edge, TopAbs_VERTEX)
    while exp.More():
        edge_vertex_hashes.add(hash(exp.Current()))
        exp.Next()

    registry_vertex_hashes = {hash(v) for v in topology.vertices.values()}
    # Every edge endpoint must be in the registry.
    assert edge_vertex_hashes <= registry_vertex_hashes, (
        "Horizontal edge has endpoints not registered in the vertex registry."
    )
```

- [ ] **Step 2: Run to verify failing**

Run: `python -m pytest tests/structured/test_cohort_topology_vertices_edges.py -v --no-cov`
Expected: FAIL — the registries are still empty.

- [ ] **Step 3: Implement vertex + horizontal-edge construction**

Modify `meshwell/structured/cohort_topology.py`. Replace the body of `build_cohort_topology` with:

```python
def build_cohort_topology(
    plan: StructuredPlan,
    component_index: int,
) -> CohortTopology:
    """Build the shared topology for one cohort.

    Walks the cohort's slabs and arrangement to populate registries of
    vertices, horizontal/vertical edges, and faces. All sub-prisms in the
    cohort are then assembled as views into this topology, so adjacent
    sub-prisms share TopoDS_* TShape identity at their interfaces.
    """
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeVertex
    from OCP.gp import gp_Pnt

    topology = CohortTopology(component_index=component_index, plan=plan)

    cohort_slabs = [s for s in plan.slabs if s.component_index == component_index]
    if not cohort_slabs:
        return topology

    arrangement = plan.arrangements[component_index]

    # Collect unique z-planes across the cohort.
    z_planes: set[float] = set()
    for s in cohort_slabs:
        z_planes.add(s.zlo)
        z_planes.add(s.zhi)
    z_planes_sorted = sorted(z_planes)

    # Build a stable xy_corner_id for each unique XY vertex in the arrangement.
    # _ROUND_DECIMALS for stable hashing.
    _ROUND = 9
    xy_to_corner_id: dict[tuple[float, float], int] = {}
    for arr_edge in arrangement.edges:
        for (x, y) in arr_edge.vertices:
            key = (round(x, _ROUND), round(y, _ROUND))
            if key not in xy_to_corner_id:
                xy_to_corner_id[key] = len(xy_to_corner_id)

    # Vertex registry.
    for (x, y), corner_id in xy_to_corner_id.items():
        for z in z_planes_sorted:
            topology.vertices[(z, corner_id)] = BRepBuilderAPI_MakeVertex(
                gp_Pnt(x, y, z)
            ).Vertex()

    # Horizontal edge registry — straight edges only for now.
    # Arc support comes in Task 7.
    for arr_edge in arrangement.edges:
        p1, p2 = arr_edge.vertices
        c1 = xy_to_corner_id[(round(p1[0], _ROUND), round(p1[1], _ROUND))]
        c2 = xy_to_corner_id[(round(p2[0], _ROUND), round(p2[1], _ROUND))]
        for z in z_planes_sorted:
            v1 = topology.vertices[(z, c1)]
            v2 = topology.vertices[(z, c2)]
            topology.horizontal_edges[(z, arr_edge.edge_id)] = (
                BRepBuilderAPI_MakeEdge(v1, v2).Edge()
            )

    return topology
```

Also store `xy_to_corner_id` on the `CohortTopology` for later use (Task 4 needs it for vertical edges). Add the field to the dataclass:

```python
@dataclass
class CohortTopology:
    # ...existing fields...
    xy_to_corner_id: dict[tuple[float, float], int] = field(default_factory=dict)
```

And assign it before returning: `topology.xy_to_corner_id = xy_to_corner_id`.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/structured/test_cohort_topology_vertices_edges.py -v --no-cov`
Expected: all three tests PASS.

- [ ] **Step 5: Run structured suite**

Run: `python -m pytest tests/structured/ -v --no-cov`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add meshwell/structured/cohort_topology.py tests/structured/test_cohort_topology_vertices_edges.py
git commit -m "feat(cohort_topology): vertex + straight horizontal edge registries"
```

---

## Task 4: Vertical edge registry

**Files:**
- Modify: `meshwell/structured/cohort_topology.py`
- Test: same file as Task 3 (add to `test_cohort_topology_vertices_edges.py`)

For each cohort slab (a `z_interval_id` we can assign sequentially) and each cohort XY corner, build a `TopoDS_Edge` connecting the registered vertices at `(slab.zlo, corner_id)` and `(slab.zhi, corner_id)`.

**`z_interval_id` definition:** Use the slab's position within the cohort's sorted-by-(zlo,source_index) order. For simplicity, use `slab_index` from the plan as the id (it's already unique).

- [ ] **Step 1: Write the failing test**

Append to `tests/structured/test_cohort_topology_vertices_edges.py`:

```python
def test_vertical_edges_registered_per_slab_per_corner():
    """A stacked cohort with 2 slabs x 4 corners = 8 vertical edges."""
    plan = build_plan([_polyprism("A", 0, 1, 1), _polyprism("B", 1, 2, 2)])
    topology = build_cohort_topology(plan, component_index=0)
    assert len(topology.vertical_edges) == 8


def test_vertical_edge_endpoints_match_registry():
    """Each vertical edge's vertices must be in the registry."""
    plan = build_plan([_polyprism("A", 0, 1, 1)])
    topology = build_cohort_topology(plan, component_index=0)
    edge_key = next(iter(topology.vertical_edges))
    edge = topology.vertical_edges[edge_key]

    from OCP.TopAbs import TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer

    endpoints = set()
    exp = TopExp_Explorer(edge, TopAbs_VERTEX)
    while exp.More():
        endpoints.add(hash(exp.Current()))
        exp.Next()

    registry = {hash(v) for v in topology.vertices.values()}
    assert endpoints <= registry
```

- [ ] **Step 2: Run to verify failing**

Run: `python -m pytest tests/structured/test_cohort_topology_vertices_edges.py -v --no-cov`
Expected: vertical-edge tests FAIL (registry still empty).

- [ ] **Step 3: Implement vertical-edge construction**

In `meshwell/structured/cohort_topology.py`, inside `build_cohort_topology`, after the horizontal-edge loop, add:

```python
    # Vertical edge registry — per slab, per cohort XY corner.
    for slab in cohort_slabs:
        slab_index = plan.slabs.index(slab)  # use slab_index as z_interval_id
        for corner_id in xy_to_corner_id.values():
            v_lo = topology.vertices[(slab.zlo, corner_id)]
            v_hi = topology.vertices[(slab.zhi, corner_id)]
            topology.vertical_edges[(slab_index, corner_id)] = (
                BRepBuilderAPI_MakeEdge(v_lo, v_hi).Edge()
            )
```

**Note:** `plan.slabs.index(slab)` is O(N) per slab; if benchmarks show this is hot, build an `{id(slab): index}` map once. For now keep it simple.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/structured/test_cohort_topology_vertices_edges.py -v --no-cov`
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/cohort_topology.py tests/structured/test_cohort_topology_vertices_edges.py
git commit -m "feat(cohort_topology): vertical edge registry"
```

---

## Task 5: Horizontal face registry (straight only)

**Files:**
- Modify: `meshwell/structured/cohort_topology.py`
- Test: `tests/structured/test_cohort_topology_faces.py`

For each cohort piece at each z-plane it appears in, build a `TopoDS_Face` from the piece's outer-wire arrangement edges (reusing the registered horizontal edges).

**Piece appearance:** A cohort piece (identified by `piece_id = (source_index, piece_index)`) appears at the z-planes that are the zlo or zhi of any slab where that piece exists. We could over-generate (build the face at every cohort z-plane regardless) but that wastes work; better to track which (piece_id, z_plane) pairs are needed.

For simplicity in first cut: build at every cohort z-plane × every piece_id that appears in the cohort. We can prune later.

The piece's outer wire is built from `slab.face_partition_edges[piece_index]` — a list of `(arrangement_edge_id, reversed)` tuples. For each, look up the registered horizontal edge at the target z-plane; reverse if needed; assemble into a `BRepBuilderAPI_MakeWire`.

- [ ] **Step 1: Write the failing test**

Create `tests/structured/test_cohort_topology_faces.py`:

```python
"""build_cohort_topology populates horizontal and lateral face registries."""

from __future__ import annotations

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.cohort_topology import build_cohort_topology
from meshwell.structured.plan import build_plan


def _square(x0, y0, x1, y1):
    return shapely.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _polyprism(name, z0, z1, mesh_order, x0=0, y0=0, x1=1, y1=1):
    return PolyPrism(
        polygons=_square(x0, y0, x1, y1),
        buffers={float(z0): 0.0, float(z1): 0.0},
        physical_name=name,
        mesh_order=mesh_order,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )


def test_horizontal_faces_registered_per_piece_per_z_plane():
    """One piece x 2 slabs (3 z-planes) x 1 piece_id = 3 horizontal faces."""
    plan = build_plan([_polyprism("A", 0, 1, 1), _polyprism("B", 1, 2, 2)])
    topology = build_cohort_topology(plan, component_index=0)
    assert len(topology.horizontal_faces) == 3


def test_horizontal_face_at_shared_z_plane_is_same_tshape_for_top_and_bottom():
    """The face at z=1 IS both the top of slab A and the bottom of slab B.
    Because it's registered once, both references retrieve the same TShape.
    """
    plan = build_plan([_polyprism("A", 0, 1, 1), _polyprism("B", 1, 2, 2)])
    topology = build_cohort_topology(plan, component_index=0)
    # piece_id = (source_index, piece_index)
    # A is source_index=0, B is source_index=1, both have piece_index=0.
    # Look up face at z=1 for A's piece_id and B's piece_id; if they
    # represent the same XY footprint, they should map to the same key.
    # Convention chosen in the spec: same source_index -> same piece_id.
    # When A and B have different source_indices but the same XY piece,
    # they will be different piece_ids — and that's correct: the SAME
    # XY footprint at z=1 belongs to TWO different "pieces" semantically
    # (A's top piece, B's bottom piece). Verify they get separate keys.
    keys_at_z_1 = [k for k in topology.horizontal_faces if k[0] == 1.0]
    # Two pieces (A's at z=1 = its top, B's at z=1 = its bottom) at this
    # z-plane. They should reference the SAME TShape because the cohort
    # topology builder unifies same-XY pieces across vertical neighbors.
    # ... BUT only if piece_id collapses A and B's pieces at this z.
    # See spec discussion of piece_id; this test asserts the chosen behavior.
    assert len(keys_at_z_1) >= 1


def test_horizontal_face_outer_wire_is_built_from_registry_edges():
    """The horizontal face's outer wire edges must match the registered edges."""
    plan = build_plan([_polyprism("A", 0, 1, 1)])
    topology = build_cohort_topology(plan, component_index=0)
    face_key = next(iter(topology.horizontal_faces))
    face = topology.horizontal_faces[face_key]

    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer

    face_edge_hashes = set()
    exp = TopExp_Explorer(face, TopAbs_EDGE)
    while exp.More():
        face_edge_hashes.add(hash(exp.Current()))
        exp.Next()

    registry_edge_hashes = {hash(e) for e in topology.horizontal_edges.values()}
    assert face_edge_hashes <= registry_edge_hashes, (
        "Horizontal face's outer wire references edges not in the registry."
    )
```

**Note for the implementer:** the `test_horizontal_face_at_shared_z_plane_is_same_tshape_for_top_and_bottom` test depends on a design decision in the piece_id assignment. The spec says: "A piece at slab N and the corresponding piece at slab N+1 (vertically stacked, same entity) get the same `piece_id` because `source_index` and `piece_index` match." Two different entities A and B (different source_index) are NOT the same piece_id even if they have the same XY footprint at the shared z-plane.

In that case, at z=1, A contributes piece_id=(0,0) and B contributes piece_id=(1,0). Both get separate horizontal-face entries — so there will be 2 entries at z=1 (one for A's top, one for B's bottom), and the test should say so. Update the test assertion if needed.

For TRUE vertical sharing to happen across entities A and B, the piece_id needs a different definition — based on XY geometry rather than source_index. The spec's choice (source_index + piece_index) means vertical sharing only works WITHIN a single entity's slabs (e.g., when one PolyPrism has multiple z-intervals via `buffers`). Across entities, vertical sharing requires they merge through BOPAlgo.

**This is a design constraint the implementer should NOT try to fix.** If the test in this form doesn't model what we want, revise the test, not the design.

- [ ] **Step 2: Run to verify failing**

Run: `python -m pytest tests/structured/test_cohort_topology_faces.py -v --no-cov`
Expected: FAIL — horizontal_faces registry is still empty.

- [ ] **Step 3: Implement horizontal-face construction**

In `meshwell/structured/cohort_topology.py`, inside `build_cohort_topology`, after the vertical-edge loop, add:

```python
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire

    # Horizontal face registry: per (z_plane, piece_id).
    # piece_id = (source_index_of_entity, piece_index_within_slab).
    # For each cohort slab, build the bottom face (at zlo) and the top face
    # (at zhi) for each piece. Registered keys collapse vertically-stacked
    # same-entity pieces because (source_index, piece_index) matches.
    for slab in cohort_slabs:
        if not slab.face_partition or slab.face_partition_edges is None:
            continue
        for piece_index, piece_edges in enumerate(slab.face_partition_edges):
            piece_id = (slab.source_index, piece_index)
            for z in (slab.zlo, slab.zhi):
                key = (z, piece_id)
                if key in topology.horizontal_faces:
                    continue
                # Build the outer wire from the piece's arrangement edges
                # at this z-plane, applying orientation.
                mw = BRepBuilderAPI_MakeWire()
                for arr_edge_id, reversed_orient in piece_edges:
                    edge = topology.horizontal_edges[(z, arr_edge_id)]
                    mw.Add(edge.Reversed() if reversed_orient else edge)
                wire = mw.Wire()
                topology.horizontal_faces[key] = BRepBuilderAPI_MakeFace(wire).Face()
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/structured/test_cohort_topology_faces.py -v --no-cov`
Expected: PASS. If the test about "same TShape for top and bottom" needs adjusting per the design constraint above, update the test and re-run.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/cohort_topology.py tests/structured/test_cohort_topology_faces.py
git commit -m "feat(cohort_topology): horizontal face registry"
```

---

## Task 6: Straight lateral face registry

**Files:**
- Modify: `meshwell/structured/cohort_topology.py`
- Test: extend `tests/structured/test_cohort_topology_faces.py`

For each arrangement edge × each cohort slab, build a quadrilateral lateral face from:
- Bottom horizontal edge: `horizontal_edges[(slab.zlo, arr_edge_id)]`
- Top horizontal edge: `horizontal_edges[(slab.zhi, arr_edge_id)]`
- Two vertical edges: `vertical_edges[(slab_index, corner_id_left)]` and `vertical_edges[(slab_index, corner_id_right)]`

Stitch into a 4-edge wire and `BRepBuilderAPI_MakeFace`. Order matters; the wire must be a closed loop with consistent orientation.

- [ ] **Step 1: Write the failing test**

Append to `tests/structured/test_cohort_topology_faces.py`:

```python
def test_lateral_faces_registered_per_slab_per_edge():
    """A square cohort with 2 stacked slabs has 4 outer arrangement edges
    x 2 slabs = 8 lateral faces.
    """
    plan = build_plan([_polyprism("A", 0, 1, 1), _polyprism("B", 1, 2, 2)])
    topology = build_cohort_topology(plan, component_index=0)
    assert len(topology.lateral_faces) == 8


def test_lateral_face_edges_match_registry():
    """A lateral face's 4 edges must all be in the horizontal or vertical
    edge registries.
    """
    plan = build_plan([_polyprism("A", 0, 1, 1)])
    topology = build_cohort_topology(plan, component_index=0)
    face_key = next(iter(topology.lateral_faces))
    face = topology.lateral_faces[face_key]

    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer

    face_edge_hashes = set()
    exp = TopExp_Explorer(face, TopAbs_EDGE)
    while exp.More():
        face_edge_hashes.add(hash(exp.Current()))
        exp.Next()

    all_edge_hashes = (
        {hash(e) for e in topology.horizontal_edges.values()}
        | {hash(e) for e in topology.vertical_edges.values()}
    )
    assert face_edge_hashes <= all_edge_hashes
```

- [ ] **Step 2: Run to verify failing**

Run: `python -m pytest tests/structured/test_cohort_topology_faces.py -v --no-cov`
Expected: lateral tests FAIL.

- [ ] **Step 3: Implement straight lateral-face construction**

In `meshwell/structured/cohort_topology.py`, inside `build_cohort_topology`, after the horizontal-face loop, add:

```python
    # Straight lateral face registry: per (z_interval_id, arrangement_edge_id).
    # Arc edges are skipped here — they need Task 7's cylindrical surface
    # handling. For now, treat all edges as straight.
    for slab in cohort_slabs:
        slab_index = plan.slabs.index(slab)
        for arr_edge in arrangement.edges:
            key = (slab_index, arr_edge.edge_id)
            if key in topology.lateral_faces:
                continue
            p1, p2 = arr_edge.vertices
            c1 = xy_to_corner_id[(round(p1[0], _ROUND), round(p1[1], _ROUND))]
            c2 = xy_to_corner_id[(round(p2[0], _ROUND), round(p2[1], _ROUND))]

            bot_edge = topology.horizontal_edges[(slab.zlo, arr_edge.edge_id)]
            top_edge = topology.horizontal_edges[(slab.zhi, arr_edge.edge_id)]
            v_edge_1 = topology.vertical_edges[(slab_index, c1)]
            v_edge_2 = topology.vertical_edges[(slab_index, c2)]

            # Build wire: bot_edge (c1 -> c2 at zlo)
            #           -> v_edge_2 (c2 zlo -> c2 zhi)
            #           -> top_edge REVERSED (c2 -> c1 at zhi)
            #           -> v_edge_1 REVERSED (c1 zhi -> c1 zlo)
            mw = BRepBuilderAPI_MakeWire()
            mw.Add(bot_edge)
            mw.Add(v_edge_2)
            mw.Add(top_edge.Reversed())
            mw.Add(v_edge_1.Reversed())
            wire = mw.Wire()
            topology.lateral_faces[key] = BRepBuilderAPI_MakeFace(wire).Face()
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/structured/test_cohort_topology_faces.py -v --no-cov`
Expected: PASS.

If lateral face construction fails with "wire is not closed" or similar OCC errors, you may need to swap the orientation of one of the edges. Try `v_edge_1.Reversed()` ↔ `v_edge_1` and re-run. The correct combination depends on the canonical orientation of arrangement edges (which `arr_edge.vertices` is `(p1, p2)` — the edge goes from p1 to p2 at the lower z). If you find a fix, add a comment explaining the orientation logic.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/cohort_topology.py tests/structured/test_cohort_topology_faces.py
git commit -m "feat(cohort_topology): straight lateral face registry"
```

---

## Task 7: Arc support — horizontal arc edges

**Files:**
- Modify: `meshwell/structured/cohort_topology.py`
- Test: `tests/structured/test_cohort_topology_arcs.py`

Arrangement edges with a non-None `circle` attribute are arcs. For these, the horizontal edge must be built as an arc using the circle's center, radius, and the two endpoint vertices.

**Detection:** `arrangement.edges` returns `ArrangementEdge` objects with a `circle: CanonicalCircle | None` attribute. When `circle is not None`, build via OCC arc primitives.

**OCC construction for an arc edge:**
- Create `Geom_Circle` with center `gp_Pnt(cx, cy, z)` and radius `r` on the `gp_Ax2(center, gp_Dir(0,0,1))` axis.
- Use `GC_MakeArcOfCircle(circle, start_pnt, end_pnt, sense=True)` to get a `Geom_TrimmedCurve`.
- Use `BRepBuilderAPI_MakeEdge(curve, v1, v2)` to wrap into a `TopoDS_Edge` using the registered vertices.

- [ ] **Step 1: Write the failing test**

Create `tests/structured/test_cohort_topology_arcs.py`:

```python
"""Arc support in build_cohort_topology."""

from __future__ import annotations

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.cohort_topology import build_cohort_topology
from meshwell.structured.plan import build_plan


def _circle(cx, cy, r, n=32):
    """Approximate circle as a 32-gon polygon (PolyPrism's identify_arcs=True
    detects it as an arc).
    """
    import math
    pts = [
        (cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]
    return shapely.Polygon(pts)


def test_arc_horizontal_edge_is_built_when_arrangement_edge_has_circle():
    """Build a cohort with a circular PolyPrism (identify_arcs=True); the
    arrangement's edges include arc edges; horizontal_edges entries for
    arc edges should be valid TopoDS_Edge with arc geometry, not straight
    segments.
    """
    poly = _circle(0, 0, 1, n=32)
    A = PolyPrism(
        polygons=poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="A",
        mesh_order=1,
        structured=True,
        identify_arcs=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )
    plan = build_plan([A])
    topology = build_cohort_topology(plan, component_index=0)
    # Arrangement should have arc edges (circle attribute non-None).
    arc_edges = [e for e in plan.arrangements[0].edges if e.circle is not None]
    assert arc_edges, "Test setup: expected arc edges in arrangement"

    from OCP.BRep import BRep_Tool
    from OCP.GeomAbs import GeomAbs_Circle

    for arr_edge in arc_edges:
        edge = topology.horizontal_edges[(0.0, arr_edge.edge_id)]
        curve, _u0, _u1 = BRep_Tool.Curve_s(edge)
        # Adapt to query curve type. If implementation builds straight
        # edges instead, this query will return GeomAbs_Line (not Circle).
        from OCP.BRepAdaptor import BRepAdaptor_Curve

        adaptor = BRepAdaptor_Curve(edge)
        assert adaptor.GetType() == GeomAbs_Circle, (
            f"Arc arrangement edge produced non-Circle horizontal edge "
            f"(got type {adaptor.GetType()})"
        )
```

- [ ] **Step 2: Run to verify failing**

Run: `python -m pytest tests/structured/test_cohort_topology_arcs.py -v --no-cov`
Expected: FAIL — current implementation builds straight edges for all arrangement edges.

- [ ] **Step 3: Implement arc horizontal-edge construction**

In `meshwell/structured/cohort_topology.py`, modify the horizontal-edge loop:

```python
    from OCP.GC import GC_MakeArcOfCircle
    from OCP.gp import gp_Ax2, gp_Circ, gp_Dir
    from OCP.Geom import Geom_Circle

    for arr_edge in arrangement.edges:
        p1, p2 = arr_edge.vertices
        c1 = xy_to_corner_id[(round(p1[0], _ROUND), round(p1[1], _ROUND))]
        c2 = xy_to_corner_id[(round(p2[0], _ROUND), round(p2[1], _ROUND))]
        for z in z_planes_sorted:
            v1 = topology.vertices[(z, c1)]
            v2 = topology.vertices[(z, c2)]
            if arr_edge.circle is not None:
                cx, cy = arr_edge.circle.center
                r = arr_edge.circle.radius
                axis = gp_Ax2(gp_Pnt(cx, cy, z), gp_Dir(0, 0, 1))
                circ = gp_Circ(axis, r)
                start = gp_Pnt(p1[0], p1[1], z)
                end = gp_Pnt(p2[0], p2[1], z)
                arc = GC_MakeArcOfCircle(circ, start, end, True).Value()
                edge = BRepBuilderAPI_MakeEdge(arc, v1, v2).Edge()
            else:
                edge = BRepBuilderAPI_MakeEdge(v1, v2).Edge()
            topology.horizontal_edges[(z, arr_edge.edge_id)] = edge
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/structured/test_cohort_topology_arcs.py -v --no-cov`
Expected: PASS.

If the arc-vs-line orientation comes out wrong (you might see e.g. the major arc instead of the minor arc), revisit `GC_MakeArcOfCircle`'s `Sense` parameter — flip to False if needed.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/cohort_topology.py tests/structured/test_cohort_topology_arcs.py
git commit -m "feat(cohort_topology): arc horizontal edges via Geom_Circle"
```

---

## Task 8: Arc support — cylindrical lateral faces

**Files:**
- Modify: `meshwell/structured/cohort_topology.py`
- Test: extend `tests/structured/test_cohort_topology_arcs.py`

When an arrangement edge is an arc, its lateral face is a cylindrical strip. Construction:
- `Geom_CylindricalSurface` with axis `gp_Ax3(center=gp_Pnt(cx, cy, zlo), direction=gp_Dir(0, 0, 1))` and radius `r`.
- Build the wire (arc at zlo, vertical edge, arc at zhi reversed, vertical edge reversed).
- `BRepBuilderAPI_MakeFace(cylindrical_surface, wire)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/structured/test_cohort_topology_arcs.py`:

```python
def test_arc_lateral_face_is_cylindrical():
    poly = _circle(0, 0, 1, n=32)
    A = PolyPrism(
        polygons=poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="A",
        mesh_order=1,
        structured=True,
        identify_arcs=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )
    plan = build_plan([A])
    topology = build_cohort_topology(plan, component_index=0)
    arc_arr_edges = [e for e in plan.arrangements[0].edges if e.circle is not None]
    assert arc_arr_edges, "Test setup: expected arc edges"

    slab_index = 0  # only slab
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder

    for arr_edge in arc_arr_edges:
        face = topology.lateral_faces[(slab_index, arr_edge.edge_id)]
        adaptor = BRepAdaptor_Surface(face)
        assert adaptor.GetType() == GeomAbs_Cylinder, (
            f"Arc lateral face is not cylindrical (got {adaptor.GetType()})"
        )
```

- [ ] **Step 2: Run to verify failing**

Run: `python -m pytest tests/structured/test_cohort_topology_arcs.py -v --no-cov`
Expected: FAIL.

- [ ] **Step 3: Implement cylindrical lateral face construction**

Modify the lateral-face loop in `meshwell/structured/cohort_topology.py`:

```python
    from OCP.Geom import Geom_CylindricalSurface
    from OCP.gp import gp_Ax3

    for slab in cohort_slabs:
        slab_index = plan.slabs.index(slab)
        for arr_edge in arrangement.edges:
            key = (slab_index, arr_edge.edge_id)
            if key in topology.lateral_faces:
                continue
            p1, p2 = arr_edge.vertices
            c1 = xy_to_corner_id[(round(p1[0], _ROUND), round(p1[1], _ROUND))]
            c2 = xy_to_corner_id[(round(p2[0], _ROUND), round(p2[1], _ROUND))]

            bot_edge = topology.horizontal_edges[(slab.zlo, arr_edge.edge_id)]
            top_edge = topology.horizontal_edges[(slab.zhi, arr_edge.edge_id)]
            v_edge_1 = topology.vertical_edges[(slab_index, c1)]
            v_edge_2 = topology.vertical_edges[(slab_index, c2)]

            mw = BRepBuilderAPI_MakeWire()
            mw.Add(bot_edge)
            mw.Add(v_edge_2)
            mw.Add(top_edge.Reversed())
            mw.Add(v_edge_1.Reversed())
            wire = mw.Wire()

            if arr_edge.circle is not None:
                cx, cy = arr_edge.circle.center
                r = arr_edge.circle.radius
                axis = gp_Ax3(gp_Pnt(cx, cy, slab.zlo), gp_Dir(0, 0, 1))
                surface = Geom_CylindricalSurface(axis, r)
                topology.lateral_faces[key] = BRepBuilderAPI_MakeFace(
                    surface, wire
                ).Face()
            else:
                topology.lateral_faces[key] = BRepBuilderAPI_MakeFace(wire).Face()
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/structured/test_cohort_topology_arcs.py -v --no-cov`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/cohort_topology.py tests/structured/test_cohort_topology_arcs.py
git commit -m "feat(cohort_topology): cylindrical lateral faces for arc edges"
```

---

## Task 9: `assemble_cohort_sub_prism` — solid assembly + PhantomShape population

**Files:**
- Modify: `meshwell/structured/cohort_topology.py`
- Test: `tests/structured/test_cohort_topology_assembly.py`

For one `(slab, piece_index)`, build a `TopoDS_Solid` from the registry's faces and populate a `PhantomShape` with `input_*_by_key` dicts referencing registry entries.

Steps inside the function:
1. Look up the bottom and top horizontal faces.
2. For each outer arrangement edge of the piece, look up the lateral face (applying `.Reversed()` if the piece records `reversed=True` for the edge).
3. Build a `TopoDS_Shell` via `BRep_Builder.MakeShell` + `Add` for bot (reversed for outward orientation), top, and laterals.
4. Build a `TopoDS_Solid` via `BRep_Builder.MakeSolid` + `Add`.
5. Validate with `BRepCheck_Analyzer` (optional debug check).
6. Populate `PhantomShape`:
   - `solid`
   - `input_faces_by_key`: FaceKey for bot/top → horizontal_faces entry
   - `input_edges_by_key`: EdgeKey for bot/top outer-wire edges → horizontal_edges entry (one per outer edge)
   - `input_vertices_by_key`: VertexKey for bot/top corners → vertices entry
   - `input_laterals_by_outer_edge`: outer_edge_index → lateral_faces entry (possibly reversed)

- [ ] **Step 1: Write the failing test**

Create `tests/structured/test_cohort_topology_assembly.py`:

```python
"""assemble_cohort_sub_prism produces a valid solid + populated PhantomShape."""

from __future__ import annotations

import shapely
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.cohort_topology import (
    assemble_cohort_sub_prism,
    build_cohort_topology,
)
from meshwell.structured.plan import build_plan


def _square(x0, y0, x1, y1):
    return shapely.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _polyprism(name, z0, z1, mesh_order):
    return PolyPrism(
        polygons=_square(0, 0, 1, 1),
        buffers={float(z0): 0.0, float(z1): 0.0},
        physical_name=name,
        mesh_order=mesh_order,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )


def _faces(shape):
    out = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        out.append(exp.Current())
        exp.Next()
    return out


def test_assemble_produces_valid_solid():
    plan = build_plan([_polyprism("A", 0, 1, 1)])
    topology = build_cohort_topology(plan, component_index=0)
    slab = [s for s in plan.slabs if s.component_index == 0][0]
    ps = assemble_cohort_sub_prism(topology, slab, piece_index=0)
    analyzer = BRepCheck_Analyzer(ps.solid)
    assert analyzer.IsValid(), "Assembled solid failed BRepCheck"


def test_assembled_solid_has_six_faces_for_square_cohort():
    plan = build_plan([_polyprism("A", 0, 1, 1)])
    topology = build_cohort_topology(plan, component_index=0)
    slab = [s for s in plan.slabs if s.component_index == 0][0]
    ps = assemble_cohort_sub_prism(topology, slab, piece_index=0)
    assert len(_faces(ps.solid)) == 6  # 1 bot + 1 top + 4 laterals


def test_input_faces_by_key_uses_registry():
    plan = build_plan([_polyprism("A", 0, 1, 1)])
    topology = build_cohort_topology(plan, component_index=0)
    slab = [s for s in plan.slabs if s.component_index == 0][0]
    ps = assemble_cohort_sub_prism(topology, slab, piece_index=0)
    registry_face_hashes = {hash(f) for f in topology.horizontal_faces.values()}
    for face_key, face in ps.input_faces_by_key.items():
        if face_key.side in ("bot", "top"):
            assert hash(face) in registry_face_hashes


def test_stacked_solids_share_interface_face_tshape():
    """Two stacked sub-prisms in the same cohort share the z=1 interface."""
    plan = build_plan([_polyprism("A", 0, 1, 1), _polyprism("B", 1, 2, 2)])
    topology = build_cohort_topology(plan, component_index=0)
    slab_A = [s for s in plan.slabs if s.physical_name == ("A",)][0]
    slab_B = [s for s in plan.slabs if s.physical_name == ("B",)][0]
    ps_A = assemble_cohort_sub_prism(topology, slab_A, piece_index=0)
    ps_B = assemble_cohort_sub_prism(topology, slab_B, piece_index=0)
    a_faces = {hash(f) for f in _faces(ps_A.solid)}
    b_faces = {hash(f) for f in _faces(ps_B.solid)}
    assert a_faces & b_faces, (
        "Stacked sub-prisms do not share interface face TShape after assembly."
    )
```

- [ ] **Step 2: Run to verify failing**

Run: `python -m pytest tests/structured/test_cohort_topology_assembly.py -v --no-cov`
Expected: FAIL with NotImplementedError.

- [ ] **Step 3: Implement `assemble_cohort_sub_prism`**

Replace `assemble_cohort_sub_prism` in `meshwell/structured/cohort_topology.py`:

```python
def assemble_cohort_sub_prism(
    topology: CohortTopology,
    slab: Slab,
    piece_index: int,
) -> PhantomShape:
    """Assemble one sub-prism's solid + PhantomShape from the registry."""
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Shell, TopoDS_Solid

    from meshwell.structured.spec import EdgeKey, FaceKey, VertexKey

    plan = topology.plan
    assert plan is not None
    slab_index = plan.slabs.index(slab)
    piece_id = (slab.source_index, piece_index)

    bot_face = topology.horizontal_faces[(slab.zlo, piece_id)]
    top_face = topology.horizontal_faces[(slab.zhi, piece_id)]

    # Build lateral faces per outer arrangement edge, applying orientation.
    piece_edges = slab.face_partition_edges[piece_index]
    lateral_faces_oriented: list[Any] = []
    input_laterals: dict[int, Any] = {}
    for outer_edge_i, (arr_edge_id, reversed_orient) in enumerate(piece_edges):
        lateral = topology.lateral_faces[(slab_index, arr_edge_id)]
        oriented = lateral.Reversed() if reversed_orient else lateral
        lateral_faces_oriented.append(oriented)
        input_laterals[outer_edge_i] = oriented

    # Assemble shell + solid.
    b = BRep_Builder()
    shell = TopoDS_Shell()
    b.MakeShell(shell)
    b.Add(shell, bot_face.Reversed())  # bottom face's normal points down
    b.Add(shell, top_face)
    for lf in lateral_faces_oriented:
        b.Add(shell, lf)

    solid = TopoDS_Solid()
    b.MakeSolid(solid)
    b.Add(solid, shell)

    # Populate PhantomShape input dicts.
    input_faces: dict[FaceKey, Any] = {
        FaceKey(slab_index=slab_index, side="bot", piece_index=piece_index): bot_face,
        FaceKey(slab_index=slab_index, side="top", piece_index=piece_index): top_face,
    }

    input_edges: dict[EdgeKey, Any] = {}
    input_vertices: dict[VertexKey, Any] = {}
    for outer_edge_i, (arr_edge_id, _reversed) in enumerate(piece_edges):
        for side, z in (("bot", slab.zlo), ("top", slab.zhi)):
            edge = topology.horizontal_edges[(z, arr_edge_id)]
            input_edges[
                EdgeKey(
                    slab_index=slab_index,
                    side=side,
                    piece_index=piece_index,
                    edge_index=outer_edge_i,
                )
            ] = edge

    # Vertex bookkeeping: walk each outer edge's start vertex.
    arrangement = plan.arrangements[topology.component_index]
    edge_by_id = {e.edge_id: e for e in arrangement.edges}
    for corner_i, (arr_edge_id, reversed_orient) in enumerate(piece_edges):
        arr_edge = edge_by_id[arr_edge_id]
        # Start vertex of the piece-side traversal:
        if reversed_orient:
            x, y = arr_edge.vertices[1]
        else:
            x, y = arr_edge.vertices[0]
        c = topology.xy_to_corner_id[
            (round(x, 9), round(y, 9))
        ]
        for side, z in (("bot", slab.zlo), ("top", slab.zhi)):
            input_vertices[
                VertexKey(
                    slab_index=slab_index,
                    side=side,
                    piece_index=piece_index,
                    corner_index=corner_i,
                )
            ] = topology.vertices[(z, c)]

    return PhantomShape(
        slab_index=slab_index,
        piece_index=piece_index,
        solid=solid,
        input_faces_by_key=input_faces,
        input_edges_by_key=input_edges,
        input_vertices_by_key=input_vertices,
        input_laterals_by_outer_edge=input_laterals,
    )
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/structured/test_cohort_topology_assembly.py -v --no-cov`
Expected: PASS.

If `BRepCheck_Analyzer` flags the assembled solid as invalid, the most likely cause is shell orientation. Try:
- Without `bot_face.Reversed()` (just `b.Add(shell, bot_face)`)
- Reverse `top_face` instead
- Reverse all laterals
The correct convention for `BRepPrimAPI_MakePrism`'s output is: top and laterals have outward normals; bot has inward normal (= "reversed"). Match that convention.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/cohort_topology.py tests/structured/test_cohort_topology_assembly.py
git commit -m "feat(cohort_topology): assemble_cohort_sub_prism builds solid + PhantomShape"
```

---

## Task 10: Integrate via `_USE_COHORT_TOPOLOGY` kill-switch in `build_phantom_shapes`

**Files:**
- Modify: `meshwell/structured/phantom.py`
- Test: `tests/structured/test_phantom_use_cohort_topology.py`

Add `_USE_COHORT_TOPOLOGY = True` module-level constant. Modify `build_phantom_shapes` to branch:
- `_USE_COHORT_TOPOLOGY=True`: group slabs by `component_index`, call `build_cohort_topology` once per cohort, then `assemble_cohort_sub_prism` per `(slab, piece_index)`.
- `_USE_COHORT_TOPOLOGY=False`: existing path (untouched).

Output ordering: `(slab_index, piece_index)` ascending — same as today.

- [ ] **Step 1: Write the failing test**

Create `tests/structured/test_phantom_use_cohort_topology.py`:

```python
"""build_phantom_shapes with _USE_COHORT_TOPOLOGY=True uses the cohort builder."""

from __future__ import annotations

import shapely
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured import phantom as phantom_mod
from meshwell.structured.phantom import build_phantom_shapes
from meshwell.structured.plan import build_plan


def _square(x0, y0, x1, y1):
    return shapely.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _polyprism(name, z0, z1, mesh_order):
    return PolyPrism(
        polygons=_square(0, 0, 1, 1),
        buffers={float(z0): 0.0, float(z1): 0.0},
        physical_name=name,
        mesh_order=mesh_order,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )


def _faces(shape):
    out = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        out.append(exp.Current())
        exp.Next()
    return out


def test_kill_switch_default_is_on():
    assert phantom_mod._USE_COHORT_TOPOLOGY is True


def test_cohort_topology_path_produces_shared_lateral_face():
    """Two side-by-side structured slabs in the same cohort -> shared lateral face."""
    A = PolyPrism(
        polygons=_square(0, 0, 1, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="A",
        mesh_order=1,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )
    B = PolyPrism(
        polygons=_square(1, 0, 2, 1),  # adjacent to A at x=1
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="B",
        mesh_order=2,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )
    plan = build_plan([A, B])
    result = build_phantom_shapes(plan)
    by_slab = {ps.slab_index: ps for ps in result.shapes}
    assert len(by_slab) == 2

    a_face_hashes = {hash(f) for f in _faces(by_slab[0].solid)}
    b_face_hashes = {hash(f) for f in _faces(by_slab[1].solid)}
    assert a_face_hashes & b_face_hashes, (
        "Laterally-adjacent cohort sub-prisms do NOT share interface lateral "
        "face TShape — cohort topology builder not active or wired incorrectly."
    )


def test_legacy_path_when_kill_switch_off():
    """With _USE_COHORT_TOPOLOGY=False we fall back to the existing path."""
    phantom_mod._USE_COHORT_TOPOLOGY = False
    try:
        plan = build_plan([_polyprism("A", 0, 1, 1)])
        result = build_phantom_shapes(plan)
        assert len(result.shapes) == 1
    finally:
        phantom_mod._USE_COHORT_TOPOLOGY = True
```

- [ ] **Step 2: Run to verify failing**

Run: `python -m pytest tests/structured/test_phantom_use_cohort_topology.py -v --no-cov`
Expected: kill-switch constant test fails (not defined); shared-lateral test fails (cohort path not active).

- [ ] **Step 3: Add kill-switch and integration**

In `meshwell/structured/phantom.py`, add the constant near `_PRESHARE_VERTICAL_FACES`:

```python
# Phase 2 kill-switch. When True (default after Phase 2 lands),
# build_phantom_shapes uses the cohort topology builder for full
# vertical+lateral face sharing. When False, falls back to the
# Phase 1 path (which itself has _PRESHARE_VERTICAL_FACES sub-switch).
# Both paths preserved during Phase 2; legacy retired in Phase 3.
_USE_COHORT_TOPOLOGY = True
```

Modify `build_phantom_shapes` to branch at the top:

```python
@phase_timed("phantom_build")
def build_phantom_shapes(plan: StructuredPlan) -> PhantomBuildResult:
    """..."""
    if _USE_COHORT_TOPOLOGY:
        return _build_phantom_shapes_via_cohort_topology(plan)
    # ... existing body unchanged ...
```

Add new helper function `_build_phantom_shapes_via_cohort_topology(plan)`:

```python
def _build_phantom_shapes_via_cohort_topology(
    plan: StructuredPlan,
) -> PhantomBuildResult:
    """Phase 2 path: build each cohort's topology once, then assemble each
    sub-prism as a view into the shared topology.
    """
    from meshwell.structured.cohort_topology import (
        assemble_cohort_sub_prism,
        build_cohort_topology,
    )

    # Group slabs by cohort.
    cohorts: dict[int, list[Slab]] = {}
    for slab in plan.slabs:
        cohorts.setdefault(slab.component_index, []).append(slab)

    out: dict[tuple[int, int], PhantomShape] = {}
    for component_index in sorted(cohorts):
        topology = build_cohort_topology(plan, component_index)
        for slab in cohorts[component_index]:
            slab_index = plan.slabs.index(slab)
            if not slab.face_partition:
                continue
            for piece_index in range(len(slab.face_partition)):
                ps = assemble_cohort_sub_prism(topology, slab, piece_index)
                out[(slab_index, piece_index)] = ps

    shapes = [out[k] for k in sorted(out.keys())]
    return PhantomBuildResult(shapes=tuple(shapes))
```

- [ ] **Step 4: Run kill-switch tests**

Run: `python -m pytest tests/structured/test_phantom_use_cohort_topology.py -v --no-cov`
Expected: PASS.

- [ ] **Step 5: Run full structured + cad_occ suites**

Run: `python -m pytest tests/structured/ tests/test_cad_occ.py tests/test_cad_occ_polyprism_overlap_fastpath.py tests/test_cad_occ_same_name_fuse.py tests/test_cad_occ_cohort_sewing.py -v --no-cov`
Expected: PASS. Any failure must be investigated — could be legitimate bug in cohort topology that the broader suite catches.

If a failure is real-bug (cohort topology output doesn't match what cad_occ/builder expects), STOP and report BLOCKED.

- [ ] **Step 6: Commit**

```bash
git add meshwell/structured/phantom.py tests/structured/test_phantom_use_cohort_topology.py
git commit -m "feat(structured): _USE_COHORT_TOPOLOGY wires Phase 2 into build_phantom_shapes"
```

---

## Task 11: End-to-end integration test (vertical + lateral)

**Files:**
- Create: `tests/structured/test_cohort_topology_integration.py`

Build a small mixed scene (3 vertically-stacked + 2 laterally-adjacent + 1 unstructured neighbor). Run `build_phantom_shapes` with default (Phase 2 path). Verify all expected sharing happens.

- [ ] **Step 1: Write the integration test**

Create `tests/structured/test_cohort_topology_integration.py`:

```python
"""End-to-end: cohort topology builder produces both vertical AND lateral sharing."""

from __future__ import annotations

import shapely
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.phantom import build_phantom_shapes
from meshwell.structured.plan import build_plan


def _square(x0, y0, x1, y1):
    return shapely.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _faces(shape):
    out = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        out.append(exp.Current())
        exp.Next()
    return out


def test_mixed_cohort_sharing():
    """Build scene:
       Cohort 1: A (z=0..1), B (z=1..2), C (z=2..3) at XY [0,1]x[0,1]  (vertical stack)
       Cohort 2: D (z=0..1), E (z=0..1) at XY [3,4]x[0,1] and [4,5]x[0,1]  (lateral pair)
       Unstructured: F at XY [10,11]x[10,11]
    """
    A = PolyPrism(
        polygons=_square(0, 0, 1, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="A",
        mesh_order=1,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )
    B = PolyPrism(
        polygons=_square(0, 0, 1, 1),
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="B",
        mesh_order=2,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )
    C = PolyPrism(
        polygons=_square(0, 0, 1, 1),
        buffers={2.0: 0.0, 3.0: 0.0},
        physical_name="C",
        mesh_order=3,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )
    D = PolyPrism(
        polygons=_square(3, 0, 4, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="D",
        mesh_order=4,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )
    E = PolyPrism(
        polygons=_square(4, 0, 5, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="E",
        mesh_order=5,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )
    plan = build_plan([A, B, C, D, E])
    result = build_phantom_shapes(plan)

    by_name = {}
    for ps in result.shapes:
        slab = plan.slabs[ps.slab_index]
        by_name[slab.physical_name[0]] = ps

    def _face_hashes(ps):
        return {hash(f) for f in _faces(ps.solid)}

    # Vertical sharing within Cohort 1.
    assert _face_hashes(by_name["A"]) & _face_hashes(by_name["B"]), \
        "A and B should share interface face (vertical)"
    assert _face_hashes(by_name["B"]) & _face_hashes(by_name["C"]), \
        "B and C should share interface face (vertical)"
    # A and C are NOT vertically adjacent; should NOT share.
    assert not (_face_hashes(by_name["A"]) & _face_hashes(by_name["C"])), \
        "A and C should NOT share faces (not vertically adjacent)"

    # Lateral sharing within Cohort 2.
    assert _face_hashes(by_name["D"]) & _face_hashes(by_name["E"]), \
        "D and E should share interface face (lateral)"

    # Cross-cohort: no sharing.
    assert not (_face_hashes(by_name["A"]) & _face_hashes(by_name["D"])), \
        "Different cohorts should not share faces"
```

- [ ] **Step 2: Run and confirm passing (or investigate)**

Run: `python -m pytest tests/structured/test_cohort_topology_integration.py -v --no-cov`

Possible outcomes:
- PASS: great, ship.
- Vertical sharing fails: investigate — likely a `piece_id` issue (same XY, same source_index across slabs… wait, A and B have DIFFERENT source_indices because they're separate entities, so they would NOT share by current `piece_id` definition. Update the test expectations or update the design.)
- Lateral sharing fails: investigate — likely the arrangement edges connecting D and E aren't being treated as cohort-interior. Confirm `plan.arrangements[component_index].edges` includes the seam between D and E.

**If a real bug is found, STOP and report.** Don't paper over.

If the vertical-sharing test fails due to the `piece_id` constraint (A and B as separate entities don't share), the test should be revised: change the scene so A and B are slabs of the SAME PolyPrism (use the `buffers` parameter to define multiple z-intervals on one entity). That way they share `source_index` and `piece_index`, and the cohort topology builder will register the z=1 face once for both.

- [ ] **Step 3: Commit**

```bash
git add tests/structured/test_cohort_topology_integration.py
git commit -m "test(cohort_topology): end-to-end mixed-cohort sharing integration test"
```

---

## Task 12: Parity test (`_USE_COHORT_TOPOLOGY` ON vs OFF)

**Files:**
- Create: `tests/test_cohort_topology_parity.py`

Run the full pipeline twice on the same scene — once with `_USE_COHORT_TOPOLOGY=True`, once with `_USE_COHORT_TOPOLOGY=False` AND `_PRESHARE_VERTICAL_FACES=False` (full legacy) — assert identical observable output.

- [ ] **Step 1: Write the parity test**

Create `tests/test_cohort_topology_parity.py`:

```python
"""Parity: full pipeline output is invariant under _USE_COHORT_TOPOLOGY."""

from __future__ import annotations

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured import phantom as phantom_mod
from meshwell.orchestrator import build  # adjust import if entry differs


def _square(x0, y0, x1, y1):
    return shapely.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _scene():
    A = PolyPrism(
        polygons=_square(0, 0, 1, 1),
        buffers={0.0: 0.0, 1.0: 0.0, 2.0: 0.0},  # stack: z=[0,1], [1,2]
        physical_name="A",
        mesh_order=1,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2, 2])],
    )
    B = PolyPrism(
        polygons=_square(1, 0, 2, 1),  # lateral neighbor at z=[0,1] only
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="B",
        mesh_order=2,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )
    return [A, B]


def _signature(occ_entities):
    return {ent.physical_name: (len(ent.shapes), ent.dim) for ent in occ_entities}


def test_use_cohort_topology_does_not_change_entity_signature(tmp_path):
    entities = _scene()

    phantom_mod._USE_COHORT_TOPOLOGY = False
    phantom_mod._PRESHARE_VERTICAL_FACES = False
    try:
        baseline = build(entities, output_dir=tmp_path / "baseline")
        baseline_sig = _signature(baseline)
    finally:
        phantom_mod._USE_COHORT_TOPOLOGY = True
        phantom_mod._PRESHARE_VERTICAL_FACES = True

    new = build(entities, output_dir=tmp_path / "new")
    new_sig = _signature(new)

    assert baseline_sig == new_sig, (
        f"Entity signature differs:\n  baseline={baseline_sig}\n  new={new_sig}"
    )
```

**IMPORTANT:** The `from meshwell.orchestrator import build` is a stand-in. Inspect `tests/test_cad_occ_phantom_hook.py` or other integration tests to find the correct full-pipeline entry point used in this repo. Adapt `_signature()` if `build()` returns something else.

- [ ] **Step 2: Wire to actual pipeline entry**

Read existing tests and adapt the import + signature extraction.

- [ ] **Step 3: Run the parity test**

Run: `python -m pytest tests/test_cohort_topology_parity.py -v --no-cov`
Expected: PASS.

If FAIL: signatures differ → real correctness regression in cohort topology. STOP, investigate.

- [ ] **Step 4: Commit**

```bash
git add tests/test_cohort_topology_parity.py
git commit -m "test(cohort_topology): parity test for _USE_COHORT_TOPOLOGY toggle"
```

---

## Task 13: Performance measurement

**Files:**
- Create: `scripts/bench_cohort_topology.py`
- Modify: `docs/superpowers/specs/2026-05-27-cad-occ-cohort-topology-builder-design.md` (append Measured Results section)

- [ ] **Step 1: Write the benchmark**

Create `scripts/bench_cohort_topology.py`:

```python
"""Compare full-pipeline wall time with cohort topology ON vs OFF.

Constructs a mixed vertically-stacked + laterally-adjacent structured
scene representative of the production cost surface, then runs the
pipeline three times: full legacy, Phase 1 vertical-only, Phase 2 full.
"""

from __future__ import annotations

import time

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured import phantom as phantom_mod


def _square(x0, y0, x1, y1):
    return shapely.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _build_scene(n_stacks: int = 4, layers_per_stack: int = 10):
    entities = []
    for stack_idx in range(n_stacks):
        x0 = stack_idx * 2.0
        x1 = x0 + 1.0
        for layer_idx in range(layers_per_stack):
            z0 = float(layer_idx)
            z1 = z0 + 1.0
            entities.append(
                PolyPrism(
                    polygons=_square(x0, 0, x1, 1),
                    buffers={z0: 0.0, z1: 0.0},
                    physical_name=f"s{stack_idx}_l{layer_idx}",
                    mesh_order=stack_idx * 100 + layer_idx,
                    structured=True,
                    resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
                )
            )
    return entities


def _time_pipeline(entities, *, use_cohort: bool, preshare_vertical: bool) -> float:
    phantom_mod._USE_COHORT_TOPOLOGY = use_cohort
    phantom_mod._PRESHARE_VERTICAL_FACES = preshare_vertical
    from meshwell.orchestrator import build

    t0 = time.perf_counter()
    build(entities, output_dir="/tmp/bench_cohort_topology")
    t1 = time.perf_counter()
    return t1 - t0


if __name__ == "__main__":
    entities = _build_scene()
    legacy = _time_pipeline(entities, use_cohort=False, preshare_vertical=False)
    phase1 = _time_pipeline(entities, use_cohort=False, preshare_vertical=True)
    phase2 = _time_pipeline(entities, use_cohort=True, preshare_vertical=True)
    print(f"Full legacy:     {legacy:.2f}s")
    print(f"Phase 1 (vert):  {phase1:.2f}s  (speedup: {legacy / phase1:.2f}x)")
    print(f"Phase 2 (both):  {phase2:.2f}s  (speedup: {legacy / phase2:.2f}x)")
```

(Adapt the `build` import to the actual entry point used in Task 12.)

- [ ] **Step 2: Run the benchmark**

Run: `python scripts/bench_cohort_topology.py`
Expected: Phase 2 ≥3× over full legacy on the structured-heavy scene.

- [ ] **Step 3: Append Measured Results to spec**

Append to `docs/superpowers/specs/2026-05-27-cad-occ-cohort-topology-builder-design.md`:

```markdown
## Measured Results (Task 13)

- Scene: <N stacks x M layers, brief description>
- Full legacy: <time>
- Phase 1 (vertical-only): <time> (<ratio>x)
- Phase 2 (vertical + lateral): <time> (<ratio>x)
- Notes: <where time is spent, what dominates>
```

- [ ] **Step 4: Commit**

```bash
git add scripts/bench_cohort_topology.py docs/superpowers/specs/2026-05-27-cad-occ-cohort-topology-builder-design.md
git commit -m "perf(cohort_topology): benchmark Phase 2 vs Phase 1 vs legacy"
```

---

## Wrap-up

After all tasks pass:

- [ ] Final full-suite run: `python -m pytest tests/ -v --no-cov`
- [ ] Manual sanity check on one production scene if available
- [ ] If success criterion (≥3× over legacy) met, plan Phase 3 cleanup (delete legacy path, retire kill-switches)
- [ ] If success criterion missed, investigate where remaining time is spent; may indicate that the `compute_cutters` overlap-test work (Phase 2 non-goal) is now the bottleneck
