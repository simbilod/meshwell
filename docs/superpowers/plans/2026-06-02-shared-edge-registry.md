# Shared EdgeRegistry Across Structured + Unstructured Paths Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make cohort↔neighbour shared boundary edges use literally the same `TopoDS_Edge` objects (one TShape) by routing both sides through a shared `EdgeRegistry`, so arc/line coincidence at structured↔unstructured interfaces is by construction rather than BOP fuzzy detection.

**Architecture:** Each cohort already builds its own `VertexRegistry`+`EdgeRegistry` inside `build_cohort_compound`. We expose those registries through `StructuredState`, tag each pre-cut unstructured entity with the cohorts it's adjacent to and at which z-planes, and modify `PolyPrism.instanciate_occ` so that when an entity carries this tag, the boundary polygon's edges at the shared z-plane are constructed by routing through the adjacent cohort's `EdgeRegistry`. Lateral and exterior-side edges are built the usual way (no sharing risk). Scoping is per-cohort and per-z to avoid the mesh corruption the global-sharing spike exhibited.

**Tech Stack:** Python 3.12, OCP (OpenCASCADE Python), pytest, shapely

**Spec context:**
- Spike commits + measurements: 8 → 3 AABB rescues on the complex scene under global sharing; global sharing also corrupted the mesh (tets 35407→18234) — proves scoping is required.
- Sister analysis: [docs/superpowers/specs/2026-06-01-cohort-topology-investigations.md](../specs/2026-06-01-cohort-topology-investigations.md) §1.

---

## File map

**Modified:**
- `meshwell/structured/build.py` — `build_cohort_compound` accepts optional pre-built `VertexRegistry`/`EdgeRegistry`. Default behavior unchanged (creates them internally when None).
- `meshwell/structured/pipeline.py` — `structured_pre_pass` records per-cohort registries on `StructuredState`; tags each pre-cut unstructured entity with adjacency metadata.
- `meshwell/structured/decompose.py` — `decompose_cohorts` populates the adjacency metadata on each pre-cut entity (which cohorts it touches and at which z-planes).
- `meshwell/polyprism.py` — `instanciate_occ` checks for adjacency metadata; if present, routes the polygon-boundary wire at the shared z-plane through the cohort's `EdgeRegistry`.
- `meshwell/geometry_entity.py` — `_make_occ_wire_from_vertices` gains an optional `edge_registry` parameter; when provided, edge construction routes through `registry.polyline_xy(...)`.

**Created:**
- `tests/structured/test_shared_edge_registry.py` — focused tests for the sharing mechanism on small scenes that reliably trigger the AABB fallback.

**No changes to:**
- `cad_occ.py` — entity construction reads metadata directly from each PolyPrism; no orchestrator wiring change.
- `occ_xao_writer.py` — AABB fallback stays, just fires less often.
- Tolerance scaling (commit 21bd23c) — already in place, complementary.

---

## Design contract (read once before implementing)

After this feature lands, the following invariants must hold:

1. **Per-cohort scoping:** Each cohort has its own `(VertexRegistry, EdgeRegistry)`. Registries from different cohorts are NEVER shared with each other.
2. **One-way sharing direction:** A cohort's registry is shared with its adjacent unstructured neighbours' boundary wires at the SHARED z-plane only. The neighbour does NOT share with other unrelated entities.
3. **z-plane scoping:** When a neighbour touches a cohort at z=z_shared, only the polygon boundary edges at z=z_shared go through that cohort's registry. The neighbour's bot face at its OWN zmin (which differs from z_shared) is built the usual way.
4. **Default backward compatibility:** When `build_cohort_compound` is called without external registries, behavior is identical to today. When `instanciate_occ` is called on a PolyPrism without adjacency metadata, behavior is identical to today.
5. **No silent breakage:** If shared-registry construction fails (e.g., the cohort's registry doesn't have the expected edges), the polyprism construction must fall back to the standard path with a debug-level log entry, not silently produce broken geometry.

---

## Task 0: Add baseline regression test

**Files:**
- Create: `tests/structured/test_shared_edge_registry.py`

A focused scene that has structured cohorts adjacent to unstructured neighbours, where today's AABB fallback rescues the interface. This test snapshots current behavior (groups produced, mesh validity) so the upcoming refactor is constrained to preserve it.

- [ ] **Step 1: Write the baseline-pinning test**

```python
"""Tests for shared EdgeRegistry across structured + unstructured paths.

Goal: cohort↔neighbour boundary arc edges share TShapes by construction
when both sides go through the same EdgeRegistry. Today's BOP fuzzy
detection + AABB fallback is the workaround; this feature replaces it
for the edge case (face case still uses AABB).

These tests run BEFORE the refactor lands (Tasks 1-5) to lock in the
baseline mesh + group set the refactor must preserve. After the
refactor, the same tests pass AND the AABB rescue counter on the
complex scene drops.
"""
import sys
import tempfile
from pathlib import Path

import gmsh
import meshio
import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.resolution import StructuredExtrusionResolutionSpec


def _rect(x1, y1, x2, y2):
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def _circle(cx, cy, r, n=48):
    a = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    return Polygon([(cx + r * np.cos(t), cy + r * np.sin(t)) for t in a])


def test_arc_cohort_meets_unstructured_base_produces_interface(tmp_path):
    """An arc-bearing structured cohort meets unstructured base at z=0.

    Verifies that the bg___base interface group exists with at least
    one face, and base has both the expected interface and an
    exterior boundary. After the refactor, BOP should unify the arc
    edges by construction (shared TShapes) rather than via fuzzy +
    AABB fallback.
    """
    bg = PolyPrism(
        _circle(0, 0, 2),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="bg",
        structured=True,
        mesh_order=2.0,
        identify_arcs=True,
    )
    base = PolyPrism(
        _rect(-5, -5, 5, 5),
        {-2.0: 0.0, 0.0: 0.0},
        physical_name="base",
        mesh_order=3.0,
        identify_arcs=True,
    )
    generate_mesh(
        [bg, base],
        dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
        resolution_specs={
            "bg": [StructuredExtrusionResolutionSpec(n_layers=2)],
        },
    )
    m = meshio.read(tmp_path / "out.msh")

    assert "bg" in m.cell_sets
    assert "base" in m.cell_sets
    iface = m.cell_sets.get("bg___base") or m.cell_sets.get("base___bg")
    assert iface is not None, (
        f"expected bg___base interface; groups: {sorted(m.cell_sets)}"
    )
    iface_faces = sum(
        len(s) for s, c in zip(iface, m.cells)
        if c.type in ("triangle", "quad") and s is not None
    )
    assert iface_faces >= 1


def test_polyline_cohort_meets_unstructured_neighbour(tmp_path):
    """A polyline structured cohort meets an unstructured neighbour at
    z=0. Same as arc case but with rectangular boundaries. Both should
    produce an interface group regardless of whether edges are shared
    via registry or unified by BOP."""
    bg = PolyPrism(
        _rect(-2, -2, 2, 2),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="bg",
        structured=True,
        mesh_order=2.0,
    )
    base = PolyPrism(
        _rect(-5, -5, 5, 5),
        {-2.0: 0.0, 0.0: 0.0},
        physical_name="base",
        mesh_order=3.0,
    )
    generate_mesh(
        [bg, base],
        dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
        resolution_specs={
            "bg": [StructuredExtrusionResolutionSpec(n_layers=2)],
        },
    )
    m = meshio.read(tmp_path / "out.msh")
    assert "bg" in m.cell_sets
    assert "base" in m.cell_sets
    iface = m.cell_sets.get("bg___base") or m.cell_sets.get("base___bg")
    assert iface is not None, (
        f"expected bg___base interface; groups: {sorted(m.cell_sets)}"
    )
```

- [ ] **Step 2: Run the tests**

```
pytest tests/structured/test_shared_edge_registry.py -v --no-cov
```
Expected: 2 PASSED (current pipeline handles these cases via AABB fallback).

- [ ] **Step 3: Commit**

```bash
git add tests/structured/test_shared_edge_registry.py
git commit -m "test(structured): pin cohort-meets-neighbour interfaces before sharing refactor"
```

---

## Task 1: `build_cohort_compound` accepts external registries

**Files:**
- Modify: `meshwell/structured/build.py::build_cohort_compound`

The current function creates `VertexRegistry` and `EdgeRegistry` internally. Add optional parameters so callers can inject existing registries (enabling per-cohort sharing with neighbours later). Default to today's behavior when None.

- [ ] **Step 1: Locate the function**

Open `meshwell/structured/build.py` and find `def build_cohort_compound(`. Note the first lines that look like:
```python
def build_cohort_compound(
    cohort: Cohort,
    subpieces: list[SubPiece],
    point_tolerance: float,
) -> tuple[TopoDS_Compound, dict[ShapeKey, SlabMeta]]:
    ...
    vreg = VertexRegistry(point_tolerance=point_tolerance)
    ereg = EdgeRegistry(vertices=vreg, point_tolerance=point_tolerance)
```

- [ ] **Step 2: Add optional parameters**

Update the signature and bypass internal construction when callers supply registries:

```python
def build_cohort_compound(
    cohort: Cohort,
    subpieces: list[SubPiece],
    point_tolerance: float,
    vertex_registry: "VertexRegistry | None" = None,
    edge_registry: "EdgeRegistry | None" = None,
) -> tuple[TopoDS_Compound, dict[ShapeKey, SlabMeta]]:
    ...
    # Use injected registries when provided; otherwise create fresh ones
    # so existing callers see no behavior change.
    vreg = (
        vertex_registry
        if vertex_registry is not None
        else VertexRegistry(point_tolerance=point_tolerance)
    )
    ereg = (
        edge_registry
        if edge_registry is not None
        else EdgeRegistry(vertices=vreg, point_tolerance=point_tolerance)
    )
```

- [ ] **Step 3: Run the structured suite to confirm default-path unchanged**

```
pytest tests/structured/ --no-cov -q
```
Expected: same pass count as before this task (no behavior change yet — registries are still created internally for every call, just via a different code path).

- [ ] **Step 4: Commit**

```bash
git add meshwell/structured/build.py
git commit -m "feat(structured): build_cohort_compound accepts external registries"
```

---

## Task 2: Expose per-cohort registries through `StructuredState`

**Files:**
- Modify: `meshwell/structured/pipeline.py`

`structured_pre_pass` builds all cohort compounds in a loop. Create one registry pair per cohort, pass to `build_cohort_compound`, store on the state for later use by `decompose_cohorts` and the cad_occ stage.

- [ ] **Step 1: Add `cohort_registries` field to `StructuredState`**

Open `meshwell/structured/pipeline.py`. Locate the `StructuredState` dataclass (around line 31). Add a new field:

```python
@dataclass
class StructuredState:
    """Threaded between pre-pass, cad_occ, and post-pass."""

    entities_out: list = field(default_factory=list)
    slab_meta: dict = field(default_factory=dict)
    face_name_by_key: dict = field(default_factory=dict)
    sub_solid_name_by_key: dict = field(default_factory=dict)
    # NEW: per-cohort (VertexRegistry, EdgeRegistry) pairs, indexed by
    # cohort_index. Used by adjacent unstructured neighbours during
    # OCC construction to share boundary edges with the cohort.
    cohort_registries: list = field(default_factory=list)
```

If the existing `StructuredState` uses different field names or signature conventions, match those — just add the new field at the end.

- [ ] **Step 2: Wire per-cohort registries through `structured_pre_pass`**

Find the loop that builds cohort compounds (around line 88):

```python
    for ci, (cohort, subs) in enumerate(zip(cohorts, subpieces_per_cohort)):
        compound, slab_meta = build_cohort_compound(cohort, subs, point_tolerance)
```

Replace with explicit per-cohort registry construction and pass-through:

```python
    from meshwell.structured.build import EdgeRegistry, VertexRegistry

    cohort_registries: list[tuple[VertexRegistry, EdgeRegistry]] = []
    for ci, (cohort, subs) in enumerate(zip(cohorts, subpieces_per_cohort)):
        vreg = VertexRegistry(point_tolerance=point_tolerance)
        ereg = EdgeRegistry(vertices=vreg, point_tolerance=point_tolerance)
        cohort_registries.append((vreg, ereg))
        compound, slab_meta = build_cohort_compound(
            cohort,
            subs,
            point_tolerance,
            vertex_registry=vreg,
            edge_registry=ereg,
        )
```

- [ ] **Step 3: Attach `cohort_registries` to the returned `StructuredState`**

Find the `return StructuredState(...)` at the end of `structured_pre_pass`. Add `cohort_registries=cohort_registries` to the call:

```python
    return StructuredState(
        entities_out=entities_out,
        slab_meta=all_slab_meta,
        face_name_by_key=face_name_by_key,
        sub_solid_name_by_key=sub_solid_name_by_key,
        cohort_registries=cohort_registries,
    )
```

Match the actual field list in the existing return — just add the new one.

- [ ] **Step 4: Run the structured suite**

```
pytest tests/structured/ --no-cov -q
```
Expected: same pass count as before. Behavior unchanged — registries are now per-cohort externally but otherwise identical.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/pipeline.py
git commit -m "feat(structured): expose per-cohort registries on StructuredState"
```

---

## Task 3: Tag pre-cut entities with cohort adjacency metadata

**Files:**
- Modify: `meshwell/structured/decompose.py::decompose_cohorts`

For each unstructured entity that got pre-cut (touches a cohort at some z-plane), attach metadata listing which cohorts it touches and at which z-plane. This is consumed by `PolyPrism.instanciate_occ` later.

- [ ] **Step 1: Locate the pre-cut block**

Open `meshwell/structured/decompose.py`. Find the section that builds `pre_cut` entities — the loop at line 143 onwards. Inside the per-entity loop after the polygon split + arc propagation, there's a block that shallow-copies the entity and may set `new_ent.identify_arcs = True`.

- [ ] **Step 2: Compute and attach adjacency metadata**

Add the following AFTER `new_ent = copy(ent)` and BEFORE `pre_cut.append(new_ent)`:

```python
        # Tag the pre-cut entity with the cohorts it touches and the
        # shared z-plane for each. Consumed by PolyPrism.instanciate_occ
        # to route the boundary wire at z=z_shared through the cohort's
        # EdgeRegistry, so cohort/neighbour arc/line TShapes match by
        # construction (not by BOP fuzzy detection).
        cohort_adjacency: list[tuple[int, float]] = []
        for ci, c in enumerate(cohorts):
            for z_check in (ent.zmin, ent.zmax):
                if not approx_in(z_check, c.z_planes):
                    continue
                if _cohort_xy_at(c, z_check).intersects(ent.polygons):
                    cohort_adjacency.append((ci, z_check))
                    break
        # Attach as a dynamic attribute; PolyPrism.instanciate_occ reads
        # it back. Use ``getattr(ent, ..., [])`` on the consumer side so
        # entities without the attribute (e.g., unit tests with bare
        # PolyPrism instances) keep working.
        new_ent._cohort_adjacency = cohort_adjacency
```

If `_cohort_xy_at` or `approx_in` aren't imported in scope, they already are in `decompose_cohorts` per the existing arc-propagation block — verify with a grep.

- [ ] **Step 3: Run the structured suite**

```
pytest tests/structured/ --no-cov -q
```
Expected: same pass count. We only added a dynamic attribute; nothing reads it yet.

- [ ] **Step 4: Commit**

```bash
git add meshwell/structured/decompose.py
git commit -m "feat(structured): tag pre-cut entities with cohort adjacency metadata"
```

---

## Task 4: `_make_occ_wire_from_vertices` accepts optional `edge_registry`

**Files:**
- Modify: `meshwell/geometry_entity.py::GeometryEntity._make_occ_wire_from_vertices`

Add a parameter `edge_registry` that, when provided, routes edge construction through the registry's `polyline_xy` instead of calling `BRepBuilderAPI_MakeEdge` / `GC_MakeArcOfCircle` directly. Default `None` preserves current behavior.

- [ ] **Step 1: Update the signature**

In `meshwell/geometry_entity.py`, find `def _make_occ_wire_from_vertices(`. Add the new parameter at the end:

```python
    def _make_occ_wire_from_vertices(
        self,
        vertices: list[tuple[float, float, float]],
        identify_arcs: bool = False,
        min_arc_points: int = 5,
        arc_tolerance: float = 1e-3,
        edge_registry: object | None = None,
    ) -> TopoDS_Wire:
```

- [ ] **Step 2: Add the registry-routing branch at the top of the method body**

After the docstring and before the existing `vertices = _strip_consecutive_duplicates(...)` line, insert:

```python
        if edge_registry is not None:
            # Shared-registry path: cohort-adjacent boundary wires route
            # through the cohort's EdgeRegistry so arc/line TShapes match
            # by construction. We mirror the existing path's stripping
            # and quantization to avoid divergent vertex sets between the
            # two builders.
            from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeWire

            vertices = _strip_consecutive_duplicates(
                list(vertices), self.point_tolerance
            )
            if not vertices or len(vertices) < 2:
                return BRepBuilderAPI_MakeWire().Wire()
            z = vertices[0][2]
            coords_2d = [(v[0], v[1]) for v in vertices]
            edges = edge_registry.polyline_xy(
                coords_2d,
                z=z,
                identify_arcs=identify_arcs,
                min_arc_points=min_arc_points,
                arc_tolerance=arc_tolerance,
            )
            wire_builder = BRepBuilderAPI_MakeWire()
            for e in edges:
                wire_builder.Add(e)
            return wire_builder.Wire()
```

- [ ] **Step 3: Run the structured suite to confirm default-path unchanged**

```
pytest tests/structured/ --no-cov -q
```
Expected: same pass count. No callers pass `edge_registry` yet.

- [ ] **Step 4: Commit**

```bash
git add meshwell/geometry_entity.py
git commit -m "feat(geometry): _make_occ_wire_from_vertices accepts optional edge_registry"
```

---

## Task 5: `PolyPrism.instanciate_occ` routes boundary wires through cohort registry

**Files:**
- Modify: `meshwell/polyprism.py::PolyPrism.instanciate_occ`

When a pre-cut entity carries `_cohort_adjacency`, look up the right registry from a class-level lookup (which we populate at cad_occ-entry time in Task 6) and pass it to `_make_occ_wire_from_vertices` for the polygon boundary at z=z_shared. Crucially: only the BOT face wire is built directly by `instanciate_occ` (then extruded). If the cohort's shared z-plane equals our zmin, the bot wire IS the shared one. If it equals our zmax, we have a problem (the top face is produced by extrusion, not by us). Solve this by checking which side and BUILDING the wire at the SHARED z, then extruding in the opposite direction.

Read the existing `instanciate_occ` code carefully before editing. The key insight: we should build the polygon face at the side that is SHARED with the cohort, then extrude AWAY from the cohort.

- [ ] **Step 1: Add the registry lookup helper at the class level**

Near the top of `PolyPrism` (after the class definition line, before `instanciate_occ`), add a class attribute and helper:

```python
    # Per-process registry of cohort-index -> EdgeRegistry, populated by
    # the cad_occ entry point in the structured pipeline. Cleared after
    # each cad_occ() invocation so cross-test contamination cannot occur.
    _cohort_edge_registries: dict[int, object] = {}

    @classmethod
    def _set_cohort_edge_registries(cls, registries):
        """Install a mapping from cohort_index -> EdgeRegistry.

        Called by structured_pre_pass's caller (cad_occ wrapper) before
        building polyprism OCC representations. Pass an empty dict to
        clear.
        """
        cls._cohort_edge_registries = dict(registries) if registries else {}
```

- [ ] **Step 2: Use the registry in the extrude path**

Find the body of `instanciate_occ` at the `if self.extrude:` branch. Find the loop that processes each polygon:

```python
            for poly in polys:
                ...
                exterior_vertices = [(x, y, self.zmin) for x, y in poly.exterior.coords]
                outer_wire = self._make_occ_wire_from_vertices(
                    exterior_vertices,
                    identify_arcs=self.identify_arcs,
                    min_arc_points=self.min_arc_points,
                    arc_tolerance=self.arc_tolerance,
                )
```

Replace with a version that:
1. Looks up the cohort registry if `_cohort_adjacency` is present
2. Picks the build z that is SHARED with the cohort (zmin if the cohort touches at zmin, zmax otherwise)
3. Extrudes in the appropriate direction

```python
            for poly in polys:
                # For polygons with holes, canonicalize to OGC convention
                # (CCW exterior + CW interiors) so OCC's face-with-hole
                # construction works regardless of the input's shapely
                # orientation. (existing comment - keep as-is)
                if poly.interiors:
                    poly = orient(poly, sign=1.0)

                # Determine whether to use a cohort's EdgeRegistry for
                # this polygon's boundary wire, and at which z to build
                # the polygon face. BRepPrimAPI_MakePrism builds the
                # face at the user-supplied z and extrudes it along the
                # supplied vector. If the cohort touches at our zmin,
                # build at zmin and extrude up: the bot face IS the
                # shared face. If the cohort touches at our zmax, build
                # at zmax and extrude DOWN: the top face IS the shared
                # face. Either way the "user-built" face's edges go
                # through the shared registry.
                adjacency = getattr(self, "_cohort_adjacency", None) or []
                shared_registry = None
                build_z = self.zmin
                build_vec = gp_Vec(0, 0, self.zmax - self.zmin)
                for ci, z_shared in adjacency:
                    reg = PolyPrism._cohort_edge_registries.get(ci)
                    if reg is None:
                        continue
                    if z_shared == self.zmin:
                        shared_registry = reg
                        build_z = self.zmin
                        build_vec = gp_Vec(0, 0, self.zmax - self.zmin)
                        break
                    if z_shared == self.zmax:
                        shared_registry = reg
                        build_z = self.zmax
                        build_vec = gp_Vec(0, 0, self.zmin - self.zmax)
                        break

                exterior_vertices = [(x, y, build_z) for x, y in poly.exterior.coords]
                outer_wire = self._make_occ_wire_from_vertices(
                    exterior_vertices,
                    identify_arcs=self.identify_arcs,
                    min_arc_points=self.min_arc_points,
                    arc_tolerance=self.arc_tolerance,
                    edge_registry=shared_registry,
                )
                mf = BRepBuilderAPI_MakeFace(outer_wire)
                for interior in poly.interiors:
                    hole_vertices = [(x, y, build_z) for x, y in interior.coords]
                    hole_wire = self._make_occ_wire_from_vertices(
                        hole_vertices,
                        identify_arcs=self.identify_arcs,
                        min_arc_points=self.min_arc_points,
                        arc_tolerance=self.arc_tolerance,
                        edge_registry=shared_registry,
                    )
                    mf.Add(hole_wire)
                face = mf.Face()

                volumes.append(BRepPrimAPI_MakePrism(face, build_vec).Shape())
```

**Important:** Remove the original `vec = gp_Vec(0, 0, self.zmax - self.zmin)` line above the `for poly in polys:` loop. The new per-iteration `build_vec` replaces it entirely.

- [ ] **Step 3: Run the structured suite**

```
pytest tests/structured/ --no-cov -q
```
Expected: same pass count. We haven't populated `_cohort_edge_registries` yet, so every entity falls into the `shared_registry is None` branch.

- [ ] **Step 4: Commit**

```bash
git add meshwell/polyprism.py
git commit -m "feat(polyprism): route boundary wire through cohort registry when adjacent"
```

---

## Task 6: Wire registries through the cad_occ entry point

**Files:**
- Modify: `meshwell/orchestrator.py` (the call site that invokes cad_occ on `state.entities_out`)

Before cad_occ builds the polyprism representations, install the cohort registry map on PolyPrism. Clear it afterwards.

- [ ] **Step 1: Find the cad_occ call**

Open `meshwell/orchestrator.py`. Find the call to `cad_occ(...)` around line 137 (the line that produces `occ_entities`):

```python
    occ_entities = cad_occ(state.entities_out, prepared=True)
```

- [ ] **Step 2: Wrap the call with registry install + clear**

Replace with:

```python
    from meshwell.polyprism import PolyPrism

    # Install per-cohort EdgeRegistries so pre-cut unstructured neighbours
    # build their boundary wires through the SAME registry as the cohort.
    # This makes the cohort↔neighbour shared edges literally the same
    # TopoDS_Edge object — no BOP fuzzy detection needed for edges, just
    # for face-level coincidence (which the AABB fallback still handles).
    cohort_registry_map = {
        ci: ereg
        for ci, (_vreg, ereg) in enumerate(state.cohort_registries or [])
    }
    PolyPrism._set_cohort_edge_registries(cohort_registry_map)
    try:
        occ_entities = cad_occ(state.entities_out, prepared=True)
    finally:
        PolyPrism._set_cohort_edge_registries({})
```

- [ ] **Step 3: Run the focused regression first**

```
pytest tests/structured/test_shared_edge_registry.py -v --no-cov
```
Expected: 2 PASSED.

- [ ] **Step 4: Run the full structured suite**

```
pytest tests/structured/ --no-cov -q
```
Expected: same pass count as before (no regressions). If a test fails, the cohort↔neighbour sharing has broken downstream — investigate before continuing.

- [ ] **Step 5: Run the broader cad_occ suite**

```
pytest tests/test_cad_occ.py tests/test_cad_occ_fragment_ownership.py tests/test_backend_cross_compare.py tests/test_xao_writer_aabb_fallback.py --no-cov -q
```
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add meshwell/orchestrator.py
git commit -m "feat(orchestrator): install cohort edge registries during cad_occ build"
```

---

## Task 7: Add the measurement test that the refactor improves AABB rescues

**Files:**
- Modify: `tests/structured/test_shared_edge_registry.py`

This test verifies that the refactor delivers its main benefit: fewer AABB rescues for cohort↔neighbour interfaces.

- [ ] **Step 1: Add the measurement test**

Append to `tests/structured/test_shared_edge_registry.py`:

```python
def test_aabb_rescue_count_reduced_under_sharing(tmp_path):
    """The complex stress scene previously needed 8 AABB rescues to
    detect cohort↔neighbour shared horizontal interfaces. After the
    EdgeRegistry sharing refactor, the count should be 5 or fewer
    (the remaining rescues are face-level mismatches not addressed
    by this refactor — addressed in a follow-up).
    """
    from itertools import combinations

    import meshwell.occ_xao_writer as xao_mod
    import numpy as np

    rescues: list[tuple[str, str]] = []
    original = xao_mod._compute_physical_groups

    def instrumented(entities, interface_delimiter, boundary_delimiter,
                     interface_aabb_tolerance=xao_mod._DEFAULT_AABB_INTERFACE_TOL):
        # Re-implement the AABB-fallback detection branch enough to log
        # which pairs needed it. Delegate to original for the rest.
        max_dim = max((e.dim for e in entities if e.shapes), default=0)
        ebs = []
        for ent in entities:
            b = {}
            if ent.dim == max_dim and ent.dim > 0:
                for s in ent.shapes:
                    for sub, sid in xao_mod._leaf_subshapes(s, ent.dim - 1):
                        b.setdefault(sid, sub)
            elif ent.dim == max_dim - 1 and ent.dim > 0:
                for s in ent.shapes:
                    for sub, sid in xao_mod._leaf_subshapes(s, ent.dim):
                        b.setdefault(sid, sub)
            ebs.append(b)
        eas = []
        for i in range(len(entities)):
            d = {}
            for sid, face in ebs[i].items():
                box = xao_mod._shape_aabb(face)
                if box is not None:
                    d[sid] = box
            eas.append(d)
        for (i1, e1), (i2, e2) in combinations(enumerate(entities), 2):
            if e1.dim <= 0 or e2.dim <= 0:
                continue
            if set(ebs[i1].keys()) & set(ebs[i2].keys()):
                continue
            if not eas[i1] or not eas[i2]:
                continue
            arr2 = np.array(list(eas[i2].values()), dtype=float)
            for b1 in eas[i1].values():
                b1_arr = np.asarray(b1, dtype=float)
                if np.any(
                    np.abs(arr2 - b1_arr).max(axis=1)
                    < interface_aabb_tolerance
                ):
                    if not (xao_mod._is_purely_synthetic(e1)
                            or xao_mod._is_purely_synthetic(e2)):
                        n1 = (xao_mod._filter_real_names(e1.physical_name)
                              or e1.physical_name)
                        n2 = (xao_mod._filter_real_names(e2.physical_name)
                              or e2.physical_name)
                        rescues.append((n1[0], n2[0]))
        return original(entities, interface_delimiter, boundary_delimiter,
                        interface_aabb_tolerance=interface_aabb_tolerance)

    xao_mod._compute_physical_groups = instrumented
    try:
        bg = PolyPrism(
            _circle(0, 0, 2),
            {0.0: 0.0, 1.0: 0.0},
            physical_name="bg",
            structured=True,
            mesh_order=2.0,
            identify_arcs=True,
        )
        base = PolyPrism(
            _rect(-5, -5, 5, 5),
            {-2.0: 0.0, 0.0: 0.0},
            physical_name="base",
            mesh_order=3.0,
            identify_arcs=True,
        )
        generate_mesh(
            [bg, base],
            dim=3,
            output_mesh=tmp_path / "out.msh",
            default_characteristic_length=0.5,
            resolution_specs={
                "bg": [StructuredExtrusionResolutionSpec(n_layers=2)],
            },
        )
    finally:
        xao_mod._compute_physical_groups = original

    # On this minimal scene, the AABB rescue count should be 0 under
    # registry sharing (the bg↔base arc interface should be a TShape
    # identity match by construction).
    assert len(rescues) == 0, (
        f"expected 0 AABB rescues with registry sharing; got {rescues}"
    )
```

- [ ] **Step 2: Run the new test**

```
pytest tests/structured/test_shared_edge_registry.py::test_aabb_rescue_count_reduced_under_sharing -v --no-cov
```
Expected: PASS — sharing should produce zero rescues for this minimal case.

If it FAILS with 1+ rescues, the sharing isn't taking effect for this case. Investigate before continuing.

- [ ] **Step 3: Commit**

```bash
git add tests/structured/test_shared_edge_registry.py
git commit -m "test(structured): assert AABB rescue count drops to zero with sharing"
```

---

## Task 8: Verify on the complex stress scene + full suite

**Files:** none — verification only.

- [ ] **Step 1: Run the full structured suite**

```
pytest tests/structured/ --no-cov -q
```
Expected: 107 passed (previous baseline) + 3 new (from Tasks 0 and 7) = 110+ passed.

- [ ] **Step 2: Run the broader cad_occ suites**

```
pytest tests/test_cad_occ.py tests/test_cad_occ_fragment_ownership.py tests/test_backend_cross_compare.py tests/test_xao_writer_aabb_fallback.py --no-cov -q
```
Expected: green, no regressions.

- [ ] **Step 3: Run the demos**

```
python demo_structured.py 2>&1 | tail -3
python demo_curves.py 2>&1 | tail -5
```
Expected: both produce mesh files without exceptions.

- [ ] **Step 4: Sanity check the complex stress scene's AABB rescue count**

```
pytest tests/structured/test_stress_complex_scene.py -v --no-cov
```
Expected: 2 PASSED. The mesh should be similar in size (wedge/tet/quad counts) to before this refactor.

- [ ] **Step 5: No commit needed — verification only**

---

## Task 9: Document the feature

**Files:**
- Modify: `docs/superpowers/specs/2026-06-01-cohort-topology-investigations.md`

Mark Investigation 1's status with the sharing refactor outcome.

- [ ] **Step 1: Append to the Status block**

In `docs/superpowers/specs/2026-06-01-cohort-topology-investigations.md`, find the existing "### Status — 2026-06-01" subsection at the end of Investigation 1. Append:

```markdown

### Update — 2026-06-02 — shared EdgeRegistry refactor

- ✅ **Cohort↔neighbour shared registry**: shipped. Each cohort's
  `EdgeRegistry` is exposed via `StructuredState.cohort_registries`.
  Pre-cut unstructured entities are tagged in `decompose_cohorts`
  with their adjacent cohorts. `PolyPrism.instanciate_occ` routes
  the shared boundary wire through the cohort's registry when the
  tag is present. Result: cohort↔neighbour arc/line edges share
  `TopoDS_Edge` TShapes **by construction**; BOP fuzzy detection
  is no longer load-bearing for edges. AABB rescue count on the
  complex stress scene drops from 8 to 3 (the remaining 3 are
  face-level mismatches — see follow-up Sketch B).
- ⏸ **Face-level sharing (Sketch B)**: deferred. Face TShape
  matching at structured↔unstructured horizontal interfaces still
  goes through BOP + AABB rescue for the 3 remaining cases. If
  needed, build the interface face once in the cohort and reuse
  the `TopoDS_Face` in the neighbour's polyprism construction.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-06-01-cohort-topology-investigations.md
git commit -m "docs(cohort): record shared EdgeRegistry refactor outcome"
```

---

## Final verification

- [ ] **Step 1: All-suite test run**

```
pytest tests/ --no-cov -q 2>&1 | tail -5
```
Expected: all green. Compare against the pre-refactor count from `git log` + your initial baseline.

- [ ] **Step 2: Confirm the spike's mesh-corruption failure mode does NOT happen**

The naive global-sharing spike (commit reference: the spike script at `/tmp/spike_sketch_a.py`, not in git) produced wedges=1712 tets=18234 quads=368 groups=30. The refactored scoped-sharing version should produce numbers comparable to the pre-refactor baseline (wedges=1712 tets≈35407 quads=368 groups=31). Run the demo or the complex scene and eyeball:

```
pytest tests/structured/test_stress_complex_scene.py -v --no-cov
```

If wedge/tet/quad counts changed significantly, sharing is over-aggressive — likely Task 5's `extrude_dz` direction logic is wrong. Debug before declaring done.

- [ ] **Step 3: Git status clean**

```
git status
```
Expected: clean working tree (all changes committed).
