# Clean structured-polyprism design

**Date:** 2026-05-15
**Branch target:** `feat/structured-clean`, cut from `main`
**Supersedes:** the `feat/structured` branch (≈8.3k LOC of additions on a 5.7k base; `structured_polyprism.py` alone reached 2707 lines). The current branch will be archived; this design reimplements its features cleanly.

---

## Why a rewrite

The feat/structured branch shipped real value (structured-phantom strategy, conformal interfaces, arc-provenance partition) but accreted complexity faster than it could be paid down:

- A 2707-LOC single file mixing planning, CAD, mesh-building, periodic-mapping, and arc heuristics.
- Two parallel topology-matching strategies (`setPeriodic` + coord-snap fallback) that mask each other's failures.
- A cascade resolver for overlapping structured prisms that distributes `n_layers` proportionally and silently drops `identify_arcs`, with three layers of recovery code stacked on top.
- A `_StructuredPolyPrism` subclass dispatched from `PolyPrism.__new__`, mixing user-visible class identity with structured-mode-only state.

The pre-existing failure in `test_overlapping_structured_priority_resolution` (broken since the test was introduced) and the recurring "Unknown node N" / "No xy-twin" diagnostics confirm that the current architecture has run out of headroom.

This spec re-implements the same user-facing capability with one structural prevention strategy per failure source.

---

## Decisions taken during brainstorming

| Area | Decision |
| --- | --- |
| API surface | `PolyPrism(..., structured=True)`. No `__new__` dispatch, no subclass. Structured-ness is data, not type. `structured=True` with no `StructuredExtrusionResolutionSpec` in `resolutions=` raises at construction time; `StructuredExtrusionResolutionSpec` attached to a non-structured prism is ignored with a warning. |
| Layering parameters | New `StructuredExtrusionResolutionSpec(n_layers: list[int], recombine: bool = False)`, attached via the existing `resolutions=` list. `n_layers` length must equal `len(z_breakpoints) - 1`. Buffers must be uniform across z (no taper) when `structured=True`; non-uniform buffers raise. |
| Arc parameters | Stay on the entity (CAD-level): `identify_arcs`, `min_arc_points`, `arc_tolerance`. Not in the resolution spec. |
| Slab data | CAD-only. `Slab` carries geometry + provenance, not `n_layers` / `recombine`. Mesh parameters resolved in a second pass at mesh time. |
| Overlap policy | Volumetric overlap of structured prisms is allowed *iff* z-extents match exactly within tolerance AND `n_layers` agrees in the overlap region. Lower `mesh_order` wins. No proportional layer distribution. No cascade. Mismatch raises `StructuredOverlapError`. |
| Phantom strategy | Keep. Phantom punches voids in OCC fragmentation, mesh-stage refills with discrete 3D entities. |
| Top↔bottom matching | Layer A (mirror-symmetric face_partition) + Layer B (OCC vertex map via BOP history) + Layer C (mesh stage owns the top mesh, never lets gmsh mesh top OCC faces independently). |
| Lateral conformality | Lateral faces are 4-corner by construction when no neighbour cuts mid-height. When a neighbour does cut, drop transfinite for that face only — fall back to unstructured (conformality is preserved via shared edges). |
| Speed | No hard targets up front. Ship a structured logger + `bench_structured.py` from day 1, treat regressions as bugs once a baseline lands. |

---

## Phase-0 spike outcome

Run at `docs/superpowers/spikes/discrete_entity_displaced_vertex.py`, three phases:

- **P1 (synthetic baseline):** `addDiscreteEntity(3)` with 6-node prisms whose top vs bottom nodes differ in coords by 0, 1e-6, 1e-3 — all pass; quality drops from 0.5 by ≤ 3e-4. **Confirms gmsh accepts the mechanism.**
- **P2 (OCC phantom + BOP fragment):** OCC box fragmented against a stick, non-recursive remove, independent 2D mesh on top + bottom sub-faces, attempt to bridge with wedges. **Fails because independent meshing produces non-isomorphic triangulations** (different node counts on twin sub-faces). This invalidates any approach that meshes top and bottom independently and matches node-by-node.
- **P3 (stacked phantoms):** writes/reopens but produces negative-Jacobian elements when triangle orientation isn't controlled. Confirms element orientation is load-bearing in the discrete-entity build.

**Key spike conclusions:**

1. The discrete 3D entity mechanism is sound at the gmsh-API level.
2. Independent meshing of "twin" top/bottom OCC faces is **not** a viable building block — topology, not just coords, can diverge.
3. The mesh stage must own the top mesh derivation (Layer C), not delegate it to gmsh.
4. Element orientation must be explicitly maintained by the builder.

---

## The three-layer correctness strategy

### Layer A — Plan stage guarantees mirror-symmetric topology

Before any OCC work, the planner computes, for each slab, a single `face_partition: list[Polygon]` that is applied to *both* z=zlo and z=zhi. The phantom builds one OCC sub-prism per partition piece and fuses them. Result: bottom and top OCC face counts match by construction; sub-face *i* on the bottom has a 1:1 correspondence with sub-face *i* on the top.

Each sub-prism has 4-corner lateral faces *by construction* (one per piece-boundary segment). "Mid-height" cut = a neighbour OCC vertex landing on the lateral face's interior with `zlo < z < zhi`. When detected (during phantom build, via the BOP history walked for Layer B), that face is excluded from transfinite hints and meshed unstructured. Conformality is preserved via shared edges with neighbour OCC faces.

The face partition is computed once per slab from:
- the slab's own footprint
- the footprints of every non-structured entity that touches z=zlo or z=zhi within the slab's xy-bbox
- the footprints of sibling slabs sharing a horizontal interface

Arc segments are preserved through the partition by the provenance mechanism from the existing branch (ported as-is — the `2026-05-13-arc-provenance-face-partition.md` design is sound and reusable cleanly).

### Layer B — CAD stage builds an explicit OCC vertex map via BOP history

The structured pipeline owns the `BRepAlgoAPI_BuilderAlgo` (or equivalent) object that fragments the phantom against its neighbours. For each phantom sub-prism *p*:

1. At build time, record the input OCC vertex tags for each corner of *p*'s bottom face and each corner of *p*'s top face, keyed by `(piece_index, corner_index)`.
2. After running the BOP, walk `algo.Modified(in_vertex)`, `algo.Generated(in_vertex)`, `algo.IsDeleted(in_vertex)` for every recorded input vertex.
3. Build the map: `(piece i, bottom_corner k) → output_bot_vertex_tag` and `(piece i, top_corner k) → output_top_vertex_tag`.

The map is keyed by piece/corner identity, not by tag or coords, so it survives BOP-induced splits, merges, and ~fuzzy displacement. This mirrors the pattern already established in `meshwell/cad_occ.py` (which uses `Modified()` for fragment-piece ownership tracking).

Same idea is extended to edges: `(piece i, edge k) → output_edge_tag`, used by the builder to identify the OCC curves on which top boundary nodes must land.

### Layer C — Mesh stage owns the top mesh; gmsh never independently meshes top OCC faces

The mesh stage:

1. Marks all top sub-face entities as "no auto-mesh" before calling `mesh.generate(2)`. The 2D mesher only runs on bottom sub-faces and lateral faces.
2. For each phantom sub-prism, derives its top sub-face mesh by transferring the bottom triangulation:
   - **Boundary mesh nodes**: look up the corresponding top OCC vertex/edge via the Layer B map. Use the *actual position of that top OCC vertex* (BOP-displaced, read directly from OCC). For top boundary curve interior nodes, walk the OCC top edge's parametric curve and place nodes at the corresponding parameter as on the bottom.
   - **Interior mesh nodes**: `bottom_xy + (0, 0, h)` — these don't correspond to any OCC vertex, so a pure translation is correct.
3. Stamps the derived mesh directly onto the top sub-face entities (`gmsh.model.mesh.addNodes(2, top_sub_face, ...)` + `addElements(2, top_sub_face, ...)`).
4. Builds the slab volume as a single `addDiscreteEntity(3, -1, [])` and adds wedge (or hex when `recombine=True`) elements bridging bottom→top via the node-by-node map. Element orientation is explicitly maintained (bottom triangle CCW when viewed from below).
5. Runs a single global `gmsh.model.mesh.removeDuplicateNodes` at the very end. Tolerance = `2 × max(slab.fragment_fuzzy_value for slab in plan.slabs)`. The 2× factor allows for the worst case where both a bottom and a top OCC vertex moved by `fuzzy` in *opposite* directions during BOP, putting nominally-coincident nodes up to `2 × fuzzy` apart.

---

## Module layout

```
meshwell/structured/
    __init__.py        # public exports
    spec.py            # StructuredExtrusionResolutionSpec, Slab, StructuredPlan
    plan.py            # validator + planner; produces StructuredPlan + OverlapPair list
    phantom.py         # CAD-stage _StructuredPhantom proxy + Layer B vertex map
    builder.py         # mesh-stage Layer C: top mesh derivation + discrete 3D entity
    logging.py         # structured logger + per-phase timing decorators
```

`meshwell/polyprism.py` gains one new keyword: `structured: bool = False`. When `True`, the polyprism participates in the structured pipeline; when `False`, behaviour is unchanged.

The orchestrator detects `StructuredExtrusionResolutionSpec` instances on entities flagged `structured=True`, gathers them, and hands them to the planner. Non-structured entities pass through unchanged.

---

## Data model

### Public

```python
@dataclass(frozen=True)
class StructuredExtrusionResolutionSpec:
    n_layers: list[int]                     # one per z-interval of owning entity
    recombine: bool = False
```

### Internal — Plan stage

```python
@dataclass
class Slab:                                 # CAD-only; no mesh data
    footprint: Polygon | MultiPolygon
    zlo: float
    zhi: float
    physical_name: tuple[str, ...]
    source_index: int                       # → owning entity → owning spec
    z_interval_index: int                   # which (zlo,zhi) pair of the owner
    mesh_order: float
    identify_arcs: bool
    min_arc_points: int
    arc_tolerance: float
    fragment_fuzzy_value: float | None
    face_partition: list[Polygon]
    face_partition_provenance: list["PieceProvenance"]

@dataclass(frozen=True)
class OverlapPair:
    winner_slab_index: int                  # index into StructuredPlan.slabs
    loser_source_index: int
    loser_z_interval_index: int
    z_extent: tuple[float, float]

@dataclass(frozen=True)
class StructuredPlan:
    slabs: list[Slab]
    z_planes: list[float]                   # sorted unique z-planes
    overlaps: list[OverlapPair]             # for mesh-stage cross-check
```

### Internal — CAD stage adds the vertex map

```python
@dataclass(frozen=True)
class VertexKey:
    slab_index: int
    piece_index: int
    corner_index: int                       # within the piece's boundary
    side: Literal["bot", "top"]

@dataclass(frozen=True)
class EdgeKey:
    slab_index: int
    piece_index: int
    edge_index: int
    side: Literal["bot", "top"]

@dataclass(frozen=True)
class PhantomMap:
    bot_vertex_to_top_vertex: dict[int, int]       # OCC vertex tags
    bot_edge_to_top_edge: dict[int, int]
    occ_vertex_by_key: dict[VertexKey, int]
    occ_edge_by_key: dict[EdgeKey, int]
```

### Internal — Mesh stage resolves mesh parameters

```python
@dataclass(frozen=True)
class StructuredMeshPlan:
    slabs: list[Slab]                       # same slabs
    n_layers: list[int]                     # parallel to slabs
    recombine: list[bool]                   # parallel to slabs
```

Built in `builder.py` by walking `StructuredPlan.slabs`, looking each slab back up via `(source_index, z_interval_index)` to its `StructuredExtrusionResolutionSpec`, and for each `OverlapPair` verifying that loser and winner agree on `n_layers` (otherwise raise `StructuredMeshOverlapError`).

---

## Pipeline

```
1. User builds PolyPrism(..., structured=True, resolutions=[StructuredExtrusionResolutionSpec(...)])
2. Orchestrator gathers structured entities → calls plan.build(entities) → StructuredPlan
3. CAD backend (cad_occ or cad_gmsh) calls phantom.instantiate(plan) → registers
   phantom sub-prisms in the OCC scene and returns PhantomMap from BOP history
4. CAD backend runs the global fragment, marks phantoms for non-recursive removal
5. Mesh backend calls builder.apply(plan, phantom_map) which:
   a. resolves StructuredMeshPlan from plan + entity resolutions
   b. marks top sub-faces "no auto-mesh"
   c. calls gmsh.model.mesh.generate(2) (bottom + lateral faces only)
   d. for each slab: derives top mesh from bottom + phantom_map, stamps onto top sub-faces,
      builds slab volume as discrete 3D entity with wedge/hex elements
   e. calls gmsh.model.mesh.removeDuplicateNodes(tol = 2 * max_fuzzy)
6. Mesh backend runs the 3D mesh for non-structured volumes (the discrete slabs are
   already meshed and won't be re-touched)
```

---

## Testing

A clean test directory `tests/structured/` mirroring `meshwell/structured/`:

```
tests/structured/
    test_spec.py                # data-class validation
    test_plan.py                # planner: overlap rule, face partition, arc provenance
    test_phantom.py             # CAD-stage: phantom map correctness via OCP fixtures
    test_builder.py             # mesh-stage: top-mesh derivation, prism orientation
    test_end_to_end.py          # full pipeline on representative scenes
    test_overlap_validation.py  # the policy-B mismatch errors
    test_lateral_cuts.py        # neighbour cuts mid-height → unstructured lateral fallback
    test_logging.py             # structured logger captures per-phase timing
```

Tests carried over from feat/structured (those that exercise behaviour the new design preserves) ported individually after each module lands. Tests that exercise the cascade (`test_overlapping_structured_priority_resolution` and similar) are **deliberately not ported** — the new overlap policy rejects those scenes; replacement tests cover the new error.

Phase-0 spike stays as a regression artifact at `docs/superpowers/spikes/discrete_entity_displaced_vertex.py`.

---

## Performance approach

Day-1 wiring:

- `meshwell/structured/logging.py` provides a `structured_logger` plus a `@phase_timed("plan")` / `@phase_timed("phantom")` / `@phase_timed("mesh")` decorator that records wall time and arbitrary counter increments.
- `scripts/bench_structured.py` is ported and adapted to print the structured logger's tape at the end.
- No hard numeric targets in v1. Once a baseline lands on a representative scene, regressions are bugs.

Logged hotspots from feat/structured to watch:
- Per-slab `gmsh.model.occ.getEntities(2)` rescans (O(n_slabs × n_faces) — replace with one global snapshot).
- Phantom build amortization (one BOP call per *entity*, not per slab).
- `removeDuplicateNodes` cost (O(n_nodes log n_nodes) — measure on dense scenes).

---

## Migration

1. Cut new branch from `main`: `git switch -c feat/structured-clean main`.
2. Land modules in pipeline order (`spec.py` → `plan.py` → `phantom.py` → `builder.py` → `logging.py`), each with its tests, each green before the next starts.
3. Wire into orchestrator behind a single feature path (presence of `StructuredExtrusionResolutionSpec` activates it).
4. Once the new pipeline passes its own test suite + the carried-over feat/structured tests, archive `feat/structured` (tag it for reference) and treat the new branch as the structured baseline.

No effort is spent on staying compatible with feat/structured's intermediate APIs (`_StructuredPolyPrism`, `Slab.n_layers` field, `PolyPrism(n_layers=...)` keyword). Those weren't released — clean break is cheaper than parallel maintenance.

---

## Out of scope (v1)

- Tapered structured prisms. `buffers` values must be uniform across z when `structured=True` (the planner rejects non-uniform buffers with a clear error).
- Mixed-element-type slabs (hex + wedge in the same slab).
- Distributed-memory phantom construction.
- Automatic recovery from neighbour cuts producing >4-corner lateral faces *and* requiring transfinite hints simultaneously — v1 falls back to unstructured on such faces; users who need structured laterals there will get an explicit warning telling them how to refactor.

---

## Open questions to resolve during implementation

- Exact API for marking top sub-faces "no auto-mesh". Candidates: per-entity `setMeshSize(0)` (probably no-op for this), `removeEmbedded`, or a custom plan flag respected by the mesh-stage hook. Pick one in `builder.py` design pass.
- Whether to expose the structured logger as a public API (so users can attach their own handlers) or keep it internal for v1. Lean: internal, promote in v2 if users ask.
- Boundary curve mesh transfer in Layer C: gmsh stores 1D mesh nodes on the OCC edges. Need to decide whether Layer C *also* derives top edge meshes from bottom edge meshes (likely yes, for consistency with face transfer) or relies on gmsh's 1D mesher producing matching counts on twin edges (fragile per spike learnings).
