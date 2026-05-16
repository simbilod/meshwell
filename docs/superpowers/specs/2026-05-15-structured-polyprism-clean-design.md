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

### Layer A — Plan stage records the partition; CAD stage encodes it as one sub-prism per piece, no fuse

Before any OCC work, the planner computes, for each slab, a `face_partition: list[Polygon]` of the slab's union footprint into pairwise-disjoint pieces.

The CAD stage builds **one OCC sub-prism per partition piece** (e.g. `BRepPrimAPI_MakePrism` on the piece's bottom face, extruded by `(0, 0, h)`). Adjacent sub-prisms have geometrically coincident vertical faces. **No per-phantom fuse step is run.**

Why this works without fusing:

- The global fragment (`BRepAlgoAPI_BuilderAlgo`) runs against all entities anyway. It natively merges coincident input shapes: when sub-prism A's vertical face touches sub-prism B's vertical face, both inputs map to the same output face via `algo.Modified()`. Layer B's tag-tracking handles this with `dict[..., list[int]]`.
- The per-phantom fuse step in `feat/structured` (`structured_polyprism.py:989-1005`) was both redundant and harmful: the in-code comment at L1005 explicitly states it "over-attributes pieces" in the global Modified() tracking. Dropping it eliminates that failure mode.
- Internal seams don't constrain the slab volume mesh: the slab is built as a single `addDiscreteEntity(3, -1, [])` spanning all pieces, with our own wedge/hex elements. The OCC interior seam faces are marked "no auto-mesh" before `generate(2)` so they don't generate spurious 2D mesh inside the slab. (They still exist as OCC topology — the post-BOP map records them — but they carry no mesh data.)

Result of CAD stage: N sub-prisms registered per slab, all with known input OCC tags (bottom face, top face, lateral faces, all vertices, all edges). Top and bottom of each piece remain matched **by piece index**, so mesh-time routing is pure mapping — no point-in-polygon, no bbox lookup, no coordinate matching.

If a neighbour cuts only the top of one piece, `algo.Modified(piece_top_face)` yields multiple output sub-faces — all known to belong to that piece. The Python-side bookkeeping in Layer B records them as `list[output_tag]` for the same `(slab, "top", piece_index)` key.

Lateral OCC faces are 4-corner by construction (one rectangular vertical face per outer polygon edge, plus arc-rectangle for arc edges). "Mid-height" cut = a neighbour OCC vertex landing on a lateral face's interior with `zlo < z < zhi`. When detected (via `algo.Generated()` of the lateral face during the global fragment), that face is excluded from transfinite hints and meshed unstructured. Conformality is preserved via shared edges with neighbour OCC faces.

The face partition is computed once per slab from:
- the slab's own footprint
- the footprints of every non-structured entity that touches z=zlo or z=zhi within the slab's xy-bbox
- the footprints of sibling slabs sharing a horizontal interface

Arc segments are preserved through the partition by the provenance mechanism from the existing branch (ported as-is — the `2026-05-13-arc-provenance-face-partition.md` design is sound and reusable cleanly).

### Layer B — CAD stage builds an explicit OCC vertex/edge/face map via BOP history

Because the phantom is constructed piece-by-piece (Layer A), every piece's bottom face, top face, vertices, and edges are created by *our* code with known input OCC tags. We record at construction time:

```
input_face_by_key:    (slab i, "bot"/"top", piece k)            → input OCC face tag
input_edge_by_key:    (slab i, "bot"/"top", piece k, edge j)    → input OCC edge tag
input_vertex_by_key:  (slab i, "bot"/"top", piece k, corner j)  → input OCC vertex tag
input_lateral_by_key: (slab i, "outer_edge" m)                  → input OCC lateral face tag
```

After the global fragment runs against neighbours, walk `algo.Modified(in)`, `algo.Generated(in)`, `algo.IsDeleted(in)` for every recorded input tag to produce the post-BOP map:

```
output_faces_by_key:    (slab i, side, piece k)         → list[output face tag]
output_edges_by_key:    (slab i, side, piece k, edge j) → list[output edge tag]
output_vertices_by_key: (slab i, side, piece k, corner) → list[output vertex tag]
output_laterals_by_key: (slab i, outer_edge m)          → list[output lateral face tag]
```

`Modified()` returning a list (rather than a single tag) is the natural representation for neighbour-induced splits: a piece-top face cut by a neighbour becomes several output faces, all still belonging to that piece — mesh routing in Layer C iterates the list.

The map is keyed by slab/side/piece/element-index, not by OCC tag or coords, so it survives BOP-induced splits, merges, and ~fuzzy displacement. This mirrors the pattern already established in `meshwell/cad_occ.py` (which uses `Modified()` for fragment-piece ownership tracking).

When neighbour entities introduce new vertices on a lateral face mid-height, those vertices appear in `algo.Generated(input_lateral_face)`. The Layer A "mid-height cut" detection iterates those Generated() lists to decide which lateral faces lose their transfinite hint.

### Layer C — Mesh stage owns the top mesh; gmsh never independently meshes top OCC faces

For each slab, for each piece *k*:

1. Mark all `output_faces_by_key[(slab, "top", k)]` entries as "no auto-mesh" before calling `mesh.generate(2)`. The 2D mesher only runs on bottom faces and lateral faces.
2. After `mesh.generate(2)`, gather the 2D mesh from every `output_faces_by_key[(slab, "bot", k)]` entry (one or several output faces, all belonging to piece *k*). This is the piece's bottom mesh — routed by **mapping**, not by point-in-polygon or bbox.
3. Derive the piece's top mesh by transferring the bottom triangulation:
   - **Boundary mesh nodes** that sit on a piece corner: look up via `output_vertices_by_key[(slab, "top", k, corner)]`, use that OCC vertex's actual position.
   - **Boundary mesh nodes** that sit on a piece edge interior: look up via `output_edges_by_key[(slab, "top", k, edge)]`, walk the OCC top edge's parametric curve at the same parameter as the bottom edge, use that position.
   - **Interior mesh nodes**: `bottom_xy + (0, 0, h)` — these don't correspond to any OCC vertex/edge, so a pure translation is correct.
4. Stamp the derived mesh onto the corresponding output top face(s). If `output_faces_by_key[(slab, "top", k)]` has multiple entries (neighbour cut the top into several sub-faces), partition the derived mesh by point-in-polygon against each output top face's footprint. This is the *only* geometric routing in the pipeline, and it only fires when the user's neighbour actually introduced top-side cuts.
5. Build the slab volume as a single `addDiscreteEntity(3, -1, [])` per slab, with wedge (or hex when `recombine=True`) elements bridging bottom→top via the node-by-node map. Element orientation is explicitly maintained (bottom triangle CCW when viewed from below).
6. Run a single global `gmsh.model.mesh.removeDuplicateNodes` at the very end. Tolerance = `2 × max(slab.fragment_fuzzy_value for slab in plan.slabs)`. The 2× factor allows for the worst case where both a bottom and a top OCC vertex moved by `fuzzy` in *opposite* directions during BOP, putting nominally-coincident nodes up to `2 × fuzzy` apart.

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

### Internal — CAD stage adds the OCC map

```python
Side = Literal["bot", "top"]

@dataclass(frozen=True)
class FaceKey:
    slab_index: int
    side: Side
    piece_index: int

@dataclass(frozen=True)
class EdgeKey:
    slab_index: int
    side: Side
    piece_index: int
    edge_index: int

@dataclass(frozen=True)
class VertexKey:
    slab_index: int
    side: Side
    piece_index: int
    corner_index: int

@dataclass(frozen=True)
class LateralKey:
    slab_index: int
    outer_edge_index: int

@dataclass(frozen=True)
class PhantomMap:
    # Post-BOP OCC tags (lists because Modified() can split one input into many).
    output_faces:    dict[FaceKey, list[int]]
    output_edges:    dict[EdgeKey, list[int]]
    output_vertices: dict[VertexKey, list[int]]
    output_laterals: dict[LateralKey, list[int]]
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
3. CAD backend (cad_occ or cad_gmsh) calls phantom.instantiate(plan) → builds N
   sub-prisms per slab (one per partition piece via BRepPrimAPI_MakePrism), no
   per-phantom fuse. Records input OCC tags for every vertex/edge/face/lateral.
4. CAD backend runs the global fragment (against all entities including the
   sub-prisms). Adjacent sub-prisms' coincident vertical faces are merged by the
   global BOP. phantom.extract_map(algo) walks Modified()/Generated()/IsDeleted()
   to produce PhantomMap. Phantom volumes are marked for non-recursive removal
   so their faces survive for the mesh stage. Internal seam OCC faces are marked
   "no auto-mesh" so they don't produce spurious 2D mesh inside the slab.
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
