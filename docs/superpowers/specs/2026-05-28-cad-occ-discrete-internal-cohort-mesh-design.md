# Phase 3 — Discrete Internal Cohort Mesh

**Date:** 2026-05-28
**Status:** Spec (brainstormed)
**Supersedes:** `2026-05-27-cad-occ-cohort-preshared-faces-design.md` (Phase 1) and
`2026-05-27-cad-occ-cohort-topology-builder-design.md` (Phase 2) once stable.
**Companion plans:** to be created via `superpowers:writing-plans`.

## Goal

Sidestep the N² BOP fragment cost on structured slabs by not making
internal piece volumes OCC entities at all. Each connected z-component of
structured slabs (a *cohort*) becomes a single OCC solid whose outer
envelope cad_occ fragments against unstructured neighbors. All interior
piece volumes and piece-to-piece interfaces are pure gmsh discrete
entities, built directly during the mesh stage.

The cohort outer envelope has:

- **Top shell** — per-piece OCC sub-faces (subdivided) so that an
  unstructured neighbor above shares OCC topology with each piece (for
  fragment conformality and per-piece BCs).
- **Bottom shell** — symmetric to top.
- **Lateral wall** — one OCC face per outline edge (or arc strip),
  *not* subdivided by interior piece boundaries. See the **Invariants**
  section below for the constraint this depends on.

cad_occ sees N cohort solids instead of M sub-prisms (M ≫ N), shrinking
the fragment graph by ~the average cohort size.

## Why this supersedes Phase 1+2

Phase 1 (pre-shared vertical faces) and Phase 2 (full cohort topology
builder with shared vertical+lateral faces) both still ship one
TopoDS_Solid per (slab, piece). They reduce *duplicate* sub-shapes inside
the BOP graph but don't reduce its *node count*. Phase 3 collapses the
node count itself by hiding interior topology from OCC.

The Phase 2 cohort_topology builder remains useful: a stripped-down
subset of it constructs the cohort envelope (no interior bookkeeping).

## Architecture

```
StructuredPlan
   │
   ├─► group slabs by component_index  (already populated)
   │
   ▼
build_cohort_envelopes(plan)  ───►  CohortEnvelopeResult
   │                                  ├── per-cohort TopoDS_Solid
   │                                  ├── per-piece top OCC sub-face
   │                                  ├── per-piece bottom OCC sub-face
   │                                  └── per-outline-edge OCC lateral
   ▼
cad_occ.fragment_all(cohort_solids + unstructured_solids)
   │
   ▼
PhantomMap  (FaceKey → gmsh face tags, EdgeKey → gmsh edge tags;
             keys for interior pieces stay symbolic — no OCC face exists)
   │
   ▼
apply_structured_mesh
   ├── stamp per-piece top sub-face mesh onto cohort top sub-faces
   ├── stamp per-piece bottom sub-face mesh onto cohort bottom sub-faces
   ├── stamp un-subdivided lateral mesh onto cohort lateral OCC faces
   ├── for each interior interface (piece A | piece B inside cohort):
   │       gmsh.model.addDiscreteEntity(2, …) + addElements(2, …)
   │       record FaceKey → discrete tag in PhantomMap
   └── for each piece volume:
           gmsh.model.addDiscreteEntity(3, …) + addElements(3, …)
           assign physical group per piece.physical_name
```

## Components

### 1. Cohort envelope builder (`meshwell/structured/cohort_envelope.py`)

Replaces and dramatically simplifies `cohort_topology.py`. Inputs: a
`StructuredPlan` and a `component_index`. Output: one
`CohortEnvelope` per cohort containing the OCC solid and accessors to:

- Per-piece top OCC sub-face by `FaceKey(slab_index, "top", piece_index)`.
- Per-piece bottom OCC sub-face by `FaceKey(slab_index, "bot", piece_index)`.
- Per-outline-edge lateral OCC face by `LateralKey(slab_index, piece_index, outer_edge_index)`
  — but keyed by *cohort outline edge*, not per-piece. Pieces whose
  exterior edge maps to the same outline edge share the lateral face.

The builder reuses the validated subset of Phase 2's vertex+edge dedup
infrastructure:

- Outline vertex dedup by snapped (x, y) → `corner_id`, with multi-arc
  vertex averaging + OCC tolerance for residual (already validated in
  `test_cohort_topology_multi_arc_corner.py`).
- Outline edge dedup by `(zlo, zhi, corner_id)` for vertical edges and
  `(z, edge_signature)` for horizontal outline edges (already validated
  in `test_cohort_topology_lateral_validity.py`).
- Arc lateral construction via `BRepFill::Face_s(bot_arc, top_arc)`
  (already validated in `test_stacked_concentric_arc_discs_mesh_clean`).

What it does NOT do (vs. Phase 2 cohort_topology):

- No interior horizontal edges between pieces of the same slab.
- No interior vertical edges between stacked slabs.
- No interior lateral faces between adjacent pieces.
- No per-piece lateral face subdivision. (See Invariants.)

For each slab in the cohort, the top shell is the union of per-piece top
sub-faces, all assembled into a single TopoDS_Shell with shared
boundary edges at piece-to-piece seams. Symmetric for bottom. Lateral
walls close the cohort by joining the outermost top edges to the
outermost bottom edges over the full z-extent of the cohort, with one
OCC lateral face per outline edge of the cohort's exterior boundary.

The cohort solid is built with `BRep_Builder.Add` (manual assembly) into
a closed shell, then wrapped in a `TopoDS_Solid`. Outer-normal
orientation is established by `shell.Reversed()` at solid level when
needed (same pattern as cohort_topology.py:assemble_cohort_sub_prism).

### 2. Cohort lateral wall — vertical extent

A cohort spans multiple z-intervals (one per slab in the stack). The
**lateral wall** must therefore include the slab-to-slab z-planes as
horizontal edges on its OCC representation, even though no interior
piece subdivision crosses them.

Concretely: each outline edge of the cohort generates one OCC lateral
face per slab z-interval (because slabs may have different outline
shapes per z, when one slab has a piece that another doesn't). When two
adjacent slabs share an outline edge with the same XY geometry, the
lateral OCC faces stack and their shared horizontal edge is a single
TopoDS_Edge.

This is the same lateral-edge dedup-by-`(zlo, zhi, corner_id)` rule that
Phase 2 validated; the envelope builder reuses it.

### 3. PhantomMap extensions

Today `PhantomMap` maps `FaceKey` to a *gmsh tag of an OCC face*. Phase 3
adds a second flavor: `FaceKey` for an interior piece-to-piece
interface maps to a *gmsh tag of a discrete 2D entity*.

```python
@dataclass
class PhantomMap:
    # Existing: OCC-backed
    face_keys_to_occ: dict[FaceKey, int]
    edge_keys_to_occ: dict[EdgeKey, int]
    ...
    # New (Phase 3): discrete-backed (interior cohort interfaces only)
    face_keys_to_discrete: dict[FaceKey, int]
```

Consumers (boundary assignment, mesh-size constraints) treat both flavors
uniformly: a `FaceKey` maps to a gmsh dim-2 entity tag, and the consumer
queries gmsh for boundary conditions or surface integrals as today.

### 4. Mesh stage (`apply_structured_mesh` extensions)

The existing `apply_structured_mesh` already uses
`addDiscreteEntity(3, -1, [])` and `addElements(3, vol_tag, …)` for
per-piece volume stamping. Phase 3 extends it:

- **Per-piece discrete 3D entity creation.** For each piece, allocate a
  discrete 3D entity, assign it the piece's `physical_name`, stamp the
  structured volume cells. This is largely already wired today.
- **Interior interface discrete 2D entity creation.** For each FaceKey
  that lies on a piece-to-piece interface inside a cohort (not on the
  cohort envelope), allocate a discrete 2D entity, stamp it with the
  triangulation/quadrangulation of that interface, and record the tag
  in `face_keys_to_discrete`. Two flavors:
  - **Horizontal interior interface** (between two stacked pieces of
    adjacent slabs in the same cohort): the interface is exactly the
    top sub-face mesh of the lower piece (equivalently, the bottom
    sub-face mesh of the upper piece — they're conformal by
    construction). Stamp the discrete 2D entity using the same node
    tags as the OCC top sub-face mesh.
  - **Vertical interior interface** (between two laterally-adjacent
    pieces of the same slab): the interface is a vertical strip whose
    horizontal extent is the shared piece-to-piece edge on the top
    (and bottom) shells, extruded over the slab's z-layers. Because
    the top shell is subdivided per piece, the shared edge is a single
    TopoDS_Edge with one shared row of node tags; vertical extrusion
    uses the same per-layer node tags as the two piece volumes. The
    discrete 2D entity is stamped quad-by-quad (or tri-by-tri if the
    slab is not recombined).
- **Node sharing.** Interior interfaces share nodes with the adjacent
  pieces' volumes by construction: the piece-volume stamping uses the
  same node tags from the top/bottom mesh that the interface stamps
  from. No `removeDuplicateNodes` magic needed at the interior. Cohort
  envelope sharing across cohorts still uses the existing
  `removeDuplicateNodes` pass.

### 5. Routing: piece → cohort owner

Each piece needs to know:

- Which cohort solid (TopoDS_Solid) it lives inside (for fragment-history
  lookup of containing volume after cad_occ fragments cohort solids
  against unstructured neighbors).
- Which envelope top/bottom OCC sub-face it owns (for mesh stamping).

This routing data lives in a new field on `CohortEnvelope` and is
threaded through `PhantomMap` for the mesh stage.

## Data flow

1. **Planner** (unchanged) produces `StructuredPlan` with
   `component_index` on each slab and `arrangements[component_index]`.
2. **Phantom stage** (new branch behind kill-switch): for each cohort,
   call `build_cohort_envelopes(plan, component_index)` to produce one
   OCC solid + per-piece top/bottom OCC sub-faces + per-outline-edge
   lateral OCC faces. PhantomShape entries become *per-cohort*
   bookkeeping objects rather than per-piece sub-prisms.
3. **cad_occ.fragment_all** consumes the cohort solids alongside
   unstructured solids. Fragment graph size drops from M (pieces) to
   N (cohorts).
4. **PhantomMap extraction** walks BOP history for cohort envelope OCC
   faces/edges. FaceKey/EdgeKey/VertexKey for interior pieces are
   created without OCC backing — they're symbolic until the mesh stage
   materializes them as discrete entities.
5. **apply_structured_mesh** consumes the plan + PhantomMap + cohort
   envelope routing. For each cohort: stamp top/bottom OCC sub-face
   meshes, stamp lateral OCC face meshes, then for each piece allocate
   a discrete 3D entity (or two — see Q4 (b) interior interfaces) and
   stamp its volume cells using shared top/bottom node tags. Allocate
   discrete 2D entities for interior interfaces.

## Kill-switch + coexistence

New kill-switch in `meshwell/structured/phantom.py`:

```python
# Phase 3 kill-switch. When True, build_phantom_shapes routes through
# build_cohort_envelopes (single OCC solid per cohort, discrete elements
# for interior pieces/interfaces). When False (default during
# stabilization), routes through Phase 1+2 path (per-piece OCC
# sub-prisms). Promote to default True once the Phase 3 path passes the
# full structured test suite end-to-end. Once that's done and the new
# path has soaked in production, delete the Phase 1+2 cohort code
# entirely (cohort_topology.py, _USE_COHORT_TOPOLOGY,
# _PRESHARE_VERTICAL_FACES, _build_phantom_shapes_via_cohort_topology).
_USE_DISCRETE_COHORT_MESH = False
```

Coexistence rules during stabilization:

- Default OFF. Existing structured pipeline behavior unchanged.
- Flip ON in tests that exercise the supported subset, parallel to how
  Phase 2 tests flip `_USE_COHORT_TOPOLOGY`.
- Phase 1 and Phase 2 paths remain reachable. Phase 3 is a separate
  third branch in `build_phantom_shapes`.

Promotion criteria (when to flip default to True):

1. Full structured test suite passes with the flag ON, including:
   - test_stress_stacked_patterns (annular arc xfail can stay xfail
     if the discrete path also can't handle the annular case; that's
     orthogonal to this work).
   - test_backend_cross_compare (gmsh-vs-cad_occ parity).
   - All `tests/structured/` and `tests/test_cad_occ_*` scenes.
2. Bench shows expected speedup (target: >3× on `_fragment_all` for
   scenes with M/N ≥ 5, matching Phase 2's measured 3.31×–3.87×).
3. No regressions in `test_palace_mixed_mesh_check.py`-class scenes
   (mixed structured + unstructured + Palace solver round-trip).

Cleanup commit (after soak):

- Delete `meshwell/structured/cohort_topology.py`.
- Delete kill-switches `_USE_COHORT_TOPOLOGY`, `_PRESHARE_VERTICAL_FACES`,
  `_USE_DISCRETE_COHORT_MESH`.
- Delete `_group_slabs_into_vertical_stacks`,
  `_build_phantom_shapes_via_cohort_topology`, and the Phase 1
  pre-share branch in `build_phantom_shapes`.
- Delete Phase 1+2 tests that are now obsolete (the cohort envelope
  builder has its own focused tests).

## Invariants this design depends on

**Load-bearing invariant: structured slabs have no XY unstructured
neighbors.** A cohort's lateral wall is a single un-subdivided OCC face
per outline edge. If an unstructured neighbor touched a cohort
laterally and the contact spanned a piece-to-piece interior boundary,
cad_occ's fragment would fail to align that boundary with the
neighbor's mesh (the OCC topology doesn't expose the boundary).

This invariant is currently enforced by the planner: structured slabs
that overlap unstructured entities in XY are either:

- Carved by the unstructured entity (becoming a thinner footprint), or
- Rejected by the StructuredOverlapError / mid-height-cut check.

If a future requirement weakens this invariant, **per-piece lateral
OCC subdivision becomes required** and the cohort envelope builder
must be extended (essentially: re-introduce the subset of Phase 2's
cohort_topology lateral path that we deliberately stripped here). See
the **Future work** section.

Other invariants:

- A cohort's top and bottom envelopes expose per-piece OCC sub-faces.
  Unstructured neighbors above/below fragment against these and inherit
  per-piece tagging via FaceKey → gmsh-tag routing.
- Interior interfaces between pieces of the same slab and between
  stacked pieces of adjacent slabs are addressable via FaceKey but
  materialized only as discrete 2D entities.

## Future work

**Unstructured XY neighbors.** If/when meshwell needs to support
structured slabs laterally adjacent to unstructured solids, the cohort
envelope must subdivide lateral OCC faces along piece-to-piece interior
boundaries that meet the cohort's exterior. The cohort envelope builder
gains a per-piece lateral path that reuses Phase 2's BRepFill-based arc
lateral construction. Mark this as a TODO at the top of
`cohort_envelope.py` and reference this section.

**Annular arc pieces** (already-known xfail in
`test_stress_stacked_patterns.py`). The discrete approach doesn't
directly fix annular face_partition pieces — transfinite meshing still
rejects them. If anything, the discrete path may make it *easier* to
support annular pieces (custom triangulation per piece is allowed),
but that's a separate spec.

**Per-piece mesh-size hints on interior interfaces.** Today
mesh-size constraints attached to interior FaceKeys would route through
gmsh's `setSize` on the OCC face. With Phase 3 those FaceKeys point at
discrete 2D entities; gmsh `setSize` on discrete entities behaves
differently. May need a parallel constraint-propagation path. Defer
until a concrete consumer surfaces.

## Testing strategy

New unit tests:

1. `tests/structured/test_cohort_envelope_build.py` — single cohort,
   two stacked slabs, simple polygon footprint. Verify the envelope
   solid is closed, top/bottom shells have correct per-piece sub-face
   counts, lateral wall has correct outline-edge count.
2. `tests/structured/test_cohort_envelope_arc.py` — single cohort with
   an arc on its outline. Verify the arc lateral is built via
   BRepFill::Face_s and has valid PCurves (bbox sanity, gmsh.open(xao)
   round-trip).
3. `tests/structured/test_cohort_envelope_concentric.py` — concentric
   arc discs in one cohort (the Phase 2 multi-arc snap scenario).
   Verify the envelope build succeeds with multi-arc vertex averaging.
4. `tests/structured/test_phase3_discrete_volumes.py` — end-to-end
   mesh: build a cohort with 4 pieces × 2 slabs = 8 piece volumes,
   verify 8 discrete 3D entities exist with correct physical names and
   total element count matches the structured layer counts.
5. `tests/structured/test_phase3_interior_interfaces.py` — end-to-end
   mesh: verify interior interfaces are materialized as discrete 2D
   entities with conformal node sharing (no orphan nodes, no duplicate
   nodes on the interface).
6. `tests/structured/test_phase3_top_bottom_conformality.py` — cohort
   with an unstructured neighbor above and below. Verify the
   unstructured tet mesh shares nodes with the cohort's per-piece top
   sub-face mesh stamps.

Existing tests run with `_USE_DISCRETE_COHORT_MESH=True` as a parity
gate:

- `test_stress_stacked_patterns.py` (excluding annular xfail).
- `test_backend_cross_compare.py`.
- `test_palace_mixed_mesh_check.py`.

Bench:

- `scripts/bench_cohort_envelope.py` — like `bench_cohort_topology.py`
  but for the new path. Target: full structured pipeline including
  discrete mesh stamping, compared head-to-head against Phase 1 and
  Phase 2.
- `scripts/bench_fragment_all.py` — extend to a third mode for
  Phase 3. Expected: fragment cost on Phase 3 should be N/M times
  Phase 2's cost (where N = cohort count, M = piece count).

## Open questions / risks

- **gmsh discrete-entity boundary inheritance.** When a discrete 3D
  entity is created with `addDiscreteEntity(3, -1, [])` (empty boundary
  list), gmsh doesn't track its boundary. If a discrete 2D interior
  interface needs to be in the boundary of two adjacent piece volumes,
  we must either: (a) include the discrete 2D tag in the discrete 3D
  boundary list, or (b) accept that interior interfaces are
  "free-floating" addressable entities not registered as volume
  boundaries. Investigate during implementation. Affects whether
  `gmsh.model.getBoundary` of a piece volume returns the interfaces.
- **Physical group assignment on discrete entities.** Confirm via a
  smoke test that `gmsh.model.addPhysicalGroup(3, [discrete_tag])` on
  a discrete 3D entity works and that the physical name is exported
  correctly to XAO/MSH/Palace.
- **PhantomMap key disambiguation.** Today `face_keys` is a single
  dict. Splitting into `face_keys_to_occ` and `face_keys_to_discrete`
  needs every consumer audited to handle both flavors. Could
  alternatively use one dict with a tagged value
  (`("occ" | "discrete", tag)`). Decide during implementation.

## Out of scope

- Recombination of discrete piece volumes into hexes. Today's recombine
  flag still operates on OCC volumes via gmsh's transfinite path. Phase
  3 piece volumes are built directly from `bot_cells` × layers, so
  recombine remains supported (just stamp hex elements instead of
  wedges) — but the existing element-stamp code already does this
  branch and needs no further work.
- Distributed/parallel mesh runs. Orthogonal.
- Changes to the planner. The planner already produces everything Phase
  3 needs (component_index, arrangements, face_partition_provenance).
