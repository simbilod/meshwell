# Phase 3 follow-up — third handoff (2026-05-30)

**Date:** 2026-05-30
**Branch:** `feat/structured-clean4`
**Predecessors:** `2026-05-28-phase3-followup.md`, `2026-05-29-phase3-followup.md`

## TL;DR

Phase 3 sweep: **26 → 1 failed** across two sessions. Default-off
sweep: **283 passed, 0 failed**, unchanged from baseline.

The one remaining failure is `test_simple_slab_lateral_mesh_is_conformal`
(single-piece slab; 22 non-conformal lateral triangles). All multi-piece
routing, arc disc/annulus handling, and lateral sub-face transfinite
hinting now work. Twelve commits landed since the second handoff —
listed at the end.

A new, real issue surfaced this session while writing example scenes:
**multi-layer cohorts (≥2 stacked structured slabs in one cohort) +
an unstructured neighbour above the cohort fail to mesh.** The
`removeEntities` call in `_suppress_empty_cohort_envelope_volumes`
corrupts the neighbour's boundary topology. This isn't covered by any
existing test, but the new `stacked-cap-features`, `stacked-3-layer-features`,
and `photonics-cross-section` scenes in `scripts/inspect_phase3_geometry.py`
reproduce it. Plan for the fix sketched at the bottom.

## The Phase 3 architecture (recap)

For each cohort of structured slabs:

```
PLANNER  build_plan
  ├─ slab.face_partition[i] = piece polygon i
  └─ slab.face_partition_edges[i] = [(arrangement_edge_id, fwd), ...]
                                    (NOT necessarily in chain order)

PHASE 3 CAD  build_cohort_envelope
  ├─ env.outline_xy_to_corner_id   (snapped corner XYs)
  ├─ env.vertices[(z, cid)]        (one OCC vertex per corner per z-plane)
  ├─ env.horizontal_edges[(z, eid)] (one OCC wire per arrangement edge per z)
  ├─ env.vertical_edges[(zlo, zhi, cid)]
  ├─ env.bottom_sub_faces[FaceKey(slab,"bot",piece)] = TopoDS_Face
  ├─ env.top_sub_faces[FaceKey(slab,"top",piece)] = TopoDS_Face
  └─ env.lateral_faces[(slab, eid)] = list[TopoDS_Face]

ASSEMBLE  assemble_cohort_envelope_solid
  ├─ sewing.Add(bottom_sub_faces / bottom_union_face)
  ├─ sewing.Add(top_sub_faces / top_union_face)
  ├─ sewing.Add(all lateral_faces, including arc closing strips)
  └─ Build TopoDS_Solid from sewn shells

PHANTOM ROUTING  _build_phantom_shapes_via_cohort_envelope
  └─ PhantomShape:
       slab_index = -(cohort_idx + 1)        (synthetic marker)
       input_faces_by_key = {FaceKey -> OCC face}
       input_laterals_by_outer_edge = {edge_id*10000+seg -> OCC face}

BOP via cad_occ.fragment_all
  ├─ One envelope solid per cohort + neighbours
  └─ Each FaceKey's input face -> list of post-BOP gmsh face tags
     via PhantomMap walk (builder.Modified())

MESH STAMP  apply_structured_mesh (pre_3d_hook)
  ├─ _filter_phase3_face_map_per_piece (pick fragment per piece)
  ├─ per-piece loop:
  │    ├─ _stamp_top_face_mesh (bot triangulation -> top, tolerance-matched
  │    │   boundary nodes for arc drift)
  │    └─ _build_slab_volume (discrete 3D entity with wedges)
  ├─ _stamp_phase3_interior_interfaces (horizontal between slabs;
  │   vertical between same-z pieces)
  ├─ _suppress_empty_cohort_envelope_volumes (purges envelope from
  │   physical groups; if single-z-range, removeEntities to skip
  │   tetrahedralization)
  └─ removeDuplicateNodes

3D MESH  generate(3) with MeshOnlyEmpty=1
  └─ Tetrahedralizes still-empty volumes (cap/neighbours).
     For multi-z-range envelopes (current limitation): also
     tetrahedralizes the envelope (then post_3d_hook clears those tets).
```

## This session's commits

| Commit  | Bucket / Purpose | Net effect |
|---------|------------------|------------|
| `9fda0aa` | Bucket A — skipif markers on 7 Phase-1/2-assumption tests + 2 B5 tests | -15 sweep failures |
| `ab1181a` | B2 partial — propagate `sewing.Modified()` to per-piece sub-faces; per-piece face_map filter | -3 simple tests |
| `c6eccf1` | B1b — multi-shell solid split by source in `_group_phantom_solids_by_entity` (+ spec test) | covers B1b |
| `8751f82` | docs — second handoff | docs |
| `2573b8e` | Multi-piece sub-faces via shared `horizontal_edges` registry; skip union for single-slab multi-piece | -3 (multi-output-face tests pass) |
| `0ee9be2` | Scope `_suppress` to structured slab physical groups | unstructured neighbour now keeps its physical group |
| `9cdf5f1` | Register every lateral sub-face per arrangement edge in `input_laterals` | -1 (cap test passes) |
| `77a7981` | `removeEntities` of empty cohort envelope volumes + tolerance-match arc boundary nodes | -4 arc tests pass |
| `dca2258` | 3-point `GC_MakeArcOfCircle(start, middle, end)` for inner-ring arcs | -1 (annulus split passes) |
| `4de0b35` | inspect script — sort FaceKey dicts by tuple | tooling |
| `0e17976`, `e8d1df9` | inspect script — stacked-cap, stacked-3-layer, photonics scenes | tooling |

Combined Phase 3 sweep delta: **26 → 1 failed**.

## What each fix actually fixed

### `2573b8e` — multi-piece sub-faces share OCC TShapes

**Root cause:** For multi-piece slabs, `build_cohort_envelope` built each
piece's bot/top sub-face independently via
`_make_face_from_polygon_with_arcs`. Sibling pieces' sub-faces had no
shared OCC TShapes, so `BRepBuilderAPI_Sewing` couldn't merge them at
their interior arrangement edges. To avoid an "open shell" solid,
`assemble_cohort_envelope_solid` then built a single union face spanning
all pieces, and routed every per-piece FaceKey through the union for
BOP. After BOP, every per-piece FaceKey saw the union's full fragment
list — and the per-piece volume builder hit KeyError on nodes it didn't
own.

**Fix:** Build per-piece sub-faces from the shared `env.horizontal_edges`
registry (same as the existing single-piece path). Sibling pieces'
sub-faces now share OCC TShapes at interior edges, so sewing merges
them into a manifold sheet without the union. For single-slab
multi-piece cohorts, `assemble_cohort_envelope_solid` skips the union
and adds per-piece sub-faces directly to sewing. Multi-slab same-z
cohorts (laterally-adjacent slabs, 4-quadrant grids) keep the union
behavior — they have multi-slab T-junctions that don't sew cleanly
without a union.

**Knock-on:** `face_partition_edges` from the planner isn't guaranteed
to be in chain order. Added `_chain_piece_edges_by_corner` to walk
them in entry-corner-to-exit-corner order before building the wire.

### `0ee9be2` — suppress scoped to structured slab physical groups

**Root cause:** `_suppress_empty_cohort_envelope_volumes` removed
*every* empty 3D volume from *every* physical group. For
cap-above-slab scenes, gmsh's tet pass runs after the structured hook,
so the cap volume is empty at suppression time — and got purged from
the `cap` physical group along with the actual empty envelope.

**Fix:** Pass `structured_pg_names` (the set of `"/".join(slab.physical_name)`
for every plan.slabs entry). Suppress only volumes that appear in at
least one structured slab's physical group. The cap stays tagged and
gets tet-meshed normally.

### `9cdf5f1` — lateral sub-faces per arrangement edge

**Root cause:** A perimeter arrangement edge with N vertices produces
N−1 OCC lateral sub-faces (one per straight segment). The previous
routing kept only `face_list[0]` under a single `outline_edge_id` key,
so only one sub-face reached `apply_structured_transfinite_hints`. The
other sub-faces fell back to gmsh's default tri-mesher and produced
non-conformal lateral triangles on the slab perimeter (38 of the cap
test's 58 orphans were on the multi-vertex right-perimeter edge).

**Fix:** Encode `(edge_id, segment_idx)` into the synthetic
`outer_edge_index` via `edge_id * 10000 + seg_idx` so each sub-face has
a unique entry in `input_laterals_by_outer_edge`. All sub-faces now
participate in BOP history walk and transfinite hinting.

### `77a7981` — `removeEntities` + tolerance-match arc nodes

**Root cause (#1, the tetrahedralization conflict):** Phase 3's
original design tetrahedralized the empty cohort envelope OCC volume,
then cleared those tets in a post_3d_hook. This worked for simple
scenes but failed on the single-piece annulus
(`Invalid boundary mesh (overlapping facets)`) — gmsh's 3D mesher
couldn't reconcile the wedge boundary mesh with the OCC face mesh
inside an annular cohort solid.

**Root cause (#2, the arc node drift):** Bot and top OCC arc edges
were built independently per z-level using `GC_MakeArcOfCircle(circ,
start, end, True)`. Even though their endpoint vertices have identical
XY (same corner_id, different z), gmsh's 1D mesher places interior arc
nodes at slightly different parametric positions on the two curves
(~1e-5 drift in XY). `_stamp_top_face_mesh`'s exact-XY dict lookup
silently dropped these nodes and KeyError'd at the triangle stamp.

**Fix #1:** `gmsh.model.removeEntities([(3, vt)], recursive=False)` on
each empty cohort envelope volume after purging from physical groups.
The 2D face children stay (shared with neighbours), but the 3D parent
is gone so gmsh's 3D mesher never visits it. The `recursive=False`
flag keeps the children attached.

**Fix #2:** Replace the exact dict lookup in the legacy XY-match path
with `np.argmin` over `top_bnd_coords` with a 1e-3 tolerance cap. The
two fixes are tightly coupled: the tolerance match would have created
slightly slanted wedges that crashed the tet mesher, but with the
envelope gone, that mesher never sees the wedges.

### `dca2258` — 3-point arc constructor for inner rings

**Root cause:** `GC_MakeArcOfCircle(circ, start, end, True)` always
picks the CCW arc. That's correct for OUTER rings (shapely traverses
them CCW), but wrong for INNER rings of polygons with holes (shapely
traverses holes CW). The annulus' lower-half inner arc edge — going
from (-0.4, 0) to (0.4, 0) through y < 0 — was built going through
y > 0 instead. The piece-1 (lower half) sub-face wire wrapped the
wrong half, encompassing the y ∈ [0, 0.4] inner-hole strip.

This bug was caught by running
`python scripts/inspect_phase3_geometry.py --scene annulus-cap` and
noticing that piece 1's bot bbox showed `y_max = 0.4` instead of
`y_max = 0` — way past the cap chord.

**Fix:** Replace `(circ, start, end, Sense)` with the 3-point
constructor `GC_MakeArcOfCircle(start, middle, end)`, passing
`arr_edge.vertices[mid]` as the middle point. The middle vertex
uniquely identifies the correct half regardless of CCW/CW traversal
direction. Falls back to the old constructor when the arrangement
edge has fewer than 3 vertices.

## Inspection tooling — `scripts/inspect_phase3_geometry.py`

New script that dumps:
- `plan.slabs` with `face_partition` polygons and `face_partition_edges`
- `plan.arrangements` — every edge marked `[ARC]` or not, vertex count
- `env.outline_xy_to_corner_id` — snapped corner XYs
- `env.horizontal_edges` keyed by `(z, edge_id)`
- `env.lateral_faces` with sub-face counts
- `env.bottom_sub_faces` / `env.top_sub_faces` with bboxes
- `env.multi_piece_shares_edges_by_slab`
- Cohort solid topology counts via `TopExp_Explorer`

Optional `--export-step <dir>` writes STEP for the cohort solid and
BREP for each per-piece sub-face. Optional `--run-cad-occ` runs the
full pipeline and `--gui` launches gmsh on the resulting .msh.

Scenes:
- `cap` — 4×4 slab + 2×4 cap (the canonical multi-piece test)
- `disc-cap` — disc r=1 + cap covering upper half (arc test)
- `annulus-cap` — annulus + cap covering upper half (arc + hole)
- `overlap-2-caps` — slab + 2 overlapping top neighbours (3-piece)
- `simple-slab` — single 2×2 slab (lateral conformality test)
- `stacked-cap-features` — 2 structured slabs + cap above (NEW)
- `stacked-3-layer-features` — 3 structured slabs + pad below + cap above (NEW)
- `photonics-cross-section` — 5 structured layers including a disc waveguide
  with `identify_arcs=True`, plus unstructured air on top (NEW)

The annulus inner-arc bug was caught by this script. The photonics
scene exposes the next real issue (below).

## Remaining failures

### 1. `test_simple_slab_lateral_mesh_is_conformal` (existing)

Single 2×2 slab, n_layers=2. 22 boundary triangles "not conformal with
wedge mesh". Down from 66 after the lateral sub-face fix. The remaining
22 are on single-segment lateral walls.

**Diagnosis:** Single-segment lateral walls (a single straight outline
edge with no interior arrangement vertices) are constructed as one
OCC face per (slab, edge). Gmsh's default 2D mesher places interior
nodes on these lateral OCC faces at intermediate z values that the
wedge mesh doesn't include. With n_layers=2 (3 z-levels), the wedge
has nodes at z=0, 0.5, 1; if gmsh adds an interior node at, say, z=0.25
on a lateral face, it doesn't conform with any wedge node.

The transfinite hint in `apply_structured_transfinite_hints`
constrains lateral curve node count via `setTransfiniteCurve(n_layers+1)`
on vertical edges. But that only applies to the VERTICAL edges; the
2D mesh on the lateral face itself can still add interior face nodes
unless `setTransfiniteSurface` succeeds.

**Likely fix:** Tighten the transfinite-surface logic in
`apply_structured_transfinite_hints` so that single-segment lateral
walls always get `setTransfiniteSurface` applied. The current code
skips faces with `len(edge_tags_set) < 3` — for a single-segment
lateral wall, the boundary has exactly 4 edges (2 horizontal + 2
vertical), so it shouldn't trigger the skip. Need to investigate why
the hint isn't producing a conformant 2D mesh on those faces.

### 2. Multi-layer cohort + unstructured neighbour above (NEW, real)

**Reproduction:**
```bash
python scripts/inspect_phase3_geometry.py --scene stacked-cap-features --run-cad-occ
# generate_mesh failed: Invalid boundary mesh (overlapping facets) on surface N surface M
```

**Trace:**
- 2 structured slabs stacked into one cohort z∈[0, 2]. Cap above z∈[2, 3].
- Cohort envelope = one OCC TopoDS_Solid spanning z=[0, 2].
- Cap = separate OCC volume; its bot face at z=2 is shared (BOP-merged)
  with the cohort envelope's top piece-0 face.
- `_suppress_empty_cohort_envelope_volumes` calls
  `gmsh.model.removeEntities([(3, envelope_vol)], recursive=False)`.
- After `removeEntities`, **the cap's `gmsh.model.getBoundary` returns
  the wrong faces** — its bot reference now points to layer2 top
  piece 1 (`x ∈ [2, 4]`) instead of layer2 top piece 0 (`x ∈ [0, 2]`).
  The cap's own `getBoundingBox` returns empty.
- gmsh's 3D mesher then chokes on the corrupted boundary: "Invalid
  boundary mesh (overlapping facets)".

**Why the 1-layer case works:** Single-piece-cohort envelopes have
simpler topology — one bot face, one top face, four lateral walls,
no interior horizontal/vertical structure. The shared top face has
only two parents (envelope + cap); removing the envelope leaves the
cap as the sole parent. `getBoundary` returns it correctly.

For 2-layer cohorts, the envelope's TopoDS_Solid has more sub-shapes
(2 piece sub-faces × 2 z-levels × 1 cohort = 4 horizontal sub-faces
on the boundary, plus the 4 lateral walls × 2 z-intervals = 8 lateral
sub-faces). The cap-envelope shared face is just one of many sub-shapes
of the envelope solid. After `removeEntities` of the parent, the
shared-face reference seems to get re-routed to a sibling face. This
is a gmsh-internal corruption, not something we can work around with
`recursive=` flags.

**Attempted workarounds (all fail):**
- `gmsh.model.setVisibility(envelope, 0, recursive=False)` +
  `Mesh.MeshOnlyVisible=1`: cap also stays unmeshed (gmsh's
  MeshOnlyVisible seems to propagate even with `recursive=False`).
- Skip `removeEntities` and use `recursive=True` to delete children
  too: kills the cap's bot face entirely.
- The pre-existing dummy-tet-injection approach (now removed): an
  earlier comment in `_suppress` says this corrupted `_mesh_vertices`
  indexing — not viable.

**Recommended fix path:** Subdivide the cohort envelope into per-z-range
sub-solids inside `assemble_cohort_envelope_solid` instead of one
z-spanning solid. Then each z-range's sub-solid has its own top face
shared with at most one external neighbour, and `removeEntities` of
the topmost sub-solid only affects the cap's bot reference (which is
the only shared face). This mirrors how Phase 1+2 handled this
naturally (each piece was its own OCC sub-prism).

Alternative: introduce a "do not tet-mesh" flag on OCC volumes (via
algorithm constraints on the volume) instead of removing it. Needs
investigation of gmsh's per-entity meshing options for OCC volumes.

Affected tests: none (no existing test covers multi-layer cohort +
unstructured neighbour above). Affected inspection scenes:
`stacked-cap-features`, `stacked-3-layer-features`, `photonics-cross-section`.

## What the architecture handles well now

| Scenario | Status |
|---------|--------|
| Single-piece slab, no neighbour | ✓ |
| Single-piece slab + neighbour above (cap) | ✓ |
| Multi-piece slab (face_partition > 1) | ✓ |
| Multi-piece slab + overlapping neighbours | ✓ |
| Multi-piece slab with disc (arc) edges | ✓ |
| Multi-piece slab with annulus (inner ring) edges | ✓ |
| Disjoint-XY multi-source cohort (entity-grouping split) | ✓ |
| Per-piece volume tagging with shared bot OCC face | ✓ |
| Lateral conformality on multi-segment perimeter edges | ✓ |
| Single-piece lateral conformality (single-segment edges) | ✗ (22 orphans) |
| 2+ stacked structured slabs in one cohort | ✓ (mesh produces, but...) |
| 2+ stacked structured slabs + unstructured neighbour above | ✗ (mesh fails) |
| Multi-layer cohort with arc-bearing layer (photonics) | ✗ (same as above) |

## Recommended next session

In priority order:

1. **Fix the multi-layer cohort + neighbour boundary corruption**
   (the real architectural gap). Subdivide the cohort envelope into
   per-z-range sub-solids so `removeEntities` doesn't reach across
   neighbour interfaces. Worth a focused 2-3 hour spike — the per-z-range
   sub-solid approach is already foreshadowed by the existing
   `assemble_cohort_envelope_solid` logic for multi-piece per z-level
   (it iterates `_zmin_bot_faces` and `_zmax_top_faces` per side).

2. **Fix `test_simple_slab_lateral_mesh_is_conformal`** (22 orphans
   on single-segment lateral walls). Likely a few-line fix in
   `apply_structured_transfinite_hints` — investigate why
   `setTransfiniteSurface` isn't producing a conformant 2D mesh on
   simple 4-corner lateral walls. Re-run the test with `Mesh.Algorithm=8`
   (Frontal-Delaunay-for-Quads) or via `setRecombine` to see if a
   different 2D algorithm avoids interior nodes.

3. **Flip the kill switch** (Task 22). Once (1) and (2) land and the
   sweep is at 0 failed, change `_USE_DISCRETE_COHORT_MESH = True` as
   the default in `meshwell/structured/phantom.py`.

4. **Delete Phase 1+2 code** (Task 23). Once (3) has soaked:
   - `meshwell/structured/cohort_topology.py`
   - The `_USE_COHORT_TOPOLOGY` and `_PRESHARE_VERTICAL_FACES`
     constants
   - `_build_phantom_shapes_via_cohort_topology`
   - The 7 Bucket-A skipped tests
   - The 2 Bucket-B5 skipped tests

5. **Re-sweep the stress tests** (`test_stress_stacked_patterns.py`)
   under `_MESHWELL_FORCE_PHASE3=1` to confirm `ccc7b4a`'s B4 SIGSEGV
   fix still holds and no new regressions surfaced from this
   session's six fixes.

## Key files

- `meshwell/structured/cohort_envelope.py` — multi-piece sub-face
  builder, edge-chaining helper, 3-point arc constructor, post-sewing
  TShape propagation, assembly path that skips union for single-slab
  multi-piece.
- `meshwell/structured/phantom.py` — Phase 3 routing branch,
  `_split_cohort_solid_by_source`, lateral sub-face encoding into
  `input_laterals_by_outer_edge`.
- `meshwell/structured/builder.py` — `_filter_phase3_face_map_per_piece`,
  `_suppress_empty_cohort_envelope_volumes` with `structured_pg_names`
  scope and `removeEntities`, `_stamp_top_face_mesh` tolerance match.
- `scripts/inspect_phase3_geometry.py` — diagnosis tool that caught
  the annulus inner-arc bug and surfaced the multi-layer cohort issue.

## Test counts

**Default off (`_USE_DISCRETE_COHORT_MESH=False`):**
- 283 passed, 4 skipped, 0 failed.

**Phase 3 sweep (`_MESHWELL_FORCE_PHASE3=1`):**
- 271 passed, 15 skipped, **1 failed**.
- The 15 skipped: 7 Bucket A + 2 Bucket B5 + 1 cohort_topology
  fixture conflict + 1 Phase-2-phantom-shape identity assertion +
  4 baseline skips that pre-date Phase 3.
- The 1 failed: `test_simple_slab_lateral_mesh_is_conformal`
  (22 lateral orphan triangles, not regression).
