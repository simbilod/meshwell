# Structured meshing — what happens after `build_plan`

**Date:** 2026-05-30
**Companion to:** [2026-05-30-structured-polygon-preprocessing.md](2026-05-30-structured-polygon-preprocessing.md)

The previous doc covered what `build_plan(entities)` produces. This
doc covers what happens between that `StructuredPlan` and the final
.msh file on disk. All stages run inside
[meshwell/orchestrator.py::generate_mesh](meshwell/orchestrator.py).

Phase 3 (`_USE_DISCRETE_COHORT_MESH=True`) is documented as the
default path; differences from Phase 1+2 are noted inline.

## High-level flow

```
StructuredPlan  ─→  Phantom solids        ─→  cad_occ BOP        ─→  XAO → gmsh
       │                  │                          │                      │
       │                  │                          │                      ▼
       │                  │                          │              2D mesh + transfinite
       │                  │                          │                      │
       │                  │                          │                      ▼
       │                  │                          │              apply_structured_mesh
       │                  │                          │              (the magic happens here)
       │                  │                          │                      │
       │                  │                          │                      ▼
       │                  │                          │              3D mesh (only empty vols)
       │                  │                          │                      │
       │                  │                          │                      ▼
       │                  │                          │              clear dummy tets, write
       │                  │                          │
       └──────────────────┴──────────────────────────┘
                  fed forward as data through PhantomMap
```

## Stage 1 — phantom shape construction

[build_phantom_shapes](meshwell/structured/phantom.py#L1273) runs
right after `build_plan`. Under Phase 3 it routes through
[_build_phantom_shapes_via_cohort_envelope](meshwell/structured/phantom.py#L1199).

Per cohort:
1. [build_cohort_envelope](meshwell/structured/cohort_envelope.py#L142):
   builds the outline-only topology — vertex registry, horizontal edge
   wires per z-plane, per-piece bot/top sub-faces, lateral wall faces.
2. [assemble_cohort_envelope_solid](meshwell/structured/cohort_envelope.py#L744):
   sews the bot/top/lateral faces into a single `TopoDS_Solid`.
3. Build one `PhantomShape` per cohort with synthetic
   `slab_index = -(cohort_idx + 1)`, packaging:
   - `solid` — the assembled cohort envelope
   - `input_faces_by_key` — `FaceKey(slab, side, piece) → TopoDS_Face`
   - `input_laterals_by_outer_edge` — synthetic-key →
     `TopoDS_Face` (one entry per outline-edge sub-segment)

The PhantomShape becomes the BOP input on behalf of every structured
slab in the cohort.

## Stage 2 — entity grouping for cad_occ

[_group_phantom_solids_by_entity](meshwell/structured/phantom.py#L982)
maps each cohort's envelope solid back onto the input entities'
`source_index` via:
- For disjoint XY cohorts (multi-shell solid), split the cohort
  envelope into per-shell single-shell solids and assign each shell
  to its source by XY bbox match
  ([_split_cohort_solid_by_source](meshwell/structured/phantom.py#L1017)).
- For connected cohorts, assign the envelope to the lowest non-carved
  source; all other non-carved sources get an empty `[]` entry, which
  tells `cad_occ` to skip them entirely (their geometry is already in
  the envelope).
- Fully-carved sources get `[]` too — `cad_occ` doesn't reinstantiate
  them.

The output is `overrides: dict[source_index, list[TopoDS_Solid]]` that
overrides what `cad_occ` would otherwise build from each entity.

## Stage 3 — cad_occ.fragment_all

[cad_occ.fragment_all](meshwell/cad_occ.py) runs `BOPAlgo_GeneralFuse`
(or the equivalent cad_occ wrapper) on all entity solids:
- Structured entities: their `overrides` are used as input (i.e., the
  cohort envelopes, not the original PolyPrism solids).
- Unstructured entities: their original solids are used.

After BOP:
- Coincident faces are merged into one shared TopoDS_Face with
  multiple parent volumes.
- Volumes overlapping in 3D are split into sub-volumes.
- The history (which input face became which output face/faces) is
  preserved by `BOPAlgo_Builder.Modified(face)`.

The output is written to a temporary XAO file
(BREP + XML for tags/groups).

## Stage 4 — PhantomMap extraction

[extract_phantom_map](meshwell/structured/phantom.py#L1462) walks the
post-BOP history (still in OCC, before gmsh loads the XAO):

For each PhantomShape's input face/edge/lateral, asks
`builder.Modified(in_shape)` to recover the post-BOP output shape(s).
The result is a `PhantomMap`:
- `output_faces: dict[FaceKey, list[TopoDS_Face]]`
- `output_laterals: dict[LateralKey, list[TopoDS_Face]]`
- `output_edges: dict[EdgeKey, list[TopoDS_Edge]]`
- `lateral_has_midheight_cut: dict[LateralKey, bool]`

These are pure OCC references — gmsh hasn't seen them yet.

## Stage 5 — gmsh loads the XAO

The orchestrator calls `gmsh.open(<xao_path>)`. gmsh's OCC kernel
reads the BREP compound + XAO XML metadata, and assigns gmsh tags to
every shape in `TopExp::MapShapes` order. Each `TopoDS_Face` in
`phantom_map.output_faces` now corresponds to a specific gmsh face
tag.

[_map_phantom_faces_to_gmsh](meshwell/structured/builder.py#L1747)
does this mapping: build a `TopTools_IndexedMapOfShape` of all faces
in the loaded XAO compound, look up each `phantom_map.output_faces`
entry via `FindIndex`. The result is
`face_map: dict[FaceKey, list[gmsh_face_tag]]`.

Similarly for edges (`_map_phantom_edges_to_gmsh`) and laterals
(`_map_phantom_laterals_to_gmsh`).

## Stage 6 — transfinite hints (pre_2d_hook)

[apply_structured_transfinite_hints](meshwell/structured/builder.py#L1633)
runs as gmsh's `pre_2d_hook`, before `generate(2)`:

For each `LateralKey` in `phantom_map.output_laterals`:
- Get the gmsh face tag and find its 1D boundary edges.
- Each VERTICAL edge (endpoints differ in z): apply
  `setTransfiniteCurve(edge, n_layers+1)` so the lateral wall has
  exactly the right number of z-subdivisions.
- Apply `setTransfiniteSurface(face)` so the 2D mesh is a structured
  quad grid (no interior face nodes).
- `setRecombine` if `recombine_lateral_faces=True`.

Faces that gmsh would reject the transfinite hint on (mid-height cut
detected via `lateral_has_midheight_cut`, multi-wire faces, fewer than
3 boundary edges) are silently skipped — they fall back to the
default 2D mesher.

This is what makes the structured slab's lateral walls produce quads
that match the wedge volume's lateral faces.

## Stage 7 — gmsh.model.mesh.generate(2)

gmsh meshes every OCC 2D face:
- Lateral OCC faces with transfinite hints: structured quads.
- Bot/top OCC sub-faces: 2D triangulation (gmsh's default mesher).
- 1D OCC edges shared between faces: meshed once; node tags shared.

At this point the model has 2D meshes on every OCC face but no 3D
elements anywhere.

## Stage 8 — apply_structured_mesh (pre_3d_hook)

[apply_structured_mesh](meshwell/structured/builder.py#L1015) is the
heart of Phase 3. It runs after `generate(2)` and before `generate(3)`.

### 8a. Filter face_map per piece

[_filter_phase3_face_map_per_piece](meshwell/structured/builder.py#L1773)
filters out BOP fragments that don't belong to a piece. For multi-piece
zmin/zmax cohorts where all per-piece FaceKeys route through one
union face for BOP, the same fragment list ends up on every per-piece
FaceKey. The filter picks the fragment whose 2D bbox contains the
piece polygon's representative point.

### 8b. Pre-create discrete 2D entities for interior interfaces

For horizontal interior interfaces (between stacked slabs at a shared
z-plane), `_create_discrete_interior_face` creates a discrete 2D
gmsh entity ahead of the per-piece loop. Each piece's `bot_key` and
`top_key` referencing this z-plane is mapped to the discrete entity
in `interior_discrete`.

### 8c. Per-piece loop

For each `(slab_idx, piece_idx)`:

1. Look up `bot_tag` and `top_tag` via `face_map`. If the FaceKey
   points to an interior interface, use `interior_discrete[key]`.
2. Build `edge_correspondence: dict[bot_edge → top_edge]` from
   `edge_map` — for Phase 5(a) boundary node matching.
3. [_stamp_top_face_mesh](meshwell/structured/builder.py#L85):
   - Read the bot face's triangulation.
   - Clear the top face's existing mesh.
   - For each bot node:
     - Boundary node (on a 1D edge): match by tolerance nearest-XY to
       a top boundary node and reuse the existing top tag.
     - Interior node: allocate a new top tag at (bot_x, bot_y, zhi).
   - Stamp the bot triangulation onto the top face using
     `bot_to_top_tag`.
   - Returns `bot_to_top_tag` for the volume builder.
4. For multi-layer slabs (n_layers > 1), allocate intermediate-layer
   node maps between bot and top.
5. [_build_slab_volume](meshwell/structured/builder.py#L321):
   - Create (or reuse) a discrete 3D gmsh entity.
   - For each bot triangle, emit a wedge element using
     `bot_to_top_tag` to look up the corresponding top vertex per
     layer.
   - Recombine into hexes if `recombine=True`.
6. Register a physical group on the discrete 3D entity with the
   slab's `physical_name`.

### 8d. Shared-face cache

When two same-z slabs share `(bot_tag, top_tag)` (e.g. laterally-
adjacent slabs in one cohort that share a union top), the second
slab reuses the first's cached `(top_map, layer_maps, vol_tag)` and
just registers a new physical group pointing at the same volume.

### 8e. Stamp interior interfaces

After the per-piece loop:
- [_stamp_phase3_interior_interfaces](meshwell/structured/builder.py) —
  horizontal interfaces (between stacked z-slabs) already created in
  8b; vertical interfaces (between same-z laterally-adjacent pieces)
  stamped now if `n_layers ≥ 1`.

### 8f. Suppress empty cohort envelope volumes

[_suppress_empty_cohort_envelope_volumes](meshwell/structured/builder.py#L842):
- Find every 3D volume that still has zero elements (i.e., the
  cohort envelope's OCC solid — wedges live in separate discrete
  entities).
- Purge those volumes from every 3D physical group that's a
  structured-slab group.
- Call `gmsh.model.removeEntities([(3, vt)], recursive=False)` to
  delete the empty 3D entity outright. The 2D face children stay
  (still shared with unstructured neighbours).

Return the list of suppressed volume tags so the post_3d_hook can
clear any dummy tets (currently unused — `removeEntities` makes the
post-hook a no-op).

### 8g. Global node dedup

`gmsh.model.mesh.removeDuplicateNodes()` merges ~coincident nodes
across discrete and OCC entities. Important when interior interface
nodes (placed by piece stamping) coincide with OCC face boundary
nodes.

## Stage 9 — gmsh.model.mesh.generate(3)

With `Mesh.MeshOnlyEmpty=1`, gmsh tetrahedralizes every volume that
still has zero 3D elements:
- Unstructured entities (caps, claddings, voids) — get tet meshes.
- Cohort envelope volumes — were removed in Stage 8f, so gmsh
  doesn't see them.

## Stage 10 — gmsh.model.mesh.removeDuplicateNodes (again) + write

Final dedup pass to clean up coincident nodes between tet and wedge
regions. Then `gmsh.write(<output.msh>)`.

## Output guarantees

The resulting .msh contains:
- One physical group per `slab.physical_name` for structured slabs,
  with discrete wedge/hex elements.
- One physical group per unstructured entity, with tet elements.
- Internal interface physical groups (named via `"___"` separator,
  e.g. `slab___cap`) for shared boundary faces.
- Lateral OCC face boundary triangles for non-shared surfaces.

Conformality:
- Wedge bot/top triangles exactly match the OCC bot/top face mesh
  (same node tags) by construction (Stage 8c.3).
- Wedge lateral quads match the OCC lateral face quads when the
  lateral got a transfinite hint (Stage 6).
- Interior interfaces (between pieces, between stacked slabs) are
  stamped as discrete 2D entities sharing nodes with both sides
  (Stage 8b, 8e).

## Where the remaining issues live

[2026-05-30-phase3-followup.md](docs/superpowers/followup/2026-05-30-phase3-followup.md):

- **Single-segment lateral conformality** — Stage 6's transfinite
  surface hint isn't catching one specific shape, producing 22
  non-conformal triangles on a single-piece slab.
- **Multi-layer cohort + unstructured neighbour above** — Stage 8f's
  `removeEntities` corrupts the neighbour's `getBoundary` references
  when the envelope spans 2+ z-intervals. The fix likely involves
  subdividing the envelope into per-z-range sub-solids in Stage 1
  (`assemble_cohort_envelope_solid`).

## Differences in Phase 1+2 (legacy)

Phase 1+2 doesn't build a cohort envelope. Each face_partition piece
becomes its own OCC sub-prism in Stage 1; Stages 2-7 run identically;
Stage 8 is much simpler (each piece has a 1-to-1 OCC↔gmsh face
correspondence, no shared bot/top across pieces, no interior
interface stamping). Stage 8f doesn't exist — there's nothing to
suppress because each piece IS its own OCC volume.

The Phase 3 path lets the cohort envelope BOP run with one input per
cohort instead of one per piece. For complex scenes (concentric
arcs, multi-piece-per-slab stacks), this is dramatically more robust
in OCC than the Phase 1+2 path — fewer BOP arguments, no internal
sewing/PCurves issues, simpler topology — at the cost of needing
Stage 8's per-piece machinery at mesh time.
