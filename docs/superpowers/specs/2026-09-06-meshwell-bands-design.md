# Structured bands (2D) — design

Status: approved design, pre-implementation.
Origin: `laser/plans/02a-meshwell-structured-bands.md` (phase 1 consumer:
laser plan 02 transport meshes). This document supersedes that plan's API
sketch; required plan-text updates are listed at the end.

## Problem

Finite-volume transport (Voronoi/Scharfetter–Gummel) needs meshes with
nonnegative Voronoi edge weights at ~100:1 anisotropy near junctions.
Tensor-product cells satisfy this by construction; unstructured
triangulation does not. meshwell needs a way to lay a structured,
anisotropically graded band of cells along an interface, conformal with
unstructured triangles everywhere else.

The band is **pure mesh structure**: it owns no material and defines no
new region. It subdivides existing regions, as if imprinted with the
highest mesh priority after all structural CAD is done.

## API

Two objects, split exactly along the CAD/mesh stage boundary (the same
split the 3D structured pipeline uses: `structured=True` is a CAD-side
flag; `n_layers` is a mesh-side spec).

### `Band` — CAD input, geometry only

```python
Band(
    name="qw_band",                 # pairing key for the mesh-side spec
    on="sch_top___well_1",          # a dim-(N-1) physical name
    thickness={"well_1": 0.02,      # per-side growth distance
               "sch_top": 0.006},
)
```

- `on` accepts any dim-1 physical name: an interface (`a___b`), a domain
  boundary (`a___None`), or an embedded `PolyLine`'s physical name.
- `thickness` keys select the growth side(s), and the valid key set
  depends on the attachment:

  | attachment | valid keys | rationale |
  |---|---|---|
  | interface `a___b` | `"a"`, `"b"` | region names are orientation-robust (BOP scrambles curve orientation) |
  | boundary | the inside region name | only one side exists |
  | embedded PolyLine | `"left"`, `"right"` | orientation = user-supplied coordinate order; matches the material-left-of-travel convention |

  One key = one-sided band; two keys = two-sided. Mixed or invalid keys
  are a hard error naming the admissible key set.
- `Band` is not an entity: it produces no physical group of its own, so
  it is passed via a separate `bands=[...]` argument, preserving the
  "everything in `entities` yields a physical group" invariant. It
  serializes via `to_dict`/`from_dict` like entities do.

### `BandDiscretizationSpec` — mesh input, discretization only

```python
resolution_specs={
    "qw_band": [BandDiscretizationSpec(
        tangential=0.05,            # float (uniform spacing) | explicit array (arclength from curve start)
        normal={"well_1": Graded(h0=1e-3, ratio=1.3),
                "sch_top": np.array([0.0, 1e-3, 2.5e-3, 6e-3])},
        element_type="triangle",    # | "quad"
    )],
}
```

- Keyed by `Band.name` in the existing `resolution_specs` dict (string
  keys survive serialization; object identity would not).
- `normal` keys follow the same rule as `Band.thickness` keys.
- `Graded(h0, ratio)` carries **no thickness** — it fills whatever
  thickness the CAD declared (last cell adjusted), so the one field both
  stages care about is stated once. Explicit arrays restate the extent
  and are cross-validated against the imprinted geometry (hard error on
  mismatch). Uniform spacing = `ratio=1`.
- `element_type="quad"` skips the diagonal split when stamping. The
  diagonal of a right-angle band cell carries exactly zero Voronoi
  weight, so triangle and quad output are numerically identical for FV;
  quads exist for other consumers. Hexahedra in 3D are out of scope; the
  field name is chosen so `"hexahedron"` can slot in later.

## CAD stage: clip, then imprint

Runs after the structural `cad_occ` fragment, before XAO write (so
interface naming, which happens at write time, sees final topology).

1. **Clip (shapely, before touching OCC).** Reconstruct the nominal band
   footprint (attachment curve × thickness) and the final region
   polygon(s) (from entity polygons + mesh_order precedence, via the
   same polygonize approach as the 3D `Arrangement`). Keep the band only
   over tangential intervals where the **full** normal extent lies
   inside the target region. Dropped areas (corners, terminations,
   collisions with other interfaces) are left to unstructured fill.
   Clip ends land wherever geometry puts them — they become explicit
   stamped nodes, NOT snapped to the tangential grid, so imprint
   geometry depends only on `Band` fields, never on the discretization.
   Overlapping band footprints: hard error (phase 1).
2. **Imprint (OCC).** A second fragment pass with the clipped band
   rectangles as tools — "highest mesh order, last". Band sub-faces
   inherit the underlying region's physical name via the fragment
   `Modified()` maps (a band owns no material; it only subdivides).
3. **Synthetic groups.** Each band sub-face and seam curve gets a
   synthetic physical group written INTO the XAO —
   `__band_<name>__<side>__<i>` — following the `__cohort_*` precedent.
   Band seam curves get no `a___b` interface names (both sides share a
   physical name). Synthetics are stripped before `.msh` write.

## Mesh stage: discovery-based stamping

The XAO is the entire contract between stages. The mesh step never
receives in-memory band state; it discovers `__band_*` groups after XAO
load and pairs them with `resolution_specs` entries by band name. This
works identically for `generate_mesh` and for separate
CAD-then-`mesh(input_file=...)` steps — one implementation. (The 3D
wedge pipeline threads in-memory `StructuredState` and therefore only
works through the orchestrator; bands deliberately do not copy that.
Migrating wedges to discovery is a separate, later refactor.)

Inside a single composed `pre_2d` hook:

1. Recover band face/curve tags from synthetic group names.
2. Stamp seam nodes on every band boundary curve (`addNodes` with
   explicit coordinates — transfinite curves only support uniform or
   progression spacing, not arbitrary arrays). Shared curves between
   stacked bands are stamped once; disagreement between two bands'
   node sets on a shared seam is a hard error.
3. **Split edges**: BOP legitimately splits band edges (a ridge splits
   the top edge of the layer below it). The stamper selects the
   coordinate subset inside each curve fragment and requires fragment
   endpoints to be members of the tangential coordinates (within
   `point_tolerance`) — error otherwise, naming the missing coordinate.
   Note the deliberate asymmetry with clip ends: a BOP split marks
   user-placed geometry the grid must align with (its nodes are shared
   with neighbor regions), so silent node insertion is wrong there; a
   clip end is a band-internal termination the user never placed, so it
   is auto-inserted. Tangential grids from a `float` spacing are
   generated from the attachment curve's start (then clipped), keeping
   stacked bands aligned regardless of per-band clip extents.
4. `generate(1)` inside the hook (as `freeze_lateral_mesh` does), then
   stamp interior nodes and elements (right-triangle pairs or quads).
5. `Mesh.MeshOnlyEmpty=1`; the outer `generate(2)` fills only
   unstructured surfaces, conforming to the frozen seam nodes. All-band
   scenes (laser transport meshes) leave nothing to fill.

Pairing errors are hard errors in both directions: a `__band_*` group
with no spec, or a band spec with no matching group.

## Validation & errors

New exceptions in the `structured/exceptions.py` style:

- invalid `thickness`/`normal` keys for the attachment type
- explicit normal array extent ≠ imprinted thickness
- seam node-set mismatch between adjacent bands
- split-edge endpoint not in tangential coordinates
- band footprint overlap
- unpaired band group / unpaired band spec
- curved attachment curve (phase 3) → clear not-yet-supported error

Spec-side field validation (monotonicity, positivity, `h0 > 0`,
`ratio >= 1`) in the pydantic model. The plan-02a admissibility
validator (opposite-angle-sum check) remains phase 2; phase-1 output is
admissible by construction and the laser keeps its Julia-side gate.

## Tests

`tests/test_band.py`:

- stack of bands + lateral unstructured cladding: exact tensor-product
  node coordinates, all right triangles, `a___b` interface groups
  survive to `.msh`
- ridge geometry: split-edge stamping across fragments
- quad variant
- two-sided band on an embedded PolyLine (left/right grading differs)
- clip: band termination at a corner falls back to unstructured fill
- separate steps: `cad()` → `.xao` → standalone `mesh()` reproduces the
  `generate_mesh` result bit-for-bit
- each error path

Laser integration gate (lives in laser repo, plan 02): `transport_b`/
`transport_c` generate, load into ExtendableGrids with correct
`cellregions`/`bfaceregions`, pass the admissibility invariant exactly.

## Future (design hooks only, no implementation)

- **3D**: attachment generalizes to dim-2 physicals (interface
  surfaces); `Graded` normal spec is dimension-invariant; flat-surface
  bands ≈ existing structured prisms with grading retrofitted onto
  `StructuredExtrusionResolutionSpec`.
- **Phase 3 (curved bands)**: data-derived curves enter as embedded
  `PolyLine` entities; `Band.on` already points at them. Marching,
  arclength redistribution, and fan terminations extend the clip/imprint
  pass; the API does not change.
- **Wedge harmonization**: migrate the 3D pipeline to the
  discovery-based mesh step so separate-steps works there too.

## Plan 02a text updates required

- "Explicit coordinate arrays in the spec — no growth-ratio DSL":
  revised; `Graded(h0, ratio)` is in (it eliminates thickness
  redundancy between stages), explicit arrays remain supported.
- "One new mesh entity: band": revised; the band is not an entity but a
  CAD-side declaration attached to an existing dim-(N-1) physical name,
  plus a mesh-side discretization spec.
- "Output is always triangles": revised; `element_type="quad"` is a
  supported option (2D only), triangles remain the default and the
  laser consumer.
