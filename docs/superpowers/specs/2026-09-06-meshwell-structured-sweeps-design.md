# Structured sweeps — design (unifies 2D bands with 3D structured extrusion)

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

A band is **pure mesh structure**: it owns no material and defines no
new region. It subdivides existing regions, as if imprinted with the
highest mesh priority after all structural CAD is done.

## One concept: the structured sweep

meshwell already has structured meshing: `PolyPrism(structured=True)`
sweeps a 2D footprint along z into wedges. A band is the same operation
one dimension down: sweep a 1D source along its normal into
triangle-pairs/quads. This design introduces no second concept — it
names the existing one (**structured sweep**) and extends it with a new
declaration form and a 2D stamping kernel.

Shared discipline, both dimensions: build conformal structure ahead of
gmsh, freeze the shared boundaries (seams), stamp elements by hand,
`Mesh.MeshOnlyEmpty=1` so gmsh fills only the unstructured rest.

The only irreducible difference is the per-dimension stamping kernel —
different arithmetic, not different design:

| | 3D kernel (shipped, `structured/wedge.py`) | 2D kernel (new, `structured/sweep2d.py`) |
|---|---|---|
| source | prism footprint (2D face) | interface/boundary/PolyLine (1D curve) |
| in-source discretization | unstructured triangulation | tangential coordinate array |
| sweep discretization | layers | layers (graded) |
| frozen seams | lateral-face quads | seam-curve nodes |
| stamped elements | wedges | triangle pairs / quads |

Everything else — declaration, spec, cohort grouping, synthetic-group
bookkeeping, validation — is one implementation with shared names.

## API

One declaration class (CAD input, geometry only) and one spec class
(mesh input, discretization only), split along the CAD/mesh stage
boundary.

### `StructuredSweep` — CAD input

```python
StructuredSweep(
    name="qw_sweep",                # pairing key for the mesh-side spec
    on="sch_top___well_1",          # a dim-(N-1) physical name
    thickness={"well_1": 0.02,      # per-side growth distance
               "sch_top": 0.006},
)
```

- `on` accepts any dim-1 physical name (2D): an interface (`a___b`), a
  domain boundary (`a___None`), or an embedded `PolyLine`'s physical
  name. In 3D (phase 4) the same class takes a dim-2 physical name; the
  API does not change.
- `thickness` keys select the growth side(s); the valid key set depends
  on the attachment:

  | attachment | valid keys | rationale |
  |---|---|---|
  | interface `a___b` | `"a"`, `"b"` | region names are orientation-robust (BOP scrambles curve orientation) |
  | boundary | the inside region name | only one side exists |
  | embedded PolyLine | `"left"`, `"right"` | orientation = user-supplied coordinate order; matches the material-left-of-travel convention |

  One key = one-sided; two keys = two-sided. Mixed or invalid keys are
  a hard error naming the admissible key set.
- Passed via a separate `sweeps=[...]` argument to `generate_mesh` /
  the CAD step — it is not an entity (produces no physical group of its
  own), preserving the "everything in `entities` yields a physical
  group" invariant. Serializes via `to_dict`/`from_dict`.
- **`structured=True` on PolyPrism becomes documented sugar** for
  self-attachment: "sweep this entity along its own extrusion axis,
  thickness = its height." The flag stays (shipped API); the docs
  explain it as the second spelling of the same declaration.

### `StructuredSweepResolutionSpec` — mesh input

```python
resolution_specs={
    "qw_sweep": [StructuredSweepResolutionSpec(
        tangential=0.05,            # float (uniform spacing) | explicit array (arclength from curve start); None in 3D
        normal={"well_1": Graded(h0=1e-3, ratio=1.3),
                "sch_top": np.array([0.0, 1e-3, 2.5e-3, 6e-3])},
        element_type="triangle",    # | "quad"
    )],
}
```

- `normal` per side accepts `int` (n uniform layers) | `Graded(h0,
  ratio)` | explicit array. **This subsumes the existing 3D spec**:
  `StructuredExtrusionResolutionSpec(n_layers=3)` becomes a thin
  deprecated alias for `StructuredSweepResolutionSpec(normal=3)`, and
  the 3D wedge kernel gains graded layers for free once it reads the
  unified field.
- `Graded(h0, ratio)` and `int` carry **no thickness** — they fill
  whatever thickness the CAD declared (last cell adjusted), so the one
  field both stages care about is stated once. Explicit arrays restate
  the extent and are cross-validated against the imprinted geometry
  (hard error on mismatch).
- Keyed by `StructuredSweep.name` in the existing `resolution_specs`
  dict (string keys survive serialization; object identity would not).
  Owns the no-op `apply()` (structured specs are consumed by stamping
  kernels, not gmsh size fields) — that exception exists in one class.
- `normal` keys follow the same rule as `thickness` keys.
- `element_type="quad"` skips the diagonal split when stamping. The
  diagonal of a right-angle sweep cell carries exactly zero Voronoi
  weight, so triangle and quad output are numerically identical for FV;
  quads exist for other consumers. Hexahedra in 3D are out of scope;
  the field name admits `"hexahedron"` later.

## CAD stage: clip, then imprint

Runs after the structural `cad_occ` fragment, before XAO write (so
interface naming, which happens at write time, sees final topology).

1. **Clip (shapely, before touching OCC).** Reconstruct the nominal
   sweep footprint (attachment curve × thickness) and the final region
   polygon(s) (from entity polygons + mesh_order precedence, via the
   same polygonize approach as the 3D `Arrangement`). Keep the sweep
   only over tangential intervals where the **full** normal extent lies
   inside the target region. Dropped areas (corners, terminations,
   collisions with other interfaces) are left to unstructured fill.
   Clip ends land wherever geometry puts them — they become explicit
   stamped nodes, NOT snapped to the tangential grid, so imprint
   geometry depends only on `StructuredSweep` fields, never on the
   discretization. Overlapping sweep footprints: hard error (phase 1).
2. **Imprint (OCC).** A second fragment pass with the clipped sweep
   rectangles as tools — "highest mesh order, last". Sweep sub-faces
   inherit the underlying region's physical name via the fragment
   `Modified()` maps (a sweep owns no material; it only subdivides).
3. **Synthetic groups.** Each sweep sub-face and seam curve gets a
   synthetic physical group written INTO the XAO —
   `__sweep_<name>__<side>__<i>` — following the `__cohort_*`
   precedent. Seam curves get no `a___b` interface names (both sides
   share a physical name). Synthetics are stripped before `.msh` write.

Note: the shipped 3D path instead pre-builds cohort compounds BEFORE
BOP (performance-motivated) and threads in-memory state. That is
historical, not conceptual; it converges on this imprint+discovery
pattern in a later refactor (see Future).

## Mesh stage: discovery-based stamping

The XAO is the entire contract between stages. The mesh step never
receives in-memory sweep state; it discovers `__sweep_*` groups after
XAO load and pairs them with `resolution_specs` entries by sweep name.
This works identically for `generate_mesh` and for separate
CAD-then-`mesh(input_file=...)` steps — one implementation. (The 3D
wedge kernel currently only works through the orchestrator because of
its state-threading; the 2D kernel deliberately does not copy that.)

Inside a single composed `pre_2d` hook:

1. Recover sweep face/curve tags from synthetic group names.
2. Stamp seam nodes on every sweep boundary curve (`addNodes` with
   explicit coordinates — transfinite curves only support uniform or
   progression spacing, not arbitrary arrays). Shared curves between
   stacked sweeps (a cohort, same union-find grouping as 3D) are
   stamped once; disagreement between two sweeps' node sets on a shared
   seam is a hard error.
3. **Split edges**: BOP legitimately splits sweep edges (a ridge splits
   the top edge of the layer below it). The stamper selects the
   coordinate subset inside each curve fragment and requires fragment
   endpoints to be members of the tangential coordinates (within
   `point_tolerance`) — error otherwise, naming the missing coordinate.
   Deliberate asymmetry with clip ends: a BOP split marks user-placed
   geometry the grid must align with (its nodes are shared with
   neighbor regions), so silent node insertion is wrong there; a clip
   end is a sweep-internal termination the user never placed, so it is
   auto-inserted. Tangential grids from a `float` spacing are generated
   from the attachment curve's start (then clipped), keeping stacked
   sweeps aligned regardless of per-sweep clip extents.
4. `generate(1)` inside the hook (as the 3D kernel's
   `freeze_lateral_mesh` does), then stamp interior nodes and elements
   (right-triangle pairs or quads).
5. `Mesh.MeshOnlyEmpty=1`; the outer `generate(2)` fills only
   unstructured surfaces, conforming to the frozen seam nodes. All-sweep
   scenes (laser transport meshes) leave nothing to fill.

Pairing errors are hard errors in both directions: a `__sweep_*` group
with no spec, or a sweep spec with no matching group.

## Validation & errors

New exceptions in `structured/exceptions.py`:

- invalid `thickness`/`normal` keys for the attachment type
- explicit normal array extent ≠ imprinted thickness
- seam node-set mismatch within a cohort
- split-edge endpoint not in tangential coordinates
- sweep footprint overlap
- unpaired sweep group / unpaired sweep spec
- curved attachment curve (phase 3) → clear not-yet-supported error

Spec-side field validation (monotonicity, positivity, `h0 > 0`,
`ratio >= 1`) in the pydantic model. The plan-02a admissibility
validator (opposite-angle-sum check) remains phase 2; phase-1 output is
admissible by construction and the laser keeps its Julia-side gate.

## Tests

`tests/test_structured_sweep.py`:

- stack of sweeps + lateral unstructured cladding: exact tensor-product
  node coordinates, all right triangles, `a___b` interface groups
  survive to `.msh`
- ridge geometry: split-edge stamping across fragments
- quad variant
- two-sided sweep on an embedded PolyLine (left/right grading differs)
- clip: termination at a corner falls back to unstructured fill
- separate steps: `cad()` → `.xao` → standalone `mesh()` reproduces the
  `generate_mesh` result bit-for-bit
- alias: `StructuredExtrusionResolutionSpec(n_layers=3)` still drives
  the 3D kernel unchanged
- each error path

Laser integration gate (lives in laser repo, plan 02): `transport_b`/
`transport_c` generate, load into ExtendableGrids with correct
`cellregions`/`bfaceregions`, pass the admissibility invariant exactly.

## Future (design hooks only, no implementation)

- **Phase 3 (curved sources)**: data-derived curves enter as embedded
  `PolyLine` entities; `StructuredSweep.on` already points at them.
  Marching, arclength redistribution, and fan terminations extend the
  clip/imprint pass; the API does not change.
- **Phase 4 (3D sweeps from surfaces)**: `on=<dim-2 physical>`, same
  classes; the 3D kernel already exists for the flat case.
- **Wedge-kernel convergence**: migrate the 3D pipeline off pre-BOP
  compounds and state-threading onto the imprint + discovery pattern,
  making separate-steps work there too; read `normal` from the unified
  spec to gain graded z-layers.

## Plan 02a text updates required

- "Explicit coordinate arrays in the spec — no growth-ratio DSL":
  revised; `Graded(h0, ratio)` is in (it eliminates thickness
  redundancy between stages), explicit arrays remain supported.
- "One new mesh entity: band": revised; a band is a `StructuredSweep` —
  a CAD-side declaration attached to an existing dim-(N-1) physical
  name plus a mesh-side spec, unifying with (not paralleling) the
  existing 3D structured machinery.
- "Output is always triangles": revised; `element_type="quad"` is a
  supported option (2D only), triangles remain the default and the
  laser consumer.
