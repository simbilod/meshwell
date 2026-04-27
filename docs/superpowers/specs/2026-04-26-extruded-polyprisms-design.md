# Structured Extruded PolyPrisms — Design Spec

Date: 2026-04-26
Status: Draft (pending user review)

## Goal

Add a `StructuredPolyPrism` entity to meshwell that produces a **swept,
layered mesh** along z (gmsh-tutorial-t3 style: prismatic / wedge elements
stacked in user-controlled layer counts), while co-existing cleanly with
the existing unstructured CAD pipeline used by `PolyPrism`, `PolySurface`,
etc.

The structured layered mesh comes from `gmsh.model.geo.extrude(..., Layers, Heights, recombine)`,
which natively supports arbitrary polygonal bases — unlike transfinite
meshing, which requires four-corner topology. The new entity is therefore
a **meshing-strategy entity**, not a new CAD primitive.

## Scope (explicitly v1)

- **In:** Straight z-translation with user-supplied per-interval layer counts
  (zero-buffer `PolyPrism` analog).
- **In:** Multi-z-breakpoint stacks with per-interval layer counts, mirroring
  t3.py `[8, 2]` / `[0.5, 1]` semantics.
- **In:** Auto-resolution of overlapping structured slabs via shapely
  cascade (mesh_order priority).
- **In:** Both `cad_occ` and `cad_gmsh` backends.
- **Out:** Tapered (non-zero buffer) structured prisms.
- **Out:** Revolve / twist (gmsh tutorial t3 also demonstrates these — deferred).
- **Out:** Auto-mating of layer counts across touching slabs. Mismatch is
  a user error and is reported as such.
- **Out:** Hex-only output guarantees. We produce wedge/prismatic by
  default; `recombine=True` produces hex elements only where the base
  2D mesh contains quads (i.e. usually nothing for arbitrary polygons —
  documented as such).

## Public API

New entity class in `meshwell/structured_polyprism.py`:

```python
class StructuredPolyPrism(GeometryEntity):
    def __init__(
        self,
        polygons: Polygon | list[Polygon] | MultiPolygon | list[MultiPolygon],
        buffers: dict[float, float],          # all values must be 0
        n_layers: list[int],                  # len == len(buffers) - 1
        physical_name: str | tuple[str, ...] | None = None,
        mesh_order: float | None = None,
        mesh_bool: bool = True,
        recombine: bool = False,
        point_tolerance: float = 1e-3,
        identify_arcs: bool = False,
        min_arc_points: int = 4,
        arc_tolerance: float = 1e-3,
        translation: tuple[float, float, float] | None = None,
        rotation_axis: tuple[float, float, float] | None = None,
        rotation_point: tuple[float, float, float] | None = None,
        rotation_angle: float = 0.0,
    ): ...
```

**Validation at construction:**
- All `buffers.values()` must be exactly `0` (no taper). Raises `ValueError`.
- `len(n_layers) == len(buffers) - 1`. Raises `ValueError`.
- All `n_layers` entries are `int >= 1`.
- Each interval height must be > 0 (strictly increasing z keys).

**Why a separate class** rather than overloading `PolyPrism`:
- Behavior diverges meaningfully (CAD phantom + geo-kernel mesh path).
- Validation rules differ (zero-buffer required).
- Makes the slab-resolution pre-pass (see below) easy to discover with
  `isinstance` filtering.
- Sets up future `RevolvedPolyPrism` / `TwistedPolyPrism` cleanly.

## Architecture overview

The lifecycle of a `StructuredPolyPrism` is split between the CAD stage
(where it acts as a phantom that fragments neighbors and is then removed)
and the mesh stage (where the layered swept mesh is built in the gmsh geo
kernel inside the void left behind).

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Pre-CAD shapely pass                      │
│ - Existing perturbation buffer (unchanged)                          │
│ - NEW: structured-slab cascade → list[Slab]                         │
│   Each Slab = (footprint_polygon, [zlo, zhi], n_layers, recombine,  │
│                physical_name, source_index)                         │
│   Cascade ensures all Slabs are pairwise 3D-disjoint.               │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                              CAD stage                              │
│ - Each Slab → standard OCC extrude (BRepPrimAPI_MakePrism /         │
│   gmsh.model.occ.extrude). Just a regular 3D body.                  │
│ - Slab body participates in normal fragmentation.                   │
│ - Slab's mesh_order is honored among non-structured neighbors;      │
│   among structured slabs, conflicts already resolved by cascade.    │
│ - Slab body is marked keep=False at top-dim → removed after         │
│   fragmentation (existing keep=False machinery).                    │
│ - The void's boundary faces remain in the model, owned by neighbor  │
│   entities. They carry no physical group from the slab itself.      │
│ - SIDE EFFECT: physical name + slab metadata are persisted          │
│   (in the model manager for cad_gmsh; in an XAO sidecar JSON for    │
│   cad_occ) so the mesh stage can find each void.                    │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                            Mesh stage                               │
│ - For each persisted Slab metadata record:                          │
│   1. Locate the void's bounding OCC faces (the neighbor faces that  │
│      enclose the void): bottom face at z=zlo, top at z=zhi, lateral │
│      side faces. Match by xy footprint + z coordinate query within  │
│      point_tolerance. These faces ARE owned by neighbor volumes;    │
│      they must NOT be removed.                                      │
│   2. Recreate the bottom polygon in the geo kernel                  │
│      (geo.addPoint / addLine / addCurveLoop / addPlaneSurface)      │
│      at z=zlo, using the SAME 3D vertex coordinates as the void's   │
│      bottom OCC face.                                               │
│   3. geo.extrude(geo_surface, 0, 0, zhi-zlo, [n_layers], [1.0],     │
│                  recombine=recombine). This creates geo-kernel      │
│      bottom face, top face, lateral side faces, and a volume.       │
│   4. geo.synchronize().                                             │
│   5. Embed the geo kernel boundary into the OCC neighbor faces:     │
│      gmsh.model.mesh.embed(2, [geo_bottom_face], 2, occ_bottom_face)│
│      (and same for top and sides). This forces the neighbor's 2D    │
│      mesh on those shared faces to use the geo extrude's            │
│      vertices/edges, achieving node mating across kernels without   │
│      removing any OCC face.                                         │
│   6. Tag the geo-kernel volume with the slab's physical group.      │
│ - Generate mesh as usual.                                           │
└─────────────────────────────────────────────────────────────────────┘
```

## Slab resolution cascade (detailed)

Implemented in `meshwell/structured_polyprism.py` as a module-level
function `resolve_structured_slabs(entities)` invoked from
`cad_common.prepare_entities`:

```python
def resolve_structured_slabs(entities: list[Any]) -> list[Slab]:
    """Decompose StructuredPolyPrism entities into 3D-disjoint slabs.

    Steps:
    1. For each StructuredPolyPrism, expand into per-z-interval slabs
       using its `buffers` keys and `n_layers` list.
    2. Sort all slabs by (mesh_order, source_index) ascending — same
       tie-break as the CAD fragmentation cascade.
    3. For each pair (S_hi, S_lo) where S_hi has higher priority,
       compute their z-interval intersection [z_lo_max, z_hi_min].
       If empty, skip.
    4. In the overlapping z-sub-interval, compute
          S_lo.footprint.difference(S_hi.footprint)
       If the result is empty: drop S_lo's portion in that sub-interval
       (split S_lo's z-interval into [original_zlo, z_lo_max] +
       [z_hi_min, original_zhi]).
       Else: replace S_lo's footprint in the sub-interval with the
       difference (potentially splitting S_lo into multiple z-intervals).
    5. Drop slabs whose footprint is empty or whose z-interval has zero
       height after resolution.

    Returns a flat list of Slab records, each tagged with its
    originating entity index so physical names can be applied.
    """
```

**Key invariant after resolution:** for any two slabs in the returned
list, either their footprints are 2D-disjoint, or their z-intervals are
disjoint, or both. Touching (shared boundary, zero-volume intersection)
is allowed.

## Phantom CAD body details

After slab resolution, each `Slab` is converted into a transient
"phantom entity" appended to the entities list before fragmentation.
The phantom has:
- `dimension = 3`
- `mesh_order = original.mesh_order` (lets it cut unstructured neighbors
  per the user's intended priority)
- `mesh_bool = False` (top-dim keep=False → removed after fragmentation)
- `physical_name = original.physical_name` — used to name the slab's
  bounding interfaces with neighbors, AND carried forward to the
  geo-kernel volume in the mesh stage.
- `instanciate` / `instanciate_occ` builds a plain OCC extrude.

**Why mesh_bool=False:** The OCC body must vanish so the geo-kernel
extrude can occupy that void cleanly without overlapping CAD volumes.
The existing keep=False top-dim machinery in `cad_gmsh._remove_keep_false_top_dim`
and the XAO writer's keep filter handle this.

## Metadata persistence between CAD and mesh stages

The mesh stage needs each slab's `(footprint, zlo, zhi, n_layers,
recombine, physical_name)` record. Two transport mechanisms:

- **`cad_gmsh`:** ModelManager grows a `structured_slabs: list[Slab]`
  attribute populated during CAD. The mesh entrypoint (`mesh.mesh`)
  reads it directly.
- **`cad_occ` (XAO):** The XAO writer accepts an optional sidecar JSON
  file (`<filename>.structured_slabs.json`) written next to the `.xao`.
  The mesh entrypoint reads this sidecar when present after XAO import.

Both mechanisms feed into a single mesh-stage helper
`apply_structured_slabs(model_manager, slabs)` that performs the
geo-kernel extrude and embed.

## Mesh-stage geo-kernel reinstantiation

`apply_structured_slabs(model_manager, slabs)` runs after the CAD model
is loaded into gmsh but before `gmsh.model.mesh.generate`. For each
slab:

1. **Locate void boundary.** Query OCC entities at `z = zlo` matching
   the slab footprint. Bottom-face candidates are 2D entities whose
   bounding box centroid sits at `z ≈ zlo` and whose xy projection
   matches the slab footprint within `point_tolerance`. Vertical-edge
   candidates: 1D entities with constant xy and z spanning `[zlo, zhi]`.

2. **Build geo replica.** Recreate the slab's bottom polygon in the
   geo kernel, reusing the OCC void's exact vertex coordinates so
   downstream node mating is automatic. Add curve loops for holes if
   the polygon has interiors. Call `geo.addPlaneSurface`.

3. **Geo extrude with layers.** Call
   ```python
   ov = gmsh.model.geo.extrude(
       [(2, geo_surface)], 0, 0, zhi - zlo,
       numElements=[n_layers], heights=[1.0],
       recombine=recombine,
   )
   ```
   `ov[1]` is the new geo-kernel volume.

4. **Embed geo boundary into OCC neighbors.** For each OCC face that
   bounds the void at the slab's bottom/top/sides, use
   `gmsh.model.mesh.embed(dim, geo_tags, target_dim, occ_tag)` to embed
   the geo replica's points/edges/face into the OCC neighbor face.
   This forces gmsh to mesh the OCC neighbor's face using the geo
   replica's vertices/edges, achieving node mating across kernels.

5. **Tag.** Add the slab's physical group to the geo volume (and to its
   bounding faces if the entity contributes to interface naming).

6. **Synchronize.** `geo.synchronize()` once at the end.

**Note:** No OCC face removal is performed. The void's bounding OCC
faces (bottom / top / sides) are owned by neighbor volumes and remain
in place; `mesh.embed` of the geo replica into them is what forces node
mating. Removing them would damage the neighbor volumes.

**Why embed rather than identify-and-merge:** gmsh has no built-in
"merge two faces from different kernels" operation. `embed` is the
documented way to force one entity's mesh to honor another's
discretization, and it works across kernels.

## Backend-specific notes

### `cad_gmsh`
- Phantom slabs flow through `process_entities` like any other entity.
- After fragmentation + keep=False removal, `model_manager.structured_slabs`
  holds the slab list.
- `mesh.mesh(model=model_manager)` calls `apply_structured_slabs` before
  `gmsh.model.mesh.generate`.

### `cad_occ`
- Phantom slabs flow through `cad_occ.process_entities` like any other
  OCC entity.
- The XAO writer skips their bodies (already does, via keep=False).
- A new sidecar writer (`occ_xao_writer.write_structured_slabs_sidecar`)
  serializes the slab list as JSON next to the `.xao`.
- `mesh.mesh(filename=...)` reads the sidecar after XAO import and
  invokes `apply_structured_slabs`.

## Error handling and validation

- Construction-time errors (caught immediately):
  - non-zero buffer values
  - `n_layers` length mismatch
  - non-positive interval heights
  - `n_layers[i] < 1`

- Pre-CAD validation (in `resolve_structured_slabs`):
  - footprint validity (shapely `is_valid`)
  - empty footprint after `set_precision` snap

- Mesh-stage validation (in `apply_structured_slabs`):
  - void boundary faces found at expected z (else
    `MeshwellStructuredError("could not locate slab void at z=...")`)
  - layer-count mismatch on shared vertical faces between two slabs
    (detected by inspecting embedded vertex counts on the shared
    OCC edge): raises with a clear message identifying the two
    physical names and their conflicting `n_layers`.

## Testing strategy

New test file `tests/test_structured_polyprism.py` covering:

1. **Single straight prism, single layer interval.** One slab, no
   neighbors. Verify mesh has exactly `n_layers` layers of wedges
   between zlo and zhi (count nodes on a vertical line).
2. **Single prism, multi-interval (t3 [8, 2] analog).** Verify two
   stacked layer groups with correct counts.
3. **Two structured slabs, xy-disjoint, same z.** Verify both end up
   structured, no fragmentation between them.
4. **Two structured slabs, xy-overlapping, z-disjoint.** Verify
   resolution does not split them (z disjoint).
5. **Two structured slabs, xy-overlapping AND z-overlapping, different
   mesh_order.** Verify lower-`mesh_order` slab wins; loser splits
   into pre-overlap and post-overlap z-intervals.
6. **Structured slab + unstructured `PolySurface` neighbor.** Verify
   structured slab survives intact (neighbor cut at slab boundary).
7. **Layer-mating mismatch error.** Two stacked structured slabs with
   different `n_layers` on shared vertical face: assert clear error.
8. **Recombine flag.** Verify `recombine=True` produces hex elements
   on a rectangular base (regression — pure quad bottom face should
   yield hexes).
9. **Identical OCC and GMSH backend output.** Run the same scene
   through both backends and verify equal element counts + physical
   group membership.

## File-level changes

| File | Change |
|------|--------|
| `meshwell/structured_polyprism.py` | NEW — `StructuredPolyPrism`, `Slab`, `resolve_structured_slabs` |
| `meshwell/cad_common.py` | Hook `resolve_structured_slabs` into `prepare_entities`; emit phantom entities |
| `meshwell/model.py` | Add `structured_slabs: list[Slab]` to `ModelManager` |
| `meshwell/cad_occ.py` | No code changes; phantom slabs already work via keep=False |
| `meshwell/cad_gmsh.py` | No code changes for CAD; phantom slabs already work |
| `meshwell/occ_xao_writer.py` | Write structured-slabs sidecar JSON next to `.xao` |
| `meshwell/mesh.py` | NEW — `apply_structured_slabs` call before `mesh.generate`; sidecar read on XAO path |
| `meshwell/__init__.py` | Export `StructuredPolyPrism` |
| `tests/test_structured_polyprism.py` | NEW — tests above |
| `docs/prisms.py` (notebook) | Add structured-slab example |

## Open risks and mitigations

- **Risk:** `gmsh.model.mesh.embed` across OCC and geo kernels is
  underdocumented and may behave inconsistently.
  **Mitigation:** Build a minimal-reproducer test (test 6 above) early
  in implementation. If embed proves unreliable, fall back to
  rebuilding the slab's bounding faces in the geo kernel and removing
  the corresponding OCC faces, accepting a node-coordinate-based merge
  as a final step (`gmsh.model.mesh.removeDuplicateNodes` after
  generation).

- **Risk:** Slab resolution cascade may produce slivers or self-touching
  polygons under shapely `difference`.
  **Mitigation:** Apply the existing meshwell `point_tolerance` snap +
  `set_precision` after each difference. Drop slabs whose post-snap
  footprint is empty.

- **Risk:** The void's bottom face may not be perfectly co-planar with
  `z = zlo` after CAD fragmentation due to BOP tolerance.
  **Mitigation:** Use `point_tolerance` for the locate-void coordinate
  query, with a clear error if no candidate is within tolerance.

- **Risk:** Users may expect tapered structured prisms (the `buffers`
  dict suggests it's possible).
  **Mitigation:** Document explicitly in docstring that `StructuredPolyPrism`
  requires zero buffers; for tapered, use unstructured `PolyPrism`.
  Construction error message points users to `PolyPrism`.
