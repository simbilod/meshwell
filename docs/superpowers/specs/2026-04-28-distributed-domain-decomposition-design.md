# Distributed Domain Decomposition — Design Spec

Date: 2026-04-28
Status: Draft (pending user review)

## Goal

Add an opt-in distributed pipeline to meshwell that subdivides the
input polygonal domain (the union of polygon-bearing entities) into
user-supplied subdomains, runs CAD + meshing for each subdomain in a
separate process (or on a separate machine), and stitches the results
into one final `.msh` with consistent physical tagging.

The conformity contract at subdomain seams is **mesh-level by
construction**: a pre-pass meshes the seam surfaces once, with full
ResolutionSpec context from both sides, and the volume meshing of each
subdomain is constrained to honor that frozen 2D mesh. Stitching at the
end is a no-op merge plus a tolerance-based safety net.

This realizes the "Distributed memory processing with domain
decomposition" item already advertised in `README.md`.

## Scope (v1)

- **In:** User-supplied subdomain polygons (`list[Polygon]`).
- **In:** Helper `subdomains_from_grid(bbox, nx, ny[, nz])` that emits
  the same `list[Polygon]` from a regular grid.
- **In:** File-based job bundles (one directory per subdomain job)
  exchanged via a pluggable `Executor` protocol; default
  `SubprocessExecutor` uses `concurrent.futures.ProcessPoolExecutor`.
- **In:** Two-phase scheduling: phase 1 meshes interface + junction
  subdomains (surface mesh only); phase 2 meshes volume subdomains
  with the phase-1 results as fixed boundary constraints.
- **In:** User-supplied interface slab width (scalar or per-cut dict).
- **In:** Phantom-imprint of the cut polyline inside each interface
  slab so the seam surface is identified by physical name, not by
  coordinate matching.
- **In:** Junction subdomains for points/edges where ≥3 volume
  subdomains meet.
- **In:** Conformal stitching by construction; physical group merging
  by name equality across subdomain results.
- **Out:** Auto-derivation of `interface_width` from ResolutionSpec
  influence radii (deferred; v1 requires user-supplied width).
- **Out:** Auto-routed cuts (METIS-style adjacency partitioning).
- **Out:** In-memory-only transport (file bundles always).
- **Out:** Restart-from-partial-failure beyond "rerun a single job by
  replaying its bundle."
- **Out:** Heterogeneous-worker load balancing (executor handles
  scheduling; library does not rebalance).
- **Out:** Non-conformal seam joins for visualization-only use cases.

## Public API

New module `meshwell/distributed.py`:

```python
from pathlib import Path
from typing import Any, Protocol
from concurrent.futures import Future
from shapely.geometry import Polygon


class Executor(Protocol):
    def submit(self, job_dir: Path) -> Future: ...


def generate_mesh_distributed(
    entities: list[Any],
    subdomains: list[Polygon],
    output_mesh: Path | str,
    work_dir: Path | str,
    interface_width: float | dict[tuple[int, int], float],
    executor: Executor | None = None,
    keep_bundles: bool = False,
    registry: dict[str, callable] | None = None,
    **mesh_kwargs,
) -> Any: ...


def subdomains_from_grid(
    bbox: tuple[float, float, float, float],
    nx: int,
    ny: int,
) -> list[Polygon]: ...
```

The existing `generate_mesh` entrypoint is unchanged. The distributed
flow is purely additive.

**Argument semantics:**

- `entities`: same shape as `generate_mesh` accepts (PolyPrism,
  PolySurface, etc., or their dict serializations).
- `subdomains`: list of 2D polygons. v1 assumes the same set of polygons
  applies at every z. The full extruded domain is implicitly
  `subdomain_polygon × (-inf, +inf)` for clipping purposes; the actual
  z extent comes from the entities themselves.
- `interface_width`: required. Scalar (one width for every
  interface) or dict keyed by `(i, j)` subdomain pair. Must be
  large enough to encompass any ResolutionSpec influence that should
  affect the seam mesh; the user picks this based on their
  ResolutionSpec characteristic lengths. Auto-derivation is deferred
  to a later version (see Out of Scope).
- `executor`: defaults to `SubprocessExecutor()` (process pool,
  `meshwell run-job` CLI per job).
- `work_dir`: parent directory for job bundles. Created if absent.
- `keep_bundles`: if `False`, bundles are deleted after a successful
  glue.
- `**mesh_kwargs`: forwarded to each worker's `generate_mesh` call.

**Validation at construction:**

- `subdomains` must be non-empty, all `is_valid`, pairwise interior-
  disjoint (touching boundaries allowed). Raises `ValueError`.
- Their union must cover the bounding union of polygon-bearing entities
  within `point_tolerance`. Raises `ValueError` listing uncovered area.
- `interface_width`, scalar form: must be `> 0`. Dict form: every
  pair `(i, j)` with a non-empty shared boundary must have an entry,
  each `> 0`.

## Architecture overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                              MASTER                                 │
│ 1. prepare_entities(entities, perturbation)   — run ONCE globally   │
│ 2. validate(subdomains)                                             │
│ 3. plan = build_subdomain_plan(subdomains, entities, width)         │
│      → volumes:    one per subdomain polygon                        │
│      → interfaces: one per shared edge (slab of width w)            │
│      → junctions:  one per multi-way meeting locus                  │
│ 4. clip every entity against every {volume, interface, junction}    │
│    region (shapely). Preserve physical_name, mesh_order, resolutions│
│ 5. write phase-1 bundles (interfaces + junctions)                   │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  PHASE 1: INTERFACE MESHING (parallel)              │
│ For each interface / junction bundle, executor runs:                │
│   meshwell run-job <bundle>                                         │
│ Worker:                                                             │
│   - deserialize entities (already pre-buffered, includes the        │
│     master's phantom imprint along the cut polyline tagged          │
│     "_seam___volume_i___volume_j", keep=False)                      │
│   - call generate_mesh(dim=3, _pre_buffered=True,                   │
│                        _emit_only_seam_surfaces=True)               │
│   - meshes the slab volumetrically, then exports ONLY the           │
│     physical groups whose name starts with "_seam___"               │
│   - write result.msh containing the 2D mesh on the seam surfaces    │
│   - write result.json with seam-surface inventory + node count      │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│            MASTER: write phase-2 bundles                            │
│ For each volume subdomain, package:                                 │
│   - clipped entities                                                │
│   - subdomain polygon                                               │
│   - paths to all interface/junction result.msh that touch this      │
│     subdomain (placed under bundle/interface_meshes/)               │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  PHASE 2: VOLUME MESHING (parallel)                 │
│ For each volume bundle, executor runs:                              │
│   meshwell run-job <bundle>                                         │
│ Worker:                                                             │
│   - deserialize entities (already pre-buffered)                     │
│   - call generate_mesh(dim=3, _pre_buffered=True,                   │
│                        _interface_constraints=[...interface_meshes])│
│   - inside generate_mesh: after CAD load, gmsh.merge each interface │
│     mesh, then gmsh.model.mesh.embed the interface 2D entities into │
│     the matching OCC face of the volume. Node positions on the seam │
│     are taken from the interface mesh.                              │
│   - write result.msh                                                │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                          MASTER: GLUE                               │
│ - new gmsh model                                                    │
│ - gmsh.merge each volume_*.msh                                      │
│ - gmsh.model.mesh.removeDuplicateNodes(tol=point_tolerance/2)       │
│ - reconcile physical groups by name equality                        │
│   (same name across multiple imported tiles → one merged group)     │
│ - write output_mesh                                                 │
│ - if not keep_bundles: shutil.rmtree(work_dir)                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Subdomain plan construction

Module-level function in `meshwell/distributed.py`:

```python
def build_subdomain_plan(
    subdomains: list[Polygon],
    entities: list[Any],
    interface_width: float | dict | None,
) -> SubdomainPlan: ...
```

Steps:

1. **Volume regions.** Each input polygon becomes one volume region.
   Volume region `i` has id `volume_{i:04d}`.
2. **Pairwise edges.** For each pair `(i, j)` with `i < j`, compute
   `boundary_ij = subdomains[i].boundary.intersection(subdomains[j].boundary)`.
   Discard empty intersections and zero-length results. The remaining
   `boundary_ij` is one or more LineStrings.
3. **Junction loci.** Cluster shared boundary endpoints across all
   `boundary_ij`. Any point shared by ≥3 distinct subdomain polygons
   becomes a junction locus. In 3D the junction is a vertical edge
   spanning the entity z-range; in 2D it is a point.
4. **Interface slabs.** For each `boundary_ij` LineString, generate a
   slab `boundary_ij.buffer(width_ij / 2)` minus a small disk around
   each junction locus that the slab touches (the junction owns those
   neighborhoods). Slab id `interface_{k:04d}`. The slab carries its
   originating `boundary_ij` polyline as a separate field on the plan
   record — this polyline is imprinted into the slab CAD as a phantom
   (see "Interface CAD construction" below) so the seam surface can
   be identified by physical name.
5. **Junction subdomains.** For each junction locus, generate a small
   box `point.buffer(width_jct)` where `width_jct = max(width_ij)`
   over all interfaces meeting at the junction. The junction's plan
   record carries the set of meeting boundary polylines so each can
   be imprinted in the junction CAD.
6. **Coverage check.** Union of all volumes must cover the union of all
   polygon-bearing entity polygons (within `point_tolerance`).

The plan is serialized to `manifest.json` and reused by both phases.

## Interface CAD construction (phase 1)

The phase-1 worker needs to mesh a thin slab around a cut polyline,
then emit *only* the surface mesh that lies on the cut polyline (the
seam). To make seam identification unambiguous, the master imprints
the cut polyline into the slab CAD as a phantom feature with a known
physical name.

**Master, building a phase-1 bundle for interface `(i, j)`:**

1. `cut_line = boundary(subdomains[i]).intersection(boundary(subdomains[j]))`
2. `slab = cut_line.buffer(interface_width / 2)`
3. **Geometry entities into the bundle:** for each pre-buffered entity:
   - `sliver = entity.polygons.intersection(slab)`. If non-empty, ship
     a clipped copy (via `_clip_entity_to_polygon`).
   - Else if the entity is within `interface_width` of the slab AND
     carries any ResolutionSpec, ship the entity *unclipped* with a
     marker `_resolution_only=True`. Phase-1 worker uses it only as a
     ResolutionSpec source (see "Worker handling of resolution-only
     entities" below).
4. **Phantom seam imprint:** add a `keep=False` entity along
   `cut_line` with `physical_name = f"_seam___volume_{i:04d}___volume_{j:04d}"`.
   Concretely this is a meshwell `InterfaceTag` (preferred — it
   already imprints linestrings into the CAD as internal 1D / 2D
   features) or, where InterfaceTag is not applicable, a `keep=False`
   PolySurface.
5. Worker meshes at `dim=3`. The slab is meshed volumetrically; the
   imprint produces a 2D face along the cut surface (vertical
   extrusion of `cut_line` through the entity z-stack) carrying the
   `_seam___volume_i___volume_j` physical name. Existing fragmentation
   propagates that name to every fragment of the seam face.
6. Worker `result.msh` export filter (`_emit_only_seam_surfaces=True`):
   restrict written elements to physical groups whose name starts with
   `_seam___`. Other elements (the slab's volumetric mesh, the slab's
   outer boundary surfaces) are discarded — they were scratch.

**Junction CAD** uses the same mechanism with multiple imprints: each
boundary polyline meeting at the junction is imprinted as a separate
`InterfaceTag`, producing one seam face per pairwise contact in the
junction mesh, each with its own `_seam___volume_i___volume_j` name.

**Why a phantom instead of a coordinate filter:** unambiguous physical
naming survives fragmentation, propagates through XAO export, and
matches at phase-2 import via `gmsh.merge` without any post-mesh
geometry queries. The same `keep=False` machinery is already in use
elsewhere in meshwell (see structured-polyprism design).

### Worker handling of resolution-only entities

An entity shipped with `_resolution_only=True` is added to the
worker's entity list with `mesh_bool=False` and an empty geometry
proxy: its `instanciate_occ` returns an empty shape. After CAD,
its tag list is empty so it produces no fragments and no physical
group. Its `resolutions` list is still passed to the resolution
engine, where any `restrict_to` / `sharing` / `not_sharing` references
to physical names that exist in the worker's local fragment inventory
take effect; references to absent names are no-ops (per
`_global_physical_names` handling). The net effect: the size field
defined by such a ResolutionSpec applies to whatever local entities
match, even though the source entity itself contributes no geometry.

This is a no-op when the ResolutionSpec is keyed only on its own
entity (the common case for `ConstantInField` on "this material") —
that field needs the source entity's geometry to apply. For v1 we
document that ResolutionSpecs whose size field requires the source
entity's own geometry will not extend across subdomain seams; users
who need such extension should choose `interface_width` large enough
that the source entity's geometry intersects the slab directly (i.e.
the entity is shipped fully clipped, not as resolution-only).

## Job bundle format

```
work_dir/
  manifest.json
  jobs/
    interface_0001/
      job.json
      entities.json
      subdomain.wkt
      mesh_kwargs.json
      [outputs:]
      result.msh
      result.json
    junction_0001/
      ...                    # same shape as interface
    volume_0003/
      job.json
      entities.json
      subdomain.wkt
      mesh_kwargs.json
      interface_meshes/      # populated between phase 1 and phase 2
        interface_0001.msh
        junction_0002.msh
      [outputs:]
      result.msh
      result.json
```

**`manifest.json`** schema:

```json
{
  "version": 1,
  "perturbation": 1.0e-5,
  "point_tolerance": 1.0e-3,
  "physical_names_seen": ["silicon", "oxide", "air", ...],
  "interface_delimiter": "___",
  "boundary_delimiter": "None",
  "subdomains": {
    "volume_0000": {"polygon_wkt": "...", "neighbors": ["volume_0001", ...]},
    "interface_0001": {"polygon_wkt": "...", "between": ["volume_0000", "volume_0001"]},
    "junction_0002": {"polygon_wkt": "...", "between": ["volume_0000", "volume_0001", "volume_0002"]}
  },
  "phase_order": [
    ["interface_0001", "junction_0002", ...],   // phase 1, all parallel
    ["volume_0000", "volume_0001", ...]         // phase 2, all parallel
  ]
}
```

**`job.json`** schema:

```json
{
  "id": "volume_0003",
  "role": "volume",                     // "interface" | "junction" | "volume"
  "dim": 3,
  "interface_inputs": [                  // empty for phase 1
    {"id": "interface_0001", "path": "interface_meshes/interface_0001.msh"},
    {"id": "junction_0002", "path": "interface_meshes/junction_0002.msh"}
  ],
  "neighbors": ["volume_0001", "volume_0004"],
  "manifest_ref": "../../manifest.json"
}
```

**`entities.json`** uses the existing `to_dict` serialization that
PolyPrism / PolySurface / etc. already implement. ResolutionSpec
already has `to_dict` (see `meshwell/resolution.py`).

**`result.json`** schema:

```json
{
  "status": "ok",
  "elapsed_s": 12.3,
  "physical_groups": {"silicon": {"dim": 3, "n_elements": 12345}, ...},
  "seam_surface_ids": ["interface_0001:silicon___oxide", ...]   // phase 1 only
}
```

## Worker entrypoint

New CLI subcommand wired into `meshwell/__init__.py` via
`pyproject.toml`:

```bash
meshwell run-job <work_dir>/jobs/<job_id>
```

Implementation sketch in `meshwell/distributed.py`:

```python
def run_job(job_dir: Path) -> None:
    job = json.loads((job_dir / "job.json").read_text())
    manifest = json.loads((job_dir / job["manifest_ref"]).read_text())
    entities = deserialize(
        json.loads((job_dir / "entities.json").read_text()),
        registry=...,  # propagated via env or job.json
    )
    mesh_kwargs = json.loads((job_dir / "mesh_kwargs.json").read_text())

    extra = {
        "_pre_buffered": True,
        "perturbation": 0.0,            # never re-buffer
        "point_tolerance": manifest["point_tolerance"],
        "interface_delimiter": manifest["interface_delimiter"],
        "boundary_delimiter": manifest["boundary_delimiter"],
        "_global_physical_names": manifest["physical_names_seen"],
    }

    if job["role"] in ("interface", "junction"):
        extra["_emit_only_seam_surfaces"] = True
        # No _seam_targets needed: filter is purely by physical-name
        # prefix "_seam___" (set by the master via the phantom imprint).

    if job["role"] == "volume":
        extra["_interface_constraints"] = [
            (job_dir / inp["path"]) for inp in job["interface_inputs"]
        ]

    result = generate_mesh(
        entities,
        dim=job["dim"],
        output_mesh=job_dir / "result.msh",
        **mesh_kwargs,
        **extra,
    )

    (job_dir / "result.json").write_text(json.dumps({...}, indent=2))
```

## Changes to `generate_mesh` / mesh.py / cad_common

The distributed flow needs three new private kwargs threaded through
the existing pipeline. They are underscored to mark them internal;
distributed.py is the only intended caller.

### `_pre_buffered: bool = False`

In `cad_common.prepare_entities`: skip the buffer pass when this flag
is set. InterfaceTag resolution still runs (it's idempotent and still
needed). The flag is plumbed via a new `prepare_entities(...,
skip_buffer: bool = False)` parameter.

### `_emit_only_seam_surfaces: bool = False`

In `meshwell/mesh.py`, after `mesh.generate(dim=3)` and before
writing `output_mesh`: restrict the emitted physical groups + element
export to those whose name starts with the prefix `_seam___`. These
are the phantom-imprint physical groups put in place by the master
(see "Interface CAD construction"). All other physical groups and
their elements are stripped before writing.

Implementation: walk `gmsh.model.getPhysicalGroups()`, identify
target groups by name prefix, then write a filtered `.msh`. The
slab's volumetric mesh and outer-boundary surfaces are scratch and
do not appear in `result.msh`.

### `_interface_constraints: list[Path] = []`

In `meshwell/mesh.py`, after CAD load and synchronize, before
`mesh.generate(dim=3)`:

```python
for path in _interface_constraints:
    gmsh.merge(str(path))
    # Imported entities show up as discrete entities in gmsh.
    # For each imported 2D discrete entity, locate the matching OCC
    # face by physical name + bbox + planar-equation match.
    # Then gmsh.model.mesh.embed(2, [discrete_2d_tag], 2, occ_face_tag)
    # forces the OCC face's mesh to honor the imported nodes.
    _embed_imported_seam_into_occ_face(...)
```

The matching helper `_embed_imported_seam_into_occ_face` lives in
`meshwell/mesh.py`. It uses the same physical name discipline (interface
name = `lhs___rhs` ordered alphabetically) so both phases agree.

### `_global_physical_names: list[str]`

In `meshwell/mesh.py` resolution-spec resolver: when a ResolutionSpec
references a name via `restrict_to` / `sharing` / `not_sharing` that is
not present locally, but IS in `_global_physical_names`, log a debug
message and treat as an empty match (no-op). Today the same situation
raises. This stops a worker from failing because a referenced material
lives in another subdomain.

## Master-side clipping helper

```python
def _clip_entity_to_polygon(entity: Any, mask: Polygon) -> Any | None:
    """Return a copy of ``entity`` whose .polygons are intersected with mask.

    Returns None if the intersection is empty (entity drops out of this
    subdomain). Preserves physical_name, mesh_order, mesh_bool,
    additive, resolutions, all transformation params, etc.

    For PolyPrism / PolySurface this is a shapely .intersection on
    the entity's polygons attribute. OCC_entity is not clipped; per
    risk R7 it must be fully contained in a single subdomain (validated
    at plan time).
    """


def _resolution_only_proxy(entity: Any) -> Any:
    """Wrap an entity so it contributes no geometry but keeps its specs.

    Used by phase-1 bundles for entities near (but not intersecting)
    a slab whose ResolutionSpecs may still affect the seam mesh.
    Sets mesh_bool=False and overrides instanciate_occ to return an
    empty TopoDS_Compound. The resolutions list is preserved verbatim.
    """
```

The clip is applied:
- to volume subdomain masks for phase-2 entities,
- to interface slab masks (LineString.buffer) for phase-1 entities,
- to junction box masks for phase-1 junction entities.

Resolutions travel verbatim with each clipped entity.

## Stitching contract details

**Conformity:** The interface mesh from phase 1 is the single source of
truth for node coordinates on every shared seam. Phase 2 workers
embed it into their OCC seam faces, so node coordinates and 2D
connectivity match exactly between adjacent volumes by construction.

**Tolerance safety net:** `removeDuplicateNodes(tol=point_tolerance/2)`
runs after final merge purely to absorb floating-point drift from the
gmsh `merge` round-trip. If this pass changes node count by more than
0.1% the master raises `MeshwellSeamConformityError` (a real bug, not
a numerical hiccup).

**Physical group reconciliation:** After `gmsh.merge` of all
`volume_*.msh`, multiple physical groups may exist with the same name
(one per source tile). Master walks `gmsh.model.getPhysicalGroups()`,
groups by name, calls `gmsh.model.removePhysicalGroups` and
re-adds one consolidated group per name covering the union of all
constituent entity tags.

**Junction handling:** A junction subdomain's `result.msh` contains the
2D mesh on every face that touches the junction locus, including the
slivers near pairwise interfaces. Each adjacent volume worker imports
both the relevant pairwise interfaces AND any touching junctions, and
embeds them all. Where two seams share an edge at a junction, the
embedded edge nodes match because both phase-1 jobs computed them in
the same global coordinate system from the same pre-buffered geometry.

## Failure handling

- **Phase 1 job failure:** master collects the failure, refuses to start
  phase 2, leaves `work_dir` intact, raises `MeshwellJobError` with the
  failed job id and a hint to rerun `meshwell run-job <bundle>` for
  reproduction.
- **Phase 2 job failure:** same — master gives up, leaves bundles for
  inspection.
- **Glue failure:** raises with the offending physical name set or the
  conformity-violation node count delta.
- **`keep_bundles=True`:** even on success, work_dir is preserved.

## Testing strategy

New test file `tests/test_distributed.py`:

1. **Trivial 2x1 grid, single material.** One slab cut, single
   PolyPrism spanning both subdomains. Assert merged mesh is identical
   (within tolerance) to the non-distributed result on the same input.
2. **2x1 grid, two materials with shared interface.** Verify the
   `silicon___oxide` interface physical group exists in the merged
   mesh and contains the same elements as the non-distributed run.
3. **2x2 grid, four materials, one junction at center.** Verify the
   junction subdomain is generated, the four-way edge mesh is shared
   by all four volume workers, and the merged mesh is conformal at
   the junction.
4. **ResolutionSpec near a cut.** Define a ConstantInField on an
   entity adjacent to (but not crossing) the cut, with an
   `interface_width` chosen to enclose the entity. Verify the
   resolution-only proxy mechanism ships the entity to the phase-1
   worker, the seam mesh node spacing reflects the spec, and the
   phase-2 volume meshes inherit those spacings on the seam.
5. **Per-cut interface_width dict.** Pass a dict with two different
   widths for two different cuts; verify each interface bundle uses
   its assigned width.
6. **Phantom imprint identity.** Inspect a phase-1 `result.msh`;
   verify the only physical groups present have names starting with
   `_seam___` and they cover exactly the cut surface area predicted
   by the slab geometry.
7. **Coverage validation.** Pass subdomains that don't cover the
   entity union; verify clean ValueError listing missing area.
8. **Worker reproducibility.** After a successful run with
   `keep_bundles=True`, manually run `meshwell run-job` on a single
   volume bundle; verify identical `result.msh`.
9. **Executor swap.** Subclass Executor with a synchronous-in-process
   variant; verify same final mesh.
10. **Glue physical-group dedup.** Create a scenario where the same
    material appears in 4 tiles; verify one consolidated physical
    group in the final mesh, not 4.
11. **Failure path.** Force one phase-1 job to crash (inject malformed
    entity); verify `MeshwellJobError`, work_dir intact, no phase-2
    submitted.

Cross-cutting: every distributed test also runs its input through the
non-distributed `generate_mesh` and asserts physical-group inventories
match (element counts can differ, but materials and interfaces must
all be present in both).

## File-level changes

| File | Change |
|------|--------|
| `meshwell/distributed.py` | NEW — `generate_mesh_distributed`, `build_subdomain_plan`, `run_job`, `Executor`, `SubprocessExecutor`, `subdomains_from_grid`, `_clip_entity_to_polygon`, `_resolution_only_proxy` |
| `meshwell/cad_common.py` | Add `skip_buffer: bool = False` to `prepare_entities`. |
| `meshwell/mesh.py` | Accept `_pre_buffered`, `_emit_only_seam_surfaces`, `_interface_constraints`, `_global_physical_names` kwargs. Add `_embed_imported_seam_into_occ_face` helper. Tolerate unresolved ResolutionSpec name refs when `_global_physical_names` provided. Filter export by `_seam___` physical-name prefix when `_emit_only_seam_surfaces` is set. |
| `meshwell/orchestrator.py` | Forward the new private kwargs to `mesh()`. |
| `meshwell/__init__.py` | Export `generate_mesh_distributed`, `subdomains_from_grid`, `Executor`, `SubprocessExecutor`. |
| `pyproject.toml` | Add `meshwell` console script → `meshwell.distributed:cli_main`. |
| `tests/test_distributed.py` | NEW — tests above. |
| `docs/distributed.md` | NEW — narrative guide. |
| `README.md` | Move "Distributed memory processing" out of "Planned" into "Key Features". |

## Open risks and mitigations

- **R1: ResolutionSpec cross-references break at clip time** when the
  referenced physical name lives in another subdomain.
  **Mitigation:** ship `physical_names_seen` in the manifest; the
  worker's resolver treats absent-but-globally-known names as empty
  matches with a debug log instead of erroring.

- **R2: User picks `interface_width` too small** to capture relevant
  ResolutionSpec influence, producing a seam mesh that does not match
  the size field the volume workers expect at the seam.
  **Mitigation:** v1 documents the rule of thumb (≥ 4× the largest
  characteristic length of any ResolutionSpec near the cut). Phase-2
  workers detect a > 2× sizing mismatch between the embedded seam
  nodes and the locally-computed size field at the seam, and warn.
  Auto-derivation of `interface_width` is deferred (see Out of Scope).

- **R3: `gmsh.model.mesh.embed` from a `gmsh.merge`-imported discrete
  entity into an OCC face may behave inconsistently** (same risk
  flagged in the structured-polyprism design).
  **Mitigation:** build a minimal repro early. Fallback: bypass embed
  and inject node positions directly via `gmsh.model.mesh.setNodes` on
  the OCC face's boundary points, then let gmsh mesh the face with
  those points fixed. A second fallback is to remove the OCC seam face
  entirely and replace it with the imported discrete face (analogous
  to the structured-polyprism phantom-removal trick).

- **R4: Polygon spans a subdomain boundary at a sharp feature** (e.g. a
  thin rectangle whose long axis crosses a cut), creating a sliver
  after clipping that is too thin for stable BOP.
  **Mitigation:** reject clipped entity copies whose post-snap area is
  below `point_tolerance²` per polygon, with a warning naming the
  entity. Document that the user should reroute the cut through a
  quieter region or use a finer subdivision in that area.

- **R5: Junction in 3D between subdomains with different vertical
  entity stacks** (entity zmin/zmax mismatch across subdomains) yields
  a junction edge that is not actually shared.
  **Mitigation:** plan-construction step computes the junction edge as
  the z-range intersection of the touching entity stacks; degenerate
  (zero-height) junctions are dropped with a warning.

- **R6: `keep_bundles=False` deletes evidence of a near-miss bug.**
  **Mitigation:** default `keep_bundles=False` for happy path, but
  always preserve bundles when any worker exits non-zero or the glue
  raises.

- **R7: Master-side clipping for `OCC_entity` (user-supplied callable)
  is not a shapely op** — there is no polygon to intersect.
  **Mitigation:** v1 documents that `OCC_entity` instances must be
  fully contained within a single subdomain (validated at plan time
  via their bounding box hint); spanning-OCC-entity support is
  deferred.
