# Distributed Domain Decomposition

Meshwell can subdivide a polygonal scene into per-subdomain CAD + meshing
jobs that run in separate processes (or on separate machines via a
pluggable `Executor` protocol), then stitch the results into one
unified mesh with conformal seams and consistent physical-group
tagging.

This is the implementation of the "Distributed memory processing with
domain decomposition" feature mentioned in the README.

See the design spec at
[`docs/superpowers/specs/2026-04-28-distributed-domain-decomposition-design.md`](superpowers/specs/2026-04-28-distributed-domain-decomposition-design.md)
for the full architectural rationale.

## When to use

- The scene is large enough that a single-process meshing run is too
  slow or runs out of memory.
- You have multiple cores or multiple machines available.
- Per-subdomain meshing time dominates the master's I/O / clip /
  glue cost (which is bounded by shapely operations on polygons; very
  cheap relative to OCC fragmentation + 3D meshing).

## v1 limitations

- **Subdomain layouts must be strip-shaped (`Nx1` or `1xN`)**, i.e. each
  volume tile shares boundaries with at most one other tile per side.
  2D grids (e.g. `2x2`, `3x3`) and any layout with interior corner
  junctions are not supported in v1; they currently fail with
  `"The 1D mesh seems not to be forming a closed loop"` during phase-2
  meshing. See the spec's "Out (v1 limitation)" section for the cause
  and the v2 fix path.
- `interface_width` must be supplied explicitly (no auto-derivation
  from ResolutionSpec radii in v1).
- `OCC_entity` instances must be fully contained within a single
  subdomain (no spanning support in v1).

## Quick start

```python
import shapely
from meshwell.distributed import (
    InProcessExecutor,
    generate_mesh_distributed,
    subdomains_from_grid,
)
from meshwell.polyprism import PolyPrism

# Two materials abutting at x=1.
silicon = PolyPrism(
    polygons=shapely.box(0, 0, 1, 1),
    buffers={0.0: 0.0, 1.0: 0.0},
    physical_name="silicon",
    mesh_order=1,
)
oxide = PolyPrism(
    polygons=shapely.box(1, 0, 2, 1),
    buffers={0.0: 0.0, 1.0: 0.0},
    physical_name="oxide",
    mesh_order=2,
)

# Strip layout: two tiles, no interior corners (v1-supported).
subdomains = subdomains_from_grid((0, 0, 2, 1), nx=2, ny=1)

generate_mesh_distributed(
    entities=[silicon, oxide],
    subdomains=subdomains,
    output_mesh="merged.msh",
    work_dir="dist_work",
    interface_width=0.1,
    executor=InProcessExecutor(),     # use SubprocessExecutor() for parallel
    default_characteristic_length=0.3,
)
```

## How it works

```
┌─────────────────────────────────────────────────────────────────────┐
│  MASTER                                                             │
│  1. prepare_entities once globally (perturbation buffer)            │
│  2. build subdomain plan: volumes, interface slabs, junctions       │
│  3. clip entities to each subdomain (drop slivers below             │
│     point_tolerance²; clip against mask eroded by perturbation      │
│     to keep buffer halos out of neighbouring subdomains)            │
│  4. write per-job bundles (entities.json, subdomain.wkt, job.json)  │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1: interface meshing (parallel)                              │
│  Each interface slab gets a phantom seam imprint along the cut      │
│  polyline, tagged "_seam___volume_i___volume_j". Worker meshes      │
│  the slab in 3D, exports only the _seam___-tagged surface mesh.     │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 2: volume meshing (parallel)                                 │
│  Each volume worker imports the seam meshes from its neighbours,    │
│  seeds them onto the matching OCC face via gmsh.model.mesh.addNodes │
│  with parametric (u,v) coordinates and Mesh.MeshOnlyEmpty=1, then   │
│  meshes the volume in 3D.                                           │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│  MASTER: glue                                                       │
│  Concatenate all volume_*.msh via meshio (preserves per-file        │
│  physical-group names through the merge), dedup nodes within        │
│  point_tolerance/2, consolidate same-named groups across tiles.     │
└─────────────────────────────────────────────────────────────────────┘
```

## Choosing `interface_width`

The interface slab needs to be wide enough to capture any
ResolutionSpec influence that should affect the seam mesh sizing. Rule
of thumb: pick `interface_width >= 4 * (largest characteristic length
of any ResolutionSpec near the cut)`. Smaller widths mean the seam
mesh may not reflect the resolution refinements the ResolutionSpecs
would have produced in a serial run.

In v2, this will be auto-derived; for v1 it is a required input.

## Plugging a custom Executor

The `Executor` protocol is one method:

```python
from concurrent.futures import Future
from pathlib import Path

class MyExecutor:
    def submit(self, job_dir: Path) -> Future:
        # ... dispatch `meshwell run-job <job_dir>` to your scheduler ...
        # Return a Future whose result() blocks until the job finishes.
```

Built-in implementations:

- `InProcessExecutor` — synchronous, runs in the calling process. For
  tests and debugging.
- `SubprocessExecutor(max_workers=N)` — `concurrent.futures.ProcessPoolExecutor`
  invoking the `meshwell run-job` CLI. Default for parallelism on a
  single machine.

For Slurm/Ray/k8s, write an adapter that submits the job to your
scheduler and returns a Future. Each job is a self-contained
directory; bundles can be shipped over network filesystems or
copied to remote workers.

## Debugging

Pass `keep_bundles=True` to preserve the per-job bundle directories
after a successful run. To rerun a single job in isolation:

```bash
meshwell run-job <work_dir>/jobs/<job_id>
```

The CLI is the same code path workers use; you can step through it
with a debugger.

## Architectural notes

- **OCC face seeding via parametric `addNodes`.** Phase-2 workers do
  not use `gmsh.merge` + `gmsh.model.mesh.embed` (which fails for
  cross-kernel imports — see the R3 spike at
  `tests/test_distributed_spike.py`). Instead they read the seam
  `.msh` directly, compute parametric `(u, v)` coordinates on the OCC
  face via `gmsh.model.getParametrization`, and inject nodes +
  triangles via `gmsh.model.mesh.addNodes` + `addElementsByType`. The
  spike test pins the working recipe so future gmsh upgrades that
  break it are caught.
- **Master-side perturbation + eroded-mask clipping.** The master
  buffers all polygons by `perturbation` (1e-5 default) once globally
  for fragmentation robustness, then clips each entity to a
  subdomain mask eroded inward by the same `perturbation`. This
  prevents buffer halos from leaking into adjacent subdomains and
  spuriously erasing materials there.
- **Seam imprint via PolySurface.** Each cut polyline is imprinted in
  the phase-1 slab CAD as a phantom `keep=False` PolySurface tagged
  `_seam___volume_i___volume_j`. The seam stripe is widened to
  `max(slab.width * 0.001, point_tolerance * 2)` to survive shapely
  precision snapping.
- **meshio-based stitch.** The final merge uses meshio rather than
  `gmsh.merge` because gmsh.merge collides physical-group tag IDs
  across files (only the first file's name survives). meshio's
  `field_data` carries names per-file and is consolidated by name in
  the master.
