# Distributed Domain Decomposition

Meshwell can subdivide a polygonal scene into per-subdomain CAD + meshing
jobs that run in separate processes (or on separate machines via a
pluggable `Executor` protocol), then stitch the resulting per-tile
meshes into one unified mesh with conformal seams and consistent
physical-group tagging.

This is the implementation of the "Distributed memory processing with
domain decomposition" feature mentioned in the README.

See the design spec at
[`docs/superpowers/specs/2026-04-28-distributed-domain-decomposition-design.md`](superpowers/specs/2026-04-28-distributed-domain-decomposition-design.md)
for the full architectural rationale (including the v2 amendment) and
the v2 plan at
[`docs/superpowers/plans/2026-04-28-distributed-v2.md`](superpowers/plans/2026-04-28-distributed-v2.md).

## When to use

- The scene is large enough that a single-process meshing run is too
  slow or runs out of memory.
- You have multiple cores or multiple machines available.
- Per-subdomain meshing time dominates the master's I/O / clip /
  glue cost (which is bounded by shapely operations on polygons; very
  cheap relative to OCC fragmentation + 3D meshing).

## Limitations

The v2 pipeline supports arbitrary `NxM` grids (no corner-tile or
N>=3 strip restrictions). The narrower limitations are:

- **Adjacent tiles must use the same characteristic length.** The
  stitch step welds coincident nodes via `removeDuplicateNodes`; if
  neighbouring tiles mesh their shared face with different sizing,
  only the corner nodes will weld and the seam will be non-conformal
  in the interior. Use a single `default_characteristic_length` (or
  matched `ResolutionSpec`s along shared faces) when conformity
  matters. Refining one tile globally relative to another is fine
  away from seams.
- **Material-material interface naming convention shifts.** In v1
  (single-CAD), abutting `silicon` and `oxide` materials produced a
  synthesized `silicon___oxide` interface group. In v2, each tile
  meshes independently and only sees its own materials, so each tile
  emits its own `<material>___None` boundary group at subdomain
  edges. After stitch, the welded face cells stay tagged with their
  per-tile boundary names — there is no `silicon___oxide` group. Users
  who need cross-material interface naming should post-process the
  output mesh to detect coincident face cells from differently-named
  groups and re-tag them.
- **`OCC_entity` instances must fit in one subdomain** (unchanged
  from v1). Master-side clipping is a shapely op; arbitrary OCC
  callables can't be clipped that way, so spanning-OCC-entity support
  is deferred.

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

# Any NxM grid works in v2; here a 2x1 split.
subdomains = subdomains_from_grid((0, 0, 2, 1), nx=2, ny=1)

generate_mesh_distributed(
    entities=[silicon, oxide],
    subdomains=subdomains,
    output_mesh="merged.msh",
    work_dir="dist_work",
    executor=InProcessExecutor(),     # use SubprocessExecutor() for parallel
    default_characteristic_length=0.3,
)
```

## How it works

```
┌─────────────────────────────────────────────────────────────────────┐
│  MASTER                                                             │
│  1. prepare_entities once globally (perturbation buffer)            │
│  2. clip entities to each subdomain (drop slivers below             │
│     point_tolerance²; clip against mask eroded by perturbation      │
│     to keep buffer halos out of neighbouring subdomains)            │
│  3. write per-tile bundles (entities.json, subdomain.wkt, job.json) │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│  WORKERS (parallel)                                                 │
│  Each tile meshes independently with _hashed_physical_tags=True,    │
│  so the integer tag for each physical-group name is identical       │
│  across all per-tile .msh files.                                    │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│  MASTER: stitch                                                     │
│  gmsh.merge each per-tile .msh; gmsh auto-unions entities sharing   │
│  a (dim, tag); removeDuplicateNodes welds coincident nodes at the   │
│  seams; gmsh.write the unified mesh.                                │
└─────────────────────────────────────────────────────────────────────┘
```

## Why hashed tags

The stitch step relies on `gmsh.merge`, which loads each per-tile
`.msh` into one model and unions entities that share a `(dim, tag)`
key. Without coordination, gmsh assigns physical-group tags based on
local insertion order, so silicon's tag in `volume_0000.msh` would
collide with oxide's tag in `volume_0001.msh` and the merged mesh
would have the wrong material everywhere.

To avoid this, workers run with `_hashed_physical_tags=True`. After
the local mesh is built, every physical group is re-tagged using
`_name_to_tag(name, dim) = sha1("<dim>:<name>") mod 1_000_000`. The
result is deterministic across processes, runs, and machines: every
file calls silicon at dim=3 the same integer. After `gmsh.merge`,
gmsh consolidates by `(dim, tag)` so silicon from all tiles ends up
in one physical group named "silicon". No post-merge consolidation
pass is needed.

The 1,000,000 tag space is chosen to sit safely above gmsh's
auto-tag range (which starts at 1 and grows with the model), so
hash-derived tags don't collide with internally-generated tags
during merge. The empirical merge spike at `tests/test_merge_spike.py`
validates that this approach handles arbitrary `NxM` grids without
seam meshes or OCC face seeding.

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

- **Master-side perturbation + eroded-mask clipping.** The master
  buffers all polygons by `perturbation` (1e-5 default) once globally
  for fragmentation robustness, then clips each entity to a
  subdomain mask eroded inward by the same `perturbation`. This
  prevents buffer halos from leaking into adjacent subdomains and
  spuriously erasing materials there.
- **gmsh.merge + removeDuplicateNodes.** v1 used a meshio-based stitch
  to work around `gmsh.merge`'s physical-group tag collisions. v2
  fixes the collision at the source (hashed tags) and goes back to
  `gmsh.merge`, which is faster and preserves OCC topology hints.
