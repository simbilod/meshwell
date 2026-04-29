"""Complex distributed-meshing example: a ridge waveguide on a stack.

Builds a 2 µm x 1 µm x 1 µm domain with four extruded materials
(substrate, slab, ridge, cladding) — a simple photonics-style cross-
section — and splits it in half along x for distributed meshing. Two
processes mesh the two halves and the result is stitched into a single
mesh preserving every material and the material-material interfaces
across the cut.

v1 supports only **N=2 strip layouts** because interior tiles in
N>=3 strips import seams from both neighbours, and those seams share
the tile's top/bottom OCC edges — the same node-closure issue that
blocks 2D grids. See ``docs/distributed.md`` for the full v1 limit
and the v2 fix path.

Run::

    python docs/distributed_example.py

Outputs:
- ``distributed_example.msh`` — final stitched mesh.
- ``distributed_example_serial.msh`` — non-distributed baseline for
  comparison (same input, no domain decomposition).
- ``distributed_example_work/`` — per-tile bundle directories
  (preserved for inspection; delete after use).

Inspect with gmsh::

    gmsh distributed_example.msh
    gmsh distributed_example_serial.msh

The two meshes should have identical physical-group inventories
(materials + interfaces). Element counts will differ because the
distributed run threads conformal seam meshes between adjacent tiles
while the serial run meshes the whole scene at once.
"""
from __future__ import annotations

import time
from pathlib import Path

import meshio
import shapely

from meshwell.distributed import (
    SubprocessExecutor,
    generate_mesh_distributed,
    subdomains_from_grid,
)
from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism

# Geometry: 2 µm long x 1 µm wide, vertical stack of substrate / slab /
# ridge / cladding. All dimensions in microns.
X_MIN, X_MAX = 0.0, 2.0
Y_MIN, Y_MAX = 0.0, 1.0

SUBSTRATE_Z = (0.0, 0.3)
SLAB_Z = (0.3, 0.45)
RIDGE_Z = (0.45, 0.7)
CLADDING_Z = (0.7, 1.0)

# Ridge runs the full x extent, centered in y, 0.4 µm wide.
RIDGE_Y = (0.3, 0.7)


def build_entities() -> list[PolyPrism]:
    """Construct the four-material stack as a list of PolyPrisms.

    Materials are extruded polygons. mesh_order resolves overlaps
    (lower wins): the ridge cuts the cladding above the slab.
    """
    full_xy = shapely.box(X_MIN, Y_MIN, X_MAX, Y_MAX)
    ridge_xy = shapely.box(X_MIN, RIDGE_Y[0], X_MAX, RIDGE_Y[1])

    substrate = PolyPrism(
        polygons=full_xy,
        buffers={SUBSTRATE_Z[0]: 0.0, SUBSTRATE_Z[1]: 0.0},
        physical_name="substrate",
        mesh_order=4,
    )
    slab = PolyPrism(
        polygons=full_xy,
        buffers={SLAB_Z[0]: 0.0, SLAB_Z[1]: 0.0},
        physical_name="slab",
        mesh_order=3,
    )
    ridge = PolyPrism(
        polygons=ridge_xy,
        buffers={RIDGE_Z[0]: 0.0, RIDGE_Z[1]: 0.0},
        physical_name="ridge",
        mesh_order=1,  # wins over cladding wherever they overlap
    )
    cladding = PolyPrism(
        polygons=full_xy,
        buffers={RIDGE_Z[0]: 0.0, CLADDING_Z[1]: 0.0},
        physical_name="cladding",
        mesh_order=2,
    )
    return [substrate, slab, ridge, cladding]


def report_inventory(label: str, mesh_path: Path) -> dict[str, int]:
    """Read a .msh, print physical-group element counts, return them."""
    m = meshio.read(mesh_path)
    counts: dict[str, int] = {}
    field_data = m.field_data or {}
    for cb_idx, _cb in enumerate(m.cells):
        if not m.cell_data or "gmsh:physical" not in m.cell_data:
            continue
        phys = m.cell_data["gmsh:physical"][cb_idx]
        # Map tag -> name
        tag_to_name = {int(arr[0]): name for name, arr in field_data.items()}
        for tag in phys:
            name = tag_to_name.get(int(tag), f"<tag {tag}>")
            counts[name] = counts.get(name, 0) + 1

    print(f"\n=== {label} ({mesh_path.name}) ===")
    for name in sorted(counts):
        print(f"  {name:40s} {counts[name]:>8d} cells")
    print(f"  total cells: {sum(counts.values())}")
    print(f"  total nodes: {len(m.points)}")
    return counts


def main() -> None:
    here = Path(__file__).parent
    work_dir = here / "distributed_example_work"
    out_distributed = here / "distributed_example.msh"
    out_serial = here / "distributed_example_serial.msh"

    # ------------------------------------------------------------------
    # 1. Serial baseline.
    # ------------------------------------------------------------------
    print(">>> Serial run (single process)...")
    t0 = time.time()
    generate_mesh(
        entities=build_entities(),
        dim=3,
        output_mesh=out_serial,
        default_characteristic_length=0.15,
    )
    print(f"   serial wall time: {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 2. Distributed run: 2 tiles along x (v1-supported shape).
    # ------------------------------------------------------------------
    subdomains = subdomains_from_grid(
        bbox=(X_MIN, Y_MIN, X_MAX, Y_MAX),
        nx=2,
        ny=1,
    )
    print("\n>>> Distributed run (2 tiles, SubprocessExecutor)...")
    t0 = time.time()
    generate_mesh_distributed(
        entities=build_entities(),
        subdomains=subdomains,
        output_mesh=out_distributed,
        work_dir=work_dir,
        executor=SubprocessExecutor(max_workers=2),
        keep_bundles=True,  # leave bundles for inspection
        default_characteristic_length=0.15,
    )
    print(f"   distributed wall time: {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 3. Inventory comparison.
    # ------------------------------------------------------------------
    serial_inv = report_inventory("Serial baseline", out_serial)
    dist_inv = report_inventory("Distributed (stitched)", out_distributed)

    # The distributed merged mesh adds _seam___volume_*___volume_* groups
    # for each interface between tiles; filter those out of the
    # comparison since they don't exist in the serial run.
    dist_no_seam = {n: c for n, c in dist_inv.items() if not n.startswith("_seam___")}
    serial_names = set(serial_inv)
    dist_names = set(dist_no_seam)

    print("\n=== Inventory comparison ===")
    print(f"Names in serial only:      {sorted(serial_names - dist_names)}")
    print(f"Names in distributed only: {sorted(dist_names - serial_names)}")
    print(f"Names in both:             {sorted(serial_names & dist_names)}")

    print("\nDone. Inspect with:")
    print(f"  gmsh {out_distributed}")
    print(f"  gmsh {out_serial}")
    print(f"\nPer-tile bundles preserved at {work_dir}/jobs/")
    print("  Each volume_<id>/ has entities.json, subdomain.wkt, result.msh")
    print("  Each interface_<id>/ has the seam mesh shipped to phase 2")
    print("\nTo rerun a single bundle in isolation:")
    print(f"  meshwell run-job {work_dir}/jobs/volume_0000")


if __name__ == "__main__":
    main()
