"""Distributed-meshing example: a ridge waveguide on a stack across a 3x2 grid.

Builds a 3 µm x 2 µm x 1 µm domain with four extruded materials
(substrate, slab, ridge, cladding) — a simple photonics-style cross-
section — and decomposes it into a 3x2 grid of 6 tiles for distributed
meshing. Workers mesh each tile independently and the master stitches
them into a single mesh preserving every material across the cuts.

The v2 distributed pipeline supports arbitrary `NxM` grids (no v1-era
N=2 strip restriction). See ``docs/distributed.md`` for the narrower
v2 limitations (matched-lc requirement and the cross-material naming
convention shift).

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

The two meshes should have the same materials present (substrate,
slab, ridge, cladding). Element counts will differ because the
distributed run weld-stitches independent per-tile meshes while the
serial run meshes the whole scene at once. The distributed mesh also
carries per-tile boundary groups (``<material>___None``) at the
subdomain edges; see ``docs/distributed.md`` for the convention.
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

# Geometry: 3 µm long x 2 µm wide, vertical stack of substrate / slab /
# ridge / cladding. All dimensions in microns.
X_MIN, X_MAX = 0.0, 3.0
Y_MIN, Y_MAX = 0.0, 2.0

SUBSTRATE_Z = (0.0, 0.3)
SLAB_Z = (0.3, 0.45)
RIDGE_Z = (0.45, 0.7)
CLADDING_Z = (0.7, 1.0)

# Ridge runs the full x extent, centered in y, 0.4 µm wide.
RIDGE_Y = (0.8, 1.2)


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
    # 2. Distributed run: 3x2 grid (6 tiles), v2 supports arbitrary NxM.
    # ------------------------------------------------------------------
    nx, ny = 3, 2
    subdomains = subdomains_from_grid(
        bbox=(X_MIN, Y_MIN, X_MAX, Y_MAX),
        nx=nx,
        ny=ny,
    )
    print(f"\n>>> Distributed run ({nx}x{ny} = {nx * ny} tiles, SubprocessExecutor)...")
    t0 = time.time()
    generate_mesh_distributed(
        entities=build_entities(),
        subdomains=subdomains,
        output_mesh=out_distributed,
        work_dir=work_dir,
        executor=SubprocessExecutor(max_workers=nx * ny),
        keep_bundles=True,  # leave bundles for inspection
        default_characteristic_length=0.15,
    )
    print(f"   distributed wall time: {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 3. Inventory comparison.
    # ------------------------------------------------------------------
    serial_inv = report_inventory("Serial baseline", out_serial)
    dist_inv = report_inventory("Distributed (stitched)", out_distributed)

    # The distributed merged mesh adds <material>___None boundary groups
    # at each subdomain edge (v2 convention shift); filter those out of
    # the bare-material comparison.
    dist_materials = {n: c for n, c in dist_inv.items() if "___" not in n}
    serial_names = set(serial_inv)
    dist_names = set(dist_materials)

    print("\n=== Inventory comparison (materials only) ===")
    print(f"Names in serial only:      {sorted(serial_names - dist_names)}")
    print(f"Names in distributed only: {sorted(dist_names - serial_names)}")
    print(f"Names in both:             {sorted(serial_names & dist_names)}")

    print("\nDone. Inspect with:")
    print(f"  gmsh {out_distributed}")
    print(f"  gmsh {out_serial}")
    print(f"\nPer-tile bundles preserved at {work_dir}/jobs/")
    print("  Each volume_<id>/ has entities.json, subdomain.wkt, result.msh")
    print("\nTo rerun a single bundle in isolation:")
    print(f"  meshwell run-job {work_dir}/jobs/volume_0000")


if __name__ == "__main__":
    main()
