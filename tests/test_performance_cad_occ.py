"""Performance benchmark comparing OCC and GMSH CAD engines."""

import os
import time

import shapely

from meshwell.cad_gmsh import CAD as CAD_GMSH
from meshwell.cad_occ import CAD_OCC
from meshwell.model import ModelManager
from meshwell.polyprism import PolyPrism


def test_performance_comparison():
    """Benchmark CAD processing time for a large number of entities."""
    # Create a grid of prisms that overlap slightly or are adjacent
    # to trigger fragmentation/cuts
    n = 15  # 15x15 = 225 entities
    entities = []
    for i in range(n):
        for j in range(n):
            poly = shapely.Polygon(
                [[i, j], [i + 1.1, j], [i + 1.1, j + 1.1], [i, j + 1.1]]
            )
            entities.append(
                PolyPrism(
                    poly,
                    buffers={0: 0, 1: 0},
                    physical_name=f"p_{i}_{j}",
                    mesh_order=1,  # All in same layer for maximal parallelization
                )
            )

    print(f"\nBenchmarking with {len(entities)} entities...")
    time.sleep(2)  # Give system a breather

    # 1. GMSH Backend
    mm_gmsh = ModelManager()
    mm_gmsh.ensure_initialized("perf_gmsh.xao")
    cad_gmsh = CAD_GMSH()

    start_gmsh = time.perf_counter()
    cad_gmsh.process_entities(entities)
    end_gmsh = time.perf_counter()
    duration_gmsh = end_gmsh - start_gmsh
    print(f"GMSH backend processing time: {duration_gmsh:.4f}s")
    mm_gmsh.finalize()

    time.sleep(2)

    # 2. OCC Backend (New)
    # No need for gmsh model during OCC processing
    cad_occ = CAD_OCC(n_threads=os.cpu_count())

    start_occ = time.perf_counter()
    cad_occ.process_entities(entities)
    end_occ = time.perf_counter()
    duration_occ = end_occ - start_occ
    print(f"OCC backend (Parallel) processing time: {duration_occ:.4f}s")

    speedup = duration_gmsh / duration_occ if duration_occ > 0 else float("inf")
    print(f"Speedup: {speedup:.2f}x")

    # Optional: Verify OCC to GMSH bridge speed too
    mm_bridge = ModelManager()
    mm_bridge.ensure_initialized("perf_bridge.xao")
    from meshwell.occ_to_gmsh import inject_occ_entities_into_gmsh

    # We need the entities processed by OCC
    occ_entities = cad_occ.process_entities(entities)

    start_bridge = time.perf_counter()
    inject_occ_entities_into_gmsh(occ_entities, mm_bridge)
    end_bridge = time.perf_counter()
    duration_bridge = end_bridge - start_bridge
    print(f"OCC to GMSH bridge injection time: {duration_bridge:.4f}s")
    mm_bridge.finalize()

    # Note: Even with bridge overhead, pure OCC CAD + bridge
    # should ideally be competitive with pure GMSH CAD for large N.
    total_new = duration_occ + duration_bridge
    print(speedup)
    print(f"Total New (OCC+Bridge) time: {total_new:.4f}s")


if __name__ == "__main__":
    test_performance_comparison()

if __name__ == "__main__":
    test_performance_comparison()
