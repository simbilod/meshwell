"""Performance sanity check for the OCC CAD engine."""

import os
import time

import pytest
import shapely

from meshwell.cad_occ import CAD_OCC
from meshwell.model import ModelManager
from meshwell.polyprism import PolyPrism


@pytest.mark.skip(reason="Too slow - run manually for perf investigations")
def test_performance_occ():
    """Benchmark the OCC CAD pipeline for a large entity grid."""
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
                    mesh_order=1,
                )
            )

    print(f"\nBenchmarking with {len(entities)} entities...")

    cad = CAD_OCC(n_threads=os.cpu_count())
    t0 = time.perf_counter()
    occ_entities = cad.process_entities(entities)
    t1 = time.perf_counter()
    print(f"OCC CAD time: {t1 - t0:.4f}s")

    mm = ModelManager()
    mm.ensure_initialized("perf_bridge.xao")
    t2 = time.perf_counter()
    mm.load_occ_entities(occ_entities)
    t3 = time.perf_counter()
    print(f"OCC->GMSH bridge time: {t3 - t2:.4f}s")
    print(f"Total: {(t1 - t0) + (t3 - t2):.4f}s")
    mm.finalize()


if __name__ == "__main__":
    test_performance_occ()
