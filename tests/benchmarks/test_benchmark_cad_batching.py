import time

import shapely

from meshwell.cad_occ import cad_occ
from meshwell.polysurface import PolySurface


def create_overlapping_polysurfaces(n=50):
    entities = []
    for i in range(n):
        poly = shapely.box(float(i), float(i), float(i + 2), float(i + 2))
        entities.append(
            PolySurface(polygons=poly, physical_name=f"surf_{i}", mesh_order=i)
        )
    return entities


def test_benchmark_cad_occ_scaling():
    entities = create_overlapping_polysurfaces(50)
    start = time.time()
    cad_occ(entities)
    end = time.time()
    print(f"\nCAD OCC took {end - start:.4f}s for 50 entities")
