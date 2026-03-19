import shapely
import time
import pytest
from meshwell.polysurface import PolySurface
from meshwell.cad_occ import cad_occ

def create_complex_polysurfaces(n=50, n_holes=20):
    entities = []
    for i in range(n):
        # Base square
        poly = shapely.box(float(i*2), 0.0, float(i*2 + 1), 1.0)
        # Add many small holes
        holes = []
        for j in range(n_holes):
            # Create a small hole inside the box
            h_x = i*2 + 0.1 + (j % 5) * 0.15
            h_y = 0.1 + (j // 5) * 0.2
            hole = shapely.box(h_x, h_y, h_x + 0.05, h_y + 0.05)
            holes.append(hole)
        
        # Use MultiPolygon with holes
        poly_with_holes = shapely.Polygon(poly.exterior, [h.exterior for h in holes])
        entities.append(PolySurface(polygons=poly_with_holes, physical_name=f"complex_{i}", mesh_order=i))
    return entities

def test_benchmark_cad_parallel_instantiation():
    entities = create_complex_polysurfaces(50, 20)
    start = time.time()
    cad_occ(entities, n_threads=1)
    end = time.time()
    print(f"CAD OCC (1 thread) took {end - start:.4f}s")

    start = time.time()
    cad_occ(entities, n_threads=4)
    end = time.time()
    print(f"CAD OCC (4 threads) took {end - start:.4f}s")
