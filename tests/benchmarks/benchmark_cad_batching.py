import shapely
import time
import pytest
from meshwell.polysurface import PolySurface
from meshwell.cad_occ import cad_occ
from meshwell.cad_gmsh import CAD as CAD_GMSH

def create_overlapping_polysurfaces(n=50):
    entities = []
    for i in range(n):
        # Create squares that overlap sequentially: [i, i] to [i+2, i+2]
        poly = shapely.box(float(i), float(i), float(i + 2), float(i + 2))
        entities.append(PolySurface(polygons=poly, physical_name=f"surf_{i}", mesh_order=i))
    return entities

def test_benchmark_cad_occ_scaling():
    entities = create_overlapping_polysurfaces(50)
    start = time.time()
    cad_occ(entities)
    end = time.time()
    print(f"\nCAD OCC took {end - start:.4f}s for 50 entities")

def test_benchmark_cad_gmsh_scaling():
    entities = create_overlapping_polysurfaces(50)
    processor = CAD_GMSH()
    start = time.time()
    # Mocking progress bars if necessary
    processor.process_entities(entities)
    end = time.time()
    print(f"\nCAD GMSH took {end - start:.4f}s for 50 entities")
