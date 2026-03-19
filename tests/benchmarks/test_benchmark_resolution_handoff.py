import time
from pathlib import Path

import shapely

from meshwell.cad_gmsh import cad
from meshwell.mesh import Mesh
from meshwell.polysurface import PolySurface
from meshwell.resolution import ConstantInField


def test_benchmark_resolution_handoff():
    n = 50
    entities = []
    physical_names = [f"surf_{i}" for i in range(n)]

    # Create side-by-side squares
    for i in range(n):
        poly = shapely.box(float(i), 0.0, float(i + 1), 1.0)
        entities.append(PolySurface(polygons=poly, physical_name=physical_names[i]))

    # Each entity shares with ALL others
    # Use curves to avoid boundary check error
    resolution_specs = {
        name: [
            ConstantInField(resolution=0.1, apply_to="curves", sharing=physical_names)
        ]
        for name in physical_names
    }

    # 1. Generate CAD (to get XAO)
    cad_file = Path("benchmark_handoff.xao")
    cad(entities, cad_file)

    # 2. Measure Mesh refinement
    m = Mesh()
    m.load_xao_file(cad_file)
    m._initialize_model()

    start = time.time()
    m._apply_entity_refinement(
        boundary_delimiter="None", resolution_specs=resolution_specs
    )
    end = time.time()
    print(f"Resolution handoff (sharing with {n} entities) took {end - start:.4f}s")

    if cad_file.exists():
        cad_file.unlink()
