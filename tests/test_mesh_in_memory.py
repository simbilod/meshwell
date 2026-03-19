import shapely
import pytest
from meshwell.polysurface import PolySurface
from meshwell.mesh import mesh
from meshwell.model import ModelManager
from meshwell.cad_gmsh import CAD
from pathlib import Path

def test_mesh_in_memory():
    """Test meshing directly from an in-memory ModelManager."""
    # 1. Create entities
    poly = shapely.box(0.0, 0.0, 1.0, 1.0)
    surf = PolySurface(polygons=poly, physical_name="surf")
    
    # 2. Instantiate in GMSH directly (GMSH backend)
    mm = ModelManager()
    cad_proc = CAD(model=mm)
    cad_proc.process_entities([surf])
    
    # 3. Mesh from MM
    output_msh = Path("test_in_memory.msh")
    
    # This should fail if mesh() requires input_file
    try:
        mesh_obj = mesh(
            dim=2,
            input_file=None,
            output_file=output_msh,
            model=mm,
            default_characteristic_length=0.1
        )
        assert mesh_obj is not None
    finally:
        if output_msh.exists():
            output_msh.unlink()
        mm.finalize()
