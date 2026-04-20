from pathlib import Path

import shapely

from meshwell.cad_occ import cad_occ
from meshwell.mesh import mesh
from meshwell.model import ModelManager
from meshwell.occ_xao_writer import inject_occ_entities_into_gmsh
from meshwell.polysurface import PolySurface


def test_mesh_in_memory():
    """Mesh directly from an in-memory ModelManager populated by the OCC bridge."""
    poly = shapely.box(0.0, 0.0, 1.0, 1.0)
    surf = PolySurface(polygons=poly, physical_name="surf")

    # OCC fragment, then inject into a shared ModelManager
    occ_entities = cad_occ([surf])
    mm = ModelManager()
    inject_occ_entities_into_gmsh(occ_entities, model_manager=mm)

    output_msh = Path("test_in_memory.msh")
    try:
        mesh_obj = mesh(
            dim=2,
            input_file=None,
            output_file=output_msh,
            model=mm,
            default_characteristic_length=0.1,
        )
        assert mesh_obj is not None
    finally:
        if output_msh.exists():
            output_msh.unlink()
        mm.finalize()
