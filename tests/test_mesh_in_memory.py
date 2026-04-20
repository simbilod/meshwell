import tempfile
from pathlib import Path

import shapely

from meshwell.cad_occ import cad_occ
from meshwell.mesh import mesh
from meshwell.occ_xao_writer import write_xao
from meshwell.polysurface import PolySurface


def test_mesh_in_memory():
    """Mesh directly from a transient XAO (no permanent CAD file on disk)."""
    poly = shapely.box(0.0, 0.0, 1.0, 1.0)
    surf = PolySurface(polygons=poly, physical_name="surf")

    with tempfile.TemporaryDirectory() as tmpdir:
        xao = Path(tmpdir) / "cad.xao"
        write_xao(cad_occ([surf]), xao)

        output_msh = Path(tmpdir) / "cad.msh"
        mesh_obj = mesh(
            dim=2,
            input_file=xao,
            output_file=output_msh,
            default_characteristic_length=0.1,
        )
    assert mesh_obj is not None
