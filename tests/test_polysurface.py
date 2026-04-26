from __future__ import annotations

import meshio
import shapely

from meshwell.cad_occ import cad_occ
from meshwell.mesh import mesh
from meshwell.occ_xao_writer import write_xao
from meshwell.polysurface import PolySurface


def test_polysurface(tmp_path):
    polygon1 = shapely.Polygon(
        [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
        holes=([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]],),
    )
    polygon2 = shapely.Polygon([[-1, -1], [-2, -1], [-2, -2], [-1, -2], [-1, -1]])
    polygon = shapely.MultiPolygon([polygon1, polygon2])

    polysurface_obj = PolySurface(polygons=polygon, physical_name="polysurface")
    write_xao(cad_occ([polysurface_obj]), str(tmp_path / "test_polysurface.xao"))
    msh_path = tmp_path / "test_polysurface.msh"
    mesh(
        input_file=str(tmp_path / "test_polysurface.xao"),
        output_file=str(msh_path),
        dim=2,
        default_characteristic_length=0.5,
        n_threads=1,
    )
    # Verify the mesh was generated and contains triangles for a 2D mesh
    m = meshio.read(str(msh_path))
    assert any(c.type == "triangle" and len(c.data) > 0 for c in m.cells)
    assert "polysurface" in m.cell_sets


def test_coinciding_polysurface(tmp_path):
    width = 1
    height = 1

    core = shapely.geometry.box(-width / 2, -0.2, +width / 2, height)
    cladding = shapely.geometry.box(-width * 2, 0, width * 2, height * 3)
    buried_oxide = shapely.geometry.box(-width * 2, -height * 2, width * 2, 0)

    core_surface = PolySurface(polygons=core, physical_name="core", mesh_order=0)
    cladding_surface = PolySurface(
        polygons=cladding, physical_name="cladding", mesh_order=1
    )
    buried_oxide_surface = PolySurface(
        polygons=buried_oxide, physical_name="buried_oxide", mesh_order=2
    )

    entities = [core_surface, cladding_surface, buried_oxide_surface]

    write_xao(
        cad_occ(
            entities,
            n_threads=1,
        ),
        str(tmp_path / "test_polysurface_coinciding.xao"),
    )
    msh_path = tmp_path / "test_polysurface_coinciding.msh"
    mesh(
        input_file=str(tmp_path / "test_polysurface_coinciding.xao"),
        output_file=str(msh_path),
        dim=2,
        default_characteristic_length=0.5,
        n_threads=1,
    )
    # Verify all three regions were meshed
    m = meshio.read(str(msh_path))
    assert any(c.type == "triangle" and len(c.data) > 0 for c in m.cells)
    for region in ("core", "cladding", "buried_oxide"):
        assert region in m.cell_sets, f"missing region: {region}"


if __name__ == "__main__":
    test_coinciding_polysurface()
