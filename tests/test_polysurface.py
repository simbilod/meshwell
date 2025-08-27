from __future__ import annotations

import shapely
from meshwell.polysurface import PolySurface
from meshwell.cad import cad
from meshwell.mesh import mesh
from meshwell.utils import compare_gmsh_files
from pathlib import Path


def test_polysurface():
    polygon1 = shapely.Polygon(
        [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
        holes=([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]],),
    )
    polygon2 = shapely.Polygon([[-1, -1], [-2, -1], [-2, -2], [-1, -2], [-1, -1]])
    polygon = shapely.MultiPolygon([polygon1, polygon2])

    polysurface_obj = PolySurface(polygons=polygon, physical_name="polysurface")
    cad(entities_list=[polysurface_obj], output_file="test_polysurface")
    mesh(
        input_file="test_polysurface.xao",
        output_file="test_polysurface.msh",
        dim=2,
        default_characteristic_length=0.5,
        n_threads=1,
    )
    compare_gmsh_files(Path("test_polysurface.msh"))


def test_coinciding_polysurface():
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

    cad(entities_list=entities, output_file="test_polysurface_coinciding.xao")
    mesh(
        input_file="test_polysurface_coinciding.xao",
        output_file="test_polysurface_coinciding.msh",
        dim=2,
        default_characteristic_length=0.5,
        n_threads=1,
    )
    compare_gmsh_files(Path("test_polysurface_coinciding.msh"))


if __name__ == "__main__":
    test_coinciding_polysurface()
