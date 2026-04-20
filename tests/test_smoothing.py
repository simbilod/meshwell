from __future__ import annotations

import shapely

from meshwell.cad_occ import cad_occ
from meshwell.mesh import mesh
from meshwell.occ_entity import OCC_entity
from meshwell.occ_xao_writer import write_xao
from meshwell.polyprism import PolyPrism
from meshwell.resolution import ConstantInField
from tests.test_occ_helpers import _occ_sphere


def test_smoothing():
    polygon1 = shapely.Polygon(
        [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
        holes=([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]],),
    )
    polygon2 = shapely.Polygon([[-1, -1], [-2, -1], [-2, -2], [-1, -2], [-1, -1]])
    polygon = shapely.MultiPolygon([polygon1, polygon2])

    poly3D = PolyPrism(
        polygons=polygon,
        buffers={0.0: 0.0, 1.0: -0.1},
        physical_name="first_entity",
        mesh_order=1,
    )

    sphere_entity = OCC_entity(
        occ_function=_occ_sphere(xc=0, yc=0, zc=0, radius=1),
        physical_name="second_entity",
        mesh_order=2,
        dimension=3,
    )

    write_xao(cad_occ([poly3D, sphere_entity]), "mesh3D_smoothing.xao")
    mesh(
        dim=3,
        input_file="mesh3D_smoothing.xao",
        output_file="mesh3D_smoothing.msh",
        resolution_specs={
            "first_entity": [ConstantInField(resolution=0.5, apply_to="volumes")],
            "second_entity": [],
        },
        optimization_flags=[("HighOrderElastic", 5), ("HighOrderFastCurving", 5)],
        verbosity=10,
        n_threads=1,
        default_characteristic_length=0.7,
        mesh_element_order=1,
    )


if __name__ == "__main__":
    test_smoothing()
