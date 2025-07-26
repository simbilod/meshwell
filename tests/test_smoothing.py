from __future__ import annotations

from pathlib import Path
import shapely
import gmsh
from functools import partial
from meshwell.prism import Prism
from meshwell.gmsh_entity import GMSH_entity
from meshwell.resolution import ConstantInField
from meshwell.cad import cad
from meshwell.mesh import mesh

from meshwell.utils import compare_gmsh_files


def test_smoothing():
    polygon1 = shapely.Polygon(
        [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
        holes=([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]],),
    )
    polygon2 = shapely.Polygon([[-1, -1], [-2, -1], [-2, -2], [-1, -2], [-1, -1]])
    polygon = shapely.MultiPolygon([polygon1, polygon2])

    buffers = {0.0: 0.0, 1.0: -0.1}

    poly3D = Prism(
        polygons=polygon,
        buffers=buffers,
        physical_name="first_entity",
        mesh_order=1,
    )

    gmsh_entity = GMSH_entity(
        gmsh_partial_function=partial(
            gmsh.model.occ.addSphere, xc=0, yc=0, zc=0, radius=1
        ),
        physical_name="second_entity",
        mesh_order=2,
    )

    entities_list = [poly3D, gmsh_entity]

    # Split into CAD and mesh operations
    cad(
        entities_list=entities_list,
        output_file="mesh3D_smoothing.xao",
    )

    mesh(
        dim=3,
        input_cad_file="mesh3D_smoothing.xao",
        output_mesh_file="mesh3D_smoothing.msh",
        resolution_specs={
            "first_entity": [ConstantInField(resolution=0.5, apply_to="volumes")],
            "second_entity": [],  # No specific resolution for sphere
        },
        optimization_flags=[("HighOrderElastic", 5), ("HighOrderFastCurving", 5)],
        verbosity=10,
        n_threads=1,
        default_characteristic_length=10,
    )

    compare_gmsh_files(Path("mesh3D_smoothing.msh"))


if __name__ == "__main__":
    test_smoothing()
