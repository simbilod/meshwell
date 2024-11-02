from __future__ import annotations

import shapely
from meshwell.polysurface import PolySurface
from meshwell.model import Model
from pathlib import Path
import pytest


# fmt: off
@pytest.mark.parametrize("config", ["mesh_msh.msh",
                                    "mesh_stp.stp",
                                    "mesh_msh.msh2",
                                    "mesh_msh.step",
                                    Path("mesh_msh.msh"),
                                    Path("mesh_stp.stp"),
                                    Path("mesh_msh.msh2"),
                                    Path("mesh_msh.step"),
                                    ]
                                )
def test_msh(config):
    polygon1 = shapely.Polygon(
        [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
        holes=([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]],),
    )
    polygon2 = shapely.Polygon([[-1, -1], [-2, -1], [-2, -2], [-1, -2], [-1, -1]])
    polygon = shapely.MultiPolygon([polygon1, polygon2])

    model = Model(n_threads=1)
    poly2D = PolySurface(
        polygons=polygon,
        model=model,
        physical_name="first_entity",
        mesh_order=1,
    )

    entities_list = [poly2D]

    model.mesh(
        entities_list=entities_list,
        default_characteristic_length=0.5,
        verbosity=False,
        filename=config),

    pass
