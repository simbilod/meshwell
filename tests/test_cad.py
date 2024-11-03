from __future__ import annotations

import shapely
from meshwell.prism import Prism
from meshwell.model import Model
from meshwell.resolution import ConstantInField
import time


def test_cad_load():
    buffers = {0.0: 0.0, 1.0: -0.1}

    model = Model(n_threads=1)

    entities_list = []

    for stagger in range(10):
        entities_list.append(
            Prism(
                polygons=shapely.Polygon(
                    [
                        [-5 + stagger, -5 + stagger],
                        [5 + stagger, -5 + stagger],
                        [5 + stagger, 5 + stagger],
                        [-5 + stagger, 5 + stagger],
                        [-5 + stagger, -5 + stagger],
                    ],
                ),
                buffers=buffers,
                model=model,
                physical_name=f"prism_{stagger}",
                mesh_order=stagger,
                resolutions=[
                    ConstantInField(resolution=10, apply_to="volumes"),
                ],
            )
        )

    start_time = time.time()
    model.mesh(
        entities_list=entities_list,
        filename_cad="test_cad.xao",
        load_cad=False,
        filename="test_cad.msh",
    )
    end_time = time.time()
    init_time = end_time - start_time

    model2 = Model(n_threads=1)
    start_time = time.time()
    model2.mesh(
        entities_list=entities_list,
        filename_cad="test_cad.xao",
        load_cad=True,
        filename="test_load_cad.msh",
    )
    end_time = time.time()
    load_time = end_time - start_time

    assert load_time < init_time


if __name__ == "__main__":
    test_cad_load()
