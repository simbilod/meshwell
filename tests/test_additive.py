from __future__ import annotations

import shapely
from meshwell.prism import Prism
from meshwell.model import Model
from meshwell.resolution import ResolutionSpec
from meshwell.utils import compare_meshes
from pathlib import Path
from shapely.ops import clip_by_rect


def test_mesh_additive_3D():
    polygon = shapely.Polygon(
        [[-5, -5], [5, -5], [5, 5], [-5, 5], [-5, -5]],
    )

    buffers = {0.0: 0.0, 1.0: -0.1}

    model = Model(n_threads=1)
    big_prism = Prism(
        polygons=polygon.buffer(10, join_style="mitre"),
        buffers=buffers,
        model=model,
        physical_name="big_prism",
        mesh_order=3,
        resolutions=[ResolutionSpec(resolution_volumes=10)],
    )
    medium_prism = Prism(
        polygons=clip_by_rect(
            polygon.buffer(5, join_style="mitre"), xmin=0, ymin=-100, ymax=100, xmax=100
        ),
        buffers=buffers,
        model=model,
        physical_name="medium_prism",
        resolutions=[ResolutionSpec(resolution_volumes=1)],
        additive=True,
        mesh_order=2,
    )
    small_prism = Prism(
        polygons=polygon,
        buffers=buffers,
        model=model,
        physical_name="small_prism",
        mesh_order=1,
        resolutions=[ResolutionSpec(resolution_volumes=10)],
    )
    entities_list = [big_prism, medium_prism, small_prism]
    # entities_list = [medium_prism]

    model.mesh(
        entities_list=entities_list,
        default_characteristic_length=10,
        verbosity=False,
        filename="mesh3D_additive.msh",
    )

    compare_meshes(Path("mesh3D_additive.msh"))


if __name__ == "__main__":
    test_mesh_additive_3D()
