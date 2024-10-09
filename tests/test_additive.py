from __future__ import annotations

import shapely
from meshwell.prism import Prism
from meshwell.model import Model
from meshwell.resolution import ResolutionSpec


def test_mesh_additive_3D():
    polygon = shapely.Polygon(
        [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
        holes=([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]],),
    )

    buffers = {0.0: 0.0, 1.0: -0.1}

    model = Model(n_threads=1)
    big_prism = Prism(
        polygons=polygon.buffer(5),
        buffers=buffers,
        model=model,
        physical_name="big_prism",
        mesh_order=1,
    )
    medium_prism = Prism(
        polygons=polygon.buffer(2),
        buffers=buffers,
        model=model,
        physical_name="medium_prism",
        resolutions=[ResolutionSpec(resolution_volumes=0.5)],
        additive=True,
        mesh_order=1,
    )
    small_prism = Prism(
        polygons=polygon,
        buffers=buffers,
        model=model,
        physical_name="small_prism",
        mesh_order=1,
    )
    entities_list = [big_prism, medium_prism, small_prism]

    model.mesh(
        entities_list=entities_list,
        default_characteristic_length=1,
        verbosity=False,
        filename="mesh3D_additive.msh",
    )

    # compare_meshes(Path("mesh3D.msh"))


if __name__ == "__main__":
    test_mesh_additive_3D()
