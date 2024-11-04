from __future__ import annotations

import shapely
from meshwell.polysurface import PolySurface
from meshwell.model import Model
from meshwell.prism import Prism
from meshwell.resolution import ConstantInField, ThresholdField, ExponentialField
from meshwell.utils import compare_meshes
from pathlib import Path
import pytest


def test_2D_resolution():
    large_rect = 20
    small_rect = 5

    polygon1 = shapely.Polygon(
        [[0, 0], [large_rect, 0], [large_rect, large_rect], [0, large_rect], [0, 0]],
    )
    polygon2 = shapely.Polygon(
        [
            [small_rect, small_rect],
            [large_rect - small_rect, small_rect],
            [large_rect - small_rect, large_rect - small_rect],
            [small_rect, 6],
            [small_rect, small_rect],
        ],
    )

    model = Model(n_threads=1)  # 1 thread for deterministic mesh
    poly_obj1 = PolySurface(
        polygons=polygon1,
        model=model,
        mesh_order=2,
        physical_name="outer",
        resolutions=[ConstantInField(apply_to="surfaces", resolution=0.5)],
    )
    poly_obj2 = PolySurface(
        polygons=polygon2,
        model=model,
        mesh_order=1,
        physical_name="inner",
        resolutions=[
            ThresholdField(sizemin=0.1, distmax=2, sizemax=1, apply_to="curves")
        ],
    )

    entities_list = [poly_obj1, poly_obj2]

    model.mesh(
        entities_list=entities_list,
        default_characteristic_length=1,
        verbosity=0,
        filename="mesh_test_2D_resolution.msh",
    )

    compare_meshes(Path("mesh_test_2D_resolution.msh"))


def test_3D_resolution():
    polygon1 = shapely.Polygon(
        [[0, 0], [9, 0], [9, 9], [0, 9], [0, 0]],
    )
    polygon2 = shapely.Polygon(
        [[3, 3], [6, 3], [6, 6], [3, 6], [3, 3]],
    )

    buffers = {0.0: 0.0, 3: 0.0}

    model = Model(n_threads=1)
    prism_obj1 = Prism(
        polygons=polygon1,
        buffers=buffers,
        model=model,
        mesh_order=2,
        physical_name="outer",
        resolutions=[ConstantInField(resolution=1, apply_to="volumes")],
    )
    prism_obj2 = Prism(
        polygons=polygon2,
        buffers=buffers,
        model=model,
        mesh_order=1,
        physical_name="inner",
        resolutions=[
            ConstantInField(resolution=1, apply_to="volumes"),
            ThresholdField(sizemin=0.2, distmax=1, sizemax=1, apply_to="surfaces"),
        ],
    )

    entities_list = [prism_obj1, prism_obj2]

    model.mesh(
        entities_list=entities_list,
        default_characteristic_length=1,
        verbosity=0,
        filename="mesh_test_3D_resolution.msh",
    )

    compare_meshes(Path("mesh_test_3D_resolution.msh"))


def test_exponential_field():
    large_rect = 40
    small_rect = 5

    polygon1 = shapely.Polygon(
        [[0, 0], [large_rect, 0], [large_rect, large_rect], [0, large_rect], [0, 0]],
    )
    polygon2 = shapely.Polygon(
        [
            [small_rect, small_rect],
            [large_rect - small_rect, small_rect],
            [large_rect - small_rect, large_rect - small_rect],
            [small_rect, 6],
            [small_rect, small_rect],
        ],
    )

    model = Model(n_threads=1)  # 1 thread for deterministic mesh
    poly_obj1 = PolySurface(
        polygons=polygon1,
        model=model,
        mesh_order=2,
        physical_name="outer",
        resolutions=[ConstantInField(apply_to="surfaces", resolution=5)],
    )
    poly_obj2 = PolySurface(
        polygons=polygon2,
        model=model,
        mesh_order=1,
        physical_name="inner",
        resolutions=[
            ExponentialField(
                growth_factor=2,
                sizemin=0.3,
                max_samplings=200,
                apply_to="curves",
                lengthscale=2,
            )
        ],
    )

    entities_list = [poly_obj1, poly_obj2]

    model.mesh(
        entities_list=entities_list,
        default_characteristic_length=5,
        verbosity=0,
        filename="mesh_test_2D_exponential.msh",
    )


@pytest.mark.parametrize(
    "field",
    [
        ConstantInField(apply_to="surfaces", resolution=1),
        ExponentialField(
            growth_factor=2,
            sizemin=0.3,
            max_samplings=200,
            apply_to="curves",
            lengthscale=2,
        ),
        ThresholdField(
            sizemin=1,
            sizemax=5,
            distmin=0,
            distmax=5,
            apply_to="curves",
        ),
    ],
)
def test_refine(field):
    large_rect = 10

    polygon1 = shapely.Polygon(
        [[0, 0], [large_rect, 0], [large_rect, large_rect], [0, large_rect], [0, 0]],
    )

    points = []
    for factor in [0.5, 1, 2]:
        model = Model(n_threads=1)  # 1 thread for deterministic mesh
        poly_obj2 = PolySurface(
            polygons=polygon1,
            model=model,
            mesh_order=1,
            physical_name="inner",
            resolutions=[field.refine(factor)],
        )

        entities_list = [poly_obj2]

        output = model.mesh(
            entities_list=entities_list,
            default_characteristic_length=5,
            verbosity=0,
            filename="mesh_test_2D_exponential.msh",
        )

        points.append(len(output.points))

    assert points[0] > points[1] > points[2]


if __name__ == "__main__":
    test_refine(ConstantInField(apply_to="surfaces", resolution=1))
    # test_3D_resolution()
