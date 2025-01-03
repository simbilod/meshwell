from __future__ import annotations

import shapely
from meshwell.polysurface import PolySurface
from meshwell.model import Model
from meshwell.prism import Prism
from meshwell.resolution import ConstantInField, ThresholdField, ExponentialField
from meshwell.utils import compare_meshes
from pathlib import Path
import pytest
import numpy as np


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


# FIXME: add regression
@pytest.mark.parametrize(
    ("apply_to", "min_mass", "max_mass"),
    [
        ("volumes", 5**3, np.inf),
        ("volumes", 0, 5**3),
        ("surfaces", 5**2, np.inf),
        ("surfaces", 0, 5**2),
        ("curves", 5, np.inf),
        ("curves", 0, 5),
        ("points", 5, np.inf),
        ("points", 0, 5),
    ],
)
def test_filter(apply_to, min_mass, max_mass):
    large_rect = 6
    small_rect = 4

    buffers1 = {-6: 0, 6: 0}
    buffers2 = {-4: 0, 4: 0}

    polygon1 = shapely.Polygon(
        [
            [-large_rect / 2, -large_rect / 2],
            [large_rect / 2, -large_rect / 2],
            [large_rect / 2, large_rect / 2],
            [-large_rect / 2, large_rect / 2],
            [-large_rect / 2, -large_rect / 2],
        ],
    )
    polygon2 = shapely.Polygon(
        [
            [-small_rect / 2, -small_rect / 2],
            [small_rect / 2, -small_rect / 2],
            [small_rect / 2, small_rect / 2],
            [-small_rect / 2, small_rect / 2],
            [-small_rect / 2, -small_rect / 2],
        ],
    )

    model = Model(n_threads=1)  # 1 thread for deterministic mesh
    prism1 = Prism(
        polygons=polygon1,
        buffers=buffers1,
        model=model,
        mesh_order=2,
        physical_name="outer",
        resolutions=[
            ConstantInField(
                apply_to=apply_to, resolution=0.5, min_mass=min_mass, max_mass=max_mass
            )
        ],
    )
    prism2 = Prism(
        polygons=polygon2,
        buffers=buffers2,
        model=model,
        mesh_order=1,
        physical_name="inner",
        resolutions=[
            ThresholdField(
                apply_to=apply_to,
                sizemin=0.5,
                min_mass=min_mass,
                max_mass=max_mass,
                sizemax=10,
                distmin=0,
                distmax=10,
            )
        ],
    )

    entities_list = [prism1, prism2]

    model.mesh(
        entities_list=entities_list,
        default_characteristic_length=5,
        verbosity=0,
        filename="mesh_filter.msh",
    )


def test_filter_shared():
    large_rect = 8
    small_rect = 4

    buffers1 = {0: 0, 8: 0}
    buffers2 = {0: 0, 4: 0}
    buffers3 = {0: 0, 8: 0}

    polygon1 = shapely.Polygon(
        [
            [0, 0],
            [large_rect, 0],
            [large_rect, large_rect],
            [0, large_rect],
            [0, 0],
        ],
    )
    polygon2 = shapely.Polygon(
        [
            [0, 0],
            [small_rect, 0],
            [small_rect, small_rect],
            [0, small_rect],
            [0, 0],
        ],
    )
    polygon3 = shapely.affinity.translate(polygon2, xoff=small_rect, yoff=0.0, zoff=0.0)

    model = Model(n_threads=1)  # 1 thread for deterministic mesh
    prism1 = Prism(
        polygons=polygon1,
        buffers=buffers1,
        model=model,
        mesh_order=3,
        physical_name="outer",
    )
    prism2 = Prism(
        polygons=polygon2,
        buffers=buffers2,
        model=model,
        mesh_order=1,
        physical_name="bot",
        resolutions=[
            ExponentialField(
                apply_to="surfaces",
                sizemin=0.1,
                lengthscale=0.2,
                growth_factor=2,
                sharing=["outer", "None"],
            )
        ],
    )
    prism3 = Prism(
        polygons=polygon3,
        buffers=buffers3,
        model=model,
        mesh_order=2,
        physical_name="right",
    )

    entities_list = [prism1, prism2, prism3]

    model.mesh(
        entities_list=entities_list,
        default_characteristic_length=5,
        verbosity=0,
        filename="mesh_filter_shared.msh",
    )


def test_filter_not_shared():
    large_rect = 8
    small_rect = 4

    buffers1 = {0: 0, 8: 0}
    buffers2 = {0: 0, 4: 0}
    buffers3 = {0: 0, 8: 0}

    polygon1 = shapely.Polygon(
        [
            [0, 0],
            [large_rect, 0],
            [large_rect, large_rect],
            [0, large_rect],
            [0, 0],
        ],
    )
    polygon2 = shapely.Polygon(
        [
            [0, 0],
            [small_rect, 0],
            [small_rect, small_rect],
            [0, small_rect],
            [0, 0],
        ],
    )
    polygon3 = shapely.affinity.translate(polygon2, xoff=small_rect, yoff=0.0, zoff=0.0)

    model = Model(n_threads=1)  # 1 thread for deterministic mesh
    prism1 = Prism(
        polygons=polygon1,
        buffers=buffers1,
        model=model,
        mesh_order=3,
        physical_name="outer",
    )
    prism2 = Prism(
        polygons=polygon2,
        buffers=buffers2,
        model=model,
        mesh_order=1,
        physical_name="bot",
        resolutions=[
            ExponentialField(
                apply_to="surfaces",
                sizemin=0.1,
                lengthscale=0.2,
                growth_factor=2,
                not_sharing=["outer", "None"],
            )
        ],
    )
    prism3 = Prism(
        polygons=polygon3,
        buffers=buffers3,
        model=model,
        mesh_order=2,
        physical_name="right",
    )

    entities_list = [prism1, prism2, prism3]

    model.mesh(
        entities_list=entities_list,
        default_characteristic_length=5,
        verbosity=0,
        filename="mesh_filter_shared.msh",
    )


@pytest.mark.parametrize(
    ("restrict_to",),
    [
        (None,),
        (["inner_left"],),
        (["inner_right"],),
        (["outer", "inner_right"],),
    ],
)
def test_restrict(restrict_to):
    large_rect = 20
    small_rect = 5

    polygon1 = shapely.Polygon(
        [[0, 0], [large_rect, 0], [large_rect, large_rect], [0, large_rect], [0, 0]],
    )
    polygon2 = shapely.Polygon(
        [
            [large_rect / 2 - small_rect / 2, large_rect / 2 - small_rect / 2],
            [large_rect / 2 + small_rect / 2, large_rect / 2 - small_rect / 2],
            [large_rect / 2 + small_rect / 2, large_rect / 2 + small_rect / 2],
            [large_rect / 2 - small_rect / 2, large_rect / 2 + small_rect / 2],
            [large_rect / 2 - small_rect / 2, large_rect / 2 - small_rect / 2],
        ],
    )

    model = Model(n_threads=1)
    poly_outer = PolySurface(
        polygons=polygon1,
        model=model,
        mesh_order=2,
        physical_name="outer",
    )

    if restrict_to is None:
        restrict_to = None
    else:
        restrict_to = restrict_to
    poly_left = PolySurface(
        polygons=shapely.affinity.translate(polygon2, xoff=-3.1),
        model=model,
        mesh_order=1,
        physical_name="inner_left",
        resolutions=[
            ThresholdField(
                sizemin=0.05,
                distmax=10,
                sizemax=1,
                apply_to="curves",
                restrict_to=restrict_to,
            )
        ],
    )
    poly_right = PolySurface(
        polygons=shapely.affinity.translate(polygon2, xoff=3.1),
        model=model,
        mesh_order=1,
        physical_name="inner_right",
    )
    entities_list = [poly_outer, poly_left, poly_right]
    model.mesh(
        entities_list=entities_list,
        default_characteristic_length=1,
        verbosity=0,
        filename=f"mesh_restrict_{restrict_to}.msh",
    )
    compare_meshes(Path(f"mesh_restrict_{restrict_to}.msh"))


def test_interface_thresholds():
    """Test different threshold fields on each side of an interface."""
    # Create outer square
    outer_square = shapely.Polygon([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]])

    # Create inner square
    inner_square = shapely.Polygon([[4, 4], [6, 4], [6, 6], [4, 6], [4, 4]])

    model = Model(n_threads=1)

    # Create outer polysurface with coarser exponential field
    poly_outer = PolySurface(
        polygons=outer_square,
        model=model,
        mesh_order=2,
        physical_name="outer",
        resolutions=[
            ExponentialField(
                sizemin=0.05,
                lengthscale=2.0,
                growth_factor=3,
                apply_to="curves",
                not_sharing=["None"],
                restrict_to=["outer"],
            )
        ],
    )

    # Create inner polysurface with finer exponential field
    poly_inner = PolySurface(
        polygons=inner_square,
        model=model,
        mesh_order=1,
        physical_name="inner",
        resolutions=[
            ExponentialField(
                sizemin=0.05,
                lengthscale=0.5,  # Shorter transition distance
                growth_factor=3.0,
                apply_to="curves",
                restrict_to=["inner"],
            )
        ],
    )

    # Mesh the model
    model.mesh(
        entities_list=[poly_outer, poly_inner],
        default_characteristic_length=2.0,  # Increased default size
        verbosity=0,
        filename="mesh_interface_thresholds.msh",
    )

    compare_meshes(Path("mesh_interface_thresholds.msh"))


if __name__ == "__main__":
    test_interface_thresholds()
