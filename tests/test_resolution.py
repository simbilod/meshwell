from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import shapely

from meshwell.cad import cad
from meshwell.mesh import mesh
from meshwell.polyprism import PolyPrism
from meshwell.polysurface import PolySurface
from meshwell.resolution import ConstantInField, ExponentialField, ThresholdField
from meshwell.utils import compare_gmsh_files


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

    poly_obj1 = PolySurface(
        polygons=polygon1,
        mesh_order=2,
        physical_name="outer",
    )
    poly_obj2 = PolySurface(
        polygons=polygon2,
        mesh_order=1,
        physical_name="inner",
    )

    entities_list = [poly_obj1, poly_obj2]

    cad(
        entities_list=entities_list,
        output_file="test_2D_resolution",
    )

    mesh(
        dim=2,
        input_file="test_2D_resolution.xao",
        output_file="test_2D_resolution.msh",
        resolution_specs={
            "inner": [
                ThresholdField(sizemin=0.1, distmax=2, sizemax=1, apply_to="curves")
            ],
            "outer": [ConstantInField(apply_to="surfaces", resolution=0.5)],
        },
        n_threads=1,
        default_characteristic_length=100,
    )

    compare_gmsh_files(Path("test_2D_resolution.msh"))


def test_3D_resolution():
    polygon1 = shapely.Polygon(
        [[0, 0], [9, 0], [9, 9], [0, 9], [0, 0]],
    )
    polygon2 = shapely.Polygon(
        [[3, 3], [6, 3], [6, 6], [3, 6], [3, 3]],
    )

    buffers = {0.0: 0.0, 3: 0.0}

    prism_obj1 = PolyPrism(
        polygons=polygon1,
        buffers=buffers,
        mesh_order=2,
        physical_name="outer",
    )
    prism_obj2 = PolyPrism(
        polygons=polygon2,
        buffers=buffers,
        mesh_order=1,
        physical_name="inner",
    )

    entities_list = [prism_obj1, prism_obj2]

    cad(
        entities_list=entities_list,
        output_file="test_3D_resolution.xao",
    )

    mesh(
        dim=3,
        input_file="test_3D_resolution.xao",
        output_file="test_3D_resolution.msh",
        resolution_specs={
            "inner": [
                ConstantInField(resolution=1, apply_to="volumes"),
                ThresholdField(sizemin=0.2, distmax=1, sizemax=1, apply_to="surfaces"),
            ],
            "outer": [ConstantInField(resolution=1, apply_to="volumes")],
        },
        n_threads=1,
        default_characteristic_length=100,
    )

    compare_gmsh_files(Path("test_3D_resolution.msh"))


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

    poly_obj1 = PolySurface(
        polygons=polygon1,
        mesh_order=2,
        physical_name="outer",
    )
    poly_obj2 = PolySurface(
        polygons=polygon2,
        mesh_order=1,
        physical_name="inner",
    )

    entities_list = [poly_obj1, poly_obj2]

    cad(
        entities_list=entities_list,
        output_file="test_exponential_field.xao",
    )

    mesh(
        dim=3,
        input_file="test_exponential_field.xao",
        output_file="test_exponential_field.msh",
        resolution_specs={
            "inner": [
                ExponentialField(
                    growth_factor=2,
                    sizemin=0.3,
                    max_samplings=200,
                    apply_to="curves",
                    lengthscale=2,
                )
            ],
            "outer": [ConstantInField(apply_to="surfaces", resolution=5)],
        },
        n_threads=1,
        default_characteristic_length=100,
    )

    compare_gmsh_files(Path("test_exponential_field.msh"))


@pytest.mark.parametrize(
    ("field", "label"),
    [
        (ConstantInField(apply_to="surfaces", resolution=1), "constant"),
        (
            ExponentialField(
                growth_factor=2,
                sizemin=0.3,
                max_samplings=200,
                apply_to="curves",
                lengthscale=2,
            ),
            "exponential",
        ),
        (
            ThresholdField(
                sizemin=1,
                sizemax=5,
                distmin=0,
                distmax=5,
                apply_to="curves",
            ),
            "threshold",
        ),
    ],
)
def test_refine(field, label):
    large_rect = 10

    polygon1 = shapely.Polygon(
        [[0, 0], [large_rect, 0], [large_rect, large_rect], [0, large_rect], [0, 0]],
    )

    poly_obj2 = PolySurface(
        polygons=polygon1,
        mesh_order=1,
        physical_name="inner",
    )

    cad(
        entities_list=[poly_obj2],
        output_file=f"test_refine_{label}.xao",
    )

    points = []

    for factor in [0.5, 1, 2]:
        poly_obj2 = PolySurface(
            polygons=polygon1,
            mesh_order=1,
            physical_name="inner",
        )
        output = mesh(
            dim=3,
            input_file=f"test_refine_{label}.xao",
            output_file=f"test_refine_{label}.msh",
            resolution_specs={
                "inner": [field.refine(factor)],
            },
            n_threads=1,
            default_characteristic_length=100,
        )

        points.append(len(output.points))

    print(points)

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
    label = f"{apply_to}_{min_mass}_{max_mass}"

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

    prism1 = PolyPrism(
        polygons=polygon1,
        buffers=buffers1,
        mesh_order=2,
        physical_name="outer",
    )
    prism2 = PolyPrism(
        polygons=polygon2,
        buffers=buffers2,
        mesh_order=1,
        physical_name="inner",
    )

    entities_list = [prism1, prism2]

    cad(
        entities_list=entities_list,
        output_file=f"test_filter_{label}.xao",
    )

    mesh(
        dim=3,
        input_file=f"test_filter_{label}.xao",
        output_file=f"test_filter_{label}.msh",
        resolution_specs={
            "inner": [
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
            "outer": [
                ConstantInField(
                    apply_to=apply_to,
                    resolution=0.5,
                    min_mass=min_mass,
                    max_mass=max_mass,
                )
            ],
        },
        n_threads=1,
        default_characteristic_length=100,
    )

    compare_gmsh_files(Path(f"test_filter_{label}.msh"))


@pytest.mark.parametrize(
    "restrict_to",
    [
        None,
        ["inner_left"],
        ["inner_right"],
        ["outer", "inner_right"],
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

    poly_outer = PolySurface(
        polygons=polygon1,
        mesh_order=2,
        physical_name="outer",
    )

    restrict_to = None if restrict_to is None else restrict_to
    poly_left = PolySurface(
        polygons=shapely.affinity.translate(polygon2, xoff=-3.1),
        mesh_order=1,
        physical_name="inner_left",
    )
    poly_right = PolySurface(
        polygons=shapely.affinity.translate(polygon2, xoff=3.1),
        mesh_order=1,
        physical_name="inner_right",
    )
    entities_list = [poly_outer, poly_left, poly_right]

    cad(
        entities_list=entities_list,
        output_file=f"test_restrict_{restrict_to}.xao",
    )

    mesh(
        dim=3,
        input_file=f"test_restrict_{restrict_to}.xao",
        output_file=f"test_restrict_{restrict_to}.msh",
        resolution_specs={
            "inner_left": [
                ThresholdField(
                    sizemin=0.05,
                    distmax=10,
                    sizemax=1,
                    apply_to="curves",
                    restrict_to=restrict_to,
                )
            ],
        },
        n_threads=1,
        default_characteristic_length=100,
    )

    compare_gmsh_files(Path(f"test_restrict_{restrict_to}.msh"))


def test_interface_thresholds():
    """Test different threshold fields on each side of an interface."""
    # Create outer square
    outer_square = shapely.Polygon([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]])

    # Create inner square
    inner_square = shapely.Polygon([[4, 4], [6, 4], [6, 6], [4, 6], [4, 4]])

    # Create outer polysurface with coarser exponential field
    poly_outer = PolySurface(
        polygons=outer_square,
        mesh_order=2,
        physical_name="outer",
    )

    # Create inner polysurface with finer exponential field
    poly_inner = PolySurface(
        polygons=inner_square,
        mesh_order=1,
        physical_name="inner",
    )

    entities_list = [poly_outer, poly_inner]

    cad(
        entities_list=entities_list,
        output_file="test_interface_thresholds.xao",
    )

    mesh(
        dim=3,
        input_file="test_interface_thresholds.xao",
        output_file="test_interface_thresholds.msh",
        resolution_specs={
            "inner": [
                ExponentialField(
                    sizemin=0.05,
                    lengthscale=0.5,  # Shorter transition distance
                    growth_factor=3.0,
                    apply_to="curves",
                    restrict_to=["inner"],
                )
            ],
            "outer": [
                ExponentialField(
                    sizemin=0.05,
                    lengthscale=2.0,
                    growth_factor=3,
                    apply_to="curves",
                    not_sharing=["None"],
                    restrict_to=["outer"],
                )
            ],
        },
        n_threads=1,
        default_characteristic_length=100,
    )

    compare_gmsh_files(Path("test_interface_thresholds.msh"))


if __name__ == "__main__":
    test_2D_resolution()
    # test_refine(ConstantInField(apply_to="surfaces", resolution=1))
