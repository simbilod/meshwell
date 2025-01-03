from __future__ import annotations

import shapely
from meshwell.prism import Prism
from meshwell.model import Model
from meshwell.resolution import ConstantInField
from shapely import clip_by_rect
import pytest


def test_multiple_physicals():
    polygon = shapely.Polygon(
        [[-5, -5], [5, -5], [5, 5], [-5, 5], [-5, -5]],
    )

    buffers = {0.0: 0.0, 1.0: -0.1}

    model = Model(n_threads=1)
    big_prism = Prism(
        polygons=polygon.buffer(10, join_style="mitre"),
        buffers=buffers,
        model=model,
        physical_name=("big_prism", "domain"),
        mesh_order=3,
        resolutions=[
            ConstantInField(resolution=10, apply_to="volumes"),
        ],
    )
    medium_prism = Prism(
        polygons=polygon.buffer(5, join_style="mitre"),
        buffers=buffers,
        model=model,
        physical_name=("medium_prism", "center"),
        resolutions=[
            ConstantInField(resolution=10, apply_to="volumes"),
        ],
        mesh_order=2,
    )
    small_prism = Prism(
        polygons=polygon,
        buffers=buffers,
        model=model,
        physical_name="small_prism",
        mesh_order=1,
        resolutions=[
            ConstantInField(resolution=10, apply_to="volumes"),
        ],
    )
    entities_list = [big_prism, medium_prism, small_prism]

    mesh = model.mesh(
        entities_list=entities_list,
        default_characteristic_length=10,
        verbosity=False,
        filename="mesh3D_multiplePhysicals.msh",
    )

    # Equivalence of volumes
    for key1, key2 in [["domain", "big_prism"], ["center", "medium_prism"]]:
        assert (
            mesh.cell_sets_dict[key1]["tetra"].all()
            == mesh.cell_sets_dict[key2]["tetra"].all()
        )
    # Equivalence of surfaces
    assert (
        mesh.cell_sets_dict["small_prism___medium_prism"]["triangle"].all()
        == mesh.cell_sets_dict["small_prism___center"]["triangle"].all()
    )
    assert (
        mesh.cell_sets_dict["medium_prism___big_prism"]["triangle"].all()
        == mesh.cell_sets_dict["medium_prism___domain"]["triangle"].all()
    )
    assert (
        mesh.cell_sets_dict["medium_prism___big_prism"]["triangle"].all()
        == mesh.cell_sets_dict["center___domain"]["triangle"].all()
    )
    # Equivalence of boundaries
    assert (
        mesh.cell_sets_dict["big_prism___None"]["triangle"].all()
        == mesh.cell_sets_dict["domain___None"]["triangle"].all()
    )


ref1_physical_present = ["big", "small"]
ref1_physical_absent = ["medium", "small+medium", "big+medium"]
ref2_physical_present = ["big", "small", "medium"]
ref2_physical_absent = ["small+medium", "big+medium"]
ref3_physical_present = ["big", "small", "small+medium", "big+medium"]
ref3_physical_absent = ["medium"]


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ((True, False, False), (ref1_physical_present, ref1_physical_absent)),
        ((False, True, False), (ref2_physical_present, ref2_physical_absent)),
        ((False, False, True), (ref3_physical_present, ref3_physical_absent)),
    ],
)
def test_multiple_physicals_additive(test_input, expected):
    polygon = shapely.Polygon(
        [[-5, -5], [5, -5], [5, 5], [-5, 5], [-5, -5]],
    )
    """TODO: make the test actually compare the overlapping elements."""

    buffers = {0.0: 0.0, 1.0: -0.1}

    model = Model(n_threads=1)
    big_prism = Prism(
        polygons=polygon.buffer(10, join_style="mitre"),
        buffers=buffers,
        model=model,
        physical_name=("big"),
        mesh_order=3,
        resolutions=[
            ConstantInField(resolution=10, apply_to="volumes"),
        ],
    )
    medium_prism = Prism(
        polygons=clip_by_rect(
            polygon.buffer(5, join_style="mitre"), xmin=0, ymin=-100, ymax=100, xmax=100
        ),
        buffers=buffers,
        model=model,
        physical_name=("medium"),
        resolutions=[
            ConstantInField(resolution=10, apply_to="volumes"),
        ],
        mesh_order=2,
        additive=True,
    )
    small_prism = Prism(
        polygons=polygon,
        buffers=buffers,
        model=model,
        physical_name="small",
        mesh_order=1,
        resolutions=[
            ConstantInField(resolution=10, apply_to="volumes"),
        ],
    )
    entities_list = [big_prism, medium_prism, small_prism]

    mesh = model.mesh(
        entities_list=entities_list,
        default_characteristic_length=10,
        verbosity=False,
        filename="mesh3D_multiplePhysicalsAdditive.msh",
        addition_structural_physicals=test_input[0],
        addition_addition_physicals=test_input[1],
        addition_intersection_physicals=test_input[2],
    )

    for entry in expected[0]:
        assert entry in mesh.cell_sets_dict
    for entry in expected[1]:
        assert entry not in mesh.cell_sets_dict


if __name__ == "__main__":
    test_multiple_physicals_additive()
