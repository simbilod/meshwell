from __future__ import annotations

import shapely
from meshwell.polyprism import PolyPrism
from meshwell.cad import cad
from meshwell.mesh import mesh


def test_multiple_physicals():
    polygon = shapely.Polygon(
        [[-5, -5], [5, -5], [5, 5], [-5, 5], [-5, -5]],
    )

    buffers = {0.0: 0.0, 1.0: -0.1}

    big_prism = PolyPrism(
        polygons=polygon.buffer(10, join_style="mitre"),
        buffers=buffers,
        physical_name=("big_prism", "domain"),
        mesh_order=3,
    )
    medium_prism = PolyPrism(
        polygons=polygon.buffer(5, join_style="mitre"),
        buffers=buffers,
        physical_name=("medium_prism", "center"),
        mesh_order=2,
    )
    small_prism = PolyPrism(
        polygons=polygon,
        buffers=buffers,
        physical_name="small_prism",
        mesh_order=1,
    )
    entities_list = [big_prism, medium_prism, small_prism]

    cad(
        entities_list=entities_list,
        output_file="test_multiple_physicals.xao",
    )

    mesh_obj = mesh(
        dim=3,
        input_file="test_multiple_physicals.xao",
        output_file="test_multiple_physicals.msh",
        n_threads=1,
        default_characteristic_length=100,
    )

    # Equivalence of volumes
    for key1, key2 in [["domain", "big_prism"], ["center", "medium_prism"]]:
        assert (
            mesh_obj.cell_sets_dict[key1]["tetra"].all()
            == mesh_obj.cell_sets_dict[key2]["tetra"].all()
        )
    # Equivalence of surfaces
    assert (
        mesh_obj.cell_sets_dict["small_prism___medium_prism"]["triangle"].all()
        == mesh_obj.cell_sets_dict["small_prism___center"]["triangle"].all()
    )
    assert (
        mesh_obj.cell_sets_dict["medium_prism___big_prism"]["triangle"].all()
        == mesh_obj.cell_sets_dict["medium_prism___domain"]["triangle"].all()
    )
    assert (
        mesh_obj.cell_sets_dict["medium_prism___big_prism"]["triangle"].all()
        == mesh_obj.cell_sets_dict["center___domain"]["triangle"].all()
    )
    # Equivalence of boundaries
    assert (
        mesh_obj.cell_sets_dict["big_prism___None"]["triangle"].all()
        == mesh_obj.cell_sets_dict["domain___None"]["triangle"].all()
    )


ref1_physical_present = ["big", "small"]
ref1_physical_absent = ["medium", "small+medium", "big+medium"]
ref2_physical_present = ["big", "small", "medium"]
ref2_physical_absent = ["small+medium", "big+medium"]
ref3_physical_present = ["big", "small", "small+medium", "big+medium"]
ref3_physical_absent = ["medium"]


if __name__ == "__main__":
    test_multiple_physicals()
