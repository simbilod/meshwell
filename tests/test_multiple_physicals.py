from __future__ import annotations

import shapely
from meshwell.prism import Prism
from meshwell.model import Model
from meshwell.resolution import ResolutionSpec


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
        resolutions=[ResolutionSpec(resolution_volumes=10)],
    )
    medium_prism = Prism(
        polygons=polygon.buffer(5, join_style="mitre"),
        buffers=buffers,
        model=model,
        physical_name=("medium_prism", "center"),
        resolutions=[ResolutionSpec(resolution_volumes=10)],
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


if __name__ == "__main__":
    test_multiple_physicals()
