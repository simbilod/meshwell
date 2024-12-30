from __future__ import annotations

import shapely
from meshwell.polysurface import PolySurface
from meshwell.model import Model

import numpy as np
from meshwell.utils import compare_meshes
from pathlib import Path
from shapely.geometry import box
from meshwell.resolution import ThresholdField


def test_fuse():
    polygon1 = shapely.box(xmin=0, ymin=0, xmax=2, ymax=2)
    polygon2 = shapely.box(xmin=-1, ymin=-1, xmax=1, ymax=1)
    polygon3 = shapely.box(xmin=-3, ymin=-3, xmax=-2, ymax=-2)

    model = Model(n_threads=1)

    entities = [
        PolySurface(polygons=polygon1, model=model, physical_name="cad1"),
        PolySurface(polygons=polygon2, model=model, physical_name="cad2"),
        PolySurface(polygons=polygon3, model=model, physical_name="cad3"),
    ]

    mesh_unfused = model.mesh(
        entities_list=entities,
        verbosity=False,
        filename="mesh_unfused.msh",
        fuse_entities_by_name=True,
    )

    model = Model(n_threads=1)

    entities = [
        PolySurface(polygons=polygon1, model=model, physical_name="cad1"),
        PolySurface(polygons=polygon2, model=model, physical_name="cad1"),
        PolySurface(polygons=polygon3, model=model, physical_name="cad1"),
    ]

    mesh_fused = model.mesh(
        entities_list=entities,
        verbosity=False,
        filename="mesh_fused.msh",
        fuse_entities_by_name=True,
    )

    dimtags_fused = np.unique(mesh_fused.point_data["gmsh:dim_tags"], axis=0)
    dimtags_unfused = np.unique(mesh_unfused.point_data["gmsh:dim_tags"], axis=0)

    dimtags_fused_2D = dimtags_fused[np.where(dimtags_fused[:, 0] == 2)]
    dimtags_unfused_2D = dimtags_unfused[np.where(dimtags_unfused[:, 0] == 2)]

    dimtags_fused_1D = dimtags_fused[np.where(dimtags_fused[:, 0] == 1)]
    dimtags_unfused_1D = dimtags_unfused[np.where(dimtags_unfused[:, 0] == 1)]

    assert len(dimtags_fused_2D) == 2
    assert len(dimtags_unfused_2D) == 3
    assert len(dimtags_fused_1D) < len(dimtags_unfused_1D)  # less interfaces

    compare_meshes(Path("mesh_fused.msh"))
    compare_meshes(Path("mesh_unfused.msh"))


def test_different_resolutions_same_physical():
    model = Model()

    # Create two different boxes
    box1 = box(0, 0, 1, 1)
    box2 = box(2, 0, 3, 1)

    # Create resolution specifications
    resolution1 = ThresholdField(
        apply_to="surfaces", sizemin=0.1, sizemax=0.5, distmax=0.2
    )

    resolution2 = ThresholdField(
        apply_to="surfaces", sizemin=0.2, sizemax=1.0, distmax=0.4
    )

    # Create two polysurfaces with same physical name but different resolutions
    surface1 = PolySurface(
        polygons=box1,
        model=model,
        physical_name="test_surface",
        resolutions=[resolution1],
    )

    surface2 = PolySurface(
        polygons=box2,
        model=model,
        physical_name="test_surface",
        resolutions=[resolution2],
    )

    # Mesh the model
    model.mesh(
        entities_list=[surface1, surface2],
        default_characteristic_length=0.5,
        filename="test_different_resolutions_same_physical.msh",
    )

    compare_meshes(Path("test_different_resolutions_same_physical.msh"))


if __name__ == "__main__":
    # test_fuse()
    test_different_resolutions_same_physical()
