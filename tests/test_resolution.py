from __future__ import annotations

import shapely
from meshwell.polysurface import PolySurface
from meshwell.model import Model
from meshwell.gmsh_entity import GMSH_entity
from meshwell.prism import Prism
from meshwell.resolution import ResolutionSpec
from meshwell.utils import compare_meshes
from pathlib import Path

def test_2D_resolution():

    large_rect = 20
    small_rect = 5

    polygon1 = shapely.Polygon(
        [[0, 0], [large_rect, 0], [large_rect, large_rect], [0, large_rect], [0, 0]],
    )
    polygon2 = shapely.Polygon(
        [[small_rect, small_rect], [large_rect-small_rect, small_rect], [large_rect-small_rect, large_rect-small_rect], [small_rect, 6], [small_rect, small_rect]],
    )

    model = Model(n_threads=1) # 1 thread for deterministic mesh
    poly_obj1 = PolySurface(polygons=polygon1, 
                       model=model, 
                       mesh_order=2, 
                       physical_name="outer",
                       resolutions=[ResolutionSpec(resolution_surfaces=0.5)],
                       )
    poly_obj2 = PolySurface(polygons=polygon2, 
                       model=model, 
                       mesh_order=1, 
                       physical_name="inner",
                       resolutions=[ResolutionSpec(resolution_surfaces=2,
                                                    resolution_curves = 0.1,
                                                    distmax_curves=2,
                                                    sizemax_curves=1,
                                                  )]
                       )

    entities_list = [poly_obj1, poly_obj2]

    mesh = model.mesh(
        entities_list=entities_list,
        default_characteristic_length=1,
        verbosity=0,
        filename=f"mesh_test_2D_resolution.msh",
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
    prism_obj1 = Prism(polygons=polygon1, 
                       buffers=buffers, 
                       model=model, 
                       mesh_order=2, 
                       physical_name="outer",
                       resolutions=[ResolutionSpec(resolution_volumes=1)],
                       )
    prism_obj2 = Prism(polygons=polygon2, 
                       buffers=buffers, 
                       model=model, 
                       mesh_order=1, 
                       physical_name="inner",
                       resolutions=[ResolutionSpec(resolution_volumes=1,
                                                  resolution_surfaces=0.2,
                                                  distmax_surfaces=1,
                                                  sizemax_surfaces=1,
                                                  )]
                       )

    entities_list = [prism_obj1, prism_obj2]

    mesh = model.mesh(
        entities_list=entities_list,
        default_characteristic_length=1,
        verbosity=0,
        filename=f"mesh_test_3D_resolution.msh",
    )

    compare_meshes(Path("mesh_test_3D_resolution.msh"))

if __name__ == "__main__":
    test_2D_resolution()
    test_3D_resolution()
