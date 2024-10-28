from __future__ import annotations
import shapely
from meshwell.polysurface import PolySurface
from meshwell.model import Model
from meshwell.prism import Prism
from meshwell.resolution import ResolutionSpec
import pytest

domain = shapely.box(0, 0, 10, 10)
inside = shapely.box(4, 4, 6, 6)


@pytest.mark.skip(reason="Info not reported; check with debugger if in doubt")
def test_2D_distance_sampling():
    model = Model(n_threads=1)  # 1 thread for deterministic mesh
    poly_obj1 = PolySurface(
        polygons=domain,
        model=model,
        mesh_order=2,
        physical_name="outer",
        resolutions=[
            ResolutionSpec(
                resolution_surfaces=0.1,
            )
        ],
    )
    poly_obj2 = PolySurface(
        polygons=inside,
        model=model,
        mesh_order=1,
        physical_name="inner",
        resolutions=[
            ResolutionSpec(
                resolution_surfaces=0.1,
                resolution_curves=0.1,
                distmax_curves=2,
                sizemax_curves=1,
                length_per_sampling_curves=0.01,  # default is half resolution
            )
        ],
    )

    entities_list = [poly_obj1, poly_obj2]

    model.mesh(
        entities_list=entities_list,
        default_characteristic_length=1,
        verbosity=0,
        filename="mesh_test_2D_distance_sampling.msh",
    )


@pytest.mark.skip(reason="Info not reported; check with debugger if in doubt")
def test_3D_resolution():
    buffers_outer = {0.0: 0.0, 10: 0.0}
    buffers_inner = {4.0: 0.0, 6.0: 0.0}

    model = Model(n_threads=1)
    prism_obj1 = Prism(
        polygons=domain,
        buffers=buffers_outer,
        model=model,
        mesh_order=2,
        physical_name="outer",
        resolutions=[ResolutionSpec(resolution_volumes=0.1)],
    )
    prism_obj2 = Prism(
        polygons=inside,
        buffers=buffers_inner,
        model=model,
        mesh_order=1,
        physical_name="inner",
        resolutions=[
            ResolutionSpec(
                resolution_surfaces=0.1,
                resolution_curves=0.1,
                distmax_curves=2,
                sizemax_curves=1,
                surface_per_sampling_surfaces=1,
                length_per_sampling_curves=1,  # default is half resolution
            )
        ],
    )

    entities_list = [prism_obj1, prism_obj2]

    model.mesh(
        entities_list=entities_list,
        default_characteristic_length=1,
        verbosity=0,
        filename="mesh_test_3D_resolution.msh",
    )


if __name__ == "__main__":
    test_3D_resolution()
