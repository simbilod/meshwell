from __future__ import annotations

import shapely

from meshwell.cad_occ import cad_occ
from meshwell.occ_entity import OCC_entity
from meshwell.occ_to_gmsh import occ_to_xao
from meshwell.polyprism import PolyPrism
from tests.test_occ_helpers import _occ_box, _occ_rectangle, _occ_sphere


def test_composite_cad_3D():
    # Prism from a shapely polygon, extruded z=[0,1]
    polygon = shapely.Polygon([[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]])
    prism_obj = PolyPrism(
        polygons=polygon,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="prism",
        mesh_order=1,
    )

    box_obj = OCC_entity(
        occ_function=_occ_box(x=1, y=1, z=0.5, dx=2, dy=2, dz=1),
        physical_name="box",
        mesh_order=2,
        additive=False,
        dimension=3,
    )

    sphere_obj = OCC_entity(
        occ_function=_occ_sphere(xc=0, yc=0, zc=0.5, radius=0.75),
        physical_name="sphere",
        mesh_order=2,
        additive=False,
        dimension=3,
    )

    separate_box = OCC_entity(
        occ_function=_occ_box(x=4, y=4, z=0, dx=1, dy=1, dz=1),
        physical_name="separate_box",
        mesh_order=3,
        dimension=3,
    )

    entities = [prism_obj, sphere_obj, box_obj, separate_box]
    occ_to_xao(
        cad_occ(entities, progress_bars=True),
        "test_composite_3D.xao",
    )


def test_composite_cad_mixed():
    """Mixed-dimensional CAD: 3D prism with an embedded 2D rectangle."""
    polygon = shapely.Polygon([[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]])
    prism_obj = PolyPrism(
        polygons=polygon,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="prism",
        mesh_order=2,
    )

    plane_obj = OCC_entity(
        occ_function=_occ_rectangle(x=0.5, y=0.5, z=0.5, dx=1, dy=1),
        physical_name="plane",
        mesh_order=1,
        additive=False,
        dimension=2,
    )

    entities = [prism_obj, plane_obj]
    occ_to_xao(
        cad_occ(entities, progress_bars=True),
        "test_composite_mixed.xao",
    )
