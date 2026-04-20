from __future__ import annotations

import numpy as np
import shapely
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib

from meshwell.cad_occ import cad_occ
from meshwell.occ_geometry_cache import OCCGeometryCache
from meshwell.occ_to_gmsh import occ_to_xao
from meshwell.polyprism import PolyPrism


def test_prism():
    polygon1 = shapely.Polygon(
        [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
        holes=([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]],),
    )
    polygon2 = shapely.Polygon([[-1, -1], [-2, -1], [-2, -2], [-1, -2], [-1, -1]])
    polygon = shapely.MultiPolygon([polygon1, polygon2])

    buffers = {0.0: 0.0, 0.3: 0.1, 1.0: -0.2}

    prism_obj = PolyPrism(polygons=polygon, buffers=buffers, physical_name="prism")
    assert prism_obj.extrude is False
    occ_to_xao(cad_occ([prism_obj]), "test_prism.xao")


def test_prism_extruded():
    polygon1 = shapely.Polygon(
        [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
        holes=([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]],),
    )
    polygon2 = shapely.Polygon([[-1, -1], [-2, -1], [-2, -2], [-1, -2], [-1, -1]])
    polygon = shapely.MultiPolygon([polygon1, polygon2])

    buffers = {-1.0: 0.0, 1.0: 0.0}

    prism_obj = PolyPrism(
        polygons=polygon, buffers=buffers, physical_name="prism_extruded"
    )

    # Extrude path: instantiate once via the OCC cache and check its z-extent
    # before feeding through the full fragment/meshing pipeline.
    shape = prism_obj.instanciate_occ(occ_cache=OCCGeometryCache())
    assert prism_obj.extrude is True

    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    _, _, zmin, _, _, zmax = box.Get()
    assert np.isclose(zmin, min(buffers.keys()))
    assert np.isclose(zmax, max(buffers.keys()))

    occ_to_xao(cad_occ([prism_obj]), "test_prism_extruded.xao")


if __name__ == "__main__":
    test_prism_extruded()
