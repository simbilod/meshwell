from __future__ import annotations

from pathlib import Path

import gmsh
import pytest
import shapely

from meshwell.cad_occ import cad_occ
from meshwell.mesh import mesh
from meshwell.occ_entity import OCC_entity
from meshwell.occ_to_gmsh import inject_occ_entities_into_gmsh, occ_to_xao
from meshwell.polyprism import PolyPrism
from meshwell.polysurface import PolySurface


def test_occ_polysurface():
    """Test PolySurface with OCC backend."""
    polygon1 = shapely.Polygon(
        [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
        holes=([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]],),
    )
    polygon2 = shapely.Polygon([[-1, -1], [-2, -1], [-2, -2], [-1, -2], [-1, -1]])
    polygon = shapely.MultiPolygon([polygon1, polygon2])

    polysurface_obj = PolySurface(polygons=polygon, physical_name="polysurface")

    # 1. Process via OCC
    occ_entities = cad_occ(entities_list=[polysurface_obj])
    assert len(occ_entities) == 1
    assert occ_entities[0].dim == 2

    # 2. Bridge to GMSH and save XAO
    output_xao = Path("test_polysurface_occ.xao")
    occ_to_xao(
        occ_entities, output_xao, model_manager=None
    )  # Uses default model manager

    assert output_xao.exists()

    # 3. Mesh from XAO
    output_msh = Path("test_polysurface_occ.msh")
    mesh(
        input_file=output_xao,
        output_file=output_msh,
        dim=2,
        default_characteristic_length=0.5,
        n_threads=1,
    )
    assert output_msh.exists()


def test_occ_composite_3D():
    """Test composite 3D CAD with OCC backend, mirroring test_composite_cad_3D."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere
    from OCP.gp import gp_Pnt

    # Create a prism
    polygon = shapely.Polygon([[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]])
    buffers = {0.0: 0.0, 1.0: 0.0}
    prism_obj = PolyPrism(
        polygons=polygon, buffers=buffers, physical_name="prism", mesh_order=1
    )

    # Create an OCC_entity box
    box_obj = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(
            gp_Pnt(1, 1, 0.5), 2.0, 2.0, 1.0
        ).Shape(),
        physical_name="box",
        mesh_order=2,
        additive=False,
    )

    # Create an OCC_entity sphere
    sphere_obj = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeSphere(gp_Pnt(0, 0, 0.5), 0.75).Shape(),
        physical_name="sphere",
        mesh_order=2,
        additive=False,
    )

    entities = [prism_obj, sphere_obj, box_obj]

    # Process
    occ_entities = cad_occ(entities_list=entities)

    # Check that we have results
    assert len(occ_entities) > 0

    # Bridge and Tag
    # This will initialize gmsh and tag everything
    from meshwell.model import ModelManager

    mm = ModelManager()
    inject_occ_entities_into_gmsh(occ_entities, mm)

    # Verify physical names in gmsh model
    groups = gmsh.model.getPhysicalGroups()
    names = [gmsh.model.getPhysicalName(dim, tag) for dim, tag in groups]
    assert "prism" in names
    assert "box" in names
    assert "sphere" in names

    mm.finalize()


def test_occ_buffered_prism():
    """Test PolyPrism with buffering in OCC backend."""
    polygon = shapely.Polygon([[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]])
    # Non-zero buffers force the _create_occ_volume path
    buffers = {0.0: 0.0, 0.5: 0.2, 1.0: 0.0}
    prism_obj = PolyPrism(
        polygons=polygon, buffers=buffers, physical_name="buffered_prism"
    )

    occ_entities = cad_occ(entities_list=[prism_obj])
    assert len(occ_entities) == 1
    assert occ_entities[0].dim == 3

    output_xao = Path("test_buffered_prism_occ.xao")
    occ_to_xao(occ_entities, output_xao)
    assert output_xao.exists()


if __name__ == "__main__":
    pytest.main([__file__])
