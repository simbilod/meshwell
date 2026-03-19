"""Test multi-dimensional entity processing capabilities with OCC backend.

This test module validates the ability to process entities of different
dimensions (3D volumes, 2D surfaces, 1D curves, 0D points) in a single model
using fragment-based integration in the OCC backend.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeVertex,
    BRepBuilderAPI_MakeWire,
)
from OCP.BRepPrimAPI import (
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakeSphere,
)
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

from meshwell.cad_occ import cad_occ
from meshwell.mesh import mesh
from meshwell.occ_entity import OCC_entity
from meshwell.occ_to_gmsh import occ_to_xao
from meshwell.resolution import ConstantInField


def make_rectangle_occ(x, y, z, dx, dy):
    """Helper to create a rectangle face in OCC."""
    p1 = gp_Pnt(x, y, z)
    p2 = gp_Pnt(x + dx, y, z)
    p3 = gp_Pnt(x + dx, y + dy, z)
    p4 = gp_Pnt(x, y + dy, z)
    e1 = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
    e2 = BRepBuilderAPI_MakeEdge(p2, p3).Edge()
    e3 = BRepBuilderAPI_MakeEdge(p3, p4).Edge()
    e4 = BRepBuilderAPI_MakeEdge(p4, p1).Edge()
    wire = BRepBuilderAPI_MakeWire(e1, e2, e3, e4).Wire()
    return BRepBuilderAPI_MakeFace(wire).Face()


def test_volume_with_internal_surface_occ():
    """Test 3D volume with 2D surface fragment integration in OCC."""
    box_obj = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(2.0, 2.0, 1.0).Shape(),
        physical_name="box",
        mesh_order=1,
    )

    surface_obj = OCC_entity(
        occ_function=lambda: make_rectangle_occ(0.5, 0.5, 0.5, 1.0, 1.0),
        physical_name="internal_surface",
        mesh_order=2,
    )

    entities = [box_obj, surface_obj]
    occ_entities = cad_occ(entities_list=entities)
    output_xao = Path("test_volume_with_surface_occ.xao")
    occ_to_xao(occ_entities, output_xao)

    mesh(
        dim=3,
        input_file=output_xao,
        output_file=output_xao.with_suffix(".msh"),
        default_characteristic_length=1,
    )


def test_lower_dim_with_multiple_higher_entities_occ():
    """Test lower-dimension object interacting with multiple higher-level entities in OCC."""
    box1_obj = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(
            gp_Pnt(0, 0, 0), 1.0, 1.0, 1.0
        ).Shape(),
        physical_name="box1",
        mesh_order=1,
    )

    box2_obj = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(
            gp_Pnt(1.5, 0, 0), 1.0, 1.0, 1.0
        ).Shape(),
        physical_name="box2",
        mesh_order=2,
    )

    spanning_surface = OCC_entity(
        occ_function=lambda: make_rectangle_occ(0.5, 0.5, 0.5, 1.5, 0.5),
        physical_name="spanning_surface",
        mesh_order=3,
    )

    entities = [box1_obj, box2_obj, spanning_surface]
    occ_entities = cad_occ(entities_list=entities)
    output_xao = Path("test_lower_dim_multiple_higher_occ.xao")
    occ_to_xao(occ_entities, output_xao)

    mesh(
        dim=3,
        input_file=output_xao,
        output_file=output_xao.with_suffix(".msh"),
        default_characteristic_length=1,
        resolution_specs={
            "spanning_surface": [ConstantInField(apply_to="surfaces", resolution=0.1)],
        },
    )


def test_surface_boundary_overlap_occ():
    """Test surface that lies exactly on boundary of volume in OCC."""
    box_obj = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(2.0, 2.0, 1.0).Shape(),
        physical_name="box",
        mesh_order=1,
    )

    boundary_surface = OCC_entity(
        occ_function=lambda: make_rectangle_occ(0.5, 0.5, 1.0, 1.0, 1.0),
        physical_name="boundary_surface",
        mesh_order=2,
    )

    entities = [box_obj, boundary_surface]
    occ_entities = cad_occ(entities_list=entities)
    output_xao = Path("test_boundary_overlap_occ.xao")
    occ_to_xao(occ_entities, output_xao)

    mesh(
        dim=3,
        input_file=output_xao,
        output_file=output_xao.with_suffix(".msh"),
        default_characteristic_length=1,
    )


def test_point_in_multiple_entities_occ():
    """Test 0D point inside multiple overlapping entities in OCC."""
    box_obj = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(2.0, 2.0, 2.0).Shape(),
        physical_name="box",
        mesh_order=1,
    )

    sphere_obj = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeSphere(gp_Pnt(1, 1, 1), 1.2).Shape(),
        physical_name="sphere",
        mesh_order=2,
    )

    surface_obj = OCC_entity(
        occ_function=lambda: make_rectangle_occ(0.5, 0.5, 1, 1, 1),
        physical_name="surface",
        mesh_order=3,
    )

    point_obj = OCC_entity(
        occ_function=lambda: BRepBuilderAPI_MakeVertex(gp_Pnt(1, 1, 1)).Vertex(),
        physical_name="central_point",
        mesh_order=4,
    )

    entities = [box_obj, sphere_obj, surface_obj, point_obj]
    occ_entities = cad_occ(entities_list=entities)
    output_xao = Path("test_point_multiple_entities_occ.xao")
    occ_to_xao(occ_entities, output_xao)

    mesh(
        dim=3,
        input_file=output_xao,
        output_file=output_xao.with_suffix(".msh"),
        default_characteristic_length=1,
    )


def test_sequential_fragmentation_complex_occ():
    """Test sequential fragmentation with multiple lower-dim entities in OCC."""
    base_volume = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(3.0, 3.0, 2.0).Shape(),
        physical_name="base_volume",
        mesh_order=1,
    )

    surface1 = OCC_entity(
        occ_function=lambda: make_rectangle_occ(0.5, 0.5, 1, 2, 2),
        physical_name="surface1",
        mesh_order=2,
    )

    surface2 = OCC_entity(
        occ_function=lambda: make_rectangle_occ(1, 1, 0.5, 1.5, 1.5),
        physical_name="surface2",
        mesh_order=3,
    )

    # 1D curve: Arc or Circle
    def make_circle():
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
        from OCP.gp import gp_Circ

        circle = gp_Circ(gp_Ax2(gp_Pnt(1.5, 1.5, 1.2), gp_Dir(0, 0, 1)), 0.8)
        return BRepBuilderAPI_MakeEdge(circle).Edge()

    curve = OCC_entity(
        occ_function=make_circle,
        physical_name="curve",
        mesh_order=4,
    )

    point1 = OCC_entity(
        occ_function=lambda: BRepBuilderAPI_MakeVertex(gp_Pnt(1.2, 1.2, 1.2)).Vertex(),
        physical_name="point1",
        mesh_order=5,
    )

    point2 = OCC_entity(
        occ_function=lambda: BRepBuilderAPI_MakeVertex(gp_Pnt(2, 2, 0.5)).Vertex(),
        physical_name="point2",
        mesh_order=6,
    )

    entities = [base_volume, surface1, surface2, curve, point1, point2]
    occ_entities = cad_occ(entities_list=entities)
    output_xao = Path("test_sequential_complex_occ.xao")
    occ_to_xao(occ_entities, output_xao)

    mesh(
        dim=3,
        input_file=output_xao,
        output_file=output_xao.with_suffix(".msh"),
        default_characteristic_length=1,
    )


if __name__ == "__main__":
    pytest.main([__file__])
