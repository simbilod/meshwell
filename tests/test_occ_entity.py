from __future__ import annotations

from functools import partial

from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeWire,
)
from OCP.gp import gp_Pnt

from meshwell.cad_occ import cad_occ
from meshwell.occ_entity import OCC_entity
from meshwell.occ_xao_writer import occ_to_xao
from tests.test_occ_helpers import _occ_box


def test_occ_entity_box():
    """Basic OCC_entity wrapping a box primitive flows through the pipeline."""
    entity = OCC_entity(
        occ_function=_occ_box(0, 0, 0, 1, 1, 1),
        physical_name="occ_entity",
        mesh_order=1,
        dimension=3,
    )
    occ_to_xao(cad_occ([entity]), "test_occ_entity.xao")


def test_occ_entity_custom_callable():
    """OCC_entity accepts any zero-arg callable returning a TopoDS_Shape."""

    def front_face_rectangle(x_min, x_max, y, z_min, z_max):
        # Planar quad in the y=const plane
        p1 = gp_Pnt(x_min, y, z_min)
        p2 = gp_Pnt(x_max, y, z_min)
        p3 = gp_Pnt(x_max, y, z_max)
        p4 = gp_Pnt(x_min, y, z_max)
        e1 = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
        e2 = BRepBuilderAPI_MakeEdge(p2, p3).Edge()
        e3 = BRepBuilderAPI_MakeEdge(p3, p4).Edge()
        e4 = BRepBuilderAPI_MakeEdge(p4, p1).Edge()
        wire = BRepBuilderAPI_MakeWire(e1, e2, e3, e4).Wire()
        return BRepBuilderAPI_MakeFace(wire).Shape()

    entity = OCC_entity(
        occ_function=partial(
            front_face_rectangle, x_min=0, x_max=1, y=0, z_min=0, z_max=1
        ),
        physical_name="custom_occ_entity",
        mesh_order=1,
        dimension=2,
    )
    occ_to_xao(cad_occ([entity]), "custom_occ_entity.xao")


def test_deserialize_no_longer_knows_gmsh_entity():
    """After deprecation, deserialize must raise for the legacy type tag."""
    import pytest

    from meshwell.utils import deserialize

    with pytest.raises(ValueError, match="Unknown entity type"):
        deserialize({"type": "GMSH_entity", "physical_name": "x"})


if __name__ == "__main__":
    test_occ_entity_box()
    test_occ_entity_custom_callable()
