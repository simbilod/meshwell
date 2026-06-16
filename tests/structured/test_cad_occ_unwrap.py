from OCP.BRep import BRep_Builder
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.TopoDS import TopoDS_Compound

from meshwell.cad_occ import CAD_OCC


class _StubCompoundEntity:
    """Entity whose instanciate_occ returns a compound of two solids."""

    dimension = 3
    mesh_order = 1.0
    mesh_bool = True
    physical_name = ("stub",)

    def instanciate_occ(self):
        b = BRep_Builder()
        c = TopoDS_Compound()
        b.MakeCompound(c)
        s1 = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Solid()
        s2 = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Solid()
        b.Add(c, s1)
        b.Add(c, s2)
        return c


def test_compound_flattens_to_constituent_solids():
    proc = CAD_OCC()
    labeled = proc._instantiate_entity_occ(0, _StubCompoundEntity())
    assert len(labeled.shapes) == 2
