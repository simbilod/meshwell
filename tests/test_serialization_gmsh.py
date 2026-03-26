import gmsh
from functools import partial
from meshwell.gmsh_entity import GMSH_entity

def test_gmsh_entity_serialization():
    gmsh.initialize()
    func = gmsh.model.occ.addBox
    p_func = partial(func, 0, 0, 0, 1, 1, 1)
    entity = GMSH_entity(gmsh_partial_function=p_func, physical_name="box", mesh_order=1)
    
    d = entity.to_dict()
    assert d["type"] == "GMSH_entity"
    assert d["function_name"] in ["addBox", "add_box"]
    assert d["physical_name"] == ("box",)
    
    new_entity = GMSH_entity.from_dict(d)
    assert new_entity.physical_name == ("box",)
    assert new_entity.mesh_order == 1
    assert new_entity.gmsh_partial_function.func == func
    gmsh.finalize()
