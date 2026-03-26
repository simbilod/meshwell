from functools import partial
from meshwell.occ_entity import OCC_entity

def mock_occ_func(x, y):
    return f"Shape at {x}, {y}"

def test_occ_entity_serialization():
    p_func = partial(mock_occ_func, 1.0, 2.0)
    entity = OCC_entity(occ_function=p_func, physical_name="occ_shape", mesh_order=2)
    
    d = entity.to_dict()
    assert d["type"] == "OCC_entity"
    assert d["function_name"] == "mock_occ_func"
    assert d["args"] == (1.0, 2.0)
    
    # Use registry for reconstruction
    registry = {"mock_occ_func": mock_occ_func}
    new_entity = OCC_entity.from_dict(d, registry=registry)
    
    assert new_entity.physical_name == ("occ_shape",)
    assert new_entity.mesh_order == 2
    assert new_entity.occ_function.func == mock_occ_func
    assert new_entity.occ_function.args == (1.0, 2.0)
