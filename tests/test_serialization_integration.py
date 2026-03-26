import gmsh
import shapely.geometry as sg
from functools import partial
from meshwell import deserialize
from meshwell.gmsh_entity import GMSH_entity
from meshwell.polyline import PolyLine
from meshwell.polysurface import PolySurface
from meshwell.polyprism import PolyPrism
from meshwell.occ_entity import OCC_entity

def mock_occ_func(x, y):
    return f"Shape at {x}, {y}"

def test_mixed_serialization_integration():
    gmsh.initialize()
    
    # 1. GMSH_entity
    p_func = partial(gmsh.model.occ.addBox, 0, 0, 0, 1, 1, 1)
    e1 = GMSH_entity(gmsh_partial_function=p_func, physical_name="box")
    
    # 2. PolyLine
    e2 = PolyLine(linestrings=sg.LineString([(0,0), (1,1)]), physical_name="line")
    
    # 3. PolySurface
    e3 = PolySurface(polygons=sg.Polygon([(0,0), (1,0), (1,1)]), physical_name="surf")
    
    # 4. PolyPrism
    e4 = PolyPrism(polygons=sg.Polygon([(0,0), (1,0), (0,1)]), buffers={0:0, 1:1}, physical_name="prism")
    
    # 5. OCC_entity
    p_func_occ = partial(mock_occ_func, 5, 5)
    e5 = OCC_entity(occ_function=p_func_occ, physical_name="occ")
    
    entities = [e1, e2, e3, e4, e5]
    
    # Serialize
    serialized = [e.to_dict() for e in entities]
    
    # Deserialize
    registry = {"mock_occ_func": mock_occ_func}
    new_entities = deserialize(serialized, registry=registry)
    
    assert len(new_entities) == 5
    assert isinstance(new_entities[0], GMSH_entity)
    assert isinstance(new_entities[1], PolyLine)
    assert isinstance(new_entities[2], PolySurface)
    assert isinstance(new_entities[3], PolyPrism)
    assert isinstance(new_entities[4], OCC_entity)
    
    assert new_entities[0].physical_name == ("box",)
    assert new_entities[1].physical_name == ("line",)
    assert new_entities[2].physical_name == ("surf",)
    assert new_entities[3].physical_name == ("prism",)
    assert new_entities[4].physical_name == ("occ",)
    
    gmsh.finalize()
