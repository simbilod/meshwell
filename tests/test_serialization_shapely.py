import shapely.geometry as sg
from meshwell.polyline import PolyLine
from meshwell.polysurface import PolySurface
from meshwell.polyprism import PolyPrism

def test_polyline_serialization():
    ls = sg.LineString([(0, 0), (1, 1)])
    entity = PolyLine(linestrings=ls, physical_name="line", mesh_order=1.5, identify_arcs=True)
    
    d = entity.to_dict()
    assert d["type"] == "PolyLine"
    # Shapely with rounding_precision=12 produces 12 decimal places
    assert "LINESTRING (0.000000000000 0.000000000000, 1.000000000000 1.000000000000)" in d["linestrings_wkt"][0]
    assert d["identify_arcs"] is True
    
    new_entity = PolyLine.from_dict(d)
    assert new_entity.physical_name == ("line",)
    assert len(new_entity.linestrings) == 1
    assert isinstance(new_entity.linestrings[0], sg.LineString)
    assert new_entity.identify_arcs is True

def test_polysurface_serialization():
    poly = sg.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    entity = PolySurface(polygons=poly, physical_name="surf", mesh_bool=False)
    
    d = entity.to_dict()
    assert d["type"] == "PolySurface"
    assert "POLYGON ((0.000000000000 0.000000000000, 1.000000000000 0.000000000000, 1.000000000000 1.000000000000, 0.000000000000 1.000000000000, 0.000000000000 0.000000000000))" in d["polygons_wkt"][0]
    
    new_entity = PolySurface.from_dict(d)
    assert new_entity.physical_name == ("surf",)
    assert new_entity.mesh_bool is False
    assert len(new_entity.polygons) == 1
    assert isinstance(new_entity.polygons[0], sg.Polygon)

def test_polyprism_serialization():
    poly = sg.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    buffers = {0.0: 0.0, 1.0: 0.5}
    entity = PolyPrism(polygons=poly, buffers=buffers, physical_name="prism", subdivision=(2, 2, 2))
    
    d = entity.to_dict()
    assert d["type"] == "PolyPrism"
    assert "POLYGON ((0.000000000000 0.000000000000, 1.000000000000 0.000000000000, 1.000000000000 1.000000000000, 0.000000000000 1.000000000000, 0.000000000000 0.000000000000))" in d["polygons_wkt"][0]
    assert d["buffers"] == {"0.0": 0.0, "1.0": 0.5}
    assert d["subdivision"] == [2, 2, 2]
    
    new_entity = PolyPrism.from_dict(d)
    assert new_entity.physical_name == ("prism",)
    assert new_entity.buffers == {0.0: 0.0, 1.0: 0.5}
    assert new_entity.subdivision == (2, 2, 2)
    assert len(new_entity.buffered_polygons) > 0 or new_entity.extrude
