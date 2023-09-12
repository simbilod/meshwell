from __future__ import annotations

import shapely
from meshwell.polysurface import PolySurface
from meshwell.model import Model
from meshwell.validation import sort_entities_by_mesh_order


def test_mesh_order():
    polygon = shapely.box(xmin=0, ymin=1, xmax=1, ymax=1)

    model = Model()

    entities = [
        PolySurface(
            polygons=polygon, model=model, mesh_order=10, physical_name="mesh10"
        ),
        PolySurface(polygons=polygon, model=model, mesh_order=2, physical_name="mesh2"),
        PolySurface(polygons=polygon, model=model, physical_name="meshdefault"),
        PolySurface(
            polygons=polygon, model=model, mesh_order=3.5, physical_name="mesh3p5"
        ),
    ]

    assert entities[0].physical_name == "mesh10"
    assert entities[1].physical_name == "mesh2"
    assert entities[2].physical_name == "meshdefault"
    assert entities[3].physical_name == "mesh3p5"

    ordered_entities = sort_entities_by_mesh_order(entities)

    assert ordered_entities[0].physical_name == "mesh2"
    assert ordered_entities[1].physical_name == "mesh3p5"
    assert ordered_entities[2].physical_name == "mesh10"
    assert ordered_entities[3].physical_name == "meshdefault"


if __name__ == "__main__":
    test_mesh_order()
