from __future__ import annotations
import gmsh
from meshwell.gmsh_entity import GMSH_entity
from meshwell.cad import cad
from functools import partial


def test_gmsh_entity():
    gmsh_obj = GMSH_entity(
        gmsh_partial_function=partial(
            gmsh.model.occ.add_box, x=0, y=0, z=0, dx=1, dy=1, dz=1
        ),
        physical_name="gmsh_entity",
        mesh_order=1,
    )

    cad(entities_list=[gmsh_obj], output_file="test_gmsh_entity.xao")
