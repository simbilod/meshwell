from __future__ import annotations

import pytest
import gmsh
import shapely
from meshwell.prism import Prism
from meshwell.cad import cad

from meshwell.gmsh_entity import GMSH_entity
from functools import partial


def test_composite_cad_3D():
    # Create a prism
    polygon = shapely.Polygon([[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]])
    buffers = {0.0: 0.0, 1.0: 0.0}  # Extrude from z=0 to z=1
    prism_obj = Prism(
        polygons=polygon, buffers=buffers, physical_name="prism", mesh_order=1
    )

    # Create a box that intersects with the prism (additive)
    box_obj = GMSH_entity(
        gmsh_partial_function=partial(
            gmsh.model.occ.add_box,
            x=1,
            y=1,
            z=0.5,  # Positioned to intersect the prism
            dx=2,
            dy=2,
            dz=1,
        ),
        physical_name="box",
        mesh_order=2,
        additive=False,  # Make this an additive entity
    )

    # Create a sphere that intersects with the prism (non-additive)
    sphere_obj = GMSH_entity(
        gmsh_partial_function=partial(
            gmsh.model.occ.add_sphere,
            xc=0,
            yc=0,
            zc=0.5,  # Center at edge of prism
            radius=0.75,
        ),
        physical_name="sphere",
        mesh_order=2,
        additive=False,  # This will cut into the prism
    )

    # Create another box that's separate
    separate_box = GMSH_entity(
        gmsh_partial_function=partial(
            gmsh.model.occ.add_box, x=4, y=4, z=0, dx=1, dy=1, dz=1
        ),
        physical_name="separate_box",
        mesh_order=3,
    )

    # Process all entities
    entities = [prism_obj, sphere_obj, box_obj, separate_box]
    cad(
        entities_list=entities,
        addition_delimiter="*",  # Custom delimiter for testing
        progress_bars=True,
    )


@pytest.mark.skip(
    "Skipping -- cannot handle different dimensions seamlessly currently!"
)
def test_composite_cad_mixed():
    # Create a prism
    polygon = shapely.Polygon([[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]])
    buffers = {0.0: 0.0, 1.0: 0.0}  # Extrude from z=0 to z=1
    prism_obj = Prism(
        polygons=polygon, buffers=buffers, physical_name="prism", mesh_order=2
    )

    # Create a plane
    # Create a plane at z=0.5
    plane_obj = GMSH_entity(
        gmsh_partial_function=partial(
            gmsh.model.occ.add_rectangle,
            x=0.5,
            y=0.5,
            z=0.5,  # Center the plane inside the prism
            dx=1,
            dy=1,
        ),
        physical_name="plane",
        mesh_order=1,
        additive=False,
    )

    # Process all entities
    entities = [prism_obj, plane_obj]
    cad(
        entities_list=entities,
        output_file="test_composite_mixed.xao",
        addition_delimiter="*",  # Custom delimiter for testing
        progress_bars=True,
    )
