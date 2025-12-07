from __future__ import annotations

from functools import partial

import gmsh

from meshwell.cad import cad
from meshwell.gmsh_entity import GMSH_entity


def test_gmsh_entity():
    gmsh_obj = GMSH_entity(
        gmsh_partial_function=partial(
            gmsh.model.occ.add_box, x=0, y=0, z=0, dx=1, dy=1, dz=1
        ),
        physical_name="gmsh_entity",
        mesh_order=1,
    )

    cad(entities_list=[gmsh_obj], output_file="test_gmsh_entity")


def test_custom_partial():
    def front_face_rectangle(x_min, x_max, y, z_min, z_max):
        """Assumes model is already initialized."""
        # Define the four corners of the rectangle in 3D space
        # Front face is at y=y_max, spanning x and z
        p1 = gmsh.model.occ.addPoint(x_min, y, z_min)
        p2 = gmsh.model.occ.addPoint(x_max, y, z_min)
        p3 = gmsh.model.occ.addPoint(x_max, y, z_max)
        p4 = gmsh.model.occ.addPoint(x_min, y, z_max)

        # Create lines connecting the points
        l1 = gmsh.model.occ.addLine(p1, p2)
        l2 = gmsh.model.occ.addLine(p2, p3)
        l3 = gmsh.model.occ.addLine(p3, p4)
        l4 = gmsh.model.occ.addLine(p4, p1)

        # Create a curve loop and plane surface
        loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
        return gmsh.model.occ.addPlaneSurface([loop])

    gmsh_obj = GMSH_entity(
        gmsh_partial_function=partial(
            front_face_rectangle, x_min=0, x_max=1, y=0, z_min=0, z_max=1
        ),
        physical_name="custom_gmsh_entity",
        mesh_order=1,
        dimension=2,
    )

    cad(entities_list=[gmsh_obj], output_file="custom_gmsh_entity.xao")


if __name__ == "__main__":
    test_custom_partial()
