"""Class definition for delayed gmhs entity evaluation."""

from functools import partial

import gmsh

from meshwell.cad import CAD
from meshwell.validation import format_physical_name

GMSH_ENTITY_DIMENSIONS = {
    # 0D (Points)
    gmsh.model.occ.addPoint: 0,
    gmsh.model.occ.add_point: 0,
    # 1D (Curves)
    gmsh.model.occ.addLine: 1,
    gmsh.model.occ.add_line: 1,
    gmsh.model.occ.addCircle: 1,
    gmsh.model.occ.add_circle: 1,
    gmsh.model.occ.addCircleArc: 1,
    gmsh.model.occ.add_circle_arc: 1,
    gmsh.model.occ.addEllipse: 1,
    gmsh.model.occ.add_ellipse: 1,
    gmsh.model.occ.addEllipseArc: 1,
    gmsh.model.occ.add_ellipse_arc: 1,
    gmsh.model.occ.addSpline: 1,
    gmsh.model.occ.add_spline: 1,
    gmsh.model.occ.addBSpline: 1,
    gmsh.model.occ.add_bspline: 1,
    gmsh.model.occ.addBezier: 1,
    gmsh.model.occ.add_bezier: 1,
    gmsh.model.occ.addWire: 1,
    gmsh.model.occ.add_wire: 1,
    # 2D (Surfaces)
    gmsh.model.occ.addPlaneSurface: 2,
    gmsh.model.occ.add_plane_surface: 2,
    gmsh.model.occ.addSurfaceFilling: 2,
    gmsh.model.occ.add_surface_filling: 2,
    gmsh.model.occ.addSurfaceLoop: 2,
    gmsh.model.occ.add_surface_loop: 2,
    gmsh.model.occ.addDisk: 2,
    gmsh.model.occ.add_disk: 2,
    gmsh.model.occ.addRectangle: 2,
    gmsh.model.occ.add_rectangle: 2,
    # 3D (Volumes)
    gmsh.model.occ.addBox: 3,
    gmsh.model.occ.add_box: 3,
    gmsh.model.occ.addSphere: 3,
    gmsh.model.occ.add_sphere: 3,
    gmsh.model.occ.addCylinder: 3,
    gmsh.model.occ.add_cylinder: 3,
    gmsh.model.occ.addCone: 3,
    gmsh.model.occ.add_cone: 3,
    gmsh.model.occ.addWedge: 3,
    gmsh.model.occ.add_wedge: 3,
    gmsh.model.occ.addTorus: 3,
    gmsh.model.occ.add_torus: 3,
    gmsh.model.occ.addVolume: 3,
    gmsh.model.occ.add_volume: 3,
}


class GMSH_entity:
    """Delayed evaluation of a gmsh occ kernel entity.

    Attributes:
        gmsh_partial_function: entity-defining function from model.occ
        gmsh_partial_function_kwargs: dict of keyword arguments for gmsh_partial_function
        physical_name: name(s) of the physical this entity will belong to
        mesh_order: priority of the entity if it overlaps with others (lower numbers override higher numbers)
        mesh_bool: if True, entity will be meshed; if not, will not be meshed

    """

    def __init__(
        self,
        gmsh_partial_function: callable,
        physical_name: str | tuple[str] | None = None,
        mesh_order: float | None = None,
        mesh_bool: bool = True,
        additive: bool = False,
        dimension: int | None = None,
    ):
        if not isinstance(gmsh_partial_function, partial):
            raise TypeError(
                "gmsh_partial_function must be a functools.partial object referencing a GMSH occ entity-defining function!"
            )
        self.gmsh_partial_function = gmsh_partial_function
        self.physical_name = format_physical_name(physical_name)
        self.mesh_order = mesh_order
        self.mesh_bool = mesh_bool
        self.additive = additive

        if gmsh_partial_function.func in GMSH_ENTITY_DIMENSIONS:
            self.dimension = GMSH_ENTITY_DIMENSIONS[gmsh_partial_function.func]
        else:
            if dimension is None:
                raise ValueError(
                    "For custom gmsh_partial_function, dimension must be specified!"
                )
            self.dimension = dimension

    def instanciate(self, cad_model: CAD):
        """Returns dim tag from entity.

        TODO: properly use cad_model instead of relying on general gmsh.model (being activated)
        """
        entity_output = self.gmsh_partial_function(
            *self.gmsh_partial_function.args, **self.gmsh_partial_function.keywords
        )
        entity_output = [(self.dimension, entity_output)]
        cad_model.model_manager.occ.synchronize()
        return entity_output
