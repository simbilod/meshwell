from __future__ import annotations

from typing import Dict, Optional

import gmsh
from meshwell.validation import validate_dimtags, unpack_dimtags
from meshwell.labeledentity import LabeledEntities
from meshwell.tag import tag_entities, tag_interfaces, tag_boundaries
from meshwell.refinement import constant_refinement
from collections import OrderedDict

import contextlib
import tempfile
import meshio


class Model:
    """Model class."""

    def __init__(
        self,
        point_tolerance=1e-3,
    ):
        # Initialize model
        gmsh.clear()
        gmsh.initialize()

        # Point snapping
        self.point_tolerance = point_tolerance

        # CAD engine
        self.model = gmsh.model
        self.occ = gmsh.model.occ

        # Track some gmsh entities for bottom-up volume definition
        self.points = {}
        self.segments = {}
        # self.points_coordinates = np.empty((1,3))
        # self.points_entities = []
        # self.segment_coordinates =  np.empty((1,6))
        # self.segment_entities = []

    # def find_point(self, xyz):
    #     """Returns an existing point entity if the provided point coordinates are within atol of the entity coordinates.

    #     TODO
    #     """
    #     diffs = np.array(xyz) - self.points_coordinates
    #     mask = np.where(diffs < self.point_tolerance)

    def add_get_point(self, x, y, z):
        """Add a point to the model, or reuse a previously-defined point.
        Args:
            x: float, x-coordinate
            y: float, y-coordinate
            z: float, z-coordinate
        Returns:
            ID of the added or retrieved point
        """
        if (x, y, z) not in self.points.keys():
            self.points[(x, y, z)] = self.occ.add_point(x, y, z)
        return self.points[(x, y, z)]

    def add_get_segment(self, xyz1, xyz2):
        """Add a segment (2-point line) to the gmsh model, or retrieve a previously-defined segment.
        The OCC kernel does not care about orientation.
        Args:
            xyz1: first [x,y,z] coordinate
            xyz2: second [x,y,z] coordinate
        Returns:
            ID of the added or retrieved line segment
        """
        if (xyz1, xyz2) in self.segments.keys():
            return self.segments[(xyz1, xyz2)]
        elif (xyz2, xyz1) in self.segments.keys():
            return self.segments[(xyz2, xyz1)]
        else:
            self.segments[(xyz1, xyz2)] = self.occ.add_line(
                self.add_get_point(xyz1[0], xyz1[1], xyz1[2]),
                self.add_get_point(xyz2[0], xyz2[1], xyz2[2]),
            )
            return self.segments[(xyz1, xyz2)]

    def _channel_loop_from_vertices(self, vertices):
        """Add a curve loop from the list of vertices.
        Args:
            vertices: list of [x,y,z] coordinates
        Returns:
            ID of the added curve loop
        """
        edges = []
        for vertex1, vertex2 in [
            (vertices[i], vertices[i + 1]) for i in range(len(vertices) - 1)
        ]:
            gmsh_line = self.add_get_segment(vertex1, vertex2)
            edges.append(gmsh_line)
        return self.occ.add_curve_loop(edges)

    def add_surface(self, vertices):
        """Add a surface composed of the segments formed by vertices.

        Args:
            vertices: List of xyz coordinates, whose subsequent entries define a closed loop.
        Returns:
            ID of the added surface
        """
        channel_loop = self._channel_loop_from_vertices(vertices)
        return self.occ.add_plane_surface([channel_loop])

    def sync_model(self):
        """Synchronize the CAD model, and update points and lines vertex mapping."""
        self.occ.synchronize()
        cad_points = self.model.getEntities(dim=0)
        new_points = {}
        for _, cad_point in cad_points:
            # OCC kernel can do whatever, so just read point coordinates
            vertices = tuple(self.model.getValue(0, cad_point, []))
            new_points[vertices] = cad_point
        cad_lines = self.model.getEntities(dim=1)
        new_segments = {}
        for _, cad_line in cad_lines:
            try:
                key = list(self.segments.keys())[
                    list(self.segments.values()).index(cad_line)
                ]
                cad_lines[key] = cad_line
            except Exception:
                continue
        self.points = new_points
        self.segments = new_segments

    def mesh(
        self,
        entities_dict: OrderedDict,
        boundaries_dict: Dict = None,
        resolutions: Optional[Dict] = None,
        default_characteristic_length: float = 0.5,
        filename: Optional[str] = None,
        verbosity: Optional[int] = 0,
        interface_delimiter: str = "___",
        boundary_delimiter: str = "None",
        finalize: bool = True,
    ):
        """Creates a GMSH mesh with proper physical tagging from a dict of {labels: list( (GMSH entity dimension, GMSH entity tag) )}.

        Args:
            entities_dict: OrderedDict of key: physical name, and value: meshwell entity
            resolutions (Dict): Pairs {"physical name": {"resolution": float, "distance": "float}}
            default_characteristic_length (float): if resolutions is not specified for this physical, will use this value instead
            filename (str, path): if True, filepath where to save the mesh
            verbosity: GMSH stdout while meshing (True or False)
            interface_delimiter: string characters to use when naming interfaces between entities
            boundary_delimiter: string characters to use when defining an interface between an entity and nothing (simulation boundary)
            finalize: if True (default), finalizes the GMSH model after execution

        Returns:
            meshio object with mesh information
        """
        resolutions = resolutions if resolutions else {}
        gmsh.option.setNumber("General.Terminal", verbosity)  # 1 verbose, 0 otherwise
        gmsh.option.setNumber(
            "Mesh.CharacteristicLengthMax", default_characteristic_length
        )

        # Initial synchronization
        self.occ.synchronize()

        # Parse strutural and boundary entities
        full_entities_dict = OrderedDict()
        keep = False
        if boundaries_dict is not None:
            for key, value in boundaries_dict.items():
                full_entities_dict[key] = (value, keep)
        keep = True
        for key, value in entities_dict.items():
            full_entities_dict[key] = (value, keep)

        # Preserve ID numbering
        gmsh.option.setNumber("Geometry.OCCBooleanPreserveNumbering", 1)

        # Main loop:
        # Iterate through OrderedDict of entities, generating and logging the volumes/surfaces in order
        # Manually remove intersections so that BooleanFragments from removeAllDuplicates does not reassign entity tags
        final_entity_list = []
        max_dim = 0
        for index, (label, (entity_obj, keep)) in enumerate(full_entities_dict.items()):
            # First create the shape
            dimtags_out = entity_obj.instanciate()
            # Parse dimension
            dim = validate_dimtags(dimtags_out)
            max_dim = max(dim, max_dim)
            dimtags = unpack_dimtags(dimtags_out)

            # Assemble with other shapes
            base_resolution = (
                resolutions[label]["resolution"]
                if label in resolutions
                else default_characteristic_length
            )
            current_entities = LabeledEntities(
                index=index,
                dimtags=dimtags,
                label=label,
                base_resolution=base_resolution,
                keep=keep,
                model=self.model,
            )
            if index != 0:
                current_dimtags_cut = []
                for current_dimtags in current_entities.dimtags:
                    for previous_entities in final_entity_list:
                        for previous_dimtags in previous_entities.dimtags:
                            if cut := self.occ.cut(
                                [current_dimtags],
                                [previous_dimtags],
                                removeObject=True,  # Only keep the difference
                                removeTool=False,  # Tool (previous entities) should remain untouched
                            ):
                                current_dimtags_cut.extend(cut[0])
                            self.sync_model()
                    # Heal interfaces now that there are no volume conflicts
                    self.occ.removeAllDuplicates()
                    self.sync_model()
                    previous_entities.update_boundaries()
                current_entities.dimtags = current_dimtags_cut
            current_entities.update_boundaries()
            final_entity_list.append(current_entities)

        # Tag entities, interfaces, and boundaries
        tag_entities(final_entity_list)
        final_entity_list = tag_interfaces(
            final_entity_list, max_dim, interface_delimiter
        )
        tag_boundaries(
            final_entity_list, max_dim, interface_delimiter, boundary_delimiter
        )

        # Remove boundary entities
        for entity in final_entity_list:
            if not entity.keep:
                gmsh.model.occ.remove(entity.dimtags, recursive=True)

        # Perform refinement
        refinement_fields = []
        refinement_index = 0
        refinement_fields_constant, refinement_index = constant_refinement(
            final_entity_list, refinement_field_index=0, model=self.model
        )
        refinement_fields.extend(refinement_fields_constant)

        # Use the smallest element size overall
        gmsh.model.mesh.field.add("Min", refinement_index)
        gmsh.model.mesh.field.setNumbers(
            refinement_index, "FieldsList", refinement_fields
        )
        gmsh.model.mesh.field.setAsBackgroundMesh(refinement_index)

        # Turn off default meshing options
        gmsh.model.mesh.MeshSizeFromPoints = 0
        gmsh.model.mesh.MeshSizeFromCurvature = 0
        gmsh.model.mesh.MeshSizeExtendFromBoundary = 0

        self.occ.synchronize()
        gmsh.model.mesh.generate(max_dim)

        if filename:
            gmsh.write(f"{filename}")

        with contextlib.redirect_stdout(None):
            with tempfile.TemporaryDirectory() as tmpdirname:
                gmsh.write(f"{tmpdirname}/mesh.msh")
                if finalize:
                    gmsh.finalize()
                return meshio.read(f"{tmpdirname}/mesh.msh")
