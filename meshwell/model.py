from __future__ import annotations

from typing import Dict, Optional

import gmsh
from meshwell.validation import validate_dimtags, unpack_dimtags
from meshwell.entity import LabeledEntities
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
    ):
        # Initialize model
        if not gmsh.isInitialized():
            gmsh.initialize()

        # Track some gmsh entities for bottom-up volume definition
        self.points = {}
        self.segments = {}

    def _add_get_point(self, x, y, z):
        """Add a point to the model, or reuse a previously-defined point.
        Args:
            x: float, x-coordinate
            y: float, y-coordinate
            z: float, z-coordinate
        Returns:
            ID of the added or retrieved point
        """
        if (x, y, z) not in self.points.keys():
            self.points[(x, y, z)] = gmsh.model.occ.add_point(x, y, z)
        return self.points[(x, y, z)]

    def _add_get_segment(self, xyz1, xyz2):
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
            self.segments[(xyz1, xyz2)] = gmsh.model.occ.add_line(
                self._add_get_point(xyz1[0], xyz1[1], xyz1[2]),
                self._add_get_point(xyz2[0], xyz2[1], xyz2[2]),
            )
            return self.segments[(xyz1, xyz2)]

    def _channel_loop_from_vertices(self, vertices):
        """Add a curve loop from the list of vertices.
        Args:
            model: GMSH model
            vertices: list of [x,y,z] coordinates
        Returns:
            ID of the added curve loop
        """
        edges = []
        for vertex1, vertex2 in [
            (vertices[i], vertices[i + 1]) for i in range(len(vertices) - 1)
        ]:
            gmsh_line = self._add_get_segment(vertex1, vertex2)
            edges.append(gmsh_line)
        return gmsh.model.occ.add_curve_loop(edges)

    def _add_surface(self, vertices):
        """Add a surface composed of the segments formed by vertices.

        Args:
            vertices: List of xyz coordinates, whose subsequent entries define a closed loop.
        Returns:
            ID of the added surface
        """
        channel_loop = self._channel_loop_from_vertices(vertices)
        return gmsh.model.occ.add_plane_surface([channel_loop])

    def mesh(
        self,
        dimtags_dict: OrderedDict,
        resolutions: Optional[Dict] = None,
        default_characteristic_length: float = 0.5,
        filename: Optional[str] = None,
        verbosity: Optional[bool] = False,
        interface_delimiter: str = "___",
        boundary_delimiter: str = "None",
        finalize: bool = True,
    ):
        """Creates a GMSH mesh with proper physical tagging from a dict of {labels: list( (GMSH entity dimension, GMSH entity tag) )}.

        Args:
            dimtags_dict: OrderedDict of key: physical name, and value: (dim, tags) of  GMSH entities
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
        gmsh.option.setNumber(
            "Mesh.CharacteristicLengthMax", default_characteristic_length
        )

        # Initial synchronization
        gmsh.model.occ.synchronize()

        # Validate and unpack dim tags, and detect mesh dimension
        max_dim = 0
        for label, dimtags in dimtags_dict.items():
            dim = validate_dimtags(dimtags)
            max_dim = max(dim, max_dim)
            dimtags_dict[label] = unpack_dimtags(dimtags)

        # Preserve ID numbering
        gmsh.option.setNumber("Geometry.OCCBooleanPreserveNumbering", 1)

        # Iterate through OrderedDict of entities, logging the volumes/surfaces in order
        final_entity_list = []
        for index, (label, dimtags) in enumerate(dimtags_dict.items()):
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
            )
            if index != 0:
                current_dimtags_cut = []
                for current_dimtags in current_entities.dimtags:
                    for previous_entities in final_entity_list:
                        if cut := gmsh.model.occ.cut(
                            [current_dimtags],
                            previous_entities.dimtags,
                            removeObject=True,  # Only keep the difference
                            removeTool=False,  # Tool (previous entities) should remain untouched
                        ):
                            current_dimtags_cut.append(cut[0])
                        gmsh.model.occ.synchronize()
                        gmsh.model.occ.removeAllDuplicates()  # Heal interfaces now
                        gmsh.model.occ.synchronize()
                        previous_entities.update_boundaries()
                current_entities.dimtags = [
                    item for sublist in current_dimtags_cut for item in sublist
                ]
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

        # Perform refinement
        refinement_fields = []
        refinement_index = 0
        refinement_fields_constant, refinement_index = constant_refinement(
            final_entity_list, refinement_field_index=0
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

        gmsh.model.occ.synchronize()

        gmsh.option.setNumber(
            "General.Terminal", 1 if verbosity else 0
        )  # 1 verbose, 0 otherwise
        gmsh.model.mesh.generate(max_dim)

        if filename:
            gmsh.write(f"{filename}")

        with contextlib.redirect_stdout(None):
            with tempfile.TemporaryDirectory() as tmpdirname:
                gmsh.write(f"{tmpdirname}/mesh.msh")
                if finalize:
                    gmsh.finalize()
                return meshio.read(f"{tmpdirname}/mesh.msh")