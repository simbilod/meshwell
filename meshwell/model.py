from __future__ import annotations

from os import cpu_count
from typing import Dict, Optional

import gmsh
from meshwell.validation import validate_dimtags, unpack_dimtags, parse_entities
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
        point_tolerance: float = 1e-3,
        n_threads: int = cpu_count(),
    ):
        # Initialize model
        if gmsh.is_initialized():
            gmsh.clear()
        gmsh.initialize()

        # Set paralllel processing
        gmsh.option.setNumber("General.NumThreads", n_threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads1D", n_threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", n_threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", n_threads)

        # Point snapping
        self.point_tolerance = point_tolerance

        # CAD engine
        self.model = gmsh.model
        self.occ = self.model.occ

        # Track some gmsh entities for bottom-up volume definition
        self.points = {}
        self.segments = {}

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
        resolutions: Optional[Dict] = None,
        default_characteristic_length: float = 0.5,
        global_scaling: float = 1.0,
        global_2D_algorithm: int = 6,
        global_3D_algorithm: int = 1,
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
            global_scaling: factor to scale all mesh coordinates by (e.g. 1E-6 to go from um to m)
            global_2D_algorithm: gmsh surface default meshing algorithm, see https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options
            global_3D_algorithm: gmsh volume default meshing algorithm, see https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options
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
        gmsh.option.setNumber("Mesh.Algorithm", global_2D_algorithm)
        gmsh.option.setNumber("Mesh.Algorithm3D", global_3D_algorithm)

        # Initial synchronization
        self.occ.synchronize()

        # Parse dimensionality of entries
        entities_3D, entities_2D, entities_1D, entities_0D = parse_entities(entities_dict)
        
        # Parse mesh dimension
        if entities_3D:
            max_dim = 3
        elif entities_2D:
            max_dim = 2
        elif entities_1D:
            max_dim = 1
        else:
            max_dim = 0

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
                    # Make sure the most up-to-date surfaces are logged as boundaries
                    previous_entities.update_boundaries()
                current_entities.dimtags = list(set(current_dimtags_cut))
            current_entities.update_boundaries()
            if current_entities.dimtags:
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
                self.model.occ.remove(entity.dimtags, recursive=True)

        # Perform refinement
        refinement_fields = []
        refinement_index = 0
        refinement_fields_constant, refinement_index = constant_refinement(
            final_entity_list, refinement_field_index=0, model=self.model
        )
        refinement_fields.extend(refinement_fields_constant)

        # Use the smallest element size overall
        self.model.mesh.field.add("Min", refinement_index)
        self.model.mesh.field.setNumbers(
            refinement_index, "FieldsList", refinement_fields
        )
        self.model.mesh.field.setAsBackgroundMesh(refinement_index)

        # Turn off default meshing options
        self.model.mesh.MeshSizeFromPoints = 0
        self.model.mesh.MeshSizeFromCurvature = 0
        self.model.mesh.MeshSizeExtendFromBoundary = 0

        # Global resizing
        gmsh.option.setNumber("Mesh.ScalingFactor", global_scaling)

        self.occ.synchronize()
        self.model.mesh.generate(max_dim)

        if filename:
            gmsh.write(f"{filename}")

        with contextlib.redirect_stdout(None):
            with tempfile.TemporaryDirectory() as tmpdirname:
                gmsh.write(f"{tmpdirname}/mesh.msh")
                if finalize:
                    gmsh.finalize()
                return meshio.read(f"{tmpdirname}/mesh.msh")
