from __future__ import annotations

from os import cpu_count
from typing import Dict, Optional, List, Tuple
import numpy as np

from pathlib import Path

import gmsh
from meshwell.validation import (
    validate_dimtags,
    unpack_dimtags,
    order_entities,
    consolidate_entities_by_physical_name,
)
from meshwell.labeledentity import LabeledEntities
from meshwell.tag import tag_entities, tag_interfaces, tag_boundaries

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
        gmsh.option.setNumber("Geometry.OCCParallel", 1)

        # Point snapping
        self.point_tolerance = point_tolerance

        # CAD engine
        self.model = gmsh.model
        self.occ = self.model.occ

        # Track some gmsh entities for bottom-up volume definition
        self.points: Dict[Tuple[float, float, float], int] = {}
        self.segments: Dict[
            Tuple[Tuple[float, float, float], Tuple[float, float, float]], int
        ] = {}

    def add_get_point(self, x: float, y: float, z: float) -> int:
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

    def add_get_segment(
        self, xyz1: Tuple[float, float, float], xyz2: Tuple[float, float, float]
    ) -> int:
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

    def _channel_loop_from_vertices(
        self, vertices: List[Tuple[float, float, float]]
    ) -> int:
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

    def add_surface(self, vertices: List[Tuple[float, float, float]]) -> int:
        """Add a surface composed of the segments formed by vertices.

        Args:
            vertices: List of xyz coordinates, whose subsequent entries define a closed loop.
        Returns:
            ID of the added surface
        """
        channel_loop = self._channel_loop_from_vertices(vertices)
        try:
            return self.occ.add_plane_surface([channel_loop])
        except Exception as e:
            print("Failed vertices:")
            print(vertices)
            raise e

    def sync_model(self):
        """Synchronize the CAD model, and update points and lines vertex mapping."""
        self.occ.synchronize()
        cad_points = self.model.getEntities(dim=0)
        new_points: Dict[Tuple[float, float, float], int] = {}
        for _, cad_point in cad_points:
            # OCC kernel can do whatever, so just read point coordinates
            vertices = tuple(self.model.getValue(0, cad_point, []))
            new_points[vertices] = cad_point
        cad_lines = self.model.getEntities(dim=1)
        new_segments: Dict[
            Tuple[Tuple[float, float, float], Tuple[float, float, float]], int
        ] = {}
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
        entities_list: List,
        background_remeshing_file: Optional[Path] = None,
        default_characteristic_length: float = 0.5,
        global_scaling: float = 1.0,
        global_2D_algorithm: int = 6,
        global_3D_algorithm: int = 1,
        filename: Optional[str | Path] = None,
        verbosity: Optional[int] = 5,
        progress_bars: bool = True,
        interface_delimiter: str = "___",
        boundary_delimiter: str = "None",
        addition_delimiter: str = "+",
        addition_intersection_physicals: bool = True,
        addition_addition_physicals: bool = True,
        addition_structural_physicals: bool = True,
        gmsh_version: Optional[float] = None,
        finalize: bool = True,
        reinitialize: bool = True,
        periodic_entities: List[Tuple[str, str]] = None,
        fuse_entities_by_name: bool = False,
        optimization_flags: tuple[tuple[str, int]] | None = None,
    ) -> meshio.Mesh:
        """Creates a GMSH mesh with proper physical tagging from a dict of {labels: list( (GMSH entity dimension, GMSH entity tag) )}.

        Args:
            entities_list: list of meshwell entities (GMSH_entity, Prism, or PolySurface)
            background_remeshing (Path): path to a .pos file for background remeshing. If not None, is used instead of entity resolutions.
            default_characteristic_length (float): if resolutions is not specified for this physical, will use this value instead
            global_scaling: factor to scale all mesh coordinates by (e.g. 1E-6 to go from um to m)
            global_2D_algorithm: gmsh surface default meshing algorithm, see https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options
            global_3D_algorithm: gmsh volume default meshing algorithm, see https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options
            filename (str, path): if True, filepath where to save the mesh
            verbosity: GMSH stdout while meshing (True or False)
            interface_delimiter: string characters to use when naming interfaces between entities
            boundary_delimiter: string characters to use when defining an interface between an entity and nothing (simulation boundary)
            addition_delimiter: string characters to use with addition_intersection_physicals
            addition_addition_physicals: if True, the intersecting additive entities also carry the additive physical name
            addition_structural_physicals: if True, the intersecting additive entities also carry the underlying entity names
            addition_intersection_physicals: if True, the intersecting additive entities also carry a unique {underlying}{addition_delimiter}{additive} name
            gmsh_version: Gmsh mesh format version. For example, Palace requires an older version of 2.2,
                see https://mfem.org/mesh-formats/#gmsh-mesh-formats.
            finalize: if True (default), finalizes the GMSH model after execution
            periodic_entities: enforces mesh periodicity between the physical entities
            fuse_entities_by_name: if True, fuses CAD entities sharing the same physical_name
            optimization_flags: list of (method, niters) for smoothing. See https://gitlab.onelab.info/gmsh/gmsh/blob/gmsh_4_12_2/api/gmsh.py#L2087

        Returns:
            meshio object with mesh information
        """
        # Parse filename
        filename = str(filename)

        # If background mesh, create separate model
        if background_remeshing_file:
            # path = os.path.dirname(os.path.abspath(__file__))
            gmsh.merge(background_remeshing_file)
            gmsh.model.add("temp")

        gmsh.option.setNumber("General.Terminal", 10)  # 1 verbose, 0 otherwise
        gmsh.option.setNumber(
            "Mesh.CharacteristicLengthMax", default_characteristic_length
        )
        gmsh.option.setNumber("Mesh.Algorithm", global_2D_algorithm)
        gmsh.option.setNumber("Mesh.Algorithm3D", global_3D_algorithm)
        if gmsh_version is not None:
            gmsh.option.setNumber("Mesh.MshFileVersion", gmsh_version)

        # Initial synchronization
        self.occ.synchronize()

        # Parse and order the entities
        structural_entities = [
            entity for entity in entities_list if entity.additive is False
        ]
        additive_entities = [
            entity for entity in entities_list if entity.additive is True
        ]
        structural_entities = order_entities(structural_entities)
        additive_entities = order_entities(additive_entities)

        # Preserve ID numbering
        gmsh.option.setNumber("Geometry.OCCBooleanPreserveNumbering", 1)

        # Main loop:
        # Iterate through list of entities, generating and logging the volumes/surfaces in order
        # Manually remove intersections so that BooleanFragments from removeAllDuplicates does not reassign entity tags
        structural_entity_list = []
        max_dim = 0

        enumerator = enumerate(structural_entities)
        if progress_bars:
            from tqdm.auto import tqdm

            enumerator = tqdm(list(enumerator))

        for index, entity_obj in enumerator:
            physical_name = entity_obj.physical_name
            keep = entity_obj.mesh_bool
            resolutions = entity_obj.resolutions
            if progress_bars:
                if physical_name:
                    enumerator.set_description(
                        f"{str(physical_name):<30} - {'instanciate':<15}"
                    )
            # First create the shape
            dimtags_out = entity_obj.instanciate()

            if progress_bars:
                if physical_name:
                    enumerator.set_description(
                        f"{str(physical_name):<30} - {'dimtags':<15}"
                    )

            # Parse dimension
            dim = validate_dimtags(dimtags_out)
            max_dim = max(dim, max_dim)
            dimtags = unpack_dimtags(dimtags_out)

            if progress_bars:
                if physical_name:
                    enumerator.set_description(
                        f"{str(physical_name):<30} - {'entities':<15}"
                    )
            # Assemble with other shapes
            current_entities = LabeledEntities(
                index=index,
                dimtags=dimtags,
                physical_name=physical_name,
                keep=keep,
                model=self.model,
                resolutions=resolutions,
            )
            if progress_bars:
                if physical_name:
                    enumerator.set_description(
                        f"{str(physical_name):<30} - {'boolean':<15}"
                    )
            if index != 0:
                cut = self.occ.cut(
                    current_entities.dimtags,
                    [
                        dimtag
                        for previous_entities in structural_entity_list
                        for dimtag in previous_entities.dimtags
                    ],
                    removeObject=True,  # Only keep the difference
                    removeTool=False,  # Tool (previous entities) should remain untouched
                )
                # Heal interfaces now that there are no volume conflicts
                if progress_bars:
                    if physical_name:
                        enumerator.set_description(
                            f"{str(physical_name):<30} - {'duplicates':<15}"
                        )
                self.occ.removeAllDuplicates()
                if progress_bars:
                    if physical_name:
                        enumerator.set_description(
                            f"{str(physical_name):<30} - {'sync':<15}"
                        )
                self.sync_model()
                current_entities.dimtags = list(set(cut[0]))
            if current_entities.dimtags:
                structural_entity_list.append(current_entities)

        # Make sure the most up-to-date surfaces are logged as boundaries
        consolidated_entity_list = consolidate_entities_by_physical_name(
            structural_entity_list
        )
        structural_entity_list = []
        if fuse_entities_by_name:
            for entities in consolidated_entity_list:
                if len(entities.dimtags) != 1:
                    entities.dimtags = self.occ.fuse(
                        [entities.dimtags[0]],
                        entities.dimtags[1:],
                        removeObject=True,
                        removeTool=True,
                    )[0]
                    self.occ.synchronize()
                structural_entity_list.append(entities)
        else:
            structural_entity_list = consolidated_entity_list
        for entity in structural_entity_list:
            entity.update_boundaries()

        # Now that the structure is defined, merge with additive entities
        if additive_entities:
            for additive_entity in additive_entities:
                additive_dimtags = unpack_dimtags(additive_entity.instanciate())
                additive_entity_resolutions = (
                    []
                    if additive_entity.resolutions is None
                    else additive_entity.resolutions
                )
                updated_entities = []
                # Parse additive physical names
                for index, structural_entity in enumerate(structural_entity_list):
                    removeTool = False if index + 1 < len(structural_entities) else True

                    # Add all physical names
                    additive_names = []
                    if addition_addition_physicals:
                        additive_names.extend(additive_entity.physical_name)
                    if addition_structural_physicals:
                        additive_names.extend(structural_entity.physical_name)
                    if addition_intersection_physicals:
                        additive_names.extend(
                            [
                                f"{x}{addition_delimiter}{y}"
                                for x in structural_entity.physical_name
                                for y in additive_entity.physical_name
                            ]
                        )
                    additive_names = tuple(additive_names)

                    # Add the intersection and complement as new LabeledEntities
                    intersection = self.occ.intersect(
                        structural_entity.dimtags,
                        additive_dimtags,
                        removeObject=False,  # Don't remove yet
                        removeTool=removeTool,  # Only remove at the end
                    )
                    # No overlap case (no intersection)
                    if not intersection[0]:
                        self.occ.synchronize()
                        updated_entities.append(structural_entity)
                    else:
                        complement = self.occ.cut(
                            structural_entity.dimtags,
                            intersection[0],
                            removeObject=False,
                            removeTool=False,
                        )
                        self.occ.synchronize()
                        # Add the intersection and complement as new LabeledEntities
                        structural_entity_resolutions = (
                            []
                            if structural_entity.resolutions is None
                            else structural_entity.resolutions
                        )
                        if complement[0]:
                            updated_entities.append(
                                LabeledEntities(
                                    index=structural_entity.index,
                                    dimtags=complement[0],
                                    physical_name=structural_entity.physical_name,
                                    keep=structural_entity.keep,
                                    model=self.model,
                                    resolutions=structural_entity_resolutions,
                                )
                            )
                        updated_entities.append(
                            LabeledEntities(
                                index=structural_entity.index,
                                dimtags=intersection[0],
                                physical_name=additive_names,
                                keep=structural_entity.keep,
                                model=self.model,
                                resolutions=structural_entity_resolutions
                                + additive_entity_resolutions,
                            )
                        )
                # Use the updated entities in the next iteration
                structural_entity_list = updated_entities
                self.occ.removeAllDuplicates()
                self.sync_model()
        final_entity_list = structural_entity_list

        # Since we only took intersection with non-overlapping structural entities and removed the tool, there is no further conflict / tag reassignment
        # self.occ.removeAllDuplicates()
        for entity in final_entity_list:
            entity.update_boundaries()

        # Tag entities, interfaces, and boundaries
        tag_entities(final_entity_list)
        final_entity_list = tag_interfaces(
            final_entity_list, max_dim, interface_delimiter
        )
        tag_boundaries(
            final_entity_list, max_dim, interface_delimiter, boundary_delimiter
        )

        # Enforce periodic boundaries
        mapping = {}
        for dimtag in self.model.getPhysicalGroups():
            mapping[self.model.getPhysicalName(dimtag[0], dimtag[1])] = dimtag
        if periodic_entities:
            for label1, label2 in periodic_entities:
                if label1 not in mapping or label2 not in mapping:
                    continue
                tags1 = self.model.getEntitiesForPhysicalGroup(*mapping[label1])
                tags2 = self.model.getEntitiesForPhysicalGroup(*mapping[label2])

                vector1 = self.occ.getCenterOfMass(mapping[label1][0], tags1[0])
                vector2 = self.occ.getCenterOfMass(mapping[label1][0], tags2[0])
                vector = np.subtract(vector1, vector2)
                self.model.mesh.setPeriodic(
                    mapping[label1][0],
                    tags1,
                    tags2,
                    [
                        1,
                        0,
                        0,
                        vector[0],
                        0,
                        1,
                        0,
                        vector[1],
                        0,
                        0,
                        1,
                        vector[2],
                        0,
                        0,
                        0,
                        1,
                    ],
                )

        gmsh.write("cad.xao")

        # Perform refinement
        if background_remeshing_file is None:
            # Use entity information
            refinement_field_indices = []
            refinement_max_index = 0

            # Hashable final_entity_list
            final_entity_dict = {
                entity.physical_name: entity for entity in final_entity_list
            }

            for entity in final_entity_list:
                (
                    refinement_field_indices,
                    refinement_max_index,
                ) = entity.add_refinement_fields_to_model(
                    refinement_field_indices,
                    refinement_max_index,
                    default_characteristic_length,
                    final_entity_dict,
                    boundary_delimiter,
                )

            # Use the smallest element size overall
            self.model.mesh.field.add("Min", refinement_max_index)
            self.model.mesh.field.setNumbers(
                refinement_max_index, "FieldsList", refinement_field_indices
            )
            self.model.mesh.field.setAsBackgroundMesh(refinement_max_index)
        else:
            bg_field = self.model.mesh.field.add("PostView")
            self.model.mesh.field.setNumber(bg_field, "ViewIndex", 0)
            gmsh.model.mesh.field.setAsBackgroundMesh(bg_field)

        # Remove boundary entities
        for entity in final_entity_list:
            if not entity.keep:
                self.model.occ.remove(entity.dimtags, recursive=True)

        # Turn off default meshing options
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

        # Global resizing
        gmsh.option.setNumber("Mesh.ScalingFactor", global_scaling)

        self.occ.synchronize()

        if not filename.endswith((".step", ".stp")):
            if global_3D_algorithm == 1 and verbosity:
                gmsh.logger.start()
            self.model.mesh.generate(max_dim)

            # Mesh smoothing
            if optimization_flags is not None:
                for optimization_flag, niter in optimization_flags:
                    self.model.mesh.optimize(optimization_flag, niter=niter)

            if global_3D_algorithm == 1 and verbosity:
                for line in gmsh.logger.get():
                    if "Optimizing volume " in str(line):
                        number = int(str(line).split("Optimizing volume ")[1])
                        physicalTags = gmsh.model.getPhysicalGroupsForEntity(3, number)
                        physicals = []
                        if len(physicalTags):
                            for p in physicalTags:
                                physicals.append(gmsh.model.getPhysicalName(dim, p))
                    if "ill-shaped tets are" in str(line):
                        print(",".join(physicals))
                        print(str(line))

        if filename:
            gmsh.write(f"{filename}")

        if not filename.endswith((".step", ".stp")):
            with contextlib.redirect_stdout(None):
                with tempfile.TemporaryDirectory() as tmpdirname:
                    gmsh.write(f"{tmpdirname}/mesh.msh")
                    if finalize:
                        gmsh.finalize()
                    return meshio.read(f"{tmpdirname}/mesh.msh")
