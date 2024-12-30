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
        verbosity: Optional[int] = 0,
        progress_bars: bool = False,
        interface_delimiter: str = "___",
        boundary_delimiter: str = "None",
        addition_delimiter: str = "+",
        addition_intersection_physicals: bool = True,
        addition_addition_physicals: bool = True,
        addition_structural_physicals: bool = True,
        gmsh_version: Optional[float] = None,
        finalize: bool = True,
        periodic_entities: List[Tuple[str, str]] = None,
        fuse_entities_by_name: bool = False,
        optimization_flags: tuple[tuple[str, int]] | None = None,
    ) -> meshio.Mesh:
        """Creates a GMSH mesh with proper physical tagging."""
        # Initialize mesh settings
        self._initialize_mesh_settings(
            verbosity,
            default_characteristic_length,
            global_2D_algorithm,
            global_3D_algorithm,
            gmsh_version,
        )

        # Process background mesh if provided
        if background_remeshing_file:
            self._handle_background_mesh(background_remeshing_file)

        # Process entities and get max dimension
        final_entity_list, max_dim = self._process_entities(
            entities_list,
            progress_bars,
            fuse_entities_by_name,
            addition_delimiter,
            addition_intersection_physicals,
            addition_addition_physicals,
            addition_structural_physicals,
        )

        # Tag entities and boundaries
        self._tag_mesh_components(
            final_entity_list,
            max_dim,
            interface_delimiter,
            boundary_delimiter,
        )

        # Handle periodic boundaries if specified
        if periodic_entities:
            self._apply_periodic_boundaries(final_entity_list, periodic_entities)

        # Apply mesh refinement
        self._apply_mesh_refinement(
            background_remeshing_file,
            final_entity_list,
            boundary_delimiter,
        )

        # Generate and return mesh
        return self._generate_final_mesh(
            filename,
            max_dim,
            global_3D_algorithm,
            global_scaling,
            verbosity,
            optimization_flags,
            finalize,
        )

    def _initialize_mesh_settings(
        self,
        verbosity: int,
        default_characteristic_length: float,
        global_2D_algorithm: int,
        global_3D_algorithm: int,
        gmsh_version: Optional[float],
    ) -> None:
        """Initialize basic mesh settings."""
        gmsh.option.setNumber("General.Terminal", verbosity)
        gmsh.option.setNumber(
            "Mesh.CharacteristicLengthMax", default_characteristic_length
        )
        gmsh.option.setNumber("Mesh.Algorithm", global_2D_algorithm)
        gmsh.option.setNumber("Mesh.Algorithm3D", global_3D_algorithm)
        if gmsh_version is not None:
            gmsh.option.setNumber("Mesh.MshFileVersion", gmsh_version)
        self.occ.synchronize()

    def _handle_background_mesh(self, background_remeshing_file: Path) -> None:
        """Handle background mesh file if provided."""
        gmsh.merge(str(background_remeshing_file))
        gmsh.model.add("temp")

    def _process_entities(
        self,
        entities_list: List,
        progress_bars: bool,
        fuse_entities_by_name: bool,
        addition_delimiter: str,
        addition_intersection_physicals: bool,
        addition_addition_physicals: bool,
        addition_structural_physicals: bool,
    ) -> Tuple[List, int]:
        """Process structural and additive entities."""
        # Separate and order entities
        structural_entities = [e for e in entities_list if not e.additive]
        additive_entities = [e for e in entities_list if e.additive]
        structural_entities = order_entities(structural_entities)
        additive_entities = order_entities(additive_entities)

        # Process structural entities
        structural_entity_list, max_dim = self._process_structural_entities(
            structural_entities, progress_bars
        )

        # Process additive entities if present
        if additive_entities:
            structural_entity_list = self._process_additive_entities(
                structural_entity_list,
                additive_entities,
                addition_delimiter,
                addition_intersection_physicals,
                addition_addition_physicals,
                addition_structural_physicals,
            )

        # Handle entity fusion if requested
        if fuse_entities_by_name:
            structural_entity_list = self._fuse_entities(structural_entity_list)

        return structural_entity_list, max_dim

    def _process_structural_entities(
        self, structural_entities: List, progress_bars: bool
    ) -> Tuple[List, int]:
        """Process structural entities and return entity list and max dimension."""
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

        # Update boundaries for all entities after substractions
        for entity in structural_entity_list:
            entity.update_boundaries()

        return structural_entity_list, max_dim

    def _process_additive_entities(
        self,
        structural_entity_list: List,
        additive_entities: List,
        addition_delimiter: str,
        addition_intersection_physicals: bool,
        addition_addition_physicals: bool,
        addition_structural_physicals: bool,
    ) -> List:
        """Process additive entities and return updated entity list."""
        structural_entities_length = len(structural_entity_list)
        for additive_entity in additive_entities:
            # Create additive entity geometry
            additive_dimtags = unpack_dimtags(additive_entity.instanciate())
            additive_entity_resolutions = (
                []
                if additive_entity.resolutions is None
                else additive_entity.resolutions
            )

            updated_entities = []
            # Process each structural entity
            for index, structural_entity in enumerate(structural_entity_list):
                # Only remove tool on last iteration
                removeTool = index + 1 >= structural_entities_length

                # Build list of physical names for intersection
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

                # Find intersection between structural and additive entities
                intersection = self.occ.intersect(
                    structural_entity.dimtags,
                    additive_dimtags,
                    removeObject=False,
                    removeTool=removeTool,
                )

                if not intersection[0]:
                    # No overlap case - keep original structural entity
                    self.occ.synchronize()
                    updated_entities.append(structural_entity)
                else:
                    # Cut out intersection from structural entity
                    complement = self.occ.cut(
                        structural_entity.dimtags,
                        intersection[0],
                        removeObject=False,
                        removeTool=False,
                    )
                    self.occ.synchronize()

                    # Add complement if it exists
                    if complement[0]:
                        updated_entities.append(
                            LabeledEntities(
                                index=structural_entity.index,
                                dimtags=complement[0],
                                physical_name=structural_entity.physical_name,
                                keep=structural_entity.keep,
                                model=self.model,
                                resolutions=structural_entity.resolutions,
                            )
                        )

                    # Add intersection with combined physical names
                    updated_entities.append(
                        LabeledEntities(
                            index=structural_entity.index,
                            dimtags=intersection[0],
                            physical_name=additive_names,
                            keep=structural_entity.keep,
                            model=self.model,
                            resolutions=(
                                structural_entity.resolutions
                                + additive_entity_resolutions
                            ),
                        )
                    )

            # Update entity list for next iteration
            structural_entity_list = updated_entities
            self.occ.removeAllDuplicates()
            self.sync_model()

        # Update boundaries for all entities after addition
        for entity in structural_entity_list:
            entity.update_boundaries()

        return structural_entity_list

    def _fuse_entities(
        self, entity_list: List[LabeledEntities]
    ) -> List[LabeledEntities]:
        """Fuse entities that share the same physical name.

        This method:
        1. Consolidates entities by their physical names
        2. For each group of entities with the same name:
        - If there's more than one entity, fuses them together
        - Maintains the first entity's properties (index, keep status, etc.)
        3. Updates boundaries for all entities after fusion

        Args:
            entity_list: List of LabeledEntities to potentially fuse

        Returns:
            List of LabeledEntities where entities with the same physical name have been fused
        """
        # First consolidate entities by their physical names
        consolidated_entities = consolidate_entities_by_physical_name(entity_list)

        fused_entities = []
        for entities in consolidated_entities:
            if len(entities.dimtags) > 1:
                # If we have multiple entities with the same name, fuse them
                entities.dimtags = self.occ.fuse(
                    [entities.dimtags[0]],  # First entity is the target
                    entities.dimtags[1:],  # Rest are tools to fuse into target
                    removeObject=True,  # Remove the target after fusion
                    removeTool=True,  # Remove the tools after fusion
                )[0]
                self.occ.synchronize()
            fused_entities.append(entities)

        # Update boundaries for all entities after fusion
        for entity in fused_entities:
            entity.update_boundaries()

        return fused_entities

    def _tag_mesh_components(
        self,
        final_entity_list: List,
        max_dim: int,
        interface_delimiter: str,
        boundary_delimiter: str,
    ) -> None:
        """Tag entities, interfaces, and boundaries."""
        tag_entities(final_entity_list)
        final_entity_list = tag_interfaces(
            final_entity_list, max_dim, interface_delimiter
        )
        tag_boundaries(
            final_entity_list, max_dim, interface_delimiter, boundary_delimiter
        )

    def _apply_periodic_boundaries(
        self, final_entity_list: List, periodic_entities: List[Tuple[str, str]]
    ) -> None:
        """Apply periodic boundary conditions."""
        mapping = {
            self.model.getPhysicalName(dimtag[0], dimtag[1]): dimtag
            for dimtag in self.model.getPhysicalGroups()
        }

        for label1, label2 in periodic_entities:
            if label1 not in mapping or label2 not in mapping:
                continue

            self._set_periodic_pair(mapping, label1, label2)

    def _set_periodic_pair(self, mapping: Dict, label1: str, label2: str) -> None:
        """Set up periodic boundary pair."""
        tags1 = self.model.getEntitiesForPhysicalGroup(*mapping[label1])
        tags2 = self.model.getEntitiesForPhysicalGroup(*mapping[label2])

        vector1 = self.occ.getCenterOfMass(mapping[label1][0], tags1[0])
        vector2 = self.occ.getCenterOfMass(mapping[label1][0], tags2[0])
        vector = np.subtract(vector1, vector2)

        self.model.mesh.setPeriodic(
            mapping[label1][0],
            tags1,
            tags2,
            [1, 0, 0, vector[0], 0, 1, 0, vector[1], 0, 0, 1, vector[2], 0, 0, 0, 1],
        )

    def _apply_mesh_refinement(
        self,
        background_remeshing_file: Optional[Path] | None,
        final_entity_list: List,
        boundary_delimiter: str,
    ) -> None:
        """Apply mesh refinement settings."""
        if background_remeshing_file is None:
            self._apply_entity_refinement(final_entity_list, boundary_delimiter)
        else:
            self._apply_background_refinement()

    def _apply_entity_refinement(
        self,
        final_entity_list: List[LabeledEntities],
        boundary_delimiter: str,
    ) -> None:
        """Apply mesh refinement based on entity information.

        Args:
            final_entity_list: List of LabeledEntities to process
            boundary_delimiter: String used to identify boundary entities
        """
        # Create dictionary for easier entity lookup
        final_entity_dict = {
            entity.physical_name: entity for entity in final_entity_list
        }

        # Collect all refinement fields
        refinement_field_indices = []
        for entity in final_entity_list:
            refinement_field_indices.extend(
                entity.add_refinement_fields_to_model(
                    final_entity_dict,
                    boundary_delimiter,
                )
            )

        # If we have refinement fields, create a minimum field
        if refinement_field_indices:
            # Use the smallest element size overall
            min_field_index = self.model.mesh.field.add("Min")
            self.model.mesh.field.setNumbers(
                min_field_index, "FieldsList", refinement_field_indices
            )
            self.model.mesh.field.setAsBackgroundMesh(min_field_index)

        # Turn off default meshing options
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    def _apply_background_refinement(self) -> None:
        """Apply mesh refinement based on background mesh."""
        # Create background field from post-processing view
        bg_field = self.model.mesh.field.add("PostView")
        self.model.mesh.field.setNumber(bg_field, "ViewIndex", 0)
        gmsh.model.mesh.field.setAsBackgroundMesh(bg_field)

        # Turn off default meshing options
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    def _generate_final_mesh(
        self,
        filename: str | Path,
        max_dim: int,
        global_3D_algorithm: int,
        global_scaling: float,
        verbosity: int,
        optimization_flags: tuple[tuple[str, int]] | None,
        finalize: bool,
    ) -> meshio.Mesh:
        """Generate the final mesh and return meshio object."""
        gmsh.option.setNumber("Mesh.ScalingFactor", global_scaling)

        if not str(filename).endswith((".step", ".stp")):
            if global_3D_algorithm == 1 and verbosity:
                gmsh.logger.start()

            self.model.mesh.generate(max_dim)

            if optimization_flags:
                for optimization_flag, niter in optimization_flags:
                    self.model.mesh.optimize(optimization_flag, niter=niter)

        if filename:
            gmsh.write(str(filename))

        with contextlib.redirect_stdout(None):
            with tempfile.TemporaryDirectory() as tmpdirname:
                temp_mesh_path = f"{tmpdirname}/mesh.msh"
                gmsh.write(temp_mesh_path)
                if finalize:
                    gmsh.finalize()
                return meshio.read(temp_mesh_path)
