from __future__ import annotations

from os import cpu_count
from typing import Dict, List, Tuple
from pathlib import Path
import gmsh

from meshwell.validation import (
    unpack_dimtags,
)
from meshwell.labeledentity import LabeledEntities
from meshwell.tag import tag_entities, tag_interfaces, tag_boundaries


class CAD:
    """CAD class for generating geometry and saving to .xao format."""

    def __init__(
        self,
        point_tolerance: float = 1e-3,
        n_threads: int = cpu_count(),
        filename: str = "temp",
    ):
        """Initialize CAD processor."""
        # Model initialization
        self.n_threads = n_threads
        self.point_tolerance = point_tolerance
        self.filename = Path(filename)

        # Track gmsh entities for bottom-up volume definition
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
        # Snap coordinates to point tolerance
        x = round(x / self.point_tolerance) * self.point_tolerance
        y = round(y / self.point_tolerance) * self.point_tolerance
        z = round(z / self.point_tolerance) * self.point_tolerance
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

    def wire_from_vertices(
        self, vertices: List[Tuple[float, float, float]], checkClosed=False
    ) -> int:
        """Add a wire from the list of vertices.
        Args:
            vertices: list of [x,y,z] coordinates
        Returns:
            ID of the added wire
        """
        edges = []
        for vertex1, vertex2 in [
            (vertices[i], vertices[i + 1]) for i in range(len(vertices) - 1)
        ]:
            gmsh_line = self.add_get_segment(vertex1, vertex2)
            edges.append(gmsh_line)
        if checkClosed:
            return self.occ.add_wire(edges, checkClosed=checkClosed)
        else:
            return edges

    def _channel_loop_from_vertices(
        self, vertices: List[Tuple[float, float, float]]
    ) -> int:
        return self.wire_from_vertices(vertices, checkClosed=True)

    def add_surface(self, vertices: List[Tuple[float, float, float]]) -> int:
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
            key = list(self.segments.keys())[
                list(self.segments.values()).index(cad_line)
            ]
            cad_lines[key] = cad_line
        self.points = new_points
        self.segments = new_segments

    def _instantiate_entity(
        self, index: int, entity_obj, progress_bars: bool
    ) -> LabeledEntities:
        """Common logic for instantiating entities."""
        physical_name = entity_obj.physical_name
        if progress_bars and physical_name:
            print(f"Processing {physical_name} - instantiation")

        # Instantiate entity
        dimtags_out = entity_obj.instanciate(self)
        dimtags = unpack_dimtags(dimtags_out)

        return LabeledEntities(
            index=index,
            dimtags=dimtags,
            physical_name=physical_name,
            keep=entity_obj.mesh_bool,
            model=self.model,
        )

    def _initialize_model(self):
        """Ensure GMSH is initialized before operations."""

        # Create model object
        self.model = gmsh.model
        self.occ = self.model.occ

        if gmsh.is_initialized():
            gmsh.finalize()
            gmsh.initialize()
        else:
            gmsh.initialize()

        # Clear model and points
        gmsh.clear()
        self.points = {}
        self.segments = {}

        self.model.add(str(self.filename))
        self.model.setFileName(str(self.filename))
        gmsh.option.setNumber("General.NumThreads", self.n_threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads1D", self.n_threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", self.n_threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", self.n_threads)
        gmsh.option.setNumber("Geometry.OCCParallel", 1)

    def _process_entities(
        self,
        entities_list: List,
        progress_bars: bool,
        addition_delimiter: str,
        addition_intersection_physicals: bool,
        addition_addition_physicals: bool,
        addition_structural_physicals: bool,
    ) -> Tuple[List, int]:
        """Process structural and additive entities."""
        # Separate and order entities
        structural_entities = [e for e in entities_list if not e.additive]
        additive_entities = [e for e in entities_list if e.additive]

        # Process structural entities
        structural_entity_list, max_dim = self._process_multidimensional_entities(
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

        return structural_entity_list, max_dim

    def _process_multidimensional_entities(
        self, structural_entities: List, progress_bars: bool
    ) -> Tuple[List, int]:
        """Process entities with mixed dimensions using hierarchical approach."""

        entity_dimensions = []
        max_dim = 0
        for index, entity_obj in enumerate(structural_entities):
            dim = entity_obj.dimension
            max_dim = max(dim, max_dim)
            entity_dimensions.append((dim, index, entity_obj))

        # Group entities by dimension and sort by mesh_order within each dimension
        dimension_groups = {0: [], 1: [], 2: [], 3: []}
        for dim, index, entity_obj in entity_dimensions:
            dimension_groups[dim].append((index, entity_obj))

        # Sort each dimension group by mesh_order
        for dim in dimension_groups:
            dimension_groups[dim].sort(
                key=lambda x: x[1].mesh_order
                if x[1].mesh_order is not None
                else float("inf")
            )

        # Process entities by dimension (highest first)
        all_processed_entities = []
        dim = max_dim

        while dim >= 0:
            if dimension_groups[dim]:
                # Process entities of same dimension in mesh_order sequence
                current_dimension_entities = self._process_dimension_group_cuts(
                    dimension_groups[dim], progress_bars
                )

                # Second pass: Fragment against higher dimensional entities
                if all_processed_entities:
                    current_dimension_entities = (
                        self._process_dimension_group_fragments(
                            current_dimension_entities,
                            progress_bars,
                            all_processed_entities,
                        )
                    )

                all_processed_entities.extend(current_dimension_entities)
            dim -= 1

        return all_processed_entities, max_dim

    def _process_dimension_group_fragments(
        self,
        entity_group: List[LabeledEntities],
        progress_bars: bool,
        higher_dim_entities: List[LabeledEntities],
    ) -> List[LabeledEntities]:
        """Fragment processing for entities against higher dimensional entities."""
        processed_entities = []

        for current_entity in entity_group:
            if progress_bars and current_entity.physical_name:
                print(
                    f"Processing {current_entity.physical_name} - fragment integration"
                )

            # Fragment against higher-dimensional entities if they exist
            if higher_dim_entities and current_entity.dimtags:
                for higher_dim_entity in higher_dim_entities:
                    if not higher_dim_entity.dimtags or not current_entity.dimtags:
                        continue

                    fragment_result = self.occ.fragment(
                        current_entity.dimtags,
                        higher_dim_entity.dimtags,
                        removeObject=True,
                        removeTool=True,
                    )
                    self.occ.synchronize()

                    # Update current_entity dimtags based on fragment result
                    if (
                        fragment_result
                        and len(fragment_result) > 1
                        and fragment_result[1]
                    ):
                        num_object_dimtags = len(current_entity.dimtags)
                        new_object_dimtags = []

                        for idx in range(num_object_dimtags):
                            if idx < len(fragment_result[1]):
                                mapping = fragment_result[1][idx]
                                if mapping:
                                    if isinstance(mapping, list):
                                        new_object_dimtags.extend(mapping)
                                    else:
                                        new_object_dimtags.append(mapping)

                        current_entity.dimtags = new_object_dimtags

            if current_entity.dimtags:
                processed_entities.append(current_entity)

        return processed_entities

    def _process_dimension_group_cuts(
        self, entity_group: List, progress_bars: bool
    ) -> List[LabeledEntities]:
        """Process entities of same dimension using cuts."""
        processed_entities = []

        for i, (index, entity_obj) in enumerate(entity_group):
            # Instantiate entity using helper method
            current_entity = self._instantiate_entity(index, entity_obj, progress_bars)

            if i == 0:
                processed_entities.append(current_entity)
            else:
                # Cut against previously processed entities
                tool_dimtags = [
                    dimtag
                    for prev_entity in processed_entities
                    for dimtag in prev_entity.dimtags
                    if dimtag
                ]

                if tool_dimtags and current_entity.dimtags:
                    try:
                        cut = self.occ.cut(
                            current_entity.dimtags,
                            tool_dimtags,
                            removeObject=True,
                            removeTool=False,
                        )
                        if cut and cut[0]:
                            current_entity.dimtags = list(set(cut[0]))
                    except Exception as e:
                        if progress_bars:
                            print(
                                f"Cut operation failed for {current_entity.physical_name}: {e}"
                            )

                if current_entity.dimtags:
                    processed_entities.append(current_entity)

                try:
                    self.occ.removeAllDuplicates()
                    self.sync_model()
                except Exception as e:
                    if progress_bars:
                        print(
                            f"Warning: cleanup failed after {current_entity.physical_name}: {e}"
                        )

        return processed_entities

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
            additive_dimtags = unpack_dimtags(additive_entity.instanciate(self))

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
                        )
                    )

            # Update entity list for next iteration
            structural_entity_list = updated_entities
            self.occ.removeAllDuplicates()
            self.sync_model()

        return structural_entity_list

    def _tag_mesh_components(
        self,
        final_entity_list: List,
        max_dim: int,
        interface_delimiter: str,
        boundary_delimiter: str,
    ) -> None:
        """Tag entities, interfaces, and boundaries."""
        # Update entity boundaries
        for entity in final_entity_list:
            entity.update_boundaries()

        # Filter out entities that became invalid after boundary update
        valid_final_entities = [
            entity for entity in final_entity_list if entity.dimtags and entity.dim >= 0
        ]

        # Tag entities
        if valid_final_entities:
            tag_entities(valid_final_entities, self.model)
            # Tag highest-dimension model interfaces, model boundaries
            valid_final_entities = tag_interfaces(
                valid_final_entities, max_dim, interface_delimiter, self.model
            )
            # Tag model boundaries
            tag_boundaries(
                valid_final_entities,
                max_dim,
                interface_delimiter,
                boundary_delimiter,
                self.model,
            )

    def generate(
        self,
        entities_list: List,
        output_file: Path,
        addition_delimiter: str = "+",
        addition_intersection_physicals: bool = True,
        addition_addition_physicals: bool = True,
        addition_structural_physicals: bool = True,
        interface_delimiter: str = "___",
        boundary_delimiter: str = "None",
        progress_bars: bool = False,
    ) -> None:
        """Generate CAD geometry and save to .xao format.

        Args:
            entities_list: List of entities to process
            output_file: Output file path
            ... [other args from original cad() method]
        """
        self._initialize_model()
        output_file = Path(output_file)

        # Process entities and get max dimension
        final_entity_list, max_dim = self._process_entities(
            entities_list=entities_list,
            progress_bars=progress_bars,
            addition_delimiter=addition_delimiter,
            addition_intersection_physicals=addition_intersection_physicals,
            addition_addition_physicals=addition_addition_physicals,
            addition_structural_physicals=addition_structural_physicals,
        )

        # Tag entities and boundaries (filtering happens inside _tag_mesh_components)
        self._tag_mesh_components(
            final_entity_list=final_entity_list,
            max_dim=max_dim,
            interface_delimiter=interface_delimiter,
            boundary_delimiter=boundary_delimiter,
        )

        # Save CAD to .xao format
        gmsh.write(str(output_file.with_suffix(".xao")))
        gmsh.finalize()


def cad(
    entities_list: List,
    output_file: Path,
    point_tolerance: float = 1e-3,
    n_threads: int = cpu_count(),
    filename: str = "temp",
    addition_delimiter: str = "+",
    addition_intersection_physicals: bool = True,
    addition_addition_physicals: bool = True,
    addition_structural_physicals: bool = True,
    interface_delimiter: str = "___",
    boundary_delimiter: str = "None",
    progress_bars: bool = False,
) -> None:
    """Utility function that wraps the CAD class for easier usage.

    Args:
        entities_list: List of entities to process
        output_file: Output file path
        point_tolerance: Tolerance for point merging
        n_threads: Number of threads to use for processing
        filename: Temporary filename for GMSH model
        addition_delimiter: Delimiter for additive entity names
        addition_intersection_physicals: Include intersection physical names
        addition_addition_physicals: Include addition physical names
        addition_structural_physicals: Include structural physical names
        interface_delimiter: Delimiter for interface names
        boundary_delimiter: Delimiter for boundary names
        progress_bars: Show progress bars during processing
    """
    cad_processor = CAD(
        point_tolerance=point_tolerance,
        n_threads=n_threads,
        filename=filename,
    )

    cad_processor.generate(
        entities_list=entities_list,
        output_file=output_file,
        addition_delimiter=addition_delimiter,
        addition_intersection_physicals=addition_intersection_physicals,
        addition_addition_physicals=addition_addition_physicals,
        addition_structural_physicals=addition_structural_physicals,
        interface_delimiter=interface_delimiter,
        boundary_delimiter=boundary_delimiter,
        progress_bars=progress_bars,
    )
