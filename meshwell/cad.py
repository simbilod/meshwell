from __future__ import annotations

from os import cpu_count
from typing import Dict, List, Tuple
from pathlib import Path
import gmsh

from meshwell.validation import (
    validate_dimtags,
    unpack_dimtags,
    order_entities,
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
            if progress_bars:
                if physical_name:
                    enumerator.set_description(
                        f"{str(physical_name):<30} - {'instanciate':<15}"
                    )
            # First create the shape
            dimtags_out = entity_obj.instanciate(self)

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
            )
            if progress_bars:
                if physical_name:
                    enumerator.set_description(
                        f"{str(physical_name):<30} - {'boolean':<15}"
                    )
            if index == 0:
                structural_entity_list.append(current_entities)
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
        # Tag entities
        tag_entities(final_entity_list, self.model)
        # Tag model interfaces, logging model boundaries
        final_entity_list = tag_interfaces(
            final_entity_list, max_dim, interface_delimiter, self.model
        )
        # Tag model boundaries
        tag_boundaries(
            final_entity_list,
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

        # Tag entities and boundaries
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
