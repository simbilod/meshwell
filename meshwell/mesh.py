from __future__ import annotations

from os import cpu_count
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import gmsh
import meshio
import numpy as np

from meshwell.labeledentity import LabeledEntities
from meshwell.gmsh_model import GmshModel


class Mesh(GmshModel):
    """Mesh class for generating meshes from .xao files."""

    def __init__(
        self,
        input_cad_file: Path | str | None = None,
        output_mesh_file: Path | str | None = None,
        *args,
        **kwargs,
    ):
        """Initialize mesh generator."""
        super().__init__(*args, **kwargs)

        if input_cad_file is None:
            self.input_cad_file = self.default_xao
        elif isinstance(input_cad_file, str):  # Ensure filename is a Path object
            self.input_cad_file = Path(input_cad_file)
        else:
            self.input_cad_file = input_cad_file

        # Validate if custom
        assert self.input_cad_file.suffix == ".xao", (
            "Input file must have .xao extension, got: " f"{self.input_cad_file.suffix}"
        )

        if output_mesh_file is None:
            self.output_mesh_file = self.default_msh
        elif isinstance(output_mesh_file, str):  # Ensure filename is a Path object
            self.output_mesh_file = Path(output_mesh_file)
        else:
            self.output_mesh_file = output_mesh_file

        # Validate if custom
        assert self.output_mesh_file.suffix == ".msh", (
            "Input file must have .msh extension, got: "
            f"{self.output_mesh_file.suffix}"
        )

    def _initialize_cad(self) -> None:
        # Load CAD model
        gmsh.merge(str(self.input_cad_file.with_suffix(".xao")))

    def _initialize_mesh_settings(
        self,
        verbosity: int,
        default_characteristic_length: float,
        global_2D_algorithm: int,
        global_3D_algorithm: int,
        gmsh_version: Optional[float],
        mesh_element_order: int = 1,
    ) -> None:
        """Initialize basic mesh settings."""
        gmsh.option.setNumber("General.Terminal", verbosity)
        gmsh.option.setNumber(
            "Mesh.CharacteristicLengthMax", default_characteristic_length
        )
        gmsh.option.setNumber("Mesh.Algorithm", global_2D_algorithm)
        gmsh.option.setNumber("Mesh.Algorithm3D", global_3D_algorithm)
        gmsh.option.setNumber("Mesh.ElementOrder", mesh_element_order)
        if gmsh_version is not None:
            gmsh.option.setNumber("Mesh.MshFileVersion", gmsh_version)
        self.occ.synchronize()

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
        boundary_delimiter: str,
        resolution_specs: Dict,
    ) -> None:
        """Apply mesh refinement settings.

        TODO: enable simultaneous background mesh and entity-based refinement
        """
        self._apply_entity_refinement(boundary_delimiter, resolution_specs)

    def get_top_physical_names(self) -> list[str]:
        """Get all physical names of dimension dim from the GMSH model.

        Returns:
            List of physical names as strings
        """
        # Get all physical groups
        physical_groups = self.model.getPhysicalGroups()

        # Get top dimension in physical groups dimtags
        max_dim = max(dim for dim, _ in physical_groups)

        # Filter physical groups by dimension
        physical_groups = [(dim, tag) for dim, tag in physical_groups if dim == max_dim]

        # Extract names from physical groups
        physical_names = [
            self.model.getPhysicalName(dim, tag) for dim, tag in physical_groups
        ]

        return physical_names

    def get_physical_dimtags(self, physical_name: str) -> list[tuple[int, int]]:
        """Get the dimtags associated with a physical group name.

        Args:
            physical_name: Name of the physical group

        Returns:
            List of (dim, tag) tuples for entities in the physical group
        """
        # Get all physical groups
        physical_groups = self.model.getPhysicalGroups()

        dimtags = []
        for dim, tag in physical_groups:
            # Get name of current physical group
            current_name = self.model.getPhysicalName(dim, tag)

            if current_name == physical_name:
                # Get entities in this physical group
                entity_tags = self.model.getEntitiesForPhysicalGroup(dim, tag)
                # Add dimtags for all entities
                dimtags.extend([(dim, int(t)) for t in set(entity_tags)])

        return dimtags

    def _apply_entity_refinement(
        self,
        boundary_delimiter: str,
        resolution_specs: Dict,
    ) -> None:
        """Apply mesh refinement based on entity information.

        Args:
            final_entity_list: List of LabeledEntities to process
            boundary_delimiter: String used to identify boundary entities
        """
        # Recreate LabeledEntities for applying resolutions
        final_entity_list = []
        final_entity_dict = {}
        # We address entities by top dimensional physicals:
        for index, physical_name in enumerate(self.get_top_physical_names()):
            if physical_name in resolution_specs:
                resolutions = resolution_specs[physical_name]
            else:
                resolutions = []
            entities = LabeledEntities(
                index=index,
                physical_name=physical_name,
                model=self.model,
                dimtags=self.get_physical_dimtags(physical_name=physical_name),
                resolutions=resolutions,
            )
            entities.update_boundaries()
            final_entity_list.append(entities)
            final_entity_dict[physical_name] = entities

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

    def _generate_final_mesh(
        self,
        dim: int,
        global_3D_algorithm: int,
        global_scaling: float,
        verbosity: int,
        optimization_flags: tuple[tuple[str, int]] | None,
        finalize: bool,
    ) -> meshio.Mesh:
        """Generate the final mesh and return meshio object."""
        gmsh.option.setNumber("Mesh.ScalingFactor", global_scaling)

        if global_3D_algorithm == 1 and verbosity:
            gmsh.logger.start()

        self.model.mesh.generate(dim)

        if optimization_flags:
            for optimization_flag, niter in optimization_flags:
                self.model.mesh.optimize(optimization_flag, niter=niter)

        gmsh.write(str(self.output_mesh_file))
        if finalize:
            gmsh.finalize()
        return meshio.read(str(self.output_mesh_file))

    def generate(
        self,
        dim: int,
        default_characteristic_length: float,
        global_scaling: float = 1.0,
        global_2D_algorithm: int = 6,
        global_3D_algorithm: int = 1,
        mesh_element_order: int = 1,
        verbosity: Optional[int] = 0,
        periodic_entities: Optional[List[Tuple[str, str]]] = None,
        optimization_flags: Optional[tuple[tuple[str, int]]] = None,
        boundary_delimiter: str = "None",  # noqa: B006
        resolution_specs: Dict = (),
    ) -> Optional[meshio.Mesh]:
        """Generate mesh from .xao file.

        Args:
            input_file: Path to input .xao file
            output_file: Path for output mesh file
            entities_list: Optional list of entities with mesh parameters
            ... [other args from original mesh() method]

        Returns:
            Optional[meshio.Mesh]: Generated mesh object
        """
        self._initialize_model()
        self._initialize_cad()

        # Initialize mesh settings
        self._initialize_mesh_settings(
            verbosity=verbosity,
            default_characteristic_length=default_characteristic_length,
            global_2D_algorithm=global_2D_algorithm,
            global_3D_algorithm=global_3D_algorithm,
            gmsh_version=None,
            mesh_element_order=mesh_element_order,
        )

        # # Handle periodic boundaries if specified
        # if periodic_entities and entities_list:
        #     self._apply_periodic_boundaries(entities_list, periodic_entities)

        # Apply mesh refinement
        self._apply_mesh_refinement(
            boundary_delimiter=boundary_delimiter,
            resolution_specs=resolution_specs,
        )

        # Generate and return mesh
        return self._generate_final_mesh(
            dim=dim,
            global_3D_algorithm=global_3D_algorithm,
            global_scaling=global_scaling,
            verbosity=verbosity,
            optimization_flags=optimization_flags,
            finalize=True,
        )


def mesh(
    dim: int,
    input_cad_file: Path,
    output_mesh_file: Path,
    default_characteristic_length: float,
    resolution_specs: Dict | None = None,
    global_scaling: float = 1.0,
    global_2D_algorithm: int = 6,
    global_3D_algorithm: int = 1,
    mesh_element_order: int = 1,
    verbosity: Optional[int] = 0,
    periodic_entities: Optional[List[Tuple[str, str]]] = None,
    optimization_flags: Optional[tuple[tuple[str, int]]] = None,
    boundary_delimiter: str = "None",
    n_threads: int = cpu_count(),
) -> Optional[meshio.Mesh]:
    """Utility function that wraps the Mesh class for easier usage.

    Args:
        input_cad_file: Path to input .xao file
        output_mesh_file: Path for output mesh file
        entities_list: Optional list of entities with mesh parameters
        default_characteristic_length: Default mesh size
        global_scaling: Global scaling factor
        global_2D_algorithm: GMSH 2D meshing algorithm
        global_3D_algorithm: GMSH 3D meshing algorithm
        verbosity: GMSH verbosity level
        periodic_entities: List of periodic boundary pairs
        optimization_flags: Mesh optimization flags
        boundary_delimiter: Delimiter for boundary names
        resolution_specs: Mesh resolution specifications
        n_threads: Number of threads to use
        filename: Temporary filename for GMSH model

    Returns:
        Optional[meshio.Mesh]: Generated mesh object
    """
    mesh_generator = Mesh(
        input_cad_file=input_cad_file,
        output_mesh_file=output_mesh_file,
        n_threads=n_threads,
    )

    if resolution_specs is None:
        resolution_specs = {}

    return mesh_generator.generate(
        dim=dim,
        default_characteristic_length=default_characteristic_length,
        global_scaling=global_scaling,
        global_2D_algorithm=global_2D_algorithm,
        global_3D_algorithm=global_3D_algorithm,
        mesh_element_order=mesh_element_order,
        verbosity=verbosity,
        periodic_entities=periodic_entities,
        optimization_flags=optimization_flags,
        boundary_delimiter=boundary_delimiter,
        resolution_specs=resolution_specs,
    )
