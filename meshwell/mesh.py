from __future__ import annotations

import contextlib
import tempfile
from os import cpu_count
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import gmsh
import meshio
import numpy as np

from meshwell.labeledentity import LabeledEntities
from meshwell.model import Model


class Mesh:
    """Mesh class for generating meshes from .xao files."""

    def __init__(
        self,
        n_threads: int = cpu_count(),
        filename: str = "temp",
        model: Optional[Model] = None,
    ):
        """Initialize mesh generator.

        Args:
            n_threads: Number of threads for processing
            filename: Base filename for the model
            model: Optional Model instance to use (creates new if None)
        """
        # Use provided model or create new one
        if model is None:
            self.model_instance = Model(
                n_threads=n_threads,
                filename=filename,
            )
            self._owns_model = True
        else:
            self.model_instance = model
            self._owns_model = False

        # Store parameters for backward compatibility
        self.filename = Path(filename)

    def _initialize_model(self, input_file: Optional[Path] = None) -> None:
        """Initialize GMSH model and optionally load .xao file."""
        # Initialize the model instance
        self.model_instance.ensure_initialized("temp")

        # Set references for backward compatibility
        self.model = self.model_instance.model
        self.occ = self.model_instance.occ

        # Load CAD model if input file provided
        if input_file is not None:
            input_file = Path(input_file)
            gmsh.merge(str(input_file.with_suffix(".xao")))

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
        self.model_instance.sync_model()

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

        vector1 = self.model.occ.getCenterOfMass(mapping[label1][0], tags1[0])
        vector2 = self.model.occ.getCenterOfMass(mapping[label1][0], tags2[0])
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
        boundary_delimiter: str,
        resolution_specs: Dict,
    ) -> None:
        """Apply mesh refinement settings.

        TODO: enable simultaneous background mesh and entity-based refinement
        """
        if background_remeshing_file is None:
            self._apply_entity_refinement(boundary_delimiter, resolution_specs)
        else:
            self._apply_background_refinement()

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

    def get_all_physical_names(self) -> list[str]:
        """Get all physical names of dimension dim from the GMSH model.

        Returns:
            List of physical names as strings
        """
        # Get all physical groups
        physical_groups = self.model.getPhysicalGroups()

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
        # We address entities by "named" physicals (not default):
        top_physical_names = self.get_top_physical_names()
        all_physical_names = self.get_all_physical_names()
        for index, physical_name in enumerate(all_physical_names):
            if physical_name in resolution_specs:
                resolutions = resolution_specs[physical_name]
            else:
                resolutions = []
            if not resolutions and physical_name not in top_physical_names:
                continue
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

    def process_mesh(
        self,
        dim: int,
        global_3D_algorithm: int,
        global_scaling: float,
        verbosity: int,
        optimization_flags: tuple[tuple[str, int]] | None,
    ) -> meshio.Mesh:
        """Generate mesh and return meshio object (no file I/O)."""
        gmsh.option.setNumber("Mesh.ScalingFactor", global_scaling)

        if global_3D_algorithm == 1 and verbosity:
            gmsh.logger.start()

        self.model.mesh.generate(dim)

        if optimization_flags:
            for optimization_flag, niter in optimization_flags:
                self.model.mesh.optimize(optimization_flag, niter=niter)

        # Return mesh object without writing to file
        with contextlib.redirect_stdout(None):
            with tempfile.TemporaryDirectory() as tmpdirname:
                temp_mesh_path = f"{tmpdirname}/mesh.msh"
                gmsh.write(temp_mesh_path)
                return meshio.read(temp_mesh_path)

    def save_to_file(self, output_file: Path) -> None:
        """Save current mesh to file.

        Args:
            output_file: Output mesh file path
        """
        gmsh.write(str(output_file))

    def load_from_file(self, input_file: Path) -> None:
        """Load CAD geometry from .xao file.

        Args:
            input_file: Input .xao file path
        """
        self._initialize_model()
        input_file = Path(input_file)
        gmsh.merge(str(input_file.with_suffix(".xao")))

    def process_geometry(
        self,
        dim: int,
        default_characteristic_length: float,
        background_remeshing_file: Optional[Path] = None,
        global_scaling: float = 1.0,
        global_2D_algorithm: int = 6,
        global_3D_algorithm: int = 1,
        mesh_element_order: int = 1,
        verbosity: Optional[int] = 0,
        periodic_entities: Optional[List[Tuple[str, str]]] = None,
        optimization_flags: Optional[tuple[tuple[str, int]]] = None,
        boundary_delimiter: str = "None",  # noqa: B006
        resolution_specs: Dict = (),
    ) -> meshio.Mesh:
        """Process loaded geometry into mesh (no file I/O).

        Args:
            dim: Dimension of mesh to generate
            default_characteristic_length: Default mesh size
            background_remeshing_file: Optional background mesh file for refinement
            global_scaling: Global scaling factor
            global_2D_algorithm: GMSH 2D meshing algorithm
            global_3D_algorithm: GMSH 3D meshing algorithm
            mesh_element_order: Element order
            verbosity: GMSH verbosity level
            periodic_entities: List of periodic boundary pairs
            optimization_flags: Mesh optimization flags
            boundary_delimiter: Delimiter for boundary names
            resolution_specs: Mesh resolution specifications

        Returns:
            meshio.Mesh: Generated mesh object
        """
        self._initialize_model()

        # Initialize mesh settings
        self._initialize_mesh_settings(
            verbosity=verbosity,
            default_characteristic_length=default_characteristic_length,
            global_2D_algorithm=global_2D_algorithm,
            global_3D_algorithm=global_3D_algorithm,
            gmsh_version=None,
            mesh_element_order=mesh_element_order,
        )

        # Apply mesh refinement
        self._apply_mesh_refinement(
            background_remeshing_file=background_remeshing_file,
            boundary_delimiter=boundary_delimiter,
            resolution_specs=resolution_specs,
        )

        # Generate and return mesh
        return self.process_mesh(
            dim=dim,
            global_3D_algorithm=global_3D_algorithm,
            global_scaling=global_scaling,
            verbosity=verbosity,
            optimization_flags=optimization_flags,
        )


def mesh(
    dim: int,
    input_file: Path,
    output_file: Path,
    default_characteristic_length: float,
    resolution_specs: Dict | None = None,
    background_remeshing_file: Optional[Path] = None,
    global_scaling: float = 1.0,
    global_2D_algorithm: int = 6,
    global_3D_algorithm: int = 1,
    mesh_element_order: int = 1,
    verbosity: Optional[int] = 0,
    periodic_entities: Optional[List[Tuple[str, str]]] = None,
    optimization_flags: Optional[tuple[tuple[str, int]]] = None,
    boundary_delimiter: str = "None",
    n_threads: int = cpu_count(),
    filename: str = "temp",
    model: Optional[Model] = None,
) -> Optional[meshio.Mesh]:
    """Utility function that wraps the Mesh class for easier usage.

    Args:
        input_file: Path to input .xao file
        output_file: Path for output mesh file
        entities_list: Optional list of entities with mesh parameters
        background_remeshing_file: Optional background mesh file for refinement
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
        model: Optional Model instance to use (creates new if None)

    Returns:
        Optional[meshio.Mesh]: Generated mesh object
    """
    mesh_generator = Mesh(
        n_threads=n_threads,
        filename=filename,
        model=model,
    )

    if resolution_specs is None:
        resolution_specs = {}

    # Load geometry from file
    mesh_generator.load_from_file(input_file)

    # Process geometry into mesh
    mesh_obj = mesh_generator.process_geometry(
        dim=dim,
        background_remeshing_file=background_remeshing_file,
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

    # Save to file
    mesh_generator.save_to_file(output_file)

    # Finalize if we created the model
    if model is None:
        mesh_generator.model_instance.finalize()

    return mesh_obj
