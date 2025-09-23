"""Mesh class definition."""
from __future__ import annotations

import contextlib
import tempfile
from os import cpu_count
from pathlib import Path

import gmsh
import meshio
import numpy as np

from meshwell.labeledentity import LabeledEntities
from meshwell.model import ModelManager


class Mesh:
    """Mesh class for generating meshes from cad models."""

    def __init__(
        self,
        n_threads: int = cpu_count(),
        filename: str = "temp",
        model: ModelManager | None = None,
    ):
        """Initialize mesh generator.

        Args:
            n_threads: Number of threads for processing
            filename: Base filename for the model
            model: Optional Model instance to use (creates new if None)

        """
        # Use provided model or create new one
        if model is None:
            self.model_manager = ModelManager(
                n_threads=n_threads,
                filename=filename,
            )
            self._owns_model = True
        else:
            self.model_manager = model
            self._owns_model = False

    def _initialize_model(self, input_file: Path | None = None) -> None:
        """Initialize GMSH model and optionally load .xao file."""
        # Initialize the model instance
        self.model_manager.ensure_initialized("temp")

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
        gmsh_version: float | None,
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
        self.model_manager.sync_model()

    def _apply_periodic_boundaries(
        self, periodic_entities: list[tuple[str, str]]
    ) -> None:
        """Apply periodic boundary conditions."""
        mapping = {
            self.model_manager.model.getPhysicalName(dimtag[0], dimtag[1]): dimtag
            for dimtag in self.model_manager.model.getPhysicalGroups()
        }

        for label1, label2 in periodic_entities:
            if label1 not in mapping or label2 not in mapping:
                continue

            self._set_periodic_pair(mapping, label1, label2)

    def _set_periodic_pair(self, mapping: dict, label1: str, label2: str) -> None:
        """Set up periodic boundary pair."""
        tags1 = self.model_manager.model.getEntitiesForPhysicalGroup(*mapping[label1])
        tags2 = self.model_manager.model.getEntitiesForPhysicalGroup(*mapping[label2])

        vector1 = self.model_manager.model.occ.getCenterOfMass(
            mapping[label1][0], tags1[0]
        )
        vector2 = self.model_manager.model.occ.getCenterOfMass(
            mapping[label1][0], tags2[0]
        )
        vector = np.subtract(vector1, vector2)

        self.model_manager.model.mesh.setPeriodic(
            mapping[label1][0],
            tags1,
            tags2,
            [1, 0, 0, vector[0], 0, 1, 0, vector[1], 0, 0, 1, vector[2], 0, 0, 0, 1],
        )

    def _apply_mesh_refinement(
        self,
        background_remeshing_file: Path | None | None,
        boundary_delimiter: str,
        resolution_specs: dict,
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
        return self.model_manager.get_top_physical_names()

    def get_all_physical_names(self) -> list[str]:
        """Get all physical names from the GMSH model.

        Returns:
            List of physical names as strings

        """
        return self.model_manager.get_physical_names()

    def get_physical_dimtags(self, physical_name: str) -> list[tuple[int, int]]:
        """Get the dimtags associated with a physical group name.

        Args:
            physical_name: Name of the physical group

        Returns:
            List of (dim, tag) tuples for entities in the physical group

        """
        return self.model_manager.get_physical_dimtags(physical_name)

    def _recover_labels_from_cad(self, resolution_specs: dict) -> tuple[list, dict]:
        """Recover labeled entities from loaded CAD model.

        Args:
            resolution_specs: Dictionary mapping physical names to resolution specifications

        Returns:
            Tuple of (final_entity_list, final_entity_dict)
        """
        final_entity_list = []
        final_entity_dict = {}

        # We address entities by "named" physicals (not default):
        top_physical_names = self.get_top_physical_names()
        all_physical_names = self.get_all_physical_names()

        for index, physical_name in enumerate(all_physical_names):
            resolutions = resolution_specs.get(physical_name, [])
            if not resolutions and physical_name not in top_physical_names:
                continue

            entities = LabeledEntities(
                index=index,
                physical_name=physical_name,
                model=self.model_manager.model,
                dimtags=self.get_physical_dimtags(physical_name=physical_name),
                resolutions=resolutions,
            )
            entities.update_boundaries()
            final_entity_list.append(entities)
            final_entity_dict[physical_name] = entities

        return final_entity_list, final_entity_dict

    def _apply_entity_refinement(
        self,
        boundary_delimiter: str,
        resolution_specs: dict,
    ) -> None:
        """Apply mesh refinement based on entity information.

        Args:
            boundary_delimiter: String used to identify boundary entities
            resolution_specs: Resolution specifications

        """
        # Recover labeled entities from loaded CAD model
        final_entity_list, final_entity_dict = self._recover_labels_from_cad(
            resolution_specs
        )

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
            min_field_index = self.model_manager.model.mesh.field.add("Min")
            self.model_manager.model.mesh.field.setNumbers(
                min_field_index, "FieldsList", refinement_field_indices
            )
            self.model_manager.model.mesh.field.setAsBackgroundMesh(min_field_index)

        # Turn off default meshing options
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    def _apply_background_refinement(self) -> None:
        """Apply mesh refinement based on background mesh."""
        # Create background field from post-processing view
        bg_field = self.model_manager.model.mesh.field.add("PostView")
        self.model_manager.model.mesh.field.setNumber(bg_field, "ViewIndex", 0)
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

        self.model_manager.model.mesh.generate(dim)

        if optimization_flags:
            for optimization_flag, niter in optimization_flags:
                self.model_manager.model.mesh.optimize(optimization_flag, niter=niter)

        # Return mesh object without writing to file
        with contextlib.redirect_stdout(
            None
        ), tempfile.TemporaryDirectory() as tmpdirname:
            temp_mesh_path = f"{tmpdirname}/mesh.msh"
            gmsh.write(temp_mesh_path)
            return meshio.read(temp_mesh_path)

    def save_to_file(self, output_file: Path) -> None:
        """Save current mesh to file.

        Args:
            output_file: Output mesh file path

        """
        self.model_manager.save_to_mesh(output_file)

    def to_msh(self, output_file: Path, format: str = "msh") -> None:
        """Save current mesh to .msh file.

        Args:
            output_file: Output file path (will be suffixed with .format)
            format: File format to use in gmsh

        """
        self.model_manager.save_to_mesh(output_file, format)

    def to_meshio(self) -> meshio.Mesh:
        """Convert current mesh to meshio.Mesh object.

        Returns:
            meshio.Mesh: Current mesh as meshio object

        """
        with contextlib.redirect_stdout(
            None
        ), tempfile.TemporaryDirectory() as tmpdirname:
            temp_mesh_path = f"{tmpdirname}/mesh.msh"
            gmsh.write(temp_mesh_path)
            return meshio.read(temp_mesh_path)

    def load_xao_file(self, input_file: Path) -> None:
        """Load CAD geometry from .xao file.

        Args:
            input_file: Input .xao file path

        """
        self.model_manager.load_from_xao(input_file)

    def process_geometry(
        self,
        dim: int,
        default_characteristic_length: float,
        background_remeshing_file: Path | None = None,
        global_scaling: float = 1.0,
        global_2D_algorithm: int = 6,
        global_3D_algorithm: int = 1,
        mesh_element_order: int = 1,
        verbosity: int | None = 0,
        periodic_entities: list[tuple[str, str]] | None = None,  # noqa: ARG002
        optimization_flags: tuple[tuple[str, int]] | None = None,
        boundary_delimiter: str = "None",
        resolution_specs: dict = (),
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
    resolution_specs: dict | None = None,
    background_remeshing_file: Path | None = None,
    global_scaling: float = 1.0,
    global_2D_algorithm: int = 6,
    global_3D_algorithm: int = 1,
    mesh_element_order: int = 1,
    verbosity: int | None = 0,
    periodic_entities: list[tuple[str, str]] | None = None,
    optimization_flags: tuple[tuple[str, int]] | None = None,
    boundary_delimiter: str = "None",
    n_threads: int = cpu_count(),
    filename: str = "temp",
    model: ModelManager | None = None,
) -> meshio.Mesh | None:
    """Utility function that wraps the Mesh class for easier usage.

    Args:
        dim: Dimension of mesh to generate
        input_file: Path to input .xao file
        output_file: Path for output mesh file
        entities_list: Optional list of entities with mesh parameters
        background_remeshing_file: Optional background mesh file for refinement
        default_characteristic_length: Default mesh size
        global_scaling: Global scaling factor
        global_2D_algorithm: GMSH 2D meshing algorithm
        global_3D_algorithm: GMSH 3D meshing algorithm
        mesh_element_order: Element order
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
    mesh_generator.load_xao_file(input_file)

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
        mesh_generator.model_manager.finalize()

    return mesh_obj
