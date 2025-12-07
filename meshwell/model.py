"""Main gmsh model definitions."""
from __future__ import annotations

from os import cpu_count
from pathlib import Path

import gmsh


class ModelManager:
    """Base model class that handles common GMSH model functionality.

    This class centralizes GMSH model initialization, configuration, and cleanup
    to be shared between CAD and Mesh classes.

    Can also host CAD and Mesh generators as properties for convenience.
    """

    def __init__(
        self,
        n_threads: int = cpu_count(),
        filename: str = "temp",
        point_tolerance: float | None = None,
    ):
        """Initialize Model with common settings.

        Args:
            n_threads: Number of threads for GMSH operations
            filename: Base filename for the model
            point_tolerance: Point tolerance for CAD operations (optional)

        """
        self.n_threads = n_threads
        self.filename = Path(filename)
        self.point_tolerance = point_tolerance

        # GMSH objects (initialized in _initialize)
        self.model = None
        self.occ = None

        # Initialization state
        self._is_initialized = False

        # CAD and Mesh instances (created lazily)
        self._cad = None
        self._mesh = None

    def _initialize(self, model_name: str | None = None) -> None:
        """Initialize GMSH model and set basic configuration.

        Args:
            model_name: Optional model name override

        """
        if self._is_initialized:
            return

        # Handle GMSH initialization
        if gmsh.is_initialized():
            gmsh.finalize()
            gmsh.initialize()
        else:
            gmsh.initialize()

        # Clear any existing model
        gmsh.clear()

        # Create model objects
        self.model = gmsh.model
        self.occ = self.model.occ

        # Set model name and filename
        model_name = model_name or str(self.filename)
        self.model.add(model_name)
        self.model.setFileName(str(self.filename))

        # Configure threading
        self._configure_threading()

        self._is_initialized = True

    def to_file(self, output_file: Path) -> None:
        """Save current model state.

        Args:
            output_file: Output file path.

        """
        output_file = Path(output_file)
        gmsh.write(str(output_file))

    def load_from_xao(self, input_file: Path) -> None:
        """Load CAD geometry from .xao file.

        Args:
            input_file: Input .xao file path

        """
        self.ensure_initialized("temp")
        input_file = Path(input_file)
        gmsh.merge(str(input_file.with_suffix(".xao")))

    def save_to_xao(self, output_file: Path) -> None:
        """Save current model to .xao file.

        Args:
            output_file: Output file path (will be suffixed with .xao)

        """
        output_file = Path(output_file).with_suffix(".xao")
        gmsh.write(str(output_file))

    def save_to_mesh(self, output_file: Path, format: str = "msh") -> None:
        """Save current mesh to file.

        Args:
            output_file: Output file path (will be suffixed with format)
            format: File format (msh, vtk, etc.)

        """
        output_file = Path(output_file).with_suffix(f".{format}")
        gmsh.write(str(output_file))

    def get_physical_names(self, dim: int | None = None) -> list[str]:
        """Get physical names, optionally filtered by dimension.

        Args:
            dim: Optional dimension filter (0=points, 1=curves, 2=surfaces, 3=volumes)
                If None, returns all physical names

        Returns:
            List of physical names as strings

        """
        if not self._is_initialized or self.model is None:
            return []

        physical_groups = self.model.getPhysicalGroups()

        if dim is not None:
            physical_groups = [(d, tag) for d, tag in physical_groups if d == dim]

        return [self.model.getPhysicalName(d, tag) for d, tag in physical_groups]

    def get_top_physical_names(self) -> list[str]:
        """Get physical names of highest dimension.

        Returns:
            List of physical names as strings

        """
        if not self._is_initialized or self.model is None:
            return []

        physical_groups = self.model.getPhysicalGroups()

        if not physical_groups:
            return []

        max_dim = max(dim for dim, _ in physical_groups)
        return self.get_physical_names(dim=max_dim)

    def get_physical_dimtags(self, physical_name: str) -> list[tuple[int, int]]:
        """Get dimtags for a physical group name.

        Args:
            physical_name: Name of the physical group

        Returns:
            List of (dim, tag) tuples for entities in the physical group

        """
        if not self._is_initialized or self.model is None:
            return []

        physical_groups = self.model.getPhysicalGroups()

        dimtags = []
        for dim, tag in physical_groups:
            current_name = self.model.getPhysicalName(dim, tag)
            if current_name == physical_name:
                entity_tags = self.model.getEntitiesForPhysicalGroup(dim, tag)
                dimtags.extend([(dim, int(t)) for t in set(entity_tags)])

        return dimtags

    def _configure_threading(self) -> None:
        """Configure GMSH threading options."""
        gmsh.option.setNumber("General.NumThreads", self.n_threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads1D", self.n_threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", self.n_threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", self.n_threads)
        gmsh.option.setNumber("Geometry.OCCParallel", 1)

    def sync_model(self) -> None:
        """Synchronize the OCC model."""
        if not self._is_initialized:
            return
        self.occ.removeAllDuplicates()
        self.occ.synchronize()

    def clear_and_reinitialize(self, model_name: str | None = None) -> None:
        """Clear current model and reinitialize.

        Args:
            model_name: Optional model name override

        """
        self._is_initialized = False
        self._initialize(model_name)

    def finalize(self) -> None:
        """Finalize GMSH and cleanup."""
        if gmsh.is_initialized():
            gmsh.finalize()
        self._is_initialized = False
        self.model = None
        self.occ = None
        # Clean up lazy instances
        self._cad = None
        self._mesh = None

    def is_initialized(self) -> bool:
        """Check if model is initialized."""
        return self._is_initialized

    def ensure_initialized(self, model_name: str | None = None) -> None:
        """Ensure model is initialized, initialize if not.

        Args:
            model_name: Optional model name override

        """
        if not self._is_initialized:
            self._initialize(model_name)

    @property
    def cad(self):
        """Get CAD instance for this model (created lazily).

        Returns:
            CAD instance configured to use this ModelManager

        Example:
            model = ModelManager(filename="my_project")
            model.cad.process_entities(entities_list)
            model.save_to_xao("output.xao")

        """
        if self._cad is None:
            # Import here to avoid circular imports
            from meshwell.cad import CAD

            self._cad = CAD(
                model=self,
                point_tolerance=self.point_tolerance or 1e-3,
            )
        return self._cad

    @property
    def mesh(self):
        """Get Mesh instance for this model (created lazily).

        Returns:
            Mesh instance configured to use this ModelManager

        Example:
            model = ModelManager(filename="my_project")
            model.mesh.load_xao_file("input.xao")
            mesh_obj = model.mesh.process_geometry(dim=3, default_characteristic_length=0.1)

        """
        if self._mesh is None:
            # Import here to avoid circular imports
            from meshwell.mesh import Mesh

            self._mesh = Mesh(model=self)
        return self._mesh
