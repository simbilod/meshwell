from __future__ import annotations

from os import cpu_count
from pathlib import Path
from typing import Optional
import gmsh


class Model:
    """Base model class that handles common GMSH model functionality.

    This class centralizes GMSH model initialization, configuration, and cleanup
    to be shared between CAD and Mesh classes.

    Can also host CAD and Mesh generators as properties for convenience.
    """

    def __init__(
        self,
        n_threads: int = cpu_count(),
        filename: str = "temp",
        point_tolerance: Optional[float] = None,
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

    def _initialize(self, model_name: Optional[str] = None) -> None:
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
        self.occ.synchronize()

    def clear_and_reinitialize(self, model_name: Optional[str] = None) -> None:
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

    def is_initialized(self) -> bool:
        """Check if model is initialized."""
        return self._is_initialized

    def ensure_initialized(self, model_name: Optional[str] = None) -> None:
        """Ensure model is initialized, initialize if not.

        Args:
            model_name: Optional model name override
        """
        if not self._is_initialized:
            self._initialize(model_name)

    @property
    def cad(self):
        """Get the CAD instance for this model (created lazily).

        Returns:
            CAD instance configured to use this model

        Example:
            model = Model(filename="my_project")
            model.cad.generate(entities_list, "output.xao")
        """
        if self._cad is None:
            # Import here to avoid circular imports
            from meshwell.cad import CAD

            self._cad = CAD(
                model=self,
                n_threads=self.n_threads,
                filename=str(self.filename),
                point_tolerance=self.point_tolerance or 1e-3,
            )
        return self._cad

    @property
    def mesh(self):
        """Get the Mesh instance for this model (created lazily).

        Returns:
            Mesh instance configured to use this model

        Example:
            model = Model(filename="my_project")
            mesh_obj = model.mesh.generate(2, "input.xao", "output.msh", 0.1)
        """
        if self._mesh is None:
            # Import here to avoid circular imports
            from meshwell.mesh import Mesh

            self._mesh = Mesh(
                model=self,
                n_threads=self.n_threads,
                filename=str(self.filename),
            )
        return self._mesh
