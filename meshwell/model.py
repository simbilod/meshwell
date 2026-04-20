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
        point_tolerance: float | None = 1e-3,
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

        # Configure OCC tolerance if provided
        if self.point_tolerance is not None:
            gmsh.option.setNumber("Geometry.Tolerance", self.point_tolerance)
            gmsh.option.setNumber("Geometry.ToleranceBoolean", self.point_tolerance)

        self._is_initialized = True

    def to_file(self, output_file: Path) -> None:
        """Save current model state.

        Args:
            output_file: Output file path.

        """
        output_file = Path(output_file)
        gmsh.write(str(output_file))

    def load_from_xao(
        self,
        input_file: Path,
        remove_all_duplicates: bool = False,
    ) -> None:
        """Load CAD geometry from .xao file.

        Args:
            input_file: Input .xao file path
            remove_all_duplicates: opt-in gmsh-level ``occ.fragment`` pass
                over every loaded dimtag. Only catches OCC-identical
                coincident TShapes -- it does not fix geometric-but-not-
                topological duplicates, slivers, or sub-fuzzy features,
                which are the usual root cause of ``dihedral 0`` and PLC
                errors. For those, prefer ``canonicalize_topology=True``
                and/or raising ``fuzzy_value`` at the ``cad_occ`` stage.

        """
        self.ensure_initialized("temp")
        input_file = Path(input_file)
        gmsh.open(str(input_file.with_suffix(".xao")))
        self.model.occ.synchronize()
        if remove_all_duplicates:
            self._fragment_all_loaded_dimtags()

    def load_occ_entities(
        self,
        entities: list,
        remove_all_duplicates: bool = False,
        **write_xao_kwargs,
    ) -> None:
        """Serialize ``entities`` to a transient XAO and load it into gmsh.

        Convenience wrapper around :func:`meshwell.occ_xao_writer.write_xao`
        + :meth:`load_from_xao`. Every physical group the OCP tagging pass
        produced is applied to the current gmsh model.

        Args:
            entities: list of ``OCCLabeledEntity`` from ``cad_occ``.
            remove_all_duplicates: forwarded to :meth:`load_from_xao`.
                See its docstring for the tradeoff.
            **write_xao_kwargs: forwarded to
                :func:`meshwell.occ_xao_writer.write_xao`
                (``interface_delimiter``, ``boundary_delimiter``,
                ``model_name``).
        """
        import tempfile

        from meshwell.occ_xao_writer import write_xao

        self.ensure_initialized("temp")
        with tempfile.TemporaryDirectory() as tmpdir:
            xao_path = Path(tmpdir) / "cad.xao"
            write_xao(entities, xao_path, **write_xao_kwargs)
            self.load_from_xao(xao_path, remove_all_duplicates=remove_all_duplicates)

    def _fragment_all_loaded_dimtags(self) -> None:
        """Run ``occ.fragment`` over every loaded dimtag.

        Equivalent to gmsh's own ``removeAllDuplicates`` but survives
        without a per-entity dimtag map: the refinement engine
        (``mesh.py``) rebuilds its state from physical groups
        post-fragment, so dimtag identity need not be preserved.
        """
        all_dimtags = [
            (d, t) for d in (0, 1, 2, 3) for _, t in self.model.getEntities(d)
        ]
        if not all_dimtags:
            return
        self.model.occ.fragment(all_dimtags, [], removeObject=True, removeTool=True)
        self.model.occ.synchronize()

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
        gmsh.option.setNumber("Geometry.OCCParallel", 1 if self.n_threads > 1 else 0)

    def sync_model(self) -> None:
        """Synchronize the OCC model."""
        if not self._is_initialized:
            return
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
