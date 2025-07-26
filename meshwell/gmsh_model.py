from os import cpu_count
from pathlib import Path
import gmsh
import tempfile


class GmshModel:
    """Methods to create generic GMSH models."""

    def __init__(
        self,
        n_threads: int = cpu_count(),
        default_filestem: Path | str | None = None,
    ):
        """Initialize mesh generator."""
        self.n_threads = n_threads

        if default_filestem is None:
            self.default_directory = Path(str(tempfile.TemporaryDirectory()))
        elif isinstance(default_filestem, str):  # Ensure filename is a Path object
            self.default_directory = Path(default_filestem)
        else:
            self.default_directory = default_filestem

        self.model_name = self.default_directory.name

        self.default_directory.mkdir(parents=True, exist_ok=True)
        self.default_xao = self.default_directory / "cad.xao"
        self.default_msh = self.default_directory / "mesh.msh"

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

        self.model.add(str(self.model_name))
        self.model.setFileName(str(self.model_name))
        gmsh.option.setNumber("General.NumThreads", self.n_threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads1D", self.n_threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", self.n_threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", self.n_threads)
        gmsh.option.setNumber("Geometry.OCCParallel", 1)
