"""meshwell - OpenCASCADE-based CAD and GMSH-based meshing for integrated photonics and beyond."""

from meshwell.distributed import (
    Executor,
    InProcessExecutor,
    SubprocessExecutor,
    generate_mesh_distributed,
    subdomains_from_grid,
)
from meshwell.orchestrator import generate_mesh
from meshwell.utils import deserialize

__version__ = "0.0.1"
__author__ = "Simon Bilodeau <sb30@princeton.edu>"

__all__ = [
    "Executor",
    "InProcessExecutor",
    "SubprocessExecutor",
    "deserialize",
    "generate_mesh",
    "generate_mesh_distributed",
    "subdomains_from_grid",
]
