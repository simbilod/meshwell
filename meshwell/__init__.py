"""meshwell - GMSH wrapper, with integrated photonics focus."""

from meshwell.orchestrator import generate_mesh
from meshwell.utils import deserialize

__version__ = "0.0.1"
__author__ = "Simon Bilodeau <sb30@princeton.edu>"

__all__ = ["deserialize", "generate_mesh"]
