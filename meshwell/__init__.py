"""meshwell - OpenCASCADE-based CAD and GMSH-based meshing for integrated photonics and beyond."""

import logging

from meshwell.orchestrator import generate_mesh
from meshwell.utils import deserialize

# Library convention: silent by default; host applications opt in by
# configuring a handler on the "meshwell" logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "0.0.1"
__author__ = "Simon Bilodeau <sb30@princeton.edu>"

__all__ = ["deserialize", "generate_mesh"]
