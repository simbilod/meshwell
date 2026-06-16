"""meshwell - OpenCASCADE-based CAD and GMSH-based meshing for integrated photonics and beyond."""

import logging
from importlib.metadata import PackageNotFoundError, version

from meshwell.orchestrator import generate_mesh
from meshwell.utils import deserialize

# Library convention: silent by default; host applications opt in by
# configuring a handler on the "meshwell" logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    __version__ = version("meshwell")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0"
__author__ = "Simon Bilodeau <sb30@princeton.edu>"

__all__ = ["deserialize", "generate_mesh"]
