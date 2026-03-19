"""CAD processor facade - re-exports both gmsh and OCC backends.

For backward compatibility, defaults to the gmsh backend.
"""
from meshwell.cad_gmsh import CAD as CAD_GMSH
from meshwell.cad_gmsh import cad as cad_gmsh

try:
    from meshwell.cad_occ import (
        CAD_OCC,
        cad_occ,
    )
except ImportError:
    CAD_OCC = None
    cad_occ = None

# Default to gmsh for backward compatibility
CAD = CAD_GMSH
cad = cad_gmsh
