# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: meshwell
#     language: python
#     name: python3
# ---

# # Quickstart

# + tags=["hide-input"] vscode={"languageId": "python"}
import shapely
import gmsh
from meshwell.prism import Prism
from meshwell.model import Model
from collections import OrderedDict
from skfem.visuals.matplotlib import draw_mesh3d
from skfem.io.meshio import from_meshio
import meshio

# -

# # Step 0
#
# Initialize the CAD engine:

# + vscode={"languageId": "python"}
model = Model()
# -

# ## Step 1: define GMSH entites
#
# You can use any object or transformation from the GMSH OCC kernel directly:

# + vscode={"languageId": "python"}
mysphere = gmsh.model.occ.addSphere(0, 0, 0, 1)
# -

# Meshwell also introduces new object classes (PolySurfaces and Prisms) that simplify definition of complex shapes:

# + vscode={"languageId": "python"}
# We use shapely as the interface to describe polygons
mypolygon = shapely.Polygon(
    [
        [-1, -1],
        [-1, 1],
        [1, 1],
        [1, -1],
    ]
)
# We can "extrude" the polygon in 3D, with offsets
buffers = {
    -0.5: 1.0,
    0.5: 0.0,
}

mywedge = Prism(polygons=mypolygon, buffers=buffers, model=model)
# -

# ## Step 2: define the mesh
#
# Provide meshwell with an Ordered dictionary, with physical labels as keys and a list of GMSH entities as values. Entities higher in the dict take precedence; you can perform your own GMSH booleans prior to the meshing to more finely control subregion names.

# + vscode={"languageId": "python"}
dimtags_dict = OrderedDict(
    {
        "wedge": [(3, mywedge)],
        "sphere": [(3, mysphere)],
    }
)

geometry = model.mesh(
    dimtags_dict=dimtags_dict, verbosity=False, filename="quickmesh.msh"
)
# -

# This yields:

# + vscode={"languageId": "python"}
mesh = from_meshio(meshio.read("quickmesh.msh"))
draw_mesh3d(mesh)
# -

# The gmsh gui (`gmsh quickmesh.msh` in terminal) allows easy inspection of the mesh.

# ## Step 3: use the mesh
#
# The returned mesh has all maximum dimension entities (volumes for 3D, surfaces for 2D) and [maximum - 1] dimension (surfaces for 3D, lines for 2D) properly labeled:
#

# + vscode={"languageId": "python"}
geometry.cell_sets.keys()
# -

# The mesh() function has many more arguments that can give you more control on the mesh and labels generated.
