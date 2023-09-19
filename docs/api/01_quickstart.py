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

# + tags=["hide-input"]
import shapely
from meshwell.prism import Prism
from meshwell.model import Model
from meshwell.gmsh_entity import GMSH_entity
from skfem.visuals.matplotlib import draw_mesh3d
from skfem.io.meshio import from_meshio
import meshio

# -

# # Step 0
#
# Initialize the CAD engine:

model = Model()

# ## Step 1: define GMSH entites
#
# You can use any object or transformation from the GMSH OCC kernel through a delayed evaluation wrapper:

mysphere = GMSH_entity(
    gmsh_function=model.occ.addSphere,
    gmsh_function_kwargs={"xc": 0, "yc": 0, "zc": 0, "radius": 1},
    dim=3,
    model=model,
    mesh_order=2,
    physical_name="sphere",
    dimension=3,
)

# Meshwell also introduces new object classes (PolySurfaces and Prisms) that simplify definition of complex shapes:

# +
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

mywedge = Prism(
    polygons=mypolygon,
    buffers=buffers,
    model=model,
    mesh_order=1,
    physical_name="wedge",
)
# -

# ## Step 2: define the mesh
#
# Provide meshwell with an Ordered dictionary, with physical labels as keys and a list of meshwell entities as values. Entities higher in the dict take precedence.

# +
entities_list = [mywedge, mysphere]

geometry = model.mesh(
    entities_list=entities_list, verbosity=False, filename="quickmesh.msh"
)
# -

# This yields:

mesh = from_meshio(meshio.read("quickmesh.msh"))
draw_mesh3d(mesh)

# The gmsh gui (`gmsh quickmesh.msh` in terminal) allows easy inspection of the mesh.

# ## Step 3: use the mesh
#
# The returned mesh has all maximum dimension entities (volumes for 3D, surfaces for 2D) and [maximum - 1] dimension (surfaces for 3D, lines for 2D) properly labeled:
#

geometry.cell_sets.keys()

# The mesh() function has many more arguments that can give you more control on the mesh and labels generated.
