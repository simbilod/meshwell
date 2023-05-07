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

# # Prisms
#
# Meshwell can extrude polygonal entities (not already meshed) in the z-direction to define complex 3D shapes.
#
# This extrusion is coupled to morphological grow/shrink operations ("buffers") to define complex shapes.
#
# THe Prism object takes in a (Multi)Polygon and a dict of z-values: buffer-values. A buffer value of 0 keeps the polygon intact; a negative value shrink the polygon by this amounts; and a positive value grows the polygon similarly.

# + tags=["hide-input"]
import shapely
from shapely.plotting import plot_polygon
import matplotlib.pyplot as plt
import gmsh
from skfem.io.meshio import from_meshio
import meshio
from skfem.visuals.matplotlib import draw_mesh3d
from meshwell.prism import Prism

# -

# First define some polygon:

# +
polygon = shapely.Point(0, 0).buffer(2)

fig = plt.figure()
ax = fig.add_subplot()
plot_polygon(polygon, ax=ax, add_points=False)
plt.show()


# +
# Combine with "buffers" to richly extrude in 3D as a Prism
buffers = {
    -2.0: 1.0,  # at z = -2, grow the polygon by 1 unit
    -1.0: 0.0,  # at z = -1, use the base polygon
    0.0: -0.5,  # at z = 0.0, shrink the base polygon by 1/2 a unit
    0.2: 0.0,  # etc.
    0.5: 1.0,
    0.8: 0.0,
    1.0: -0.5,
}

# Some GMSH boilerplate
gmsh.initialize()
occ = gmsh.model.occ

# This package
poly3D = Prism(polygons=polygon, buffers=buffers, model=occ)

# More GMSH boilerplate
occ.synchronize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.mesh.generate(3)
gmsh.write("mesh3D.msh")

# Plotting courtesy of scikit-fem
mesh = from_meshio(meshio.read("mesh3D.msh"))

draw_mesh3d(mesh)
