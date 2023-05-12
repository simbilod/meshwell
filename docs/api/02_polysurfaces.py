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

# # Polysurfaces

# + tags=["hide-input"]
import shapely
from shapely.plotting import plot_polygon
import matplotlib.pyplot as plt
from meshwell.model import Model
from meshwell.polysurface import PolySurface
from skfem.visuals.matplotlib import draw_mesh2d
from skfem.io.meshio import from_meshio
import meshio

# -

# GMSH is a powerful meshing engine, but non-standard shapes (e.g. arbitrary polygons) are still most easily described from the bottom-up, by defining vertices, lines, and closed loops.
#
# Meshwell has a "PolySurface" object simplifying this process. It takes as an argument a shapely (Multi)Polygon:

# +

polygon_with_holes = shapely.Polygon(
    [[-2, -2], [3, -2], [3, 2], [-2, 2], [-2, -2]],
    holes=(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [1.0, -1.0],
            [0.0, 0.0],
        ],
    ),
)
polygon_with_holes_boolean = shapely.union(
    polygon_with_holes - shapely.Point(-2, -2).buffer(2), shapely.Point(0, 2).buffer(1)
)

fig = plt.figure()
ax = fig.add_subplot()
plot_polygon(polygon_with_holes_boolean, ax=ax, add_points=False)
plt.show()

# +
model = Model()

poly2D = PolySurface(polygons=polygon_with_holes_boolean, model=model)

model.mesh(dimtags_dict={"polygon": [(2, poly2D)]}, filename="mesh2D.msh")

# Plotting courtesy of scikit-fem
mesh = from_meshio(meshio.read("mesh2D.msh"))

draw_mesh2d(mesh)
# -

# Although shapely does not work in 3D, it accepts 3D coordinates, which are used here to define surfaces in 3D space.

# ## Some notes
#
# * All polygon verticles are instanciated as 0-D points in the GMSH model
# * All polygon edges (interior and exterior) are instanciated as 1-D lines in the GMSH model
# * "PolySurface" returns the ID(s) of the polygon surfaces, but instanciating the object is enough to add the entity to the GMSH model
# * Ill-formed polygons will cause any meshing to fail
