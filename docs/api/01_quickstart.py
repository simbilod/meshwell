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
from shapely.plotting import plot_polygon
import matplotlib.pyplot as plt
import gmsh
from meshwell.polysurface import PolySurface
from skfem.visuals.matplotlib import draw_mesh2d
from skfem.io.meshio import from_meshio
import meshio
from skfem.visuals.matplotlib import draw_mesh3d
from meshwell.prism import Prism

# -

# Meshwell is a Python wrapper around GMSH that provides:
#
# (1) a Prism class that simplifies, to the point of automating, the definition of solids from arbitrary (multi)polygons with "buffered" extrusions
#
# For instance, consider some complicated polygon resulting from some upstream calculation:

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
polygon_with_holes_boolean = polygon_with_holes - shapely.Point(-2, -2).buffer(2)

fig = plt.figure()
ax = fig.add_subplot()
plot_polygon(polygon_with_holes_boolean, ax=ax, add_points=False)
plt.show()
# -

# Meshwell's PolySurface can easily convert this to a 2D GMSH entity:

# +

# Some GMSH boilerplate
gmsh.initialize()
occ = gmsh.model.occ

# This package
poly2D = PolySurface(polygons=polygon_with_holes_boolean, model=occ)

# More GMSH boilerplate
occ.synchronize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.mesh.generate(2)
gmsh.write("mesh2D.msh")

# Plotting courtesy of scikit-fem
mesh = from_meshio(meshio.read("mesh2D.msh"))

draw_mesh2d(mesh)

# +

polygon1 = shapely.Point(0, 0).buffer(2)
polygon2 = shapely.Point(3, 3).buffer(1)
polygon = shapely.MultiPolygon([polygon1, polygon2])

# Combine with "buffers" to richly extrude in 3D as a Prism
buffers = {
    -2.0: 1.0,
    -1.0: 0.0,
    0.0: -0.5,  # z-coordinate: buffer (shrink or grow) to apply to original polygon
    0.2: 0.0,
    0.5: 1.0,
    0.8: 0.0,
    1.0: -0.5,
}

# Some GMSH boilerplate
gmsh.clear()
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
