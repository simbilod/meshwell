# ruff: noqa: E402

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

# # Introduction
#
# Meshwell is a Python wrapper around GMSH that provides:
#
# (1) a Prism class that simplifies, to the point of automating, the definition of solids from arbitrary (multi)polygons with "buffered" extrusions
#
# For instance, consider some complicated polygon resulting from some upstream calculation:

# +
import shapely
from shapely.plotting import plot_polygon
import matplotlib.pyplot as plt

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
# -

# Meshwell's PolySurface can easily convert this to a 2D GMSH entity:

# +
import gmsh
from meshwell.polysurface import PolySurface
from skfem.visuals.matplotlib import draw_mesh2d
from skfem.io.meshio import from_meshio
import meshio

# Some GMSH boilerplate
gmsh.clear()
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
# -

# Given a planar polygon, it is also possible to extrude with arbitrary grow and shrink patterns as a function of the z-coordinates, using a dictionary of buffer values:

# +
from skfem.visuals.matplotlib import draw_mesh3d
from meshwell.prism import Prism

polygon1 = shapely.Point(0, 0).buffer(2)
polygon2 = shapely.Point(4, 0).buffer(1)
polygon = shapely.MultiPolygon(
    [polygon1, polygon2]
)  # showing a MultiPolygon input for flexibility

# Combine with "buffers" to richly extrude in 3D as a Prism
buffers = {
    -2.0: 1.0,  # z-coordinate: buffer (shrink or grow) to apply to original polygon
    -1.0: 0.0,
    0.0: -0.5,
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
