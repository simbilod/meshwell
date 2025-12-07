# %% [markdown]
# # Prisms
# Polygons can be associated with arbitrarily complex extrusion rules to form 3D Prisms.

# %%
import matplotlib.pyplot as plt
import shapely

from meshwell.cad import cad
from meshwell.mesh import mesh
from meshwell.polyprism import PolyPrism
from meshwell.visualization import plot3D

# %%
# We use shapely as an API to enter polygons

# Initialize GMSH and create the mesh
polygon_hull = shapely.Polygon(
    [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
)
polygon_hole1 = polygon_hull.buffer(-0.5, join_style="mitre")
polygon = polygon_hull - polygon_hole1

# %% [markdown]
# View the polygons:

# %%
# Plot the shapely polygons
plt.figure(figsize=(8, 8))
plt.plot(*polygon.exterior.xy, "b-", label="Polygon 1 exterior")
plt.plot(*polygon.interiors[0].xy, "b--", label="Polygon 1 hole")
plt.axis("equal")
plt.title("Shapely Polygons")
plt.legend()
plt.show()

# %% [markdown]
# Mesh a prism by combining with buffers
# %%

buffers = {0.0: 0.05, 1.0: -0.05}

poly3D = PolyPrism(
    polygons=polygon,
    buffers=buffers,
    physical_name="my_prism1",
)

entities_list = [poly3D]

cad(
    entities_list=entities_list,
    output_file="prism.xao",
)

output_mesh = mesh(
    dim=3,
    input_file="prism.xao",
    output_file="prism.msh",
    default_characteristic_length=100,
)


# Read and plot the mesh

# %%


plot3D(output_mesh, title="Interactive 3D Mesh")
# %%
