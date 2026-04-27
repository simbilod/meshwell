# %% [markdown]
# # Prisms
# Polygons can be associated with arbitrarily complex extrusion rules to form 3D Prisms.

# %%
import matplotlib.pyplot as plt
import shapely

from meshwell.cad_occ import cad_occ
from meshwell.mesh import mesh
from meshwell.occ_xao_writer import write_xao
from meshwell.orchestrator import generate_mesh
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

write_xao(cad_occ(entities_list), "prism.xao")

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

# %% [markdown]
# # Structured (layered) extruded prisms
#
# Passing `n_layers=` to `PolyPrism` switches it into structured mode
# (gmsh tutorial t3 style): each z-interval declared in `buffers` gets
# its own layer count from `n_layers`, producing a swept layered mesh
# whose interior xy-columns have exactly `n_layers + 1` distinct z-levels.
#
# The base polygon can be any shape -- arbitrary polygons are supported,
# unlike transfinite meshing which requires four-corner topology. Users
# can freely mix structured layered prisms with regular (taper,
# unstructured) `PolyPrism` entities and `PolySurface` entities in the
# same scene.
#
# Multiple structured prisms whose 3D extents overlap are resolved
# upfront by `mesh_order`: lower-priority prisms are sliced in xy and z
# so every structured volume in the final mesh is disjoint from every
# other structured volume. Layer counts are distributed proportionally
# across the resulting sub-pieces.

# %%
base = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

structured = PolyPrism(
    polygons=base,
    buffers={0.0: 0.0, 0.5: 0.0, 1.0: 0.0},  # two intervals
    n_layers=[8, 2],  # 8 layers, then 2
    physical_name="film_stack",
    recombine=False,
)

generate_mesh(
    entities=[structured],
    dim=3,
    output_mesh="structured_prism.msh",
    default_characteristic_length=0.2,
)

# %%
