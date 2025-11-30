# %% [markdown]
# # Adaptive Remeshing
#
# Meshwell supports adaptive remeshing based on a provided size field. This allows you to refine or coarsen the mesh in specific regions based on your requirements.
#
# The `remesh` module provides functionality to:
# 1.  Load an existing mesh and geometry.
# 2.  Interpolate a size field from a set of points (x, y, z, size).
# 3.  Generate a new mesh that respects the size field.
#
# In this example, we will create a simple 2D geometry with two physical groups and refine the mesh based on the distance from a vertical line.

# %%
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import shapely
import meshio
from meshwell.cad import cad
from meshwell.mesh import mesh
from meshwell.remesh import remesh, Remesh
from meshwell.polysurface import PolySurface
from meshwell.visualization import plot2D

# %% [markdown]
# ## Define Geometry
#
# We define two adjacent rectangles with different physical tags.

# %%
# Define geometry
large_rect = 10
mid_rect = 5

# Box 1: Left side
polygon1 = shapely.Polygon(
    [[0, 0], [mid_rect, 0], [mid_rect, large_rect], [0, large_rect], [0, 0]],
)

# Box 2: Right side
polygon2 = shapely.Polygon(
    [
        [mid_rect, 0],
        [large_rect, 0],
        [large_rect, large_rect],
        [mid_rect, large_rect],
        [mid_rect, 0],
    ],
)

poly_obj1 = PolySurface(
    polygons=polygon1,
    mesh_order=1,
    physical_name="left_box",
)
poly_obj2 = PolySurface(
    polygons=polygon2,
    mesh_order=1,
    physical_name="right_box",
)

entities_list = [poly_obj1, poly_obj2]

# Generate CAD
cad(
    entities_list=entities_list,
    output_file="remesh_example.xao",
)

# %% [markdown]
# ## Initial Mesh
#
# We generate a coarse initial mesh.

# %%
# Generate initial mesh
initial_mesh = mesh(
    dim=2,
    input_file="remesh_example.xao",
    output_file="remesh_example_initial.msh",
    default_characteristic_length=2.0,  # Coarse mesh
    n_threads=1,
)

print(f"Initial mesh points: {len(initial_mesh.points)}")
plot2D(initial_mesh, title="Initial Coarse Mesh", wireframe=True)

# %% [markdown]
# ## Define Size Field
#
# We define a size field that refines the mesh along an oval shape.
# The size will be fine near the oval boundary and coarser elsewhere.

# %%
# Create size map
# Sample points in the domain
num_points = 5000
x = np.random.uniform(0, large_rect, num_points)
y = np.random.uniform(0, large_rect, num_points)
z = np.zeros(num_points)

# Define oval parameters
center_x, center_y = 5.0, 5.0
radius_x, radius_y = 4.0, 2.5

# Compute normalized distance from center (oval equation)
# (x-h)^2/a^2 + (y-k)^2/b^2 = 1
normalized_dist = ((x - center_x) ** 2 / radius_x**2) + (
    (y - center_y) ** 2 / radius_y**2
)
dist_from_boundary = np.abs(np.sqrt(normalized_dist) - 1.0)

# Define size function: fine near boundary, coarse away
# Size = 0.1 at boundary, increasing to 1.5 away
sizes = 0.1 + 0.5 * dist_from_boundary

size_map = np.column_stack([x, y, z, sizes])

# %% [markdown]
# ## Visualize Size Field
#
# It is helpful to visualize the target size field to verify it matches expectations.

# %%
plt.figure(figsize=(10, 10))
sc = plt.scatter(x, y, c=sizes, cmap="viridis_r", s=5)
plt.colorbar(sc, label="Target Mesh Size")
plt.title("Target Size Field (Oval Refinement)")
plt.axis("equal")
plt.show()

# %% [markdown]
# ## Perform Remeshing
#
# We use the `remesh` function to generate the new mesh.

# %%
remesh(
    input_mesh=Path("remesh_example_initial.msh"),
    geometry_file=Path("remesh_example.xao"),
    output_mesh=Path("remesh_example_final.msh"),
    size_map=size_map,
    dim=2,
    field_smoothing_steps=1,
    verbosity=0,
    n_threads=1,
)

# %% [markdown]
# ## Visualize Result
#
# We load and plot the final mesh to see the refinement.

# %%
final_mesh = meshio.read("remesh_example_final.msh")
print(f"Final mesh points: {len(final_mesh.points)}")

plot2D(final_mesh, title="Adaptive Remesh (Oval Refinement)", wireframe=True)

# %% [markdown]
# ## Data-Driven Refinement
#
# In this section, we demonstrate how to refine the mesh based on solution data.
# We will use the `refine_by_value_difference` function to refine the mesh where the solution value changes abruptly.

# %%
# Define a scalar field with a step change
# f(x, y) = 1 if x > 5 else 0
# We use the coordinates from the previous final mesh
points = initial_mesh.points
y = points[:, 1]
data = np.zeros(len(points))
data[y > 5.0] = 1.0


# Extract edges for connectivity
def get_edges(mesh):
    edges = set()
    for block in mesh.cells:
        if block.type == "triangle":
            for elem in block.data:
                edges.add(tuple(sorted((elem[0], elem[1]))))
                edges.add(tuple(sorted((elem[1], elem[2]))))
                edges.add(tuple(sorted((elem[2], elem[0]))))
    return np.array(list(edges))


edges = get_edges(initial_mesh)

# Get current mesh sizes using Remesh class
remesher_instance = Remesh()
current_sizes = remesher_instance.get_current_mesh_sizes(
    Path("remesh_example_initial.msh")
)

# Refine based on value difference
new_size_map = remesher_instance.refine_by_value_difference(
    coords=points,
    connectivity=edges,
    data=data,
    current_sizes=current_sizes,
    threshold=0.5,  # Data jumps from 0 to 1
    factor=0.5,  # Halve the size at the step
    min_size=0.05,
)

# %% [markdown]
# ## Visualize Refinement Field
#
# Let's verify the target sizes.

# %%
plt.figure(figsize=(10, 10))
sc = plt.scatter(
    new_size_map[:, 0], new_size_map[:, 1], c=new_size_map[:, 3], cmap="viridis_r", s=5
)
plt.colorbar(sc, label="Target Mesh Size")
plt.title("Target Size Field (Data-Driven)")
plt.axis("equal")
plt.show()

# %% [markdown]
# ## Perform Refinement
#
# Now we generate the new mesh using this data-driven size field.

# %%
remesh(
    input_mesh=Path("remesh_example_initial.msh"),
    geometry_file=Path("remesh_example.xao"),
    output_mesh=Path("remesh_example_refined.msh"),
    size_map=new_size_map,
    dim=2,
    field_smoothing_steps=3,
    verbosity=0,
    n_threads=1,
)

refined_mesh = meshio.read("remesh_example_refined.msh")
plot2D(refined_mesh, title="Data-Driven Refinement (Step at x=5)", wireframe=True)

# %%
# Clean up files
Path("remesh_example.xao").unlink(missing_ok=True)
Path("remesh_example_initial.msh").unlink(missing_ok=True)
Path("remesh_example_final.msh").unlink(missing_ok=True)
Path("remesh_example_refined.msh").unlink(missing_ok=True)
