import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box

from meshwell.model import ModelManager
from meshwell.parallel import decompose_domain
from meshwell.polysurface import PolySurface
from meshwell.resolution import ThresholdField
from meshwell.visualization import plot2D

# 1. Define global geometry
device_poly = box(0, 0, 10, 10)
device = PolySurface(polygons=device_poly, physical_name="background", mesh_order=3)

# Features that spans across multiple future subdomains
triangle_poly = Polygon([(2, 4), (8, 4), (8, 6)])
triangle = PolySurface(polygons=triangle_poly, physical_name="triangle", mesh_order=1)

waveguide_poly = box(2, 4, 8, 6)
waveguide = PolySurface(polygons=waveguide_poly, physical_name="box", mesh_order=2)

# Another feature
waveguide_boundary = box(0, 1, 5, 2)
waveguide_boundary = PolySurface(
    polygons=waveguide_boundary, physical_name="box_coinciding", mesh_order=1
)


entities = [device, triangle, waveguide, waveguide_boundary]

# 2. Decompose into multiple subdomains
# Here, we segment a 10x10 device into 4 smaller 5x5 quadrant tiles.
subdomains = [
    box(0, 0, 5, 5),  # Bottom-Left
    box(5, 0, 10, 5),  # Bottom-Right
    box(0, 5, 5, 10),  # Top-Left
    box(5, 5, 10, 10),  # Top-Right
]

# Set a resolution field to refine the mesh inside the waveguide
res = ThresholdField(
    sizemin=0.2, sizemax=1.0, distmin=0.0, distmax=1.0, apply_to="surfaces"
)
resolution_specs = {"triangle": [res], "box_coinciding": [res]}

# %% [markdown]
# ### Visualizing Domain Decomposition
# We can visualize the original shapes and their subdivided regions before meshing occurs.

# %%

fig, axs = plt.subplots(1, 2, figsize=(10, 5))


# Plot original entities
def plot_geoms(geom, ax, **kwargs):
    if isinstance(geom, list):
        for g in geom:
            plot_geoms(g, ax, **kwargs)
    elif geom.geom_type == "Polygon":
        x, y = geom.exterior.xy
        ax.plot(x, y, **kwargs)
    elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        for g in geom.geoms:
            if g.geom_type == "Polygon":
                x, y = g.exterior.xy
                ax.plot(x, y, **kwargs)


for entity in entities:
    color = "k" if entity.physical_name == "device" else "r"
    plot_geoms(entity.polygons, axs[0], color=color, label=entity.physical_name)

axs[0].set_title("Original Entities")
axs[0].legend()

# Plot subdomains
for i, sub in enumerate(subdomains):
    x, y = sub.exterior.xy
    axs[1].plot(x, y, label=f"Subdomain {i}")
axs[1].set_title("Subdomain Decomposition")
axs[1].legend()
plt.show()

# %% [markdown]
# ### The "Halo" Buffer
# To ensure perfect deterministic stitching of meshes at the boundary of adjacent subdomains without explicit constraints, the `mesh_parallel` framework extracts the domain intersecting each subdomain along with a strict padding, expanding the subset by `halo_buffer`.

# %%

tasks = decompose_domain(entities, subdomains, halo_buffer=1.0)
task_0 = tasks[0]

fig, ax = plt.subplots(figsize=(6, 6))


def fill_geoms(geom, ax, **kwargs):
    if isinstance(geom, list):
        for g in geom:
            fill_geoms(g, ax, **kwargs)
    elif geom.geom_type == "Polygon":
        x, y = geom.exterior.xy
        ax.fill(x, y, **kwargs)
    elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        for g in geom.geoms:
            if g.geom_type == "Polygon":
                x, y = g.exterior.xy
                ax.fill(x, y, **kwargs)


# Plot Subdomain 0's domain entities (strict intersection)
for ent in task_0["domain"]:
    label = (
        "Strict Domain"
        if "Strict Domain" not in ax.get_legend_handles_labels()[1]
        else ""
    )
    fill_geoms(ent.polygons, ax, alpha=0.5, color="blue", label=label)

# Plot Subdomain 0's halo entities (buffer intersection minus domain)
for ent in task_0["halo"]:
    label = (
        "Halo Buffer" if "Halo Buffer" not in ax.get_legend_handles_labels()[1] else ""
    )
    fill_geoms(ent.polygons, ax, alpha=0.5, color="red", label=label)

ax.set_title("Worker 0: Local CAD Input (Domain + Halo)")
ax.set_xlim(-1, 11)
ax.set_ylim(-1, 11)
ax.legend()
plt.show()

# %% [markdown]
# The Dask worker assigned to this subdomain will receive this combined Subdomain + Halo CAD state. GMSH evaluates the geometry inside the halo to deterministically spawn identical mesh boundary nodes overlapping with the neighboring subdomain. Before being written, the cells belonging to the "Halo Buffer" physical tags are transparently chopped off.
#
# Let's perform this operation globally using `mesh_parallel`:

# %%
# Build model
model = ModelManager()

# Run parallel meshing (distributing across 4 parallel jobs)
final_mesh = model.mesh_parallel(
    entities_list=entities,
    subdomains=subdomains,
    halo_buffer=1.0,  # Adding 1um halo ensures accurate un-fractured stitching on boundaries
    n_jobs=4,
    default_characteristic_length=1.0,
    resolution_specs=resolution_specs,
    dim=2,
)

final_mesh.field_data

# %% [markdown]
# We can visualize the final stitched parallel mesh output:
# %%
plot2D(final_mesh, wireframe=True)
