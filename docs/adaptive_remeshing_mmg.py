# %% [markdown]
# # Adaptive Remeshing with MMG
#
# Meshwell supports adaptive remeshing using MMG as a backend. This provides an alternative to the default GMSH-based remeshing, often yielding higher quality meshes and supporting anisotropic refinement.
#
# The `remesh_mmg` module provides functionality to:
# 1.  Load an existing mesh.
# 2.  Define remeshing strategies (same as standard remeshing).
# 3.  Generate a new mesh using MMG.
#
# In this example, we will use the same geometry and strategies as the standard example but use MMG for the remeshing step.

# %%
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import shapely
from meshwell.cad import cad
from meshwell.mesh import mesh
from meshwell.remesh import (
    remesh_mmg,
    BinaryScalingStrategy,
    MMGRemeshingStrategy,
    compute_total_size_map,
)
import meshio
from meshwell.polysurface import PolySurface
from meshwell.visualization import plot2D
from meshwell.polyprism import PolyPrism
from meshwell.visualization import plot3D

# %% [markdown]
# ## Define Geometry
#
# We define two adjacent rectangles with different physical tags.

# %%
# Define geometry
large_rect = 10
mid_rect = 2

# Box 1: inner box
polygon1 = shapely.Polygon(
    [
        [-large_rect / 2, -mid_rect / 2],
        [large_rect / 2, -mid_rect / 2],
        [large_rect / 2, mid_rect / 2],
        [-large_rect / 2, mid_rect / 2],
        [-large_rect / 2, -mid_rect / 2],
    ],
)

# Box 2: global box
polygon2 = shapely.Polygon(
    [
        [-large_rect / 2, -large_rect / 2],
        [large_rect / 2, -large_rect / 2],
        [large_rect / 2, large_rect / 2],
        [-large_rect / 2, large_rect / 2],
        [-large_rect / 2, -large_rect / 2],
    ],
)

poly_obj1 = PolySurface(
    polygons=polygon1,
    mesh_order=1,
    physical_name="inner_box",
)
poly_obj2 = PolySurface(
    polygons=polygon2,
    mesh_order=2,
    physical_name="outer_box",
)

entities_list = [poly_obj1, poly_obj2]

# Generate CAD
cad(
    entities_list=entities_list,
    output_file="remesh_mmg_example.xao",
)

# %% [markdown]
# ## Initial Mesh
#
# We generate a coarse initial mesh.

# %%
# Generate initial mesh
initial_mesh = mesh(
    dim=2,
    input_file="remesh_mmg_example.xao",
    output_file="remesh_mmg_example_initial.msh",
    default_characteristic_length=1.0,  # Coarse mesh
    n_threads=1,
    gmsh_version=2.2,
)

# %%
# Load initial mesh
initial_mesh = meshio.read("remesh_mmg_example_initial.msh")
print(f"Initial mesh points: {len(initial_mesh.points)}")
plot2D(initial_mesh, title="Initial Coarse Mesh", wireframe=True)

# %% [markdown]
# ## Define Remeshing Strategy
#
# We define a strategy that refines the mesh along an oval shape.

# %%
# Define oval parameters
center_x, center_y = 0.0, 0.0
radius_x, radius_y = 3.0, 3.0


def oval_looking_data(coords, data=None):
    """Calculate solution/error based on proximity to oval boundary.
    Returns 1.0 if close to boundary, 0.0 otherwise.
    """
    x = coords[:, 0]
    y = coords[:, 1]

    # Normalized distance from center
    normalized_dist = ((x - center_x) ** 2 / radius_x**2) + (
        (y - center_y) ** 2 / radius_y**2
    )
    dist_from_boundary = np.abs(np.sqrt(normalized_dist) - 1.0)

    # Invert distance: high value near boundary
    # e.g., 1.0 at boundary, decaying to 0.0 at distance 1.0
    return np.maximum(0, 1.0 - dist_from_boundary)


# Calculate solution/error on initial mesh points
data_values = oval_looking_data(initial_mesh.points)

# refinement_data as (N, 4) -> x, y, z, data
refinement_data = np.column_stack([initial_mesh.points, data_values])

# Create strategy with refinement_data
strategy = BinaryScalingStrategy(
    threshold=0.7,
    factor=0.2,
    refinement_data=refinement_data,
    min_size=0.1,
    max_size=2.0,
    field_smoothing_steps=2,
)

# %% [markdown]
# ## Perform Remeshing with MMG
#
# We use `remesh_mmg` to refine the mesh. Note that we don't need to pass the geometry file for MMG, as it operates on the mesh directly (though preserving boundaries is handled by MMG's internal logic).

# %%
size_map = remesh_mmg(
    input_mesh=initial_mesh,
    output_mesh=Path("remesh_mmg_example_final.msh"),
    strategies=[strategy],
    dim=2,
    verbosity=1,
    mmg_executable="mmg2d_O3",  # Ensure this is in your PATH or provide full path
)

# %% [markdown]
# ## Visualize Result
#
# We load and plot the final mesh to see the refinement.

# %%
final_mesh = meshio.read("remesh_mmg_example_final.msh")
print(f"Final mesh points: {len(final_mesh.points)}")
print("Final mesh physical groups:", final_mesh.field_data)

plot2D(final_mesh, title="Adaptive Remesh with MMG (Oval Refinement)", wireframe=True)

# %% [markdown]
# ## Multiple Strategies with Grid Evaluation
#
# Just like the standard remesher, we can use multiple strategies and grid-based evaluation.

# %%
# Generate a dense grid of points
x = np.linspace(-large_rect / 2, large_rect / 2, 100)
y = np.linspace(-large_rect / 2, large_rect / 2, 100)
X, Y = np.meshgrid(x, y)
grid_coords = np.column_stack([X.ravel(), Y.ravel(), np.zeros_like(X.ravel())])

# Evaluate solution/error on the grid
grid_data = oval_looking_data(grid_coords)
grid_refinement_data = np.column_stack([grid_coords, grid_data])

grid_strategy = BinaryScalingStrategy(
    func=oval_looking_data,
    threshold=0.8,
    factor=0.2,
    refinement_data=grid_refinement_data,
    min_size=0.1,
    max_size=2.0,
)

# %% [markdown]
# ## Soft Thresholding with Fine Grid (MMG)
#
# We can use MMG's gradation parameter (hgrad) to create a smooth transition between refined and coarse regions.
# This avoids the need for a manual sigmoid function.

# %%

soft_grid_strategy = MMGRemeshingStrategy(
    threshold=0.8,
    factor=0.1,
    refinement_data=grid_refinement_data,
    min_size=0.1,
    max_size=2.0,
    hgrad=1.1,  # Control gradation for smoothness
)

# Compute size map for visualization
soft_size_map = compute_total_size_map(
    input_mesh=initial_mesh,
    strategies=[soft_grid_strategy],
)

# Visualize the soft size field
plt.figure(figsize=(10, 10))
sc = plt.scatter(
    soft_size_map[:, 0],
    soft_size_map[:, 1],
    c=soft_size_map[:, 3],
    cmap="viridis_r",
    s=3,
)
plt.colorbar(sc, label="Target Mesh Size")
plt.title("Soft Threshold Size Field (MMG hgrad)")
plt.axis("equal")
plt.show()

# Remesh with soft threshold
remesh_mmg(
    input_mesh=initial_mesh,
    output_mesh=Path("remesh_mmg_example_soft.msh"),
    strategies=[soft_grid_strategy],
    dim=2,
    verbosity=1,
)

# Visualize final mesh
soft_mesh = meshio.read("remesh_mmg_example_soft.msh")
print(f"Soft threshold mesh points: {len(soft_mesh.points)}")
plot2D(soft_mesh, title="Soft Threshold Refinement (MMG)", wireframe=True)

# %% [markdown]
# ## MMG Parameter Effects
#
# Let's demonstrate how different MMG parameters affect the mesh quality.
# We'll compare different values of `hausd` (Hausdorff distance) and `hgrad` (gradation).

# %%
# Create a simple circular refinement strategy
circle_center = np.array([0.0, 0.0, 0.0])
circle_radius = 2.0


def simple_circle_looking_data(coords, data=None):
    """Calculate solution/error based on proximity to circle boundary."""
    if data is not None:
        return data

    x = coords[:, 0]
    y = coords[:, 1]

    dist_from_center = np.sqrt(
        (x - circle_center[0]) ** 2 + (y - circle_center[1]) ** 2
    )
    dist_from_boundary = np.abs(dist_from_center - circle_radius)

    # High value near boundary
    return np.maximum(0, 1.0 - dist_from_boundary / 0.5)


# Test different parameter combinations
param_configs = [
    {"name": "Default", "hausd": None, "hgrad": None},
    {"name": "Fine Hausdorff", "hausd": 0.01, "hgrad": None},
    {"name": "Smooth Gradation", "hausd": None, "hgrad": 1.1},
    {"name": "Fine + Smooth", "hausd": 0.01, "hgrad": 1.1},
]

# Create figure for comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
axes = axes.flatten()

for idx, config in enumerate(param_configs):
    # Create strategy with specific parameters
    strategy = MMGRemeshingStrategy(
        func=simple_circle_looking_data,
        threshold=0.8,
        factor=0.2,
        min_size=0.1,
        max_size=1.0,
        refinement_data=None,
        hausd=config["hausd"],
        hgrad=config["hgrad"],
    )

    # Remesh
    output_file = Path(f"remesh_mmg_params_{idx}.msh")
    remesh_mmg(
        input_mesh=initial_mesh,
        output_mesh=output_file,
        strategies=[strategy],
        dim=2,
        verbosity=0,
    )

    # Load and plot
    result_mesh = meshio.read(output_file)

    ax = axes[idx]
    # Plot mesh edges
    for cell_block in result_mesh.cells:
        if cell_block.type == "triangle":
            for tri in cell_block.data:
                pts = result_mesh.points[tri]
                triangle = plt.Polygon(
                    pts[:, :2], fill=False, edgecolor="blue", linewidth=0.5
                )
                ax.add_patch(triangle)

    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.5, 5.5)
    ax.set_aspect("equal")
    ax.set_title(f"{config['name']}\n({len(result_mesh.points)} vertices)")
    ax.grid(True, alpha=0.3)

    # Clean up
    output_file.unlink()

plt.tight_layout()
plt.show()

print("\nParameter Effects:")
print("- hausd (Hausdorff): Controls surface approximation quality (smaller = finer)")
print(
    "- hgrad (Gradation): Controls mesh size transition smoothness (smaller = smoother)"
)


# %% [markdown]
# ## Multiple Strategies
#
# We can combine multiple strategies to refine different regions. Here we'll refine around a circle and along a vertical line.

# %%
# Define circle parameters
circle_center_x, circle_center_y = 2.0, 2.0
circle_radius = 1.5


def circle_looking_data(coords, data=None):
    """Calculate solution/error based on proximity to circle boundary."""
    if data is not None:
        return data

    x = coords[:, 0]
    y = coords[:, 1]

    # Distance from circle center
    dist_from_center = np.sqrt((x - circle_center_x) ** 2 + (y - circle_center_y) ** 2)
    dist_from_boundary = np.abs(dist_from_center - circle_radius)

    # High value near boundary
    return np.maximum(0, 1.0 - dist_from_boundary / 0.5)


# Define line parameters (vertical line at x = -2)
line_x = -2.0
line_width = 0.3


def line_looking_data(coords, data=None):
    """Calculate solution/error based on proximity to vertical line."""
    if data is not None:
        return data

    x = coords[:, 0]

    # Distance from line
    dist_from_line = np.abs(x - line_x)

    # High value near line
    return np.maximum(0, 1.0 - dist_from_line / line_width)


# Create strategies
# Note: We can pass data if we want to pre-calculate, but here we let the function evaluate on the mesh nodes
circle_strategy = BinaryScalingStrategy(
    func=circle_looking_data,
    threshold=0.5,
    factor=0.15,
    min_size=0.05,
    max_size=2.0,
    refinement_data=None,
)

line_strategy = BinaryScalingStrategy(
    func=line_looking_data,
    threshold=0.4,
    factor=0.2,
    min_size=0.08,
    max_size=2.0,
    refinement_data=None,
)

# Combine strategies
multi_size_map = remesh_mmg(
    input_mesh=initial_mesh,
    output_mesh=Path("remesh_mmg_example_multi.msh"),
    strategies=[circle_strategy, line_strategy],
    dim=2,
    verbosity=1,
)

# Load and visualize the final mesh
multi_mesh = meshio.read("remesh_mmg_example_multi.msh")
print(f"Multi-strategy mesh points: {len(multi_mesh.points)}")
plot2D(
    multi_mesh,
    title="Multi-Strategy Refinement with MMG (Circle + Line)",
    wireframe=True,
)

# %% [markdown]
# ## 3D Remeshing (MMG)
#
# We can also remesh 3D geometries using MMG3D. Here we'll create a prism using `PolyPrism` and refine it.

# %%
# Create a 3D prism geometry

# Define base polygon
polygon = shapely.Polygon([[-2, -2], [2, -2], [2, 2], [-2, 2], [-2, -2]])

# Define buffers for extrusion (z-levels)
buffers = {0.0: 0.0, 4.0: 0.0}  # Extrude from z=0 to z=4

poly_prism = PolyPrism(
    polygons=polygon,
    buffers=buffers,
    physical_name="my_prism",
)

# Generate CAD
cad(
    entities_list=[poly_prism],
    output_file="remesh_mmg_example_3d.xao",
)

# Generate initial 3D mesh
mesh_3d_initial = mesh(
    dim=3,
    input_file="remesh_mmg_example_3d.xao",
    output_file="remesh_mmg_example_3d_initial.msh",
    default_characteristic_length=1.0,
)

# Visualize initial mesh
print(f"3D Initial mesh points: {len(mesh_3d_initial.points)}")
plot3D(mesh_3d_initial, title="3D Initial Mesh")

# Define spherical strategy
sphere_center = np.array([0.0, 0.0, 2.0])  # Center in the prism
sphere_radius = 1.0


def sphere_looking_data(coords, data=None):
    """Calculate solution/error based on proximity to sphere."""
    if data is not None:
        return data

    # Distance from sphere center
    dist = np.linalg.norm(coords - sphere_center, axis=1)
    dist_from_boundary = np.abs(dist - sphere_radius)

    # High value near boundary
    return np.maximum(0, 1.0 - dist_from_boundary / 0.5)


# Define spherical strategy using MMGRemeshingStrategy to pass hausd
sphere_strategy = MMGRemeshingStrategy(
    func=sphere_looking_data,
    threshold=0.5,
    factor=0.5,
    min_size=0.2,
    max_size=1.0,
    refinement_data=None,
    hausd=0.01,  # Control Hausdorff distance for better surface approximation
    hgrad=1.3,  # Control gradation
)

# Remesh 3D with MMG
try:
    size_map_3d = remesh_mmg(
        input_mesh=Path("remesh_mmg_example_3d_initial.msh"),
        output_mesh=Path("remesh_mmg_example_3d_final.msh"),
        strategies=[sphere_strategy],
        dim=3,
        verbosity=1,
        mmg_executable="mmg3d_O3",  # Use MMG3D for 3D meshes
    )

    # Visualize final mesh
    mesh_3d_final = meshio.read("remesh_mmg_example_3d_final.msh")
    print(f"3D Final mesh points: {len(mesh_3d_final.points)}")
    plot3D(mesh_3d_final, title="3D Remeshing (MMG)")
except Exception as e:
    print(f"3D remeshing failed (likely due to missing executable): {e}")

# %%
# Clean up files
for f in [
    "remesh_mmg_example.xao",
    "remesh_mmg_example_initial.msh",
    "remesh_mmg_example_final.msh",
    "remesh_mmg_example_multi.msh",
    "remesh_mmg_example_soft.msh",
    "remesh_mmg_example_3d_initial.msh",
    "remesh_mmg_example_3d_final.msh",
]:
    Path(f).unlink(missing_ok=True)
