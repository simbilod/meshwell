# %% [markdown]
# # GMSH Entities
#
# While meshwell provides convenient polygon-based entities (PolySurface, PolyPrism), you can also use arbitrary GMSH OCC (OpenCascade) geometry directly through the `GMSH_entity` class. This gives you full access to GMSH's geometric modeling capabilities while still benefiting from meshwell's workflow.

# %%
from functools import partial
import gmsh
from meshwell.gmsh_entity import GMSH_entity
from meshwell.cad import cad
from meshwell.mesh import mesh
from meshwell.visualization import plot3D
from meshwell.polysurface import PolySurface
import shapely
from pathlib import Path


# %% [markdown]
# ## Basic Example: Box
#
# The simplest use case is wrapping a built-in GMSH primitive. Here we create a box using `gmsh.model.occ.add_box`:

# %%
# Create a box entity using GMSH's built-in primitive
box_entity = GMSH_entity(
    gmsh_partial_function=partial(
        gmsh.model.occ.add_box,
        x=0,
        y=0,
        z=0,  # Bottom corner position
        dx=2,
        dy=2,
        dz=2,  # Dimensions
    ),
    physical_name="my_box",
    mesh_order=1,
)

# Generate the CAD geometry
cad(entities_list=[box_entity], output_file="gmsh_box.xao")

# Generate and visualize the mesh
box_mesh = mesh(
    dim=3,
    input_file="gmsh_box.xao",
    output_file="gmsh_box.msh",
    default_characteristic_length=0.5,
)

print(f"Box mesh: {len(box_mesh.points)} vertices")
plot3D(box_mesh, title="GMSH Box Entity")

# %% [markdown]
# ## Custom Geometry Function
#
# For more complex shapes, you can define your own geometry construction function. The function should:
# - (Optionally) Accept geometric parameters as arguments
# - Use GMSH OCC commands to build the geometry
# - Return the tag of the created entity
#
# Since this function will be evaluated during CAD definition, we can assume that a GMSH model will already be activated.
#
# Here's an example that creates a rectangular surface in 3D space:


# %%
def front_face_rectangle(x_min, x_max, y, z_min, z_max):
    """Create a rectangular surface at a fixed y-coordinate.

    Note: This assumes the GMSH model is already initialized.
    """
    # Define the four corners of the rectangle
    p1 = gmsh.model.occ.addPoint(x_min, y, z_min)
    p2 = gmsh.model.occ.addPoint(x_max, y, z_min)
    p3 = gmsh.model.occ.addPoint(x_max, y, z_max)
    p4 = gmsh.model.occ.addPoint(x_min, y, z_max)

    # Create lines connecting the points
    l1 = gmsh.model.occ.addLine(p1, p2)
    l2 = gmsh.model.occ.addLine(p2, p3)
    l3 = gmsh.model.occ.addLine(p3, p4)
    l4 = gmsh.model.occ.addLine(p4, p1)

    # Create a curve loop and plane surface
    loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
    surface = gmsh.model.occ.addPlaneSurface([loop])

    return surface


# Create the entity with our custom function
custom_entity = GMSH_entity(
    gmsh_partial_function=partial(
        front_face_rectangle, x_min=0, x_max=3, y=0, z_min=0, z_max=2
    ),
    physical_name="custom_surface",
    mesh_order=1,
    dimension=2,  # Specify that this is a 2D surface
)

# Generate CAD and mesh
cad(entities_list=[custom_entity], output_file="custom_surface.xao")

custom_mesh = mesh(
    dim=2,
    input_file="custom_surface.xao",
    output_file="custom_surface.msh",
    default_characteristic_length=0.3,
)

print(f"Custom surface mesh: {len(custom_mesh.points)} vertices")
plot3D(custom_mesh, title="Custom GMSH Surface")

# %% [markdown]
# ## Combining GMSH Entities with Meshwell Entities
#
# You can mix GMSH entities with regular meshwell entities (PolySurface, PolyPrism) in the same model. The `mesh_order` parameter controls boolean operations between overlapping entities.

# %%

# Create a GMSH sphere
sphere_entity = GMSH_entity(
    gmsh_partial_function=partial(
        gmsh.model.occ.add_sphere, xc=0, yc=0, zc=0, radius=1.5  # Center
    ),
    physical_name="sphere",
    mesh_order=1,
)

# Create a meshwell box that will be subtracted
polygon = shapely.box(-1, -1, 1, 1)
box_surface = PolySurface(
    polygons=polygon,
    physical_name="box_cutout",
    mesh_order=2,  # Higher order = subtracted from sphere
)

# Combine both entities
cad(entities_list=[sphere_entity, box_surface], output_file="combined.xao")

combined_mesh = mesh(
    dim=2,
    input_file="combined.xao",
    output_file="combined.msh",
    default_characteristic_length=0.3,
)

print(f"Combined mesh: {len(combined_mesh.points)} vertices")
plot3D(combined_mesh, title="GMSH + Meshwell Combined")

# %% [markdown]
# ## Key Concepts
#
# - **`gmsh_partial_function`**: A `functools.partial` object that will be evaluated during CAD generation
# - **`dimension`**: Explicitly specify the topological dimension (1=line, 2=surface, 3=volume) if needed
# - **`physical_name`**: Label for the entity in the mesh file
# - **`mesh_order`**: Controls boolean operations with overlapping entities (lower order takes precedence)
#
# ## When to Use GMSH Entities
#
# Use `GMSH_entity` when you need:
# - GMSH's built-in primitives (sphere, cylinder, cone, torus, etc.)
# - Complex CAD operations (fillets, chamfers, extrusions with twist)
# - Import of STEP/IGES files
# - Precise control over GMSH's OpenCascade kernel
#
# For simple polygon-based geometries, `PolySurface` and `PolyPrism` are more convenient.

# %%
# Clean up files

for f in [
    "gmsh_box.xao",
    "gmsh_box.msh",
    "custom_surface.xao",
    "custom_surface.msh",
    "combined.xao",
    "combined.msh",
]:
    Path(f).unlink(missing_ok=True)
