# %% [markdown]
# # GMSH
# Since meshwell is a wrapper around GMSH, we will first review GMSH  Python API. For a more thorough explanation of the below, see the GMSH documentation (https://gmsh.info/doc/texinfo/gmsh.html), tutorials (https://gitlab.onelab.info/gmsh/gmsh/-/tree/master/tutorials/python) and API (https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/api/gmsh.py?ref_type=heads).

# %%
import gmsh
import meshio
from meshwell.visualization import plot2D

# %% [markdown]
# ## Bottom-up construction of CAD entities
# The most general way to create CAD entities in GMSH is "bottom-up", going from points --> lines --> closed loops --> surfaces --> closed shells --> volumes

# %%
# Initialize GMSH
gmsh.initialize()
model = gmsh.model
model.add("bottom_up")

# Create points
p1 = model.occ.addPoint(0, 0, 0)
p2 = model.occ.addPoint(2, 0, 0)
p3 = model.occ.addPoint(2, 1, 0)
p4 = model.occ.addPoint(1, 1, 0)
p5 = model.occ.addPoint(1, 2, 0)
p6 = model.occ.addPoint(0, 2, 0)

# Create lines
l1 = model.occ.addLine(p1, p2)
l2 = model.occ.addLine(p2, p3)
l3 = model.occ.addLine(p3, p4)
l4 = model.occ.addLine(p4, p5)
l5 = model.occ.addLine(p5, p6)
l6 = model.occ.addLine(p6, p1)

# Create curve loop
cl = model.occ.addCurveLoop([l1, l2, l3, l4, l5, l6])

# Create surface
s1 = model.occ.addPlaneSurface([cl])

# Create the mesh
model.occ.synchronize()
model.mesh.generate(2)

gmsh.write("bottom_up.xao")
gmsh.write("bottom_up.msh")
gmsh.finalize()

# Read the mesh
mesh = meshio.read("bottom_up.msh")

# %%
try:
    gmsh.initialize()
    gmsh.open("bottom_up.xao")
    gmsh.fltk.run()
except:  # noqa: E722
    print("Skipping CAD GUI visualization - only available when running locally")

# %%
try:
    gmsh.initialize()
    gmsh.open("bottom_up.msh")
    gmsh.fltk.run()
except:  # noqa: E722
    print("Skipping mesh GUI visualization - only available when running locally")

# %%
plot2D(mesh, wireframe=True)

# %% [markdown]
# ## Construction of CAD entities from primitives
# A limited set of primitives (rectangles, circles, arcs, spheres, boxes, etc.) are also already implemented in GMSH:
# %%
gmsh.initialize()
model = gmsh.model
model.add("bottom_up")

# Create rectangle
box = model.occ.addRectangle(0, 0, 0, 1, 1)

# Create circle
circle = model.occ.addDisk(3, 0, 0, 0.5, 0.5)

# Create the mesh
model.occ.synchronize()
model.mesh.generate(2)
gmsh.write("primitives.msh")
gmsh.finalize()

# %%
mesh = meshio.read("primitives.msh")
plot2D(mesh, wireframe=True)

# %% [markdown]
# ## Constructive geometry operations
# More complex elementary entities can also be created from constructive geometry operations (cut, fuse, intersect):

# %%
gmsh.initialize()
model = gmsh.model
model.add("bottom_up")

# Create rectangle
box = model.occ.addRectangle(-1, -1, 0, 2, 2)

# Create circle
circle = model.occ.addDisk(0, 0, 0, 0.5, 0.5)

# Keep difference, delete originals
difference = model.occ.cut(
    [(2, box)], [(2, circle)], removeObject=True, removeTool=True
)

# Create the mesh
model.occ.synchronize()
model.mesh.generate(2)
gmsh.write("booleans.msh")
gmsh.finalize()

# %%
mesh = meshio.read("booleans.msh")
plot2D(mesh, wireframe=True)

# %% [markdown]
# ## Physical entities
# It is almost always extremely useful to be able to refer to all of the mesh nodes within a set of elementary entities. In GMSH, this is achieved by assigning a "physical" group to a set of elementary entities:

# %%
gmsh.initialize()
model = gmsh.model
model.add("physicals")

# Create rectangle
box1 = model.occ.addRectangle(0, 0, 0, 1, 1)
box2 = model.occ.addRectangle(-2, -2, 0, 1, 1)

# Create circle
circle1 = model.occ.addDisk(3, 0, 0, 0.5, 0.5)
circle2 = model.occ.addDisk(2, -2, 0, 0.5, 0.5)

model.occ.synchronize()

# Create physical groups
model.addPhysicalGroup(2, [box1, box2], tag=1, name="boxes")
model.addPhysicalGroup(2, [circle1, circle2], tag=2, name="circles")

# Create the mesh
model.occ.synchronize()
model.mesh.generate(2)
gmsh.write("physicals.msh")
gmsh.finalize()

# %%
mesh = meshio.read("physicals.msh")
plot2D(mesh, ignore_lines=True)

# %% [markdown]
# ## The sharp bits
#
# ### Conflicting entities
# When adding elementary entities, overlaps and interfaces are not "merged" by default: the entities will overlap and will be meshed separately. The resulting sub-meshes will not be connected.
#
#

# %% [markdown]
# ### Keeping track of integers
# Whenever entities are created / transformed (e.g. when healing interfaces), there can be reassignment of the integer tags used to label them. In the official tutorials, entity tags are re-identified based on some characteristic like bounding extent to later assign to physical groups.

# %%
