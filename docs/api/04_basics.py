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

# # Basics
#
# The main value of this package is the automatic tagging of complex combinations of GMSH physical entities, which allows areas of the mesh to be easily accessed for later simulation definition.

# + tags=["hide-input"]
from meshwell.model import Model
from meshwell.gmsh_entity import GMSH_entity

# +
model = Model()
# model.mesh?
# -

# The keys of the ordered dictionary `entities_dict` are associated to the corresponding values (list of entities). The values are input as a list of meshwell entities (PolySurfaces, Prisms, or GMSH_entities). The dimension of the returned mesh is set by the maximum entity dimension across all entries.
#
# The interfaces between different entries are tagged as `{entity1_key}{interface_delimiter}{entity2_key}`, defaulting to `___`. The interface between entities and the mesh boundaries are `{entity_key}{interface_delimiter}{boundary_delimiter}`, defaulting to `None`

# Seeing this in action:

# +
model = Model()

box1 = GMSH_entity(
    gmsh_function=model.occ.addBox,
    gmsh_function_kwargs={"x": 0, "y": 0, "z": 0, "dx": 2, "dy": 2, "dz": 2},
    dimension=3,
    model=model,
    physical_name="box1",
    mesh_order=1,
)

box2 = GMSH_entity(
    gmsh_function=model.occ.addBox,
    gmsh_function_kwargs={"x": 1, "y": 1, "z": 1, "dx": 2, "dy": 2, "dz": 2},
    dimension=3,
    model=model,
    physical_name="box2",
    mesh_order=2,
)

entities = [box1, box2]

mesh_out = model.mesh(entities_list=entities, verbosity=True, filename="mesh.msh")
# -


# Uncomment below to dynamically inspect the mesh.

# +
# # !gmsh mesh.msh
# -

# You should see the following (toggling 3D element edges and 3D element faces):
#
# ![all](media/04_all.png)
#
# - `box1` appears earlier in the OrderedDict of entities, and hence will take precedence where it meets other entities:
#
# ![all](media/04_box1.png)
# ![all](media/04_box2.png)
#
# - Toggling 2D element edges and 2D element faces, the interface between `box1` and `box2` is rendered:
#
# ![all](media/04_box1___box2.png)
#
# - Interfaces with "nothing" (no other entities) is also rendered:
#
# ![all](media/04_box1___None.png)
