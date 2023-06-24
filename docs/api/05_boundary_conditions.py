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

# # Boundary conditions in meshwell
#
# In the last notebook we saw that interfaces between entities and entities, or entities and "nothing", were properly tagged.
#
# This can be seamlessly used to define more complicated boundary conditions, instead of relying on the usual GMSH boundary recovery methods (entity extraction in bounding boxes, or direct dimtag manipulation).
#
# The way this is done is with boundary entities that are added to the CAD model, but not meshed. These are added just like a regular `entities_dict` entry, but under `boundary_dict`:

# + tags=["hide-input"]
from meshwell.model import Model
from collections import OrderedDict
from meshwell.polysurface import PolySurface
from meshwell.gmsh_entity import GMSH_entity
import shapely

# +
model = Model()

# Mesh a rectangle
rectangle_polygon = shapely.Polygon(
    [
        (0, 2, 0),
        (2, 2, 0),
        (2, 0, 0),
        (0, 0, 0),
    ]
)
rectangle = PolySurface(polygons=rectangle_polygon, model=model)

# Create another rectangle for boundary definition
top_line_rectangle = GMSH_entity(
    gmsh_function=model.occ.add_rectangle,
    gmsh_function_kwargs={"x": 0, "y": 2, "z": 0, "dx": 2, "dy": 1},
    dim=2,
    model=model,
)

entities = OrderedDict(
    {
        "rectangle": rectangle,
    }
)

boundary_entities = {"top_line_rectangle": top_line_rectangle}

mesh_out = model.mesh(
    entities_dict=entities,
    boundaries_dict=boundary_entities,
    verbosity=False,
    filename="mesh.msh",
)


# +
# # !gmsh mesh.msh
# -

# The output is as follows:
#
# ![all](media/05_rectangle.png)
# ![all](media/05_topline.png)
# ![all](media/05_otherlines.png)
#
# Note that boundary entity labels override all other physical labels.

#
