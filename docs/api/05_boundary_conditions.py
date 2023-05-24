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
# The way this is done is with boundary entities that are added to the CAD model, but not meshed. These are added just like a regular `dimtag_dict` entry, but under `boundary_tags`:

# + tags=["hide-input"]
from meshwell.model import Model
from collections import OrderedDict
from meshwell.polysurface import PolySurface
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
top_line_rectangle = model.occ.add_rectangle(0, 2, 0, 2, 1)

entities = OrderedDict(
    {
        "rectangle": [(2, rectangle)],
    }
)

boundary_entities = {"top_line_rectangle": [(2, top_line_rectangle)]}

mesh_out = model.mesh(
    dimtags_dict=entities,
    boundary_tags=boundary_entities,
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
