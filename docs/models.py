# %% [markdown]
# # Multi-entity models
# Multiple GMSH entities (and polysurfaces or prisms) can be provided to a model to create a single mesh.

# %%
import shapely
import gmsh
from functools import partial
from meshwell.polysurface import PolySurface
from meshwell.gmsh_entity import GMSH_entity
from meshwell.visualization import plot2D
from meshwell.cad import cad
from meshwell.mesh import mesh

# %%
polygon_hull = shapely.Polygon(
    [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
)
line1 = shapely.LineString([[0.5, 0.5], [1.5, 1.5]])
polygon_hole1 = shapely.buffer(line1, 0.2)
line2 = shapely.LineString([[1.5, 0.5], [0.5, 1.5]])
polygon_hole2 = shapely.buffer(line2, 0.2)
polygon = polygon_hull - polygon_hole1 - polygon_hole2


s_curve = shapely.LineString([[-1, 0], [0, 0.5], [-1, 1], [0, 1.5], [-1, 2]])
s_shape = shapely.buffer(s_curve, 0.2)

poly2D = PolySurface(
    polygons=polygon,
    physical_name="meshwell_polygon",
    mesh_order=1,
)

s = PolySurface(
    polygons=s_shape,
    physical_name="meshwell_s",
    mesh_order=2,
)

gmsh_entity = GMSH_entity(
    gmsh_partial_function=partial(
        gmsh.model.occ.add_disk, xc=2, yc=2, zc=0, rx=1, ry=1
    ),
    physical_name="gmsh_disk",
    mesh_order=3,
)

rectangle = GMSH_entity(
    gmsh_partial_function=partial(
        gmsh.model.occ.add_rectangle, x=1.5, y=0, z=0, dx=1, dy=1
    ),
    physical_name="gmsh_rectangle",
    mesh_order=4,
)

entities_list = [poly2D, s, gmsh_entity, rectangle]

cad(
    entities_list=entities_list,
    output_file="complicated.xao",
)

output_mesh = mesh(
    dim=2,
    input_file="complicated.xao",
    output_file="complicated.msh",
    default_characteristic_length=0.5,
    mesh_element_order=1,  # set to 2 to generate a curved mesh with the disk
)

# %%
plot2D(output_mesh, wireframe=False)
# %%
plot2D(output_mesh, wireframe=False, physicals=["meshwell_polygon___gmsh_disk"])
# %% [markdown]
# mesh_order specifies which entity takes precedence if there is a conflict; lower numbers override higher numbers.

# %%
poly2D = PolySurface(
    polygons=polygon,
    physical_name="meshwell_polygon",
    mesh_order=4,
)

s = PolySurface(
    polygons=s_shape,
    physical_name="meshwell_s",
    mesh_order=3,
)

gmsh_entity = GMSH_entity(
    gmsh_partial_function=partial(
        gmsh.model.occ.add_disk, xc=2, yc=2, zc=0, rx=1, ry=1
    ),
    physical_name="gmsh_disk",
    mesh_order=2,
)

rectangle = GMSH_entity(
    gmsh_partial_function=partial(
        gmsh.model.occ.add_rectangle, x=1.5, y=0, z=0, dx=1, dy=1
    ),
    physical_name="gmsh_rectangle",
    mesh_order=1,
)

entities_list = [poly2D, s, gmsh_entity, rectangle]

cad(
    entities_list=entities_list,
    output_file="model.xao",
)

output_mesh = mesh(
    dim=2,
    input_file="model.xao",
    output_file="model.msh",
    default_characteristic_length=0.5,
    mesh_element_order=1,  # set to 2 to generate a curved mesh with the disk
)

# %%
plot2D(output_mesh, wireframe=True)


# %% [markdown]
# By default, all CAD entities get meshed. By setting meshbool to False, a CAD entity can be inserted for the purposes of cutting out regions / tagging interfaces, without adding a mesh within a region.


# %%
poly2D = PolySurface(
    polygons=polygon,
    physical_name="meshwell_polygon",
    mesh_order=4,
)

s = PolySurface(
    polygons=s_shape,
    physical_name="meshwell_s",
    mesh_order=3,
    mesh_bool=False,  # Don't mesh this entity
)

gmsh_entity = GMSH_entity(
    gmsh_partial_function=partial(
        gmsh.model.occ.add_disk, xc=2, yc=2, zc=0, rx=1, ry=1
    ),
    physical_name="gmsh_disk",
    mesh_order=2,
    mesh_bool=False,  # Don't mesh this entity
)

rectangle = GMSH_entity(
    gmsh_partial_function=partial(
        gmsh.model.occ.add_rectangle, x=1.5, y=0, z=0, dx=1, dy=1
    ),
    physical_name="gmsh_rectangle",
    mesh_order=1,
)

entities_list = [poly2D, s, gmsh_entity, rectangle]

cad(
    entities_list=entities_list,
    output_file="model.xao",
)

output_mesh = mesh(
    dim=2,
    input_file="model.xao",
    output_file="model.msh",
    default_characteristic_length=0.5,
    mesh_element_order=1,  # set to 2 to generate a curved mesh with the disk
)

# %%
plot2D(output_mesh, wireframe=True)
