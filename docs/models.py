# %% [markdown]
# # Multi-entity models
# Multiple GMSH entities (and polysurfaces or prisms) can be provided to a model to create a single mesh.

# %%
import shapely
from meshwell.model import Model
from meshwell.polysurface import PolySurface
from meshwell.gmsh_entity import GMSH_entity
from meshwell.visualization import plot2D

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


model = Model(n_threads=1)

poly2D = PolySurface(
    polygons=polygon,
    model=model,
    physical_name="meshwell_polygon",
    mesh_order=1,
)

s = PolySurface(
    polygons=s_shape,
    model=model,
    physical_name="meshwell_s",
    mesh_order=2,
)

gmsh_entity = GMSH_entity(
    gmsh_function=model.occ.add_disk,
    gmsh_function_kwargs={"xc": 2, "yc": 2, "zc": 0, "rx": 1, "ry": 1},
    dimension=2,
    model=model,
    physical_name="gmsh_disk",
    mesh_order=3,
)

rectangle = GMSH_entity(
    gmsh_function=model.occ.add_rectangle,
    gmsh_function_kwargs={"x": 1.5, "y": 0, "z": 0, "dx": 1, "dy": 1},
    dimension=2,
    model=model,
    physical_name="gmsh_rectangle",
    mesh_order=4,
)

entities_list = [poly2D, s, gmsh_entity, rectangle]

mesh = model.mesh(
    entities_list=entities_list,
    filename="model.msh",
)

# %%
plot2D(mesh, wireframe=False)
# %%
plot2D(mesh, wireframe=False, physicals=["meshwell_polygon___gmsh_disk"])
# %% [markdown]
# mesh_order specifies which entity takes precedence if there is a conflict; lower numbers override higher numbers.

# %%
model = Model(n_threads=1)

poly2D = PolySurface(
    polygons=polygon,
    model=model,
    physical_name="meshwell_polygon",
    mesh_order=4,
)

s = PolySurface(
    polygons=s_shape,
    model=model,
    physical_name="meshwell_s",
    mesh_order=3,
)

gmsh_entity = GMSH_entity(
    gmsh_function=model.occ.add_disk,
    gmsh_function_kwargs={"xc": 2, "yc": 2, "zc": 0, "rx": 1, "ry": 1},
    dimension=2,
    model=model,
    physical_name="gmsh_disk",
    mesh_order=2,
)

rectangle = GMSH_entity(
    gmsh_function=model.occ.add_rectangle,
    gmsh_function_kwargs={"x": 1.5, "y": 0, "z": 0, "dx": 1, "dy": 1},
    dimension=2,
    model=model,
    physical_name="gmsh_rectangle",
    mesh_order=1,
)

entities_list = [poly2D, s, gmsh_entity, rectangle]

mesh = model.mesh(
    entities_list=entities_list,
    filename="model.msh",
)

# %%
plot2D(mesh, wireframe=True)


# %% [markdown]
# By default, all CAD entities get meshed. By setting meshbool to False, a CAD entity can be inserted for the purposes of cutting out regions / tagging interfaces, without adding a mesh within a region.


# %%
model = Model(n_threads=1)

poly2D = PolySurface(
    polygons=polygon,
    model=model,
    physical_name="meshwell_polygon",
    mesh_order=4,
)

s = PolySurface(
    polygons=s_shape,
    model=model,
    physical_name="meshwell_s",
    mesh_order=3,
    mesh_bool=False,  # Don't mesh this entity
)

gmsh_entity = GMSH_entity(
    gmsh_function=model.occ.add_disk,
    gmsh_function_kwargs={"xc": 2, "yc": 2, "zc": 0, "rx": 1, "ry": 1},
    dimension=2,
    model=model,
    physical_name="gmsh_disk",
    mesh_order=2,
    mesh_bool=False,  # Don't mesh this entity
)

rectangle = GMSH_entity(
    gmsh_function=model.occ.add_rectangle,
    gmsh_function_kwargs={"x": 1.5, "y": 0, "z": 0, "dx": 1, "dy": 1},
    dimension=2,
    model=model,
    physical_name="gmsh_rectangle",
    mesh_order=1,
)

entities_list = [poly2D, s, gmsh_entity, rectangle]

mesh = model.mesh(
    entities_list=entities_list,
    filename="model.msh",
)

# %%
plot2D(mesh, wireframe=True)


# %% [markdown]
# On the other hand, the "additive" flag will add the entity, but instead of overriding other entities by mesh order, it will add its own physical_name to the regular entities:

# %%
model = Model(n_threads=1)

poly2D = PolySurface(
    polygons=polygon,
    model=model,
    physical_name="meshwell_polygon",
    mesh_order=4,
)

s = PolySurface(
    polygons=s_shape,
    model=model,
    physical_name="meshwell_s",
    mesh_order=3,
)

gmsh_entity = GMSH_entity(
    gmsh_function=model.occ.add_disk,
    gmsh_function_kwargs={"xc": 2, "yc": 2, "zc": 0, "rx": 1, "ry": 1},
    dimension=2,
    model=model,
    physical_name="gmsh_disk",
    mesh_order=2,
)

rectangle = GMSH_entity(
    gmsh_function=model.occ.add_rectangle,
    gmsh_function_kwargs={"x": 1.5, "y": 0, "z": 0, "dx": 1, "dy": 1},
    dimension=2,
    model=model,
    physical_name="gmsh_rectangle",
    mesh_order=1,
    additive=True,
)

entities_list = [poly2D, s, gmsh_entity, rectangle]

mesh = model.mesh(
    entities_list=entities_list,
    filename="model.msh",
)

# %%
plot2D(mesh, wireframe=False, physicals=["meshwell_polygon"])

# %%
plot2D(mesh, wireframe=False, physicals=["gmsh_rectangle"])
# %%
