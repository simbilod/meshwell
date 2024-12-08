# %% [markdown]
# # Specifying resolutions
# When defining entities, ResolutionSpecs can be attached to control the mesh size within or near the entity and its boundaries.

# %%
import shapely
from collections import OrderedDict

from meshwell.model import Model
from meshwell.polysurface import PolySurface
from meshwell.visualization import plot2D
from meshwell.resolution import ConstantInField, ThresholdField, ExponentialField

# %% [markdown]
# ## Default characteristic length
# When creating a Model mesh, default_characteristic_length is used to set the default mesh size.

# %%
box1 = shapely.box(0, 0, 10, 10)
box2 = shapely.box(2, 2, 4, 4)
box3 = shapely.box(8, 4, 12, 6)

boxes = OrderedDict()
boxes["box3"] = box3
boxes["box2"] = box2
boxes["box1"] = box1


# %%

for default_characteristic_length in [10, 1, 0.5]:
    model = Model(n_threads=1)

    polysurfaces = []
    for i, (box_name, box) in enumerate(boxes.items()):
        polysurfaces.append(
            PolySurface(
                polygons=box,
                model=model,
                physical_name=box_name,
                mesh_order=i,
            )
        )

    mesh = model.mesh(
        entities_list=polysurfaces,
        filename="default_resolution.msh",
        default_characteristic_length=default_characteristic_length,
    )

    plot2D(
        mesh,
        title=f"default_characteristic_length = {default_characteristic_length}",
        wireframe=True,
    )

# %% [markdown]
# ## Constant Resolution
# Constant resolution can be assigned within entities or their boundaries using ConstantInField specification.

# %%
model = Model(n_threads=1)

polysurface1 = PolySurface(
    polygons=box1,
    model=model,
    physical_name="box1",
    mesh_order=3,
)

polysurface2 = PolySurface(
    polygons=box2,
    model=model,
    physical_name="box2",
    mesh_order=2,
    resolutions=[ConstantInField(resolution=0.3, apply_to="surfaces")],
)

polysurface3 = PolySurface(
    polygons=box3,
    model=model,
    physical_name="box3",
    mesh_order=1,
    resolutions=[ConstantInField(resolution=0.3, apply_to="curves")],
)

mesh = model.mesh(
    entities_list=[polysurface1, polysurface2, polysurface3],
    filename="constant_resolution.msh",
    default_characteristic_length=2,
)

plot2D(mesh, title="ConstantInField", wireframe=True)

# %% [markdown]
# ## Threshold-based Resolution
# This creates a mesh that transitions from fine resolution near boundaries to coarse resolution away from them

# %%
model = Model(n_threads=1)

polysurface1 = PolySurface(
    polygons=box1,
    model=model,
    physical_name="box1",
    mesh_order=3,
)

polysurface2 = PolySurface(
    polygons=box2,
    model=model,
    physical_name="box2",
    mesh_order=2,
    resolutions=[
        ThresholdField(sizemin=0.3, distmax=5, sizemax=2.0, apply_to="curves"),
        ConstantInField(resolution=0.3, apply_to="surfaces"),
    ],
)

polysurface3 = PolySurface(
    polygons=box3,
    model=model,
    physical_name="box3",
    mesh_order=1,
    resolutions=[
        ThresholdField(sizemin=0.1, distmax=2, sizemax=0.5, apply_to="curves"),
        ThresholdField(sizemin=0.5, distmax=5, sizemax=2, distmin=2, apply_to="curves"),
    ],
)

mesh = model.mesh(
    entities_list=[polysurface1, polysurface2, polysurface3],
    filename="threshold_resolution.msh",
    default_characteristic_length=2,
)

plot2D(mesh, title="ThresholdField", wireframe=True)

# %% [markdown]
# ## Exponential Resolution
# Creates a mesh with exponentially varying resolution based on distance from boundaries

# %%
model = Model(n_threads=1)

polysurface1 = PolySurface(
    polygons=box1,
    model=model,
    physical_name="box1",
    mesh_order=3,
)

polysurface2 = PolySurface(
    polygons=box2,
    model=model,
    physical_name="box2",
    mesh_order=2,
    resolutions=[
        ExponentialField(
            sizemin=0.3, lengthscale=2, growth_factor=2.0, apply_to="curves"
        ),
        ConstantInField(resolution=0.3, apply_to="surfaces"),
    ],
)

polysurface3 = PolySurface(
    polygons=box3,
    model=model,
    physical_name="box3",
    mesh_order=1,
    resolutions=[
        ExponentialField(
            sizemin=0.2, lengthscale=2, growth_factor=2.0, apply_to="curves"
        ),
    ],
)

mesh = model.mesh(
    entities_list=[polysurface1, polysurface2, polysurface3],
    filename="threshold_resolution.msh",
    default_characteristic_length=2,
)

plot2D(mesh, wireframe=True)


# ## Background mesh
# It is also possible to pass a gmsh .pos file to fully parametrize the size field vs position. This is typically more useful once a physical solution has been calculated on an initial mesh, and a new size field has been calculated from the solution. Many solvers also have built-in remeshing, and hence only need an initial mesh before doing their own refinement.

# %%
