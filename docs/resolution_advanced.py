# %% [markdown]
# # Targeting resolutions
# ResolutionSpecs have many attributes that allow them to be targeted.

# %%
import shapely
from collections import OrderedDict

from meshwell.model import Model
from meshwell.polysurface import PolySurface
from meshwell.visualization import plot2D
from meshwell.resolution import ConstantInField, ThresholdField

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


# %% [markdown]
# ## Filtering by mass
# ResolutionSpecs can be selectively applied given the "mass" of the entity (total lengths of curves, total area of surfaces, total volumes of volumes).

# %%

# Create two boxes of different sizes
small_box = shapely.box(0, 0, 1, 1)
large_box = shapely.box(3, 0, 5, 2)

# Combine into multipolygon
multi = shapely.MultiPolygon([small_box, large_box])

model = Model()

# Create polysurface with filtered resolution specs
polysurface = PolySurface(
    polygons=multi,
    model=model,
    physical_name="filtered_boxes",
    resolutions=[
        # Fine resolution for small curves
        ConstantInField(
            resolution=0.1,
            apply_to="surfaces",
        ),
    ],
)

mesh = model.mesh(
    entities_list=[polysurface],
    filename="unfiltered_resolution.msh",
    default_characteristic_length=0.5,
)

plot2D(mesh, wireframe=True, title="Reference Resolution Example")

model = Model()

# Create polysurface with filtered resolution specs
polysurface = PolySurface(
    polygons=multi,
    model=model,
    physical_name="filtered_boxes",
    resolutions=[
        # Fine resolution for small curves
        ConstantInField(
            resolution=0.1,
            apply_to="surfaces",
            max_mass=4,  # Only applies to curves with perimeter < 4
        ),
    ],
)

mesh = model.mesh(
    entities_list=[polysurface],
    filename="filtered_resolution.msh",
    default_characteristic_length=0.5,
)

plot2D(mesh, wireframe=True, title="Mass-filtered Resolution Example")


# %% [markdown]
# ## Filtering the application by sharing/not sharing
# For surfaces/curves/points, the ResolutionSpec can be applied if the entity is touching (or not touching) another entity.


# %% [markdown]
# ## Restricting the evaluation of the field
# Once applied, a field can be restricted in its evaluation to specific entities using the `restrict_to` parameter.
# This allows you to control which entities the resolution field affects, even when the field is defined on one entity.

# %%
# Create geometry: outer square with two inner squares
large_rect = 20
small_rect = 5

# Create outer square
outer_square = shapely.Polygon(
    [[0, 0], [large_rect, 0], [large_rect, large_rect], [0, large_rect], [0, 0]],
)

# Create smaller square that will be duplicated
inner_square = shapely.Polygon(
    [
        [large_rect / 2 - small_rect / 2, large_rect / 2 - small_rect / 2],
        [large_rect / 2 + small_rect / 2, large_rect / 2 - small_rect / 2],
        [large_rect / 2 + small_rect / 2, large_rect / 2 + small_rect / 2],
        [large_rect / 2 - small_rect / 2, large_rect / 2 + small_rect / 2],
        [large_rect / 2 - small_rect / 2, large_rect / 2 - small_rect / 2],
    ],
)

# Example 1: Field restricted to inner_left only
model = Model(n_threads=1)
poly_outer = PolySurface(
    polygons=outer_square,
    model=model,
    mesh_order=2,
    physical_name="outer",
)

poly_left = PolySurface(
    polygons=shapely.affinity.translate(inner_square, xoff=-3.1),
    model=model,
    mesh_order=1,
    physical_name="inner_left",
    resolutions=[
        ThresholdField(
            sizemin=0.3,
            distmax=3,
            sizemax=1.5,
            apply_to="curves",
            restrict_to=["inner_left"],  # Field only affects inner_left
        )
    ],
)

poly_right = PolySurface(
    polygons=shapely.affinity.translate(inner_square, xoff=3.1),
    model=model,
    mesh_order=1,
    physical_name="inner_right",
)

mesh1 = model.mesh(
    entities_list=[poly_outer, poly_left, poly_right],
    default_characteristic_length=1,
    filename="restricted_field_inner_only.msh",
)

plot2D(mesh1, wireframe=True, title="Field Restricted to Inner Left Only")

# Example 2: Field restricted to both inner_left and outer
model = Model(n_threads=1)
poly_outer = PolySurface(
    polygons=outer_square,
    model=model,
    mesh_order=2,
    physical_name="outer",
)

poly_left = PolySurface(
    polygons=shapely.affinity.translate(inner_square, xoff=-3.1),
    model=model,
    mesh_order=1,
    physical_name="inner_left",
    resolutions=[
        ThresholdField(
            sizemin=0.3,
            distmax=3,
            sizemax=1.5,
            apply_to="curves",
            restrict_to=[
                "inner_left",
                "outer",
            ],  # Field affects both inner_left and outer
        )
    ],
)

poly_right = PolySurface(
    polygons=shapely.affinity.translate(inner_square, xoff=3.1),
    model=model,
    mesh_order=1,
    physical_name="inner_right",
)

mesh2 = model.mesh(
    entities_list=[poly_outer, poly_left, poly_right],
    default_characteristic_length=1,
    filename="restricted_field_inner_and_outer.msh",
)

plot2D(mesh2, wireframe=True, title="Field Restricted to Inner Left and Outer")

# %% [markdown]
# In the first example, the threshold field only affects the inner_left square, resulting in refined mesh elements only around that entity.
# In the second example, the field affects both the inner_left square and the outer square, causing mesh refinement to propagate through
# the outer square as well. The inner_right square maintains the default mesh size in both cases since it's not included in the restrict_to list.
