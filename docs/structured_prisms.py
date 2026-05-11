# %% [markdown]
# # Structured (layered) extruded prisms
#
# A regular `PolyPrism` produces an unstructured 3D mesh: gmsh fills the
# extruded volume with tetrahedra of roughly the same characteristic length
# in all directions. For thin films, stacks, and slabs this is often
# wasteful -- we know the geometry is layered, and we want a swept mesh
# whose vertical resolution is decoupled from the in-plane resolution.
#
# Passing `n_layers=` to `PolyPrism` switches it into **structured mode**
# (gmsh tutorial t3 style): each z-interval declared in `buffers` gets its
# own layer count from `n_layers`, producing a swept layered mesh whose
# interior xy-columns have exactly `n_layers + 1` distinct z-levels.
#
# Compared to gmsh's transfinite meshing, structured `PolyPrism` accepts
# **arbitrary base polygons** (no four-corner topology requirement) and
# can be freely combined with regular `PolyPrism` and `PolySurface`
# entities in the same scene with conformal interfaces.

# %%
from shapely.geometry import Polygon

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism

# %% [markdown]
# ## A single layered slab
#
# The simplest case: one base polygon, one z-interval, one layer count.
# `buffers={z0: 0.0, z1: 0.0}` says "extrude from `z0` to `z1` without
# tapering". `n_layers=[4]` says "use 4 layers along that interval".

# %%
base = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

slab = PolyPrism(
    polygons=base,
    buffers={0.0: 0.0, 1.0: 0.0},
    n_layers=[4],
    physical_name="slab",
)

generate_mesh(
    entities=[slab],
    dim=3,
    output_mesh="structured_slab.msh",
    default_characteristic_length=0.2,
)

# %% [markdown]
# ## Multiple intervals with different layer counts
#
# The number of z-intervals is `len(buffers) - 1`, and `n_layers` must
# have one entry per interval. This is the analog of gmsh tutorial t3's
# `[8, 2]` example: a fine 8-layer region stacked under a coarse 2-layer
# region.
#
# In structured mode `buffers` must declare zero taper (each entry maps
# `z` to `0.0`). Taper is only meaningful in the unstructured mode.

# %%
stack = PolyPrism(
    polygons=base,
    buffers={0.0: 0.0, 0.5: 0.0, 1.0: 0.0},
    n_layers=[8, 2],
    physical_name="film_stack",
)

generate_mesh(
    entities=[stack],
    dim=3,
    output_mesh="film_stack.msh",
    default_characteristic_length=0.2,
)

# %% [markdown]
# ## Mixing structured and unstructured prisms
#
# Structured slabs live in the same `entities` list as regular prisms and
# surfaces. The pipeline guarantees the **interfaces are conformal**:
# the bottom and top faces of a structured slab share nodes with their
# unstructured neighbors, with no duplicate (coincident-but-distinct)
# vertices. This is what makes the structured option safe to drop into
# an existing scene without splitting the mesh.

# %%
hull = Polygon([(-1, -1), (2, -1), (2, 2), (-1, 2)])

substrate = PolyPrism(
    polygons=hull,
    buffers={0.0: 0.0, 0.5: 0.0},
    physical_name="substrate",
    mesh_order=10,
)
film = PolyPrism(
    polygons=base,
    buffers={0.5: 0.0, 1.0: 0.0},
    n_layers=[3],
    physical_name="film",
    mesh_order=2,
)
cladding = PolyPrism(
    polygons=hull,
    buffers={1.0: 0.0, 1.3: 0.0},
    physical_name="cladding",
    mesh_order=10,
)

generate_mesh(
    entities=[substrate, film, cladding],
    dim=3,
    output_mesh="mixed.msh",
    default_characteristic_length=0.4,
)

# %% [markdown]
# ## Hex / wedge elements with `recombine=True`
#
# By default a structured prism produces wedge/prism tets (triangulated
# in-plane, extruded). Passing `recombine=True` asks gmsh to recombine
# the in-plane mesh before extrusion -- on rectangular bases this
# produces hexahedral elements, on arbitrary polygons it yields a mix
# of hexes and wedges.

# %%
hex_slab = PolyPrism(
    polygons=base,
    buffers={0.0: 0.0, 1.0: 0.0},
    n_layers=[4],
    physical_name="hex_slab",
    recombine=True,
)

generate_mesh(
    entities=[hex_slab],
    dim=3,
    output_mesh="hex_slab.msh",
    default_characteristic_length=0.3,
)

# %% [markdown]
# ## Overlapping structured prisms: priority resolution
#
# When two structured prisms have overlapping 3D extents, `mesh_order`
# decides the winner: lower numbers override higher numbers. The
# lower-priority prism is sliced upfront (in xy and z) so that every
# structured volume in the final mesh is disjoint from every other
# structured volume. Layer counts are distributed proportionally
# across the resulting sub-pieces.

# %%
big = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
small = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])

# Compatible layer densities (2 layers/unit) so the cascade produces
# matching n_layers on shared z-intervals after the split.
lo = PolyPrism(
    polygons=big,
    buffers={0.0: 0.0, 3.0: 0.0},
    n_layers=[6],
    physical_name="lo",
    mesh_order=2.0,
)
hi = PolyPrism(
    polygons=small,
    buffers={1.0: 0.0, 2.0: 0.0},
    n_layers=[2],
    physical_name="hi",
    mesh_order=1.0,
)

generate_mesh(
    entities=[lo, hi],
    dim=3,
    output_mesh="overlapping.msh",
    default_characteristic_length=1.0,
)

# %% [markdown]
# ## Constraints
#
# - `buffers` must declare zero taper in structured mode
#   (`{z: 0.0, ...}`).
# - `len(n_layers) == len(buffers) - 1`, all positive integers.
# - Two structured slabs that share a horizontal face must agree on
#   `n_layers` at that face (a `StructuredLayerMismatchError` is raised
#   otherwise).
