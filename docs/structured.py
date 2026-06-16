# %% [markdown]
# # Structured meshing
#
# By default meshwell fills every 3D volume with **tetrahedra**: each
# `PolyPrism` is fragmented by OpenCASCADE boolean operations (BOP) and then
# handed to gmsh's unstructured tet mesher (see [Prisms](prisms)).
#
# A `PolyPrism` can instead be meshed with **wedge** (triangular-prism)
# elements by setting `structured=True`. The XY footprint is triangulated once
# and that triangulation is *extruded* into `n_layers` stacked wedges. This is
# the natural choice for layered geometries — waveguides, thin films, doped
# slabs — where you want a controlled number of elements through the thickness
# and clean, axis-aligned layers rather than an isotropic tet fill.
#
# Crucially, structured and unstructured solids coexist in a single mesh: the
# wedge-filled cohort and the surrounding tet-filled cladding share
# **conformal** interfaces (matching nodes on the shared faces), so the result
# is a single watertight mesh.

# %% [markdown]
# ## Opting in
#
# Structured meshing is controlled by three things:
#
# 1. **`structured=True`** on the `PolyPrism`.
# 2. The prism must be a *pure extrusion* (`extrude=True`), i.e. **all buffers
#    are zero** so the XY footprint is constant in z. A z-varying (tapered)
#    buffer currently cannot be wedge-meshed and raises `StructuredExtrudeRequiredError`.
#    Setting `structured=True` flips `extrude` on automatically when the buffers
#    are all zero.
# 3. The number of layers through the thickness, supplied per physical name via
#    a `StructuredExtrusionResolutionSpec(n_layers=N)` in `resolution_specs`.
#    This allows changing the z-aligned resolution independent of CAD definition.
#    Without it the prism defaults to a single layer.

# %% [markdown]
# ## A worked example
#
# A square `core` is meshed with wedges and embedded between an unstructured
# `substrate` (below) and `cladding` (above), both of which extend laterally
# past the core. The core gets three layers through its thickness.

# %%
import shapely

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.resolution import StructuredExtrusionResolutionSpec
from meshwell.visualization import plot3D

# Structured core: a 4x4 slab, z in [0, 1], wedge-meshed.
core = PolyPrism(
    polygons=shapely.box(0, 0, 4, 4),
    buffers={0.0: 0.0, 1.0: 0.0},  # all-zero buffers => pure extrusion
    physical_name="core",
    structured=True,
)

# Unstructured substrate below the core, z in [-1, 0].
substrate = PolyPrism(
    polygons=shapely.box(-2, -2, 6, 6),
    buffers={-1.0: 0.0, 0.0: 0.0},
    physical_name="substrate",
)

# Unstructured cladding above the core, z in [1, 2].
cladding = PolyPrism(
    polygons=shapely.box(-2, -2, 6, 6),
    buffers={1.0: 0.0, 2.0: 0.0},
    physical_name="cladding",
)

mesh_obj = generate_mesh(
    entities=[core, substrate, cladding],
    dim=3,
    output_mesh="structured.msh",
    default_characteristic_length=1.0,
    resolution_specs={
        # Three wedge layers through the core's thickness.
        "core": [StructuredExtrusionResolutionSpec(n_layers=3)],
    },
)

# %% [markdown]
# The core is filled with wedges, while the substrate and cladding are filled
# with tetrahedra. The shared faces appear as conformal interface physical
# groups (`core___substrate`, `core___cladding`).

# %%
wedges = sum(cb.data.shape[0] for cb in mesh_obj.cells if cb.type == "wedge")
tets = sum(cb.data.shape[0] for cb in mesh_obj.cells if cb.type == "tetra")
print(f"wedges (structured core): {wedges}")
print(f"tetrahedra (unstructured cladding/substrate): {tets}")
print("interface groups:", [k for k in mesh_obj.cell_sets if "___" in k])

# %%
plot3D(mesh_obj, title="Structured core (wedges) in unstructured cladding (tets)")

# %% [markdown]
# ## How it works (overview)
#
# `generate_mesh` runs one pipeline for both solid types; the structured path
# splices extra passes around the shared CAD/BOP and meshing stages. At a high
# level:
#
# - **Structured pre-pass.** Structured prisms are collected and grouped into
#   *cohorts* of touching slabs. Each cohort's geometry is decomposed per
#   z-interval and rebuilt as a single OCC compound. For performance reasons,
#   the internal interfaces are conformal *by construction* instead of defined
#   as deparated solids and fused through BOP fragmentation.
# - **CAD / BOP.** The cohort enters the boolean stage as one entity, so BOP
#   fragments it against the unstructured neighbours (recovering the
#   `core___substrate` / `core___cladding` interfaces) without dissolving its
#   pre-built internal structure.
# - **Structured post-pass.** The cohort compound is expanded back into
#   per-sub-solid physical groups, and a check confirms BOP did not subdivide
#   any pre-baked cohort face -- this would introduce interfaces not captured by
#   the structured mesh.
# - **Meshing hooks.** Before gmsh's 2D pass the cohort's vertical edges are set
#   transfinite (`n_layers + 1` nodes) and its lateral faces are quad-meshed;
#   gmsh then triangulates the bottom faces; before the 3D pass that bottom
#   triangulation is copied upward layer-by-layer to stamp the wedges (while the
#   GMSH OCC kernel supports extruding a previously-meshed 2D surface, this is
#   incompatible with boolean fragments, hence manual stamping). gmsh's
#   tet mesher then fills the surrounding unstructured volumes, and a final
#   node-dedup pass welds the conformal z-interfaces.
#
# The contrast in one line: **unstructured** relies on BOP fragmenting plus
# gmsh's tet mesher; **structured** builds conformal geometry up front (shared
# OCC topology) and hand-stamps wedge elements via gmsh hooks, while still
# letting BOP and gmsh handle the unstructured surroundings.

# %%
