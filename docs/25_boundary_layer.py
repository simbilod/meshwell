# %% [markdown]
# # Boundary-layer meshing with BoundaryLayerResolutionSpec
#
# `BoundaryLayerResolutionSpec` attaches a gmsh **BoundaryLayer** field to any
# 1D physical name -- a PolyLine, an interface `a___b`, or a boundary
# `a___None` -- growing an anisotropic, geometrically graded layer off those
# curves. It complements `StructuredSweep` (notebook 24): the sweep stamps an
# exact tensor grid at straight interfaces, while this delegates to gmsh's
# boundary-layer mesher (curved walls, quads, 3D-capable), trading exactness
# for generality.

# %%
import shapely

from meshwell.orchestrator import generate_mesh
from meshwell.polysurface import PolySurface
from meshwell.resolution import BoundaryLayerResolutionSpec
from meshwell.visualization import plot2D

# %% [markdown]
# ## A graded quad boundary layer on a region's outer boundary

# %%
sheet = PolySurface(
    polygons=shapely.box(0, 0, 1, 1), physical_name="sheet", mesh_order=1
)
bl_mesh = generate_mesh(
    entities=[sheet],
    dim=2,
    output_mesh="boundary_layer_quad.msh",
    default_characteristic_length=0.1,
    resolution_specs={
        "sheet___None": [
            BoundaryLayerResolutionSpec(
                size=0.01, thickness=0.08, ratio=1.3, quads=True
            )
        ],
    },
)
n_quad = sum(cb.data.shape[0] for cb in bl_mesh.cells if cb.type == "quad")
print(f"quad cells in the layer: {n_quad}")
plot2D(bl_mesh, title="Quad boundary layer on sheet___None", wireframe=True)

# %% [markdown]
# ## Triangle layer (quads=False)

# %%
tri_mesh = generate_mesh(
    entities=[
        PolySurface(
            polygons=shapely.box(0, 0, 1, 1), physical_name="sheet", mesh_order=1
        )
    ],
    dim=2,
    output_mesh="boundary_layer_tri.msh",
    default_characteristic_length=0.1,
    resolution_specs={
        "sheet___None": [
            BoundaryLayerResolutionSpec(
                size=0.01, thickness=0.08, ratio=1.3, quads=False
            )
        ],
    },
)
plot2D(tri_mesh, title="Triangle boundary layer", wireframe=True)

# %% [markdown]
# ## Two independent boundary layers with different parameters
#
# Each BoundaryLayerResolutionSpec becomes its own gmsh boundary-layer field,
# so different curves can carry different first-layer sizes / thicknesses.

# %%
two = generate_mesh(
    entities=[
        PolySurface(
            polygons=shapely.box(0, 0, 4, 1), physical_name="lower", mesh_order=2
        ),
        PolySurface(
            polygons=shapely.box(0, 1, 4, 2), physical_name="upper", mesh_order=1
        ),
    ],
    dim=2,
    output_mesh="boundary_layer_two.msh",
    default_characteristic_length=0.2,
    resolution_specs={
        "lower___None": [BoundaryLayerResolutionSpec(size=0.005, thickness=0.05)],
        "upper___None": [BoundaryLayerResolutionSpec(size=0.02, thickness=0.1)],
    },
)
plot2D(two, title="Two boundary layers, different sizes", wireframe=True)

# %% [markdown]
# ## Naming the convex corner where a boundary layer wraps
#
# A boundary layer growing off `sheet___None` has to negotiate the sharp convex
# corners of the box. You can name such a corner with a 0D `OCC_entity` (see
# notebook 10): build a vertex, set `dimension=0` and a `physical_name`. The
# named vertex survives fragmentation and becomes a named mesh node, so it can
# carry its own resolution spec -- here `ConstantInField` refines the mesh
# locally at the corner while the boundary layer grows off the walls. This
# named-point recipe is also the building block for the planned *fan points*
# feature (wrapping graded layers around the corner).

# %%
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCP.gp import gp_Pnt

from meshwell.occ_entity import OCC_entity
from meshwell.resolution import ConstantInField

corner = OCC_entity(
    occ_function=lambda: BRepBuilderAPI_MakeVertex(gp_Pnt(1.0, 1.0, 0.0)).Vertex(),
    physical_name="corner",
    dimension=0,
)
corner_mesh = generate_mesh(
    entities=[
        PolySurface(polygons=shapely.box(0, 0, 1, 1), physical_name="sheet", mesh_order=1),
        corner,
    ],
    dim=2,
    output_mesh="boundary_layer_corner.msh",
    default_characteristic_length=0.2,
    resolution_specs={
        "sheet___None": [
            BoundaryLayerResolutionSpec(size=0.01, thickness=0.08, ratio=1.3)
        ],
        "corner": [ConstantInField(apply_to="points", resolution=0.02)],
    },
)
print("named corner is a mesh node:", "corner" in corner_mesh.cell_sets)
plot2D(corner_mesh, title="Boundary layer + refined named corner (1, 1)", wireframe=True)

# %% [markdown]
# ## Notes and limitations
#
# - Full gmsh passthrough: `size`, `thickness`, `ratio`, `quads`, and the
#   optional `size_far`, `nb_layers`, `intersect_metrics`, `aniso_max`, `beta`.
# - `quads=True` emits quadrilateral layers; to split them into triangles
#   downstream, call `gmsh.model.mesh.splitQuadrangles()`.
# - A boundary layer and a `StructuredSweep` cannot share a model (the sweep
#   sets `Mesh.MeshOnlyEmpty=1`, which suppresses boundary-layer generation) --
#   meshwell raises a clear error if both are requested.
# - Fan points (wrapping a layer around a sharp convex vertex) are a planned
#   follow-up; the named 0D point they build on is shown above and in
#   notebook 10.
