# %% [markdown]
# # Adaptive refinement of structured sweeps
#
# [Notebook 24](24_structured_sweeps) introduced structured sweep bands;
# [notebooks 30](30_adaptive_remeshing_gmsh)/[31](31_adaptive_remeshing_mmg)
# closed the solve → size map → remesh loop for unstructured meshes. This
# notebook closes the same loop for models containing structured bands:
# `remesh_structured` adapts every `ResolutionSpec` — for a band, that means
# re-deriving its tangential/normal arrays from the size map (the geometry
# `.xao` never changes), so the band stays an exact tensor grid through
# every iteration.

# %%
import matplotlib.pyplot as plt
import numpy as np
import shapely

from meshwell.orchestrator import generate_mesh
from meshwell.polysurface import PolySurface
from meshwell.remesh import remesh_structured
from meshwell.resolution import StructuredSweepResolutionSpec
from meshwell.structured.sweep import StructuredSweep
from meshwell.visualization import plot2D

# %% [markdown]
# ## Model: two-layer stack with a quantum-well band
#
# Same fixture as notebook 24: two boxes meeting at `y=1`, a `"qw"` band of
# thickness 0.4 grown into `upper`. We checkpoint the CAD to a `.xao` — the
# adaptive loop regenerates from it on every iteration.

# %%
sweep = StructuredSweep(name="qw", on="lower___upper", thickness={"upper": 0.4})
specs = {
    "qw": [StructuredSweepResolutionSpec(tangential=1.0, normal={"upper": 4})],
}
mesh0 = generate_mesh(
    entities=[
        PolySurface(polygons=shapely.box(0, 0, 4, 1), physical_name="lower", mesh_order=2),
        PolySurface(polygons=shapely.box(0, 1, 4, 2), physical_name="upper", mesh_order=1),
    ],
    sweeps=[sweep],
    dim=2,
    output_mesh="adaptive_structured_0.msh",
    checkpoint_cad="adaptive_structured.xao",
    default_characteristic_length=0.5,
    resolution_specs=specs,
)
plot2D(mesh0, title="Iteration 0", wireframe=True)

# %% [markdown]
# ## Signal: an analytic boundary-layer "solution"
#
# The signal contract is unchanged from `remesh.py`: the solver supplies
# `(x, y, z, value)` data (or a ready size map). Here we stand in for the
# solver with an analytic field u = tanh((y − 1)/0.1) — a sharp transition
# at the interface, as a depletion region or thermal boundary layer would
# produce — and turn its gradient into a target size.


# %%
def size_map_from_solution():
    xs, ys = np.meshgrid(
        np.linspace(0.0, 4.0, 81), np.linspace(0.0, 2.0, 81), indexing="ij"
    )
    grad = 10.0 / np.cosh((ys - 1.0) / 0.1) ** 2  # |du/dy|
    sizes = np.clip(0.5 / (1.0 + grad), 0.02, 0.5)
    return np.column_stack(
        [xs.ravel(), ys.ravel(), np.zeros(xs.size), sizes.ravel()]
    )


# %% [markdown]
# ## The loop: solve → size map → `remesh_structured`
#
# Three lines per iteration. The driver damps per-iteration size changes
# (`change_max`) and gradation-limits both the band arrays and the point
# cloud (`max_ratio`), so the loop converges without caller-side hygiene.

# %%
mesh_i, specs_i = mesh0, specs
node_counts = [len(mesh0.points)]
for it in range(1, 3):
    mesh_i, specs_i = remesh_structured(
        input_mesh=mesh_i,
        geometry_file="adaptive_structured.xao",
        sweeps=[sweep],
        resolution_specs=specs_i,
        size_map=size_map_from_solution(),
        output_mesh=f"adaptive_structured_{it}.msh",
        default_characteristic_length=0.5,
    )
    node_counts.append(len(mesh_i.points))
print("node counts per iteration:", node_counts)

# %%
plot2D(mesh_i, title="Iteration 2: band refined toward the interface", wireframe=True)

# %% [markdown]
# The adapted spec is inspectable state — explicit arrays, not hidden mesh:

# %%
off = np.asarray(specs_i["qw"][0].normal["upper"])
print("normal offsets:", np.round(off, 4))
plt.figure(figsize=(5, 3))
plt.step(off[:-1], np.diff(off), where="post")
plt.xlabel("η (offset from interface)")
plt.ylabel("normal cell size")
plt.title("Adapted normal grading (no Graded ratio was ever stated)")
plt.show()
