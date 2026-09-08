# %% [markdown]
# # Point refinement inside a structured band
#
# A size map that targets a single point inside a structured band shows the
# anisotropic character of sweep adaptation: the band's normal and
# tangential arrays respond *independently* (directional min-collapse), so
# refinement clusters around the point in both directions while the band
# remains an exact tensor grid — no unstructured island, no admissibility
# repair.

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
# ## Model and target
#
# The two-box stack from notebook 24, band of thickness 0.4 into `upper`.
# The target: fine resolution at the point (2.0, 1.2) — the middle of the
# band — growing linearly away from it.

# %%
sweep = StructuredSweep(name="qw", on="lower___upper", thickness={"upper": 0.4})
specs = {
    "qw": [StructuredSweepResolutionSpec(tangential=0.5, normal={"upper": 4})],
}
mesh0 = generate_mesh(
    entities=[
        PolySurface(polygons=shapely.box(0, 0, 4, 1), physical_name="lower", mesh_order=2),
        PolySurface(polygons=shapely.box(0, 1, 4, 2), physical_name="upper", mesh_order=1),
    ],
    sweeps=[sweep],
    dim=2,
    output_mesh="point_refine_0.msh",
    checkpoint_cad="point_refine.xao",
    default_characteristic_length=0.5,
    resolution_specs=specs,
)
plot2D(mesh0, title="Before: uniform band", wireframe=True)

# %%
P = np.array([2.0, 1.2])


def point_size_map():
    xs, ys = np.meshgrid(
        np.linspace(0.0, 4.0, 121), np.linspace(0.6, 1.8, 61), indexing="ij"
    )
    dist = np.hypot(xs - P[0], ys - P[1])
    sizes = np.clip(0.02 + 0.3 * dist, 0.02, 0.5)
    return np.column_stack(
        [xs.ravel(), ys.ravel(), np.zeros(xs.size), sizes.ravel()]
    )


# %% [markdown]
# ## Adapt (three iterations — the per-iteration clamp `change_max=2`
# bounds how fast sizes may move, so a deep target takes a few steps)

# %%
mesh_i, specs_i = mesh0, specs
for it in range(1, 4):
    mesh_i, specs_i = remesh_structured(
        input_mesh=mesh_i,
        geometry_file="point_refine.xao",
        sweeps=[sweep],
        resolution_specs=specs_i,
        size_map=point_size_map(),
        output_mesh=f"point_refine_{it}.msh",
        default_characteristic_length=0.5,
    )

plot2D(mesh_i, title="After: refinement clustered at (2.0, 1.2)", wireframe=True)

# %% [markdown]
# ## The anisotropy, made explicit
#
# The adapted spec carries the two 1D arrays. The tangential spacing h_t(ξ)
# dips near ξ = 2.0 and the normal spacing h_n(η) dips near η = 0.2 —
# each direction saw its own min-collapse of the same isotropic target.

# %%
spec = specs_i["qw"][0]
t = np.asarray(spec.tangential)
n = np.asarray(spec.normal["upper"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))
ax1.step(t[:-1], np.diff(t), where="post")
ax1.axvline(P[0], color="k", ls=":", label="target ξ")
ax1.set_xlabel("ξ (tangential)")
ax1.set_ylabel("h_t")
ax1.legend()
ax2.step(n[:-1], np.diff(n), where="post")
ax2.axvline(P[1] - 1.0, color="k", ls=":", label="target η")
ax2.set_xlabel("η (normal, from interface)")
ax2.set_ylabel("h_n")
ax2.legend()
fig.suptitle("Independent tangential / normal response to a point target")
fig.tight_layout()
plt.show()

# %% [markdown]
# The band is still an exact tensor grid — every interior node lies on a
# grid line — which is the admissibility-by-construction witness:

# %%
pts = mesh_i.points[:, :2]
band = pts[(pts[:, 1] >= 1.0 - 1e-9) & (pts[:, 1] <= 1.4 + 1e-9)]
xs = np.unique(np.round(band[:, 0], 9))
ys = np.unique(np.round(band[:, 1], 9))
assert len(band) == len(xs) * len(ys)
print(f"tensor grid intact: {len(xs)} x {len(ys)} = {len(band)} band nodes")
