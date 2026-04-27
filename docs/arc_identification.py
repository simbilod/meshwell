# %% [markdown]
# # Arc Identification
# Meshwell can automatically identify sequences of vertices that form circular arcs.
# This reduces the complexity of the CAD model and improves mesh quality.

# %%
import matplotlib.pyplot as plt
import numpy as np
import shapely

from meshwell.cad_occ import cad_occ
from meshwell.mesh import mesh
from meshwell.occ_xao_writer import write_xao
from meshwell.polyline import PolyLine
from meshwell.polysurface import PolySurface

# %%
# Create a circular geometry with many points
theta = np.linspace(0, np.pi / 2, 50)
vertices = [(np.cos(t), np.sin(t)) for t in theta]
vertices += [(0, 1), (0, 0), (1, 0)]
poly = shapely.Polygon(vertices)

# %% [markdown]
# ## PolySurface with Arc Identification
# We enable `identify_arcs=True` to recover the true curved boundary.

# %%
ps = PolySurface(
    poly,
    identify_arcs=True,
    min_arc_points=4,
    arc_tolerance=1e-3,
    physical_name="curved_surface",
)

# %% [markdown]
# ### Visualize Decomposition
# We can inspect how the polygon was partitioned into lines and arcs.
# Blue lines are straight segments, red lines are identified arcs.

# %%
ax = ps.plot_decomposition()
plt.show()

# %% [markdown]
# ## PolyLine with Arc Identification

# %%
# Create a wavy line that contains an arc
t = np.linspace(0, np.pi, 50)
arc_vertices = [(np.cos(t_val), np.sin(t_val)) for t_val in t]
line_vertices = [(1, 0), (2, 0), (2, 1)]
all_vertices = line_vertices + arc_vertices

pl = PolyLine(
    shapely.LineString(all_vertices), identify_arcs=True, physical_name="curved_wire"
)

# %%
ax = pl.plot_decomposition()
plt.show()

# %% [markdown]
# ## Meshing Results
# Generate the mesh. Notice how the arcs are preserved in the underlying CAD.

# %%
write_xao(cad_occ([ps]), "arc_example.xao")
output_mesh = mesh(
    dim=2,
    input_file="arc_example.xao",
    output_file="arc_example.msh",
    default_characteristic_length=0.1,
    mesh_element_order=2,
)
