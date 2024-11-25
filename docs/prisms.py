# %% [markdown]
# # Prisms
# Polygons can be associated with arbitrarily complex extrusion rules to form 3D Prisms.

# %%
import shapely

from meshwell.model import Model
from meshwell.prism import Prism
import plotly.graph_objects as go

# %%
# We use shapely as an API to enter polygons

# Initialize GMSH and create the mesh
polygon_hull = shapely.Polygon(
    [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
)
line1 = shapely.LineString([[0.5, 0.5], [1.5, 1.5]])
polygon_hole1 = shapely.buffer(line1, 0.2)
line2 = shapely.LineString([[1.5, 0.5], [0.5, 1.5]])
polygon_hole2 = shapely.buffer(line2, 0.2)
polygon = polygon_hull - polygon_hole1 - polygon_hole2

# %% [markdown]
# View the polygons:
#
# Plot the shapely polygons
# plt.figure(figsize=(8, 8))
# plt.plot(*polygon.exterior.xy, 'b-', label='Polygon 1 exterior')
# plt.plot(*polygon.interiors[0].xy, 'b--', label='Polygon 1 hole')
# plt.axis('equal')
# plt.title('Shapely Polygons')
# plt.legend()
# plt.show()

# %% [markdown]
# Mesh a prism by combining with buffers
# %%

buffers = {0.0: 0.05, 1.0: -0.05}

model = Model(n_threads=1)
poly3D = Prism(
    polygons=polygon,
    buffers=buffers,
    model=model,
    physical_name="my_prism1",
)

entities_list = [poly3D]

mesh = model.mesh(
    entities_list=entities_list,
    filename="prism.msh",
)


# Read and plot the mesh

# %%

# Create lists to store all vertices and edges
vertices_x = []
vertices_y = []
vertices_z = []
edge_x = []
edge_y = []
edge_z = []

# Plot each tetrahedron
for tet in mesh.cells_dict["tetra"]:
    vertices = mesh.points[tet]

    # Add vertices
    vertices_x.extend(vertices[:, 0])
    vertices_y.extend(vertices[:, 1])
    vertices_z.extend(vertices[:, 2])

    # Add edges by connecting vertices
    for i in range(4):
        for j in range(i + 1, 4):
            edge_x.extend([vertices[i, 0], vertices[j, 0], None])
            edge_y.extend([vertices[i, 1], vertices[j, 1], None])
            edge_z.extend([vertices[i, 2], vertices[j, 2], None])

# Create the vertices scatter plot
vertices_trace = go.Scatter3d(
    x=vertices_x,
    y=vertices_y,
    z=vertices_z,
    mode="markers",
    marker=dict(size=3, color="blue"),
    name="Vertices",
)

# Create the edges scatter plot
edges_trace = go.Scatter3d(
    x=edge_x,
    y=edge_y,
    z=edge_z,
    mode="lines",
    line=dict(color="blue", width=1),
    opacity=0.1,
    name="Edges",
)

# Create the figure and add traces
fig = go.Figure(data=[vertices_trace, edges_trace])

# Update layout for better visualization
fig.update_layout(
    title="Interactive 3D Mesh",
    scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="cube"),
    showlegend=True,
)

fig.show()
# %%