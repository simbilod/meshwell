"""Plotting routines."""
import matplotlib.pyplot as plt
import numpy as np

"""From https://en.wikipedia.org/wiki/Help:Distinguishable_colors"""
colors = [
    "#1CE6FF",
    "#FF34FF",
    "#FF4A46",
    "#008941",
    "#006FA6",
    "#A30059",
    "#FFDBE5",
    "#FFFF00",
    "#7A4900",
    "#0000A6",
    "#63FFAC",
    "#B79762",
    "#004D43",
    "#8FB0FF",
    "#997D87",
    "#5A0007",
    "#809693",
    "#FEFFE6",
    "#1B4400",
    "#4FC601",
    "#3B5DFF",
    "#4A3B53",
    "#FF2F80",
]


def plot2D(
    mesh,
    physicals=None,
    wireframe: bool = False,
    title: str | None = None,
    ignore_lines: bool = False,
) -> None:
    """Plot a 2D mesh using matplotlib.

    Args:
        mesh: The mesh object to be plotted
        physicals: Physical regions to highlight
        wireframe: Whether to display mesh as wireframe
        title: Plot title
        ignore_lines: Whether to ignore line elements in the mesh
    """
    # Create matplotlib figure with specified dimensions
    _fig, ax = plt.subplots(figsize=(10, 10))

    # Create mapping dicts from integer IDs to group names
    id_to_name = {}
    for name, (id_, _) in mesh.field_data.items():
        if id_ not in id_to_name:
            id_to_name[id_] = [name]
        else:
            id_to_name[id_].append(name)

    # Get unique physical groups if they exist, otherwise treat all cells as one group
    physical_groups_2D = []
    physical_groups_1D = []
    if "gmsh:physical" in mesh.cell_data_dict:
        if "triangle" in mesh.cell_data_dict["gmsh:physical"]:
            physical_groups_2D = np.unique(
                mesh.cell_data_dict["gmsh:physical"]["triangle"]
            )
        else:
            physical_groups_2D = [1]
        if "line" in mesh.cell_data_dict["gmsh:physical"]:
            physical_groups_1D = np.unique(mesh.cell_data_dict["gmsh:physical"]["line"])
        else:
            physical_groups_1D = [1]
    else:
        physical_groups_2D = [1]
        physical_groups_1D = [1]

    # Plot triangles for each 2D physical group
    for i, group in enumerate(physical_groups_2D):
        # Skip if physicals specified and this group not in them
        if (
            physicals is not None
            and "gmsh:physical" in mesh.cell_data_dict
            and not any(name in physicals for name in id_to_name[group])
        ):
            continue

        # Get cells for this physical group
        if (
            "gmsh:physical" in mesh.cell_data_dict
            and "triangle" in mesh.cell_data_dict["gmsh:physical"]
        ):
            group_cells = mesh.cells_dict["triangle"][
                mesh.cell_data_dict["gmsh:physical"]["triangle"] == group
            ]
        else:
            group_cells = mesh.cells_dict["triangle"]

        # Get color for this group
        color = colors[i % len(colors)]

        # Get group name for legend
        group_name = ", ".join(id_to_name[group]) if group in id_to_name else "mesh"

        # Plot triangles
        for triangle in group_cells:
            x = mesh.points[triangle, 0]
            y = mesh.points[triangle, 1]
            # Close the triangle
            x = np.append(x, x[0])
            y = np.append(y, y[0])

            if wireframe:
                ax.plot(
                    x,
                    y,
                    color=color,
                    marker="o" if wireframe else None,
                    markersize=3 if wireframe else None,
                    label=group_name,
                )
            else:
                ax.fill(x, y, color=color, alpha=0.5, label=group_name)
                ax.plot(x, y, color=color, linewidth=0.5)

            # Only include label once in legend
            group_name = "_nolegend_"

    # Plot lines for each 1D physical group
    if not ignore_lines:
        for i, group in enumerate(physical_groups_1D):
            # Skip if physicals specified and this group not in them
            if (
                physicals is not None
                and "gmsh:physical" in mesh.cell_data_dict
                and not any(name in physicals for name in id_to_name[group])
            ):
                continue

            # Get cells for this physical group
            if (
                "gmsh:physical" in mesh.cell_data_dict
                and "line" in mesh.cell_data_dict["gmsh:physical"]
            ):
                group_cells = mesh.cells_dict["line"][
                    mesh.cell_data_dict["gmsh:physical"]["line"] == group
                ]
            else:
                group_cells = mesh.cells_dict["line"]

            # Get color and group name
            color = colors[(i + len(physical_groups_2D)) % len(colors)]
            group_name = ", ".join(id_to_name[group]) if group in id_to_name else "mesh"

            # Plot lines
            for line in group_cells:
                x = mesh.points[line, 0]
                y = mesh.points[line, 1]
                ax.plot(
                    x,
                    y,
                    color=color,
                    marker="o" if wireframe else None,
                    markersize=3 if wireframe else None,
                    label=group_name,
                )
                group_name = "_nolegend_"

    # Set equal aspect ratio
    ax.set_aspect("equal")

    # Add title if provided
    if title is not None:
        ax.set_title(title)

    # Add labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Add legend
    ax.legend()

    # Enable grid
    ax.grid(True, linestyle="--", alpha=0.3)

    # Show plot
    plt.show()


def plot3D(
    mesh,
    title: str = "3D Mesh",
    output_file: str | None = None,
):
    """Plot 3D mesh using plotly.

    Args:
        mesh: meshio mesh object
        title: Plot title
        output_file: Optional path to save HTML plot
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print(
            "Plotly is required for 3D visualization. Please install it with `pip install plotly`."
        )
        return

    # Create lists to store all vertices and edges
    vertices_x = []
    vertices_y = []
    vertices_z = []
    edge_x = []
    edge_y = []
    edge_z = []

    # Handle Tetrahedra
    if "tetra" in mesh.cells_dict:
        for tet in mesh.cells_dict["tetra"]:
            vertices = mesh.points[tet]

            # Add vertices
            vertices_x.extend(vertices[:, 0])
            vertices_y.extend(vertices[:, 1])
            vertices_z.extend(vertices[:, 2])

            # Add edges by connecting vertices
            # Edges of a tetrahedron: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
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
        title=title,
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="cube"
        ),
        showlegend=True,
    )

    if output_file:
        fig.write_html(output_file)
    else:
        fig.show()
