import plotly.graph_objects as go
import numpy as np

"""From https://en.wikipedia.org/wiki/Help:Distinguishable_colors"""
colors = [
    # "#000000",
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
):
    # Create a plotly figure
    fig = go.Figure()

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
            physical_groups_2D = [1]  # Use dummy group ID if no triangle physicals
        if "line" in mesh.cell_data_dict["gmsh:physical"]:
            physical_groups_1D = np.unique(mesh.cell_data_dict["gmsh:physical"]["line"])
        else:
            physical_groups_1D = [1]  # Use dummy group ID if no line physicals
    else:
        physical_groups_2D = [1]  # Use dummy group ID
        physical_groups_1D = [1]

    # Plot triangles for each 2D physical group
    for i, group in enumerate(physical_groups_2D):
        # Skip if physicals specified and this group not in them
        if physicals is not None and "gmsh:physical" in mesh.cell_data_dict:
            # Check if any of the names for this group ID match the requested physicals
            if not any(name in physicals for name in id_to_name[group]):
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

        # Get color for this group (cycle through colors if more groups than colors)
        color = colors[i % len(colors)]

        # Get group name for legend
        group_name = ", ".join(id_to_name[group]) if group in id_to_name else "mesh"
        first_triangle = True

        # Plot each triangle in the group with larger points
        for triangle in group_cells:
            x = mesh.points[triangle, 0]
            y = mesh.points[triangle, 1]
            # Close the triangle by repeating first point
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers" if wireframe else "lines",
                    line=dict(color=color),
                    marker=dict(size=5) if wireframe else None,
                    fill="toself" if not wireframe else None,
                    fillcolor=color if not wireframe else None,
                    showlegend=first_triangle,  # Only show legend for first triangle of group
                    name=group_name,
                )
            )
            first_triangle = False

    j = i
    # Plot lines for each 1D physical group
    if not ignore_lines:
        for i, group in enumerate(physical_groups_1D):
            i += j
            # Skip if physicals specified and this group not in them
            if physicals is not None and "gmsh:physical" in mesh.cell_data_dict:
                if not any(name in physicals for name in id_to_name[group]):
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

            # Get color for this group (cycle through colors if more groups than colors)
            color = colors[i + 1 % len(colors)]

            # Get group name for legend
            group_name = ", ".join(id_to_name[group]) if group in id_to_name else "mesh"
            first_line = True

            # Plot each line in the group
            for line in group_cells:
                x = mesh.points[line, 0]
                y = mesh.points[line, 1]
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines+markers" if wireframe else "lines",
                        line=dict(color=color),
                        marker=dict(size=5) if wireframe else None,
                        showlegend=first_line,  # Only show legend for first line of group
                        name=group_name,
                    )
                )
                first_line = False

    # Update layout with interactive features
    fig.update_layout(
        xaxis_title="x",
        yaxis_title="y",
        showlegend=True,
        width=800,
        height=800,
        dragmode="zoom",  # Enable zoom by dragging
        hovermode="closest",
        modebar=dict(
            add=[
                "zoom",
                "pan",
                "select",
                "lasso2d",
                "zoomIn2d",
                "zoomOut2d",
                "autoScale2d",
                "resetScale2d",
            ]
        ),
    )

    # Make the plot aspect ratio equal
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    # Add title if provided
    if title is not None:
        fig.update_layout(title=title)

    # Add hover text
    for trace in fig.data:
        if trace.x is not None and trace.x[0] is not None:  # Skip legend-only traces
            trace.update(hoverinfo="x+y", hoverlabel=dict(bgcolor="white"))

    # Show the plot
    fig.show(config={"scrollZoom": True})  # Enable scroll to zoom
