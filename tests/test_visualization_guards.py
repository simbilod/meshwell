"""plot2D must not crash on meshes missing a cell block or field_data entry."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import meshio
import numpy as np

from meshwell.visualization import plot2D


def _line_only_mesh():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    return meshio.Mesh(points, [("line", np.array([[0, 1], [1, 2]]))])


def _triangle_only_mesh():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return meshio.Mesh(points, [("triangle", np.array([[0, 1, 2]]))])


def test_plot2d_line_only_mesh_does_not_crash():
    plot2D(_line_only_mesh())
    plt.close("all")


def test_plot2d_triangle_only_mesh_with_lines_enabled():
    plot2D(_triangle_only_mesh(), ignore_lines=False)
    plt.close("all")


def test_plot2d_physicals_filter_with_missing_field_data():
    m = _triangle_only_mesh()
    m.cell_data = {"gmsh:physical": [np.array([7])]}  # id 7 absent from field_data
    plot2D(m, physicals=["anything"])
    plt.close("all")
