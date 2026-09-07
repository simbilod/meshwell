"""plot2D rendering regression tests."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: plt.show() is a no-op, figures stay in memory

import matplotlib.pyplot as plt
import shapely

from meshwell.orchestrator import generate_mesh
from meshwell.polysurface import PolySurface
from meshwell.resolution import StructuredSweepResolutionSpec
from meshwell.structured.sweep import StructuredSweep
from meshwell.visualization import plot2D


def test_plot2d_renders_quad_cells(tmp_path):
    """plot2D must draw quad surface cells, not just triangles.

    Regression: a structured sweep with element_type="quad" produces quad
    band faces; plot2D previously only handled "triangle" cells, so the band
    rendered as one empty outline instead of its 4x2 quad grid.
    """
    lower = PolySurface(
        polygons=shapely.box(0, 0, 4, 1), physical_name="lower", mesh_order=2
    )
    upper = PolySurface(
        polygons=shapely.box(0, 1, 4, 2), physical_name="upper", mesh_order=1
    )
    mesh = generate_mesh(
        entities=[lower, upper],
        sweeps=[
            StructuredSweep(name="qw", on="lower___upper", thickness={"upper": 0.4})
        ],
        dim=2,
        output_mesh=str(tmp_path / "quad.msh"),
        default_characteristic_length=0.5,
        resolution_specs={
            "qw": [
                StructuredSweepResolutionSpec(
                    tangential=1.0, normal={"upper": 2}, element_type="quad"
                )
            ],
        },
    )
    assert sum(cb.data.shape[0] for cb in mesh.cells if cb.type == "quad") == 8

    plt.close("all")
    plot2D(mesh, title="quad", wireframe=True)
    ax = plt.gcf().axes[0]
    # In wireframe mode each closed quad is one Line2D of 5 points (4 + repeat).
    quad_lines = [ln for ln in ax.lines if len(ln.get_xdata()) == 5]
    assert len(quad_lines) == 8
    plt.close("all")
