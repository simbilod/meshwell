"""Regression: arc-bearing cohort + unstructured neighbour at shared z-plane."""
import math

from shapely.geometry import Polygon

import meshio
from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.resolution import StructuredExtrusionResolutionSpec


def _disc(cx, cy, r, n=48):
    return Polygon(
        [
            (
                cx + r * math.cos(2 * math.pi * i / n),
                cy + r * math.sin(2 * math.pi * i / n),
            )
            for i in range(n)
        ]
    )


def _square(x, y, w, h):
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def test_disc_cohort_with_square_cap_above(tmp_path):
    """Arc-bearing cohort topped by a polyline unstructured cap.

    The pre-cut on the cap at z=1 must adopt arc edges (matching the disc's
    OCC arc edges) so BOP can merge the shared face. Without arc propagation
    the cap's polyline edges don't merge with the disc's true arc edges and
    BOP fragments the disc's top face.
    """
    disc = PolyPrism(
        _disc(0, 0, 1.0),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="disc",
        structured=True,
        identify_arcs=True,
    )
    cap = PolyPrism(
        _square(-3, -3, 6, 6),
        {1.0: 0.0, 2.0: 0.0},
        physical_name="cap",
        identify_arcs=True,
    )
    msh = tmp_path / "x.msh"
    generate_mesh(
        [disc, cap],
        dim=3,
        output_mesh=msh,
        default_characteristic_length=0.4,
        resolution_specs={
            "disc": [StructuredExtrusionResolutionSpec(n_layers=2)],
        },
    )
    m = meshio.read(msh)
    # Both physical groups must be present with elements.
    assert "disc" in m.cell_sets
    assert "cap" in m.cell_sets
    wedges = sum(cb.data.shape[0] for cb in m.cells if cb.type == "wedge")
    tets = sum(cb.data.shape[0] for cb in m.cells if cb.type == "tetra")
    assert wedges > 0, "expected wedges in disc"
    assert tets > 0, "expected tets in cap"
