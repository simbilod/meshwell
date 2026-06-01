"""Regression test: closed-circle structured slabs produce full lateral wall coverage."""
import math

from shapely.geometry import Polygon

import meshio
from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.resolution import StructuredExtrusionResolutionSpec


def _disc(r, n=48):
    return Polygon(
        [
            (r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n))
            for i in range(n)
        ]
    )


def test_full_circle_lateral_wall_coverage(tmp_path):
    """A closed-circle structured slab must have a complete lateral wall mesh.

    Previously a cache-key collapse left only half the cylindrical wall
    tagged in physical groups.
    """
    disc = PolyPrism(
        _disc(1.0),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="disc",
        structured=True,
        identify_arcs=True,
    )
    generate_mesh(
        [disc],
        dim=3,
        output_mesh=tmp_path / "x.msh",
        default_characteristic_length=0.3,
        resolution_specs={"disc": [StructuredExtrusionResolutionSpec(n_layers=3)]},
    )
    m = meshio.read(tmp_path / "x.msh")
    # Quads only live on the lateral wall (bot/top are triangulated).
    n_quads = sum(cb.data.shape[0] for cb in m.cells if cb.type == "quad")
    # 1 unit radius -> circumference 2π ≈ 6.28; at cl=0.3 -> ~22 segments
    # times n_layers=3 -> at least 60 quads expected. Tolerate triangulation
    # variance: require ≥ 50 (well above the 36-quad broken state).
    assert n_quads >= 50, (
        f"only {n_quads} quads on the cylindrical wall — "
        "indicates the lateral wall is partially missing geometry "
        "(cache-collapse regression)"
    )
