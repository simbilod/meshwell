"""Verify a void volume has no wedges in the output mesh."""
from pathlib import Path

import meshio
from shapely.geometry import Polygon

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.resolution import StructuredExtrusionResolutionSpec

SQ_BIG = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
SQ_SMALL = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])


def test_void_does_not_appear_as_3d_group(tmp_path: Path):
    bg = PolyPrism(
        SQ_BIG,
        {0.0: 0.0, 1.0: 0.0},
        physical_name="bg",
        structured=True,
        mesh_order=2.0,
    )
    hole = PolyPrism(
        SQ_SMALL,
        {0.0: 0.0, 1.0: 0.0},
        physical_name="hole",
        structured=True,
        mesh_order=1.0,
        mesh_bool=False,
    )
    below = PolyPrism(
        SQ_BIG,
        {-1.0: 0.0, 0.0: 0.0},
        physical_name="below",
        mesh_order=5.0,
    )
    above = PolyPrism(
        SQ_BIG,
        {1.0: 0.0, 2.0: 0.0},
        physical_name="above",
        mesh_order=5.0,
    )
    generate_mesh(
        [bg, hole, below, above],
        dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
        resolution_specs={
            "bg": [StructuredExtrusionResolutionSpec(n_layers=2)],
        },
    )
    m = meshio.read(tmp_path / "out.msh")
    assert "bg" in m.cell_sets
    assert "hole" not in m.cell_sets, "void should NOT appear as a 3D physical group"
    # bg must have wedges.
    bg_sets = m.cell_sets["bg"]
    wedges = sum(
        len(s) for s, b in zip(bg_sets, m.cells) if b.type == "wedge" and s is not None
    )
    assert wedges > 0
