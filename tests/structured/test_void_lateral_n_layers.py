"""Verify n_layers mismatch check ignores keep=False owners."""
from pathlib import Path

from shapely.geometry import Polygon

import meshio
from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.resolution import StructuredExtrusionResolutionSpec

SQ_BIG = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
SQ_SMALL = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])


def test_void_does_not_trigger_n_layers_mismatch(tmp_path: Path):
    """Void lateral face shared with bg does not trigger n_layers mismatch.

    The void's lateral face is shared with bg's inner ring. Since the
    void carries no n_layers (no resolution_specs entry for `hole`), the
    mismatch check would otherwise fire. Filtering keep=False owners
    avoids that.
    """
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
    generate_mesh(
        [bg, hole],
        dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
        resolution_specs={
            "bg": [StructuredExtrusionResolutionSpec(n_layers=2)],
        },
    )
    m = meshio.read(tmp_path / "out.msh")
    bg_sets = m.cell_sets["bg"]
    wedges = sum(
        len(s) for s, b in zip(bg_sets, m.cells) if b.type == "wedge" and s is not None
    )
    assert wedges > 0
