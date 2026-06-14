import meshio
from shapely.geometry import Polygon

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.resolution import StructuredExtrusionResolutionSpec

SQ = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


def test_single_slab_yields_wedges_only(tmp_path):
    p = PolyPrism(
        polygons=SQ,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="slab",
        structured=True,
    )
    below = PolyPrism(
        polygons=SQ,
        buffers={-1.0: 0.0, 0.0: 0.0},
        physical_name="below",
        mesh_order=5.0,
    )
    above = PolyPrism(
        polygons=SQ,
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="above",
        mesh_order=5.0,
    )
    generate_mesh(
        [p, below, above],
        dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
        resolution_specs={"slab": [StructuredExtrusionResolutionSpec(n_layers=2)]},
    )
    m = meshio.read(tmp_path / "out.msh")
    # Count wedge/tet cells restricted to the structured slab via cell_sets.
    slab_sets = m.cell_sets["slab"]
    slab_wedges = sum(
        len(s)
        for s, b in zip(slab_sets, m.cells)
        if b.type == "wedge" and s is not None
    )
    slab_tets = sum(
        len(s)
        for s, b in zip(slab_sets, m.cells)
        if b.type == "tetra" and s is not None
    )
    assert slab_wedges > 0
    assert slab_tets == 0
    assert "slab" in m.cell_sets
