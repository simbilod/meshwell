from shapely.geometry import Polygon

import meshio
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
    generate_mesh(
        [p],
        dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
        resolution_specs={"slab": [StructuredExtrusionResolutionSpec(n_layers=2)]},
    )
    m = meshio.read(tmp_path / "out.msh")
    wedge_count = sum(cb.data.shape[0] for cb in m.cells if cb.type == "wedge")
    tet_count = sum(cb.data.shape[0] for cb in m.cells if cb.type == "tetra")
    assert wedge_count > 0
    assert tet_count == 0
    assert "slab" in m.cell_sets
