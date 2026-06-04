from shapely.geometry import Polygon

import meshio
from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.resolution import StructuredExtrusionResolutionSpec

SQ_BIG = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
SQ_SMALL = Polygon([(2, 2), (4, 2), (4, 4), (2, 4)])


def test_lower_mesh_order_void_carves_structured(tmp_path):
    big = PolyPrism(
        polygons=SQ_BIG,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="bg",
        structured=True,
        mesh_order=2.0,
    )
    void = PolyPrism(
        polygons=SQ_SMALL,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="void",
        structured=True,
        mesh_order=1.0,
        mesh_bool=False,
    )
    below = PolyPrism(
        polygons=SQ_BIG,
        buffers={-1.0: 0.0, 0.0: 0.0},
        physical_name="below",
        mesh_order=5.0,
    )
    above = PolyPrism(
        polygons=SQ_BIG,
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="above",
        mesh_order=5.0,
    )
    generate_mesh(
        [big, void, below, above],
        dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
        resolution_specs={"bg": [StructuredExtrusionResolutionSpec(n_layers=2)]},
    )
    m = meshio.read(tmp_path / "out.msh")
    wedge_count = sum(cb.data.shape[0] for cb in m.cells if cb.type == "wedge")
    assert wedge_count > 0
    assert "void" not in m.cell_sets, "void should not produce 3D group"
