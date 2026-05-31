import numpy as np
from shapely.geometry import Polygon

import meshio
from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.resolution import StructuredExtrusionResolutionSpec

SQ_BIG = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
SQ_SMALL = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])


def test_stacked_with_unstructured_cap_conformal(tmp_path):
    a = PolyPrism(
        polygons=SQ_BIG,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="a",
        structured=True,
    )
    b = PolyPrism(
        polygons=SQ_SMALL,
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="b",
        structured=True,
    )
    cap = PolyPrism(
        polygons=SQ_BIG,
        buffers={2.0: 0.0, 3.0: 0.0},
        physical_name="cap",
    )
    generate_mesh(
        [a, b, cap],
        dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
        resolution_specs={
            "a": [StructuredExtrusionResolutionSpec(n_layers=2)],
            "b": [StructuredExtrusionResolutionSpec(n_layers=2)],
        },
    )
    m = meshio.read(tmp_path / "out.msh")
    wedge_count = sum(cb.data.shape[0] for cb in m.cells if cb.type == "wedge")
    tet_count = sum(cb.data.shape[0] for cb in m.cells if cb.type == "tetra")
    assert wedge_count > 0
    assert tet_count > 0
    pts = np.round(m.points, 5)
    assert len(np.unique(pts, axis=0)) == len(pts), "duplicate node positions"
