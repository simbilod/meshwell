from shapely.geometry import LineString, Polygon

from meshwell.interface_tag import InterfaceTag
from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.resolution import StructuredExtrusionResolutionSpec


def test_structured_with_interface_tag(tmp_path):
    p = PolyPrism(
        polygons=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="slab",
        structured=True,
    )
    # Define InterfaceTag along part of the boundary of the structured prism.
    # The boundary is at y=0, x in [0, 10].
    # We put it on x in [0, 5].
    tag = InterfaceTag(
        linestrings=LineString([(0, 0), (5, 0)]),
        zmin=0.0,
        zmax=1.0,
        physical_name="iface",
    )
    generate_mesh(
        [p, tag],
        dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=1.0,
        resolution_specs={"slab": [StructuredExtrusionResolutionSpec(n_layers=2)]},
    )
