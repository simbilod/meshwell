"""End-to-end test: PolyPrism(structured=True) -> mesh -> wedge elements."""
from __future__ import annotations

from shapely.geometry import Polygon


def _square(x=0, y=0, w=1, h=1) -> Polygon:
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def test_single_structured_slab_produces_wedge_mesh(tmp_path):
    """A single structured PolyPrism with n_layers=2 produces wedges in the mesh."""
    import meshio

    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    p = PolyPrism(
        polygons=_square(0, 0, 2, 2),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="block",
    )

    out_msh = tmp_path / "structured.msh"
    generate_mesh([p], dim=3, output_mesh=out_msh, default_characteristic_length=0.5)

    m = meshio.read(out_msh)
    # Wedge cells should be present. meshio may name them "wedge" or "wedge15"
    # or similar; check for any prismatic cell type.
    cell_types = {cb.type for cb in m.cells}
    wedge_like = any(ct in cell_types for ct in ("wedge", "wedge6", "wedge15"))
    assert wedge_like, f"Expected wedge-like cells, got {cell_types}"
    # Physical "block" should be present.
    assert "block" in m.field_data


def test_single_structured_slab_default_characteristic_length(tmp_path):
    """Smoke: structured pipeline runs without exceptions."""
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    p = PolyPrism(
        polygons=_square(),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="b",
    )
    out_msh = tmp_path / "smoke.msh"
    generate_mesh([p], dim=3, output_mesh=out_msh, default_characteristic_length=1.0)
    assert out_msh.exists()
