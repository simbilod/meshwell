"""Phase 5(d): tests for common-refinement partition with overlapping neighbours.

Phase 5(d) fixes compute_face_partition to use individual neighbour boundaries
(common refinement) so each slab partition piece has exactly 1 bot + 1 top
OCC face after BOP, by construction.
"""
from __future__ import annotations

import meshio
from shapely.geometry import Polygon


def _square(x=0, y=0, w=1, h=1) -> Polygon:
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def test_two_overlapping_top_neighbours_meshes_cleanly(tmp_path):
    """Slab z=[0,1], two neighbours at z=[1,2] with overlapping xy footprints.

    Phase 5(d) common-refinement partition produces 3 pieces (A-only, AB-overlap,
    B-only). Each piece has exactly 1 bot + 1 top OCC face after BOP.
    """
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    slab = PolyPrism(
        polygons=_square(0, 0, 4, 4),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="slab",
    )
    a = PolyPrism(
        polygons=_square(0, 0, 3, 4),
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="a",
    )
    b = PolyPrism(
        polygons=_square(1, 0, 3, 4),
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="b",
    )

    out = tmp_path / "multi_output.msh"
    generate_mesh(
        [slab, a, b], dim=3, output_mesh=out, default_characteristic_length=0.5
    )
    m = meshio.read(out)
    cell_types = {cb.type for cb in m.cells}
    assert any(
        ct in cell_types for ct in ("wedge", "wedge6")
    ), f"Expected wedge cells in slab; got {cell_types}"
    assert "slab" in m.field_data


def test_two_overlapping_bottom_neighbours_meshes_cleanly(tmp_path):
    """Symmetric multi-bottom test: two neighbours below the slab with overlapping xy.

    Phase 5(d) common-refinement partition produces a 3-piece partition;
    each piece's bot has exactly 1 OCC face.
    """
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    slab = PolyPrism(
        polygons=_square(0, 0, 4, 4),
        buffers={1.0: 0.0, 2.0: 0.0},  # slab on TOP
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="slab",
    )
    a = PolyPrism(
        polygons=_square(0, 0, 3, 4),
        buffers={0.0: 0.0, 1.0: 0.0},  # under slab, left
        physical_name="a",
    )
    b = PolyPrism(
        polygons=_square(1, 0, 3, 4),
        buffers={0.0: 0.0, 1.0: 0.0},  # under slab, right, overlaps A
        physical_name="b",
    )

    out = tmp_path / "multi_bot.msh"
    generate_mesh(
        [slab, a, b], dim=3, output_mesh=out, default_characteristic_length=0.5
    )
    m = meshio.read(out)
    cell_types = {cb.type for cb in m.cells}
    assert any(
        ct in cell_types for ct in ("wedge", "wedge6")
    ), f"Expected wedge cells; got {cell_types}"
    assert "slab" in m.field_data
