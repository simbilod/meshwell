"""Phase 5(b): tests for multi-output-face per piece routing.

Triggered when BOP splits one slab partition piece's top OCC face into
multiple sub-faces (e.g., two overlapping non-structured neighbours
sharing the slab's top z-plane).
"""
from __future__ import annotations

import meshio
from shapely.geometry import Polygon


def _square(x=0, y=0, w=1, h=1) -> Polygon:
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def test_two_overlapping_top_neighbours_meshes_cleanly(tmp_path):
    """Slab z=[0,1], two neighbours at z=[1,2] with overlapping xy footprints.

    The neighbours' xy union covers the whole slab top, so Phase 1's planner
    makes only 1 partition piece. BOP splits the slab's top OCC face into 3
    sub-faces (A-only, AB-overlap, B-only). Phase 5(b) routes per-cell.
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


def test_phase_4_assertion_explicitly_dropped():
    """Sanity: apply_structured_mesh no longer asserts len(top_tags) == 1.

    With Phase 5(b), the per-piece loop should accept any number of top
    output faces and route per-cell.
    """
    import inspect

    from meshwell.structured.builder import apply_structured_mesh

    src = inspect.getsource(apply_structured_mesh)
    # The old error message should no longer appear in apply_structured_mesh.
    assert (
        "expected exactly one bottom + one top gmsh face" not in src
    ), "Phase 5(b) should have removed the 'expected exactly one' assertion."
