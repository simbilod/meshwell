"""End-to-end test: structured polyprism whose face_partition has 2+ pieces.

The partition is induced by a non-structured neighbour sharing the slab's
top z-plane.
"""
from __future__ import annotations

from shapely.geometry import Polygon

# Validator import: used by tests that call generate_mesh(validate_structured=True).
from meshwell.structured.validator import validate_structured_mesh  # noqa: F401


def _square(x=0, y=0, w=1, h=1) -> Polygon:
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def test_structured_slab_with_top_neighbour_face_partition_has_two_pieces():
    """Plan-only: confirm the 2-piece partition is built (no mesh)."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan

    s = PolyPrism(
        polygons=_square(0, 0, 4, 4),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="slab",
    )
    n = PolyPrism(
        polygons=_square(0, 0, 2, 4),
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="cap",
    )
    plan = build_plan([s, n])
    assert len(plan.slabs) == 1
    assert len(plan.slabs[0].face_partition) == 2, (
        f"Expected 2-piece partition; got " f"{len(plan.slabs[0].face_partition)}"
    )


def test_structured_slab_with_top_neighbour_produces_multi_piece_wedges(tmp_path):
    """Mesh test: structured slab with 2-piece face_partition produces wedges.

    A structured slab whose top is partially covered by a non-structured
    neighbour produces a 2-piece face_partition; both pieces mesh correctly.

    Phase 5(a): boundary node correspondence uses Layer B's output_edges map
    (edge identity + XY matching) instead of the old XY-only approach, which
    failed when BOP introduced top boundary nodes not present on the bottom.
    """
    import meshio
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    # Structured slab: 4x4 footprint, z=[0, 1], 2 layers.
    s = PolyPrism(
        polygons=_square(0, 0, 4, 4),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="slab",
    )
    # Non-structured neighbour: 2x4, sits on top of s's top half.
    n = PolyPrism(
        polygons=_square(0, 0, 2, 4),
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="cap",
    )

    out_msh = tmp_path / "multipiece.msh"
    generate_mesh(
        [s, n],
        dim=3,
        output_mesh=out_msh,
        default_characteristic_length=0.5,
        validate_structured=True,
    )

    m = meshio.read(out_msh)
    cell_types = {cb.type for cb in m.cells}
    # The slab produces wedges (one set per piece).
    wedge_like = any(ct in cell_types for ct in ("wedge", "wedge6"))
    assert wedge_like, f"Expected wedge cells, got {cell_types}"
    # Both physicals are present.
    assert "slab" in m.field_data
    assert "cap" in m.field_data
