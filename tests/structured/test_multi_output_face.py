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


def test_intra_entity_seam_not_tagged_as_None(tmp_path):
    """Internal seam between sub-solids of the same entity must not be tagged.

    The face should NOT appear in the entity___None physical group (it is
    interior to the logical volume, not an exterior surface).

    Scene: structured slab at z=[0,1], neighbours `a` (x=[0,3]) and `b`
    (x=[1,4]) at z=[1,2] with overlapping xy footprints. BOP fragments `a`
    into two sub-solids (A-only piece x=[0,1] and AB-overlap piece x=[1,3]);
    the shared internal face at x=1 used to leak into a___None.
    """
    import numpy as np

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

    out = tmp_path / "intra_seam.msh"
    generate_mesh(
        [slab, a, b], dim=3, output_mesh=out, default_characteristic_length=0.5
    )
    m = meshio.read(out)

    # If a___None exists, no triangle in it should lie entirely on the
    # internal vertical plane x=1 (which is the intra-entity seam between
    # the A-only and AB-overlap BOP sub-solids of `a`).
    if "a___None" not in m.field_data:
        return  # no exterior group at all is fine

    tag = m.field_data["a___None"][0]
    bad_count = 0
    for i, cb in enumerate(m.cells):
        if cb.type != "triangle":
            continue
        phys = m.cell_data["gmsh:physical"][i]
        mask = phys == tag
        if not np.any(mask):
            continue
        verts = m.points[cb.data[mask]]  # (n_tagged, 3, 3)
        # A triangle is on the x=1 seam if all 3 vertices have x ≈ 1.
        on_seam = np.all(np.abs(verts[:, :, 0] - 1.0) < 1e-3, axis=1)
        bad_count += int(np.sum(on_seam))

    assert (
        bad_count == 0
    ), f"{bad_count} triangle(s) wrongly tagged a___None on the x=1 internal seam"


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
