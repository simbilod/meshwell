"""Phase 6(a1) + Phase 6(a2): arc structured polyprism tests.

Phase 6(a1): single-piece arc structured polyprisms.
Phase 6(a2): split-arc provenance for multi-piece partitions.
"""
from __future__ import annotations

import math

import meshio
from shapely.geometry import Polygon


def _disc(cx=0.0, cy=0.0, r=1.0, n=32):
    return Polygon(
        [
            (
                cx + r * math.cos(2 * math.pi * i / n),
                cy + r * math.sin(2 * math.pi * i / n),
            )
            for i in range(n)
        ]
    )


def _annulus(cx=0.0, cy=0.0, r_outer=2.0, r_inner=1.0, n=32):
    outer = [
        (
            cx + r_outer * math.cos(2 * math.pi * i / n),
            cy + r_outer * math.sin(2 * math.pi * i / n),
        )
        for i in range(n)
    ]
    inner = [
        (
            cx + r_inner * math.cos(2 * math.pi * i / n),
            cy + r_inner * math.sin(2 * math.pi * i / n),
        )
        for i in range(n)
    ]
    return Polygon(outer, [inner])


def test_disc_structured_single_piece_meshes(tmp_path):
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    p = PolyPrism(
        polygons=_disc(),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
        physical_name="disc",
    )
    out = tmp_path / "disc.msh"
    generate_mesh([p], dim=3, output_mesh=out, default_characteristic_length=0.3)
    m = meshio.read(out)
    cell_types = {cb.type for cb in m.cells}
    assert any(
        ct in cell_types for ct in ("wedge", "wedge6")
    ), f"Expected wedge cells; got {cell_types}"
    assert "disc" in m.field_data


def test_annulus_structured_single_piece_meshes(tmp_path):
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    p = PolyPrism(
        polygons=_annulus(),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
        physical_name="ring",
    )
    out = tmp_path / "ring.msh"
    generate_mesh([p], dim=3, output_mesh=out, default_characteristic_length=0.3)
    m = meshio.read(out)
    cell_types = {cb.type for cb in m.cells}
    assert any(ct in cell_types for ct in ("wedge", "wedge6"))


def test_disc_embedded_in_cladding_meshes(tmp_path):
    """Disc inside larger cladding: single-piece partition.

    Cladding has zmin and zmax both outside slab z-range, so it doesn't
    split the slab top/bot.
    """
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    disc = PolyPrism(
        polygons=_disc(),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        identify_arcs=True,
        physical_name="disc",
        mesh_order=1.0,
    )
    cladding = PolyPrism(
        polygons=Polygon([(-3, -3), (3, -3), (3, 3), (-3, 3)]),
        buffers={-1.0: 0.0, 2.0: 0.0},
        physical_name="cladding",
        mesh_order=2.0,
    )
    out = tmp_path / "embedded.msh"
    generate_mesh(
        [disc, cladding], dim=3, output_mesh=out, default_characteristic_length=0.4
    )
    m = meshio.read(out)
    assert "disc" in m.field_data
    assert "cladding" in m.field_data


def test_split_disc_meshes_with_provenance(tmp_path):
    """Phase 6(a2): split disc meshes with provenance.

    A disc split by a top neighbour into multiple pieces should mesh
    correctly via arc-provenance lookup.
    """
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    disc = PolyPrism(
        polygons=_disc(),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        identify_arcs=True,
        physical_name="disc",
        mesh_order=1.0,
    )
    # Top neighbour cuts the disc into multiple pieces (two half-disc pieces).
    cap = PolyPrism(
        polygons=Polygon([(-2, 0), (2, 0), (2, 2), (-2, 2)]),
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="cap",
        mesh_order=2.0,
    )
    out = tmp_path / "split.msh"
    generate_mesh(
        [disc, cap], dim=3, output_mesh=out, default_characteristic_length=0.3
    )
    m = meshio.read(out)
    cell_types = {cb.type for cb in m.cells}
    assert any(
        ct in cell_types for ct in ("wedge", "wedge6")
    ), f"Expected wedge cells; got {cell_types}"
    assert "disc" in m.field_data
    assert "cap" in m.field_data
