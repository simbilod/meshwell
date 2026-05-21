"""Phase 6(a1) + Phase 6(a2): arc structured polyprism tests.

Phase 6(a1): single-piece arc structured polyprisms.
Phase 6(a2): split-arc provenance for multi-piece partitions.
"""
from __future__ import annotations

import math

import pytest
from shapely.geometry import Polygon

import meshio


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


def test_disc_embedded_in_cladding_rejected(tmp_path):
    """Disc inside an unstructured cladding is now rejected at plan time.

    The disc's circular lateral surface would be shared with the tet-meshed
    cladding — quad/tri face-topology mismatch, non-conformal. The
    structured pipeline raises ``StructuredLateralUnstructuredNeighbourError``
    rather than silently producing a non-conformal mesh.
    """
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import (
        StructuredExtrusionResolutionSpec,
        StructuredLateralUnstructuredNeighbourError,
    )

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
    with pytest.raises(StructuredLateralUnstructuredNeighbourError):
        generate_mesh(
            [disc, cladding], dim=3, output_mesh=out, default_characteristic_length=0.4
        )


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


def test_arc_provenance_propagates_to_neighbour_below():
    """A structured arc slab's PieceArcEdge propagates to a below-neighbour's provenance.

    Layer mid (z=[1,2]): a disc (identify_arcs=True) cut into 2 half-pieces
    by a structured strip cap above.
    Layer bottom (z=[0,1]): a slab (identify_arcs=True) whose footprint
    contains the disc's projected XY extent.

    After planning, the bottom slab's face_partition_provenance should include
    at least one PieceArcEdge inherited from the disc's two half-piece arcs.
    """
    import math

    from shapely.geometry import Polygon

    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan
    from meshwell.structured.spec import PieceArcEdge

    # 32-vertex disc, radius 1, centered at (0, 0)
    n = 32
    disc = Polygon(
        [
            (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n))
            for i in range(n)
        ]
    )

    disc_slab = PolyPrism(
        polygons=disc,
        buffers={1.0: 0.0, 2.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
        point_tolerance=1e-9,
        physical_name="DISC",
    )
    # Cap covers the upper half of the disc footprint at z=[2,3].
    cap = PolyPrism(
        polygons=Polygon([(-2, 0), (2, 0), (2, 2), (-2, 2)]),
        buffers={2.0: 0.0, 3.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="CAP",
    )
    bot = PolyPrism(
        polygons=Polygon([(-2, -2), (2, -2), (2, 2), (-2, 2)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
        point_tolerance=1e-9,
        physical_name="BOT",
    )

    plan = build_plan([bot, disc_slab, cap])
    by_name = {s.physical_name[0]: s for s in plan.slabs}

    bot_slab = by_name["BOT"]
    assert len(bot_slab.face_partition) >= 2, (
        f"BOT should be cut by the disc's piece boundary at y=0; got "
        f"{len(bot_slab.face_partition)} pieces"
    )
    assert bot_slab.face_partition_provenance is not None
    arc_edges = []
    for prov in bot_slab.face_partition_provenance:
        arc_edges.extend(
            edge for edge in prov.exterior_edges if isinstance(edge, PieceArcEdge)
        )
        for ring in prov.interior_edges:
            arc_edges.extend(edge for edge in ring if isinstance(edge, PieceArcEdge))
    assert arc_edges, (
        "BOT face_partition_provenance contains no PieceArcEdge entries; "
        "arc inheritance from the disc above did not propagate"
    )
    # Inherited arcs should have radius ~1 (the disc radius).
    radii = [round(e.radius, 2) for e in arc_edges]
    assert any(
        abs(r - 1.0) < 0.05 for r in radii
    ), f"no inherited arc has radius ~1; got radii: {radii}"


def test_no_arc_inheritance_when_neighbour_identify_arcs_false():
    """When the arc-bearing neighbour has identify_arcs=False, no arc inherits."""
    import math

    from shapely.geometry import Polygon

    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan
    from meshwell.structured.spec import PieceArcEdge

    n = 32
    disc = Polygon(
        [
            (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n))
            for i in range(n)
        ]
    )
    disc_slab = PolyPrism(
        polygons=disc,
        buffers={1.0: 0.0, 2.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        identify_arcs=False,  # KEY: arcs disabled on the neighbour
        physical_name="DISC",
    )
    cap = PolyPrism(
        polygons=Polygon([(-2, 0), (2, 0), (2, 2), (-2, 2)]),
        buffers={2.0: 0.0, 3.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="CAP",
    )
    bot = PolyPrism(
        polygons=Polygon([(-2, -2), (2, -2), (2, 2), (-2, 2)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
        physical_name="BOT",
    )
    plan = build_plan([bot, disc_slab, cap])
    by_name = {s.physical_name[0]: s for s in plan.slabs}
    bot_slab = by_name["BOT"]

    # No arcs anywhere (BOT's own footprint is a square; the neighbour is non-arc).
    if bot_slab.face_partition_provenance is None:
        return  # acceptable: provenance not even computed
    for prov in bot_slab.face_partition_provenance:
        for edge in prov.exterior_edges:
            assert not isinstance(
                edge, PieceArcEdge
            ), "BOT should not inherit arc edges when DISC has identify_arcs=False"
        for ring in prov.interior_edges:
            for edge in ring:
                assert not isinstance(edge, PieceArcEdge)
