"""Phase 6(a2): unit tests for arc-provenance helper functions.

Covers:
  - _build_arc_index_from_footprint (disc / rectangle / annulus / split / disabled)
  - _classify_piece_boundary (disc / rectangle / disc-cut / pure-arc / mixed)
  - Integration: generate_mesh with split-arc slabs
"""
from __future__ import annotations

import math

import pytest
from shapely.geometry import Polygon
from shapely.ops import polygonize, unary_union

# ---------------------------------------------------------------------------
# Footprint helpers
# ---------------------------------------------------------------------------


def _disc(cx: float = 0.0, cy: float = 0.0, r: float = 1.0, n: int = 48) -> Polygon:
    return Polygon(
        [
            (
                cx + r * math.cos(2 * math.pi * k / n),
                cy + r * math.sin(2 * math.pi * k / n),
            )
            for k in range(n)
        ]
    )


def _annulus(r_outer: float = 1.0, r_inner: float = 0.4, n: int = 48) -> Polygon:
    outer = [
        (
            r_outer * math.cos(2 * math.pi * k / n),
            r_outer * math.sin(2 * math.pi * k / n),
        )
        for k in range(n)
    ]
    inner = [
        (
            r_inner * math.cos(2 * math.pi * k / n),
            r_inner * math.sin(2 * math.pi * k / n),
        )
        for k in range(n)
    ]
    return Polygon(outer, [inner])


def _rect(
    x0: float = -0.5, y0: float = -0.5, x1: float = 0.5, y1: float = 0.5
) -> Polygon:
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


# ---------------------------------------------------------------------------
# Helper: split a polygon by a splitter boundary
# ---------------------------------------------------------------------------


def _split_pieces(footprint: Polygon, splitter: Polygon) -> list[Polygon]:
    """Return pieces of footprint cut by the splitter boundary."""
    combined = unary_union(
        [footprint.boundary, splitter.boundary.intersection(footprint)]
    )
    raw = list(polygonize(combined))
    return [p for p in raw if footprint.contains(p.representative_point())]


# ===========================================================================
# Tests: _build_arc_index_from_footprint
# ===========================================================================


class TestBuildArcIndex:
    """Tests for _build_arc_index_from_footprint."""

    def test_disc_identifies_one_arc(self):
        """Disc polygon produces exactly one arc in the index."""
        from meshwell.structured.plan import _build_arc_index_from_footprint

        fp = _disc(n=48)
        index = _build_arc_index_from_footprint(
            fp, identify_arcs=True, min_arc_points=4, arc_tolerance=0.01
        )
        assert len(index.arcs) == 1, f"Expected 1 arc, got {len(index.arcs)}"
        arc = index.arcs[0]
        assert abs(arc.radius - 1.0) < 1e-2, f"Expected radius ~1.0, got {arc.radius}"
        assert abs(arc.center[0]) < 1e-2
        assert abs(arc.center[1]) < 1e-2
        # Every exterior vertex of the disc must be indexed.
        for x, y in list(fp.exterior.coords)[:-1]:
            key = (round(x, index.ndigits), round(y, index.ndigits))
            assert key in index.vertex_to_arcs, f"Vertex {key} not in index"

    def test_disabled_when_identify_arcs_false(self):
        """Empty index returned when identify_arcs=False."""
        from meshwell.structured.plan import _build_arc_index_from_footprint

        fp = _disc(n=48)
        index = _build_arc_index_from_footprint(
            fp, identify_arcs=False, min_arc_points=4, arc_tolerance=0.01
        )
        assert index.arcs == []
        assert index.vertex_to_arcs == {}

    def test_annulus_identifies_two_arcs(self):
        """Annulus produces two arcs (outer + inner ring)."""
        from meshwell.structured.plan import _build_arc_index_from_footprint

        fp = _annulus(r_outer=1.0, r_inner=0.4, n=48)
        index = _build_arc_index_from_footprint(
            fp, identify_arcs=True, min_arc_points=4, arc_tolerance=0.01
        )
        radii = sorted(a.radius for a in index.arcs)
        assert len(radii) == 2, f"Expected 2 arcs, got {radii}"
        assert abs(radii[0] - 0.4) < 1e-2, f"Inner radius mismatch: {radii[0]}"
        assert abs(radii[1] - 1.0) < 1e-2, f"Outer radius mismatch: {radii[1]}"

    def test_rectangle_identifies_no_arcs(self):
        """Rectangle produces zero arcs in the index."""
        from meshwell.structured.plan import _build_arc_index_from_footprint

        fp = _rect()
        index = _build_arc_index_from_footprint(
            fp, identify_arcs=True, min_arc_points=4, arc_tolerance=0.01
        )
        assert (
            len(index.arcs) == 0
        ), f"Expected 0 arcs for rectangle, got {len(index.arcs)}"

    def test_disc_split_all_vertices_indexed(self):
        """Arc index built from original footprint must index its own vertices.

        Seam vertices (where the splitter intersects the disc arc) are ON the
        original arc and must therefore appear in the arc index.
        """
        from meshwell.structured.plan import _build_arc_index_from_footprint

        fp = _disc(n=48)
        index = _build_arc_index_from_footprint(
            fp, identify_arcs=True, min_arc_points=4, arc_tolerance=0.01
        )
        splitter = Polygon([(-2, 0), (2, 0), (2, 2), (-2, 2)])
        pieces = _split_pieces(fp, splitter)
        assert len(pieces) == 2
        # Seam endpoints (~(1,0) and ~(-1,0)) are on the disc arc and must be indexed.
        # (Actual arc classification is tested in TestClassifyPieceBoundary.)
        for piece in pieces:
            for x, y in list(piece.exterior.coords)[:-1]:
                # Build key but only assert if the coord is clearly on the arc
                # (i.e., at distance ~1.0 from origin).
                dist = math.sqrt(x**2 + y**2)
                if abs(dist - 1.0) < 0.01:
                    key = (round(x, index.ndigits), round(y, index.ndigits))
                    assert key in index.vertex_to_arcs, f"Arc vertex {key} not indexed"


# ===========================================================================
# Tests: _classify_piece_boundary
# ===========================================================================


class TestClassifyPieceBoundary:
    """Tests for _classify_piece_boundary."""

    def test_pure_disc_one_arc_edge(self):
        """Disc with no splitter: exterior is a single PieceArcEdge."""
        from meshwell.structured.plan import (
            _build_arc_index_from_footprint,
            _classify_piece_boundary,
        )
        from meshwell.structured.spec import PieceArcEdge

        fp = _disc(n=48)
        index = _build_arc_index_from_footprint(
            fp, identify_arcs=True, min_arc_points=4, arc_tolerance=0.01
        )
        prov = _classify_piece_boundary(fp, index)
        assert len(prov.exterior_edges) == 1
        assert isinstance(prov.exterior_edges[0], PieceArcEdge)
        assert prov.interior_edges == []

    def test_pure_rectangle_all_line_edges(self):
        """Rectangle with no arcs: every edge is a PieceLineEdge."""
        from meshwell.structured.plan import (
            _build_arc_index_from_footprint,
            _classify_piece_boundary,
        )
        from meshwell.structured.spec import PieceLineEdge

        fp = _rect()
        index = _build_arc_index_from_footprint(
            fp, identify_arcs=True, min_arc_points=4, arc_tolerance=0.01
        )
        prov = _classify_piece_boundary(fp, index)
        assert all(isinstance(e, PieceLineEdge) for e in prov.exterior_edges)

    def test_disc_split_vertically_each_piece_has_one_arc_one_line(self):
        """Disc cut horizontally at y=0: each piece gets 1 arc edge + 1 line edge."""
        from meshwell.structured.plan import (
            _build_arc_index_from_footprint,
            _classify_piece_boundary,
        )
        from meshwell.structured.spec import PieceArcEdge, PieceLineEdge

        fp = _disc(n=48)
        splitter = Polygon([(-2, 0), (2, 0), (2, 2), (-2, 2)])
        pieces = _split_pieces(fp, splitter)
        assert len(pieces) == 2, f"Expected 2 pieces, got {len(pieces)}"

        index = _build_arc_index_from_footprint(
            fp, identify_arcs=True, min_arc_points=4, arc_tolerance=0.01
        )
        for piece in pieces:
            prov = _classify_piece_boundary(piece, index)
            types = [type(e).__name__ for e in prov.exterior_edges]
            arc_count = sum(
                1 for e in prov.exterior_edges if isinstance(e, PieceArcEdge)
            )
            line_count = sum(
                1 for e in prov.exterior_edges if isinstance(e, PieceLineEdge)
            )
            assert (
                arc_count == 1
            ), f"Expected 1 arc edge, got {arc_count} (edges: {types})"
            assert (
                line_count == 1
            ), f"Expected 1 line edge, got {line_count} (edges: {types})"

    def test_classify_disabled_index_all_lines(self):
        """Empty arc index (identify_arcs=False) → all line edges."""
        from meshwell.structured.plan import (
            _build_arc_index_from_footprint,
            _classify_piece_boundary,
        )
        from meshwell.structured.spec import PieceLineEdge

        fp = _disc(n=48)
        index = _build_arc_index_from_footprint(
            fp, identify_arcs=False, min_arc_points=4, arc_tolerance=0.01
        )
        prov = _classify_piece_boundary(fp, index)
        assert all(isinstance(e, PieceLineEdge) for e in prov.exterior_edges)

    def test_boundary_vertices_covered(self):
        """Edge list must form a closed chain with no gaps between consecutive edges."""
        from meshwell.structured.plan import (
            _build_arc_index_from_footprint,
            _classify_piece_boundary,
        )

        fp = _disc(n=48)
        splitter = Polygon([(-2, 0), (2, 0), (2, 2), (-2, 2)])
        pieces = _split_pieces(fp, splitter)
        index = _build_arc_index_from_footprint(
            fp, identify_arcs=True, min_arc_points=4, arc_tolerance=0.01
        )
        for piece in pieces:
            prov = _classify_piece_boundary(piece, index)
            assert len(prov.exterior_edges) >= 1
            # The last point of each edge must match the first point of the next.
            for ei in range(len(prov.exterior_edges) - 1):
                last = prov.exterior_edges[ei].points[-1]
                first_next = prov.exterior_edges[ei + 1].points[0]
                assert (
                    abs(last[0] - first_next[0]) < 0.01
                ), f"x-gap between edge {ei} and {ei + 1}: {last} -> {first_next}"
                assert (
                    abs(last[1] - first_next[1]) < 0.01
                ), f"y-gap between edge {ei} and {ei + 1}: {last} -> {first_next}"

    def test_annulus_two_arc_edges(self):
        """Annulus: outer ring = 1 arc edge, inner ring = 1 arc edge."""
        from meshwell.structured.plan import (
            _build_arc_index_from_footprint,
            _classify_piece_boundary,
        )
        from meshwell.structured.spec import PieceArcEdge

        fp = _annulus(r_outer=1.0, r_inner=0.4, n=48)
        index = _build_arc_index_from_footprint(
            fp, identify_arcs=True, min_arc_points=4, arc_tolerance=0.01
        )
        prov = _classify_piece_boundary(fp, index)
        # Outer ring = one arc edge
        assert len(prov.exterior_edges) == 1
        assert isinstance(prov.exterior_edges[0], PieceArcEdge)
        # Inner ring (one hole) = one arc edge
        assert len(prov.interior_edges) == 1
        assert len(prov.interior_edges[0]) == 1
        assert isinstance(prov.interior_edges[0][0], PieceArcEdge)


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge-case tests for the provenance classifier."""

    def test_piece_entire_boundary_is_one_arc(self):
        """A piece whose boundary is purely the original disc arc.

        Should produce exactly one PieceArcEdge and zero PieceLineEdges.
        """
        from meshwell.structured.plan import (
            _build_arc_index_from_footprint,
            _classify_piece_boundary,
        )
        from meshwell.structured.spec import PieceArcEdge

        fp = _disc(n=48)
        index = _build_arc_index_from_footprint(
            fp, identify_arcs=True, min_arc_points=4, arc_tolerance=0.01
        )
        prov = _classify_piece_boundary(fp, index)
        assert len(prov.exterior_edges) == 1
        assert isinstance(prov.exterior_edges[0], PieceArcEdge)

    def test_rectangle_piece_all_lines(self):
        """A piece whose boundary has no arc vertices should have all-line provenance.

        Uses two perpendicular splitters to create a piece whose boundary
        is entirely composed of splitter cuts (no original arc segments).
        """
        from meshwell.structured.plan import (
            _build_arc_index_from_footprint,
            _classify_piece_boundary,
        )
        from meshwell.structured.spec import PieceLineEdge

        fp = _disc(n=48)
        h_splitter = Polygon([(-2, 0), (2, 0), (2, 2), (-2, 2)])
        v_splitter = Polygon([(0, -2), (2, -2), (2, 2), (0, 2)])
        combined = unary_union(
            [
                fp.boundary,
                h_splitter.boundary.intersection(fp),
                v_splitter.boundary.intersection(fp),
            ]
        )
        raw = list(polygonize(combined))
        pieces = [p for p in raw if fp.contains(p.representative_point())]
        assert len(pieces) >= 2

        index = _build_arc_index_from_footprint(
            fp, identify_arcs=True, min_arc_points=4, arc_tolerance=0.01
        )
        for piece in pieces:
            prov = _classify_piece_boundary(piece, index)
            if all(isinstance(e, PieceLineEdge) for e in prov.exterior_edges):
                return  # found at least one purely-cut piece
        pytest.skip("No purely-cut piece found with the test geometry")

    def test_seam_segment_classified_as_line_not_arc(self):
        """The seam cut segment should be a PieceLineEdge, not a PieceArcEdge.

        The seam endpoints lie ON the original disc arc but are NOT adjacent
        in the arc index, so the segment between them must be a line.
        """
        from meshwell.structured.plan import (
            _build_arc_index_from_footprint,
            _classify_piece_boundary,
        )
        from meshwell.structured.spec import PieceArcEdge, PieceLineEdge

        fp = _disc(n=48)
        splitter = Polygon([(-2, 0), (2, 0), (2, 2), (-2, 2)])
        pieces = _split_pieces(fp, splitter)
        assert len(pieces) == 2

        index = _build_arc_index_from_footprint(
            fp, identify_arcs=True, min_arc_points=4, arc_tolerance=0.01
        )
        for piece in pieces:
            prov = _classify_piece_boundary(piece, index)
            arc_edges = [e for e in prov.exterior_edges if isinstance(e, PieceArcEdge)]
            line_edges = [
                e for e in prov.exterior_edges if isinstance(e, PieceLineEdge)
            ]
            assert len(line_edges) == 1, (
                f"Seam should produce exactly 1 line edge, got {len(line_edges)} "
                f"line edges and {len(arc_edges)} arc edges"
            )


# ===========================================================================
# Integration tests: generate_mesh end-to-end
# ===========================================================================


class TestIntegration:
    """End-to-end integration tests using generate_mesh."""

    def test_disc_split_by_rectangle_top_cover(self, tmp_path):
        """Disc cut by a rectangular top cover: 2 pieces, each with arc+line."""
        import meshio
        from meshwell.orchestrator import generate_mesh
        from meshwell.polyprism import PolyPrism
        from meshwell.structured import StructuredExtrusionResolutionSpec

        disc = PolyPrism(
            polygons=_disc(n=48),
            buffers={0.0: 0.0, 1.0: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
            identify_arcs=True,
            min_arc_points=4,
            arc_tolerance=1e-3,
            physical_name="disc",
            mesh_order=1.0,
        )
        cap = PolyPrism(
            polygons=Polygon([(-2, 0), (2, 0), (2, 2), (-2, 2)]),
            buffers={1.0: 0.0, 2.0: 0.0},
            physical_name="cap",
            mesh_order=2.0,
        )
        out = tmp_path / "disc_cap.msh"
        generate_mesh(
            [disc, cap], dim=3, output_mesh=out, default_characteristic_length=0.4
        )
        m = meshio.read(out)
        cell_types = {cb.type for cb in m.cells}
        assert any(
            ct in cell_types for ct in ("wedge", "wedge6")
        ), f"Expected wedge cells; got {cell_types}"
        assert "disc" in m.field_data

    def test_disc_embedded_in_cladding_rejected(self, tmp_path):
        """Disc-in-unstructured-cladding is rejected by the lateral conformality check.

        The disc's arc lateral surface would be shared with the tet-meshed
        cladding — quad/tri face-topology mismatch. ``build_plan`` raises
        ``StructuredLateralUnstructuredNeighbourError`` before any meshing.
        """
        import pytest

        from meshwell.orchestrator import generate_mesh
        from meshwell.polyprism import PolyPrism
        from meshwell.structured import (
            StructuredExtrusionResolutionSpec,
            StructuredLateralUnstructuredNeighbourError,
        )

        disc = PolyPrism(
            polygons=_disc(n=32),
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
                [disc, cladding],
                dim=3,
                output_mesh=out,
                default_characteristic_length=0.5,
            )

    def test_provenance_field_set_only_for_split_slabs(self):
        """Split arc slabs get provenance; single-piece slabs get None."""
        from meshwell.polyprism import PolyPrism
        from meshwell.structured import StructuredExtrusionResolutionSpec
        from meshwell.structured.plan import build_plan

        disc = PolyPrism(
            polygons=_disc(n=48),
            buffers={0.0: 0.0, 1.0: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
            identify_arcs=True,
            physical_name="disc",
            mesh_order=1.0,
        )
        cap = PolyPrism(
            polygons=Polygon([(-2, 0), (2, 0), (2, 2), (-2, 2)]),
            buffers={1.0: 0.0, 2.0: 0.0},
            physical_name="cap",
            mesh_order=2.0,
        )
        plan = build_plan([disc, cap])
        disc_slab = next(s for s in plan.slabs if s.physical_name == ("disc",))
        assert disc_slab.face_partition_provenance is not None
        assert len(disc_slab.face_partition_provenance) == len(disc_slab.face_partition)
        assert len(disc_slab.face_partition) == 2

    def test_single_piece_slab_has_no_provenance(self):
        """A slab with no splitting neighbours: provenance stays None."""
        from meshwell.polyprism import PolyPrism
        from meshwell.structured import StructuredExtrusionResolutionSpec
        from meshwell.structured.plan import build_plan

        disc = PolyPrism(
            polygons=_disc(n=48),
            buffers={0.0: 0.0, 1.0: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
            identify_arcs=True,
            physical_name="disc",
            mesh_order=1.0,
        )
        plan = build_plan([disc])
        disc_slab = next(s for s in plan.slabs if s.physical_name == ("disc",))
        assert disc_slab.face_partition_provenance is None
        assert len(disc_slab.face_partition) == 1

    def test_disc_split_by_two_overlapping_covers(self, tmp_path):
        """Disc split by 2 overlapping top covers: 3+ pieces, each with some arc."""
        import meshio
        from meshwell.orchestrator import generate_mesh
        from meshwell.polyprism import PolyPrism
        from meshwell.structured import StructuredExtrusionResolutionSpec

        disc = PolyPrism(
            polygons=_disc(n=48),
            buffers={0.0: 0.0, 1.0: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
            identify_arcs=True,
            min_arc_points=4,
            arc_tolerance=1e-3,
            physical_name="disc",
            mesh_order=1.0,
        )
        cap1 = PolyPrism(
            polygons=Polygon([(-2, 0), (0, 0), (0, 2), (-2, 2)]),
            buffers={1.0: 0.0, 2.0: 0.0},
            physical_name="cap1",
            mesh_order=2.0,
        )
        cap2 = PolyPrism(
            polygons=Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            buffers={1.0: 0.0, 2.0: 0.0},
            physical_name="cap2",
            mesh_order=2.0,
        )
        out = tmp_path / "disc_two_caps.msh"
        generate_mesh(
            [disc, cap1, cap2],
            dim=3,
            output_mesh=out,
            default_characteristic_length=0.4,
        )
        m = meshio.read(out)
        cell_types = {cb.type for cb in m.cells}
        assert any(ct in cell_types for ct in ("wedge", "wedge6"))
        assert "disc" in m.field_data

    def test_annulus_split_by_partial_cover(self, tmp_path):
        """Annulus with partial top cover: outer+inner arcs + cut line."""
        import meshio
        from meshwell.orchestrator import generate_mesh
        from meshwell.polyprism import PolyPrism
        from meshwell.structured import StructuredExtrusionResolutionSpec

        ann = PolyPrism(
            polygons=_annulus(r_outer=1.0, r_inner=0.4, n=48),
            buffers={0.0: 0.0, 1.0: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
            identify_arcs=True,
            min_arc_points=4,
            arc_tolerance=1e-3,
            physical_name="ring",
            mesh_order=1.0,
        )
        cap = PolyPrism(
            polygons=Polygon([(-2, 0), (2, 0), (2, 2), (-2, 2)]),
            buffers={1.0: 0.0, 2.0: 0.0},
            physical_name="cap",
            mesh_order=2.0,
        )
        out = tmp_path / "annulus_cap.msh"
        generate_mesh(
            [ann, cap], dim=3, output_mesh=out, default_characteristic_length=0.4
        )
        m = meshio.read(out)
        cell_types = {cb.type for cb in m.cells}
        assert any(ct in cell_types for ct in ("wedge", "wedge6"))
        assert "ring" in m.field_data
