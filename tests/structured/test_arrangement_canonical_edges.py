"""Tests for _build_canonical_edges in meshwell.structured.decompose."""
from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from meshwell.structured.decompose import _build_canonical_edges


def _rect(x1, y1, x2, y2):
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def _circle(cx, cy, r, n=48):
    a = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    return Polygon([(cx + r * np.cos(t), cy + r * np.sin(t)) for t in a])


def test_two_overlapping_rectangles_split_into_seven_edges():
    """Two overlapping squares share a single cut — expect 7 canonical edges.

    The planar graph has 7 unique edges (4 around left square minus
    shared cut + 4 around right square minus shared cut + 1 shared cut
    = some bookkeeping). Shapely's unary_union nodes at the two crossing
    points where the shared cut meets each square's left/right side.
    """
    merged = unary_union([_rect(0, 0, 6, 10).boundary, _rect(4, 0, 10, 10).boundary])
    edges, lookup = _build_canonical_edges(
        merged,
        z=0.0,
        point_tolerance=1e-3,
        identify_arcs=False,
        min_arc_points=5,
        arc_tolerance=1e-3,
    )
    assert len(edges) == 7
    # Every consecutive vertex pair on every edge is registered exactly once.
    seen = set()
    for ei, edge in enumerate(edges):
        for i in range(len(edge.vertex_keys) - 1):
            pair = frozenset({edge.vertex_keys[i], edge.vertex_keys[i + 1]})
            assert pair not in seen, f"duplicate pair on edge {ei}"
            seen.add(pair)
            assert lookup[pair] == ei
    # Closed standalone edges (none here) are NOT pair-indexed.
    assert all(not e.is_closed for e in edges)


def test_single_closed_disc_yields_one_closed_edge():
    """A standalone circle becomes one closed canonical edge.

    No other arrangement nodes, so is_closed=True; its vertex pairs are
    NOT registered in the lookup (closed-standalone fallback).
    """
    merged = unary_union([_circle(0, 0, 1.0).boundary])
    edges, lookup = _build_canonical_edges(
        merged,
        z=0.0,
        point_tolerance=1e-3,
        identify_arcs=True,
        min_arc_points=5,
        arc_tolerance=1e-3,
    )
    assert len(edges) == 1
    assert edges[0].is_closed
    # OPEN storage convention: first key != last key.
    assert edges[0].vertex_keys[0] != edges[0].vertex_keys[-1]
    # Closed-standalone edges are NOT pair-indexed.
    assert lookup == {}
    # identify_arcs=True with a 48-point circle produces at least one arc.
    assert any(s.is_arc for s in edges[0].segments)


def test_two_overlapping_discs_produce_shared_arc_edges():
    """Two overlapping discs cut each other at two points.

    Shapely's ``unary_union`` of polygon-circle boundaries treats each
    polygon's seam vertex (the angle-0 vertex at ``(cx+r, cy)``) as a
    node where the closed LinearRing splits — so two 48-vertex
    polygon-circles produce 6 arrangement edges, not the 4 you'd
    expect from analytic circles. (The four short edges between a
    seam vertex and a true crossing point may be too short to fit an
    arc; we only require AT LEAST ONE edge to carry arc segments.)
    """
    merged = unary_union([_circle(0, 0, 1.0).boundary, _circle(1.0, 0, 1.0).boundary])
    edges, lookup = _build_canonical_edges(
        merged,
        z=0.0,
        point_tolerance=1e-3,
        identify_arcs=True,
        min_arc_points=5,
        arc_tolerance=1e-3,
    )
    assert len(edges) == 6
    # No closed standalone edges in this configuration.
    assert all(not e.is_closed for e in edges)
    # AT LEAST ONE edge has an arc segment (the long arcs through the
    # disc boundaries are arc-fit; the short edges between a seam and
    # a crossing may be too short for arc detection and stay as line
    # runs).
    assert any(any(s.is_arc for s in e.segments) for e in edges)
    # Every consecutive pair across all edges is registered uniquely.
    n_pairs = sum(len(e.vertex_keys) - 1 for e in edges)
    assert len(lookup) == n_pairs


def test_polyline_xy_with_arrangement_replays_canonical_segments():
    """Calling polyline_xy with arrangement replays canonical segments.

    Two sub-pieces sharing a boundary edge get the SAME TShape.
    """
    from shapely.geometry import Polygon

    from meshwell.structured.build import EdgeRegistry, VertexRegistry
    from meshwell.structured.decompose import build_cohort_arrangement
    from meshwell.structured.types import Cohort, StructuredSlab

    # Two overlapping rectangles -> 3 sub-pieces, the middle of which
    # shares one cut edge with each side.
    def _slab(idx, poly):
        return StructuredSlab(
            source_index=idx,
            footprint=poly,
            zlo=0.0,
            zhi=1.0,
            mesh_order=1.0,
            mesh_bool=True,
            physical_name=("x",),
            identify_arcs=False,
            arc_tolerance=1e-3,
            min_arc_points=4,
        )

    left = _slab(0, Polygon([(0, 0), (6, 0), (6, 10), (0, 10)]))
    right = _slab(1, Polygon([(4, 0), (10, 0), (10, 10), (4, 10)]))
    cohort = Cohort(slabs=(left, right), z_planes=(0.0, 1.0))
    arr = build_cohort_arrangement(
        cohort_index=0,
        cohort=cohort,
        adjacent_unstructured=[],
        point_tolerance=1e-3,
    )

    vreg = VertexRegistry(point_tolerance=1e-3)
    ereg = EdgeRegistry(vertices=vreg, point_tolerance=1e-3)

    # Build the middle (overlap) sub-piece's exterior ring through both
    # forward and reversed traversal. They must emit the SAME OCC TShapes.
    overlap = Polygon([(4, 0), (6, 0), (6, 10), (4, 10), (4, 0)])
    coords = list(overlap.exterior.coords)
    edges_fwd = ereg.polyline_xy(
        [(x, y) for x, y in coords],
        z=0.0,
        identify_arcs=False,
        arrangement=arr,
    )
    edges_rev = ereg.polyline_xy(
        [(x, y) for x, y in reversed(coords)],
        z=0.0,
        identify_arcs=False,
        arrangement=arr,
    )

    from OCP.TopTools import TopTools_ShapeMapHasher

    hasher = TopTools_ShapeMapHasher()
    fwd_ids = sorted(hasher(e) for e in edges_fwd)
    rev_ids = sorted(hasher(e) for e in edges_rev)
    assert fwd_ids == rev_ids


def test_polyline_xy_falls_back_when_arrangement_none():
    """When arrangement=None, polyline_xy preserves today's behavior."""
    from meshwell.structured.build import EdgeRegistry, VertexRegistry

    vreg = VertexRegistry(point_tolerance=1e-3)
    ereg = EdgeRegistry(vertices=vreg, point_tolerance=1e-3)
    coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
    edges = ereg.polyline_xy(coords, z=0.0, identify_arcs=False)
    assert len(edges) == 4


def test_polyline_xy_arc_direction_invariant_tshape():
    """Two-overlapping-discs arc edges: same OCC TShapes forward and reverse.

    This is the arc-path counterpart to
    test_polyline_xy_with_arrangement_replays_canonical_segments,
    which only exercised line segments. The arc_xy cache key is
    endpoint-direction-invariant on the sorted endpoint keys, so
    canonical-edge replay MUST emit segments in canonical order
    (not reverse each segment's points) for TShape sharing to hold
    on arc geometry as well as line geometry.
    """
    import numpy as np
    from shapely.geometry import Polygon

    from meshwell.structured.build import EdgeRegistry, VertexRegistry
    from meshwell.structured.decompose import build_cohort_arrangement
    from meshwell.structured.types import Cohort, StructuredSlab

    def _circle_local(cx, cy, r, n=48):
        a = np.linspace(0, 2 * np.pi, n + 1)[:-1]
        return Polygon([(cx + r * np.cos(t), cy + r * np.sin(t)) for t in a])

    def _slab(idx, poly):
        return StructuredSlab(
            source_index=idx,
            footprint=poly,
            zlo=0.0,
            zhi=1.0,
            mesh_order=1.0,
            mesh_bool=True,
            physical_name=("x",),
            identify_arcs=True,
            arc_tolerance=1e-3,
            min_arc_points=5,
        )

    left = _slab(0, _circle_local(0, 0, 1.0))
    right = _slab(1, _circle_local(1.0, 0, 1.0))
    cohort = Cohort(slabs=(left, right), z_planes=(0.0, 1.0))
    arr = build_cohort_arrangement(
        cohort_index=0,
        cohort=cohort,
        adjacent_unstructured=[],
        point_tolerance=1e-3,
    )
    # Get one of the lens sub-pieces from the arrangement.
    sub_polys = list(arr.polygons)
    # Pick the lens (the smallest-area piece — it lies in BOTH discs).
    sub_polys.sort(key=lambda p: p.area)
    lens = sub_polys[0]

    vreg = VertexRegistry(point_tolerance=1e-3)
    ereg = EdgeRegistry(vertices=vreg, point_tolerance=1e-3)

    coords = list(lens.exterior.coords)
    edges_fwd = ereg.polyline_xy(
        [(x, y) for x, y in coords],
        z=0.0,
        identify_arcs=True,
        min_arc_points=5,
        arc_tolerance=1e-3,
        arrangement=arr,
    )
    edges_rev = ereg.polyline_xy(
        [(x, y) for x, y in reversed(coords)],
        z=0.0,
        identify_arcs=True,
        min_arc_points=5,
        arc_tolerance=1e-3,
        arrangement=arr,
    )

    # TShape identity: same set of TShapes in both directions.
    from OCP.TopTools import TopTools_ShapeMapHasher

    hasher = TopTools_ShapeMapHasher()
    fwd_hashes = sorted(hasher(e) for e in edges_fwd)
    rev_hashes = sorted(hasher(e) for e in edges_rev)
    assert fwd_hashes == rev_hashes, (
        "Arc edges built forward vs reverse produced different TShapes — "
        "_polyline_xy_canonical's even-L arc mid-point asymmetry bug"
    )


def test_parallel_edges_between_same_nodes_raise():
    """Parallel edges between the same arrangement nodes raise CanonicalArrangementError.

    Constructed artificially because unary_union normally prevents it —
    but the validator must catch it if the input ever produces it.
    """
    # Two two-vertex LineStrings between the same endpoints, NOT routed
    # through unary_union so they stay as parallel components.
    from shapely.geometry import MultiLineString

    from meshwell.structured.exceptions import CanonicalArrangementError

    merged = MultiLineString(
        [
            LineString([(0, 0), (1, 0)]),
            LineString([(0, 0), (1, 0)]),
        ]
    )
    with pytest.raises(CanonicalArrangementError):
        _build_canonical_edges(
            merged,
            z=0.0,
            point_tolerance=1e-3,
            identify_arcs=False,
            min_arc_points=5,
            arc_tolerance=1e-3,
        )


def test_face_xy_passes_arrangement_to_polyline_xy():
    """FaceRegistry.face_xy forwards the arrangement.

    The underlying _build_horizontal_face / polyline_xy use canonical
    edges. Two distinct sub-pieces sharing a boundary edge then
    yield faces whose shared OCC edge is the same TShape.
    """
    from shapely.geometry import Polygon

    from meshwell.structured.build import (
        EdgeRegistry,
        FaceRegistry,
        VertexRegistry,
        _build_horizontal_face,
    )
    from meshwell.structured.decompose import build_cohort_arrangement
    from meshwell.structured.types import Cohort, StructuredSlab

    def _slab(idx, poly):
        return StructuredSlab(
            source_index=idx,
            footprint=poly,
            zlo=0.0,
            zhi=1.0,
            mesh_order=1.0,
            mesh_bool=True,
            physical_name=("x",),
            identify_arcs=False,
            arc_tolerance=1e-3,
            min_arc_points=4,
        )

    left = _slab(0, Polygon([(0, 0), (6, 0), (6, 10), (0, 10)]))
    right = _slab(1, Polygon([(4, 0), (10, 0), (10, 10), (4, 10)]))
    cohort = Cohort(slabs=(left, right), z_planes=(0.0, 1.0))
    arr = build_cohort_arrangement(
        cohort_index=0,
        cohort=cohort,
        adjacent_unstructured=[],
        point_tolerance=1e-3,
    )

    vreg = VertexRegistry(point_tolerance=1e-3)
    ereg = EdgeRegistry(vertices=vreg, point_tolerance=1e-3)
    freg = FaceRegistry(edges=ereg, point_tolerance=1e-3)

    # Two distinct sub-pieces from the arrangement -> two faces.
    # Their shared edge MUST be the same TShape.
    polys = list(arr.polygons)
    face1 = _build_horizontal_face(
        polys[0],
        0.0,
        ereg,
        identify_arcs=False,
        min_arc_points=5,
        arc_tolerance=1e-3,
        face_registry=freg,
        arrangement=arr,
    )
    face2 = _build_horizontal_face(
        polys[1],
        0.0,
        ereg,
        identify_arcs=False,
        min_arc_points=5,
        arc_tolerance=1e-3,
        face_registry=freg,
        arrangement=arr,
    )
    # If both faces share at least one edge TShape, the canonical
    # replay worked.
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopTools import TopTools_ShapeMapHasher

    def _edge_ids(face):
        hasher = TopTools_ShapeMapHasher()
        out = set()
        exp = TopExp_Explorer(face, TopAbs_EDGE)
        while exp.More():
            out.add(hasher(exp.Current()))
            exp.Next()
        return out

    assert _edge_ids(face1) & _edge_ids(
        face2
    ), "expected at least one shared edge TShape between adjacent faces"


def test_polyline_segments_uses_arrangement_when_provided():
    """polyline_segments with arrangement produces canonical segments.

    Ensures lateral faces' bot/top boundaries match the horizontal faces'
    TShapes when an arrangement is provided.
    """
    from shapely.geometry import Polygon

    from meshwell.structured.build import polyline_segments
    from meshwell.structured.decompose import build_cohort_arrangement
    from meshwell.structured.types import Cohort, StructuredSlab

    def _slab(idx, poly):
        return StructuredSlab(
            source_index=idx,
            footprint=poly,
            zlo=0.0,
            zhi=1.0,
            mesh_order=1.0,
            mesh_bool=True,
            physical_name=("x",),
            identify_arcs=False,
            arc_tolerance=1e-3,
            min_arc_points=4,
        )

    left = _slab(0, Polygon([(0, 0), (6, 0), (6, 10), (0, 10)]))
    right = _slab(1, Polygon([(4, 0), (10, 0), (10, 10), (4, 10)]))
    cohort = Cohort(slabs=(left, right), z_planes=(0.0, 1.0))
    arr = build_cohort_arrangement(
        cohort_index=0,
        cohort=cohort,
        adjacent_unstructured=[],
        point_tolerance=1e-3,
    )

    overlap_coords = [(4.0, 0.0), (6.0, 0.0), (6.0, 10.0), (4.0, 10.0), (4.0, 0.0)]
    # Both with and without arrangement should yield consistent segments
    # for a rectangle (only lines).
    segs_canon = polyline_segments(
        overlap_coords,
        identify_arcs=False,
        min_arc_points=5,
        arc_tolerance=1e-3,
        point_tolerance=1e-3,
        arrangement=arr,
        z=0.0,
    )
    assert all(s.kind == "line" for s in segs_canon)
    assert len(segs_canon) == 4


def test_build_cohort_compound_accepts_arrangement():
    """build_cohort_compound forwards arrangement to _build_horizontal_face.

    Ensures polyline_segments calls so sub-pieces sharing a boundary subset
    consume canonical edges.
    """
    from shapely.geometry import Polygon

    from meshwell.structured.build import (
        EdgeRegistry,
        FaceRegistry,
        VertexRegistry,
        build_cohort_compound,
    )
    from meshwell.structured.decompose import (
        arrangement_subpieces_for_interval,
        build_cohort_arrangement,
    )
    from meshwell.structured.types import Cohort, StructuredSlab

    def _slab(idx, poly):
        return StructuredSlab(
            source_index=idx,
            footprint=poly,
            zlo=0.0,
            zhi=1.0,
            mesh_order=1.0,
            mesh_bool=True,
            physical_name=("x",),
            identify_arcs=False,
            arc_tolerance=1e-3,
            min_arc_points=4,
        )

    left = _slab(0, Polygon([(0, 0), (6, 0), (6, 10), (0, 10)]))
    right = _slab(1, Polygon([(4, 0), (10, 0), (10, 10), (4, 10)]))
    cohort = Cohort(slabs=(left, right), z_planes=(0.0, 1.0))
    arr = build_cohort_arrangement(
        cohort_index=0,
        cohort=cohort,
        adjacent_unstructured=[],
        point_tolerance=1e-3,
    )
    subs = arrangement_subpieces_for_interval(arr, cohort, 0.0, 1.0)

    vreg = VertexRegistry(point_tolerance=1e-3)
    ereg = EdgeRegistry(vertices=vreg, point_tolerance=1e-3)
    freg = FaceRegistry(edges=ereg, point_tolerance=1e-3)

    compound, slab_meta = build_cohort_compound(
        cohort,
        subs,
        point_tolerance=1e-3,
        vertex_registry=vreg,
        edge_registry=ereg,
        face_registry=freg,
        arrangement=arr,
    )
    assert compound is not None
    assert len(slab_meta) == len(subs)


def test_validate_canonical_edge_coverage_passes_on_clean_arrangement():
    """A well-formed arrangement passes the coverage check."""
    from shapely.geometry import Polygon

    from meshwell.structured.decompose import (
        arrangement_subpieces_for_interval,
        build_cohort_arrangement,
        validate_canonical_edge_coverage,
    )
    from meshwell.structured.types import Cohort, StructuredSlab

    def _slab(idx, poly):
        return StructuredSlab(
            source_index=idx,
            footprint=poly,
            zlo=0.0,
            zhi=1.0,
            mesh_order=1.0,
            mesh_bool=True,
            physical_name=("x",),
            identify_arcs=False,
            arc_tolerance=1e-3,
            min_arc_points=4,
        )

    cohort = Cohort(
        slabs=(
            _slab(0, Polygon([(0, 0), (6, 0), (6, 10), (0, 10)])),
            _slab(1, Polygon([(4, 0), (10, 0), (10, 10), (4, 10)])),
        ),
        z_planes=(0.0, 1.0),
    )
    arr = build_cohort_arrangement(
        cohort_index=0,
        cohort=cohort,
        adjacent_unstructured=[],
        point_tolerance=1e-3,
    )
    subs = arrangement_subpieces_for_interval(arr, cohort, 0.0, 1.0)
    # Should not raise.
    validate_canonical_edge_coverage(arr, [s.sub_polygon for s in subs])


def test_pre_pass_threads_arrangement_to_cohort_compound():
    """structured_pre_pass propagates the per-cohort Arrangement to build_cohort_compound.

    The resulting compound's two overlapping rectangles produce three solids.
    """
    from shapely.geometry import Polygon

    from meshwell.polyprism import PolyPrism
    from meshwell.structured.pipeline import structured_pre_pass

    # Union bbox covering both rectangles, used for wrapping unstructured layers.
    wrapper_poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

    ents = [
        PolyPrism(
            Polygon([(0, 0), (6, 0), (6, 10), (0, 10)]),
            {0.0: 0.0, 1.0: 0.0},
            physical_name="a",
            structured=True,
            mesh_order=1.0,
        ),
        PolyPrism(
            Polygon([(4, 0), (10, 0), (10, 10), (4, 10)]),
            {0.0: 0.0, 1.0: 0.0},
            physical_name="b",
            structured=True,
            mesh_order=2.0,
        ),
        # Unstructured layers above/below so the cohort caps become
        # conformal interfaces rather than exterior faces.
        PolyPrism(wrapper_poly, {-1.0: 0.0, 0.0: 0.0}, physical_name="below"),
        PolyPrism(wrapper_poly, {1.0: 0.0, 2.0: 0.0}, physical_name="above"),
    ]
    state = structured_pre_pass(ents, point_tolerance=1e-3)
    # Expect ONE cohort with three sub-pieces (left-only, overlap, right-only).
    assert len(state.cohort_entities) == 1
    # The cohort entity's compound should hold three solids.
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    ce = state.cohort_entities[0]
    exp = TopExp_Explorer(ce.compound, TopAbs_SOLID)
    n_solids = 0
    while exp.More():
        n_solids += 1
        exp.Next()
    assert n_solids == 3
