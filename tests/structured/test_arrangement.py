"""Tests for the cohort-global arrangement builder."""
from __future__ import annotations

from shapely.geometry import Polygon

from meshwell.structured.decompose import build_cohort_arrangement
from meshwell.structured.types import Arrangement, Cohort, StructuredSlab


def _slab(idx, poly, zlo=0.0, zhi=1.0, mesh_order=1.0, mesh_bool=True):
    return StructuredSlab(
        source_index=idx,
        footprint=poly,
        zlo=zlo,
        zhi=zhi,
        mesh_order=mesh_order,
        mesh_bool=mesh_bool,
        physical_name=("x",),
        identify_arcs=False,
        arc_tolerance=1e-3,
        min_arc_points=4,
    )


def _rect(x1, y1, x2, y2):
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def test_arrangement_single_slab_yields_one_polygon():
    cohort = Cohort(slabs=(_slab(0, _rect(0, 0, 10, 10)),), z_planes=(0.0, 1.0))
    arr = build_cohort_arrangement(
        cohort_index=0, cohort=cohort, adjacent_unstructured=[]
    )
    assert isinstance(arr, Arrangement)
    assert arr.cohort_index == 0
    assert len(arr.polygons) == 1


def test_arrangement_two_overlapping_slabs_yields_three_pieces():
    # Two overlapping squares -> 3 pieces (left-only, overlap, right-only).
    cohort = Cohort(
        slabs=(
            _slab(0, _rect(0, 0, 6, 10)),
            _slab(1, _rect(4, 0, 10, 10)),
        ),
        z_planes=(0.0, 1.0),
    )
    arr = build_cohort_arrangement(
        cohort_index=0, cohort=cohort, adjacent_unstructured=[]
    )
    assert len(arr.polygons) == 3


def test_arrangement_includes_adjacent_unstructured_cuts():
    # One cohort slab + one adjacent unstructured rectangle whose boundary
    # passes through the cohort footprint -> the arrangement subdivides.
    cohort = Cohort(slabs=(_slab(0, _rect(0, 0, 10, 10)),), z_planes=(0.0, 1.0))
    # Unstructured neighbour spans 4..15 in x — boundary x=4 cuts cohort.
    neighbour_boundary = _rect(4, -5, 15, 15).boundary
    arr = build_cohort_arrangement(
        cohort_index=0,
        cohort=cohort,
        adjacent_unstructured=[neighbour_boundary],
    )
    # Cohort split at x=4 -> at least 2 pieces inside cohort + extras
    # outside cohort that lie within the union of all linework.
    cohort_pieces = [
        p for p in arr.polygons if p.representative_point().within(_rect(0, 0, 10, 10))
    ]
    assert len(cohort_pieces) == 2


def test_subpieces_for_interval_emits_one_per_owned_polygon():
    from meshwell.structured.decompose import arrangement_subpieces_for_interval

    cohort = Cohort(slabs=(_slab(0, _rect(0, 0, 10, 10)),), z_planes=(0.0, 1.0))
    arr = build_cohort_arrangement(
        cohort_index=0, cohort=cohort, adjacent_unstructured=[]
    )
    subs = arrangement_subpieces_for_interval(arr, cohort, 0.0, 1.0)
    assert len(subs) == 1
    assert subs[0].cohort_index == 0
    assert subs[0].z_interval == (0.0, 1.0)
    assert subs[0].source_slab_indices == (0,)
    # IDENTITY contract: subpiece's polygon IS arrangement's polygon
    assert subs[0].sub_polygon is arr.polygons[0]


def test_subpieces_filtered_out_when_owner_is_none():
    # Cohort has slab S at y∈[0,5]. Arrangement also contains a piece at
    # y∈[5,10] (introduced by adjacent unstructured cut) which has no owner
    # in this cohort. That piece must be dropped from sub-pieces.
    from meshwell.structured.decompose import arrangement_subpieces_for_interval

    cohort = Cohort(slabs=(_slab(0, _rect(0, 0, 10, 5)),), z_planes=(0.0, 1.0))
    neighbour = _rect(0, 0, 10, 10).boundary
    arr = build_cohort_arrangement(
        cohort_index=0,
        cohort=cohort,
        adjacent_unstructured=[neighbour],
    )
    subs = arrangement_subpieces_for_interval(arr, cohort, 0.0, 1.0)
    # Only one piece survives the owner filter (y∈[0,5]).
    assert len(subs) == 1
    assert subs[0].sub_polygon.representative_point().y < 5.0


def test_arrangement_edge_defaults_to_empty():
    """New canonical_edges / edge_by_vertex_pair default fields exist.

    Both fields start empty so existing callers / tests are unaffected.
    """
    from meshwell.structured.types import Arrangement, ArrangementEdge

    arr = Arrangement(cohort_index=0, polygons=())
    assert arr.canonical_edges == ()
    assert arr.edge_by_vertex_pair == {}

    edge = ArrangementEdge(
        vertex_keys=((0, 0, 0), (1, 0, 0)),
        z=0.0,
        segments=(),
        is_closed=False,
    )
    assert edge.vertex_keys[0] != edge.vertex_keys[-1]
    assert edge.is_closed is False


def test_build_cohort_arrangement_populates_canonical_edges():
    """build_cohort_arrangement now produces an Arrangement whose canonical_edges tuple is populated.

    ``edge_by_vertex_pair`` indexes every open-edge consecutive vertex pair.
    """
    cohort = Cohort(
        slabs=(
            _slab(0, _rect(0, 0, 6, 10)),
            _slab(1, _rect(4, 0, 10, 10)),
        ),
        z_planes=(0.0, 1.0),
    )
    arr = build_cohort_arrangement(
        cohort_index=0,
        cohort=cohort,
        adjacent_unstructured=[],
        point_tolerance=1e-3,
    )
    assert len(arr.canonical_edges) > 0
    # Lookup non-empty (rectangles have no closed standalone components).
    assert len(arr.edge_by_vertex_pair) > 0
    # Each registered pair points to an existing canonical edge index.
    for ei in arr.edge_by_vertex_pair.values():
        assert 0 <= ei < len(arr.canonical_edges)
