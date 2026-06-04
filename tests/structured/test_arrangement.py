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


def test_pre_cut_returns_polygons_by_identity():
    from shapely import equals_exact
    from shapely.geometry import MultiPolygon

    from meshwell.structured.decompose import arrangement_pre_cut_for_entity

    cohort = Cohort(slabs=(_slab(0, _rect(0, 0, 6, 10)),), z_planes=(0.0, 1.0))
    neighbour_poly = _rect(0, 0, 10, 10)
    arr = build_cohort_arrangement(
        cohort_index=0,
        cohort=cohort,
        adjacent_unstructured=[neighbour_poly.boundary],
    )
    out = arrangement_pre_cut_for_entity(arr, neighbour_poly)
    # neighbour footprint covers both arrangement polygons (left half x∈[0,6]
    # and right half x∈[6,10]). Both must be present in the MultiPolygon.
    # Shapely 2.x's MultiPolygon.geoms returns fresh Polygon wrappers, so we
    # check bit-exact geometric equality (tolerance=0) rather than Python
    # `is` identity. The underlying GEOS coordinate sequences are shared
    # by reference, so vertex coordinates match exactly.
    assert isinstance(out, MultiPolygon)
    members = list(out.geoms)
    assert len(members) == 2
    for m in members:
        assert any(
            equals_exact(m, p, tolerance=0.0) for p in arr.polygons
        ), "pre-cut member must be bit-exactly equal to an arrangement polygon"


def test_pre_cut_returns_single_polygon_when_only_one_inside():
    from meshwell.structured.decompose import arrangement_pre_cut_for_entity

    cohort = Cohort(slabs=(_slab(0, _rect(0, 0, 10, 10)),), z_planes=(0.0, 1.0))
    # neighbour fits inside cohort entirely — no extra arrangement linework
    # from neighbour; arrangement has one polygon, all inside neighbour.
    neighbour_poly = _rect(2, 2, 8, 8)
    arr = build_cohort_arrangement(
        cohort_index=0,
        cohort=cohort,
        adjacent_unstructured=[neighbour_poly.boundary],
    )
    out = arrangement_pre_cut_for_entity(arr, neighbour_poly)
    # Inside neighbour, only the middle piece (2..8 x 2..8) belongs.
    # The arrangement has 9 pieces (3x3 grid via the 2,8 cuts).
    # arrangement_pre_cut returns the one whose centroid is inside neighbour.
    from shapely.geometry import Polygon as P

    assert isinstance(out, P)
    assert out.representative_point().within(neighbour_poly)
    # Single-Polygon return path preserves Python `is` identity (no
    # MultiPolygon wrapping). Verify the contract holds where possible.
    assert any(out is p for p in arr.polygons)


def test_pre_cut_returns_entity_unchanged_when_no_polygons_inside():
    from meshwell.structured.decompose import arrangement_pre_cut_for_entity

    cohort = Cohort(slabs=(_slab(0, _rect(0, 0, 10, 10)),), z_planes=(0.0, 1.0))
    arr = build_cohort_arrangement(
        cohort_index=0, cohort=cohort, adjacent_unstructured=[]
    )
    # neighbour is far away — arrangement has no polygons inside it.
    far_neighbour = _rect(100, 100, 110, 110)
    out = arrangement_pre_cut_for_entity(arr, far_neighbour)
    assert out is far_neighbour


def test_decompose_cohorts_returns_identity_polygons_across_cohort_and_unstructured():
    """End-to-end identity contract for cohort/unstructured polygons.

    Cohort sub-piece polygon IS the same object as the unstructured pre-cut
    polygon for the same XY region.

    This is the core invariant the refactor establishes. Before, cohort
    sub-pieces and unstructured pre-cuts came from independent polygonize
    calls and were never the same Python object even when geometrically
    identical.
    """
    from meshwell.polyprism import PolyPrism
    from meshwell.structured.decompose import decompose_cohorts

    # Cohort: structured slab at z∈[0,1].
    slab_poly = _rect(0, 0, 10, 10)
    cohort = Cohort(
        slabs=(_slab(0, slab_poly, mesh_order=3.0),),
        z_planes=(0.0, 1.0),
    )
    # Unstructured neighbour at z∈[-1,0] sharing z=0 with cohort, same XY.
    neighbour = PolyPrism(
        polygons=_rect(0, 0, 10, 10),
        buffers={-1.0: 0.0, 0.0: 0.0},
        physical_name="neighbour",
        mesh_order=5.0,
    )
    subs_list, pre_cut = decompose_cohorts([cohort], [neighbour])
    cohort_subs = subs_list[0]
    assert len(cohort_subs) == 1
    sub_poly = cohort_subs[0].sub_polygon

    # Unstructured pre-cut: same geometry as cohort sub-piece.
    pre_cut_neighbour = pre_cut[0]
    assert pre_cut_neighbour.polygons is sub_poly, (
        "pre-cut polygon must be the SAME Python object as the cohort "
        "sub-piece polygon (identity, not just equality)"
    )
