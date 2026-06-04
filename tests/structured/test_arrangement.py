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
