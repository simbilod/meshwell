"""Tests for the simplified decompose_cohorts contract.

After dropping the CohortNeighbourUnstructured upgrade,
decompose_cohorts returns the unstructured-entity list unchanged
(same Python objects). BOP fragment + AABB rescue handle the
cohort↔unstructured interface detection downstream.
"""
from __future__ import annotations

from shapely.geometry import Polygon

from meshwell.polyprism import PolyPrism
from meshwell.structured.decompose import decompose_cohorts
from meshwell.structured.types import Cohort, StructuredSlab


def _rect(x1, y1, x2, y2):
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def _slab(source_index, footprint, mesh_order):
    return StructuredSlab(
        source_index=source_index,
        zlo=0.0,
        zhi=1.0,
        footprint=footprint,
        mesh_order=mesh_order,
        mesh_bool=True,
        physical_name=("slab",),
        identify_arcs=False,
        min_arc_points=24,
        arc_tolerance=1e-4,
    )


def test_decompose_cohorts_returns_touching_polyprism_unchanged():
    """A PolyPrism touching a cohort z-plane is returned by identity."""
    slab_poly = _rect(0, 0, 10, 10)
    cohort = Cohort(
        slabs=(_slab(0, slab_poly, mesh_order=3.0),),
        z_planes=(0.0, 1.0),
    )
    neighbour = PolyPrism(
        polygons=_rect(0, 0, 10, 10),
        buffers={-1.0: 0.0, 0.0: 0.0},
        physical_name="neighbour",
        mesh_order=5.0,
    )
    _subs_list, pre_cut = decompose_cohorts([cohort], [neighbour])
    assert (
        pre_cut[0] is neighbour
    ), "touching unstructured PolyPrism must pass through unchanged"


def test_decompose_cohorts_returns_non_touching_polyprism_unchanged():
    """A PolyPrism not touching any cohort z-plane is returned by identity."""
    slab_poly = _rect(0, 0, 10, 10)
    cohort = Cohort(
        slabs=(_slab(0, slab_poly, mesh_order=3.0),),
        z_planes=(0.0, 1.0),
    )
    far_neighbour = PolyPrism(
        polygons=_rect(0, 0, 10, 10),
        buffers={5.0: 0.0, 6.0: 0.0},
        physical_name="far",
        mesh_order=5.0,
    )
    _subs_list, pre_cut = decompose_cohorts([cohort], [far_neighbour])
    assert pre_cut[0] is far_neighbour


def test_decompose_cohorts_still_produces_cohort_subpieces():
    """Cohort sub-piece decomposition is unaffected by the simplification."""
    slab_poly = _rect(0, 0, 10, 10)
    cohort = Cohort(
        slabs=(_slab(0, slab_poly, mesh_order=3.0),),
        z_planes=(0.0, 1.0),
    )
    subs_list, _ = decompose_cohorts([cohort], [])
    assert len(subs_list) == 1
    assert len(subs_list[0]) == 1
    assert subs_list[0][0].sub_polygon.equals(slab_poly)
