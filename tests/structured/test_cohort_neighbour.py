"""Tests for CohortNeighbourUnstructured."""
from __future__ import annotations

from shapely.geometry import MultiPolygon, Polygon

from meshwell.polyprism import PolyPrism
from meshwell.structured.cohort_neighbour import CohortNeighbourUnstructured


def _rect(x1, y1, x2, y2):
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def test_cohort_neighbour_is_polyprism_subclass():
    assert issubclass(CohortNeighbourUnstructured, PolyPrism)


def test_cohort_neighbour_from_polyprism_copies_attributes():
    original = PolyPrism(
        _rect(0, 0, 10, 10),
        {-1.0: 0.0, 0.0: 0.0},
        physical_name="cladding",
        mesh_order=5.0,
        identify_arcs=True,
    )
    tiles = (_rect(0, 0, 5, 10), _rect(5, 0, 10, 10))
    nbr = CohortNeighbourUnstructured.from_polyprism(
        original=original,
        cohort_adjacency=[(0, 0.0)],
        tile_polygons=tiles,
    )
    assert nbr.physical_name == ("cladding",)
    assert nbr.mesh_order == 5.0
    assert nbr.identify_arcs is True
    assert nbr.zmin == -1.0
    assert nbr.zmax == 0.0
    assert nbr.cohort_adjacency == [(0, 0.0)]
    assert nbr.tile_polygons == tiles
    # polygons becomes MultiPolygon of tiles.
    assert isinstance(nbr.polygons, MultiPolygon)
    assert len(list(nbr.polygons.geoms)) == 2


def test_cohort_neighbour_single_tile_keeps_polygon_not_multipolygon():
    original = PolyPrism(
        _rect(0, 0, 10, 10),
        {-1.0: 0.0, 0.0: 0.0},
        physical_name="cladding",
        mesh_order=5.0,
    )
    one_tile = (_rect(0, 0, 10, 10),)
    nbr = CohortNeighbourUnstructured.from_polyprism(
        original=original,
        cohort_adjacency=[(0, 0.0)],
        tile_polygons=one_tile,
    )
    # Single-tile case: polygons is a Polygon, not MultiPolygon.
    assert isinstance(nbr.polygons, Polygon)


def test_cohort_neighbour_inherits_instanciate_occ_until_overridden():
    """Until Task 7, instanciate_occ inherits from PolyPrism unchanged."""
    original = PolyPrism(
        _rect(0, 0, 10, 10),
        {-1.0: 0.0, 0.0: 0.0},
        physical_name="cladding",
        mesh_order=5.0,
    )
    nbr = CohortNeighbourUnstructured.from_polyprism(
        original=original,
        cohort_adjacency=[(0, 0.0)],
        tile_polygons=(_rect(0, 0, 10, 10),),
    )
    # PolyPrism.instanciate_occ should still work because we inherit it.
    shape = nbr.instanciate_occ()
    assert shape is not None
