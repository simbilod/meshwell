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


def test_decompose_cohorts_upgrades_touching_polyprism_to_neighbour():
    from meshwell.polyprism import PolyPrism
    from meshwell.structured.cohort_neighbour import CohortNeighbourUnstructured
    from meshwell.structured.pipeline import structured_pre_pass

    cohort = PolyPrism(
        _rect(0, 0, 10, 10),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="cohort",
        structured=True,
        mesh_order=3.0,
    )
    below = PolyPrism(
        _rect(0, 0, 10, 10),
        {-1.0: 0.0, 0.0: 0.0},
        physical_name="below",
        mesh_order=5.0,
    )
    above = PolyPrism(
        _rect(0, 0, 10, 10),
        {1.0: 0.0, 2.0: 0.0},
        physical_name="above",
        mesh_order=5.0,
    )
    state = structured_pre_pass([cohort, below, above], point_tolerance=1e-3)
    by_name = {
        e.physical_name: e for e in state.entities_out if hasattr(e, "physical_name")
    }
    assert isinstance(by_name[("below",)], CohortNeighbourUnstructured)
    assert isinstance(by_name[("above",)], CohortNeighbourUnstructured)
    assert by_name[("below",)].cohort_adjacency == [(0, 0.0)]
    assert by_name[("above",)].cohort_adjacency == [(0, 1.0)]


def test_cohort_neighbour_overrides_instanciate_occ_to_use_registries():
    """The face-registry branch lives on the subclass, not PolyPrism."""
    # Verify PolyPrism no longer has the class-level FACE registry.
    from meshwell.polyprism import PolyPrism
    from meshwell.structured.cohort_neighbour import CohortNeighbourUnstructured

    assert not hasattr(PolyPrism, "_cohort_face_registries")
    assert not hasattr(PolyPrism, "_set_cohort_face_registries")

    # CohortNeighbourUnstructured does.
    assert hasattr(CohortNeighbourUnstructured, "_cohort_face_registries")
    assert hasattr(CohortNeighbourUnstructured, "_set_cohort_face_registries")


def test_polyprism_instanciate_occ_no_longer_reads_cohort_adjacency():
    """PolyPrism.instanciate_occ has no special-case for cohort adjacency.

    After the cleanup, the face-registry routing lives entirely in
    CohortNeighbourUnstructured.instanciate_occ.
    """
    import inspect

    from meshwell.polyprism import PolyPrism

    src = inspect.getsource(PolyPrism.instanciate_occ)
    # No reference to _cohort_adjacency or _cohort_face_registries.
    assert "_cohort_adjacency" not in src
    assert "_cohort_face_registries" not in src
