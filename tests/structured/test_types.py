from shapely.geometry import Polygon

from meshwell.structured.types import (
    Cohort,
    ShapeKey,
    StructuredSlab,
    SubPiece,
)

SQ = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


def test_structured_slab_is_frozen():
    s = StructuredSlab(
        source_index=0,
        footprint=SQ,
        zlo=0.0,
        zhi=1.0,
        mesh_order=1.0,
        mesh_bool=True,
        physical_name=("a",),
        identify_arcs=True,
        arc_tolerance=1e-3,
        min_arc_points=4,
    )
    assert s.zlo == 0.0
    import dataclasses

    assert dataclasses.is_dataclass(s)
    # frozen → assignment raises
    import pytest

    with pytest.raises(dataclasses.FrozenInstanceError):
        s.zlo = 5.0


def test_cohort_default_z_planes_sorted():
    s1 = StructuredSlab(0, SQ, 0.0, 1.0, 1.0, True, ("a",), True, 1e-3, 4)
    s2 = StructuredSlab(1, SQ, 1.0, 2.0, 1.0, True, ("b",), True, 1e-3, 4)
    c = Cohort(slabs=(s1, s2), z_planes=(0.0, 1.0, 2.0))
    assert c.z_planes == (0.0, 1.0, 2.0)
    assert c.zmin == 0.0
    assert c.zmax == 2.0


def test_subpiece_carries_source_indices():
    sp = SubPiece(
        cohort_index=0,
        z_interval=(0.0, 1.0),
        sub_polygon=SQ,
        source_slab_indices=(0, 3),
    )
    assert sp.source_slab_indices == (0, 3)


def test_shape_key_is_hashable():
    k = ShapeKey(tshape_id=12345, orientation=0)
    {k: "value"}  # noqa: B018 — must be hashable


def test_arrangement_is_hashable_and_holds_polygons():
    from shapely.geometry import Polygon

    from meshwell.structured.types import Arrangement

    p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    p2 = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
    arr = Arrangement(cohort_index=0, polygons=(p1, p2))
    assert arr.cohort_index == 0
    assert len(arr.polygons) == 2
    # frozen → hashable, usable as dict key
    {arr: 1}  # noqa: B018 — must be hashable
    # identity contract: the polygons stored are the exact objects passed in
    assert arr.polygons[0] is p1
    assert arr.polygons[1] is p2
