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


def test_slab_meta_synthetic_solid_round_trip():
    from meshwell.structured.types import ShapeKey, SlabMeta

    meta = SlabMeta(
        slab_index=3,
        physical_name=("metal", "port__1", "a__name_b", "c__src_d"),
        bot_face_key=ShapeKey(1, 0),
        top_face_key=ShapeKey(2, 0),
        lateral_face_keys=(ShapeKey(3, 0), ShapeKey(4, 0)),
    )
    solid_name = meta.to_synthetic_solid_name(cohort_index=1, sub_index=2)
    assert solid_name == (
        "__cohort_1__slab_2__name_metal|port__1|a__name_b|c__src_d__src_3"
    )
    # Backward-compatible trailing integer is slab_index (3), not sub_index (2).
    assert int(solid_name.rsplit("_", 1)[1]) == 3

    parsed = SlabMeta.parse_synthetic_solid_name(solid_name)
    assert parsed == (
        "__cohort_1__slab_2",
        3,
        ("metal", "port__1", "a__name_b", "c__src_d"),
    )

    # Physical names containing literal '|' and '%' characters round-trip losslessly:
    meta_pipe = SlabMeta(
        slab_index=7,
        physical_name=("port|1", "metal%a", "a|b%c"),
        bot_face_key="b",
        top_face_key="t",
        lateral_face_keys=(),
    )
    pipe_name = meta_pipe.to_synthetic_solid_name(cohort_index=0, sub_index=9)
    assert SlabMeta.parse_synthetic_solid_name(pipe_name) == (
        "__cohort_0__slab_9",
        7,
        ("port|1", "metal%a", "a|b%c"),
    )

    # Legacy bare format fallback:
    assert SlabMeta.parse_synthetic_solid_name("__cohort_2__slab_5") == (
        "__cohort_2__slab_5",
        5,
        (),
    )

    # Non-solid names must return None:
    assert SlabMeta.parse_synthetic_solid_name("__cohort_0__slab_0__bot") is None
    assert SlabMeta.parse_synthetic_solid_name("__cohort_0__slab_0__lat_1") is None
    assert SlabMeta.parse_synthetic_solid_name("__cohort_0__slab_0___None") is None
    assert SlabMeta.parse_synthetic_solid_name("regular_group") is None


def test_slab_meta_synthetic_face_round_trip():
    from meshwell.structured.types import SlabMeta

    for role in ("bot", "top", "lat_0", "lat_15"):
        fname = SlabMeta.to_synthetic_face_name(0, 4, role)
        assert SlabMeta.parse_synthetic_face_name(fname) == (
            "__cohort_0__slab_4",
            role,
        )

    assert (
        SlabMeta.parse_synthetic_face_name("__cohort_0__slab_4__name_a__src_0") is None
    )
    assert SlabMeta.parse_synthetic_face_name("__cohort_0__slab_4___None") is None


def test_slab_meta_from_synthetic_groups():
    from meshwell.structured.types import SlabMeta

    solid_groups = {
        "__cohort_0__slab_1__name_wg|core__src_0": 10,
        "__cohort_0__slab_2": 11,  # legacy bare format
    }
    face_groups = {
        "__cohort_0__slab_1__bot": 101,
        "__cohort_0__slab_1__top": 102,
        "__cohort_0__slab_1__lat_2": 105,
        "__cohort_0__slab_1__lat_0": 103,
        "__cohort_0__slab_1__lat_1": 104,
        "__cohort_0__slab_2__bot": 201,
        "__cohort_0__slab_2__top": 202,
        "__cohort_0__slab_2__lat_0": 203,
    }
    slab_meta, face_tags, solid_tags = SlabMeta.from_synthetic_groups(
        solid_groups=solid_groups,
        face_groups=face_groups,
        fallback_names_by_vol={11: ["legacy_slab"]},
    )

    assert solid_tags == {"__cohort_0__slab_1": 10, "__cohort_0__slab_2": 11}
    m1 = slab_meta["__cohort_0__slab_1"]
    assert m1.slab_index == 0
    assert m1.physical_name == ("wg", "core")
    assert face_tags[m1.bot_face_key] == 101
    assert face_tags[m1.top_face_key] == 102
    assert [face_tags[k] for k in m1.lateral_face_keys] == [103, 104, 105]

    m2 = slab_meta["__cohort_0__slab_2"]
    assert m2.slab_index == 2
    assert m2.physical_name == ("legacy_slab",)
