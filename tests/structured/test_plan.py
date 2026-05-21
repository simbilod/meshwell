"""Tests for meshwell.structured.plan."""
from __future__ import annotations

from shapely.geometry import Polygon


def _square(x=0, y=0, w=1, h=1) -> Polygon:
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def _structured(polygon, buffers, n_layers, name, mesh_order=1.0):
    """Test helper: build a structured PolyPrism with a spec attached."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    return PolyPrism(
        polygons=polygon,
        buffers=buffers,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=n_layers)],
        physical_name=name,
        mesh_order=mesh_order,
    )


def test_gather_filters_structured_entities():
    from meshwell.polyprism import PolyPrism
    from meshwell.structured.plan import gather_structured_entities

    s = _structured(_square(), {0.0: 0.0, 1.0: 0.0}, [3], "s")
    u = PolyPrism(
        polygons=_square(2, 2),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="u",
    )

    pairs = gather_structured_entities([u, s])
    assert len(pairs) == 1
    entity, spec, source_idx = pairs[0]
    assert entity is s
    assert spec.n_layers == [3]
    assert source_idx == 1  # s was at index 1 in the input list


def test_gather_returns_empty_when_no_structured_entities():
    from meshwell.polyprism import PolyPrism
    from meshwell.structured.plan import gather_structured_entities

    u = PolyPrism(
        polygons=_square(),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="u",
    )
    assert gather_structured_entities([u]) == []


def test_gather_preserves_source_indices_across_mixed_list():
    from meshwell.polyprism import PolyPrism
    from meshwell.structured.plan import gather_structured_entities

    u1 = PolyPrism(polygons=_square(), buffers={0.0: 0.0, 1.0: 0.0}, physical_name="u1")
    s1 = _structured(_square(2, 0), {0.0: 0.0, 1.0: 0.0}, [3], "s1")
    u2 = PolyPrism(
        polygons=_square(4, 0), buffers={0.0: 0.0, 1.0: 0.0}, physical_name="u2"
    )
    s2 = _structured(_square(6, 0), {0.0: 0.0, 2.0: 0.0}, [4], "s2")

    pairs = gather_structured_entities([u1, s1, u2, s2])
    assert [p[2] for p in pairs] == [1, 3]
    assert [p[0].physical_name for p in pairs] == [("s1",), ("s2",)]


def test_expand_single_interval():
    from meshwell.structured.plan import expand_to_slabs, gather_structured_entities

    s = _structured(_square(), {0.0: 0.0, 1.5: 0.0}, [4], "s", mesh_order=2.0)
    pairs = gather_structured_entities([s])
    slabs = expand_to_slabs(pairs)
    assert len(slabs) == 1
    slab = slabs[0]
    assert slab.zlo == 0.0
    assert slab.zhi == 1.5
    assert slab.physical_name == ("s",)
    assert slab.source_index == 0
    assert slab.z_interval_index == 0
    assert slab.mesh_order == 2.0


def test_expand_multi_interval():
    from meshwell.structured.plan import expand_to_slabs, gather_structured_entities

    s = _structured(_square(), {0.0: 0.0, 1.0: 0.0, 3.0: 0.0}, [2, 5], "s")
    pairs = gather_structured_entities([s])
    slabs = expand_to_slabs(pairs)
    assert len(slabs) == 2
    assert (slabs[0].zlo, slabs[0].zhi, slabs[0].z_interval_index) == (0.0, 1.0, 0)
    assert (slabs[1].zlo, slabs[1].zhi, slabs[1].z_interval_index) == (1.0, 3.0, 1)
    # Both refer to the same owning entity / source_index.
    assert slabs[0].source_index == slabs[1].source_index == 0


def test_expand_propagates_arc_settings():
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec
    from meshwell.structured.plan import expand_to_slabs, gather_structured_entities

    s = PolyPrism(
        polygons=_square(),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[3])],
        identify_arcs=True,
        min_arc_points=8,
        arc_tolerance=5e-4,
        physical_name="s",
    )
    pairs = gather_structured_entities([s])
    slabs = expand_to_slabs(pairs)
    assert len(slabs) == 1
    assert slabs[0].identify_arcs is True
    assert slabs[0].min_arc_points == 8
    assert slabs[0].arc_tolerance == 5e-4


def test_no_overlap_keeps_all_slabs():
    from meshwell.structured.plan import (
        expand_to_slabs,
        gather_structured_entities,
        validate_and_resolve_overlap,
    )

    s1 = _structured(_square(0, 0), {0.0: 0.0, 1.0: 0.0}, [3], "s1")
    s2 = _structured(_square(2, 0), {0.0: 0.0, 1.0: 0.0}, [3], "s2")  # disjoint xy
    slabs = expand_to_slabs(gather_structured_entities([s1, s2]))
    kept, overlaps = validate_and_resolve_overlap(slabs, entities=[s1, s2])
    assert len(kept) == 2
    assert overlaps == []


def test_valid_overlap_drops_loser_records_pair():
    """Same z-extent, same n_layers, footprints overlap: lower mesh_order wins."""
    from meshwell.structured.plan import (
        expand_to_slabs,
        gather_structured_entities,
        validate_and_resolve_overlap,
    )

    # Lower mesh_order (1.0) wins; higher (2.0) is dropped.
    s_lo = _structured(
        _square(0, 0, 4, 4), {0.0: 0.0, 1.0: 0.0}, [3], "lo", mesh_order=2.0
    )
    s_hi = _structured(
        _square(1, 1, 2, 2), {0.0: 0.0, 1.0: 0.0}, [3], "hi", mesh_order=1.0
    )
    slabs = expand_to_slabs(gather_structured_entities([s_lo, s_hi]))
    kept, overlaps = validate_and_resolve_overlap(slabs, entities=[s_lo, s_hi])
    # Only the winner (hi) survives.
    assert len(kept) == 1
    assert kept[0].physical_name == ("hi",)
    # The loser pair was recorded.
    assert len(overlaps) == 1
    op = overlaps[0]
    assert op.loser_source_index == 0  # s_lo's index in entities=
    assert op.z_extent == (0.0, 1.0)


def test_overlap_with_mismatched_z_extent_raises():
    """Footprints overlap but z-extents differ: Policy B rejects."""
    import pytest

    from meshwell.structured.plan import (
        expand_to_slabs,
        gather_structured_entities,
        validate_and_resolve_overlap,
    )
    from meshwell.structured.spec import StructuredOverlapError

    s_lo = _structured(_square(0, 0, 4, 4), {0.0: 0.0, 1.0: 0.0}, [3], "lo")
    s_hi = _structured(_square(1, 1, 2, 2), {0.5: 0.0, 1.5: 0.0}, [3], "hi")
    slabs = expand_to_slabs(gather_structured_entities([s_lo, s_hi]))
    with pytest.raises(StructuredOverlapError, match="z-extent"):
        validate_and_resolve_overlap(slabs, entities=[s_lo, s_hi])


def test_overlap_with_matching_z_but_mismatched_n_layers_raises():
    """Same z-extent, different n_layers: Policy B rejects."""
    import pytest

    from meshwell.structured.plan import (
        expand_to_slabs,
        gather_structured_entities,
        validate_and_resolve_overlap,
    )
    from meshwell.structured.spec import StructuredOverlapError

    s_a = _structured(_square(0, 0, 4, 4), {0.0: 0.0, 1.0: 0.0}, [3], "a")
    s_b = _structured(_square(1, 1, 2, 2), {0.0: 0.0, 1.0: 0.0}, [5], "b")  # 5 vs 3
    slabs = expand_to_slabs(gather_structured_entities([s_a, s_b]))
    with pytest.raises(StructuredOverlapError, match="n_layers"):
        validate_and_resolve_overlap(slabs, entities=[s_a, s_b])


def test_face_partition_no_neighbours_single_piece():
    """No other entities touching the slab's z-planes: partition is one piece."""
    from meshwell.structured.plan import (
        compute_face_partition,
        expand_to_slabs,
        gather_structured_entities,
    )

    s = _structured(_square(0, 0, 4, 4), {0.0: 0.0, 1.0: 0.0}, [3], "s")
    slabs = expand_to_slabs(gather_structured_entities([s]))
    compute_face_partition(slabs, entities=[s])
    assert len(slabs[0].face_partition) == 1
    # Single piece equals the footprint (within shapely's equality tolerance).
    assert slabs[0].face_partition[0].equals(slabs[0].footprint)


def test_face_partition_with_neighbour_on_top_plane():
    """A non-structured prism touching the slab's top z-plane partitions the top."""
    import pytest

    from meshwell.polyprism import PolyPrism
    from meshwell.structured.plan import (
        compute_face_partition,
        expand_to_slabs,
        gather_structured_entities,
    )

    s = _structured(_square(0, 0, 4, 4), {0.0: 0.0, 1.0: 0.0}, [3], "s")
    # Non-structured neighbour sits on top of s (its zmin == s.zhi == 1.0),
    # covering the left half of s's footprint.
    n = PolyPrism(
        polygons=_square(0, 0, 2, 4),
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="n",
    )
    slabs = expand_to_slabs(gather_structured_entities([s, n]))
    compute_face_partition(slabs, entities=[s, n])
    # Slab partition has 2 pieces: covered (xy left half) and uncovered (xy right half).
    assert len(slabs[0].face_partition) == 2
    areas = sorted(p.area for p in slabs[0].face_partition)
    # Each piece has area 8 (2 wide x 4 tall).
    assert areas == pytest.approx([8.0, 8.0])


def test_build_plan_empty_entities():
    from meshwell.structured.plan import build_plan

    plan = build_plan([])
    assert plan.slabs == ()
    assert plan.z_planes == ()
    assert plan.overlaps == ()


def test_build_plan_no_structured_entities():
    from meshwell.polyprism import PolyPrism
    from meshwell.structured.plan import build_plan

    u = PolyPrism(polygons=_square(), buffers={0.0: 0.0, 1.0: 0.0}, physical_name="u")
    plan = build_plan([u])
    assert plan.slabs == ()
    assert plan.z_planes == ()
    assert plan.overlaps == ()


def test_build_plan_simple_structured_only():
    from meshwell.structured.plan import build_plan

    s = _structured(_square(), {0.0: 0.0, 1.0: 0.0, 2.5: 0.0}, [3, 4], "s")
    plan = build_plan([s])
    assert len(plan.slabs) == 2
    assert plan.z_planes == (0.0, 1.0, 2.5)
    assert plan.overlaps == ()
    # Both slabs get a single-piece partition (no neighbours).
    assert all(len(slab.face_partition) == 1 for slab in plan.slabs)


def test_build_plan_returns_frozen():
    """StructuredPlan is frozen; reassigning fields raises."""
    import pytest

    from meshwell.structured.plan import build_plan

    plan = build_plan([])
    with pytest.raises((AttributeError, TypeError)):
        plan.slabs = ()  # frozen dataclass rejects reassignment


def test_face_partition_overlapping_neighbours_common_refinement():
    """Phase 5(d): overlapping neighbours at slab top produce a 3-piece partition.

    (A-only, AB-overlap, B-only), not 1 piece.
    """
    from meshwell.polyprism import PolyPrism
    from meshwell.structured.plan import (
        compute_face_partition,
        expand_to_slabs,
        gather_structured_entities,
    )

    s = _structured(_square(0, 0, 4, 4), {0.0: 0.0, 1.0: 0.0}, [2], "s")
    a = PolyPrism(
        polygons=_square(0, 0, 3, 4),
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="a",
    )
    b = PolyPrism(
        polygons=_square(1, 0, 3, 4),
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="b",
    )
    slabs = expand_to_slabs(gather_structured_entities([s, a, b]))
    compute_face_partition(slabs, entities=[s, a, b])
    # 3 pieces: x in [0,1] (A only), x in [1,3] (AB), x in [3,4] (B only)
    assert len(slabs[0].face_partition) == 3


def test_partition_fixed_point_cap_is_module_constant():
    """The cap must be a module-level int so tests can monkeypatch it."""
    import meshwell.structured.plan as plan_mod

    assert hasattr(plan_mod, "_PARTITION_FIXED_POINT_CAP")
    assert isinstance(plan_mod._PARTITION_FIXED_POINT_CAP, int)
    assert plan_mod._PARTITION_FIXED_POINT_CAP >= 4


def test_structured_slabs_touching_z_returns_zlo_zhi_matches():
    """A slab is z-touching if its zlo or zhi equals the query z (within tol)."""
    from shapely.geometry import Polygon

    from meshwell.structured.plan import _structured_slabs_touching_z
    from meshwell.structured.spec import Slab

    s_lo = Slab(
        footprint=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        zlo=0.0,
        zhi=1.0,
        physical_name=("A",),
        source_index=0,
        z_interval_index=0,
        mesh_order=1.0,
    )
    s_hi = Slab(
        footprint=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        zlo=1.0,
        zhi=2.0,
        physical_name=("B",),
        source_index=1,
        z_interval_index=0,
        mesh_order=1.0,
    )
    s_far = Slab(
        footprint=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        zlo=5.0,
        zhi=6.0,
        physical_name=("C",),
        source_index=2,
        z_interval_index=0,
        mesh_order=1.0,
    )

    # z=1.0 should match s_lo.zhi and s_hi.zlo, not s_far.
    result = _structured_slabs_touching_z(1.0, [s_lo, s_hi, s_far], skip_slab_ids=set())
    names = {s.physical_name[0] for s in result}
    assert names == {"A", "B"}

    # skip_slab_ids filters out by id().
    result2 = _structured_slabs_touching_z(
        1.0, [s_lo, s_hi, s_far], skip_slab_ids={id(s_lo)}
    )
    names2 = {s.physical_name[0] for s in result2}
    assert names2 == {"B"}
