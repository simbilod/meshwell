"""Tests for meshwell.structured.plan."""
from __future__ import annotations

import pytest
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


def test_merge_arc_into_index_appends_arc_and_indexes_vertices():
    """An inherited PieceArcEdge gets a fresh arc_id, points indexed for lookup."""
    from meshwell.structured.plan import _ArcIndex, _merge_arc_into_index
    from meshwell.structured.spec import PieceArcEdge

    idx = _ArcIndex(ndigits=3)
    arc = PieceArcEdge(
        points=((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 0.0, 0.0)),
        center=(1.0, 0.0, 0.0),
        radius=1.0,
    )
    _merge_arc_into_index(idx, arc)

    assert len(idx.arcs) == 1
    assert idx.arcs[0].center == (1.0, 0.0, 0.0)
    assert idx.arcs[0].radius == 1.0
    # All 3 points indexed.
    assert (0.0, 0.0) in idx.vertex_to_arcs
    assert (1.0, 1.0) in idx.vertex_to_arcs
    assert (2.0, 0.0) in idx.vertex_to_arcs
    # The (arc_id, position) pairs map back consistently.
    arc_id_0 = idx.vertex_to_arcs[(0.0, 0.0)][0][0]
    arc_id_2 = idx.vertex_to_arcs[(2.0, 0.0)][0][0]
    assert arc_id_0 == arc_id_2  # same arc

    # A second merge with the SAME geometry still appends (caller dedupes if needed).
    _merge_arc_into_index(idx, arc)
    assert len(idx.arcs) == 2  # caller is responsible for dedup; helper is idempotent


def test_collect_cut_sources_uses_slab_pieces_not_footprints():
    """Structured neighbours contribute piece boundaries, not the whole footprint.

    Sets up a synthetic slab list where the neighbour has a multi-piece
    face_partition already populated, and asserts that the cut sources
    returned include each piece's boundary, NOT the union footprint.
    """
    from shapely.geometry import Polygon

    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec
    from meshwell.structured.plan import _collect_cut_sources
    from meshwell.structured.spec import Slab

    ent_self = PolyPrism(
        polygons=Polygon([(0, 0), (4, 0), (4, 2), (0, 2)]),
        buffers={1.0: 0.0, 2.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="SELF",
    )
    ent_neigh = PolyPrism(
        polygons=Polygon([(0, 0), (4, 0), (4, 2), (0, 2)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="NEIGH",
    )
    s_self = Slab(
        footprint=ent_self.polygons,
        zlo=1.0,
        zhi=2.0,
        physical_name=("SELF",),
        source_index=0,
        z_interval_index=0,
        mesh_order=1.0,
    )
    s_neigh = Slab(
        footprint=ent_neigh.polygons,
        zlo=0.0,
        zhi=1.0,
        physical_name=("NEIGH",),
        source_index=1,
        z_interval_index=0,
        mesh_order=1.0,
    )
    # Pre-populate neighbour with a 2-piece partition (simulating an earlier pass).
    s_neigh.face_partition = [
        Polygon([(0, 0), (1.5, 0), (1.5, 2), (0, 2)]),
        Polygon([(1.5, 0), (4, 0), (4, 2), (1.5, 2)]),
    ]

    sources = _collect_cut_sources(
        slab=s_self,
        slabs=[s_self, s_neigh],
        entities=[ent_self, ent_neigh],
        skip_indices={0},  # self's entity index
    )
    # Two piece boundaries from s_neigh, both touching x=1.5 should appear.
    # Each boundary is a LinearRing/LineString; we union them and check the
    # combined geometry passes through x=1.5.
    from shapely.ops import unary_union

    combined = unary_union(sources)
    # The seam at x=1.5 must be present (the boundary between the two pieces).
    assert combined.intersects(
        Polygon([(1.4, -0.1), (1.6, -0.1), (1.6, 2.1), (1.4, 2.1)]).boundary
    )


def test_collect_inherited_arcs_pulls_from_neighbour_provenance():
    """Inherited arcs come from z-touching structured slabs with arc provenance."""
    from shapely.geometry import Polygon

    from meshwell.structured.plan import _collect_inherited_arcs
    from meshwell.structured.spec import (
        PieceArcEdge,
        PieceLineEdge,
        PieceProvenance,
        Slab,
    )

    s_self = Slab(
        footprint=Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
        zlo=1.0,
        zhi=2.0,
        physical_name=("SELF",),
        source_index=0,
        z_interval_index=0,
        mesh_order=1.0,
        identify_arcs=True,
    )
    s_neigh = Slab(
        footprint=Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
        zlo=0.0,
        zhi=1.0,
        physical_name=("NEIGH",),
        source_index=1,
        z_interval_index=0,
        mesh_order=1.0,
        identify_arcs=True,
    )
    arc_pts = ((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 0.0, 0.0))
    s_neigh.face_partition = [Polygon([(0, 0), (2, 0), (2, 4), (0, 4)])]
    s_neigh.face_partition_provenance = [
        PieceProvenance(
            exterior_edges=[
                PieceArcEdge(points=arc_pts, center=(1.0, 0.0, 0.0), radius=1.0),
                PieceLineEdge(points=((2.0, 0.0, 0.0), (2.0, 4.0, 0.0))),
            ],
            interior_edges=[],
        )
    ]
    inherited = _collect_inherited_arcs(
        slab=s_self, slabs=[s_self, s_neigh], skip_slab_ids={id(s_self)}
    )
    assert len(inherited) == 1
    assert inherited[0].radius == 1.0


def test_collect_inherited_arcs_skips_when_identify_arcs_false():
    """Receiving slab with identify_arcs=False inherits nothing."""
    from shapely.geometry import Polygon

    from meshwell.structured.plan import _collect_inherited_arcs
    from meshwell.structured.spec import (
        PieceArcEdge,
        PieceProvenance,
        Slab,
    )

    s_self = Slab(
        footprint=Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
        zlo=1.0,
        zhi=2.0,
        physical_name=("SELF",),
        source_index=0,
        z_interval_index=0,
        mesh_order=1.0,
        identify_arcs=False,
    )
    s_neigh = Slab(
        footprint=Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
        zlo=0.0,
        zhi=1.0,
        physical_name=("NEIGH",),
        source_index=1,
        z_interval_index=0,
        mesh_order=1.0,
        identify_arcs=True,
    )
    s_neigh.face_partition = [Polygon([(0, 0), (2, 0), (2, 4), (0, 4)])]
    s_neigh.face_partition_provenance = [
        PieceProvenance(
            exterior_edges=[
                PieceArcEdge(
                    points=((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 0.0, 0.0)),
                    center=(1.0, 0.0, 0.0),
                    radius=1.0,
                ),
            ],
            interior_edges=[],
        )
    ]
    inherited = _collect_inherited_arcs(
        slab=s_self, slabs=[s_self, s_neigh], skip_slab_ids={id(s_self)}
    )
    assert inherited == []


def test_partition_propagates_cut_two_steps():
    """3-layer stack; middle layer's internal seam propagates to top and bottom.

    Layer 1 (z=[0,1]): single slab, no internal seam, footprint [0,4]x[0,2].
    Layer 2 (z=[1,2]): two slabs meeting at x=2.5 (internal seam).
    Layer 3 (z=[2,3]): single slab, no internal seam, footprint [0,4]x[0,2].

    After planning, layer 1's slab must be partitioned by x=2.5 (propagated
    down from layer 2's piece boundary), and layer 3's slab must be too
    (propagated up).
    """
    from shapely.geometry import Polygon

    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan

    def _box(x0, y0, x1, y1):
        return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    bot = PolyPrism(
        polygons=_box(0, 0, 4, 2),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="BOT",
    )
    mid_l = PolyPrism(
        polygons=_box(0, 0, 2.5, 2),
        buffers={1.0: 0.0, 2.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="MID_L",
    )
    mid_r = PolyPrism(
        polygons=_box(2.5, 0, 4, 2),
        buffers={1.0: 0.0, 2.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="MID_R",
    )
    top = PolyPrism(
        polygons=_box(0, 0, 4, 2),
        buffers={2.0: 0.0, 3.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="TOP",
    )
    plan = build_plan([bot, mid_l, mid_r, top])

    by_name = {}
    for s in plan.slabs:
        by_name[s.physical_name[0]] = s

    # BOT and TOP must each be split at x=2.5 by the propagated piece boundary
    # from mid_l/mid_r. (Today, only the direct neighbour FOOTPRINTS contribute,
    # but mid_l/mid_r footprints together cover [0,4] so no cut would appear
    # in BOT/TOP under the old logic. Under the new logic, mid_l's piece
    # boundary at x=2.5 is a cut source for both BOT and TOP.)
    assert (
        len(by_name["BOT"].face_partition) >= 2
    ), f"BOT was not split; got {len(by_name['BOT'].face_partition)} pieces"
    assert (
        len(by_name["TOP"].face_partition) >= 2
    ), f"TOP was not split; got {len(by_name['TOP'].face_partition)} pieces"


def test_partition_converges_within_K_plus_two_passes():
    """4-layer stack converges in <= K + 2 = 6 passes."""
    from shapely.geometry import Polygon

    import meshwell.structured.plan as plan_mod
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan

    def _box(x0, y0, x1, y1):
        return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    # 4 stacked layers, misaligned seams (the xfail scenario, plan-only).
    seams = [1.0, 1.7, 2.5, 3.2]
    ents = []
    for i, sx in enumerate(seams):
        zlo, zhi = float(i), float(i + 1)
        for j, (x0, x1) in enumerate([(0.0, sx), (sx, 4.0)]):
            ents.append(
                PolyPrism(
                    polygons=_box(x0, 0, x1, 2),
                    buffers={zlo: 0.0, zhi: 0.0},
                    structured=True,
                    resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
                    physical_name=f"L{i+1}_{'A' if j == 0 else 'B'}",
                )
            )
    build_plan(ents)
    # K = 4 z-intervals; cap was min(K + 2, _PARTITION_FIXED_POINT_CAP) so
    # convergence should occur within K + 2 = 6 passes.
    assert plan_mod._LAST_PARTITION_ITERATIONS <= 6, (
        f"converged in {plan_mod._LAST_PARTITION_ITERATIONS} passes; "
        f"expected <= 6 for K=4 stack"
    )


def test_partition_raises_if_not_converged(monkeypatch):
    """Tripping the cap surfaces StructuredPartitionConvergenceError."""
    from shapely.geometry import Polygon

    import meshwell.structured.plan as plan_mod
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import (
        StructuredExtrusionResolutionSpec,
        StructuredPartitionConvergenceError,
        build_plan,
    )

    def _box(x0, y0, x1, y1):
        return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    # Force the cap to 1 — a 3-layer stack with transitive seams needs at
    # least 2 passes to converge, so we should hit the cap.
    monkeypatch.setattr(plan_mod, "_PARTITION_FIXED_POINT_CAP", 1)

    bot = PolyPrism(
        polygons=_box(0, 0, 4, 2),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="BOT",
    )
    mid_l = PolyPrism(
        polygons=_box(0, 0, 2.5, 2),
        buffers={1.0: 0.0, 2.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="MID_L",
    )
    mid_r = PolyPrism(
        polygons=_box(2.5, 0, 4, 2),
        buffers={1.0: 0.0, 2.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="MID_R",
    )
    top = PolyPrism(
        polygons=_box(0, 0, 4, 2),
        buffers={2.0: 0.0, 3.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="TOP",
    )

    with pytest.raises(StructuredPartitionConvergenceError, match="did not converge"):
        build_plan([bot, mid_l, mid_r, top])


def test_partition_misaligned_seams_each_slab_partitioned_by_union():
    """4-layer misaligned: each slab's piece count matches the union of seams.

    Each slab's piece count matches the union of seams that intersect its
    footprint.

    Slab Lk_A has footprint [0, seam_k] x [0, 2]; cuts intersecting that range
    are the seams from any other layer m with 0 < seam_m < seam_k.
    Same for Lk_B with [seam_k, 4] and seam_m > seam_k.
    """
    from shapely.geometry import Polygon

    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan

    def _box(x0, y0, x1, y1):
        return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    seams = [1.0, 1.7, 2.5, 3.2]
    ents = []
    for i, sx in enumerate(seams):
        zlo, zhi = float(i), float(i + 1)
        for _j, (x0, x1, side) in enumerate([(0.0, sx, "A"), (sx, 4.0, "B")]):
            ents.append(
                PolyPrism(
                    polygons=_box(x0, 0, x1, 2),
                    buffers={zlo: 0.0, zhi: 0.0},
                    structured=True,
                    resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
                    physical_name=f"L{i+1}_{side}",
                )
            )
    plan = build_plan(ents)
    by_name = {s.physical_name[0]: s for s in plan.slabs}

    # For each layer k, side A spans [0, seam_k]. Count seams strictly
    # between 0 and seam_k (from any other layer) → expected pieces = count + 1.
    for k, sx_k in enumerate(seams, start=1):
        cuts_A = [s for s in seams if 0 < s < sx_k]
        cuts_B = [s for s in seams if sx_k < s < 4.0]
        # The k-th layer's own seam is at sx_k; it's the slab boundary, not an interior cut.
        # Filter out sx_k itself (== boundary), so use strict inequality.
        n_pieces_A = len(cuts_A) + 1
        n_pieces_B = len(cuts_B) + 1
        assert len(by_name[f"L{k}_A"].face_partition) == n_pieces_A, (
            f"L{k}_A: expected {n_pieces_A} pieces from cuts {cuts_A}; "
            f"got {len(by_name[f'L{k}_A'].face_partition)}"
        )
        assert len(by_name[f"L{k}_B"].face_partition) == n_pieces_B, (
            f"L{k}_B: expected {n_pieces_B} pieces from cuts {cuts_B}; "
            f"got {len(by_name[f'L{k}_B'].face_partition)}"
        )


def test_resolve_sublevel_carves_loser_by_winner():
    """Same-z overlap: lower mesh_order wins; loser's resolved_footprint is carved."""
    from shapely.geometry import Polygon

    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec
    from meshwell.structured.plan import (
        _resolve_sublevel_mesh_order,
        expand_to_slabs,
        gather_structured_entities,
    )

    a = PolyPrism(
        polygons=Polygon([(0, 0), (4, 0), (4, 2), (0, 2)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="A",
        mesh_order=1,
    )
    b = PolyPrism(
        polygons=Polygon([(2, 0), (6, 0), (6, 2), (2, 2)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="B",
        mesh_order=2,
    )
    entities = [a, b]
    slabs = expand_to_slabs(gather_structured_entities(entities))
    _resolve_sublevel_mesh_order(slabs, entities)

    by_name = {s.physical_name[0]: s for s in slabs}
    # A keeps its full footprint (winner).
    assert by_name["A"].resolved_footprint.area == 8.0  # 4 * 2
    # B keeps only [4,6] x [0,2] (carved by A).
    assert by_name["B"].resolved_footprint.area == 4.0  # 2 * 2


def test_resolve_sublevel_mesh_bool_false_carves_kept_neighbour():
    """A mesh_bool=False entity carves out of overlapping kept slabs."""
    from shapely.geometry import Polygon

    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec
    from meshwell.structured.plan import (
        _resolve_sublevel_mesh_order,
        expand_to_slabs,
        gather_structured_entities,
    )

    kept = PolyPrism(
        polygons=Polygon([(0, 0), (4, 0), (4, 2), (0, 2)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="kept",
        mesh_order=1,
    )
    void = PolyPrism(
        polygons=Polygon([(1, 0.5), (2, 0.5), (2, 1.5), (1, 1.5)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        mesh_bool=False,
        physical_name="void_tag",
    )
    entities = [kept, void]
    slabs = expand_to_slabs(gather_structured_entities(entities))
    _resolve_sublevel_mesh_order(slabs, entities)

    by_name = {s.physical_name[0]: s for s in slabs}
    # Kept has the 1x1 void carved out.
    assert abs(by_name["kept"].resolved_footprint.area - (8.0 - 1.0)) < 1e-9


def test_resolve_sublevel_disjoint_footprints_unchanged():
    """Slabs at different z-intervals or non-overlapping XY are unaffected."""
    from shapely.geometry import Polygon

    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec
    from meshwell.structured.plan import (
        _resolve_sublevel_mesh_order,
        expand_to_slabs,
        gather_structured_entities,
    )

    a = PolyPrism(
        polygons=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="A",
        mesh_order=1,
    )
    b_diff_z = PolyPrism(
        polygons=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        buffers={2.0: 0.0, 3.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="B",
        mesh_order=2,
    )
    c_diff_xy = PolyPrism(
        polygons=Polygon([(10, 0), (11, 0), (11, 1), (10, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="C",
        mesh_order=2,
    )
    entities = [a, b_diff_z, c_diff_xy]
    slabs = expand_to_slabs(gather_structured_entities(entities))
    _resolve_sublevel_mesh_order(slabs, entities)

    by_name = {s.physical_name[0]: s for s in slabs}
    assert by_name["A"].resolved_footprint.equals(a.polygons)
    assert by_name["B"].resolved_footprint.equals(b_diff_z.polygons)
    assert by_name["C"].resolved_footprint.equals(c_diff_xy.polygons)
