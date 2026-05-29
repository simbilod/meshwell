"""Tests for meshwell.structured.phantom shape construction."""
from __future__ import annotations

import pytest
from shapely.geometry import Polygon

import meshwell.structured.phantom as _phantom_mod

_PHASE3_ON = getattr(_phantom_mod, "_USE_DISCRETE_COHORT_MESH", False)


def _unit_square() -> Polygon:
    return Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


def _square_with_hole() -> Polygon:
    outer = [(0, 0), (4, 0), (4, 4), (0, 4)]
    hole = [(1, 1), (2, 1), (2, 2), (1, 2)]
    return Polygon(outer, [hole])


def test_make_face_returns_topods_face_at_z():
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_VERTEX

    from meshwell.structured.phantom import _make_face_from_polygon
    from tests.structured._occ_helpers import count_subshapes

    face = _make_face_from_polygon(_unit_square(), z=2.5)
    assert count_subshapes(face, TopAbs_EDGE) == 4
    assert count_subshapes(face, TopAbs_VERTEX) == 4


def test_make_face_with_hole_has_two_wires():
    from OCP.TopAbs import TopAbs_WIRE

    from meshwell.structured.phantom import _make_face_from_polygon
    from tests.structured._occ_helpers import count_subshapes

    face = _make_face_from_polygon(_square_with_hole(), z=0.0)
    # 1 outer wire + 1 inner wire.
    assert count_subshapes(face, TopAbs_WIRE) == 2


def test_make_face_canonicalizes_orientation():
    """A CW-ordered shapely polygon (reversed) still produces a valid face."""
    from meshwell.structured.phantom import _make_face_from_polygon

    cw = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])  # CW exterior
    face = _make_face_from_polygon(cw, z=0.0)
    assert face is not None  # didn't crash; the canonicalize step flipped it


def test_make_face_z_is_respected():
    """The face sits at the requested z plane."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface

    from meshwell.structured.phantom import _make_face_from_polygon

    face = _make_face_from_polygon(_unit_square(), z=7.0)
    surf = BRepAdaptor_Surface(face)
    u_mid = 0.5 * (surf.FirstUParameter() + surf.LastUParameter())
    v_mid = 0.5 * (surf.FirstVParameter() + surf.LastVParameter())
    pnt = surf.Value(u_mid, v_mid)
    assert pnt.Z() == pytest.approx(7.0)


def test_build_sub_prism_returns_solid_with_expected_topology():
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID

    from meshwell.structured.phantom import _build_sub_prism
    from tests.structured._occ_helpers import count_subshapes

    out = _build_sub_prism(_unit_square(), zlo=0.0, zhi=1.0)
    assert count_subshapes(out.solid, TopAbs_SOLID) == 1
    # 1 bottom + 1 top + 4 lateral = 6 faces.
    assert count_subshapes(out.solid, TopAbs_FACE) == 6


def test_build_sub_prism_records_bottom_and_top_face_keys():
    """The returned record knows which face is bottom and which is top, by key."""
    from meshwell.structured.phantom import _build_sub_prism
    from meshwell.structured.spec import FaceKey

    out = _build_sub_prism(
        _unit_square(), zlo=0.0, zhi=1.0, slab_index=2, piece_index=3
    )
    assert FaceKey(slab_index=2, side="bot", piece_index=3) in out.input_faces_by_key
    assert FaceKey(slab_index=2, side="top", piece_index=3) in out.input_faces_by_key


def test_build_sub_prism_records_lateral_faces_per_outer_edge():
    """One lateral face per outer-edge segment, indexed by edge_index."""
    from meshwell.structured.phantom import _build_sub_prism

    out = _build_sub_prism(_unit_square(), zlo=0.0, zhi=1.0)
    # Unit square has 4 outer edges -> 4 lateral faces.
    assert len(out.input_laterals_by_outer_edge) == 4
    assert set(out.input_laterals_by_outer_edge.keys()) == {0, 1, 2, 3}


def test_build_sub_prism_with_hole_records_extra_lateral_faces():
    """A face with a hole produces lateral faces for both outer and inner edges."""
    from meshwell.structured.phantom import _build_sub_prism

    out = _build_sub_prism(_square_with_hole(), zlo=0.0, zhi=1.0)
    # 4 outer + 4 inner = 8 lateral faces total, but we only key the
    # outer ones (Layer A's outer-only contract).
    assert len(out.input_laterals_by_outer_edge) == 4


def test_build_sub_prism_records_bottom_edge_keys():
    """Bottom edge keys cover all bottom face boundary segments."""
    from meshwell.structured.phantom import _build_sub_prism

    out = _build_sub_prism(
        _unit_square(), zlo=0.0, zhi=1.0, slab_index=0, piece_index=0
    )
    bot_edges = {k for k in out.input_edges_by_key if k.side == "bot"}
    # 4 outer edges on a square.
    assert len(bot_edges) == 4
    # All have piece_index=0.
    assert all(k.piece_index == 0 for k in bot_edges)


def test_build_sub_prism_records_bottom_vertex_keys():
    from meshwell.structured.phantom import _build_sub_prism

    out = _build_sub_prism(
        _unit_square(), zlo=0.0, zhi=1.0, slab_index=0, piece_index=0
    )
    bot_verts = {k for k in out.input_vertices_by_key if k.side == "bot"}
    assert len(bot_verts) == 4


def test_build_phantom_shapes_empty_plan_returns_empty_result():
    from meshwell.structured.phantom import build_phantom_shapes
    from meshwell.structured.spec import StructuredPlan

    plan = StructuredPlan(slabs=(), z_planes=(), overlaps=())
    result = build_phantom_shapes(plan)
    assert result.shapes == ()


@pytest.mark.skipif(
    _PHASE3_ON,
    reason="Phase 1+2 path only — Phase 3 cohort shape sets slab_index = -1",
)
def test_build_phantom_shapes_one_slab_one_piece():
    """Single slab with a one-piece partition yields one PhantomShape."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan
    from meshwell.structured.phantom import build_phantom_shapes

    s = PolyPrism(
        polygons=_unit_square(),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[3])],
        physical_name="s",
    )
    plan = build_plan([s])
    result = build_phantom_shapes(plan)
    assert len(result.shapes) == 1
    assert result.shapes[0].slab_index == 0
    assert result.shapes[0].piece_index == 0


@pytest.mark.skipif(
    _PHASE3_ON,
    reason="Phase 1+2 path only — Phase 3 collapses per-piece shapes into one envelope",
)
def test_build_phantom_shapes_multi_piece_partition():
    """A slab with a 2-piece face_partition yields 2 PhantomShapes."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan
    from meshwell.structured.phantom import build_phantom_shapes

    # 4x4 structured square; a non-structured neighbour on top covers half.
    s = PolyPrism(
        polygons=Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[3])],
        physical_name="s",
    )
    n = PolyPrism(
        polygons=Polygon([(0, 0), (2, 0), (2, 4), (0, 4)]),
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="n",
    )
    plan = build_plan([s, n])
    # face_partition should have 2 pieces (covered + uncovered halves).
    assert len(plan.slabs[0].face_partition) == 2

    result = build_phantom_shapes(plan)
    assert len(result.shapes) == 2
    assert {sh.slab_index for sh in result.shapes} == {0}
    assert sorted(sh.piece_index for sh in result.shapes) == [0, 1]


def test_build_phantom_shapes_is_deterministic_ordering():
    """Output ordering is (slab_index, piece_index) ascending."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan
    from meshwell.structured.phantom import build_phantom_shapes

    s0 = PolyPrism(
        polygons=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="s0",
    )
    s1 = PolyPrism(
        polygons=Polygon([(10, 0), (11, 0), (11, 1), (10, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="s1",
    )
    plan = build_plan([s0, s1])
    result = build_phantom_shapes(plan)
    indices = [(sh.slab_index, sh.piece_index) for sh in result.shapes]
    assert indices == sorted(indices)


@pytest.mark.skipif(
    _PHASE3_ON,
    reason="Phase 1+2 path only — Phase 3 verifies per-source grouping via test_phase3_group_phantom_solids_by_entity_handles_cohort",
)
def test_group_phantom_solids_by_entity_inverts_slab_source_index():
    """Phantom solids are grouped under their entity's source_index in input order."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec
    from meshwell.structured.phantom import (
        _group_phantom_solids_by_entity,
        build_phantom_shapes,
    )
    from meshwell.structured.plan import build_plan

    a = PolyPrism(
        polygons=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="A",
    )
    b = PolyPrism(
        polygons=Polygon([(10, 0), (11, 0), (11, 1), (10, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="B",
    )
    entities = [a, b]
    plan = build_plan(entities)
    phantom_result = build_phantom_shapes(plan)

    overrides = _group_phantom_solids_by_entity(plan, phantom_result)
    assert set(overrides.keys()) == {0, 1}
    assert len(overrides[0]) == 1
    assert len(overrides[1]) == 1
    a_solid = next(
        s.solid
        for s in phantom_result.shapes
        if plan.slabs[s.slab_index].source_index == 0
    )
    b_solid = next(
        s.solid
        for s in phantom_result.shapes
        if plan.slabs[s.slab_index].source_index == 1
    )
    assert overrides[0][0] is a_solid
    assert overrides[1][0] is b_solid
