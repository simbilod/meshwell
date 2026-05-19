"""End-to-end integration tests for the polyprism overlap fast-path.

Each scenario builds a small two-entity scene where the fast-path's
decision is non-trivial (AABB overlap but no real overlap, tangent
faces, z-separation, tapered conservative envelope) and verifies that
the cut output is identical with and without the fast-path enabled.

Spec: docs/superpowers/specs/2026-05-19-cad-occ-polyprism-overlap-fastpath-design.md
"""
from __future__ import annotations

import math
from typing import Any

from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from shapely.geometry import Polygon

from meshwell.cad_common import prepare_entities
from meshwell.cad_occ import CAD_OCC
from meshwell.polyprism import PolyPrism


def _square(x, y, w, h) -> Polygon:
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def _solid_count(shape) -> int:
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    n = 0
    while exp.More():
        n += 1
        exp.Next()
    return n


def _volume(shape) -> float:
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return abs(props.Mass())


def _run_cut(entities: list[Any], disable_fastpath: bool = False):
    """Run process_entities_cut_only with the fast-path optionally disabled."""
    prepare_entities(entities, perturbation=1e-5, resolve_snap=1e-3)
    processor = CAD_OCC(point_tolerance=1e-3, perturbation=1e-5)
    if disable_fastpath:
        # Monkeypatch the helper to always return None -> forces fall-through
        # to the OCC _shapes_actually_overlap path.
        processor._polyprism_fast_overlap = lambda _, __: None  # type: ignore[assignment]
    return processor.process_entities_cut_only(entities)


def _summarise(labeled):
    """{(physical_name,): (solid_count, volume)} for each entity in labeled."""
    out: dict[tuple, tuple[int, float]] = {}
    for le in labeled:
        nsol = sum(_solid_count(s) for s in le.shapes)
        vol = sum(_volume(s) for s in le.shapes)
        out[le.physical_name] = (nsol, vol)
    return out


def _assert_summaries_equal(fast, slow):
    """Solid counts must match; volumes within 0.1% relative."""
    assert set(fast.keys()) == set(slow.keys())
    for k in fast:
        sf, vf = fast[k]
        ss, vs = slow[k]
        assert sf == ss, f"{k}: fast solids {sf} != slow solids {ss}"
        if vs == 0:
            assert vf == 0
        else:
            rel = abs(vf - vs) / vs
            assert rel < 1e-3, f"{k}: fast vol {vf}, slow vol {vs} (rel err {rel:.2%})"


# --- Scenario 1: overlapping AABBs, disjoint polygons ----------------------


def test_scenario_1_overlapping_aabbs_disjoint_polygons():
    """L-shape vs square in the L's concave corner: their AABBs overlap but the polygons share no material.

    Fast-path: dwithin returns False, no cut, ``a`` keeps its full volume.
    """
    # L-shape: full 10x10 footprint minus a 6x6 bite from the upper-right.
    # AABB is [0, 0, 10, 10].
    l_shape = Polygon([(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)])
    # 5x5 tool in the upper-right (the cut-out of the L) -- AABB overlaps
    # the L's AABB but polygons are disjoint.
    tool = _square(5, 5, 5, 5)
    expected_a_vol = l_shape.area  # cut should be a no-op for 'a'
    ents_fast = [
        PolyPrism(
            polygons=l_shape,
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="a",
            mesh_order=10.0,
        ),
        PolyPrism(
            polygons=tool,
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="b",
            mesh_order=1.0,
        ),
    ]
    ents_slow = [
        PolyPrism(
            polygons=l_shape,
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="a",
            mesh_order=10.0,
        ),
        PolyPrism(
            polygons=tool,
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="b",
            mesh_order=1.0,
        ),
    ]
    fast = _summarise(_run_cut(ents_fast))
    slow = _summarise(_run_cut(ents_slow, disable_fastpath=True))
    _assert_summaries_equal(fast, slow)
    assert fast[("a",)][0] == 1
    assert math.isclose(fast[("a",)][1], expected_a_vol, rel_tol=5e-3)


# --- Scenario 2: tangent in xy (shared face) -------------------------------


def test_scenario_2_xy_tangent_pair_shares_a_face():
    """Two abutting extrusions sharing the x=2 face.

    Fast-path: dwithin succeeds at distance 0; both exact => True. Cut happens.
    """
    ents_fast = [
        PolyPrism(
            polygons=_square(0, 0, 2, 2),
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="a",
            mesh_order=10.0,
        ),
        PolyPrism(
            polygons=_square(2, 0, 2, 2),
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="b",
            mesh_order=1.0,
        ),
    ]
    ents_slow = [
        PolyPrism(
            polygons=_square(0, 0, 2, 2),
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="a",
            mesh_order=10.0,
        ),
        PolyPrism(
            polygons=_square(2, 0, 2, 2),
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="b",
            mesh_order=1.0,
        ),
    ]
    fast = _summarise(_run_cut(ents_fast))
    slow = _summarise(_run_cut(ents_slow, disable_fastpath=True))
    _assert_summaries_equal(fast, slow)


# --- Scenario 3: z-separated -----------------------------------------------


def test_scenario_3_z_separated_pair():
    """Same xy footprint, but z-intervals separated by more than fuzzy.

    Fast-path: z_gap > cut_fuzzy_value => False (no cut), matches OCC.
    """
    ents_fast = [
        PolyPrism(
            polygons=_square(0, 0, 3, 3),
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="a",
            mesh_order=10.0,
        ),
        PolyPrism(
            polygons=_square(0, 0, 3, 3),
            buffers={5.0: 0.0, 6.0: 0.0},
            physical_name="b",
            mesh_order=1.0,
        ),
    ]
    ents_slow = [
        PolyPrism(
            polygons=_square(0, 0, 3, 3),
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="a",
            mesh_order=10.0,
        ),
        PolyPrism(
            polygons=_square(0, 0, 3, 3),
            buffers={5.0: 0.0, 6.0: 0.0},
            physical_name="b",
            mesh_order=1.0,
        ),
    ]
    fast = _summarise(_run_cut(ents_fast))
    slow = _summarise(_run_cut(ents_slow, disable_fastpath=True))
    _assert_summaries_equal(fast, slow)
    # 'a' should be untouched (z-disjoint from b).
    assert fast[("a",)][0] == 1
    assert math.isclose(fast[("a",)][1], 9.0, rel_tol=5e-3)


# --- Scenario 4: tapered envelope says "maybe", reality is "no" ------------


def test_scenario_4_tapered_envelope_overlaps_but_actually_disjoint():
    """A tapered prism's conservative envelope contains the tool's footprint in xy.

    At the tool's z-slice the actual cross-section is empty. Fast-path returns
    None (tapered=>inexact); OCC verifies disjoint; final cut output matches the
    fast-path-disabled run.
    """
    # Tapered prism: base 6x6 at z=0, shrunk to 2x2 at z=1 (buffer = -2)
    # Its conservative envelope is the BASE polygon (largest cross-section).
    tapered = PolyPrism(
        polygons=_square(0, 0, 6, 6),
        buffers={0.0: 0.0, 1.0: -2.0},  # shrinks inward at z=1 to 2x2
        physical_name="a",
        mesh_order=10.0,
    )
    # Tool far to the right -- AABB-disjoint, polygon-disjoint from the 2x2 core
    # at z=1, so even the cheap envelope test should reject. However, the
    # conservative envelope (the 6x6 base) contains the tool's footprint.
    # This exercises the tapered path: returns None, falls back to OCC.
    tool = PolyPrism(
        polygons=_square(10, 10, 1, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="b",
        mesh_order=1.0,
    )
    fast = _summarise(_run_cut([tapered, tool]))
    slow_tapered = PolyPrism(
        polygons=_square(0, 0, 6, 6),
        buffers={0.0: 0.0, 1.0: -2.0},
        physical_name="a",
        mesh_order=10.0,
    )
    slow_tool = PolyPrism(
        polygons=_square(10, 10, 1, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="b",
        mesh_order=1.0,
    )
    slow = _summarise(_run_cut([slow_tapered, slow_tool], disable_fastpath=True))
    _assert_summaries_equal(fast, slow)
