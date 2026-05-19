# tests/test_cad_occ_batched_compound_cut.py
"""Regression test for the cut-cascade correctness invariant.

The cad_occ cut cascade must produce, for the dense substrate-vs-N-bodies
scene used in the design spike, exactly 1 SOLID per cut entity with a
volume that matches the analytic expectation modulo perturbation noise.

This test pins the contract that BOTH the sequential per-tool loop AND
the batched-compound cut (see docs/superpowers/specs/2026-05-19-cad-occ-
batched-compound-cut-design.md) must satisfy. It also guards against
accidental loosening of the _shapes_actually_overlap pre-filter, which
is load-bearing for the batched form's correctness.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer

from meshwell.cad_common import prepare_entities
from meshwell.cad_occ import CAD_OCC

# The dense-scene builder lives in scripts/. Import it directly.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from bench_cut_strategies_dense import build_dense_scene


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


@pytest.mark.parametrize("n_bodies", [6, 12, 20])
def test_dense_substrate_cut_yields_one_solid_per_entity(n_bodies):
    """Substrate + box + top_clad each cut against n_bodies metal+via pairs.

    After the cut cascade (before fragment), every cut entity must hold
    exactly 1 SOLID summing to the expected analytic volume. This locks
    in the invariant that EITHER cut strategy (sequential per-tool OR
    batched compound) must satisfy. The original bug the design spec
    references manifested as zero SOLIDs after cut.
    """
    entities = build_dense_scene(n_bodies=n_bodies)
    prepare_entities(entities, perturbation=1e-5, resolve_snap=1e-3)

    processor = CAD_OCC(point_tolerance=1e-3, perturbation=1e-5)
    labeled = processor.process_entities_cut_only(entities)

    by_name = {str(le.physical_name): le for le in labeled}

    # Expected analytic volumes (XY area * z-extent), tolerating the
    # ~2e-5 perturbation buffer (which puffs each face by ~1e-5 per side).
    # Substrate, box, top_clad all use the big_xy = 20x20 footprint.
    big_xy_area = 20.0 * 20.0  # 400
    expected = {
        "('substrate',)": (big_xy_area * 1.0, 1),  # z=[-1, 0]
        "('box',)": (big_xy_area * 0.5, 1),  # z=[0, 0.5]
        "('top_clad',)": (big_xy_area * 1.5, 1),  # z=[1.5, 3.0]
    }

    for name, (expected_vol, expected_solids) in expected.items():
        assert name in by_name, f"{name} missing from cut output"
        le = by_name[name]
        total_solids = sum(_solid_count(s) for s in le.shapes)
        total_volume = sum(_volume(s) for s in le.shapes)
        assert (
            total_solids == expected_solids
        ), f"{name}: got {total_solids} SOLIDs, expected {expected_solids}"
        # Allow 0.5% absolute tolerance on volume (perturbation buffer adds
        # ~0.05% on the 20x20 face; small bodies cut some material too).
        assert math.isclose(total_volume, expected_vol, rel_tol=5e-3), (
            f"{name}: volume {total_volume:.4f} differs from expected "
            f"{expected_vol:.4f} (rel err {abs(total_volume - expected_vol) / expected_vol:.2%})"
        )


def test_substrate_cut_against_many_bodies_does_not_zero_out():
    """Reproduce-and-deny: the original empty-result bug.

    The pre-aa77f21 code switched away from compound cuts because
    substrate-vs-~10-bodies was observed to produce zero SOLIDs. With the
    modern _shapes_actually_overlap pre-filter in place, neither the
    sequential loop NOR the batched compound cut should ever return zero
    SOLIDs for the substrate. This test would fail if anyone disabled or
    weakened the _shapes_actually_overlap gate.
    """
    entities = build_dense_scene(n_bodies=20)
    prepare_entities(entities, perturbation=1e-5, resolve_snap=1e-3)
    processor = CAD_OCC(point_tolerance=1e-3, perturbation=1e-5)
    labeled = processor.process_entities_cut_only(entities)
    by_name = {str(le.physical_name): le for le in labeled}
    sub = by_name["('substrate',)"]
    n_solids = sum(_solid_count(s) for s in sub.shapes)
    assert n_solids >= 1, (
        f"substrate cut returned {n_solids} SOLIDs; the cut cascade has "
        "lost all material. This is the original empty-result regression."
    )
