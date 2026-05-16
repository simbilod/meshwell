"""Tests for StructuredMeshPlan resolution from StructuredPlan + entities."""
from __future__ import annotations

import pytest
from shapely.geometry import Polygon


def _square(x=0, y=0, w=1, h=1) -> Polygon:
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def _structured(polygon, buffers, n_layers, name, recombine=False, mesh_order=1.0):
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    return PolyPrism(
        polygons=polygon,
        buffers=buffers,
        structured=True,
        resolutions=[
            StructuredExtrusionResolutionSpec(n_layers=n_layers, recombine=recombine),
        ],
        physical_name=name,
        mesh_order=mesh_order,
    )


def test_resolve_mesh_plan_single_slab():
    from meshwell.structured import build_plan
    from meshwell.structured.builder import resolve_mesh_plan

    s = _structured(_square(), {0.0: 0.0, 1.0: 0.0}, [3], "s")
    plan = build_plan([s])
    mp = resolve_mesh_plan(plan, [s])
    assert len(mp.slabs) == 1
    assert mp.n_layers == (3,)
    assert mp.recombine == (False,)


def test_resolve_mesh_plan_multi_z_intervals():
    from meshwell.structured import build_plan
    from meshwell.structured.builder import resolve_mesh_plan

    s = _structured(_square(), {0.0: 0.0, 1.0: 0.0, 2.5: 0.0}, [2, 5], "s")
    plan = build_plan([s])
    mp = resolve_mesh_plan(plan, [s])
    # Two slabs; n_layers parallel by slab order.
    assert mp.n_layers == (2, 5)


def test_resolve_mesh_plan_recombine_true():
    from meshwell.structured import build_plan
    from meshwell.structured.builder import resolve_mesh_plan

    s = _structured(_square(), {0.0: 0.0, 1.0: 0.0}, [2], "s", recombine=True)
    plan = build_plan([s])
    mp = resolve_mesh_plan(plan, [s])
    assert mp.recombine == (True,)


def test_resolve_mesh_plan_overlap_mismatch_raises():
    """Two overlapping slabs whose owning specs have different n_layers raise."""
    from meshwell.structured.builder import resolve_mesh_plan

    # Synthesize an OverlapPair via direct construction (Phase-1 Policy B
    # would have caught this at plan time, so we bypass it).
    from meshwell.structured.spec import (
        OverlapPair,
        Slab,
        StructuredMeshOverlapError,
        StructuredPlan,
    )

    winner = Slab(
        footprint=_square(),
        zlo=0.0,
        zhi=1.0,
        physical_name=("a",),
        source_index=0,
        z_interval_index=0,
        mesh_order=1.0,
        face_partition=[_square()],
    )
    plan = StructuredPlan(
        slabs=(winner,),
        z_planes=(0.0, 1.0),
        overlaps=(
            OverlapPair(
                winner_slab_index=0,
                loser_source_index=1,
                loser_z_interval_index=0,
                z_extent=(0.0, 1.0),
            ),
        ),
    )

    e_winner = _structured(_square(), {0.0: 0.0, 1.0: 0.0}, [3], "a")
    e_loser = _structured(_square(), {0.0: 0.0, 1.0: 0.0}, [5], "b")
    with pytest.raises(StructuredMeshOverlapError, match="n_layers"):
        resolve_mesh_plan(plan, [e_winner, e_loser])
