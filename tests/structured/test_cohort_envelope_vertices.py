"""Vertex registry tests for cohort_envelope."""

from __future__ import annotations

import math

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.cohort_envelope import build_cohort_envelope
from meshwell.structured.plan import build_plan


def _square_slab(zlo, zhi, name, side=1.0):
    return PolyPrism(
        polygons=shapely.box(0, 0, side, side),
        buffers={zlo: 0.0, zhi: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name=name,
    )


def _disc(cx, cy, r, n=32):
    return shapely.Polygon(
        [
            (
                cx + r * math.cos(2 * math.pi * i / n),
                cy + r * math.sin(2 * math.pi * i / n),
            )
            for i in range(n)
        ]
    )


def _arc_slab(r, zlo, zhi, name):
    return PolyPrism(
        polygons=_disc(0, 0, r),
        buffers={zlo: 0.0, zhi: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
        physical_name=name,
    )


def test_vertex_registry_populated_for_simple_cohort():
    """A two-slab square cohort registers 4 corners at each z-plane (2 z-planes => 8 vertices)."""
    plan = build_plan(
        [
            _square_slab(0.0, 1.0, "L1"),
            _square_slab(1.0, 2.0, "L2"),
        ]
    )
    env = build_cohort_envelope(plan, component_index=0)
    # 4 outline corners x 3 z-planes (z=0, z=1, z=2) = 12 vertices.
    assert len(env.vertices) == 12
    assert len(env.outline_xy_to_corner_id) == 4


def test_multi_arc_vertex_snap_carries_tolerance():
    """Concentric arc discs produce multi-arc corners; vertex has positive OCC tolerance."""
    from OCP.BRep import BRep_Tool

    plan = build_plan(
        [
            _arc_slab(1.0, 0.0, 1.0, "L1"),
            _arc_slab(0.7, 1.0, 2.0, "L2"),
        ]
    )
    env = build_cohort_envelope(plan, component_index=0)
    # At least one vertex must have nontrivial tolerance from multi-arc snap.
    saw_tol = False
    for v in env.vertices.values():
        tol = BRep_Tool.Tolerance_s(v)
        if tol > 1e-9:
            saw_tol = True
            break
    assert saw_tol, "Expected at least one vertex with multi-arc snap tolerance"
