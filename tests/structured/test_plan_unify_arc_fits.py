"""Verify the planner unifies fitted circles across half-arcs of the same disc.

When an annular face partition bisects a disc with radial cuts, the
resulting half-arcs each get their own independent circle fit, producing
nominally-identical circles whose center/radius differ by ~1e-5 due to
floating-point fit variance. `_unify_concentric_arc_fits` clusters arcs
by (rounded center, rounded radius) and re-fits each cluster from the
union of vertices so every cluster member shares one canonical circle.

Without this unification, the Phase 2 cohort topology builder snaps
multi-arc corners differently per arc (depending on which arc's circle
the snap uses), creating tolerance issues in downstream OCC topology.
"""

from __future__ import annotations

import math

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.plan import build_plan


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


def test_half_arcs_of_same_disc_share_canonical_circle():
    """Concentric stacked discs → annular partition → half-arcs share circle."""
    entities = [
        _arc_slab(1.0, 0.0, 1.0, "L1"),
        _arc_slab(0.7, 1.0, 2.0, "L2"),
    ]
    plan = build_plan(entities)
    arr = plan.arrangements[0]

    # Group arc edges by their canonical circle identity.
    by_circle: dict[tuple[float, float, float], list[int]] = {}
    for e in arr.edges:
        if e.circle is None:
            continue
        key = (e.circle.center[0], e.circle.center[1], e.circle.radius)
        by_circle.setdefault(key, []).append(e.edge_id)

    # The r=1.0 disc and the r=0.7 disc each get bisected by annular radial
    # cuts → 2 half-arcs per disc. After unification each disc's two half-
    # arcs MUST share the same canonical circle (same key in by_circle).
    multi_member_clusters = [eids for eids in by_circle.values() if len(eids) >= 2]
    assert len(multi_member_clusters) >= 2, (
        f"Expected >=2 unified clusters (one per disc); got {len(multi_member_clusters)}. "
        f"All circles: {list(by_circle.keys())}"
    )


def test_unification_no_op_for_single_disc():
    """Single disc with no annular partition: nothing to unify, no change."""
    entities = [_arc_slab(1.0, 0.0, 1.0, "L1")]
    plan = build_plan(entities)
    arr = plan.arrangements[0]
    # Should not crash; just verify arc edges exist.
    arc_edges = [e for e in arr.edges if e.circle is not None]
    assert arc_edges, "Test setup: expected arc edges"
