"""Vertical edge registry tests for cohort_envelope."""

from __future__ import annotations

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


def test_vertical_edge_per_z_interval_per_outline_corner():
    """Two stacked square slabs: 2 z-intervals x 4 corners = 8 vertical edges."""
    plan = build_plan([_square_slab(0.0, 1.0, "L1"), _square_slab(1.0, 2.0, "L2")])
    env = build_cohort_envelope(plan, component_index=0)
    assert len(env.vertical_edges) == 8


def test_vertical_edges_shared_across_slabs_at_same_z_interval():
    """Two slabs at the same z-interval share TopoDS_Edge per corner.

    Two square slabs at z=[0,1], side-by-side in XY, sharing one outline edge.
    The shared corners must reference the SAME TopoDS_Edge for both slabs.
    """
    s1 = PolyPrism(
        polygons=shapely.box(0, 0, 1, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="L1",
    )
    s2 = PolyPrism(
        polygons=shapely.box(1, 0, 2, 1),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="L2",
    )
    plan = build_plan([s1, s2])
    env = build_cohort_envelope(plan, component_index=0)
    # Each unique outline corner at this single z-interval should map
    # to ONE TopoDS_Edge — only one entry in the registry per
    # (zlo, zhi, corner_id) tuple.
    keys = list(env.vertical_edges.keys())
    z_intervals = {(zlo, zhi) for (zlo, zhi, _cid) in keys}
    assert z_intervals == {(0.0, 1.0)}
    # Cardinality = N outline corners x 1 z-interval.
    n_corners = len(env.outline_xy_to_corner_id)
    assert len(env.vertical_edges) == n_corners
