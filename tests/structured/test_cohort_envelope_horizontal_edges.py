"""Horizontal edge registry tests for cohort_envelope."""

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


def _arc_slab(r, zlo, zhi, name):
    n = 32
    pts = [
        (r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]
    return PolyPrism(
        polygons=shapely.Polygon(pts),
        buffers={zlo: 0.0, zhi: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
        physical_name=name,
    )


def test_horizontal_edge_registry_size_matches_outline_x_zplanes():
    """One wire per (z_plane, outline_edge_id)."""
    plan = build_plan([_square_slab(0.0, 1.0, "L1"), _square_slab(1.0, 2.0, "L2")])
    env = build_cohort_envelope(plan, component_index=0)
    arr = plan.arrangements[0]
    expected = 3 * len(arr.edges)  # 3 z-planes x N outline edges
    assert len(env.horizontal_edges) == expected


def test_horizontal_edge_for_arc_outline_is_arc_wire():
    """An arc arrangement edge produces a wire containing an arc-typed OCC edge."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepTools import BRepTools_WireExplorer
    from OCP.GeomAbs import GeomAbs_Circle

    plan = build_plan([_arc_slab(1.0, 0.0, 1.0, "L1")])
    env = build_cohort_envelope(plan, component_index=0)
    arr = plan.arrangements[0]
    arc_edges = [e for e in arr.edges if e.circle is not None]
    assert arc_edges, "Test setup expected at least one arc outline edge"
    wire = env.horizontal_edges[(0.0, arc_edges[0].edge_id)]
    exp = BRepTools_WireExplorer(wire)
    assert exp.More(), "Wire should have at least one edge"
    curve = BRepAdaptor_Curve(exp.Current())
    assert curve.GetType() == GeomAbs_Circle
