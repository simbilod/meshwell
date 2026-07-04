"""WP4 Task 6 — tolerance discipline + narrow arc-edge exception catches.

Pins:
  * z-plane dict keys in build.py are quantized (ULP-invariant), so a z
    differing by 1e-12 from a registered plane still shares arc params and
    shared horizontal interior faces (cross-plane conformality).
  * ``EdgeRegistry.vertical``'s cache key is endpoint-order-invariant.
  * ``_is_arc_edge`` warns + propagates (does not silently swallow) on an
    unexpected, non-OCC failure.
"""
import dataclasses
import math
import warnings

import pytest
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeVertex
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from shapely.geometry import Polygon

from meshwell.polyprism import PolyPrism
from meshwell.structured.build import (
    EdgeRegistry,
    VertexRegistry,
    _is_arc_edge,
    build_cohort_compound,
)
from meshwell.structured.cohort import build_cohorts
from meshwell.structured.collect import collect_structured_slabs
from meshwell.structured.decompose import decompose_cohorts
from meshwell.structured.types import quantize_key


def _make_line_edge():
    a = BRepBuilderAPI_MakeVertex(gp_Pnt(0, 0, 0)).Vertex()
    b = BRepBuilderAPI_MakeVertex(gp_Pnt(1, 0, 0)).Vertex()
    return BRepBuilderAPI_MakeEdge(a, b).Edge()


def _half_disc(r=1.0, n=24):
    """Straight diameter + an n-point circular arc (one long arc run)."""
    pts = [(-r, 0.0), (r, 0.0)]
    for i in range(1, n):
        a = math.pi * i / n
        pts.append((r * math.cos(a), r * math.sin(a)))
    return Polygon(pts)


def _count_arc_edges(shape) -> int:
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    n = 0
    while exp.More():
        if _is_arc_edge(TopoDS.Edge_s(exp.Current())):
            n += 1
        exp.Next()
    return n


# ---------------------------------------------------------------------------
# 1. Quantized z-plane keys are ULP-tolerant.
# ---------------------------------------------------------------------------


def test_quantize_z_key_ulp_invariant():
    """A z differing by 1e-12 quantizes to the SAME plane key."""
    pt = 1e-3
    for z in (0.0, 1.0, 3.7, -2.25):
        assert (
            quantize_key(0.0, 0.0, z, pt)[2] == quantize_key(0.0, 0.0, z + 1e-12, pt)[2]
        )


def _stacked_arc_cohort():
    """Two stacked half-discs sharing z=1, with DIFFERENT min_arc_points.

    The shared z=1 plane propagates the *stricter* (larger) min_arc_points
    across both slabs, so the lower slab's laterals at that plane suppress
    the arc. This propagation is what the z-plane dict keying gates.
    """
    lower = PolyPrism(
        _half_disc(),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="lo",
        structured=True,
        identify_arcs=True,
        min_arc_points=6,
    )
    upper = PolyPrism(
        _half_disc(),
        {1.0: 0.0, 2.0: 0.0},
        physical_name="hi",
        structured=True,
        identify_arcs=True,
        min_arc_points=40,
    )
    slabs, unstr = collect_structured_slabs([lower, upper])
    cohorts = build_cohorts(slabs)
    subs_per_cohort, _, _arr = decompose_cohorts(cohorts, unstr)
    return cohorts[0], subs_per_cohort[0]


def test_zplane_arc_propagation_ulp_invariant():
    """1e-12 z noise must not change cross-plane arc-parameter propagation.

    ``arc_params_for_z`` looks up per-z-plane (identify_arcs, min_arc_points,
    arc_tolerance). With raw-float dict keys, a subpiece z differing by 1e-12
    from the registered plane z silently MISSES the plane entry and falls
    back to that subpiece's own source-slab params — flipping arc detection
    on its laterals. Quantized z keys make the lookup ULP-invariant.
    """
    cohort, subs = _stacked_arc_cohort()

    def _perturb(sp):
        zlo, zhi = sp.z_interval
        nzlo = zlo + 1e-12 if abs(zlo - 1.0) < 1e-9 else zlo
        nzhi = zhi + 1e-12 if abs(zhi - 1.0) < 1e-9 else zhi
        return dataclasses.replace(sp, z_interval=(nzlo, nzhi))

    base_compound, _ = build_cohort_compound(cohort, subs, point_tolerance=1e-3)
    pert_compound, _ = build_cohort_compound(
        cohort, [_perturb(sp) for sp in subs], point_tolerance=1e-3
    )
    assert _count_arc_edges(pert_compound) == _count_arc_edges(base_compound), (
        "arc-parameter propagation across the shared z-plane must be "
        "invariant to 1e-12 float noise on subpiece z"
    )


# ---------------------------------------------------------------------------
# 2. EdgeRegistry.vertical cache key is endpoint-order-invariant.
# ---------------------------------------------------------------------------


def test_vertical_edge_cache_order_invariant():
    """vertical(z_a, z_b) and vertical(z_b, z_a) share one TShape."""
    vreg = VertexRegistry(point_tolerance=1e-3)
    ereg = EdgeRegistry(vertices=vreg, point_tolerance=1e-3)
    e_up = ereg.vertical(0.0, 0.0, 0.0, 1.0)
    e_down = ereg.vertical(0.0, 0.0, 1.0, 0.0)
    assert e_up.IsSame(e_down)
    assert len(ereg._store) == 1


# ---------------------------------------------------------------------------
# 3. _is_arc_edge warns (not silences) on an unexpected failure.
# ---------------------------------------------------------------------------


def test_is_arc_edge_warns_and_propagates_on_unexpected_error(monkeypatch):
    """An unexpected (non-OCC) failure is surfaced, not swallowed as False."""
    import OCP.GeomAdaptor as GA

    edge = _make_line_edge()
    # Sanity: a real line edge classifies as non-arc without warnings.
    assert _is_arc_edge(edge) is False

    def _boom(*_a, **_k):
        raise RuntimeError("injected non-OCC failure")

    monkeypatch.setattr(GA, "GeomAdaptor_Curve", _boom)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError):
            _is_arc_edge(edge)
    assert any(
        "edge" in str(w.message).lower() for w in captured
    ), "unexpected _is_arc_edge failure must emit a warning with edge context"
