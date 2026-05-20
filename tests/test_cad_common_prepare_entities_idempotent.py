"""prepare_entities is idempotent: double-call must not double-buffer."""
from __future__ import annotations

from shapely.geometry import Polygon

from meshwell.cad_common import clear_preparation, prepare_entities
from meshwell.polyprism import PolyPrism


def _make_prism(name, mesh_order):
    return PolyPrism(
        polygons=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name=name,
        mesh_order=mesh_order,
    )


def _polygon_bounds(entity):
    """Get bounds of the entity's polygon for comparison."""
    poly = entity.polygons
    if hasattr(poly, "geoms"):
        # MultiPolygon
        return tuple(tuple(g.bounds) for g in poly.geoms)
    return poly.bounds


def test_double_call_does_not_double_buffer():
    entities = [_make_prism("a", 1), _make_prism("b", 2)]
    perturbation = 1e-3
    prepare_entities(entities, perturbation=perturbation, resolve_snap=1e-3)
    bounds_after_first = [_polygon_bounds(e) for e in entities]
    prepare_entities(entities, perturbation=perturbation, resolve_snap=1e-3)
    bounds_after_second = [_polygon_bounds(e) for e in entities]
    assert bounds_after_first == bounds_after_second, (
        f"Double-buffer detected: first call -> {bounds_after_first}, "
        f"second call -> {bounds_after_second}"
    )


def test_fresh_entity_added_to_already_prepared_list_gets_buffered():
    a = _make_prism("a", 1)
    b = _make_prism("b", 2)
    prepare_entities([a, b], perturbation=1e-3, resolve_snap=1e-3)
    bounds_a_after_first = _polygon_bounds(a)
    c = _make_prism("c", 3)
    bounds_c_unbuffered = _polygon_bounds(c)
    prepare_entities([a, b, c], perturbation=1e-3, resolve_snap=1e-3)
    # a unchanged (already prepared)
    assert _polygon_bounds(a) == bounds_a_after_first
    # c was buffered (its bounds changed by ~perturbation)
    assert _polygon_bounds(c) != bounds_c_unbuffered


def test_prepared_flag_attribute_is_set():
    e = _make_prism("a", 1)
    assert not hasattr(e, "_meshwell_prepared") or e._meshwell_prepared is False
    prepare_entities([e], perturbation=1e-3, resolve_snap=1e-3)
    assert e._meshwell_prepared is True


def test_clear_preparation_allows_rebuffer():
    e = _make_prism("a", 1)
    prepare_entities([e], perturbation=1e-3, resolve_snap=1e-3)
    bounds_before = _polygon_bounds(e)
    clear_preparation([e])
    assert not hasattr(e, "_meshwell_prepared")
    prepare_entities([e], perturbation=1e-3, resolve_snap=1e-3)
    bounds_after = _polygon_bounds(e)
    # Re-buffered: bounds should differ (expanded further).
    assert bounds_after != bounds_before
