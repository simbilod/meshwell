"""Integration tests for the redesigned tolerance chain."""
from __future__ import annotations

import pytest


def test_interior_buffer_radius_relative_small_radius():
    """At r=1e-3 with arc_chord_height_fraction=0.01, buffer is r*0.01=1e-5."""
    from shapely.geometry import Point

    from meshwell.structured.plan import _interior_buffer_for_radius
    from meshwell.structured.spec import Slab

    slab = Slab(
        footprint=Point(0, 0).buffer(1e-3, resolution=32),
        zlo=0.0,
        zhi=1e-3,
        physical_name=("small_disc",),
        source_index=0,
        z_interval_index=0,
        mesh_order=0,
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
        arc_chord_height_fraction=0.01,
    )
    assert _interior_buffer_for_radius(slab, r=1e-3) == pytest.approx(1e-5)


def test_interior_buffer_radius_relative_large_radius():
    """At r=100 with fraction=0.01, buffer is r*0.01=1.0 (proportional)."""
    from shapely.geometry import Point

    from meshwell.structured.plan import _interior_buffer_for_radius
    from meshwell.structured.spec import Slab

    slab = Slab(
        footprint=Point(0, 0).buffer(100.0, resolution=64),
        zlo=0.0,
        zhi=1.0,
        physical_name=("big",),
        source_index=0,
        z_interval_index=0,
        mesh_order=0,
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
        arc_chord_height_fraction=0.01,
    )
    assert _interior_buffer_for_radius(slab, r=100.0) == pytest.approx(1.0)


def test_aggregate_slab_fuzzy_warns_on_heterogeneous(caplog):
    """Mesh stage warns when slabs disagree on fragment_fuzzy_value."""
    import logging

    from meshwell.structured.builder import _aggregate_slab_fuzzy

    caplog.set_level(logging.WARNING, logger="meshwell.structured.builder")
    result = _aggregate_slab_fuzzy([1e-6, 1e-4, None], default=1e-6)
    assert result == 1e-4
    assert any(
        "heterogeneous fragment_fuzzy_value" in rec.message.lower()
        for rec in caplog.records
    )


def test_aggregate_slab_fuzzy_no_warn_when_uniform(caplog):
    """No warning when all non-None values agree."""
    import logging

    from meshwell.structured.builder import _aggregate_slab_fuzzy

    caplog.set_level(logging.WARNING, logger="meshwell.structured.builder")
    result = _aggregate_slab_fuzzy([1e-5, 1e-5, None], default=1e-6)
    assert result == 1e-5
    assert not any("heterogeneous" in rec.message.lower() for rec in caplog.records)


def test_aggregate_slab_fuzzy_all_none_uses_default():
    from meshwell.structured.builder import _aggregate_slab_fuzzy

    assert _aggregate_slab_fuzzy([None, None], default=1e-7) == 1e-7
