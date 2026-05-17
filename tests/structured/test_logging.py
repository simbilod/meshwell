"""Smoke tests for meshwell.structured.logging."""
from __future__ import annotations


def test_phase_timer_records_wall_time():
    import time

    from meshwell.structured.logging import (
        get_timings,
        phase_timer,
        reset_timings,
    )

    reset_timings()
    with phase_timer("test_phase"):
        time.sleep(0.01)
    timings = get_timings()
    assert "test_phase" in timings["phases"]
    assert timings["phases"]["test_phase"]["calls"] == 1
    assert timings["phases"]["test_phase"]["total_s"] >= 0.01


def test_counters_increment():
    from meshwell.structured.logging import counter, counter_inc, reset_timings

    reset_timings()
    counter_inc("foo")
    counter_inc("foo")
    counter_inc("bar", 5)
    assert counter("foo") == 2
    assert counter("bar") == 5
    assert counter("missing") == 0


def test_phase_timed_decorator():
    from meshwell.structured.logging import (
        get_timings,
        phase_timed,
        reset_timings,
    )

    reset_timings()

    @phase_timed("decorated")
    def f(x):
        return x * 2

    assert f(5) == 10
    assert f(3) == 6
    timings = get_timings()
    assert timings["phases"]["decorated"]["calls"] == 2


def test_end_to_end_pipeline_records_timings_for_structured_run():
    import tempfile
    from pathlib import Path

    from shapely.geometry import Polygon

    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec
    from meshwell.structured.logging import get_timings, reset_timings

    reset_timings()
    p = PolyPrism(
        polygons=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="slab",
    )
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "t.msh"
        generate_mesh([p], dim=3, output_mesh=out, default_characteristic_length=0.5)

    timings = get_timings()
    # At minimum the plan + mesh_apply phases should have been hit.
    assert "plan" in timings["phases"]
    assert "mesh_apply" in timings["phases"]
    assert timings["phases"]["plan"]["calls"] >= 1
    assert timings["phases"]["mesh_apply"]["calls"] >= 1
