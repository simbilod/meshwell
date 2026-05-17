"""Performance instrumentation for the structured-polyprism pipeline.

Provides a module-level ``structured_logger`` and a thread-local
``StructuredTimings`` accumulator that records wall time per phase plus
arbitrary integer counter increments. The ``@phase_timed("name")``
decorator wraps an entry-point function and records its wall time.
The ``counter(name)`` and ``counter_inc(name, n)`` helpers expose
incremental counters for hotspot tracking (e.g. number of
``gmsh.model.occ.getEntities(2)`` calls).

Usage::

    from meshwell.structured.logging import (
        phase_timed,
        counter_inc,
        get_timings,
        reset_timings,
    )

    @phase_timed("plan")
    def build_plan(entities): ...

    @phase_timed("mesh_apply")
    def apply_structured_mesh(...):
        # ... per-slab loop
        for slab in slabs:
            counter_inc("getEntities_2D_calls")
            # ...

    # After a run:
    print_timings()  # human-readable summary
    timings = get_timings()  # dict for programmatic use
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from typing import Any, TypeVar

structured_logger = logging.getLogger("meshwell.structured")

# Module-level accumulators. Not thread-safe — meshwell isn't either.
_phase_times: dict[str, list[float]] = {}
_counters: dict[str, int] = {}

F = TypeVar("F", bound=Callable[..., Any])


def reset_timings() -> None:
    """Clear all accumulated phase times + counters."""
    _phase_times.clear()
    _counters.clear()


def get_timings() -> dict[str, Any]:
    """Return a snapshot of accumulated timings + counters."""
    return {
        "phases": {
            name: {
                "calls": len(times),
                "total_s": sum(times),
                "mean_s": (sum(times) / len(times)) if times else 0.0,
                "min_s": min(times) if times else 0.0,
                "max_s": max(times) if times else 0.0,
            }
            for name, times in _phase_times.items()
        },
        "counters": dict(_counters),
    }


def counter_inc(name: str, n: int = 1) -> None:
    """Increment a named counter (creates if missing)."""
    _counters[name] = _counters.get(name, 0) + n


def counter(name: str) -> int:
    """Read the current value of a named counter."""
    return _counters.get(name, 0)


@contextmanager
def phase_timer(name: str):
    """Context-manager that times the block and records under ``name``."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        _phase_times.setdefault(name, []).append(dt)


def phase_timed(name: str) -> Callable[[F], F]:
    """Decorator: time the wrapped function and record under ``name``."""

    def _wrap(fn: F) -> F:
        @wraps(fn)
        def _inner(*args: Any, **kwargs: Any) -> Any:
            with phase_timer(name):
                return fn(*args, **kwargs)

        return _inner  # type: ignore[return-value]

    return _wrap


def print_timings(stream: Any = None) -> None:
    """Human-readable timing summary to ``stream`` (default: stdout via print)."""
    import sys

    out = stream if stream is not None else sys.stdout
    timings = get_timings()
    out.write("\n" + "=" * 70 + "\n")
    out.write("structured-pipeline timings\n")
    out.write("=" * 70 + "\n")
    if not timings["phases"]:
        out.write("(no phases recorded)\n")
    else:
        out.write(
            f"{'phase':25s} {'calls':>6s} {'total_s':>10s} "
            f"{'mean_s':>10s} {'max_s':>10s}\n"
        )
        for name, stats in sorted(timings["phases"].items()):
            out.write(
                f"{name:25s} {stats['calls']:>6d} {stats['total_s']:>10.4f} "
                f"{stats['mean_s']:>10.4f} {stats['max_s']:>10.4f}\n"
            )
    if timings["counters"]:
        out.write("\ncounters:\n")
        for name, val in sorted(timings["counters"].items()):
            out.write(f"  {name:30s} {val:>10d}\n")
    out.write("=" * 70 + "\n")
