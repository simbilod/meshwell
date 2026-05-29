"""Planner-level validation that cohort XY footprint is constant across z.

Phase 3's cohort envelope architecture only supports cohorts whose
combined-by-piece XY footprint is the same at every z-interval. Stepped
stacks (e.g. concentric arc discs) would require modelling exposed
annular rings on slab-to-slab interfaces and are rejected at plan time
with a clear remediation hint.
"""

from __future__ import annotations

import math

import pytest
import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.plan import build_plan
from meshwell.structured.spec import StructuredCohortFootprintMismatchError


def _disc(r, n=32):
    return shapely.Polygon(
        [
            (r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n))
            for i in range(n)
        ]
    )


def _square_slab(zlo, zhi, name, x0=0.0, y0=0.0, x1=1.0, y1=1.0):
    return PolyPrism(
        polygons=shapely.box(x0, y0, x1, y1),
        buffers={zlo: 0.0, zhi: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name=name,
    )


def _disc_slab(r, zlo, zhi, name):
    return PolyPrism(
        polygons=_disc(r),
        buffers={zlo: 0.0, zhi: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
        physical_name=name,
    )


def test_uniform_stack_passes():
    """Two stacked equal-footprint slabs are accepted."""
    plan = build_plan([_square_slab(0.0, 1.0, "L1"), _square_slab(1.0, 2.0, "L2")])
    assert len(plan.slabs) == 2


def test_single_z_interval_cohort_passes():
    """A cohort with a single z-interval is trivially constant."""
    plan = build_plan([_square_slab(0.0, 1.0, "L1")])
    assert len(plan.slabs) == 1


def test_disjoint_cohorts_pass():
    """Each cohort is validated independently.

    A small-then-big stack in one cohort is fine if it doesn't share z
    with another cohort that has a different footprint.
    """
    plan = build_plan(
        [
            _square_slab(0.0, 1.0, "A1", x0=0, y0=0, x1=1, y1=1),
            _square_slab(1.0, 2.0, "A2", x0=0, y0=0, x1=1, y1=1),
            _square_slab(10.0, 11.0, "B1", x0=10, y0=10, x1=12, y1=12),
            _square_slab(11.0, 12.0, "B2", x0=10, y0=10, x1=12, y1=12),
        ]
    )
    assert len({s.component_index for s in plan.slabs}) == 2


def test_concentric_arc_discs_rejected():
    """Concentric arc discs of decreasing radius form a stepped cohort.

    Rejected because the cohort's XY footprint shrinks at higher z.
    """
    with pytest.raises(StructuredCohortFootprintMismatchError) as exc_info:
        build_plan(
            [
                _disc_slab(1.0, 0.0, 1.0, "L1"),
                _disc_slab(0.7, 1.0, 2.0, "L2"),
                _disc_slab(0.5, 2.0, 3.0, "L3"),
            ]
        )
    msg = str(exc_info.value)
    assert "inconsistent XY footprint" in msg
    assert "Fill the missing area" in msg


def test_stepped_square_stack_rejected():
    """Same rejection applies to straight-edged stepped stacks."""
    with pytest.raises(StructuredCohortFootprintMismatchError):
        build_plan(
            [
                _square_slab(0.0, 1.0, "Base", x0=0, y0=0, x1=2, y1=2),
                _square_slab(1.0, 2.0, "Top", x0=0.5, y0=0.5, x1=1.5, y1=1.5),
            ]
        )


def test_filled_stepped_stack_passes():
    """Filler rectangles around a smaller upper slab restore constancy.

    Four surrounding rectangles at the upper z-interval fill the
    annulus around the smaller central slab, so the cohort's union
    footprint matches the canonical and the plan is accepted.
    """
    plan = build_plan(
        [
            _square_slab(0.0, 1.0, "Base", x0=0, y0=0, x1=2, y1=2),
            _square_slab(1.0, 2.0, "Top", x0=0.5, y0=0.5, x1=1.5, y1=1.5),
            # Frame around the top: 4 surrounding rectangles to fill
            # the annulus.
            _square_slab(1.0, 2.0, "FrameW", x0=0, y0=0, x1=0.5, y1=2),
            _square_slab(1.0, 2.0, "FrameE", x0=1.5, y0=0, x1=2, y1=2),
            _square_slab(1.0, 2.0, "FrameS", x0=0.5, y0=0, x1=1.5, y1=0.5),
            _square_slab(1.0, 2.0, "FrameN", x0=0.5, y0=1.5, x1=1.5, y1=2),
        ]
    )
    assert len(plan.slabs) == 6
