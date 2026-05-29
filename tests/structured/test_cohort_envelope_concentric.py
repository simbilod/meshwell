"""Concentric arc discs are rejected at plan time.

The multi-radius stepped stack would require modelling annular rings on
slab-to-slab interfaces — not supported by Phase 3's cohort envelope
architecture. The planner's `_validate_cohort_footprint_constancy`
catches the case before it reaches the envelope builder. See
tests/structured/test_cohort_footprint_constancy.py for the full
validator coverage and the remediation contract.

The multi-arc snap behaviour this spec test originally targeted is
covered by
tests/structured/test_cohort_envelope_vertices.py::test_multi_arc_vertex_snap_carries_tolerance.
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


def _arc_slab(r, zlo, zhi, name):
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


def test_concentric_arc_discs_rejected_at_plan_time():
    """build_plan must reject a stepped stack of concentric arc discs."""
    with pytest.raises(StructuredCohortFootprintMismatchError):
        build_plan(
            [
                _arc_slab(1.0, 0.0, 1.0, "L1"),
                _arc_slab(0.7, 1.0, 2.0, "L2"),
                _arc_slab(0.5, 2.0, 3.0, "L3"),
            ]
        )
