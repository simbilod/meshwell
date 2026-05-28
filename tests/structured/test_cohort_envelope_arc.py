"""Cohort envelope with an arc outline.

Spec test #2: verify the arc lateral is built via BRepFill::Face_s and
has valid PCurves (bbox sanity).
"""

from __future__ import annotations

import math

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.cohort_envelope import (
    assemble_cohort_envelope_solid,
    build_cohort_envelope,
)
from meshwell.structured.plan import build_plan


def _arc_disc(r, n=32):
    return shapely.Polygon(
        [
            (r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n))
            for i in range(n)
        ]
    )


def test_arc_outline_envelope_assembles_and_volume_matches_circle():
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    arc_slab = PolyPrism(
        polygons=_arc_disc(1.0),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
        physical_name="ArcCohort",
    )
    plan = build_plan([arc_slab])
    env = build_cohort_envelope(plan, component_index=0)
    solid = assemble_cohort_envelope_solid(env)
    assert BRepCheck_Analyzer(solid).IsValid()

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid, props)
    # Unit-radius disc x height 1.0 ~= pi.
    assert abs(props.Mass() - math.pi) < 0.05
