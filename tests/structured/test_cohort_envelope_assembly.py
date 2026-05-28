"""Assembly tests for cohort envelope solid."""

from __future__ import annotations

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.cohort_envelope import (
    assemble_cohort_envelope_solid,
    build_cohort_envelope,
)
from meshwell.structured.plan import build_plan


def _square_slab(zlo, zhi, name, side=1.0):
    return PolyPrism(
        polygons=shapely.box(0, 0, side, side),
        buffers={zlo: 0.0, zhi: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name=name,
    )


def test_assembled_solid_is_valid_and_closed():
    from OCP.BRepCheck import BRepCheck_Analyzer

    plan = build_plan([_square_slab(0.0, 1.0, "L1"), _square_slab(1.0, 2.0, "L2")])
    env = build_cohort_envelope(plan, component_index=0)
    solid = assemble_cohort_envelope_solid(env)
    assert solid is not None
    analyzer = BRepCheck_Analyzer(solid)
    assert analyzer.IsValid(), "Cohort envelope solid must be BRepCheck-valid"


def test_assembled_solid_has_positive_volume():
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    plan = build_plan([_square_slab(0.0, 1.0, "L1")])
    env = build_cohort_envelope(plan, component_index=0)
    solid = assemble_cohort_envelope_solid(env)
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid, props)
    # A unit cube has volume 1.0.
    assert abs(props.Mass() - 1.0) < 1e-3
