"""End-to-end cohort envelope build for two stacked simple slabs.

Spec test #1: verify envelope solid is closed, top/bottom shells have
correct per-piece sub-face counts, lateral wall has correct outline-edge
count.
"""

from __future__ import annotations

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.cohort_envelope import (
    assemble_cohort_envelope_solid,
    build_cohort_envelope,
)
from meshwell.structured.plan import build_plan


def _square_slab(zlo, zhi, name):
    return PolyPrism(
        polygons=shapely.box(0, 0, 1, 1),
        buffers={zlo: 0.0, zhi: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name=name,
    )


def test_two_stacked_slabs_envelope_end_to_end():
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    plan = build_plan([_square_slab(0.0, 1.0, "L1"), _square_slab(1.0, 2.0, "L2")])
    env = build_cohort_envelope(plan, component_index=0)
    solid = assemble_cohort_envelope_solid(env)

    assert BRepCheck_Analyzer(solid).IsValid()

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid, props)
    # Two stacked unit boxes: total volume 2.0.
    assert abs(props.Mass() - 2.0) < 1e-3

    # 2 slabs x 1 piece per slab = 2 top sub-faces, 2 bot sub-faces.
    assert len(env.top_sub_faces) == 2
    assert len(env.bottom_sub_faces) == 2

    # 2 slabs x 4 outline edges = 8 lateral face lists.
    assert len(env.lateral_faces) == 8
