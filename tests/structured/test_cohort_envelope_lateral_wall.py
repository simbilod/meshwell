"""Lateral wall registry tests for cohort_envelope."""

from __future__ import annotations

import math

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.cohort_envelope import build_cohort_envelope
from meshwell.structured.plan import build_plan


def _square_slab(zlo, zhi, name, side=1.0):
    return PolyPrism(
        polygons=shapely.box(0, 0, side, side),
        buffers={zlo: 0.0, zhi: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name=name,
    )


def _arc_slab(r, zlo, zhi, name):
    n = 32
    pts = [
        (r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]
    return PolyPrism(
        polygons=shapely.Polygon(pts),
        buffers={zlo: 0.0, zhi: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
        physical_name=name,
    )


def test_lateral_wall_one_per_slab_per_outline_edge():
    """One lateral face list per (slab_index, outline_edge_id)."""
    plan = build_plan([_square_slab(0.0, 1.0, "L1"), _square_slab(1.0, 2.0, "L2")])
    env = build_cohort_envelope(plan, component_index=0)
    arr = plan.arrangements[0]
    n_edges = len([e for e in arr.edges if e.edge_id not in env.skipped_edge_ids])
    # 2 slabs x N outline edges.
    assert len(env.lateral_faces) == 2 * n_edges


def test_arc_lateral_has_valid_bbox():
    """Arc lateral face built via BRepFill::Face_s has a bounded bbox (no plus-or-minus 1e+100)."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    plan = build_plan([_arc_slab(1.0, 0.0, 1.0, "L1")])
    env = build_cohort_envelope(plan, component_index=0)
    arr = plan.arrangements[0]
    arc_edge = next(e for e in arr.edges if e.circle is not None)
    face_list = env.lateral_faces[(0, arc_edge.edge_id)]
    assert face_list, "Arc outline edge should produce at least one lateral face"
    for face in face_list:
        bbox = Bnd_Box()
        BRepBndLib.Add_s(face, bbox)
        x0, y0, z0, x1, y1, z1 = bbox.Get()
        # Sane finite bbox under ~10 units (our arc has r=1).
        assert all(
            abs(v) < 10.0 for v in (x0, y0, z0, x1, y1, z1)
        ), f"BRepFill arc lateral has unbounded bbox: {bbox.Get()}"
