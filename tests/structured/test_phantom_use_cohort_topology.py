"""build_phantom_shapes with _USE_COHORT_TOPOLOGY uses the cohort builder.

The kill-switch defaults to False during stabilization (see comment in
phantom.py). Tests that exercise the cohort topology path flip it on
within a try/finally.
"""

from __future__ import annotations

import pytest
import shapely
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured import phantom as phantom_mod
from meshwell.structured.phantom import build_phantom_shapes
from meshwell.structured.plan import build_plan


def _square(x0, y0, x1, y1):
    return shapely.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _polyprism(name, z0, z1, mesh_order, x0=0, y0=0, x1=1, y1=1):
    return PolyPrism(
        polygons=_square(x0, y0, x1, y1),
        buffers={float(z0): 0.0, float(z1): 0.0},
        physical_name=name,
        mesh_order=mesh_order,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )


def _faces(shape):
    out = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        out.append(exp.Current())
        exp.Next()
    return out


@pytest.fixture
def cohort_topology_on():
    """Flip _USE_COHORT_TOPOLOGY ON for the duration of the test."""
    prior = phantom_mod._USE_COHORT_TOPOLOGY
    phantom_mod._USE_COHORT_TOPOLOGY = True
    try:
        yield
    finally:
        phantom_mod._USE_COHORT_TOPOLOGY = prior


def test_kill_switch_default_is_off_during_stabilization():
    """Stabilization default: False. Phase 3 will flip back to True."""
    assert phantom_mod._USE_COHORT_TOPOLOGY is False


@pytest.mark.skipif(
    getattr(phantom_mod, "_USE_DISCRETE_COHORT_MESH", False),
    reason="Phase 1+2 path only — Phase 3 global flag overrides cohort_topology_on routing",
)
def test_cohort_topology_path_produces_shared_lateral_face(
    cohort_topology_on,  # noqa: ARG001  pytest fixture
):
    """Two laterally-adjacent structured slabs -> shared lateral TShape."""
    A = _polyprism("A", 0, 1, 1, x0=0, y0=0, x1=1, y1=1)
    B = _polyprism("B", 0, 1, 2, x0=1, y0=0, x1=2, y1=1)
    plan = build_plan([A, B])
    result = build_phantom_shapes(plan)
    by_slab = {ps.slab_index: ps for ps in result.shapes}
    assert len(by_slab) == 2

    a_face_hashes = {hash(f) for f in _faces(by_slab[0].solid)}
    b_face_hashes = {hash(f) for f in _faces(by_slab[1].solid)}
    assert a_face_hashes & b_face_hashes, (
        "Laterally-adjacent cohort sub-prisms do NOT share interface lateral "
        "face TShape — cohort topology builder not active or wired incorrectly."
    )


def test_legacy_path_when_kill_switch_off():
    """Default is False; legacy path produces a single PhantomShape."""
    plan = build_plan([_polyprism("A", 0, 1, 1)])
    result = build_phantom_shapes(plan)
    assert len(result.shapes) == 1


def test_output_ordering_preserved(cohort_topology_on):  # noqa: ARG001
    """PhantomBuildResult.shapes is in (slab_index, piece_index) ascending order."""
    plan = build_plan(
        [
            _polyprism("A", 0, 1, 1),
            _polyprism("B", 1, 2, 2),
            _polyprism("C", 2, 3, 3),
        ]
    )
    result = build_phantom_shapes(plan)
    indices = [(ps.slab_index, ps.piece_index) for ps in result.shapes]
    assert indices == sorted(indices)
