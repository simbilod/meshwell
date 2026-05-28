"""build_phantom_shapes with _USE_COHORT_TOPOLOGY=True uses the cohort builder."""

from __future__ import annotations

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


def test_kill_switch_default_is_on():
    assert phantom_mod._USE_COHORT_TOPOLOGY is True


def test_cohort_topology_path_produces_shared_lateral_face():
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
    """With _USE_COHORT_TOPOLOGY=False we fall back to the existing path."""
    phantom_mod._USE_COHORT_TOPOLOGY = False
    try:
        plan = build_plan([_polyprism("A", 0, 1, 1)])
        result = build_phantom_shapes(plan)
        assert len(result.shapes) == 1
    finally:
        phantom_mod._USE_COHORT_TOPOLOGY = True


def test_output_ordering_preserved():
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
