"""End-to-end: cohort topology builder produces vertical AND lateral sharing.

Uses a synthetic scene that avoids the known-regressed cases (concentric
arcs, the unidentified hanging scene). Flips _USE_COHORT_TOPOLOGY ON via
a fixture for the duration of each test so the default-off behavior is
not affected.
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


def _faces(shape):
    out = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        out.append(exp.Current())
        exp.Next()
    return out


def _polyprism(name, z0, z1, mesh_order, x0=0, y0=0, x1=1, y1=1):
    return PolyPrism(
        polygons=_square(x0, y0, x1, y1),
        buffers={float(z0): 0.0, float(z1): 0.0},
        physical_name=name,
        mesh_order=mesh_order,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )


@pytest.fixture
def cohort_topology_on():
    prior = phantom_mod._USE_COHORT_TOPOLOGY
    phantom_mod._USE_COHORT_TOPOLOGY = True
    try:
        yield
    finally:
        phantom_mod._USE_COHORT_TOPOLOGY = prior


def _frame(name, zlo, zhi, x0, y0, x1, y1):
    """Low-priority wrapping slab so the cohort footprint stays constant."""
    return PolyPrism(
        polygons=_square(x0, y0, x1, y1),
        buffers={float(zlo): 0.0, float(zhi): 0.0},
        physical_name=name,
        mesh_order=10.0,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )


@pytest.mark.skipif(
    getattr(phantom_mod, "_USE_DISCRETE_COHORT_MESH", False),
    reason="Phase 2 cohort_topology only — Phase 3 global flag overrides cohort_topology_on routing",
)
def test_mixed_cohort_sharing(cohort_topology_on):  # noqa: ARG001  pytest fixture
    """Scene: vertical stack of 3 PolyPrisms + 2 lateral neighbors + 1 unstructured.

    - Cohort: A at z=[0,1], B at z=[1,2], C at z=[2,3] on XY [0,1]x[0,1]
      (vertical stack) plus D, E at z=[0,1] on [5,6]x[0,1] and [6,7]x[0,1]
      (lateral neighbors in the same cohort as A via same-z grouping).
      Frame slabs fill in D and E's footprint at z=[1,2] and z=[2,3] so the
      cohort footprint remains constant.
    - Unstructured: F as a non-structured PolyPrism somewhere disjoint
    """
    A = _polyprism("A", 0, 1, 1, x0=0, y0=0, x1=1, y1=1)
    B = _polyprism("B", 1, 2, 2, x0=0, y0=0, x1=1, y1=1)
    C = _polyprism("C", 2, 3, 3, x0=0, y0=0, x1=1, y1=1)
    D = _polyprism("D", 0, 1, 4, x0=5, y0=0, x1=6, y1=1)
    E = _polyprism("E", 0, 1, 5, x0=6, y0=0, x1=7, y1=1)
    # Frame slabs to fill D and E's footprint at z=[1,2] and z=[2,3]
    FD1 = _frame("Frame_D_z1", 1, 2, 5, 0, 6, 1)
    FE1 = _frame("Frame_E_z1", 1, 2, 6, 0, 7, 1)
    FD2 = _frame("Frame_D_z2", 2, 3, 5, 0, 6, 1)
    FE2 = _frame("Frame_E_z2", 2, 3, 6, 0, 7, 1)
    # F: same shape as PolyPrism but structured=False (unstructured neighbor).
    F = PolyPrism(
        polygons=_square(20, 20, 21, 21),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="F",
        mesh_order=6,
        structured=False,
        resolutions=[],
    )
    plan = build_plan([A, B, C, D, E, FD1, FE1, FD2, FE2, F])
    result = build_phantom_shapes(plan)

    by_name: dict[str, object] = {}
    for ps in result.shapes:
        slab = plan.slabs[ps.slab_index]
        by_name[slab.physical_name[0]] = ps

    def _face_hashes(name: str) -> set[int]:
        return {hash(f) for f in _faces(by_name[name].solid)}

    # Vertical sharing within cohort 1.
    # Note: A and B are different entities (different source_index) so per the
    # piece_id convention they DON'T share by source-based piece_id alone. The
    # cohort topology builder's piece_fingerprint keyword (added in Task 10)
    # collapses pieces with identical face_partition_edges fingerprints to share
    # the same horizontal face. If this assertion fails, the fingerprint
    # de-duplication isn't kicking in for this scene; revise expectation.
    a_b_share = bool(_face_hashes("A") & _face_hashes("B"))
    b_c_share = bool(_face_hashes("B") & _face_hashes("C"))
    a_c_share = bool(_face_hashes("A") & _face_hashes("C"))
    assert a_b_share, "A and B (vertical neighbors) should share interface face"
    assert b_c_share, "B and C (vertical neighbors) should share interface face"
    assert not a_c_share, "A and C should NOT share (not vertically adjacent)"

    # Lateral sharing within cohort 2.
    assert _face_hashes("D") & _face_hashes(
        "E"
    ), "D and E (lateral neighbors) should share interface face"

    # Cross-cohort: no sharing.
    assert not (
        _face_hashes("A") & _face_hashes("D")
    ), "Different cohorts should not share faces"
