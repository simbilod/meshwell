from shapely.geometry import Polygon

from meshwell.structured.cohort import build_cohorts
from meshwell.structured.types import StructuredSlab


def slab(idx, poly, zlo, zhi):
    return StructuredSlab(
        source_index=idx,
        footprint=poly,
        zlo=zlo,
        zhi=zhi,
        mesh_order=1.0,
        mesh_bool=True,
        physical_name=("x",),
        identify_arcs=False,
        arc_tolerance=1e-3,
        min_arc_points=4,
    )


SQ_A = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
SQ_B = Polygon([(3, 3), (8, 3), (8, 8), (3, 8)])  # overlaps A
SQ_FAR = Polygon([(20, 20), (25, 20), (25, 25), (20, 25)])


def test_disjoint_slabs_yield_separate_cohorts():
    s1 = slab(0, SQ_A, 0, 1)
    s2 = slab(1, SQ_FAR, 0, 1)
    cohorts = build_cohorts([s1, s2])
    assert len(cohorts) == 2


def test_lateral_touch_merges():
    s1 = slab(0, SQ_A, 0, 1)
    s2 = slab(1, SQ_B, 0, 1)  # same z-interval, overlapping XY
    cohorts = build_cohorts([s1, s2])
    assert len(cohorts) == 1
    assert len(cohorts[0].slabs) == 2


def test_face_touch_merges():
    # s1 at [0,1] and s2 at [1,2], share z=1 plane with overlap.
    s1 = slab(0, SQ_A, 0, 1)
    s2 = slab(1, SQ_B, 1, 2)  # shares z=1 with s1, XY overlaps
    cohorts = build_cohorts([s1, s2])
    assert len(cohorts) == 1
    assert cohorts[0].z_planes == (0.0, 1.0, 2.0)


def test_transitive_merge():
    # s1 ←lateral→ s2 ←face→ s3, all merge into one cohort.
    s1 = slab(0, SQ_A, 0, 1)
    s2 = slab(1, SQ_B, 0, 1)  # lateral with s1
    s3 = slab(2, SQ_B, 1, 2)  # face with s2 (same poly)
    cohorts = build_cohorts([s1, s2, s3])
    assert len(cohorts) == 1
    assert {s.source_index for s in cohorts[0].slabs} == {0, 1, 2}


def test_face_touch_requires_xy_overlap():
    s1 = slab(0, SQ_A, 0, 1)
    s2 = slab(1, SQ_FAR, 1, 2)  # same z-plane, no XY overlap
    cohorts = build_cohorts([s1, s2])
    assert len(cohorts) == 2
