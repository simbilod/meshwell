from shapely.geometry import MultiPolygon, Polygon

from meshwell.polyprism import PolyPrism
from meshwell.structured.cohort import build_cohorts
from meshwell.structured.collect import collect_structured_slabs
from meshwell.structured.decompose import decompose_cohorts

SQ_BIG = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
SQ_SMALL = Polygon([(2, 2), (5, 2), (5, 5), (2, 5)])


def make(poly, zlo, zhi, name, structured=False):
    return PolyPrism(
        polygons=poly,
        buffers={zlo: 0.0, zhi: 0.0},
        physical_name=name,
        structured=structured,
    )


def test_simple_cohort_one_subpiece_per_interval():
    structured = make(SQ_BIG, 0, 1, "s", structured=True)
    slabs, unstr = collect_structured_slabs([structured])
    cohorts = build_cohorts(slabs)
    subpieces_per_cohort, _ = decompose_cohorts(cohorts, unstr)
    assert len(subpieces_per_cohort) == 1
    subpieces = subpieces_per_cohort[0]
    assert len(subpieces) == 1
    assert subpieces[0].z_interval == (0.0, 1.0)
    assert subpieces[0].sub_polygon.equals(SQ_BIG)


def test_stepped_cohort_creates_frame_and_center():
    a = make(SQ_BIG, 0, 1, "a", structured=True)
    b = make(SQ_SMALL, 1, 2, "b", structured=True)
    slabs, unstr = collect_structured_slabs([a, b])
    cohorts = build_cohorts(slabs)
    subpieces_per_cohort, _ = decompose_cohorts(cohorts, unstr)
    assert len(subpieces_per_cohort) == 1
    subs = subpieces_per_cohort[0]
    intervals = sorted({sp.z_interval for sp in subs})
    assert intervals == [(0.0, 1.0), (1.0, 2.0)]
    # [0,1] interval split into 2 subpieces by B's outline.
    lower = [sp for sp in subs if sp.z_interval == (0.0, 1.0)]
    assert len(lower) == 2
    upper = [sp for sp in subs if sp.z_interval == (1.0, 2.0)]
    assert len(upper) == 1


def test_unstructured_above_gets_pre_cut():
    # Cohort = SQ_SMALL at [0,1]; unstructured cap = SQ_BIG at [1,2].
    structured = make(SQ_SMALL, 0, 1, "s", structured=True)
    cap = make(SQ_BIG, 1, 2, "cap")
    slabs, unstr = collect_structured_slabs([structured, cap])
    cohorts = build_cohorts(slabs)
    _, pre_cut_unstr = decompose_cohorts(cohorts, unstr)
    assert len(pre_cut_unstr) == 1
    pre_cut_cap = pre_cut_unstr[0]
    # Cap's polygons attribute should now be MultiPolygon with 2 parts
    # (SQ_SMALL inside + SQ_BIG - SQ_SMALL outside).
    polys = pre_cut_cap.polygons
    assert isinstance(polys, MultiPolygon)
    assert len(polys.geoms) == 2
    # physical_name preserved
    assert pre_cut_cap.physical_name == ("cap",)


def test_decompose_touch_uses_z_tolerance():
    from meshwell.polyprism import PolyPrism

    structured = PolyPrism(
        polygons=SQ_SMALL,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="s",
        structured=True,
    )
    # Cap z-plane differs by 1e-12 from cohort's z=1.0.
    cap = PolyPrism(
        polygons=SQ_BIG,
        buffers={1.0 + 1e-12: 0.0, 2.0: 0.0},
        physical_name="cap",
    )
    slabs, unstr = collect_structured_slabs([structured, cap])
    cohorts = build_cohorts(slabs)
    _, pre_cut_unstr = decompose_cohorts(cohorts, unstr)
    # Cap should be pre-cut (despite the 1e-12 z mismatch) → MultiPolygon
    assert isinstance(pre_cut_unstr[0].polygons, MultiPolygon)


def test_unstructured_not_touching_cohort_unchanged():
    structured = make(SQ_SMALL, 0, 1, "s", structured=True)
    far = make(Polygon([(50, 50), (60, 50), (60, 60), (50, 60)]), 1, 2, "far")
    slabs, unstr = collect_structured_slabs([structured, far])
    cohorts = build_cohorts(slabs)
    _, pre_cut_unstr = decompose_cohorts(cohorts, unstr)
    assert len(pre_cut_unstr) == 1
    # Polygon untouched.
    assert pre_cut_unstr[0].polygons.equals(far.polygons)
