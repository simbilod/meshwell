from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from shapely.geometry import Polygon

from meshwell.polyprism import PolyPrism
from meshwell.structured.build import build_cohort_compound
from meshwell.structured.cohort import build_cohorts
from meshwell.structured.collect import collect_structured_slabs
from meshwell.structured.decompose import decompose_cohorts


def _count(shape, kind) -> int:
    exp = TopExp_Explorer(shape, kind)
    out = 0
    while exp.More():
        out += 1
        exp.Next()
    return out


def _collect_faces(shape):
    """Return list of all TopoDS_Face objects visited by TopExp_Explorer."""
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    faces = []
    while exp.More():
        faces.append(exp.Current())
        exp.Next()
    return faces


def _count_issame_pairs(faces) -> int:
    """Count pairs of faces that are IsSame (shared TShape = conformality)."""
    count = 0
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            if faces[i].IsSame(faces[j]):
                count += 1
    return count


SQ = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])


def test_single_subpiece_compound_has_one_solid():
    ent = PolyPrism(
        polygons=SQ,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="s",
        structured=True,
    )
    slabs, unstr = collect_structured_slabs([ent])
    cohorts = build_cohorts(slabs)
    subs_per_cohort, _ = decompose_cohorts(cohorts, unstr)
    compound, slab_meta = build_cohort_compound(
        cohorts[0],
        subs_per_cohort[0],
        point_tolerance=1e-3,
    )
    assert _count(compound, TopAbs_SOLID) == 1
    # 1 bot + 1 top + 4 lateral = 6 faces.
    assert _count(compound, TopAbs_FACE) == 6
    assert len(slab_meta) == 1


def test_two_stacked_subpieces_share_interior_face():
    a = PolyPrism(
        polygons=SQ,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="a",
        structured=True,
    )
    b = PolyPrism(
        polygons=SQ,
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="b",
        structured=True,
    )
    slabs, unstr = collect_structured_slabs([a, b])
    cohorts = build_cohorts(slabs)
    subs_per_cohort, _ = decompose_cohorts(cohorts, unstr)
    compound, slab_meta = build_cohort_compound(
        cohorts[0],
        subs_per_cohort[0],
        point_tolerance=1e-3,
    )
    assert _count(compound, TopAbs_SOLID) == 2
    # TopExp_Explorer visits each face once per solid that owns it.
    # 1 bot (z=0) + 1 top (z=2) + 1 interior (z=1) counted twice
    # (once per adjacent solid) + 4+4 laterals = 12.
    assert _count(compound, TopAbs_FACE) == 12
    # Conformality check: the shared interior face (z=1) is the SAME
    # TShape in both solids — exactly 1 IsSame pair among all faces.
    faces = _collect_faces(compound)
    assert _count_issame_pairs(faces) == 1
    assert len(slab_meta) == 2
