"""Same-name fuse collapses entities with matching (physical_name, keep, dim)."""
from __future__ import annotations

import math

from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from shapely.geometry import Polygon

from meshwell.cad_occ import CAD_OCC, _same_name_fuse
from meshwell.polyprism import PolyPrism
from meshwell.tolerances import Tolerances


def _solid_count(shape) -> int:
    n = 0
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        n += 1
        exp.Next()
    return n


def _volume(shape) -> float:
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return props.Mass()


def _make_labeled(prism, idx):
    proc = CAD_OCC()
    return proc._instantiate_entity_occ(idx, prism)


def _square_prism(x, y, w, h, name, mesh_order, keep=True):
    p = PolyPrism(
        polygons=Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name=name,
        mesh_order=mesh_order,
    )
    p.mesh_bool = keep  # PolyPrism uses mesh_bool for the keep flag
    return p


def test_unique_names_pass_through_unchanged():
    a = _make_labeled(_square_prism(0, 0, 1, 1, "a", 1), 0)
    b = _make_labeled(_square_prism(5, 5, 1, 1, "b", 2), 1)
    out = _same_name_fuse([a, b], Tolerances.from_characteristic_length(1.0))
    assert len(out) == 2
    assert {o.physical_name for o in out} == {("a",), ("b",)}


def test_two_disjoint_same_named_pillars_combined_into_one_record():
    a = _make_labeled(_square_prism(0, 0, 1, 1, "ox", 1), 0)
    b = _make_labeled(_square_prism(5, 5, 1, 1, "ox", 1), 1)
    out = _same_name_fuse([a, b], Tolerances.from_characteristic_length(1.0))
    assert len(out) == 1
    fused = out[0]
    assert fused.physical_name == ("ox",)
    assert sum(_solid_count(s) for s in fused.shapes) == 2  # geometrically disjoint
    total_v = sum(_volume(s) for s in fused.shapes)
    assert math.isclose(total_v, 2.0, rel_tol=1e-6)


def test_abutting_same_named_pillars_fuse_into_single_solid():
    # Two unit cubes sharing the x=1 face.
    a = _make_labeled(_square_prism(0, 0, 1, 1, "ox", 1), 0)
    b = _make_labeled(_square_prism(1, 0, 1, 1, "ox", 1), 1)
    out = _same_name_fuse([a, b], Tolerances.from_characteristic_length(1.0))
    assert len(out) == 1
    fused = out[0]
    assert sum(_solid_count(s) for s in fused.shapes) == 1
    total_v = sum(_volume(s) for s in fused.shapes)
    assert math.isclose(total_v, 2.0, rel_tol=1e-3)


def test_different_keep_values_not_fused():
    a = _make_labeled(_square_prism(0, 0, 1, 1, "ox", 1, keep=True), 0)
    b = _make_labeled(_square_prism(5, 5, 1, 1, "ox", 1, keep=False), 1)
    out = _same_name_fuse([a, b], Tolerances.from_characteristic_length(1.0))
    assert len(out) == 2


def test_fused_entity_takes_min_mesh_order():
    a = _make_labeled(_square_prism(0, 0, 1, 1, "ox", 5), 0)
    b = _make_labeled(_square_prism(5, 5, 1, 1, "ox", 1), 1)
    c = _make_labeled(_square_prism(10, 10, 1, 1, "ox", 3), 2)
    out = _same_name_fuse([a, b, c], Tolerances.from_characteristic_length(1.0))
    assert len(out) == 1
    assert out[0].mesh_order == 1
    assert out[0].index == 0  # min input index


def test_structured_members_are_not_fused():
    # Two same-named entities; if either is structured, the group must pass
    # through with both records preserved. Structured entities carry per-
    # entity mesh metadata that merging would erase.
    a = _make_labeled(_square_prism(0, 0, 1, 1, "ox", 1), 0)
    b = _make_labeled(_square_prism(5, 5, 1, 1, "ox", 1), 1)
    a.structured = True  # mark one member structured
    out = _same_name_fuse([a, b], Tolerances.from_characteristic_length(1.0))
    assert len(out) == 2
    assert all(o.physical_name == ("ox",) for o in out)


def test_two_unstructured_same_named_still_fuse():
    # Sanity: regression check that the default unstructured path still fuses.
    a = _make_labeled(_square_prism(0, 0, 1, 1, "ox", 1), 0)
    b = _make_labeled(_square_prism(5, 5, 1, 1, "ox", 1), 1)
    assert a.structured is False
    assert b.structured is False
    out = _same_name_fuse([a, b], Tolerances.from_characteristic_length(1.0))
    assert len(out) == 1
