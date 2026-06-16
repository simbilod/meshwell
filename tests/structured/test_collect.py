# tests/structured/test_collect.py
import pytest
from shapely.geometry import Polygon

from meshwell.polyprism import PolyPrism
from meshwell.structured.collect import collect_structured_slabs
from meshwell.structured.exceptions import StructuredEntityTypeError

SQ = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


def make_prism(name, zlo=0.0, zhi=1.0, structured=False, mesh_order=1.0):
    return PolyPrism(
        polygons=SQ,
        buffers={zlo: 0.0, zhi: 0.0},
        physical_name=name,
        structured=structured,
        mesh_order=mesh_order,
    )


def test_separates_structured_from_unstructured():
    a = make_prism("a", structured=True)
    b = make_prism("b", structured=False)
    slabs, unstructured = collect_structured_slabs([a, b])
    assert len(slabs) == 1
    assert len(unstructured) == 1
    assert slabs[0].source_index == 0
    assert unstructured[0] is b


def test_multi_z_polyprism_emits_one_slab_per_interval():
    # buffers={0:0, 1:0, 2:0} → two intervals (0,1) and (1,2)
    p = PolyPrism(
        polygons=SQ,
        buffers={0.0: 0.0, 1.0: 0.0, 2.0: 0.0},
        physical_name="multi",
        structured=True,
    )
    slabs, _ = collect_structured_slabs([p])
    assert len(slabs) == 2
    intervals = sorted([(s.zlo, s.zhi) for s in slabs])
    assert intervals == [(0.0, 1.0), (1.0, 2.0)]
    assert all(s.source_index == 0 for s in slabs)


def test_non_polyprism_structured_raises():
    class FakeStructured:
        structured = True
        physical_name = "fake"

    with pytest.raises(StructuredEntityTypeError):
        collect_structured_slabs([FakeStructured()])


def test_carries_arc_metadata():
    # Explicit identify_arcs=True to test that arc metadata is carried through.
    # structured=True alone no longer implies identify_arcs=True (that caused
    # rectangular polygons to be built as arc-inflated disk solids).
    p = PolyPrism(
        polygons=SQ,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="a",
        structured=True,
        identify_arcs=True,
        arc_tolerance=5e-4,
        min_arc_points=5,
    )
    slabs, _ = collect_structured_slabs([p])
    assert slabs[0].arc_tolerance == 5e-4
    assert slabs[0].min_arc_points == 5
    assert slabs[0].identify_arcs is True
