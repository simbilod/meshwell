"""compute_cutters: deterministic per-entity predecessor list."""
from __future__ import annotations

from shapely.geometry import Polygon

from meshwell.cad_occ import CAD_OCC, compute_cutters
from meshwell.polyprism import PolyPrism


def _square_prism(x, y, w, h, name, mesh_order):
    return PolyPrism(
        polygons=Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name=name,
        mesh_order=mesh_order,
    )


def test_disjoint_pillars_have_empty_cutters():
    entities = [
        _square_prism(0, 0, 1, 1, "a", 1),
        _square_prism(5, 5, 1, 1, "b", 2),
        _square_prism(10, 10, 1, 1, "c", 3),
    ]
    proc = CAD_OCC()
    cutters = compute_cutters(entities, proc)
    assert cutters == {0: [], 1: [], 2: []}


def test_three_stacked_prisms_form_cascade():
    # Three overlapping prisms with strictly increasing mesh_order.
    entities = [
        _square_prism(0, 0, 5, 5, "a", 1),
        _square_prism(0, 0, 5, 5, "b", 2),
        _square_prism(0, 0, 5, 5, "c", 3),
    ]
    proc = CAD_OCC()
    cutters = compute_cutters(entities, proc)
    assert cutters[0] == []
    assert cutters[1] == [0]
    assert cutters[2] == [0, 1]


def test_cutter_lists_are_sorted_by_mesh_order_then_idx():
    # Build entities in mixed insertion order.
    a = _square_prism(0, 0, 5, 5, "a", 3)
    b = _square_prism(0, 0, 5, 5, "b", 1)
    c = _square_prism(0, 0, 5, 5, "c", 2)
    entities = [a, b, c]  # indices: a=0, b=1, c=2
    proc = CAD_OCC()
    cutters = compute_cutters(entities, proc)
    # b (mesh_order 1) precedes c (mesh_order 2) precedes a (mesh_order 3).
    assert cutters[1] == []  # b has no precedessors
    assert cutters[2] == [1]  # c is preceded by b
    assert cutters[0] == [1, 2]  # a is preceded by b then c


def test_equal_mesh_order_ties_break_by_insertion_idx():
    entities = [
        _square_prism(0, 0, 5, 5, "a", 1),  # idx 0
        _square_prism(0, 0, 5, 5, "b", 1),  # idx 1
        _square_prism(0, 0, 5, 5, "c", 1),  # idx 2
    ]
    proc = CAD_OCC()
    cutters = compute_cutters(entities, proc)
    assert cutters[0] == []
    assert cutters[1] == [0]
    assert cutters[2] == [0, 1]
