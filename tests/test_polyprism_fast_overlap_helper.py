"""Unit tests for CAD_OCC._polyprism_fast_overlap (pure logic, no OCC).

Constructs lightweight OCCLabeledEntity records with synthetic shapely
footprints and z-ranges and asserts the helper returns True/False/None
correctly. See spec
``docs/superpowers/specs/2026-05-19-cad-occ-polyprism-overlap-fastpath-design.md``.
"""
from __future__ import annotations

from shapely.geometry import Polygon

from meshwell.cad_occ import CAD_OCC, OCCLabeledEntity


def _square(x, y, w, h) -> Polygon:
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def _le(footprint, zrange, exact):
    return OCCLabeledEntity(
        shapes=[],
        physical_name=("x",),
        index=0,
        keep=True,
        dim=3,
        overlap_footprint=footprint,
        overlap_zrange=zrange,
        overlap_exact=exact,
    )


def _proc(fuzzy: float = 1e-5):
    return CAD_OCC(point_tolerance=1e-3, perturbation=1e-5, cut_fuzzy_value=fuzzy)


def test_no_metadata_returns_none():
    proc = _proc()
    a = _le(None, None, False)
    b = _le(_square(0, 0, 1, 1), (0.0, 1.0), True)
    assert proc._polyprism_fast_overlap(a, b) is None
    assert proc._polyprism_fast_overlap(b, a) is None


def test_z_separated_pair_returns_false():
    """Polygons overlap in xy, but z-intervals separated > fuzzy => disjoint."""
    proc = _proc(fuzzy=1e-5)
    a = _le(_square(0, 0, 10, 10), (0.0, 1.0), True)
    b = _le(_square(0, 0, 10, 10), (2.0, 3.0), True)
    assert proc._polyprism_fast_overlap(a, b) is False


def test_z_touching_pair_inside_fuzzy_returns_true_exact():
    """Two exact extrusions sharing the z=1 plane: distance 0, both exact => True."""
    proc = _proc(fuzzy=1e-5)
    a = _le(_square(0, 0, 1, 1), (0.0, 1.0), True)
    b = _le(_square(0, 0, 1, 1), (1.0, 2.0), True)
    assert proc._polyprism_fast_overlap(a, b) is True


def test_xy_disjoint_within_z_overlap_returns_false():
    """z-intervals overlap; xy polygons separated more than fuzzy => disjoint."""
    proc = _proc(fuzzy=1e-5)
    a = _le(_square(0, 0, 1, 1), (0.0, 1.0), True)
    b = _le(_square(10, 10, 1, 1), (0.0, 1.0), True)
    assert proc._polyprism_fast_overlap(a, b) is False


def test_xy_and_z_overlap_with_both_exact_returns_true():
    proc = _proc(fuzzy=1e-5)
    a = _le(_square(0, 0, 2, 2), (0.0, 1.0), True)
    b = _le(_square(1, 1, 2, 2), (0.5, 1.5), True)
    assert proc._polyprism_fast_overlap(a, b) is True


def test_xy_and_z_overlap_with_one_inexact_returns_none():
    """If either side is a tapered envelope, the cheap test is only necessary.

    Helper returns None and the caller must fall back to OCC.
    """
    proc = _proc(fuzzy=1e-5)
    a = _le(_square(0, 0, 2, 2), (0.0, 1.0), True)
    b = _le(_square(1, 1, 2, 2), (0.5, 1.5), False)  # tapered
    assert proc._polyprism_fast_overlap(a, b) is None


def test_xy_separated_by_less_than_fuzzy_is_within():
    """Sub-fuzzy gap counts as overlapping.

    Matches OCC's <= cut_fuzzy_value semantics.
    """
    proc = _proc(fuzzy=0.01)
    a = _le(_square(0, 0, 1, 1), (0.0, 1.0), True)
    b = _le(_square(1.005, 0, 1, 1), (0.0, 1.0), True)  # 0.005 gap < 0.01 fuzzy
    assert proc._polyprism_fast_overlap(a, b) is True
