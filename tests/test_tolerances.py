import pytest

from meshwell.tolerances import OCCT_CONFUSION, ToleranceHierarchyError, Tolerances


def test_occt_confusion_constant():
    assert OCCT_CONFUSION == 1e-7


def test_tolerances_explicit_construction():
    tol = Tolerances(
        point_tolerance=1e-4,
        perturbation=1e-5,
        cut_fuzzy_value=1e-6,
        fragment_fuzzy_value=1e-5,
        geometry_tolerance=1e-6,
        tolerance_boolean=1e-5,
        arc_chord_height_fraction=0.01,
    )
    assert tol.point_tolerance == 1e-4
    assert tol.perturbation == 1e-5
    assert tol.cut_fuzzy_value == 1e-6
    assert tol.fragment_fuzzy_value == 1e-5


def _ok_kwargs():
    return dict(
        point_tolerance=1e-4,
        perturbation=1e-5,
        cut_fuzzy_value=1e-6,
        fragment_fuzzy_value=1e-5,
        geometry_tolerance=1e-6,
        tolerance_boolean=1e-5,
        arc_chord_height_fraction=0.01,
    )


def test_cut_fuzzy_must_not_exceed_fragment_fuzzy():
    kw = _ok_kwargs()
    kw["cut_fuzzy_value"] = 2e-5  # > fragment_fuzzy_value
    with pytest.raises(ToleranceHierarchyError, match="cut_fuzzy_value"):
        Tolerances(**kw)


def test_fragment_fuzzy_must_not_exceed_perturbation():
    kw = _ok_kwargs()
    kw["fragment_fuzzy_value"] = 2e-5  # > perturbation=1e-5
    with pytest.raises(ToleranceHierarchyError, match="fragment_fuzzy_value"):
        Tolerances(**kw)


def test_perturbation_must_not_exceed_point_tolerance():
    kw = _ok_kwargs()
    kw["perturbation"] = 2e-4  # > point_tolerance=1e-4
    with pytest.raises(ToleranceHierarchyError, match="perturbation"):
        Tolerances(**kw)


def test_cut_fuzzy_must_exceed_occt_confusion():
    kw = _ok_kwargs()
    kw["cut_fuzzy_value"] = OCCT_CONFUSION / 2
    with pytest.raises(ToleranceHierarchyError, match="OCCT_CONFUSION"):
        Tolerances(**kw)


def test_arc_chord_height_fraction_must_be_in_unit_interval():
    kw = _ok_kwargs()
    kw["arc_chord_height_fraction"] = 1.5
    with pytest.raises(ToleranceHierarchyError, match="arc_chord_height_fraction"):
        Tolerances(**kw)


def test_perturbation_must_exceed_cut_fuzzy_by_safety_factor():
    """Perturbation gap must exceed cut_fuzzy by at least 2x or OCC may merge."""
    kw = _ok_kwargs()
    kw["cut_fuzzy_value"] = 1e-6
    kw["fragment_fuzzy_value"] = 1.5e-6
    kw["perturbation"] = 1.5e-6  # only 1.5x cut_fuzzy=1e-6
    with pytest.raises(ToleranceHierarchyError, match=r"perturbation.*cut_fuzzy"):
        Tolerances(**kw)
