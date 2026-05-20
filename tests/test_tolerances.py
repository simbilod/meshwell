from meshwell.tolerances import OCCT_CONFUSION, Tolerances


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
