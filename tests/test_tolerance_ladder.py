"""The cad_occ tolerance ladder must be validated, not assumed.

The pipeline invariants (documented in CAD_OCC.__init__):
  cut_fuzzy_value < perturbation          (loose cut erases the carved face)
  2*perturbation  < fragment_fuzzy_value  (tight fragment leaves coincident
                                           faces with distinct TShapes and
                                           silently drops A___B interfaces)
"""
import pytest

from meshwell.cad_occ import CAD_OCC
from meshwell.validation import validate_tolerance_ladder


def test_default_ladder_is_silent(recwarn):
    CAD_OCC()
    ladder_warnings = [w for w in recwarn.list if "fuzzy" in str(w.message)]
    assert not ladder_warnings


def test_tight_point_tolerance_warns():
    # point_tolerance=1e-5 -> fragment fuzzy 1e-5 < 2*perturbation (2e-5):
    # the documented interface-dropping regime must not be silent.
    with pytest.warns(UserWarning, match="2\\*perturbation"):
        CAD_OCC(point_tolerance=1e-5)


def test_inverted_ladder_raises():
    with pytest.raises(ValueError, match="fragment_fuzzy_value"):
        CAD_OCC(cut_fuzzy_value=5e-6, fragment_fuzzy_value=1e-6)


def test_loose_cut_fuzzy_warns():
    with pytest.warns(UserWarning, match="cut_fuzzy_value"):
        CAD_OCC(cut_fuzzy_value=2e-5)  # >= perturbation (1e-5)


def test_validate_function_direct():
    # No perturbation -> no perturbation-relative checks fire.
    validate_tolerance_ladder(
        perturbation=0.0, cut_fuzzy_value=1e-6, fragment_fuzzy_value=1e-3
    )
