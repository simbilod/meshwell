"""Single source of truth for the OCC/shapely/gmsh tolerance chain.

See docs/superpowers/plans/2026-05-19-tolerance-chain-redesign.md for
the hierarchy rationale.
"""
from __future__ import annotations

from dataclasses import dataclass

# OCCT's natural floor: Precision::Confusion() in OCP/OCCT.
# No fuzzy value below this is meaningful to the BOP algorithms.
OCCT_CONFUSION: float = 1e-7


@dataclass(frozen=True)
class Tolerances:
    """Validated tolerance chain for the structured meshing pipeline."""

    point_tolerance: float
    perturbation: float
    cut_fuzzy_value: float
    fragment_fuzzy_value: float
    geometry_tolerance: float
    tolerance_boolean: float
    arc_chord_height_fraction: float
