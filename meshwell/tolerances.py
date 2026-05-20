"""Single source of truth for the OCC/shapely/gmsh tolerance chain.

See docs/superpowers/plans/2026-05-19-tolerance-chain-redesign.md for
the hierarchy rationale.
"""
from __future__ import annotations

from dataclasses import dataclass

# OCCT's natural floor: Precision::Confusion() in OCP/OCCT.
# No fuzzy value below this is meaningful to the BOP algorithms.
OCCT_CONFUSION: float = 1e-7


class ToleranceHierarchyError(ValueError):
    """Raised when a Tolerances instance violates the required hierarchy."""


# Minimum safety factor between perturbation gap and cut_fuzzy.
# 2x means the buffered overlap must be at least twice the BOP merge
# distance, which prevents accumulated shape-tolerance drift from
# silently welding the carved face. Smaller margins (1.5x) have been
# observed to fail under the cut cascade's tolerance bloat.
_PERTURBATION_SAFETY_FACTOR: float = 2.0


def _validate(t: "Tolerances") -> None:
    if t.cut_fuzzy_value < OCCT_CONFUSION:
        raise ToleranceHierarchyError(
            f"cut_fuzzy_value={t.cut_fuzzy_value} < OCCT_CONFUSION={OCCT_CONFUSION}; "
            "OCC BOP algorithms cannot resolve below Precision::Confusion."
        )
    if t.cut_fuzzy_value > t.fragment_fuzzy_value:
        raise ToleranceHierarchyError(
            f"cut_fuzzy_value={t.cut_fuzzy_value} > "
            f"fragment_fuzzy_value={t.fragment_fuzzy_value}; "
            "fragment pass must be at least as loose as per-cut pass."
        )
    if t.fragment_fuzzy_value > t.perturbation:
        raise ToleranceHierarchyError(
            f"fragment_fuzzy_value={t.fragment_fuzzy_value} > "
            f"perturbation={t.perturbation}; fragment fuzzy must not exceed "
            "the buffered overlap or it would weld the carved face."
        )
    if t.perturbation < _PERTURBATION_SAFETY_FACTOR * t.cut_fuzzy_value:
        raise ToleranceHierarchyError(
            f"perturbation={t.perturbation} < "
            f"{_PERTURBATION_SAFETY_FACTOR}x cut_fuzzy_value={t.cut_fuzzy_value}; "
            "perturbation must exceed cut_fuzzy by safety factor or "
            "tolerance bloat will weld the buffered gap."
        )
    if t.perturbation > t.point_tolerance:
        raise ToleranceHierarchyError(
            f"perturbation={t.perturbation} > point_tolerance={t.point_tolerance}; "
            "shapely set_precision (using point_tolerance) would erase the "
            "buffer before OCC ever sees it."
        )
    if not (0.0 < t.arc_chord_height_fraction <= 1.0):
        raise ToleranceHierarchyError(
            f"arc_chord_height_fraction={t.arc_chord_height_fraction} "
            "must be in (0, 1]."
        )


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

    def __post_init__(self) -> None:
        """Validate the tolerance hierarchy after initialization."""
        _validate(self)

    @classmethod
    def from_characteristic_length(cls, L: float) -> "Tolerances":
        """Derive a coherent tolerance chain from one length scale.

        ``L`` is the characteristic geometric size of the scene (e.g.
        1.0 for unit-scale, 1e-3 if the user works in metres on
        mm-scale geometry, 1e-6 for micron-scale photonics in metres).

        Defaults at L=1:
            point_tolerance      1e-4 L  (shapely dedup; loose)
            perturbation         1e-5 L  (polygon outward buffer)
            cut_fuzzy_value      1e-6 L  (BRepAlgoAPI_Cut fuzzy; tight)
            fragment_fuzzy_value 1e-5 L  (BOPAlgo_Builder fuzzy)
            geometry_tolerance   1e-6 L  (gmsh vertex snap)
            tolerance_boolean    1e-5 L  (gmsh Geometry.ToleranceBoolean)
            arc_chord_height_fraction 0.01  (dimensionless)
        """
        return cls(
            point_tolerance=1e-4 * L,
            perturbation=1e-5 * L,
            cut_fuzzy_value=1e-6 * L,
            fragment_fuzzy_value=1e-5 * L,
            geometry_tolerance=1e-6 * L,
            tolerance_boolean=1e-5 * L,
            arc_chord_height_fraction=0.01,
        )
