"""StructuredSweep: CAD-side declaration of a structured band.

Geometry only (attachment + per-side thickness). Discretization lives in
StructuredSweepResolutionSpec, paired by ``name`` — see the design doc.
"""
from __future__ import annotations

from meshwell.structured.exceptions import SweepKeyError

_INTERFACE_DELIMITER = "___"
_BOUNDARY_SUFFIX = "None"
_POLYLINE_KEYS = frozenset({"left", "right"})


class StructuredSweep:
    """Declares a structured band grown from a dim-(N-1) physical name."""

    def __init__(self, name: str, on: str, thickness: dict[str, float]):
        if "|" in name or "|" in on:
            raise ValueError(f"'|' not allowed in sweep name/attachment: {name!r}, {on!r}")
        if not thickness:
            raise ValueError(f"Sweep {name!r}: thickness dict must not be empty")
        for side, t in thickness.items():
            if not t > 0:
                raise ValueError(f"Sweep {name!r}: thickness[{side!r}] must be > 0, got {t!r}")
        self.name = name
        self.on = on
        self.thickness = dict(thickness)
        self._validate_keys()

    @property
    def attachment_kind(self) -> str:
        parts = self.on.split(_INTERFACE_DELIMITER)
        if len(parts) == 2:
            return "boundary" if parts[1] == _BOUNDARY_SUFFIX else "interface"
        return "polyline"

    def _admissible_keys(self) -> frozenset[str]:
        kind = self.attachment_kind
        parts = self.on.split(_INTERFACE_DELIMITER)
        if kind == "interface":
            return frozenset(parts)
        if kind == "boundary":
            return frozenset({parts[0]})
        return _POLYLINE_KEYS

    def _validate_keys(self) -> None:
        admissible = self._admissible_keys()
        bad = set(self.thickness) - admissible
        if bad:
            raise SweepKeyError(self.name, self.attachment_kind, bad, admissible)

    def sides(self) -> list[str]:
        return list(self.thickness)

    def to_dict(self) -> dict:
        return {
            "type": "StructuredSweep",
            "name": self.name,
            "on": self.on,
            "thickness": self.thickness,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StructuredSweep":
        return cls(name=data["name"], on=data["on"], thickness=data["thickness"])
