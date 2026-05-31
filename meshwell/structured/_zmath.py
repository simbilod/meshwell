"""Float tolerance helpers for z-plane comparisons."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def approx_in(z: float, zs: Iterable[float], tol: float = 1e-9) -> bool:
    """True if any element of zs is within tol of z (exact-equality-safe)."""
    return any(abs(z - zp) <= tol for zp in zs)


def approx_key(z: float, mapping: dict[float, Any], tol: float = 1e-9) -> float | None:
    """Return the first key in mapping within tol of z, or None."""
    for k in mapping:
        if abs(z - k) <= tol:
            return k
    return None
