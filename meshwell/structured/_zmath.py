"""Float tolerance helpers for z-plane comparisons."""
from __future__ import annotations

from collections.abc import Iterable


def approx_in(z: float, zs: Iterable[float], tol: float = 1e-9) -> bool:
    """True if any element of zs is within tol of z (exact-equality-safe)."""
    return any(abs(z - zp) <= tol for zp in zs)
