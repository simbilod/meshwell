"""Small float / geometry helpers shared across the structured pipeline."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def approx_in(z: float, zs: Iterable[float], tol: float = 1e-9) -> bool:
    """True if any element of zs is within tol of z (exact-equality-safe)."""
    return any(abs(z - zp) <= tol for zp in zs)


def signed_axis(pts):
    """Signed direction of a (near-)collinear 2D point cloud's own axis.

    Returns the vector between the two farthest-apart points -- the actual
    endpoint-to-endpoint direction, preserving sign, for ANY orientation.
    A bounding-box diagonal ``max - min`` is only correct for non-negative
    slopes; for a negative-slope source it y-reflects the axis. The sign is
    canonicalised (dominant component positive) so axis-aligned sources keep
    their previous +x/+y tangent.
    """
    pts = np.asarray(pts, dtype=float)
    diffs = pts[:, None, :] - pts[None, :, :]
    i, j = np.unravel_index(
        int(np.argmax((diffs**2).sum(-1))), (len(pts), len(pts))
    )
    d = pts[j] - pts[i]
    if d[0] < 0 or (d[0] == 0 and d[1] < 0):
        d = -d
    return d
