"""Benchmark arc-fit clustering.

``CircleRegistry.from_fits`` tests every ordered pair of fits. Done in Python
that is an interpreter-level hypot per pair, and layouts with thousands of
arcs (ring resonators, curved waveguides) pay it in full.
"""
import time

import numpy as np
import pytest

from meshwell.circle_registry import ArcFit, CircleRegistry

FIT_COUNTS = (500, 2000, 8000)


@pytest.mark.slow
def test_benchmark_circle_registry_clustering():
    rng = np.random.default_rng(0)
    for count in FIT_COUNTS:
        points = rng.uniform(0, 1000, size=(count, 2))
        fits = [
            ArcFit(center=(float(x), float(y)), radius=5.0, npoints=6)
            for x, y in points
        ]
        start = time.process_time()
        reg = CircleRegistry.from_fits(fits, cluster_tolerance=1e-3)
        elapsed = time.process_time() - start

        assert len(reg.clusters) == count
        print(
            f"\nArc clustering: {count} fits, {len(reg.clusters)} clusters, "
            f"cpu={elapsed:.4f}s"
        )
