"""Sweep N x workers x executor for cad_occ parallel pipeline.

Records per-scene wall-clock. See
docs/superpowers/specs/2026-05-19-cad-occ-parallel-entity-cuts-design.md
for acceptance criteria.
"""
from __future__ import annotations

import argparse
import time

from shapely.geometry import Polygon

from meshwell.cad_occ import cad_occ
from meshwell.polyprism import PolyPrism


def _scene(n: int):
    sub = PolyPrism(
        polygons=Polygon([(-5, -5), (n + 5, -5), (n + 5, n + 5), (-5, n + 5)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="sub",
        mesh_order=2,
    )
    pillars = [
        PolyPrism(
            polygons=Polygon([(i, j), (i + 0.5, j), (i + 0.5, j + 0.5), (i, j + 0.5)]),
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name=f"p_{i}_{j}",
            mesh_order=1,
        )
        for i in range(n)
        for j in range(n)
    ]
    return [sub, *pillars]


def main():
    """Run the cad_occ parallel pipeline sweep and print wall-clock timings."""
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, nargs="+", default=[5, 10])
    p.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--executors", nargs="+", default=["serial", "thread", "process"])
    args = p.parse_args()

    print(f"{'N':>5} {'workers':>8} {'executor':>10} {'wall_ms':>10}")
    for n in args.N:
        scene_template = _scene(n)
        for w in args.workers:
            for ex in args.executors:
                t0 = time.perf_counter()
                cad_occ(list(scene_template), n_threads=w, executor=ex)
                ms = (time.perf_counter() - t0) * 1000.0
                print(f"{n:>5} {w:>8} {ex:>10} {ms:>10.1f}")


if __name__ == "__main__":
    main()
