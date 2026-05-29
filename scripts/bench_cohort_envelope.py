"""Bench the cohort envelope builder vs Phase 1+2 paths.

Times build_phantom_shapes under each kill-switch setting on a
synthetic NxN grid of stacked square slabs.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from unittest.mock import patch

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.phantom import build_phantom_shapes
from meshwell.structured.plan import build_plan


@contextmanager
def _timed(label: str):
    t0 = time.perf_counter()
    yield
    print(f"{label}: {time.perf_counter() - t0:.3f}s")


def _grid(n_xy: int, n_z: int):
    return [
        PolyPrism(
            polygons=shapely.box(i, j, i + 1, j + 1),
            buffers={k + 0.0: 0.0, k + 1.0: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
            physical_name=f"S_{i}_{j}_{k}",
        )
        for k in range(n_z)
        for i in range(n_xy)
        for j in range(n_xy)
    ]


def main() -> None:
    """Run the bench across multiple grid sizes."""
    for n_xy, n_z in [(4, 2), (6, 3), (8, 4)]:
        plan = build_plan(_grid(n_xy, n_z))
        n_pieces = sum(len(s.face_partition) for s in plan.slabs)
        n_cohorts = len({s.component_index for s in plan.slabs})
        print(
            f"\n=== grid={n_xy}x{n_xy} z={n_z} pieces={n_pieces} "
            f"cohorts={n_cohorts} ==="
        )
        try:
            with _timed("phase1 (pre-shared faces)"):
                build_phantom_shapes(plan)
        except Exception as exc:
            print(f"phase1 ERROR: {exc}")
        try:
            with patch(
                "meshwell.structured.phantom._USE_COHORT_TOPOLOGY", True
            ), _timed("phase2 (cohort topology)"):
                build_phantom_shapes(plan)
        except Exception as exc:
            print(f"phase2 ERROR: {exc}")
        try:
            with patch(
                "meshwell.structured.phantom._USE_DISCRETE_COHORT_MESH", True
            ), _timed("phase3 (cohort envelope)"):
                build_phantom_shapes(plan)
        except Exception as exc:
            print(f"phase3 ERROR: {exc}")


if __name__ == "__main__":
    main()
