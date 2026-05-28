"""Benchmark: full pipeline wall time with cohort topology ON vs OFF.

Builds a structured-heavy scene (N stacks x M layers per stack) and runs
the full pipeline three times:
1. Full legacy: _USE_COHORT_TOPOLOGY=False, _PRESHARE_VERTICAL_FACES=False
2. Phase 1 vertical-only: _USE_COHORT_TOPOLOGY=False, _PRESHARE_VERTICAL_FACES=True
3. Phase 2 full: _USE_COHORT_TOPOLOGY=True

Reports wall time and speedup ratio for each.

Only call this on structured-heavy scenes that avoid the known Phase 2
regressions (concentric arcs, hangs).
"""

from __future__ import annotations

import time

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured import phantom as phantom_mod


def _square(x0, y0, x1, y1):
    return shapely.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _build_scene(n_stacks: int = 4, layers_per_stack: int = 10):
    """N_STACKS lateral stacks * LAYERS_PER_STACK vertical layers each.

    Each stack occupies a unit-x XY footprint; stacks are separated by 1
    unit in x (so they're cohort-disjoint).
    """
    entities = []
    for stack_idx in range(n_stacks):
        x0 = stack_idx * 2.0
        x1 = x0 + 1.0
        for layer_idx in range(layers_per_stack):
            z0 = float(layer_idx)
            z1 = z0 + 1.0
            entities.append(
                PolyPrism(
                    polygons=_square(x0, 0, x1, 1),
                    buffers={z0: 0.0, z1: 0.0},
                    physical_name=f"s{stack_idx}_l{layer_idx}",
                    mesh_order=stack_idx * 100 + layer_idx,
                    structured=True,
                    resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
                )
            )
    return entities


def _time_phantom_build(entities) -> float:
    """Time build_plan + build_phantom_shapes. Returns wall-clock seconds."""
    from meshwell.structured.phantom import build_phantom_shapes
    from meshwell.structured.plan import build_plan

    plan = build_plan(entities)
    t0 = time.perf_counter()
    build_phantom_shapes(plan)
    t1 = time.perf_counter()
    return t1 - t0


def _bench(use_cohort: bool, preshare_vertical: bool, entities) -> float:
    phantom_mod._USE_COHORT_TOPOLOGY = use_cohort
    phantom_mod._PRESHARE_VERTICAL_FACES = preshare_vertical
    # Two warmup runs to amortize import / JIT effects.
    _time_phantom_build(entities)
    _time_phantom_build(entities)
    # Three timed runs; report min.
    runs = [_time_phantom_build(entities) for _ in range(3)]
    return min(runs)


def main() -> None:
    """Run the three-way benchmark and print timing results."""
    entities = _build_scene(n_stacks=4, layers_per_stack=10)
    n = len(entities)

    print(f"Scene: {n} structured PolyPrism entities (4 stacks x 10 layers)")
    print()

    legacy = _bench(False, False, entities)
    phase1 = _bench(False, True, entities)
    phase2 = _bench(True, True, entities)

    print(f"Full legacy (no sharing):             {legacy:.4f}s  (1.00x)")
    print(
        f"Phase 1 (vertical sharing only):      {phase1:.4f}s  ({legacy / phase1:.2f}x)"
    )
    print(
        f"Phase 2 (vertical + lateral sharing): {phase2:.4f}s  ({legacy / phase2:.2f}x)"
    )
    print()
    print("Note: this measures build_phantom_shapes only. The full _fragment_all")
    print("BOP perf gain (the original Phase 1 success criterion) depends on")
    print("running through cad_occ; that benchmark needs the orchestrator entry.")


if __name__ == "__main__":
    main()
