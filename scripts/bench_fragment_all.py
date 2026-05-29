"""Benchmark: cad_occ._fragment_all wall time with cohort topology ON vs OFF.

Drives the cad_occ pipeline (the part the orchestrator runs for structured
scenes) and times the _fragment_all BOP pass specifically by monkey-patching
the CAD_OCC class method to record timing on each invocation.

Three configurations measured:
1. Full legacy: _USE_COHORT_TOPOLOGY=False, _PRESHARE_VERTICAL_FACES=False
2. Phase 1 vertical-only: _USE_COHORT_TOPOLOGY=False, _PRESHARE_VERTICAL_FACES=True
3. Phase 2 full: _USE_COHORT_TOPOLOGY=True

The scene includes both vertical stacking (multiple z-layers per XY footprint)
and lateral adjacency (adjacent stacks sharing an edge) to exercise the cases
where Phase 1 and Phase 2 each give benefit.

Avoids the known Phase 2 regressions (concentric arcs, hangs).
"""

from __future__ import annotations

import time

import shapely

from meshwell.cad_occ import CAD_OCC, cad_occ
from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured import phantom as phantom_mod


def _square(x0, y0, x1, y1):
    return shapely.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _build_scene(n_stacks: int = 4, layers_per_stack: int = 6):
    """N_STACKS lateral neighbors x LAYERS_PER_STACK vertical layers each.

    Stacks are placed edge-to-edge in X so adjacent stacks share a lateral
    seam. Layers in the same stack share top/bottom interfaces.
    """
    entities = []
    for stack_idx in range(n_stacks):
        x0 = float(stack_idx)
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


def _run_cad_occ_structured(entities) -> None:
    """Replicate the orchestrator's structured pipeline up to and including cad_occ.

    Stops before XAO emit + mesh so we only measure the CAD-side cost.
    """
    from meshwell.cad_common import prepare_entities
    from meshwell.structured import build_phantom_shapes, build_plan
    from meshwell.structured.phantom import _group_phantom_solids_by_entity

    prepare_entities(entities, perturbation=1e-5, resolve_snap=1e-3)
    plan = build_plan(entities)
    phantom_result = build_phantom_shapes(plan)
    overrides = _group_phantom_solids_by_entity(plan, phantom_result)
    cad_occ(entities, entity_shape_overrides=overrides)


def _bench_fragment_all(
    use_cohort: bool,
    preshare_vertical: bool,
    entities,
    use_discrete_cohort_mesh: bool = False,
) -> float:
    """Return min wall-clock seconds spent in _fragment_all over 3 timed runs.

    Monkey-patches CAD_OCC._fragment_all to capture per-call duration.
    """
    phantom_mod._USE_COHORT_TOPOLOGY = use_cohort
    phantom_mod._PRESHARE_VERTICAL_FACES = preshare_vertical
    phantom_mod._USE_DISCRETE_COHORT_MESH = use_discrete_cohort_mesh

    times: list[float] = []
    original = CAD_OCC._fragment_all

    def timed_fragment_all(self, entities, *args, **kwargs):
        t0 = time.perf_counter()
        result = original(self, entities, *args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        return result

    CAD_OCC._fragment_all = timed_fragment_all
    try:
        # One warmup run.
        _run_cad_occ_structured(entities)
        times.clear()
        # Three timed runs.
        for _ in range(3):
            _run_cad_occ_structured(entities)
    finally:
        CAD_OCC._fragment_all = original
        phantom_mod._USE_DISCRETE_COHORT_MESH = False

    if not times:
        msg = "_fragment_all was not invoked during benchmark"
        raise RuntimeError(msg)
    return min(times)


def _run_size(n_stacks: int, layers: int) -> None:
    entities = _build_scene(n_stacks=n_stacks, layers_per_stack=layers)
    n = len(entities)
    print(
        f"--- Scene: {n} entities ({n_stacks} stacks x {layers} layers, "
        f"edge-to-edge laterally) ---"
    )
    legacy = _bench_fragment_all(False, False, entities)
    phase1 = _bench_fragment_all(False, True, entities)
    phase2 = _bench_fragment_all(True, True, entities)
    phase3 = _bench_fragment_all(False, False, entities, use_discrete_cohort_mesh=True)
    print(f"  Full legacy:                {legacy:.4f}s  (1.00x)")
    print(f"  Phase 1 (vertical only):    {phase1:.4f}s  ({legacy / phase1:.2f}x)")
    print(f"  Phase 2 (vertical+lateral): {phase2:.4f}s  ({legacy / phase2:.2f}x)")
    print(f"  Phase 3 (cohort envelope):  {phase3:.4f}s  ({legacy / phase3:.2f}x)")
    print()


def main() -> None:
    """Run the bench across multiple scene sizes."""
    print("_fragment_all wall time (min of 3 runs, after 1 warmup)")
    print()
    for n_stacks, layers in [(4, 6), (4, 12), (8, 6), (8, 12)]:
        _run_size(n_stacks, layers)


if __name__ == "__main__":
    main()
