"""Parity: pipeline output is invariant under _USE_COHORT_TOPOLOGY for supported scenes.

The cohort topology builder must produce the same observable entity
structure (piece counts per source entity) as the legacy path. This guards
against silent semantic drift.

Uses a synthetic supported-subset scene (straight squares only) to avoid
the known regressions (concentric arcs, hang).
"""

from __future__ import annotations

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured import phantom as phantom_mod
from meshwell.structured.phantom import build_phantom_shapes
from meshwell.structured.plan import build_plan


def _square(x0, y0, x1, y1):
    return shapely.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _polyprism(name, z0, z1, mesh_order, x0=0, y0=0, x1=1, y1=1, structured=True):
    return PolyPrism(
        polygons=_square(x0, y0, x1, y1),
        buffers={float(z0): 0.0, float(z1): 0.0},
        physical_name=name,
        mesh_order=mesh_order,
        structured=structured,
        resolutions=(
            [StructuredExtrusionResolutionSpec(n_layers=[2])] if structured else []
        ),
    )


def _scene():
    """3 vertically stacked + 2 lateral neighbors + 1 unstructured neighbor."""
    A = _polyprism("A", 0, 1, 1, x0=0, y0=0, x1=1, y1=1)
    B = _polyprism("B", 1, 2, 2, x0=0, y0=0, x1=1, y1=1)
    C = _polyprism("C", 2, 3, 3, x0=0, y0=0, x1=1, y1=1)
    D = _polyprism("D", 0, 1, 4, x0=5, y0=0, x1=6, y1=1)
    E = _polyprism("E", 0, 1, 5, x0=6, y0=0, x1=7, y1=1)
    F = _polyprism("F", 0, 1, 6, x0=20, y0=20, x1=21, y1=21, structured=False)
    return [A, B, C, D, E, F]


def _run_pipeline(entities):
    """Build plan + phantom shapes; return per-source-entity (source_index, piece_count)."""
    plan = build_plan(entities)
    result = build_phantom_shapes(plan)
    by_source: dict[int, int] = {}
    for ps in result.shapes:
        slab = plan.slabs[ps.slab_index]
        by_source[slab.source_index] = by_source.get(slab.source_index, 0) + 1
    return tuple(sorted(by_source.items()))


def test_use_cohort_topology_does_not_change_entity_signature():
    """Same entities -> same per-entity (source_index, piece_count) signature."""
    entities = _scene()

    prior = phantom_mod._USE_COHORT_TOPOLOGY
    phantom_mod._USE_COHORT_TOPOLOGY = False
    try:
        baseline_sig = _run_pipeline(entities)
    finally:
        phantom_mod._USE_COHORT_TOPOLOGY = prior

    phantom_mod._USE_COHORT_TOPOLOGY = True
    try:
        new_sig = _run_pipeline(entities)
    finally:
        phantom_mod._USE_COHORT_TOPOLOGY = prior

    assert baseline_sig == new_sig, (
        f"Pipeline output signature differs between cohort path off vs on:\n"
        f"  baseline (off): {baseline_sig}\n"
        f"  new (on):       {new_sig}"
    )
