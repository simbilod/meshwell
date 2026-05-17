# ruff: noqa: S108, D103
"""Performance benchmark for the structured-polyprism pipeline.

Builds a moderately complex scene (multi-piece slabs, overlapping
neighbours, stacked slabs) and reports per-phase wall times via the
structured_logger tape.

Run::

    .venv/bin/python scripts/bench_structured.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import meshio
from shapely.geometry import Polygon

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.logging import get_timings, print_timings, reset_timings


def _square(x, y, w, h):
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def build_complex_scene():
    """A scene that exercises many code paths.

    - 1 large structured slab (4x4, n_layers=2) at z=[0, 1]
    - 4 non-structured neighbours on top, with overlapping xy footprints
      (forces multi-piece face_partition via Phase 5(d))
    - 1 structured slab stacked on top (n_layers=3) at z=[2, 3]
    - 1 unstructured cladding around it all
    """
    entities = []

    slab = PolyPrism(
        polygons=_square(0, 0, 4, 4),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="slab_lower",
        mesh_order=1.0,
    )
    entities.append(slab)

    # 4 overlapping neighbours at z=[1, 2]
    for i, (x, w) in enumerate([(0, 1.5), (1, 1.5), (2, 1.5), (2.5, 1.5)]):
        entities.append(
            PolyPrism(
                polygons=_square(x, 0, w, 4),
                buffers={1.0: 0.0, 2.0: 0.0},
                physical_name=f"top_nb_{i}",
                mesh_order=2.0 + i * 0.01,
            )
        )

    # Upper structured slab sharing the z=2 plane with the neighbours' tops
    upper = PolyPrism(
        polygons=_square(0, 0, 4, 4),
        buffers={2.0: 0.0, 3.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[3])],
        physical_name="slab_upper",
        mesh_order=1.0,
    )
    entities.append(upper)

    # Cladding around everything
    cladding = PolyPrism(
        polygons=_square(-2, -2, 8, 8),
        buffers={-1.0: 0.0, 4.0: 0.0},
        physical_name="cladding",
        mesh_order=10.0,  # lowest priority
    )
    entities.append(cladding)

    return entities


def main():
    print("=" * 70)
    print("structured-polyprism benchmark")
    print("=" * 70)

    reset_timings()
    entities = build_complex_scene()
    out_dir = Path("/tmp/bench_structured")
    out_dir.mkdir(exist_ok=True)
    out_msh = out_dir / "bench.msh"

    t0 = time.perf_counter()
    generate_mesh(
        entities,
        dim=3,
        output_mesh=out_msh,
        default_characteristic_length=0.4,
    )
    total_wall = time.perf_counter() - t0

    print(f"\nTotal wall time: {total_wall:.3f}s")
    print(f"Output: {out_msh} ({out_msh.stat().st_size:,} bytes)")

    m = meshio.read(out_msh)
    print(f"Total points: {len(m.points):,}")
    cell_summary = {}
    for cb in m.cells:
        cell_summary[cb.type] = cell_summary.get(cb.type, 0) + len(cb.data)
    print("Cell type totals:")
    for ct, n in sorted(cell_summary.items()):
        print(f"  {ct:12s} {n:>8,}")
    print(f"Physical groups: {len(m.field_data)}")

    print_timings()

    # Return as JSON for CI / programmatic use
    timings = get_timings()
    print(f"\nJSON timings keys: {sorted(timings['phases'].keys())}")
    print(f"counters: {sorted(timings['counters'].keys())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
