"""Evidence-gathering benchmark for the structured-slab feature.

Builds a deliberately complex scene (many slabs, stacked structured prisms,
arc-bearing footprints, overlapping unstructured neighbours, fillers) and
times the suspected hotspots by monkey-patching them with accumulators.

Run:
    python bench_structured.py

Reports:
    1. Total wall time + per-phase (CAD vs mesh stage).
    2. Per-function cumulative time + call count for the suspected hotspots.
    3. Total number of `gmsh.model.occ.getEntities(2)` calls during the
       mesh stage (probes the per-slab repeated-scan issue).
    4. Top-25 cProfile entries by tottime, filtered to meshwell + gmsh.
"""
from __future__ import annotations

import cProfile
import math
import pstats
import time
from collections import defaultdict
from io import StringIO
from pathlib import Path

from shapely.geometry import Polygon

from meshwell import structured_polyprism as sp
from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism

OUT = Path("bench_structured.msh")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def square(x0: float, x1: float, y0: float, y1: float) -> Polygon:
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def annulus(
    cx: float, cy: float, r_outer: float, r_inner: float, n: int = 64
) -> Polygon:
    outer = [
        (
            cx + r_outer * math.cos(2 * math.pi * i / n),
            cy + r_outer * math.sin(2 * math.pi * i / n),
        )
        for i in range(n)
    ]
    inner = [
        (
            cx + r_inner * math.cos(2 * math.pi * i / n),
            cy + r_inner * math.sin(2 * math.pi * i / n),
        )
        for i in range(n)
    ]
    return Polygon(outer, holes=[inner])


def disk(cx: float, cy: float, r: float, n: int = 64) -> Polygon:
    return Polygon(
        [
            (
                cx + r * math.cos(2 * math.pi * i / n),
                cy + r * math.sin(2 * math.pi * i / n),
            )
            for i in range(n)
        ]
    )


# -----------------------------------------------------------------------------
# Scene: lots of slabs, stacks, arcs, overlaps
# -----------------------------------------------------------------------------

entities: list = []

# Layout strategy: every structured slab lives in a *protected* x-strip
# along the bottom of the domain. Unstructured stress entities live above
# in non-overlapping x-strips. They still get paired against every slab
# in the O(S * E) validators (same z-range), but they do NOT fragment
# slab top/bottom faces, so the mesh stage stays valid.

# Big unstructured cladding underneath (full domain).
entities.append(
    PolyPrism(
        polygons=square(-1, 41, -1, 17),
        buffers={0.0: 0.0, 0.4: 0.0},
        physical_name="cladding",
        mesh_order=20,
    )
)

# Work-zone unstructured filler covering ONLY the y>=8 half (away from
# structured wires). Overlaps slab z-range -> validators pair it with
# every slab, but its xy is disjoint from wires.
entities.append(
    PolyPrism(
        polygons=square(-1, 41, 8.0, 17.0),
        buffers={0.4: 0.0, 1.0: 0.0},
        physical_name="filler_top",
        mesh_order=15,
    )
)

# Encapsulant above the work zone.
entities.append(
    PolyPrism(
        polygons=square(-1, 41, -1, 17),
        buffers={1.0: 0.0, 1.5: 0.0},
        physical_name="encapsulant",
        mesh_order=12,
    )
)

# 6x4 = 24 structured wires over z=[0.4, 1.0], xy-disjoint, n_layers=4.
# Placed in y=[1, 7] strip.
WIRE_BUFFERS = {0.4: 0.0, 1.0: 0.0}
for ix in range(6):
    for iy in range(4):
        x0 = 1.0 + 4.0 * ix
        y0 = 1.0 + 1.5 * iy
        entities.append(
            PolyPrism(
                polygons=square(x0, x0 + 1.2, y0, y0 + 1.0),
                buffers=WIRE_BUFFERS,
                n_layers=[4],
                physical_name=f"wire_{ix}_{iy}",
                mesh_order=2,
            )
        )

# 4 multi-interval structured stacks over z=[0.4, 0.6, 0.8, 1.0]
# with n_layers=[2, 3, 4] -> each spawns 3 sub-slabs in the cascade.
# Located in y=[14, 16] strip, xy-disjoint from wires.
for k in range(4):
    x0 = 1.0 + 6.0 * k
    entities.append(
        PolyPrism(
            polygons=square(x0, x0 + 1.5, 14.0, 15.5),
            buffers={0.4: 0.0, 0.6: 0.0, 0.8: 0.0, 1.0: 0.0},
            n_layers=[2, 3, 4],
            physical_name=f"stack_{k}",
            mesh_order=3,
        )
    )

# 3 arc-bearing STRUCTURED rings (annuli) in y=[10, 13] strip.
for k in range(3):
    entities.append(
        PolyPrism(
            polygons=annulus(cx=4.0 + 6.0 * k, cy=11.5, r_outer=1.2, r_inner=0.6, n=48),
            buffers=WIRE_BUFFERS,
            n_layers=[4],
            physical_name=f"struct_ring_{k}",
            mesh_order=2,
            identify_arcs=True,
        )
    )

# 3 unstructured arc-bearing rings (identify_arcs=True) overlap filler_top
# in y>=8 strip -> exercises arc geometry on the unstructured side.
for k in range(3):
    entities.append(
        PolyPrism(
            polygons=annulus(cx=4.0 + 6.0 * k, cy=9.5, r_outer=0.9, r_inner=0.4, n=48),
            buffers={0.4: 0.0, 1.0: 0.0},
            physical_name=f"ring_{k}",
            mesh_order=4,
            identify_arcs=True,
        )
    )

# 8 unstructured pillars in y>=8 (NOT overlapping wires) but fully crossing
# z=[0.0, 1.5]. Each pairs with every slab in both validators.
for k in range(8):
    entities.append(
        PolyPrism(
            polygons=disk(cx=2.5 + 4.0 * k, cy=12.0, r=0.5, n=24),
            buffers={0.0: 0.0, 1.5: 0.0},
            physical_name=f"pillar_{k}",
            mesh_order=18,
        )
    )

print(
    f"Scene: {len(entities)} entities; expected ~{20 + 3*3 + 2} structured slabs after cascade"
)


# -----------------------------------------------------------------------------
# Instrumentation: monkey-patch hot functions to accumulate (time, calls).
# -----------------------------------------------------------------------------

TIMINGS: dict[str, list[float]] = defaultdict(list)


def wrap(module, name: str):
    orig = getattr(module, name)

    def wrapped(*a, **kw):
        t0 = time.perf_counter()
        try:
            return orig(*a, **kw)
        finally:
            TIMINGS[name].append(time.perf_counter() - t0)

    wrapped.__name__ = name
    setattr(module, name, wrapped)


for fn in (
    "_apply_lateral_transfinite_hints",
    "_apply_slab_horizontal_periodicity",
    "_apply_slab_vertical_periodicity",
    "_collect_slab_vertical_edges",
    "_apply_horizontal_recombine_hints",
    "_build_one_slab_conformal",
    "_find_all_occ_faces_for_slab",
    "_find_occ_face_for_slab",
    "_validate_slab_face_topology_symmetry",
    "_validate_slab_neighbour_mesh_order",
    "apply_structured_slabs",
    "resolve_structured_slabs",
):
    if hasattr(sp, fn):
        wrap(sp, fn)


# Count getEntities(2) calls (the suspected per-slab full-model scan).
import gmsh as _gmsh

_orig_get_entities = _gmsh.model.occ.getEntities
GET_ENTITIES_CALLS: dict[int, int] = defaultdict(int)


def _counting_get_entities(dim=-1):
    GET_ENTITIES_CALLS[dim] += 1
    return _orig_get_entities(dim)


_gmsh.model.occ.getEntities = _counting_get_entities


# -----------------------------------------------------------------------------
# Run under cProfile
# -----------------------------------------------------------------------------

prof = cProfile.Profile()
t_total = time.perf_counter()
prof.enable()
try:
    generate_mesh(
        entities=entities,
        dim=3,
        output_mesh=OUT,
        default_characteristic_length=0.6,
    )
    run_ok = True
    run_err = None
except Exception as exc:
    run_ok = False
    run_err = repr(exc)
prof.disable()
t_total = time.perf_counter() - t_total
print(f"Run OK: {run_ok}; err={run_err}")


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------

print()
print("=" * 78)
print(f"TOTAL wall time: {t_total:6.2f} s")
print("=" * 78)

# Phase summary.
cad_t = (
    sum(TIMINGS["resolve_structured_slabs"])
    + sum(TIMINGS["_validate_slab_face_topology_symmetry"])
    + sum(TIMINGS["_validate_slab_neighbour_mesh_order"])
)
mesh_t = sum(TIMINGS["apply_structured_slabs"])

print()
print("Phase totals:")
print(
    f"  CAD-stage slab work     : {cad_t:6.2f} s " f"({100 * cad_t / t_total:4.1f} %)"
)
print(
    f"  Mesh-stage apply_slabs  : {mesh_t:6.2f} s " f"({100 * mesh_t / t_total:4.1f} %)"
)

# Function-level breakdown.
print()
print(f"{'function':45s} {'calls':>8s} {'total s':>10s} {'mean ms':>10s}")
print("-" * 78)
rows = []
for name, times in TIMINGS.items():
    if not times:
        continue
    rows.append((sum(times), len(times), name))
rows.sort(reverse=True)
for total, count, name in rows:
    mean_ms = 1000.0 * total / count
    print(f"{name:45s} {count:8d} {total:10.3f} {mean_ms:10.2f}")

# getEntities call counts.
print()
print("gmsh.model.occ.getEntities calls (dim -> count):")
for dim, n in sorted(GET_ENTITIES_CALLS.items()):
    print(f"  dim={dim:>3d}: {n:5d} calls")

# cProfile top entries filtered to meshwell / gmsh.
print()
print("cProfile top-30 (tottime), filtered to meshwell + gmsh:")
print("-" * 78)
buf = StringIO()
stats = pstats.Stats(prof, stream=buf).sort_stats("tottime")
stats.print_stats(r"meshwell|gmsh|shapely", 30)
print(buf.getvalue())
