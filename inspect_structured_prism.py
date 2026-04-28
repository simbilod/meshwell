"""Inspect the structured PolyPrism feature with a richly-mixed scene.

Layout — three xy-disjoint structured "islands" sitting in a sea of
unstructured material, sandwiched between unstructured top/bottom slabs:

         X                                                   X
         X    UNSTRUCTURED FILLER  (mesh_order=10)           X
         X                                                   X
         X     ┌─ring resonator    ┌─multi-interval         X
         X     │  (UNSTRUCTURED    │  STRUCTURED stack       X
         X     │   annulus)        │  n_layers=[2, 3, 4]     X
         X     └────────────────   └──────────────           X
         X                                                   X
         X     ┌─wire ┌─wire ┌─wire   STRUCTURED                X
         X     │  L   │  M   │  R     wires (n_layers=4)      X
         X     │      │      │                                X
         X     └──────┴──────┴──                              X
         X                                                   X
    z=1.3 ───────────  ENCAPSULANT (unstructured)  ────────────
    z=1.0 ─────────────────  WORK ZONE  ────────────────────
                       (structured + unstructured filler)
    z=0.4 ───────────  CLADDING (unstructured, tapered)  ─────
    z=0.0

The three structured island groups (1 multi-interval stack, 3 disjoint
wires) live alongside a free-form ring resonator and an unstructured
filler that covers the rest of the work zone. Above and below are
unstructured cap / cladding layers. Because the structured islands are
xy-disjoint from each other AND the unstructured layers are above/below
(no structured-on-structured stacking), no horizontal-face mating
constraints fire.

Run:
    python inspect_structured_prism.py

Outputs:
    structured_inspect.msh   -- open in `gmsh structured_inspect.msh`
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import meshio
from shapely.geometry import Polygon

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism

OUT = Path("structured_inspect.msh")


def square(x0: float, x1: float, y0: float, y1: float) -> Polygon:
    """Axis-aligned rectangle convenience helper."""
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def annulus(
    cx: float, cy: float, r_outer: float, r_inner: float, n: int = 64
) -> Polygon:
    """Annulus polygon (outer ring with inner hole)."""
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


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------

# Cladding: tapered (z-buffered) UNSTRUCTURED prism beneath the device.
# The non-zero z-buffer forces the unstructured (BRepOffsetAPI thru-sections)
# path so the underside slopes inward.
cladding = PolyPrism(
    polygons=square(-0.5, 8.5, -0.5, 5.5),
    buffers={0.0: 0.05, 0.4: -0.05},
    physical_name="cladding",
    mesh_order=20,  # always loses
)

# Three structured "wires" (xy-disjoint), n_layers=4 over [0.4, 1.0]
WIRE_BUFFERS = {0.4: 0.0, 1.0: 0.0}
wire_l = PolyPrism(
    polygons=square(0.6, 0.9, 1.5, 4.5),
    buffers=WIRE_BUFFERS,
    n_layers=[4],
    physical_name="wire_left",
    mesh_order=2,
)
wire_m = PolyPrism(
    polygons=square(1.4, 1.7, 1.5, 4.5),
    buffers=WIRE_BUFFERS,
    n_layers=[4],
    physical_name="wire_mid",
    mesh_order=2,
)
wire_r = PolyPrism(
    polygons=square(2.2, 2.5, 1.5, 4.5),
    buffers=WIRE_BUFFERS,
    n_layers=[4],
    physical_name="wire_right",
    mesh_order=2,
)

# Multi-interval STRUCTURED stack -- one entity, three z-intervals with
# DIFFERENT layer counts per interval. Demonstrates t3-style stacked
# layer counts on a single prism. xy-disjoint from the wires.
multi_stack = PolyPrism(
    polygons=square(4.0, 5.5, 1.5, 4.5),
    buffers={0.4: 0.0, 0.6: 0.0, 0.8: 0.0, 1.0: 0.0},
    n_layers=[2, 3, 4],
    physical_name="multi_stack",
    mesh_order=3,
)

# Free-form UNSTRUCTURED ring resonator (annulus) over the cladding.
# xy-disjoint from the structured features.
ring_res = PolyPrism(
    polygons=annulus(cx=6.8, cy=3.0, r_outer=1.0, r_inner=0.5),
    buffers={0.4: 0.0, 1.0: 0.0},
    physical_name="ring",
    mesh_order=4,
)

# Big UNSTRUCTURED filler covering the rest of the work-zone xy.
# Lowest priority among work-zone entities so it loses everywhere the
# islands cut into it -- the cad_occ fragmenter resolves this.
filler = PolyPrism(
    polygons=square(-0.5, 8.5, -0.5, 5.5),
    buffers={0.4: 0.0, 1.0: 0.0},
    physical_name="filler",
    mesh_order=15,
)

# Encapsulant: UNSTRUCTURED slab above the work zone.
encapsulant = PolyPrism(
    polygons=square(-0.5, 8.5, -0.5, 5.5),
    buffers={1.0: 0.0, 1.3: 0.0},
    physical_name="encapsulant",
    mesh_order=12,
)


entities = [
    cladding,
    wire_l,
    wire_m,
    wire_r,
    multi_stack,
    ring_res,
    filler,
    encapsulant,
]


# -----------------------------------------------------------------------------
# Generate
# -----------------------------------------------------------------------------

print(f"Building mesh into {OUT} ({len(entities)} entities) ...")
generate_mesh(
    entities=entities,
    dim=3,
    output_mesh=OUT,
    default_characteristic_length=0.4,
)


# -----------------------------------------------------------------------------
# Inspect
# -----------------------------------------------------------------------------

m = meshio.read(OUT)

# Build a tag -> name map (3D entries only)
tag_to_name = {int(v[0]): k for k, v in m.field_data.items() if int(v[1]) == 3}

print()
print("=== 3D regions (volumetric physical groups) ===")
total_3d = 0
for i, block in enumerate(m.cells):
    if block.type not in ("tetra", "wedge", "hexahedron"):
        continue
    tags = m.cell_data["gmsh:physical"][i]
    unique_tags = sorted({int(t) for t in tags})
    names = [tag_to_name.get(t, f"<UNTAGGED:{t}>") for t in unique_tags]
    print(f"  {block.type:8s} count={len(block.data):>7d}  -> {', '.join(names)}")
    total_3d += len(block.data)
print(f"  {'TOTAL':8s} count={total_3d:>7d}")

print()
print("=== All physical groups ===")
for name in sorted(m.field_data.keys()):
    tag, dim = m.field_data[name]
    print(f"  dim={dim}  tag={tag:3d}  {name}")

print()
print("=== Z-layering signature ===")
z_by_xy = defaultdict(set)
for x, y, z in m.points:
    z_by_xy[(round(x, 4), round(y, 4))].add(round(z, 6))
hist = defaultdict(int)
for zs in z_by_xy.values():
    hist[len(zs)] += 1
for n_z in sorted(hist):
    print(f"  columns with {n_z:3d} z-levels: {hist[n_z]:>5d}")

# Sanity: any cells in unnamed physical groups?
untagged = sum(
    1
    for i, block in enumerate(m.cells)
    if block.type in ("tetra", "wedge", "hexahedron")
    for t in m.cell_data["gmsh:physical"][i]
    if int(t) not in tag_to_name
)
print()
if untagged:
    print(f"WARNING: {untagged} cells live in unnamed physical groups.")
else:
    print("All 3D cells are tagged with a named physical group ✓")

print()
print(f"Wrote {OUT}.  Open with:  gmsh {OUT}")
