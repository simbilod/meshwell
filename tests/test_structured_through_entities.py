"""Regression tests for 3D entities that pass through a structured slab in z.

The bench failure that motivated these tests:
    bench_structured.py crashed with `Surface N is transfinite but has K
    corners` because BOP fragmented struct_ring slabs against pillars that
    fully crossed the slab in z, producing asymmetric bottom/top sub-face
    decompositions. _compute_face_partition did not list crossing entities
    as splitters (it only listed entities *touching* z=zlo/z=zhi), so the
    cascade's pre-CAD partition didn't account for the cuts BOP would
    later introduce.
"""
from __future__ import annotations

import math
from pathlib import Path

from shapely.geometry import Polygon

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism


def _disk(cx: float, cy: float, r: float, n: int = 24) -> Polygon:
    return Polygon(
        [
            (
                cx + r * math.cos(2 * math.pi * i / n),
                cy + r * math.sin(2 * math.pi * i / n),
            )
            for i in range(n)
        ]
    )


def _square(x0, x1, y0, y1) -> Polygon:
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _annulus(cx, cy, ro, ri, n=48):
    outer = [
        (
            cx + ro * math.cos(2 * math.pi * i / n),
            cy + ro * math.sin(2 * math.pi * i / n),
        )
        for i in range(n)
    ]
    inner = [
        (
            cx + ri * math.cos(2 * math.pi * i / n),
            cy + ri * math.sin(2 * math.pi * i / n),
        )
        for i in range(n)
    ]
    return Polygon(outer, holes=[inner])


def test_pillar_crossing_slab_partitions_symmetrically(tmp_path: Path) -> None:
    """Pillar crossing a structured slab in z induces matching cuts on z=zlo/z=zhi.

    Without ``crosses_z`` in _compute_face_partition, BOP introduces the
    pillar's intersection as a post-cascade fragmentation that ends up
    asymmetric (e.g., 2 bottom sub-faces vs 1 top sub-face), which then
    crashes mesh.generate(2) with `transfinite has 5 corners`.
    """
    slab = PolyPrism(
        polygons=_square(0, 4, 0, 4),
        buffers={0.4: 0.0, 1.0: 0.0},
        n_layers=[4],
        physical_name="slab",
        mesh_order=2,
    )
    pillar = PolyPrism(
        polygons=_disk(2.0, 2.0, 0.6, n=24),
        buffers={0.0: 0.0, 1.5: 0.0},
        physical_name="pillar",
        mesh_order=18,
    )
    out = tmp_path / "pillar_through.msh"
    generate_mesh(
        entities=[slab, pillar],
        dim=3,
        output_mesh=out,
        default_characteristic_length=0.4,
    )
    assert out.exists()


def test_multiple_pillars_through_arc_slab(tmp_path: Path) -> None:
    """Two pillars crossing an arc-bearing structured ring slab.

    Mirrors the bench's struct_ring_0 + pillar interaction that
    crashed before the crosses_z fix.
    """
    ring = PolyPrism(
        polygons=_annulus(0.0, 0.0, 1.5, 0.7, n=48),
        buffers={0.4: 0.0, 1.0: 0.0},
        n_layers=[4],
        physical_name="ring",
        mesh_order=2,
        identify_arcs=True,
    )
    pillars = [
        PolyPrism(
            polygons=_disk(1.0, 0.0, 0.3, n=24),
            buffers={0.0: 0.0, 1.5: 0.0},
            physical_name="pillar_a",
            mesh_order=18,
        ),
        PolyPrism(
            polygons=_disk(-1.0, 0.0, 0.3, n=24),
            buffers={0.0: 0.0, 1.5: 0.0},
            physical_name="pillar_b",
            mesh_order=18,
        ),
    ]
    out = tmp_path / "pillars_through_ring.msh"
    generate_mesh(
        entities=[ring, *pillars],
        dim=3,
        output_mesh=out,
        default_characteristic_length=0.3,
    )
    assert out.exists()


def test_crosses_z_predicate_adds_pillar_to_partition() -> None:
    """Cascade's face_partition has at least 2 pieces when a pillar crosses the slab.

    White-box test that locks in the crosses_z fix at unit-test
    granularity: slab minus pillar + pillar => >=2 pieces.
    """
    from meshwell.structured_polyprism import resolve_structured_slabs

    slab = PolyPrism(
        polygons=_square(0, 4, 0, 4),
        buffers={0.4: 0.0, 1.0: 0.0},
        n_layers=[4],
        physical_name="slab",
        mesh_order=2,
    )
    pillar = PolyPrism(
        polygons=_disk(2.0, 2.0, 0.6, n=24),
        buffers={0.0: 0.0, 1.5: 0.0},
        physical_name="pillar",
        mesh_order=18,
    )
    slabs = resolve_structured_slabs([slab, pillar])
    assert len(slabs) == 1
    partition = slabs[0].face_partition
    assert (
        partition is not None
    ), "Expected non-None face_partition (crossing pillar should split slab)"
    assert len(partition) >= 2, (
        f"Expected at least 2 partition pieces (slab + pillar cut), "
        f"got {len(partition)}"
    )
