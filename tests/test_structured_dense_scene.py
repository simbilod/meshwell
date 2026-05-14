"""Regression tests for dense structured-slab scenes.

bench_structured.py raised:
    Conformal slab build for ('stack_0',) at z=[0.4, 0.6] could not
    locate bottom OCC face(s). This indicates a bug in phantom sub-face
    preservation; check that _StructuredPhantom volumes are removed
    non-recursively in _remove_keep_false_top_dim.

The diagnostic shows the face IS in the OCC scene with >=50% slab
footprint coverage; ``_face_belongs_to_slab`` (physical-group predicate
inside ``_find_all_occ_faces_for_slab``) rejects it. These tests
isolate the smallest scene that reproduces the rejection.
"""
from __future__ import annotations

import math
from pathlib import Path

from shapely.geometry import Polygon

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism


def _square(x0, x1, y0, y1) -> Polygon:
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _disk(cx, cy, r, n=24):
    return Polygon(
        [
            (
                cx + r * math.cos(2 * math.pi * i / n),
                cy + r * math.sin(2 * math.pi * i / n),
            )
            for i in range(n)
        ]
    )


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


def test_multi_interval_stack_with_many_neighbours(tmp_path: Path) -> None:
    """Stack of 3 structured sub-slabs alongside several non-structured neighbours.

    Stacks should mesh regardless of how many neighbours populate the
    OCC scene.
    """
    entities: list = [
        PolyPrism(
            polygons=_square(-1, 21, -1, 17),
            buffers={0.0: 0.0, 0.4: 0.0},
            physical_name="cladding",
            mesh_order=20,
        ),
        PolyPrism(
            polygons=_square(-1, 21, -1, 17),
            buffers={1.0: 0.0, 1.5: 0.0},
            physical_name="encapsulant",
            mesh_order=12,
        ),
        # Multi-interval stack -- the entity that fails in the bench.
        PolyPrism(
            polygons=_square(2.0, 4.0, 6.0, 8.0),
            buffers={0.4: 0.0, 0.6: 0.0, 0.8: 0.0, 1.0: 0.0},
            n_layers=[2, 3, 4],
            physical_name="stack",
            mesh_order=3,
        ),
        # Filler in y>=10 strip (xy-disjoint from stack).
        PolyPrism(
            polygons=_square(-1, 21, 10.0, 17.0),
            buffers={0.4: 0.0, 1.0: 0.0},
            physical_name="filler",
            mesh_order=15,
        ),
    ]
    out = tmp_path / "stack_dense.msh"
    generate_mesh(
        entities=entities, dim=3, output_mesh=out, default_characteristic_length=0.5
    )
    assert out.exists()


def test_multi_interval_stack_with_pillars_and_rings(tmp_path: Path) -> None:
    """Closer to bench_structured.py: stack + pillars + struct_rings.

    Pillars do NOT overlap the stack in xy. struct_rings are xy-disjoint
    from the stack as well. So any face-location failure on the stack
    is purely due to the broader OCC scene's complexity, not direct
    fragmentation against the stack.
    """
    entities = [
        PolyPrism(
            polygons=_square(-1, 25, -1, 18),
            buffers={0.0: 0.0, 0.4: 0.0},
            physical_name="cladding",
            mesh_order=20,
        ),
        PolyPrism(
            polygons=_square(-1, 25, -1, 18),
            buffers={1.0: 0.0, 1.5: 0.0},
            physical_name="encapsulant",
            mesh_order=12,
        ),
        PolyPrism(
            polygons=_square(2.0, 4.0, 14.0, 16.0),
            buffers={0.4: 0.0, 0.6: 0.0, 0.8: 0.0, 1.0: 0.0},
            n_layers=[2, 3, 4],
            physical_name="stack",
            mesh_order=3,
        ),
        PolyPrism(
            polygons=_annulus(8.0, 10.0, 1.2, 0.6, n=48),
            buffers={0.4: 0.0, 1.0: 0.0},
            n_layers=[4],
            physical_name="ring",
            mesh_order=2,
            identify_arcs=True,
        ),
    ]
    entities.extend(
        PolyPrism(
            polygons=_disk(2.5 + 5.0 * k, 12.0, 0.5, n=24),
            buffers={0.0: 0.0, 1.5: 0.0},
            physical_name=f"pillar_{k}",
            mesh_order=18,
        )
        for k in range(4)
    )
    out = tmp_path / "stack_pillars_rings.msh"
    generate_mesh(
        entities=entities, dim=3, output_mesh=out, default_characteristic_length=0.5
    )
    assert out.exists()
