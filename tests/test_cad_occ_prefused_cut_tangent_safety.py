"""Regression test: cut cascade must not corrupt the substrate when tools are tangent."""
from __future__ import annotations

import math

from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from shapely.geometry import Polygon

from meshwell.cad_occ import CAD_OCC
from meshwell.polyprism import PolyPrism


def _square(x: float, y: float, w: float, h: float) -> Polygon:
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def _disc(cx: float, cy: float, r: float, n: int = 64) -> Polygon:
    return Polygon(
        [
            (
                cx + r * math.cos(2 * math.pi * i / n),
                cy + r * math.sin(2 * math.pi * i / n),
            )
            for i in range(n)
        ]
    )


def _solid_count(shape) -> int:
    n = 0
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        n += 1
        exp.Next()
    return n


def _volume(shape) -> float:
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return props.Mass()


def _run(entities):
    # ``entities[0]`` is the substrate. In cad_occ's cascade the substrate
    # is the HIGHER-mesh_order entity (it's the one being carved by the
    # lower-mesh_order tools — same convention as the dense-scene bench).
    # ``process_entities_cut_only`` returns entities in *original*
    # insertion order, so ``labeled[0]`` is the substrate.
    labeled = CAD_OCC().process_entities_cut_only(entities)
    return labeled[0]


def test_two_tangent_squares_on_square_substrate():
    sub = PolyPrism(
        polygons=_square(0, 0, 10, 10),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="sub",
        mesh_order=2,
    )
    t1 = PolyPrism(
        polygons=_square(2, 2, 3, 3),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="t1",
        mesh_order=1,
    )
    t2 = PolyPrism(
        polygons=_square(5, 2, 3, 3),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="t2",
        mesh_order=1,
    )
    labeled = _run([sub, t1, t2])
    assert sum(_solid_count(s) for s in labeled.shapes) == 1
    v = sum(_volume(s) for s in labeled.shapes)
    assert 0.5 * 100.0 < v < 100.0  # material removed, substrate didn't vanish


def test_ten_tangent_squares_on_square_substrate():
    sub = PolyPrism(
        polygons=_square(0, 0, 20, 5),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="sub",
        mesh_order=2,
    )
    # Tools sit flush to the bottom edge so the post-cut substrate stays
    # connected (single SOLID with a strip of material along the top).
    # Tangency between adjacent tools (they share the vertical face at
    # x = i*2.0) is the load-bearing condition this test exercises.
    tools = [
        PolyPrism(
            polygons=_square(i * 2.0, 0.0, 2.0, 2.0),
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name=f"t{i}",
            mesh_order=1,
        )
        for i in range(10)
    ]
    labeled = _run([sub, *tools])
    assert sum(_solid_count(s) for s in labeled.shapes) == 1
    v = sum(_volume(s) for s in labeled.shapes)
    assert 0.5 * 100.0 < v < 100.0


def test_three_tangent_discs_on_disc_substrate():
    sub = PolyPrism(
        polygons=_disc(0, 0, 10),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="sub",
        mesh_order=2,
    )
    # Three tangent discs along the x-axis: centers at -4, 0, 4, radius 2.
    tools = [
        PolyPrism(
            polygons=_disc(cx, 0.0, 2.0),
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name=f"d{i}",
            mesh_order=1,
        )
        for i, cx in enumerate([-4.0, 0.0, 4.0])
    ]
    labeled = _run([sub, *tools])
    assert sum(_solid_count(s) for s in labeled.shapes) == 1
    v_uncut = math.pi * 100.0
    v = sum(_volume(s) for s in labeled.shapes)
    assert v < v_uncut
    assert v > 0.5 * v_uncut
