"""executor ∈ {serial, thread, process} produce the same model."""
from __future__ import annotations

import math

import pytest
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from shapely.geometry import Polygon

from meshwell.cad_occ import cad_occ
from meshwell.polyprism import PolyPrism


def _square_prism(x, y, w, h, name, mesh_order):
    return PolyPrism(
        polygons=Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name=name,
        mesh_order=mesh_order,
    )


def _pillar_grid_2x2():
    sub = _square_prism(-1, -1, 10, 10, "sub", 2)
    pillars = [
        _square_prism(x, y, 1, 1, f"p_{x}_{y}", 1) for x in (1, 5) for y in (1, 5)
    ]
    return [sub, *pillars]


def _two_tangent_squares():
    sub = _square_prism(0, 0, 10, 10, "sub", 2)
    t1 = _square_prism(2, 2, 3, 3, "t1", 1)
    t2 = _square_prism(5, 2, 3, 3, "t2", 1)
    return [sub, t1, t2]


def _same_named_disjoint_pillars():
    sub = _square_prism(-1, -1, 10, 10, "sub", 2)
    ox1 = _square_prism(1, 1, 1, 1, "ox", 1)
    ox2 = _square_prism(5, 5, 1, 1, "ox", 1)
    return [sub, ox1, ox2]


def _total_volume(labeled_entities):
    out: dict[str, float] = {}
    for ent in labeled_entities:
        if not ent.keep:
            continue
        v = 0.0
        for s in ent.shapes:
            props = GProp_GProps()
            BRepGProp.VolumeProperties_s(s, props)
            v += props.Mass()
        for name in ent.physical_name:
            out[name] = out.get(name, 0.0) + v
    return out


@pytest.mark.parametrize(
    "scene_builder",
    [_pillar_grid_2x2, _two_tangent_squares, _same_named_disjoint_pillars],
)
def test_executors_produce_same_volumes(scene_builder):
    v_serial = _total_volume(cad_occ(scene_builder(), executor="serial"))
    v_thread = _total_volume(cad_occ(scene_builder(), executor="thread"))
    v_process = _total_volume(cad_occ(scene_builder(), executor="process"))
    assert set(v_serial.keys()) == set(v_thread.keys()) == set(v_process.keys())
    for k in v_serial:
        assert math.isclose(v_serial[k], v_thread[k], rel_tol=1e-6), k
        assert math.isclose(v_serial[k], v_process[k], rel_tol=1e-6), k
