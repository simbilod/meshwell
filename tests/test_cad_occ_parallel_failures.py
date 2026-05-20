"""Executor failure modes surface cleanly."""
from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from meshwell.cad_occ import cad_occ
from meshwell.polyprism import PolyPrism


def test_unknown_executor_raises():
    entities = [
        PolyPrism(
            polygons=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="a",
            mesh_order=1,
        )
    ]
    with pytest.raises(ValueError, match="unknown executor"):
        cad_occ(entities, executor="nope")


@pytest.mark.parametrize("mode", ["serial", "thread", "process", "auto", "legacy"])
def test_all_executors_handle_empty_input(mode):
    assert cad_occ([], executor=mode) == []
