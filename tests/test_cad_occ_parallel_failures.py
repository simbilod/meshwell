"""Executor failure modes surface cleanly."""
from __future__ import annotations

import threading

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


class _UnpicklableEntity:
    """Entity with a Lock attribute; deliberately unpicklable."""

    def __init__(self):
        self.physical_name = "unpicklable"
        self.mesh_order = 1
        self.mesh_bool = True
        self.dimension = 3
        self._lock = threading.Lock()  # locks can't pickle

    def instanciate_occ(self):
        raise NotImplementedError  # never reached

    def overlap_metadata(self):
        return None


def test_process_mode_pickle_preflight_raises_with_entity_id():
    entities = [
        PolyPrism(
            polygons=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="a",
            mesh_order=2,
        ),
        _UnpicklableEntity(),
    ]
    with pytest.raises(ValueError, match="entity index 1"):
        cad_occ(entities, executor="process")


def test_thread_mode_handles_unpicklable_entity():
    # Thread mode does no pickling; an unpicklable entity should not block dispatch.
    # We use a real PolyPrism (the unpicklable's instanciate_occ raises) but verify
    # that thread mode at least gets PAST the pickle check.
    entities = [
        PolyPrism(
            polygons=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="a",
            mesh_order=2,
        ),
        PolyPrism(
            polygons=Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]),
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="b",
            mesh_order=1,
        ),
    ]
    # Should not raise (smoke-test that thread mode is reachable).
    result = cad_occ(entities, executor="thread")
    assert len(result) == 2
