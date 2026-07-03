"""APIs removed in the 2026-07-03 code-improvement effort stay removed."""
import inspect

import pytest
from shapely.geometry import LineString, Polygon

from meshwell.mesh import Mesh, mesh
from meshwell.occ_entity import OCC_entity
from meshwell.polyline import PolyLine
from meshwell.polyprism import PolyPrism
from meshwell.polysurface import PolySurface


def test_periodic_entities_removed_from_mesh_wrapper():
    assert "periodic_entities" not in inspect.signature(mesh).parameters


def test_periodic_entities_removed_from_process_geometry():
    assert (
        "periodic_entities" not in inspect.signature(Mesh.process_geometry).parameters
    )


def test_periodic_helper_methods_deleted():
    assert not hasattr(Mesh, "_apply_periodic_boundaries")
    assert not hasattr(Mesh, "_set_periodic_pair")


_SQUARE = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


@pytest.mark.parametrize(
    "build",
    [
        lambda: PolyPrism(
            polygons=_SQUARE,
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="x",
            additive=True,
        ),
        lambda: PolySurface(polygons=_SQUARE, physical_name="x", additive=True),
        lambda: PolyLine(
            linestrings=LineString([(0, 0), (1, 1)]), physical_name="x", additive=True
        ),
        lambda: OCC_entity(occ_function=lambda: None, physical_name="x", additive=True),
    ],
    ids=["polyprism", "polysurface", "polyline", "occ_entity"],
)
def test_additive_true_raises(build):
    with pytest.raises(NotImplementedError, match="additive"):
        build()


def test_additive_false_round_trips():
    ps = PolySurface(polygons=_SQUARE, physical_name="x", additive=False)
    assert ps.to_dict()["additive"] is False
    assert PolySurface.from_dict(ps.to_dict()).additive is False


def test_resolution_specs_default_is_none():
    default = (
        inspect.signature(Mesh.process_geometry).parameters["resolution_specs"].default
    )
    assert default is None, "tuple default () crashes .get() calls downstream"
