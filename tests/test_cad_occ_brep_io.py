"""BREP bytes round-trip preserves shape structure and volume."""
from __future__ import annotations

import math

from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from shapely.geometry import Polygon

from meshwell._brep_io import brep_from_bytes, brep_to_bytes
from meshwell.cad_occ import CAD_OCC
from meshwell.polyprism import PolyPrism


def _make_shape():
    prism = PolyPrism(
        polygons=Polygon([(0, 0), (2, 0), (2, 3), (0, 3)]),
        buffers={0.0: 0.0, 1.5: 0.0},
        physical_name="x",
        mesh_order=1,
    )
    return CAD_OCC()._instantiate_entity_occ(0, prism).shapes[0]


def _volume(shape):
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return props.Mass()


def _solid_count(shape) -> int:
    n = 0
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        n += 1
        exp.Next()
    return n


def test_round_trip_preserves_volume():
    shape = _make_shape()
    blob = brep_to_bytes(shape)
    assert isinstance(blob, bytes)
    assert len(blob) > 0
    restored = brep_from_bytes(blob)
    assert math.isclose(_volume(restored), _volume(shape), rel_tol=1e-9)


def test_round_trip_preserves_solid_count():
    shape = _make_shape()
    restored = brep_from_bytes(brep_to_bytes(shape))
    assert _solid_count(restored) == _solid_count(shape)
