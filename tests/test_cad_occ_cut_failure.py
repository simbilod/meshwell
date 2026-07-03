"""A failed BRepAlgoAPI_Cut must warn and keep the uncut shape, not crash."""
import pytest
from shapely.geometry import Polygon

import meshwell.cad_occ as cad_occ_mod
from meshwell.polysurface import PolySurface


class _FailingCut:
    def __init__(self, *args):
        pass

    def SetFuzzyValue(self, v):
        pass

    def Build(self):
        pass

    def IsDone(self):
        return False

    def Shape(self):  # pragma: no cover — must not be reached
        raise AssertionError("Shape() must not be called when IsDone() is False")


def test_failed_cut_warns_and_keeps_shape(monkeypatch):
    monkeypatch.setattr(cad_occ_mod, "BRepAlgoAPI_Cut", _FailingCut)
    # two overlapping same-dim entities with different mesh_order → a cut is attempted
    a = PolySurface(
        polygons=Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        physical_name="a",
        mesh_order=1,
    )
    b = PolySurface(
        polygons=Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
        physical_name="b",
        mesh_order=2,
    )
    proc = cad_occ_mod.CAD_OCC()
    with pytest.warns(UserWarning, match="Cut"):
        labeled = proc.process_entities_cut_only([a, b])
    # entity b keeps its (uncut) shape rather than losing it
    assert all(le.shapes for le in labeled)
