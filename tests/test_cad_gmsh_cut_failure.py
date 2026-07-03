"""gmsh cut failures must warn even with progress_bars=False (the default)."""
import gmsh
import pytest
from shapely.geometry import Polygon

from meshwell.cad_gmsh import CAD_GMSH
from meshwell.polysurface import PolySurface


def test_cut_failure_warns_without_progress_bars(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("synthetic cut failure")

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
    proc = CAD_GMSH()
    monkeypatch.setattr(gmsh.model.occ, "cut", boom)
    try:
        with pytest.warns(UserWarning, match="Cut failed"):
            proc.process_entities([a, b], progress_bars=False)
    finally:
        proc.model_manager.finalize()
