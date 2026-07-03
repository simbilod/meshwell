"""A failed thru-sections loft must warn with the entity's physical_name."""
import gmsh
import pytest
from shapely.geometry import Polygon

from meshwell.polyprism import PolyPrism


def test_thrusections_failure_warns(monkeypatch):
    prism = PolyPrism(
        polygons=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        buffers={0.0: 0.0, 1.0: 0.1},
        physical_name="my_prism",
    )

    def boom(*_args, **_kwargs):
        raise RuntimeError("synthetic loft failure")

    monkeypatch.setattr(gmsh.model.occ, "addThruSections", boom)
    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.model.add("loft_failure_test")
    try:
        with pytest.warns(UserWarning, match="my_prism"):
            tag = prism._create_volume_directly(prism.buffered_polygons[0])
        assert tag == 0
    finally:
        gmsh.model.remove()
