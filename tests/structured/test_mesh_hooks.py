from shapely.geometry import Polygon

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism


def test_pre_2d_and_pre_3d_hooks_called(tmp_path):
    SQ = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    p = PolyPrism(polygons=SQ, buffers={0.0: 0.0, 1.0: 0.0}, physical_name="x")
    calls: dict[str, int] = {"pre_2d": 0, "pre_3d": 0}

    def hook_2d():
        calls["pre_2d"] += 1

    def hook_3d():
        calls["pre_3d"] += 1

    generate_mesh(
        entities=[p],
        dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
        pre_2d_hook=hook_2d,
        pre_3d_hook=hook_3d,
    )
    assert calls["pre_2d"] >= 1
    assert calls["pre_3d"] >= 1
