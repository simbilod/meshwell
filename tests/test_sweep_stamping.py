import numpy as np
import pytest
import shapely

from meshwell.orchestrator import generate_mesh
from meshwell.polysurface import PolySurface
from meshwell.resolution import StructuredSweepResolutionSpec
from meshwell.structured.sweep import StructuredSweep


def _run(tmp_path, element_type="triangle", tangential=1.0, normal=2, thickness=0.4):
    return generate_mesh(
        entities=[
            PolySurface(
                polygons=shapely.box(0, 0, 4, 1), physical_name="lower", mesh_order=2
            ),
            PolySurface(
                polygons=shapely.box(0, 1, 4, 2), physical_name="upper", mesh_order=1
            ),
        ],
        sweeps=[
            StructuredSweep(
                name="qw", on="lower___upper", thickness={"upper": thickness}
            )
        ],
        dim=2,
        output_mesh=str(tmp_path / "out.msh"),
        default_characteristic_length=0.5,
        resolution_specs={
            "qw": [
                StructuredSweepResolutionSpec(
                    tangential=tangential,
                    normal={"upper": normal},
                    element_type=element_type,
                )
            ],
        },
    )


def _band_nodes(m, thickness=0.4):
    pts = m.points[:, :2]
    return pts[(pts[:, 1] >= 1.0 - 1e-9) & (pts[:, 1] <= 1.0 + thickness + 1e-9)]


def test_band_nodes_are_exact_tensor_grid(tmp_path):
    m = _run(tmp_path, tangential=1.0, normal=2, thickness=0.4)
    band = _band_nodes(m)
    xs = np.unique(np.round(band[:, 0], 9))
    ys = np.unique(np.round(band[:, 1], 9))
    np.testing.assert_allclose(xs, [0.0, 1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(ys, [1.0, 1.2, 1.4])
    # every grid point exists exactly once
    assert len(band) == len(xs) * len(ys)


def test_band_cells_are_right_triangles(tmp_path):
    m = _run(tmp_path)
    tri = next(cb.data for cb in m.cells if cb.type == "triangle")
    pts = m.points[:, :2]
    # collect triangles fully inside the band
    for conn in tri:
        p = pts[conn]
        if p[:, 1].min() >= 1.0 - 1e-9 and p[:, 1].max() <= 1.4 + 1e-9:
            # right triangle: one vertex has a 90 deg angle
            v = [p[(i + 1) % 3] - p[i] for i in range(3)]
            dots = [abs(np.dot(v[i], -v[(i - 1) % 3])) for i in range(3)]
            assert min(dots) == pytest.approx(0.0, abs=1e-9)


def test_quad_variant(tmp_path):
    m = _run(tmp_path, element_type="quad")
    quads = sum(cb.data.shape[0] for cb in m.cells if cb.type == "quad")
    assert quads == 4 * 2  # 4 tangential cells x 2 normal layers


def test_interface_groups_survive(tmp_path):
    m = _run(tmp_path)
    assert "lower___upper" in m.cell_sets
    assert "lower" in m.cell_sets
    assert "upper" in m.cell_sets
    assert not any(k.startswith("__sweep") for k in m.cell_sets)


def test_conformal_no_duplicate_nodes(tmp_path):
    m = _run(tmp_path)
    pts = np.round(m.points[:, :2], 9)
    assert len(np.unique(pts, axis=0)) == len(pts)
