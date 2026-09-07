import shapely

from meshwell.orchestrator import generate_mesh
from meshwell.polysurface import PolySurface
from meshwell.resolution import BoundaryLayerResolutionSpec


def _sheet():
    return [PolySurface(polygons=shapely.box(0, 0, 1, 1), physical_name="sheet", mesh_order=1)]


def _first_offwall_offset(mesh, xlo=0.2, xhi=0.8):
    """Smallest positive y among interior-x nodes = first boundary-layer row off y=0."""
    pts = mesh.points[:, :2]
    interior = pts[(pts[:, 0] > xlo) & (pts[:, 0] < xhi) & (pts[:, 1] > 1e-9)]
    return float(interior[:, 1].min())


def test_boundary_layer_grades_and_quads(tmp_path):
    mesh = generate_mesh(
        entities=_sheet(),
        dim=2,
        output_mesh=str(tmp_path / "bl.msh"),
        default_characteristic_length=0.1,
        resolution_specs={
            "sheet___None": [
                BoundaryLayerResolutionSpec(
                    size=0.01, thickness=0.08, ratio=1.3, quads=True
                )
            ],
        },
    )
    # quad cells are present in the layer
    n_quad = sum(cb.data.shape[0] for cb in mesh.cells if cb.type == "quad")
    assert n_quad > 0
    # first off-wall node sits ~size (0.01) away -- far below the default CL (0.1)
    first = _first_offwall_offset(mesh)
    assert 0.004 < first < 0.02


def test_boundary_layer_triangles(tmp_path):
    mesh = generate_mesh(
        entities=_sheet(),
        dim=2,
        output_mesh=str(tmp_path / "bl_tri.msh"),
        default_characteristic_length=0.1,
        resolution_specs={
            "sheet___None": [
                BoundaryLayerResolutionSpec(
                    size=0.01, thickness=0.08, ratio=1.3, quads=False
                )
            ],
        },
    )
    n_quad = sum(cb.data.shape[0] for cb in mesh.cells if cb.type == "quad")
    assert n_quad == 0
    first = _first_offwall_offset(mesh)
    assert 0.004 < first < 0.02


def test_two_boundary_layers_distinct_params(tmp_path):
    entities = [
        PolySurface(polygons=shapely.box(0, 0, 4, 1), physical_name="lower", mesh_order=2),
        PolySurface(polygons=shapely.box(0, 1, 4, 2), physical_name="upper", mesh_order=1),
    ]
    mesh = generate_mesh(
        entities=entities,
        dim=2,
        output_mesh=str(tmp_path / "bl2.msh"),
        default_characteristic_length=0.2,
        resolution_specs={
            "lower___None": [BoundaryLayerResolutionSpec(size=0.005, thickness=0.05)],
            "upper___None": [BoundaryLayerResolutionSpec(size=0.02, thickness=0.1)],
        },
    )
    pts = mesh.points[:, :2]
    # near the bottom wall (y=0), first row ~0.005
    bot = pts[(pts[:, 0] > 1) & (pts[:, 0] < 3) & (pts[:, 1] > 1e-9)]
    assert bot[:, 1].min() < 0.012
    # near the top wall (y=2), first row ~0.02
    top = pts[(pts[:, 0] > 1) & (pts[:, 0] < 3) & (pts[:, 1] < 2 - 1e-9)]
    assert (2.0 - top[:, 1].max()) < 0.04


def test_boundary_layer_validation():
    import pytest

    with pytest.raises(Exception):
        BoundaryLayerResolutionSpec(size=-1.0, thickness=0.05)
    with pytest.raises(Exception):
        BoundaryLayerResolutionSpec(size=0.01, thickness=0.0)
    with pytest.raises(Exception):
        BoundaryLayerResolutionSpec(size=0.01, thickness=0.05, ratio=0.5)
