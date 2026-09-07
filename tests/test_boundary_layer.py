import pytest
import shapely

from meshwell.orchestrator import generate_mesh
from meshwell.polysurface import PolySurface
from meshwell.resolution import (
    BoundaryLayerResolutionSpec,
    StructuredSweepResolutionSpec,
)
from meshwell.structured.sweep import StructuredSweep


def _sheet():
    return [
        PolySurface(
            polygons=shapely.box(0, 0, 1, 1), physical_name="sheet", mesh_order=1
        )
    ]


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
        PolySurface(
            polygons=shapely.box(0, 0, 4, 1), physical_name="lower", mesh_order=2
        ),
        PolySurface(
            polygons=shapely.box(0, 1, 4, 2), physical_name="upper", mesh_order=1
        ),
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
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        BoundaryLayerResolutionSpec(size=-1.0, thickness=0.05)
    with pytest.raises(ValidationError):
        BoundaryLayerResolutionSpec(size=0.01, thickness=0.0)
    with pytest.raises(ValidationError):
        BoundaryLayerResolutionSpec(size=0.01, thickness=0.05, ratio=0.5)


def test_global_none_returning_spec_does_not_crash(tmp_path):
    """A None-returning spec under the global (None) key must not corrupt the Min field.

    In this gmsh build, an unguarded None in the Min field's FieldsList doesn't
    raise -- numpy silently coerces it to NaN, which gmsh casts to an out-of-range
    int and logs "Unknown Field <garbage>" instead of crashing. We capture the
    gmsh logger to catch that silent corruption, in addition to the basic
    not-None check.
    """
    import gmsh

    if not gmsh.is_initialized():
        gmsh.initialize()
    gmsh.logger.start()
    mesh = generate_mesh(
        entities=_sheet(),
        dim=2,
        output_mesh=str(tmp_path / "glob.msh"),
        default_characteristic_length=0.1,
        resolution_specs={
            None: [BoundaryLayerResolutionSpec(size=0.01, thickness=0.05)],
        },
    )
    log = gmsh.logger.get()
    gmsh.logger.stop()
    assert mesh is not None
    assert not any("Unknown Field" in line for line in log)


def _two_boxes():
    return [
        PolySurface(
            polygons=shapely.box(0, 0, 4, 1), physical_name="lower", mesh_order=2
        ),
        PolySurface(
            polygons=shapely.box(0, 1, 4, 2), physical_name="upper", mesh_order=1
        ),
    ]


def test_boundary_layer_with_sweep_raises(tmp_path):
    with pytest.raises(ValueError, match="boundary layer"):
        generate_mesh(
            entities=_two_boxes(),
            sweeps=[
                StructuredSweep(name="qw", on="lower___upper", thickness={"upper": 0.4})
            ],
            dim=2,
            output_mesh=str(tmp_path / "conflict.msh"),
            default_characteristic_length=0.5,
            resolution_specs={
                "qw": [
                    StructuredSweepResolutionSpec(tangential=1.0, normal={"upper": 2})
                ],
                "lower___None": [
                    BoundaryLayerResolutionSpec(size=0.01, thickness=0.05)
                ],
            },
        )
