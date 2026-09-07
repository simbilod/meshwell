import pytest
import shapely

from meshwell.mesh import mesh
from meshwell.orchestrator import generate_mesh
from meshwell.polysurface import PolySurface
from meshwell.resolution import StructuredSweepResolutionSpec
from meshwell.structured.exceptions import SweepPairingError
from meshwell.structured.sweep import StructuredSweep


def _entities():
    lower = PolySurface(
        polygons=shapely.box(0, 0, 4, 1), physical_name="lower", mesh_order=2
    )
    upper = PolySurface(
        polygons=shapely.box(0, 1, 4, 2), physical_name="upper", mesh_order=1
    )
    return [lower, upper]


def _cad(tmp_path):
    xao = tmp_path / "model.xao"
    generate_mesh(
        entities=_entities(),
        sweeps=[
            StructuredSweep(name="qw", on="lower___upper", thickness={"upper": 0.4})
        ],
        dim=2,
        checkpoint_cad=xao,
        output_mesh=str(tmp_path / "cadstep.msh"),
        default_characteristic_length=0.5,
        resolution_specs={
            "qw": [StructuredSweepResolutionSpec(tangential=0.5, normal={"upper": 2})]
        },
    )
    return xao


def test_unpaired_group_raises(tmp_path):
    xao = _cad(tmp_path)
    with pytest.raises(SweepPairingError, match="qw"):
        mesh(
            dim=2,
            input_file=xao,
            output_file=str(tmp_path / "out.msh"),
            default_characteristic_length=0.5,
            resolution_specs={},
        )


def test_unpaired_spec_raises(tmp_path):
    xao = _cad(tmp_path)
    with pytest.raises(SweepPairingError, match="ghost"):
        mesh(
            dim=2,
            input_file=xao,
            output_file=str(tmp_path / "out.msh"),
            default_characteristic_length=0.5,
            resolution_specs={
                "qw": [
                    StructuredSweepResolutionSpec(tangential=0.5, normal={"upper": 2})
                ],
                "ghost": [
                    StructuredSweepResolutionSpec(tangential=0.5, normal={"upper": 2})
                ],
            },
        )


def test_paired_separate_steps_mesh_succeeds(tmp_path):
    xao = _cad(tmp_path)
    m = mesh(
        dim=2,
        input_file=xao,
        output_file=str(tmp_path / "out.msh"),
        default_characteristic_length=0.5,
        resolution_specs={
            "qw": [StructuredSweepResolutionSpec(tangential=0.5, normal={"upper": 2})],
        },
    )
    assert m is not None
