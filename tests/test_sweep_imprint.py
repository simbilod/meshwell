import gmsh
import pytest
import shapely

from meshwell.orchestrator import generate_mesh
from meshwell.polysurface import PolySurface
from meshwell.resolution import Graded, StructuredSweepResolutionSpec
from meshwell.structured.sweep import StructuredSweep


def _entities():
    lower = PolySurface(polygons=shapely.box(0, 0, 4, 1), physical_name="lower", mesh_order=2)
    upper = PolySurface(polygons=shapely.box(0, 1, 4, 2), physical_name="upper", mesh_order=1)
    return [lower, upper]


def test_imprint_writes_sweep_groups(tmp_path):
    xao = tmp_path / "model.xao"
    generate_mesh(
        entities=_entities(),
        sweeps=[StructuredSweep(name="qw", on="lower___upper", thickness={"upper": 0.4})],
        dim=2,
        checkpoint_cad=xao,
        output_mesh=str(tmp_path / "out.msh"),
        default_characteristic_length=0.5,
        resolution_specs={
            "qw": [StructuredSweepResolutionSpec(
                tangential=0.5, normal={"upper": Graded(h0=0.05, ratio=1.5)})],
        },
    )
    gmsh.initialize()
    try:
        gmsh.merge(str(xao))
        names = {
            gmsh.model.getPhysicalName(d, t)
            for d, t in gmsh.model.getPhysicalGroups()
        }
    finally:
        gmsh.finalize()
    assert "__sweep|qw|upper|0" in names
    assert "__sweepsrc|qw" in names
    assert "lower___upper" in names       # real interface preserved
    assert "lower" in names and "upper" in names


def test_sweep_groups_stripped_from_msh(tmp_path):
    import meshio

    out = tmp_path / "out.msh"
    generate_mesh(
        entities=_entities(),
        sweeps=[StructuredSweep(name="qw", on="lower___upper", thickness={"upper": 0.4})],
        dim=2,
        output_mesh=str(out),
        default_characteristic_length=0.5,
        resolution_specs={
            "qw": [StructuredSweepResolutionSpec(tangential=0.5, normal={"upper": 2})],
        },
    )
    m = meshio.read(out)
    assert not any(k.startswith("__sweep") for k in m.cell_sets)
