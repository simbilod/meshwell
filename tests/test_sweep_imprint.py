import gmsh
import shapely

from meshwell.orchestrator import generate_mesh
from meshwell.polysurface import PolySurface
from meshwell.resolution import Graded, StructuredSweepResolutionSpec
from meshwell.structured.sweep import StructuredSweep


def _entities():
    lower = PolySurface(
        polygons=shapely.box(0, 0, 4, 1), physical_name="lower", mesh_order=2
    )
    upper = PolySurface(
        polygons=shapely.box(0, 1, 4, 2), physical_name="upper", mesh_order=1
    )
    return [lower, upper]


def test_imprint_writes_sweep_groups(tmp_path):
    xao = tmp_path / "model.xao"
    generate_mesh(
        entities=_entities(),
        sweeps=[
            StructuredSweep(name="qw", on="lower___upper", thickness={"upper": 0.4})
        ],
        dim=2,
        checkpoint_cad=xao,
        output_mesh=str(tmp_path / "out.msh"),
        default_characteristic_length=0.5,
        resolution_specs={
            "qw": [
                StructuredSweepResolutionSpec(
                    tangential=0.5, normal={"upper": Graded(h0=0.05, ratio=1.5)}
                )
            ],
        },
    )
    gmsh.initialize()
    try:
        gmsh.merge(str(xao))
        names = {
            gmsh.model.getPhysicalName(d, t) for d, t in gmsh.model.getPhysicalGroups()
        }
    finally:
        gmsh.finalize()
    assert "__sweep|qw|upper|0" in names
    assert "__sweepsrc|qw" in names
    assert "lower___upper" in names  # real interface preserved
    assert "lower" in names
    assert "upper" in names


def test_sweep_groups_stripped_from_msh(tmp_path):
    import meshio

    out = tmp_path / "out.msh"
    generate_mesh(
        entities=_entities(),
        sweeps=[
            StructuredSweep(name="qw", on="lower___upper", thickness={"upper": 0.4})
        ],
        dim=2,
        output_mesh=str(out),
        default_characteristic_length=0.5,
        resolution_specs={
            "qw": [StructuredSweepResolutionSpec(tangential=0.5, normal={"upper": 2})],
        },
    )
    m = meshio.read(out)
    assert not any(k.startswith("__sweep") for k in m.cell_sets)


def test_sweep_internal_seam_not_in_boundary_group(tmp_path):
    """The band/remainder seam inside a swept region is interior, not a boundary.

    Sweeping into "upper" (box y in [1, 2]) with thickness 0.4 splits it into
    a band face (y in [1, 1.4]) and a remainder face (y in [1.4, 2]) within the
    SAME "upper" entity. The shared y=1.4 edge is interior to the same material
    and must NOT leak into the ``upper___None`` exterior boundary group.
    """
    import meshio

    out = tmp_path / "seam.msh"
    generate_mesh(
        entities=_entities(),
        sweeps=[
            StructuredSweep(name="qw", on="lower___upper", thickness={"upper": 0.4})
        ],
        dim=2,
        output_mesh=str(out),
        default_characteristic_length=0.5,
        resolution_specs={
            "qw": [StructuredSweepResolutionSpec(tangential=1.0, normal={"upper": 2})],
        },
    )
    m = meshio.read(out)
    pts = m.points[:, :2]
    line_blocks = [i for i, cb in enumerate(m.cells) if cb.type == "line"]

    def horizontal_ys(name):
        ys = set()
        for bi in line_blocks:
            idx = m.cell_sets[name][bi]
            if idx is None:
                continue
            for ci in idx:
                a, b = m.cells[bi].data[ci]
                if abs(pts[a, 1] - pts[b, 1]) < 1e-6:
                    ys.add(round(float(pts[a, 1]), 4))
        return ys

    # The only legitimate horizontal exterior edge of "upper" is its top, y=2.0.
    assert horizontal_ys("upper___None") == {
        2.0
    }, "internal band seam (y=1.4) leaked into upper___None"
    # Sanity: real interface and the other region's boundary are unaffected.
    assert horizontal_ys("lower___upper") == {1.0}
    assert horizontal_ys("lower___None") == {0.0}
