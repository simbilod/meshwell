import numpy as np
import pytest
import shapely

from meshwell.mesh import mesh
from meshwell.orchestrator import generate_mesh
from meshwell.polyline import PolyLine
from meshwell.polysurface import PolySurface
from meshwell.resolution import Graded, StructuredSweepResolutionSpec
from meshwell.structured.exceptions import SweepSplitCoordinateError
from meshwell.structured.sweep import StructuredSweep


def test_ridge_split_edges(tmp_path):
    """A ridge on top splits the band's top seam into 3 curves; explicit
    tangential array containing the ridge corners (x=1.5, 2.5) succeeds."""
    entities = [
        PolySurface(polygons=shapely.box(0, 0, 4, 1), physical_name="layer", mesh_order=3),
        PolySurface(polygons=shapely.box(1.5, 1, 2.5, 2), physical_name="ridge", mesh_order=1),
        PolySurface(polygons=shapely.box(0, 1, 4, 2), physical_name="clad", mesh_order=2),
    ]
    m = generate_mesh(
        entities=entities,
        sweeps=[StructuredSweep(name="top", on="layer___clad", thickness={"layer": 0.3})],
        dim=2,
        output_mesh=str(tmp_path / "ridge.msh"),
        default_characteristic_length=0.5,
        resolution_specs={
            "top": [StructuredSweepResolutionSpec(
                tangential=[0.0, 0.75, 1.5, 2.0, 2.5, 3.25, 4.0],
                normal={"layer": 3})],
        },
    )
    band = m.points[(m.points[:, 1] >= 0.7 - 1e-9) & (m.points[:, 1] <= 1.0 + 1e-9), 0]
    assert {1.5, 2.5} <= set(np.round(np.unique(band), 9))


def test_ridge_split_without_member_coordinate_raises(tmp_path):
    entities = [
        PolySurface(polygons=shapely.box(0, 0, 4, 1), physical_name="layer", mesh_order=3),
        PolySurface(polygons=shapely.box(1.5, 1, 2.5, 2), physical_name="ridge", mesh_order=1),
        PolySurface(polygons=shapely.box(0, 1, 4, 2), physical_name="clad", mesh_order=2),
    ]
    with pytest.raises(SweepSplitCoordinateError):
        generate_mesh(
            entities=entities,
            sweeps=[StructuredSweep(name="top", on="layer___clad", thickness={"layer": 0.3})],
            dim=2,
            output_mesh=str(tmp_path / "ridge2.msh"),
            default_characteristic_length=0.5,
            resolution_specs={
                "top": [StructuredSweepResolutionSpec(tangential=1.0, normal={"layer": 3})],
            },
        )


def test_polyline_two_sided_different_grading(tmp_path):
    entities = [
        PolySurface(polygons=shapely.box(0, 0, 4, 2), physical_name="bulk", mesh_order=1),
        PolyLine(linestrings=shapely.LineString([(0.0, 1.0), (4.0, 1.0)]), physical_name="jn"),
    ]
    m = generate_mesh(
        entities=entities,
        sweeps=[StructuredSweep(name="j", on="jn", thickness={"left": 0.4, "right": 0.2})],
        dim=2,
        output_mesh=str(tmp_path / "jn.msh"),
        default_characteristic_length=0.5,
        resolution_specs={
            "j": [StructuredSweepResolutionSpec(
                tangential=1.0,
                normal={"left": Graded(h0=0.05, ratio=2.0), "right": 2})],
        },
    )
    ys = np.round(np.unique(m.points[:, 1]), 9)
    # right side (below, travel +x -> right is -y): uniform 2 layers of 0.1
    assert {0.8, 0.9, 1.0} <= set(ys)
    # left side (above): first cell exactly h0
    assert 1.05 in set(ys)


def test_clip_corner_falls_back_to_unstructured(tmp_path):
    """Band would exit its region near x in [3,4] (notched region):
    that shadow has no band; the mesh still generates and is conformal."""
    entities = [
        PolySurface(polygons=shapely.box(0, 0, 4, 1), physical_name="lower", mesh_order=2),
        PolySurface(
            polygons=shapely.box(0, 1, 4, 2).difference(shapely.box(3, 1, 4, 1.2)),
            physical_name="upper", mesh_order=1),
    ]
    m = generate_mesh(
        entities=entities,
        sweeps=[StructuredSweep(name="qw", on="lower___upper", thickness={"upper": 0.4})],
        dim=2,
        output_mesh=str(tmp_path / "clip.msh"),
        default_characteristic_length=0.3,
        resolution_specs={
            "qw": [StructuredSweepResolutionSpec(tangential=0.5, normal={"upper": 2})],
        },
    )
    pts = np.round(m.points[:, :2], 9)
    assert len(np.unique(pts, axis=0)) == len(pts)  # conformal, no dups


def test_separate_steps_equivalent(tmp_path):
    ents = [
        PolySurface(polygons=shapely.box(0, 0, 4, 1), physical_name="lower", mesh_order=2),
        PolySurface(polygons=shapely.box(0, 1, 4, 2), physical_name="upper", mesh_order=1),
    ]
    specs = {"qw": [StructuredSweepResolutionSpec(tangential=1.0, normal={"upper": 2})]}
    sweeps = [StructuredSweep(name="qw", on="lower___upper", thickness={"upper": 0.4})]
    m1 = generate_mesh(
        entities=ents, sweeps=sweeps, dim=2, checkpoint_cad=tmp_path / "m.xao",
        output_mesh=str(tmp_path / "one.msh"),
        default_characteristic_length=0.5, resolution_specs=specs,
    )
    m2 = mesh(
        dim=2, input_file=tmp_path / "m.xao", output_file=str(tmp_path / "two.msh"),
        default_characteristic_length=0.5, resolution_specs=specs,
    )
    b1 = np.unique(np.round(m1.points[(m1.points[:, 1] >= 1 - 1e-9) & (m1.points[:, 1] <= 1.4 + 1e-9), :2], 9), axis=0)
    b2 = np.unique(np.round(m2.points[(m2.points[:, 1] >= 1 - 1e-9) & (m2.points[:, 1] <= 1.4 + 1e-9), :2], 9), axis=0)
    np.testing.assert_array_equal(b1, b2)
