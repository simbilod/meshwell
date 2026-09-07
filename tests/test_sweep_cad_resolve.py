import numpy as np
import pytest
import shapely

from meshwell.polyline import PolyLine
from meshwell.polysurface import PolySurface
from meshwell.structured.exceptions import (
    SweepAttachmentNotFoundError,
    SweepCurvedSourceError,
)
from meshwell.structured.sweep import StructuredSweep
from meshwell.structured.sweep_cad import (
    final_region_polygons,
    resolve_attachment,
    side_normal,
)


def _stack():
    lower = PolySurface(
        polygons=shapely.box(0, 0, 4, 1), physical_name="lower", mesh_order=2
    )
    upper = PolySurface(
        polygons=shapely.box(0, 1, 4, 2), physical_name="upper", mesh_order=1
    )
    return [lower, upper]


def test_final_region_polygons_mesh_order_precedence():
    # overlapping boxes: mesh_order 1 wins the overlap strip
    a = PolySurface(polygons=shapely.box(0, 0, 2, 2), physical_name="a", mesh_order=2)
    b = PolySurface(polygons=shapely.box(1, 0, 3, 2), physical_name="b", mesh_order=1)
    regions = final_region_polygons([a, b])
    assert regions["b"].area == pytest.approx(4.0)
    assert regions["a"].area == pytest.approx(2.0)  # lost the overlap


def test_resolve_interface_attachment():
    entities = _stack()
    sweep = StructuredSweep(name="s", on="lower___upper", thickness={"upper": 0.5})
    regions = final_region_polygons(entities)
    p0, p1 = resolve_attachment(sweep, entities, regions, point_tolerance=1e-6)
    assert p0[1] == pytest.approx(1.0)
    assert p1[1] == pytest.approx(1.0)
    assert {min(p0[0], p1[0]), max(p0[0], p1[0])} == {0.0, 4.0}


def test_resolve_boundary_attachment():
    entities = _stack()
    sweep = StructuredSweep(name="s", on="lower___None", thickness={"lower": 0.5})
    regions = final_region_polygons(entities)
    # lower's hull boundary is 3 sides (bottom + two laterals) -> not a single
    # straight segment -> curved-source error is the phase-1 contract
    with pytest.raises(SweepCurvedSourceError):
        resolve_attachment(sweep, entities, regions, point_tolerance=1e-6)


def test_resolve_boundary_attachment_straight():
    # single region: its bottom edge selected via a PolyLine-free trick is not
    # possible; use a wide flat region whose hull IS a rectangle -> still 4 sides.
    # Straight boundary attachment therefore uses an embedded PolyLine in
    # practice; this test pins the error message mentions PolyLine.
    entities = _stack()
    sweep = StructuredSweep(name="s", on="lower___None", thickness={"lower": 0.5})
    regions = final_region_polygons(entities)
    with pytest.raises(SweepCurvedSourceError, match="PolyLine"):
        resolve_attachment(sweep, entities, regions, point_tolerance=1e-6)


def test_resolve_polyline_attachment_preserves_orientation():
    entities = _stack()
    pl = PolyLine(
        linestrings=shapely.LineString([(3.0, 0.5), (1.0, 0.5)]),
        physical_name="jline",
    )
    entities.append(pl)
    sweep = StructuredSweep(name="s", on="jline", thickness={"left": 0.2})
    regions = final_region_polygons(entities)
    p0, p1 = resolve_attachment(sweep, entities, regions, point_tolerance=1e-6)
    np.testing.assert_allclose(p0, [3.0, 0.5])
    np.testing.assert_allclose(p1, [1.0, 0.5])
    # travel direction is -x, so "left" is -y
    n = side_normal(p0, p1, "left", sweep, regions, point_tolerance=1e-6)
    np.testing.assert_allclose(n, [0.0, -1.0], atol=1e-12)


def test_missing_attachment_raises():
    entities = _stack()
    sweep = StructuredSweep(name="s", on="lower___nosuch", thickness={"lower": 0.5})
    regions = final_region_polygons(entities)
    with pytest.raises(SweepAttachmentNotFoundError):
        resolve_attachment(sweep, entities, regions, point_tolerance=1e-6)


def test_interface_side_normals_point_into_regions():
    entities = _stack()
    sweep = StructuredSweep(
        name="s", on="lower___upper", thickness={"upper": 0.5, "lower": 0.3}
    )
    regions = final_region_polygons(entities)
    p0, p1 = resolve_attachment(sweep, entities, regions, point_tolerance=1e-6)
    n_up = side_normal(p0, p1, "upper", sweep, regions, point_tolerance=1e-6)
    n_lo = side_normal(p0, p1, "lower", sweep, regions, point_tolerance=1e-6)
    assert n_up[1] == pytest.approx(1.0)
    assert n_lo[1] == pytest.approx(-1.0)


def test_resolve_interface_attachment_rejects_stray_point_contact():
    # lower/upper each have a second lobe that only touches at one far
    # corner. boundary.intersection then returns a GeometryCollection
    # mixing the shared straight edge (LineString) with the stray corner
    # touch (Point) -> must raise SweepCurvedSourceError, not silently
    # merge to the edge and drop the extra contact.
    lower = PolySurface(
        polygons=[shapely.box(0, 0, 4, 1), shapely.box(6, 1, 7, 2)],
        physical_name="lower",
        mesh_order=2,
    )
    upper = PolySurface(
        polygons=[shapely.box(0, 1, 4, 2), shapely.box(7, 2, 8, 3)],
        physical_name="upper",
        mesh_order=1,
    )
    entities = [lower, upper]
    sweep = StructuredSweep(name="s", on="lower___upper", thickness={"upper": 0.5})
    regions = final_region_polygons(entities)
    shared = regions["lower"].boundary.intersection(regions["upper"].boundary)
    assert shared.geom_type == "GeometryCollection"  # sanity: mixed intersection
    with pytest.raises(SweepCurvedSourceError):
        resolve_attachment(sweep, entities, regions, point_tolerance=1e-6)


def test_curved_polyline_rejected():
    entities = _stack()
    pl = PolyLine(
        linestrings=shapely.LineString([(1, 0.5), (2, 0.7), (3, 0.5)]),
        physical_name="curvy",
    )
    entities.append(pl)
    sweep = StructuredSweep(name="s", on="curvy", thickness={"left": 0.1})
    regions = final_region_polygons(entities)
    with pytest.raises(SweepCurvedSourceError):
        resolve_attachment(sweep, entities, regions, point_tolerance=1e-6)
