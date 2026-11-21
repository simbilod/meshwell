from __future__ import annotations

from functools import partial

import gmsh
import pytest
import shapely

from meshwell.cad_settings import DEFAULT_POINT_TOLERANCE
from meshwell.polyprism import PolyPrism
from meshwell.polysurface import PolySurface


def _multihole_polygon(n_holes: int) -> shapely.Polygon:
    outer = shapely.box(0.0, 0.0, 10.0, 10.0)
    holes = [
        shapely.box(1.0 + 1.5 * i, 1.0, 1.4 + 1.5 * i, 1.4) for i in range(n_holes)
    ]
    return shapely.Polygon(outer.exterior.coords, [h.exterior.coords for h in holes])


@pytest.fixture
def model():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        yield gmsh.model
    finally:
        gmsh.finalize()


def _count_cuts(monkeypatch):
    calls = []
    original = gmsh.model.occ.cut

    def spy(objects, tools, *args, **kwargs):
        calls.append((len(objects), len(tools)))
        return original(objects, tools, *args, **kwargs)

    monkeypatch.setattr(gmsh.model.occ, "cut", spy)
    return calls


@pytest.mark.usefixtures("model")
def test_surface_holes_cut_in_one_boolean(monkeypatch):
    surface = PolySurface(
        polygons=_multihole_polygon(5), physical_name="A", mesh_order=1
    )
    calls = _count_cuts(monkeypatch)
    dimtags = surface.instanciate()

    assert calls == [(1, 5)]
    (dim, tag), *_ = dimtags
    assert dim == 2
    # 10x10 outer minus five 0.4x0.4 holes.
    assert gmsh.model.occ.getMass(dim, tag) == pytest.approx(100.0 - 5 * 0.16, rel=1e-9)


@pytest.mark.usefixtures("model")
def test_prism_holes_cut_in_one_boolean(monkeypatch):
    prism = PolyPrism(
        polygons=_multihole_polygon(5),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="A",
        mesh_order=1,
    )
    calls = _count_cuts(monkeypatch)
    dimtags = prism.instanciate(None)

    assert calls == [(1, 5)]
    (dim, tag), *_ = dimtags
    assert dim == 3
    # Same footprint, extruded 1.0.
    assert gmsh.model.occ.getMass(dim, tag) == pytest.approx(100.0 - 5 * 0.16, rel=1e-9)


@pytest.mark.usefixtures("model")
@pytest.mark.parametrize(
    "holes",
    [
        # Disjoint, strictly interior.
        [(1.0, 1.0, 2.0, 2.0), (4.0, 1.0, 5.0, 2.0), (7.0, 1.0, 8.0, 2.0)],
        # Two holes abutting along x=3, so their union is one connected region.
        [(1.0, 1.0, 3.0, 3.0), (3.0, 1.0, 5.0, 3.0)],
        # Five boxes forming a plus: the union is connected and non-convex.
        [
            (2.0, 4.0, 4.0, 6.0),
            (4.0, 2.0, 6.0, 4.0),
            (4.0, 4.0, 6.0, 6.0),
            (4.0, 6.0, 6.0, 8.0),
            (6.0, 4.0, 8.0, 6.0),
        ],
    ],
)
def test_hole_cut_area_matches_the_shapely_reference(holes):
    outer = shapely.box(0.0, 0.0, 10.0, 10.0)
    hole_polys = [shapely.box(*h) for h in holes]
    polygon = shapely.Polygon(
        outer.exterior.coords, [h.exterior.coords for h in hole_polys]
    )

    # PolySurface snaps its input to the default tolerance grid, so the
    # reference is snapped the same way rather than decoupled from the
    # geometry the code actually builds.
    snap = partial(
        shapely.set_precision, grid_size=DEFAULT_POINT_TOLERANCE, mode="pointwise"
    )
    expected = (
        snap(outer).difference(shapely.unary_union([snap(h) for h in hole_polys])).area
    )

    surface = PolySurface(polygons=polygon, physical_name="A", mesh_order=1)
    (dim, tag), *_ = surface.instanciate()
    assert gmsh.model.occ.getMass(dim, tag) == pytest.approx(expected, rel=1e-9)


@pytest.mark.usefixtures("model")
def test_no_holes_no_boolean(monkeypatch):
    surface = PolySurface(
        polygons=shapely.box(0.0, 0.0, 10.0, 10.0), physical_name="A", mesh_order=1
    )
    calls = _count_cuts(monkeypatch)
    surface.instanciate()
    assert calls == []
