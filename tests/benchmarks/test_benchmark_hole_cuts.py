import time

import gmsh
import pytest
import shapely

from meshwell.polysurface import PolySurface


def create_polysurface_with_holes(n_holes: int) -> PolySurface:
    outer = shapely.box(0.0, 0.0, 10.0, 10.0)
    # Grid-packed inside the exterior so every tool really intersects it.
    holes = [
        shapely.box(
            (x := 0.3 + 0.6 * (i % 15)),
            (y := 0.3 + 0.6 * (i // 15)),
            x + 0.4,
            y + 0.4,
        )
        for i in range(n_holes)
    ]
    polygon = shapely.Polygon(outer.exterior.coords, [h.exterior.coords for h in holes])
    assert polygon.is_valid
    return PolySurface(polygons=polygon, physical_name="holed", mesh_order=1)


def test_benchmark_hole_cuts():
    """Time to instantiate one face with many interior holes."""
    n_holes = 200
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        surface = create_polysurface_with_holes(n_holes)
        expected_area = sum(p.area for p in surface.polygons)
        start = time.perf_counter()
        dimtags = surface.instanciate()
        elapsed = time.perf_counter() - start

        assert len(dimtags) == 1
        (dim, tag), *_ = dimtags
        assert gmsh.model.occ.getMass(dim, tag) == pytest.approx(
            expected_area, rel=1e-9
        )
        print(f"\n{n_holes} holes: {elapsed:.4f}s")
    finally:
        gmsh.finalize()
