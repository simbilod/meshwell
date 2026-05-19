"""Unit tests for the overlap_metadata() API used by cad_occ's fast-path.

Spec: docs/superpowers/specs/2026-05-19-cad-occ-polyprism-overlap-fastpath-design.md
"""
from __future__ import annotations

import shapely
from shapely.geometry import Polygon

from meshwell.polyprism import PolyPrism
from meshwell.polysurface import PolySurface


def _square(x, y, w, h) -> Polygon:
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def test_extrude_polyprism_metadata_is_exact():
    """Axis-aligned extrusion: footprint == input polygon, zrange exact, is_exact=True."""
    foot = _square(0, 0, 10, 5)
    ent = PolyPrism(polygons=foot, buffers={0.0: 0.0, 2.5: 0.0}, physical_name="p")
    md = ent.overlap_metadata()
    assert md is not None
    footprint, (zmin, zmax), is_exact = md
    assert is_exact is True
    assert zmin == 0.0
    assert zmax == 2.5
    # Footprint matches the input polygon (modulo set_precision snap at default
    # point_tolerance=1e-3, which won't move integer coords).
    assert shapely.equals(footprint, foot)


def test_extrude_polyprism_with_list_of_polygons_unions_them():
    """List of polygons => union as the footprint, still exact."""
    p1 = _square(0, 0, 1, 1)
    p2 = _square(2, 0, 1, 1)
    ent = PolyPrism(polygons=[p1, p2], buffers={0.0: 0.0, 1.0: 0.0}, physical_name="p")
    md = ent.overlap_metadata()
    assert md is not None
    footprint, _, is_exact = md
    assert is_exact is True
    expected = shapely.unary_union([p1, p2])
    assert shapely.equals(footprint, expected)


def test_tapered_polyprism_metadata_is_conservative():
    """Non-zero buffer => tapered.

    Footprint = union of all buffered cross-sections, zrange = full extent,
    is_exact=False (conservative envelope).
    """
    base = _square(0, 0, 4, 4)
    ent = PolyPrism(
        polygons=base,
        buffers={0.0: 0.0, 1.0: 1.0},  # outward buffer at z=1 => tapered
        physical_name="p",
    )
    md = ent.overlap_metadata()
    assert md is not None
    footprint, (zmin, zmax), is_exact = md
    assert is_exact is False
    assert zmin == 0.0
    assert zmax == 1.0
    # Conservative envelope must contain the BASE polygon (z=0 buffered by 0)
    # and the LARGER polygon (z=1 buffered outward by 1.0).
    assert footprint.contains(base.buffer(-1e-6))
    larger = base.buffer(1.0)
    assert footprint.contains(larger.buffer(-1e-6))


def test_polysurface_metadata_defaults_to_none():
    """Non-polyprism entities opt out of the fast-path by returning None."""
    ps = PolySurface(polygons=_square(0, 0, 1, 1), physical_name="s")
    assert ps.overlap_metadata() is None
