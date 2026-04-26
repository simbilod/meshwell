"""Shared pytest fixtures for meshwell tests.

Centralizes scenes that recur across test files: unit-square polysurfaces,
standard extruded prisms, and tiled-prism arrangements. Tests can request
these by parameter name and avoid re-constructing the same scene inline.
"""
from __future__ import annotations

import pytest
import shapely

from meshwell.polyprism import PolyPrism
from meshwell.polysurface import PolySurface

# --- 2D scenes -----------------------------------------------------------


@pytest.fixture
def unit_square_polysurface() -> PolySurface:
    """Single 1x1 PolySurface at the origin, mesh_order=1, name='A'."""
    return PolySurface(
        polygons=shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        physical_name="A",
        mesh_order=1,
    )


@pytest.fixture
def overlapping_unit_squares() -> list[PolySurface]:
    """Two unit squares overlapping on x in [0.5, 1]: A wins (mesh_order=1)."""
    return [
        PolySurface(
            polygons=shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            physical_name="A",
            mesh_order=1,
        ),
        PolySurface(
            polygons=shapely.Polygon([(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)]),
            physical_name="B",
            mesh_order=2,
        ),
    ]


# --- 3D scenes (prisms) --------------------------------------------------


@pytest.fixture
def standard_prism_buffers() -> dict[float, float]:
    """Vertical extrusion 0->1 with no lateral buffering at either height."""
    return {0.0: 0.0, 1.0: 0.0}


@pytest.fixture
def unit_box_prism(standard_prism_buffers) -> PolyPrism:
    """5x5x1 PolyPrism at the origin, mesh_order=1, name='A'."""
    return PolyPrism(
        polygons=shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),
        buffers=standard_prism_buffers,
        physical_name="A",
        mesh_order=1,
    )


@pytest.fixture
def two_abutting_prisms(standard_prism_buffers) -> list[PolyPrism]:
    """Prisms A and B sharing an interface at x=5; A is winner (mesh_order=1)."""
    return [
        PolyPrism(
            polygons=shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),
            buffers=standard_prism_buffers,
            physical_name="A",
            mesh_order=1,
        ),
        PolyPrism(
            polygons=shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)]),
            buffers=standard_prism_buffers,
            physical_name="B",
            mesh_order=2,
        ),
    ]


@pytest.fixture
def three_abutting_prisms(standard_prism_buffers) -> list[PolyPrism]:
    """Prisms A, B, C in a row sharing interfaces at x=2 and x=5."""
    return [
        PolyPrism(
            polygons=shapely.Polygon([(0, 0), (2, 0), (2, 5), (0, 5)]),
            buffers=standard_prism_buffers,
            physical_name="A",
            mesh_order=1,
        ),
        PolyPrism(
            polygons=shapely.Polygon([(2, 0), (5, 0), (5, 5), (2, 5)]),
            buffers=standard_prism_buffers,
            physical_name="B",
            mesh_order=2,
        ),
        PolyPrism(
            polygons=shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)]),
            buffers=standard_prism_buffers,
            physical_name="C",
            mesh_order=3,
        ),
    ]
