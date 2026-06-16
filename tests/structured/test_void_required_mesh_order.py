"""Verify a void without mesh_order raises StructuredVoidMeshOrderRequiredError."""
import pytest
from shapely.geometry import Polygon

from meshwell.polyprism import PolyPrism
from meshwell.structured.collect import collect_structured_slabs
from meshwell.structured.exceptions import StructuredVoidMeshOrderRequiredError

SQ_BIG = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
SQ_SMALL = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])


def test_void_without_mesh_order_raises():
    bg = PolyPrism(
        SQ_BIG,
        {0.0: 0.0, 1.0: 0.0},
        physical_name="bg",
        structured=True,
        mesh_order=1.0,
    )
    hole = PolyPrism(
        SQ_SMALL,
        {0.0: 0.0, 1.0: 0.0},
        physical_name="hole",
        structured=True,
        mesh_bool=False,
    )
    with pytest.raises(StructuredVoidMeshOrderRequiredError):
        collect_structured_slabs([bg, hole])


def test_void_with_mesh_order_does_not_raise():
    bg = PolyPrism(
        SQ_BIG,
        {0.0: 0.0, 1.0: 0.0},
        physical_name="bg",
        structured=True,
        mesh_order=2.0,
    )
    hole = PolyPrism(
        SQ_SMALL,
        {0.0: 0.0, 1.0: 0.0},
        physical_name="hole",
        structured=True,
        mesh_bool=False,
        mesh_order=1.0,
    )
    collect_structured_slabs([bg, hole])  # no raise


def test_solid_without_mesh_order_does_not_raise():
    """Only voids require mesh_order; solids without it default to inf.

    That's fine — voids are the only ones that need explicit ordering.
    """
    bg = PolyPrism(SQ_BIG, {0.0: 0.0, 1.0: 0.0}, physical_name="bg", structured=True)
    collect_structured_slabs([bg])  # no raise
