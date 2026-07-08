"""Shared CAD helpers: backend-agnostic behavior of cad_common.prepare_entities."""
import pytest
from shapely.geometry import Polygon

from meshwell.cad_common import prepare_entities
from meshwell.polysurface import PolySurface


def test_prepare_entities_rejects_double_call():
    """Reject a second call on the same entities.

    prepare_entities is not idempotent (the buffer would compound);
    a second call on the same entities must fail loudly, not distort
    geometry by another perturbation.
    """
    ps = PolySurface(
        polygons=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        physical_name="a",
    )
    prepare_entities([ps], perturbation=1e-5)
    with pytest.raises(RuntimeError, match="prepare_entities"):
        prepare_entities([ps], perturbation=1e-5)
