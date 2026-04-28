"""Tests for the distributed-meshing pipeline."""
from __future__ import annotations

import shapely

from meshwell.cad_common import prepare_entities
from meshwell.polysurface import PolySurface


def test_prepare_entities_skip_buffer_preserves_polygons():
    p = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    ent = PolySurface(polygons=p, physical_name="A", mesh_order=1)
    original = list(ent.polygons)

    prepare_entities([ent], perturbation=1e-3, skip_buffer=True)

    # polygons must be unchanged (no outward buffer applied).
    assert len(ent.polygons) == len(original)
    assert all(a.equals(b) for a, b in zip(ent.polygons, original))
