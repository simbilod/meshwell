import shapely

from meshwell.orchestrator import generate_mesh
from meshwell.polysurface import PolySurface


def test_unified_pipeline():
    """Smoke-test the unified generate_mesh API on a simple surface."""
    poly = shapely.box(0.0, 0.0, 1.0, 1.0)
    surf = PolySurface(polygons=poly, physical_name="surf")

    m = generate_mesh([surf], dim=2, default_characteristic_length=0.1)

    assert m is not None
    assert len(m.points) > 0
