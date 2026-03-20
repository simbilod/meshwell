import shapely

from meshwell.orchestrator import generate_mesh
from meshwell.polysurface import PolySurface


def test_unified_pipeline():
    """Test the unified generate_mesh API with both backends."""
    poly = shapely.box(0.0, 0.0, 1.0, 1.0)
    surf = PolySurface(polygons=poly, physical_name="surf")

    # Test OCC backend
    m_occ = generate_mesh(
        [surf], dim=2, backend="occ", default_characteristic_length=0.1
    )
    assert m_occ is not None
    assert len(m_occ.points) > 0

    # Test GMSH backend
    m_gmsh = generate_mesh(
        [surf], dim=2, backend="gmsh", default_characteristic_length=0.1
    )
    assert m_gmsh is not None
    assert len(m_gmsh.points) > 0
