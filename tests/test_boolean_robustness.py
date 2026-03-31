import shapely

from meshwell.orchestrator import generate_mesh
from meshwell.polysurface import PolySurface


def test_overlapping_boxes_robustness():
    """Verify that overlapping boxes with specific mesh_order resolve correctly.

    This scenario replicates the conditions where coordinate snapping previously
    caused incorrect vertex connections.
    """
    width = 1
    height = 1

    # Core: priority 0
    core = shapely.geometry.box(-width / 2, -0.2, +width / 2, height)
    # Cladding: priority 1
    cladding = shapely.geometry.box(-width * 2, 0, width * 2, height * 3)
    # Buried Oxide: priority 2
    buried_oxide = shapely.geometry.box(-width * 2, -height * 2, width * 2, 0)

    core_surface = PolySurface(polygons=core, physical_name="core", mesh_order=0)
    cladding_surface = PolySurface(
        polygons=cladding, physical_name="cladding", mesh_order=1
    )
    buried_oxide_surface = PolySurface(
        polygons=buried_oxide, physical_name="buried_oxide", mesh_order=2
    )

    entities = [core_surface, cladding_surface, buried_oxide_surface]

    # Generate mesh using unified gmsh backend
    m = generate_mesh(
        entities=entities,
        dim=2,
        backend="gmsh",
        default_characteristic_length=0.5,
        n_threads=1,
    )

    # Check that we have the expected physical groups
    # 'core___cladding', 'core___buried_oxide', 'cladding___buried_oxide' should exist
    physical_names = list(m.field_data.keys())
    assert "core" in physical_names
    assert "cladding" in physical_names
    assert "buried_oxide" in physical_names
    assert "core___cladding" in physical_names
    assert "core___buried_oxide" in physical_names
    assert "cladding___buried_oxide" in physical_names

    # Ensure no node connection artifacts
    # The new robust logic produces 131 nodes for this geometry.
    assert len(m.points) == 130


if __name__ == "__main__":
    test_overlapping_boxes_robustness()
