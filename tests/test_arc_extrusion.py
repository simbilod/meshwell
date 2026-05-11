import gmsh
import numpy as np
from shapely.geometry import Polygon

from meshwell.model import ModelManager
from meshwell.polyprism import PolyPrism


def create_stadium_polygon(radius, length, num_points=50):
    """Creates a polygon with two arcs of the same radius, offset by 'length' and connected by lines."""
    angles_right = np.linspace(-np.pi / 2, np.pi / 2, num_points)
    angles_left = np.linspace(np.pi / 2, 3 * np.pi / 2, num_points)

    # Right arc (center at x=length/2)
    right_arc = [
        (length / 2 + radius * np.cos(a), radius * np.sin(a)) for a in angles_right
    ]

    # Left arc (center at x=-length/2)
    left_arc = [
        (-length / 2 + radius * np.cos(a), radius * np.sin(a)) for a in angles_left
    ]

    # Connect them (the arrays are ordered such that they form a continuous loop)
    coords = right_arc + left_arc
    return Polygon(coords)


def test_arc_extrusion():
    gmsh.initialize()
    try:
        # Create a stadium shape polygon
        poly1 = create_stadium_polygon(radius=10.0, length=20.0, num_points=50)

        # Instantiate PolyPrism with arc identification enabled
        prism1 = PolyPrism(
            polygons=poly1,
            buffers={0.0: 0.0, 5.0: 0.0},  # extrude from z=0 to z=5
            physical_name="stadium1",
            identify_arcs=True,
            arc_tolerance=1e-3,
        )

        manager = ModelManager()
        dimtags1 = prism1.instanciate(manager)

        assert len(dimtags1) > 0
        assert dimtags1[0][0] == 3  # Volume

        manager.sync_model()

        # Test fusing two identical intersecting prisms
        poly2 = create_stadium_polygon(radius=10.0, length=20.0, num_points=50)
        from shapely.affinity import translate

        poly2 = translate(poly2, xoff=5.0, yoff=5.0)

        prism2 = PolyPrism(
            polygons=poly2,
            buffers={0.0: 0.0, 5.0: 0.0},
            physical_name="stadium2",
            identify_arcs=True,
            arc_tolerance=1e-3,
        )
        dimtags2 = prism2.instanciate(manager)
        manager.sync_model()

        # Try fusion
        fuse_result = gmsh.model.occ.fuse(
            dimtags1, dimtags2, removeObject=True, removeTool=True
        )
        manager.sync_model()

        assert fuse_result is not None

    finally:
        gmsh.finalize()


if __name__ == "__main__":
    test_arc_extrusion()
    print("Test passed!")
