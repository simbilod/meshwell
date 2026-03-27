from __future__ import annotations

import gmsh
import numpy as np
import shapely

from meshwell.cad_occ import cad_occ
from meshwell.polyprism import PolyPrism
from meshwell.polysurface import PolySurface


def test_polysurface_transformation_occ():
    # Square 2x2 centered at origin (0,0)
    polygon = shapely.Polygon([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])

    # Rotate 90 degrees around X-axis, translate by (0, 0, 10)
    # Pivot should default to (0,0,0) as it's the centroid
    surface = PolySurface(
        polygons=polygon,
        rotation_axis=(1, 0, 0),
        rotation_angle=90,
        translation=(0, 0, 10),
    )

    labeled_entities = cad_occ(entities_list=[surface])
    shape = labeled_entities[0].shape

    # Get bounding box
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    bbox = Bnd_Box()
    BRepBndLib.Add_s(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

    # After 90 deg rotation around X, the square [-1,1] in Y is now in Z.
    # Original Y range [-1, 1] becomes Z range [-1, 1] (relative to pivot).
    # Then translate by (0, 0, 10) in Z.
    # Final Z range should be [9, 11].
    # Final Y range should be 0 (it's flat in YZ plane, so Y should be near 0).
    # Final X range should be [-1, 1].

    assert np.isclose(xmin, -1)
    assert np.isclose(xmax, 1)
    assert np.isclose(ymin, 0, atol=1e-6)
    assert np.isclose(ymax, 0, atol=1e-6)
    assert np.isclose(zmin, 9)
    assert np.isclose(zmax, 11)


def test_polysurface_transformation_gmsh():
    gmsh.initialize()
    try:
        polygon = shapely.Polygon([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
        surface = PolySurface(
            polygons=polygon,
            rotation_axis=(1, 0, 0),
            rotation_angle=90,
            translation=(0, 0, 10),
        )
        dimtags = surface.instanciate()

        # Check bounding box in GMSH
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(*dimtags[0])

        assert np.isclose(xmin, -1)
        assert np.isclose(xmax, 1)
        assert np.isclose(ymin, 0, atol=1e-6)
        assert np.isclose(ymax, 0, atol=1e-6)
        assert np.isclose(zmin, 9)
        assert np.isclose(zmax, 11)
    finally:
        gmsh.finalize()


def test_polyprism_transformation_occ():
    # Square 2x2 centered at origin (0,0)
    polygon = shapely.Polygon([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])

    # Prism from z=0 to z=1
    prism = PolyPrism(
        polygons=polygon,
        buffers={0.0: 0.0, 1.0: 0.0},
        rotation_axis=(0, 1, 0),  # Around Y
        rotation_angle=90,
        translation=(10, 0, 0),
    )

    labeled_entities = cad_occ(entities_list=[prism])
    shape = labeled_entities[0].shape

    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    bbox = Bnd_Box()
    BRepBndLib.Add_s(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

    # Original prism: X [-1,1], Y [-1,1], Z [0,1]
    # Centroid: (0,0,0.5) - Wait, rotation_point defaults to centroid of *polygons* (2D)
    # polygons centroid is (0,0).
    # Rotation 90 deg around Y: X becomes -Z, Z becomes X.
    # X [-1,1] -> Z [1, -1] (Wait, X positive becomes Z negative?)
    # gp_Trsf rotation: positive angle is counter-clockwise.
    # X -> Z, Z -> -X?
    # Let's say: (1,0,0) rotated 90 around (0,1,0) -> (0,0,-1)
    # (0,0,1) rotated 90 around (0,1,0) -> (1,0,0)
    # Original X [-1,1] -> Z [1, -1]
    # Original Z [0,1] -> X [0,1]
    # Then translate by (10, 0, 0)
    # X becomes [10, 11]
    # Y stays [-1, 1]
    # Z becomes [-1, 1]

    assert np.isclose(xmin, 10)
    assert np.isclose(xmax, 11)
    assert np.isclose(ymin, -1)
    assert np.isclose(ymax, 1)
    assert np.isclose(zmin, -1)
    assert np.isclose(zmax, 1)


def test_serialization():
    polygon = shapely.Polygon([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
    surface = PolySurface(
        polygons=polygon,
        rotation_axis=(1, 2, 3),
        rotation_angle=45,
        translation=(5, 6, 7),
        rotation_point=(1, 1, 1),
    )

    d = surface.to_dict()
    surface2 = PolySurface.from_dict(d)

    assert surface2.rotation_axis == (1, 2, 3)
    assert surface2.rotation_angle == 45
    assert surface2.translation == (5, 6, 7)
    assert surface2.rotation_point == (1, 1, 1)

    prism = PolyPrism(
        polygons=polygon, buffers={0.0: 0.0, 1.0: 0.0}, translation=(1, 2, 3)
    )
    d = prism.to_dict()
    prism2 = PolyPrism.from_dict(d)
    assert prism2.translation == (1, 2, 3)


if __name__ == "__main__":
    test_polysurface_transformation_occ()
    test_polysurface_transformation_gmsh()
    test_polyprism_transformation_occ()
    test_serialization()
