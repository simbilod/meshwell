"""_load_mesh_data must prefer volume cells for 3D meshio inputs."""
import meshio
import numpy as np

from meshwell.remesh import Remesher


def _mesh_3d():
    # one tetra with its four boundary triangles — mimics any real 3D mesh
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    tetra = np.array([[0, 1, 2, 3]])
    tris = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    return meshio.Mesh(points, [("triangle", tris), ("tetra", tetra)])


def test_meshio_input_prefers_tetra():
    r = Remesher()
    r._load_mesh_data(_mesh_3d())
    assert r.triangles.shape == (1, 4), "must load tetra, not boundary triangles"


def test_meshio_input_falls_back_to_triangles():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    m = meshio.Mesh(points, [("triangle", np.array([[0, 1, 2]]))])
    r = Remesher()
    r._load_mesh_data(m)
    assert r.triangles.shape == (1, 3)
