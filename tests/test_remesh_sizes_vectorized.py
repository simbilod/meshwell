"""Vectorized edge/size extraction must match the per-edge reference loop."""

from __future__ import annotations

import numpy as np

from meshwell.remesh import Remesher

EDGE_PAIRS = {
    3: ((0, 1), (1, 2), (2, 0)),
    4: ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
}


def _reference_edges(triangles: np.ndarray) -> set[tuple[int, int]]:
    edges = set()
    for elem in triangles:
        for i, j in EDGE_PAIRS[triangles.shape[1]]:
            edges.add(tuple(sorted((int(elem[i]), int(elem[j])))))
    return edges


def _reference_sizes(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    sums = np.zeros(len(points))
    counts = np.zeros(len(points))
    for n1, n2 in _reference_edges(triangles):
        length = np.linalg.norm(points[n1] - points[n2])
        sums[n1] += length
        sums[n2] += length
        counts[n1] += 1
        counts[n2] += 1

    sizes = np.zeros(len(points))
    connected = counts > 0
    sizes[connected] = sums[connected] / counts[connected]
    return sizes


def _remesher(points: np.ndarray, triangles: np.ndarray | None) -> Remesher:
    # __init__ only builds ModelManager bookkeeping, so no gmsh setup is needed
    # to exercise the size computation.
    remesher = Remesher(n_threads=1, filename="test_remesh_sizes")
    remesher.vxyz = points
    remesher.triangles = triangles
    return remesher


def test_triangle_mesh_sizes_match_reference():
    rng = np.random.default_rng(42)
    # The last two nodes belong to no cell and must come out as 0.0.
    points = rng.random((32, 3))
    triangles = rng.integers(0, 30, size=(60, 3))
    remesher = _remesher(points, triangles)

    assert {tuple(edge) for edge in remesher._extract_edges()} == _reference_edges(
        triangles
    )
    np.testing.assert_allclose(
        remesher.get_current_mesh_sizes(),
        _reference_sizes(points, triangles),
        atol=1e-12,
    )


def test_tet_mesh_sizes_match_reference():
    rng = np.random.default_rng(7)
    points = rng.random((40, 3))
    tets = rng.integers(0, 38, size=(50, 4))
    remesher = _remesher(points, tets)

    assert {tuple(edge) for edge in remesher._extract_edges()} == _reference_edges(tets)
    np.testing.assert_allclose(
        remesher.get_current_mesh_sizes(), _reference_sizes(points, tets), atol=1e-12
    )


def test_no_elements_yields_zero_sizes():
    remesher = _remesher(np.zeros((5, 3)), None)

    assert remesher._extract_edges().shape == (0, 2)
    np.testing.assert_array_equal(remesher.get_current_mesh_sizes(), np.zeros(5))
