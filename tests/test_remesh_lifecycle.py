"""gmsh lifecycle ownership and error propagation in remesh.py / model.py.

Covers:
- corrupt node references in ``_extract_gmsh_mesh_data`` must raise, not
  silently reinterpret the mesh as 2D;
- element type is selected by count (tetra preferred, triangle fallback,
  ``None`` when neither is present);
- ``ModelManager`` warns when displacing another live gmsh session, and
  stays silent for same-manager (re-)initialization;
- ``ModelManager.load_geometry`` loads files through the manager, and
  ``RemeshGMSH.remesh`` routes geometry loading through it;
- ``remesh_mmg`` / ``compute_total_size_map`` finalize the gmsh state
  they created (and only that) even on failure.
"""
import shutil
import subprocess
import warnings
from pathlib import Path

import gmsh
import meshio
import numpy as np
import pytest

from meshwell.model import ModelManager
from meshwell.remesh import (
    Remesher,
    RemeshGMSH,
    RemeshingStrategy,
    RemeshMMG,
    compute_total_size_map,
    remesh_mmg,
)


@pytest.fixture(autouse=True)
def _clean_gmsh():
    """Every test starts and ends without a live gmsh session."""
    if gmsh.isInitialized():
        gmsh.finalize()
    yield
    if gmsh.isInitialized():
        gmsh.finalize()


def _triangle_mesh() -> meshio.Mesh:
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    return meshio.Mesh(points, [("triangle", np.array([[0, 1, 2], [1, 3, 2]]))])


def _tetra_mesh() -> meshio.Mesh:
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    tris = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    return meshio.Mesh(
        points, [("triangle", tris), ("tetra", np.array([[0, 1, 2, 3]]))]
    )


def _write_msh(mesh: meshio.Mesh, path: Path) -> None:
    meshio.write(path, mesh, file_format="gmsh22")


def _identity_strategy(points: np.ndarray) -> RemeshingStrategy:
    """Strategy with (N, 4) refinement data and identity thresholding."""
    data = np.column_stack([points, np.ones(len(points))])
    return RemeshingStrategy(refinement_data=data)


# ---------------------------------------------------------------------------
# _extract_gmsh_mesh_data: count-based selection, KeyError propagation
# ---------------------------------------------------------------------------


def test_corrupt_node_references_raise(tmp_path, monkeypatch):
    """Elements referencing nonexistent node tags must raise, not 'try 2D'."""
    msh = tmp_path / "in.msh"
    _write_msh(_tetra_mesh(), msh)

    def corrupt_get_elements(_element_type, _tag=-1):
        # Non-empty element list whose node tags are absent from getNodes.
        return (
            np.array([1], dtype=np.uint64),
            np.array([999991, 999992, 999993, 999994], dtype=np.uint64),
        )

    monkeypatch.setattr(gmsh.model.mesh, "getElementsByType", corrupt_get_elements)

    remesher = Remesher(n_threads=1)
    with pytest.raises(KeyError):
        remesher._load_mesh_data(msh)


def test_path_input_prefers_tetra(tmp_path):
    msh = tmp_path / "tet.msh"
    _write_msh(_tetra_mesh(), msh)
    remesher = Remesher(n_threads=1)
    remesher._load_mesh_data(msh)
    assert remesher.triangles.shape == (1, 4)


def test_path_input_falls_back_to_triangles(tmp_path):
    msh = tmp_path / "tri.msh"
    _write_msh(_triangle_mesh(), msh)
    remesher = Remesher(n_threads=1)
    remesher._load_mesh_data(msh)
    assert remesher.triangles.shape == (2, 3)


def test_path_input_without_elements_sets_none(tmp_path):
    msh = tmp_path / "pts.msh"
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    _write_msh(meshio.Mesh(points, [("vertex", np.array([[0], [1]]))]), msh)
    remesher = Remesher(n_threads=1)
    remesher._load_mesh_data(msh)
    assert remesher.triangles is None


# ---------------------------------------------------------------------------
# ModelManager: displacement warning + load_geometry
# ---------------------------------------------------------------------------


def test_displacing_another_live_session_warns():
    manager_a = ModelManager(filename="model_a")
    manager_a.ensure_initialized("model_a")

    manager_b = ModelManager(filename="model_b")
    with pytest.warns(UserWarning, match="model_a"):
        manager_b.ensure_initialized("model_b")
    manager_b.finalize()


def test_fresh_initialization_does_not_warn():
    manager = ModelManager(filename="model_fresh")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        manager.ensure_initialized("model_fresh")
    manager.finalize()


def test_same_manager_reinit_does_not_warn():
    manager = ModelManager(filename="model_same")
    manager.ensure_initialized("model_same")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Re-enters _initialize with a live session owned by this manager.
        manager.clear_and_reinitialize("model_same")
        manager.ensure_initialized("model_same")
    manager.finalize()


def test_load_geometry_loads_into_managed_model(tmp_path):
    msh = tmp_path / "tri.msh"
    _write_msh(_triangle_mesh(), msh)

    manager = ModelManager(filename="loader")
    manager.load_geometry(msh)
    node_tags, _, _ = gmsh.model.mesh.getNodes()
    assert len(node_tags) == 4
    manager.finalize()


def test_remesh_gmsh_routes_geometry_through_manager(tmp_path, monkeypatch):
    """RemeshGMSH.remesh must load the geometry via ModelManager, not gmsh.open."""

    class _StopFlow(Exception):
        pass

    loaded = []

    def spy_load_geometry(_self, path):
        loaded.append(Path(path))
        raise _StopFlow

    monkeypatch.setattr(ModelManager, "load_geometry", spy_load_geometry)

    input_mesh = _triangle_mesh()
    remesher = RemeshGMSH(n_threads=1)
    with pytest.raises(_StopFlow):
        remesher.remesh(
            input_mesh=input_mesh,
            geometry_file=tmp_path / "geo.xao",
            strategies=[_identity_strategy(input_mesh.points)],
            dim=2,
        )
    assert loaded == [tmp_path / "geo.xao"]


# ---------------------------------------------------------------------------
# remesh_mmg / compute_total_size_map: finalize what they create
# ---------------------------------------------------------------------------


def _fake_mmg_run(cmd, **_kwargs):
    """Stand-in for the MMG binary: identity remesh (copy input to output)."""
    src = Path(cmd[cmd.index("-in") + 1])
    dst = Path(cmd[cmd.index("-out") + 1])
    shutil.copy(src, dst)


def test_remesh_mmg_finalizes_gmsh(tmp_path, monkeypatch):
    msh = tmp_path / "in.msh"
    mesh = _triangle_mesh()
    _write_msh(mesh, msh)

    monkeypatch.setattr(subprocess, "run", _fake_mmg_run)
    monkeypatch.setattr(RemeshMMG, "_find_executable", lambda _self: "mmg2d_O3")

    size_map = remesh_mmg(
        input_mesh=msh,
        output_mesh=tmp_path / "out.msh",
        strategies=[_identity_strategy(mesh.points)],
        dim=2,
    )
    assert size_map.shape[1] == 4
    assert not gmsh.isInitialized(), "remesh_mmg must finalize the gmsh it created"


def test_remesh_mmg_finalizes_gmsh_on_failure(tmp_path, monkeypatch):
    msh = tmp_path / "in.msh"
    mesh = _triangle_mesh()
    _write_msh(mesh, msh)

    def failing_run(cmd, **_kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd)

    monkeypatch.setattr(subprocess, "run", failing_run)
    monkeypatch.setattr(RemeshMMG, "_find_executable", lambda _self: "mmg2d_O3")

    with pytest.raises(RuntimeError, match="MMG failed"):
        remesh_mmg(
            input_mesh=msh,
            output_mesh=tmp_path / "out.msh",
            strategies=[_identity_strategy(mesh.points)],
            dim=2,
        )
    assert not gmsh.isInitialized(), "remesh_mmg must finalize even on failure"


def test_compute_total_size_map_finalizes_gmsh(tmp_path):
    msh = tmp_path / "in.msh"
    mesh = _triangle_mesh()
    _write_msh(mesh, msh)

    size_map = compute_total_size_map(msh, [_identity_strategy(mesh.points)])
    assert size_map.shape[1] == 4
    assert not gmsh.isInitialized()


def test_compute_total_size_map_preserves_caller_session():
    """A caller-owned gmsh session must survive compute_total_size_map."""
    caller = ModelManager(filename="caller_session")
    caller.ensure_initialized("caller_session")
    gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)

    compute_total_size_map(caller, [_identity_strategy(np.zeros((1, 3)))])

    assert gmsh.isInitialized(), "must not finalize a session it did not create"
    caller.finalize()
