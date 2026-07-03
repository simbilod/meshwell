"""APIs removed in the 2026-07-03 code-improvement effort stay removed."""
import inspect

from meshwell.mesh import Mesh, mesh


def test_periodic_entities_removed_from_mesh_wrapper():
    assert "periodic_entities" not in inspect.signature(mesh).parameters


def test_periodic_entities_removed_from_process_geometry():
    assert (
        "periodic_entities" not in inspect.signature(Mesh.process_geometry).parameters
    )


def test_periodic_helper_methods_deleted():
    assert not hasattr(Mesh, "_apply_periodic_boundaries")
    assert not hasattr(Mesh, "_set_periodic_pair")
