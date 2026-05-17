"""Unit tests for check 5: near-duplicate nodes."""
import gmsh
import pytest

from meshwell.structured.spec import (
    PhantomMap,
    StructuredMeshPlan,
    StructuredPlan,
)
from meshwell.structured.validator import validate_structured_mesh


@pytest.fixture
def empty_inputs():
    """Empty plan / mesh_plan / phantom_map — exercises only mesh-level checks."""
    plan = StructuredPlan(slabs=(), z_planes=(), overlaps=())
    mesh_plan = StructuredMeshPlan(slabs=(), n_layers=(), recombine=())
    phantom_map = PhantomMap()
    return plan, mesh_plan, phantom_map


def _add_lone_node(x: float, y: float, z: float) -> int:
    """Helper: add a node to a fresh discrete 3D entity, return its tag."""
    ent = gmsh.model.addDiscreteEntity(3, -1, [])
    max_tag = gmsh.model.mesh.getMaxNodeTag()
    new_tag = int(max_tag) + 1
    gmsh.model.mesh.addNodes(3, ent, [new_tag], [x, y, z])
    return new_tag


def test_unique_nodes_no_issue(empty_inputs):
    plan, mesh_plan, phantom_map = empty_inputs
    gmsh.initialize()
    gmsh.model.add("unique")
    _add_lone_node(0.0, 0.0, 0.0)
    _add_lone_node(1.0, 0.0, 0.0)
    _add_lone_node(0.0, 1.0, 0.0)

    result = validate_structured_mesh(
        plan, mesh_plan, phantom_map, occ_entities=[], vol_tags=[], tol=1e-6
    )
    duplicate_issues = [
        i for i in result.errors + result.warnings if i.check == "near_duplicate_nodes"
    ]
    assert duplicate_issues == []
    gmsh.finalize()


def test_exact_duplicate_nodes_reported_as_error(empty_inputs):
    plan, mesh_plan, phantom_map = empty_inputs
    gmsh.initialize()
    gmsh.model.add("exact_dup")
    _add_lone_node(0.5, 0.5, 0.5)
    _add_lone_node(0.5, 0.5, 0.5)  # Exact duplicate.

    result = validate_structured_mesh(
        plan, mesh_plan, phantom_map, occ_entities=[], vol_tags=[], tol=1e-6
    )
    exact_errors = [i for i in result.errors if i.check == "near_duplicate_nodes"]
    assert len(exact_errors) >= 1
    gmsh.finalize()


def test_near_duplicate_nodes_reported_as_warning(empty_inputs):
    plan, mesh_plan, phantom_map = empty_inputs
    gmsh.initialize()
    gmsh.model.add("near_dup")
    _add_lone_node(0.5, 0.5, 0.5)
    _add_lone_node(0.5 + 1e-9, 0.5, 0.5)  # 1 nm offset.

    result = validate_structured_mesh(
        plan, mesh_plan, phantom_map, occ_entities=[], vol_tags=[], tol=1e-7
    )
    near_warnings = [i for i in result.warnings if i.check == "near_duplicate_nodes"]
    assert len(near_warnings) >= 1
    gmsh.finalize()
