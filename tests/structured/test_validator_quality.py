"""Unit tests for check 7: element quality (opt-in)."""
import gmsh
import pytest

from meshwell.structured.spec import (
    PhantomMap,
    StructuredMeshPlan,
    StructuredPlan,
)
from meshwell.structured.validator import validate_structured_mesh


@pytest.fixture
def empty_plan_inputs():
    plan = StructuredPlan(slabs=(), z_planes=(), overlaps=())
    mesh_plan = StructuredMeshPlan(slabs=(), n_layers=(), recombine=())
    phantom_map = PhantomMap()
    return plan, mesh_plan, phantom_map


@pytest.fixture
def gmsh_session():
    gmsh.initialize()
    gmsh.model.add("test")
    try:
        yield
    finally:
        gmsh.finalize()


@pytest.fixture
def meshed_cube(gmsh_session):  # noqa: ARG001
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.4)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.4)
    gmsh.model.mesh.generate(3)


def test_quality_check_off_by_default(meshed_cube, empty_plan_inputs):  # noqa: ARG001
    plan, mesh_plan, phantom_map = empty_plan_inputs
    result = validate_structured_mesh(
        plan,
        mesh_plan,
        phantom_map,
        occ_entities=[],
        vol_tags=[],
        tol=1e-6,
    )
    quality_issues = [
        i for i in result.errors + result.warnings if i.check == "element_quality"
    ]
    assert quality_issues == []


def test_good_quality_cube_passes(meshed_cube, empty_plan_inputs):  # noqa: ARG001
    plan, mesh_plan, phantom_map = empty_plan_inputs
    result = validate_structured_mesh(
        plan,
        mesh_plan,
        phantom_map,
        occ_entities=[],
        vol_tags=[],
        tol=1e-6,
        include_quality=True,
    )
    # A cube meshed at uniform CL=0.4 should have no quality errors.
    quality_errors = [i for i in result.errors if i.check == "element_quality"]
    assert quality_errors == []


def test_negative_jacobian_reported_as_error(
    gmsh_session, empty_plan_inputs  # noqa: ARG001
):
    """A wedge with reversed top-bottom orientation has negative minSICN."""
    plan, mesh_plan, phantom_map = empty_plan_inputs
    ent = gmsh.model.addDiscreteEntity(3, -1, [])
    # Wedge node order [bot0, bot1, bot2, top0, top1, top2] with top
    # winding reversed → negative Jacobian.
    coords = [
        0.0,
        0.0,
        0.0,  # 1: bot0
        1.0,
        0.0,
        0.0,  # 2: bot1
        0.0,
        1.0,
        0.0,  # 3: bot2
        0.0,
        1.0,
        1.0,  # 4: top0  (reversed)
        1.0,
        0.0,
        1.0,  # 5: top1
        0.0,
        0.0,
        1.0,  # 6: top2
    ]
    gmsh.model.mesh.addNodes(3, ent, [1, 2, 3, 4, 5, 6], coords)
    # Relies on gmsh interpreting the prism's local frame from node order:
    # reversing the top winding produces a negative determinant.
    # This is stable across recent gmsh versions; if it ever changes,
    # use gmsh.model.mesh.getJacobian directly to confirm sign.
    # Element type 6 = 6-node prism.
    gmsh.model.mesh.addElements(3, ent, [6], [[100]], [[1, 2, 3, 4, 5, 6]])

    result = validate_structured_mesh(
        plan,
        mesh_plan,
        phantom_map,
        occ_entities=[],
        vol_tags=[],
        tol=1e-6,
        include_quality=True,
    )
    quality_errors = [i for i in result.errors if i.check == "element_quality"]
    assert len(quality_errors) >= 1
