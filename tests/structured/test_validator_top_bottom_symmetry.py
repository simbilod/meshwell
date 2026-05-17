"""Unit tests for check 8: top↔bottom z-translation symmetry."""
import gmsh
import pytest

from meshwell.structured.spec import (
    PhantomMap,
    StructuredMeshPlan,
    StructuredPlan,
)
from meshwell.structured.validator import validate_structured_mesh


@pytest.fixture
def empty_pm():
    return (
        StructuredPlan(slabs=(), z_planes=(), overlaps=()),
        StructuredMeshPlan(slabs=(), n_layers=(), recombine=()),
        PhantomMap(),
    )


@pytest.fixture
def gmsh_session():
    gmsh.initialize()
    gmsh.model.add("symmetry")
    try:
        yield
    finally:
        gmsh.finalize()


def test_symmetric_wedge_passes(gmsh_session, empty_pm):  # noqa: ARG001
    plan, mesh_plan, pm = empty_pm
    vol = gmsh.model.addDiscreteEntity(3, -1, [])
    coords = [
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        1,
        0,
        1,
        1,
    ]
    gmsh.model.mesh.addNodes(3, vol, [1, 2, 3, 4, 5, 6], coords)
    gmsh.model.mesh.addElements(3, vol, [6], [[100]], [[1, 2, 3, 4, 5, 6]])

    result = validate_structured_mesh(
        plan,
        mesh_plan,
        pm,
        occ_entities=[],
        vol_tags=[vol],
        tol=1e-6,
    )
    sym_errors = [i for i in result.errors if i.check == "top_bottom_symmetry"]
    assert sym_errors == []


def test_misaligned_top_reported(gmsh_session, empty_pm):  # noqa: ARG001
    plan, mesh_plan, pm = empty_pm
    vol = gmsh.model.addDiscreteEntity(3, -1, [])
    coords = [
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0.01,
        0,
        1,  # x offset by 0.01 — > tol=1e-3
        1.01,
        0,
        1,
        0.01,
        1,
        1,
    ]
    gmsh.model.mesh.addNodes(3, vol, [1, 2, 3, 4, 5, 6], coords)
    gmsh.model.mesh.addElements(3, vol, [6], [[100]], [[1, 2, 3, 4, 5, 6]])

    result = validate_structured_mesh(
        plan,
        mesh_plan,
        pm,
        occ_entities=[],
        vol_tags=[vol],
        tol=1e-3,
    )
    sym_errors = [i for i in result.errors if i.check == "top_bottom_symmetry"]
    assert len(sym_errors) >= 1


def test_non_structured_volume_skipped(gmsh_session, empty_pm):  # noqa: ARG001
    """Tet volumes are not subject to z-translation symmetry."""
    plan, mesh_plan, pm = empty_pm
    vol = gmsh.model.addDiscreteEntity(3, -1, [])
    gmsh.model.mesh.addNodes(3, vol, [1, 2, 3, 4], [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1])
    gmsh.model.mesh.addElements(3, vol, [4], [[100]], [[1, 2, 3, 4]])

    result = validate_structured_mesh(
        plan,
        mesh_plan,
        pm,
        occ_entities=[],
        vol_tags=[vol],
        tol=1e-6,
    )
    sym_errors = [i for i in result.errors if i.check == "top_bottom_symmetry"]
    assert sym_errors == []


def test_intermediate_layer_misalignment_reported(
    gmsh_session, empty_pm  # noqa: ARG001
):
    """A 3-layer wedge stack with an intermediate layer offset in xy must be reported.

    The original bot↔top check would have missed this; the all-planes check
    catches intermediate layer drift.
    """
    plan, mesh_plan, pm = empty_pm
    vol = gmsh.model.addDiscreteEntity(3, -1, [])

    # Three planes: z=0 (bot), z=0.5 (mid, offset by 0.05 in x), z=1 (top, aligned with bot).
    coords = [
        0,
        0,
        0,  # 1 bot
        1,
        0,
        0,  # 2 bot
        0,
        1,
        0,  # 3 bot
        0.05,
        0,
        0.5,  # 4 mid (shifted +0.05 in x)
        1.05,
        0,
        0.5,
        0.05,
        1,
        0.5,
        0,
        0,
        1,  # 7 top (aligned with bot)
        1,
        0,
        1,
        0,
        1,
        1,
    ]
    gmsh.model.mesh.addNodes(3, vol, list(range(1, 10)), coords)
    # Two stacked wedges using these 9 nodes.
    gmsh.model.mesh.addElements(
        3, vol, [6], [[100, 101]], [[1, 2, 3, 4, 5, 6, 4, 5, 6, 7, 8, 9]]
    )

    result = validate_structured_mesh(
        plan,
        mesh_plan,
        pm,
        occ_entities=[],
        vol_tags=[vol],
        tol=1e-3,
    )
    sym_errors = [i for i in result.errors if i.check == "top_bottom_symmetry"]
    # Mid layer at z=0.5 is offset by 0.05; the check should report it.
    assert len(sym_errors) >= 1
    assert any("z=0.5" in i.message or "0.5" in i.message for i in sym_errors)
