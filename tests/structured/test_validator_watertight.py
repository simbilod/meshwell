"""Unit tests for check 1: watertight (per-volume face occurrence)."""
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
    return (
        StructuredPlan(slabs=(), z_planes=(), overlaps=()),
        StructuredMeshPlan(slabs=(), n_layers=(), recombine=()),
        PhantomMap(),
    )


@pytest.fixture
def gmsh_session():
    gmsh.initialize()
    gmsh.model.add("watertight")
    try:
        yield
    finally:
        gmsh.finalize()


def _add_single_tet(vol_tag: int):
    """Add a unit tet to vol_tag."""
    gmsh.model.mesh.addNodes(
        3,
        vol_tag,
        [1, 2, 3, 4],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    )
    gmsh.model.mesh.addElements(3, vol_tag, [4], [[10]], [[1, 2, 3, 4]])


def test_single_tet_passes_watertight(gmsh_session, empty_inputs):  # noqa: ARG001
    plan, mesh_plan, phantom_map = empty_inputs
    vol = gmsh.model.addDiscreteEntity(3, -1, [])
    _add_single_tet(vol)
    result = validate_structured_mesh(
        plan,
        mesh_plan,
        phantom_map,
        occ_entities=[],
        vol_tags=[vol],
        tol=1e-6,
    )
    wt_errors = [i for i in result.errors if i.check == "watertight"]
    assert wt_errors == []


def test_three_tets_sharing_a_face_reported_as_error(
    gmsh_session,  # noqa: ARG001
    empty_inputs,
):
    """Build 3 tets that share a single face — same volume, face count = 3."""
    plan, mesh_plan, phantom_map = empty_inputs
    vol = gmsh.model.addDiscreteEntity(3, -1, [])

    # 6 nodes: 3 nodes form a "shared face" + 3 distinct apex nodes
    # so 3 tets each contain the same base face.
    gmsh.model.mesh.addNodes(
        3,
        vol,
        [1, 2, 3, 4, 5, 6],
        [
            0,
            0,
            0,  # 1 (face)
            1,
            0,
            0,  # 2 (face)
            0,
            1,
            0,  # 3 (face)
            0,
            0,
            1,  # 4 apex a
            0,
            0,
            -1,  # 5 apex b
            0,
            0,
            2,  # 6 apex c
        ],
    )
    gmsh.model.mesh.addElements(
        3,
        vol,
        [4],
        [[100, 101, 102]],
        [[1, 2, 3, 4, 1, 2, 3, 5, 1, 2, 3, 6]],
    )

    result = validate_structured_mesh(
        plan,
        mesh_plan,
        phantom_map,
        occ_entities=[],
        vol_tags=[vol],
        tol=1e-6,
    )
    wt_errors = [i for i in result.errors if i.check == "watertight"]
    assert len(wt_errors) >= 1
    assert any(
        "3 elements" in i.message or "3 occurrences" in i.message for i in wt_errors
    )
