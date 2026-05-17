"""Unit tests for check 3: internal-seam faces unmeshed."""
import gmsh
import pytest

from meshwell.structured.spec import (
    LateralKey,
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
    )


@pytest.fixture
def gmsh_session():
    gmsh.initialize()
    gmsh.model.add("seams")
    try:
        yield
    finally:
        gmsh.finalize()


def test_no_internal_seams_passes(gmsh_session, empty_pm):  # noqa: ARG001
    plan, mesh_plan = empty_pm
    pm = PhantomMap()  # No laterals.
    result = validate_structured_mesh(
        plan,
        mesh_plan,
        pm,
        occ_entities=[],
        vol_tags=[],
        tol=1e-6,
    )
    seam_issues = [i for i in result.errors if i.check == "internal_seam_unmeshed"]
    assert seam_issues == []


def test_internal_seam_with_2d_elements_reported(
    gmsh_session,  # noqa: ARG001
    empty_pm,
):
    """Construct a PhantomMap with two LateralKeys for the same slab.

    Both pieces map to the same face_tag that carries 2D elements — the
    validator must report it.
    """
    plan, mesh_plan = empty_pm

    # Create a real surface entity with a triangle on it.
    face_tag = gmsh.model.addDiscreteEntity(2, -1, [])
    gmsh.model.mesh.addNodes(
        2,
        face_tag,
        [1, 2, 3],
        [0, 0, 0, 1, 0, 0, 0, 1, 0],
    )
    gmsh.model.mesh.addElements(2, face_tag, [2], [[10]], [[1, 2, 3]])

    # PhantomMap: two LateralKeys (same slab, different pieces) both
    # mapping to the same face_tag → that face IS an internal seam.
    # int-passthrough convention: tests inject gmsh face tags directly
    # in output_laterals values (Task 8 will wire OCC→gmsh resolution).
    pm = PhantomMap()
    key_a = LateralKey(slab_index=0, piece_index=0, outer_edge_index=0)
    key_b = LateralKey(slab_index=0, piece_index=1, outer_edge_index=0)
    pm.output_laterals[key_a] = [face_tag]
    pm.output_laterals[key_b] = [face_tag]

    result = validate_structured_mesh(
        plan,
        mesh_plan,
        pm,
        occ_entities=[],
        vol_tags=[],
        tol=1e-6,
    )
    seam_errors = [i for i in result.errors if i.check == "internal_seam_unmeshed"]
    assert len(seam_errors) >= 1


def test_internal_seam_without_2d_elements_passes(
    gmsh_session,  # noqa: ARG001
    empty_pm,
):
    plan, mesh_plan = empty_pm
    face_tag = gmsh.model.addDiscreteEntity(2, -1, [])  # No elements.

    pm = PhantomMap()
    pm.output_laterals[LateralKey(0, 0, 0)] = [face_tag]
    pm.output_laterals[LateralKey(0, 1, 0)] = [face_tag]

    result = validate_structured_mesh(
        plan,
        mesh_plan,
        pm,
        occ_entities=[],
        vol_tags=[],
        tol=1e-6,
    )
    seam_errors = [i for i in result.errors if i.check == "internal_seam_unmeshed"]
    assert seam_errors == []
