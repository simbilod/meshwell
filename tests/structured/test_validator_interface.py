"""Unit tests for check 2: prism <-> tet interface conformality."""
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
    gmsh.model.add("interface")
    try:
        yield
    finally:
        gmsh.finalize()


def test_conformal_wedge_tet_share_quad_passes(gmsh_session, empty_pm):  # noqa: ARG001
    """Conformal wedge/tet quad share should pass.

    One wedge sharing its lateral quad face with two coplanar tets,
    same 4 nodes. Should pass.
    """
    plan, mesh_plan, pm = empty_pm

    # Shared quad nodes: 1, 2, 5, 4 (y=0 plane)
    coords = [
        0,
        0,
        0,  # 1
        1,
        0,
        0,  # 2
        0,
        1,
        0,  # 3
        0,
        0,
        1,  # 4 (= node above 1)
        1,
        0,
        1,  # 5 (= node above 2)
        0,
        1,
        1,  # 6 (= node above 3)
        2,
        0,
        0,  # 7
        2,
        0,
        1,  # 8
    ]
    gmsh.model.mesh.addNodes(
        3,
        gmsh.model.addDiscreteEntity(3, -1, []),
        [1, 2, 3, 4, 5, 6, 7, 8],
        coords,
    )

    # Wedge on the left.
    wedge_vol = gmsh.model.addDiscreteEntity(3, -1, [])
    gmsh.model.mesh.addElements(3, wedge_vol, [6], [[100]], [[1, 2, 3, 4, 5, 6]])

    # Two tets on the right sharing the quad (1-2-5-4) split into triangles
    # (1-2-5) and (1-5-4), each completed with an apex node.
    tet_vol = gmsh.model.addDiscreteEntity(3, -1, [])
    gmsh.model.mesh.addElements(
        3,
        tet_vol,
        [4],
        [[200, 201, 202, 203]],
        [
            [
                1,
                2,
                5,
                7,
                1,
                5,
                4,
                8,
                7,
                8,
                5,
                2,
                7,
                1,
                5,
                8,
            ]
        ],
    )

    result = validate_structured_mesh(
        plan,
        mesh_plan,
        pm,
        occ_entities=[],
        vol_tags=[wedge_vol, tet_vol],
        tol=1e-6,
    )
    iface_errors = [i for i in result.errors if i.check == "prism_tet_interface"]
    # No T-junction or hanging node -- should pass.
    assert iface_errors == []


def test_wedge_quad_with_t_junction_on_tet_side_reported(
    gmsh_session, empty_pm  # noqa: ARG001
):
    """Wedge quad with a tet-side Steiner node is reported.

    Wedge quad face (1-2-5-4) but tet side introduces extra node 99 on
    edge 1-2, splitting into a non-matching triangulation.
    """
    plan, mesh_plan, pm = empty_pm

    coords = [
        0,
        0,
        0,  # 1
        1,
        0,
        0,  # 2
        0,
        1,
        0,  # 3
        0,
        0,
        1,  # 4
        1,
        0,
        1,  # 5
        0,
        1,
        1,  # 6
        2,
        0,
        0,  # 7
        0.5,
        0,
        0,  # 99 -- Steiner node on shared edge (T-junction!)
    ]
    gmsh.model.mesh.addNodes(
        3,
        gmsh.model.addDiscreteEntity(3, -1, []),
        [1, 2, 3, 4, 5, 6, 7, 99],
        coords,
    )

    wedge_vol = gmsh.model.addDiscreteEntity(3, -1, [])
    gmsh.model.mesh.addElements(3, wedge_vol, [6], [[100]], [[1, 2, 3, 4, 5, 6]])

    tet_vol = gmsh.model.addDiscreteEntity(3, -1, [])
    # 3 tets on the right, with the shared face built using node 99 instead
    # of node 1 or 2 -- so the wedge's quad-face {1,2,5,4} is NOT matched by
    # any set of two tet triangles sharing the same 4 nodes.
    gmsh.model.mesh.addElements(
        3,
        tet_vol,
        [4],
        [[200, 201, 202]],
        [
            [
                99,
                2,
                5,
                7,
                1,
                99,
                5,
                7,
                1,
                5,
                4,
                7,
            ]
        ],
    )

    result = validate_structured_mesh(
        plan,
        mesh_plan,
        pm,
        occ_entities=[],
        vol_tags=[wedge_vol, tet_vol],
        tol=1e-6,
    )
    iface_errors = [i for i in result.errors if i.check == "prism_tet_interface"]
    assert len(iface_errors) >= 1
