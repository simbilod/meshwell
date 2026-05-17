"""Unit tests for check 4: plan ↔ mesh consistency."""
import gmsh
import pytest

from meshwell.structured.spec import (
    PhantomMap,
    Slab,
    StructuredMeshPlan,
    StructuredPlan,
)
from meshwell.structured.validator import validate_structured_mesh


def _empty_slab(z_interval_index: int, n_pieces: int) -> Slab:
    """Make a Slab with `n_pieces` empty footprint entries.

    The validator doesn't need real geometry for this check.
    """
    from shapely.geometry import Polygon

    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    return Slab(
        footprint=poly,
        zlo=0.0,
        zhi=1.0,
        physical_name=("test",),
        source_index=0,
        z_interval_index=z_interval_index,
        mesh_order=1.0,
        face_partition=[poly] * n_pieces,
    )


def _make_wedges(vol_tag: int, count: int, start_node: int = 1) -> int:
    """Add `count` disjoint wedge elements to `vol_tag`. Returns next free node.

    Topology correctness doesn't matter for this check: the validator only
    queries element counts via gmsh.model.mesh.getElements; the wedges
    don't need to share nodes or form a valid mesh.
    """
    nodes: list[int] = []
    coords: list[float] = []
    elem_node_lists: list[int] = []
    next_node = start_node
    for k in range(count):
        base_x = float(k) * 2.0  # offset so wedges don't overlap.
        n_ids = [next_node + i for i in range(6)]
        next_node += 6
        nodes.extend(n_ids)
        coords.extend(
            [
                base_x + 0.0,
                0.0,
                0.0,
                base_x + 1.0,
                0.0,
                0.0,
                base_x + 0.0,
                1.0,
                0.0,
                base_x + 0.0,
                0.0,
                1.0,
                base_x + 1.0,
                0.0,
                1.0,
                base_x + 0.0,
                1.0,
                1.0,
            ]
        )
        elem_node_lists.extend(n_ids)
    gmsh.model.mesh.addNodes(3, vol_tag, nodes, coords)
    elem_tag_start = int(gmsh.model.mesh.getMaxElementTag()) + 1
    gmsh.model.mesh.addElements(
        3,
        vol_tag,
        [6],
        [list(range(elem_tag_start, elem_tag_start + count))],
        [elem_node_lists],
    )
    return next_node


@pytest.fixture
def gmsh_session():
    gmsh.initialize()
    gmsh.model.add("plan_consistency")
    try:
        yield
    finally:
        gmsh.finalize()


def test_one_slab_one_piece_correct_count_passes(gmsh_session):  # noqa: ARG001
    slab = _empty_slab(0, n_pieces=1)
    plan = StructuredPlan(slabs=(slab,), z_planes=(0.0, 1.0), overlaps=())
    mesh_plan = StructuredMeshPlan(slabs=(slab,), n_layers=(2,), recombine=(False,))

    vol = gmsh.model.addDiscreteEntity(3, -1, [])
    _make_wedges(vol, count=4)  # 2 layers x 2 triangles = 4.

    result = validate_structured_mesh(
        plan,
        mesh_plan,
        PhantomMap(),
        occ_entities=[],
        vol_tags=[vol],
        tol=1e-6,
    )
    plan_errors = [i for i in result.errors if i.check == "plan_mesh_consistency"]
    assert plan_errors == []


def test_vol_tag_count_mismatch_reported(gmsh_session):  # noqa: ARG001
    """Plan says 2 pieces but only 1 vol_tag — error."""
    slab = _empty_slab(0, n_pieces=2)
    plan = StructuredPlan(slabs=(slab,), z_planes=(0.0, 1.0), overlaps=())
    mesh_plan = StructuredMeshPlan(slabs=(slab,), n_layers=(2,), recombine=(False,))

    vol = gmsh.model.addDiscreteEntity(3, -1, [])
    _make_wedges(vol, count=4)

    result = validate_structured_mesh(
        plan,
        mesh_plan,
        PhantomMap(),
        occ_entities=[],
        vol_tags=[vol],
        tol=1e-6,
    )
    plan_errors = [i for i in result.errors if i.check == "plan_mesh_consistency"]
    assert any("vol_tag count" in i.message for i in plan_errors)


def test_element_count_not_multiple_of_n_layers_reported(gmsh_session):  # noqa: ARG001
    slab = _empty_slab(0, n_pieces=1)
    plan = StructuredPlan(slabs=(slab,), z_planes=(0.0, 1.0), overlaps=())
    mesh_plan = StructuredMeshPlan(slabs=(slab,), n_layers=(3,), recombine=(False,))

    vol = gmsh.model.addDiscreteEntity(3, -1, [])
    _make_wedges(vol, count=5)  # 5 is not a multiple of 3.

    result = validate_structured_mesh(
        plan,
        mesh_plan,
        PhantomMap(),
        occ_entities=[],
        vol_tags=[vol],
        tol=1e-6,
    )
    plan_errors = [i for i in result.errors if i.check == "plan_mesh_consistency"]
    assert any("not a multiple" in i.message for i in plan_errors)


def test_empty_volume_reported(gmsh_session):  # noqa: ARG001
    slab = _empty_slab(0, n_pieces=1)
    plan = StructuredPlan(slabs=(slab,), z_planes=(0.0, 1.0), overlaps=())
    mesh_plan = StructuredMeshPlan(slabs=(slab,), n_layers=(2,), recombine=(False,))

    vol = gmsh.model.addDiscreteEntity(3, -1, [])  # No elements added.

    result = validate_structured_mesh(
        plan,
        mesh_plan,
        PhantomMap(),
        occ_entities=[],
        vol_tags=[vol],
        tol=1e-6,
    )
    plan_errors = [i for i in result.errors if i.check == "plan_mesh_consistency"]
    assert any("zero elements" in i.message for i in plan_errors)
