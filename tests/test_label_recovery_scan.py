"""Label recovery must index the model's physical groups once, not per lookup.

``_recover_labels_from_cad`` walks every entity and, for each, every
interface group naming it. Asking the model for one group's dimtags at a
time would rescan all physical groups (and call ``getPhysicalName`` on
each) per lookup, which is quadratic in the group count. The tests below
pin the result against the physical groups the CAD stage actually wrote,
and bound how often the model is queried.
"""
from __future__ import annotations

import gmsh
import pytest
import shapely

from meshwell._mesh_entity import _MeshEntity
from meshwell.mesh import Mesh, _interfaces_by_side
from meshwell.model import ModelManager
from meshwell.orchestrator import cad
from meshwell.polyprism import PolyPrism
from meshwell.resolution import ConstantInField

INTERFACE_DELIMITER = "___"
BOUNDARY_DELIMITER = "None"


def _grid_scene(tmp_path, n: int, name: str = "grid") -> Mesh:
    """Build an ``n x n`` grid of abutting unit-height prisms and load it into a Mesh."""
    entities = [
        PolyPrism(
            polygons=shapely.box(i * 2.0, j * 2.0, (i + 1) * 2.0, (j + 1) * 2.0),
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name=f"cell_{i}_{j}",
            mesh_order=1,
        )
        for i in range(n)
        for j in range(n)
    ]
    xao = tmp_path / f"{name}_{n}.xao"
    cad(entities, output_file=xao)
    model_manager = ModelManager()
    model_manager.load_from_xao(xao)
    return Mesh(model=model_manager)


def _group_dimtags(model_manager: ModelManager) -> dict[str, list[tuple[int, int]]]:
    """Re-derive name -> dimtags straight from gmsh, independently of the code under test."""
    model = model_manager.model
    by_name: dict[str, list[tuple[int, int]]] = {}
    for dim, group_tag in model.getPhysicalGroups():
        name = model.getPhysicalName(dim, group_tag)
        by_name.setdefault(name, []).extend(
            (dim, tag) for tag in model.getEntitiesForPhysicalGroup(dim, group_tag)
        )
    return by_name


def _one_spec_per_cell(n: int) -> dict:
    """A global spec plus a per-cell spec, so every cell survives filtering."""
    specs: dict = {
        None: [ConstantInField(resolution=0.5, apply_to="volumes", entity_str="global")]
    }
    for i in range(n):
        for j in range(n):
            name = f"cell_{i}_{j}"
            specs[name] = [
                ConstantInField(resolution=0.2, apply_to="volumes", entity_str=name)
            ]
    return specs


@pytest.mark.parametrize("n", [2, 3])
def test_recovery_recovers_every_cell(tmp_path, n):
    """Cells keep their identity, dimension, and the volume tag CAD gave them."""
    mesh_obj = _grid_scene(tmp_path, n)
    try:
        groups = _group_dimtags(mesh_obj.model_manager)
        recovered_list, recovered = mesh_obj._recover_labels_from_cad(
            _one_spec_per_cell(n)
        )

        expected_names = sorted(f"cell_{i}_{j}" for i in range(n) for j in range(n))
        assert [e.physical_name for e in recovered_list] == expected_names
        for name in expected_names:
            entity = recovered[name]
            assert entity.dim == 3
            assert entity._explicit_dim is None
            assert entity.dimtags == groups[name]
    finally:
        mesh_obj.model_manager.finalize()


@pytest.mark.parametrize("n", [2, 3])
def test_interface_tags_are_shared_by_both_cells(tmp_path, n):
    """Each ``A___B`` group lands in A's and B's interfaces, boundaries included."""
    mesh_obj = _grid_scene(tmp_path, n)
    try:
        groups = _group_dimtags(mesh_obj.model_manager)
        _, recovered = mesh_obj._recover_labels_from_cad(_one_spec_per_cell(n))

        interface_groups = 0
        boundary_groups = 0
        for group_name, dimtags in groups.items():
            parts = group_name.split(INTERFACE_DELIMITER)
            if len(parts) != 2:
                continue
            left, right = parts
            tags = [tag for _, tag in dimtags]
            assert left in recovered
            if right == BOUNDARY_DELIMITER:
                boundary_groups += 1
                assert all(
                    tag in recovered[left].mesh_edge_name_interfaces for tag in tags
                )
            else:
                interface_groups += 1
                assert right in recovered
                for tag in tags:
                    assert tag in recovered[left].interfaces
                    assert tag in recovered[right].interfaces

        assert interface_groups > 0
        assert boundary_groups > 0
    finally:
        mesh_obj.model_manager.finalize()


def test_interior_cell_sees_all_four_neighbours(tmp_path):
    """A cell in the middle of the grid is named by four interface groups."""
    n = 3
    mesh_obj = _grid_scene(tmp_path, n)
    try:
        _, recovered = mesh_obj._recover_labels_from_cad(_one_spec_per_cell(n))
        assert len(recovered["cell_1_1"].interfaces) == 4
        # A corner cell abuts only its two in-grid neighbours.
        assert len(recovered["cell_0_0"].interfaces) == 2
    finally:
        mesh_obj.model_manager.finalize()


def _expected_interfaces_in_scan_order(mesh_obj, physical_name) -> list[int]:
    """Interface tags for one cell, in the order a name-by-name scan appends them.

    The ``A___B`` form contributes when the cell is the left side, the ``B___A``
    form when it is the right side, and a ``___None`` group goes to the
    edge-interface list instead.
    """
    group_by_name = _group_dimtags(mesh_obj.model_manager)
    tags: list[int] = []
    for other, dimtags in group_by_name.items():
        parts = other.split(INTERFACE_DELIMITER)
        if len(parts) != 2:
            continue
        if parts[0] == physical_name:
            if parts[1] != BOUNDARY_DELIMITER:
                tags.extend(tag for _, tag in dimtags)
        elif parts[1] == physical_name:
            tags.extend(tag for _, tag in dimtags)
    return tags


def _reference_scan_order(
    names: list[str], physical_name: str
) -> list[tuple[str, bool, str]]:
    """What a name-by-name scan contributes for one entity, in order.

    The scan walks every name and takes the first matching branch, so a
    name matches at most once per entity and the first side wins.
    """
    out: list[tuple[str, bool, str]] = []
    for other in names:
        parts = other.split(INTERFACE_DELIMITER)
        if len(parts) != 2:
            continue
        if parts[0] == physical_name:
            out.append((other, True, parts[1]))
        elif parts[1] == physical_name:
            out.append((other, False, parts[0]))
    return out


@pytest.mark.parametrize(
    "names",
    [
        ["A___B", "B___A", "A___None", "C___A", "not_an_interface"],
        # Deliberately not in sorted order, with a self-named group and a
        # duplicated name (two groups can share a name at different dims).
        ["z___a", "a___A", "m___z", "z___a", "a___m"],
        ["___B", "A___", "A___A", "A___None"],
    ],
)
def test_interfaces_by_side_matches_the_scan_order(names):
    """Bucketing by side reproduces the name-by-name scan exactly."""
    by_side = _interfaces_by_side(names, INTERFACE_DELIMITER)
    sides = {n for name in names for n in name.split(INTERFACE_DELIMITER)}
    for side in sides:
        assert by_side.get(side, []) == _reference_scan_order(names, side), side


def test_interfaces_by_side_drops_multi_part_names():
    """Only ``<A><delim><B>`` is an interface group."""
    by_side = _interfaces_by_side(["A___B___C", "A___B"], INTERFACE_DELIMITER)
    assert by_side["A"] == [("A___B", True, "B")]


@pytest.mark.parametrize("n", [2, 3])
def test_interface_tags_follow_the_scan_order(tmp_path, n):
    """The recovered interfaces are what the scan-order reference produces."""
    mesh_obj = _grid_scene(tmp_path, n)
    try:
        _, recovered = mesh_obj._recover_labels_from_cad(_one_spec_per_cell(n))
        for i in range(n):
            for j in range(n):
                name = f"cell_{i}_{j}"
                assert recovered[name].interfaces == (
                    _expected_interfaces_in_scan_order(mesh_obj, name)
                ), name
    finally:
        mesh_obj.model_manager.finalize()


class _ModelQueryCounter:
    """Count physical-group queries against the live gmsh model."""

    def __init__(self, model):
        self.model = model
        self.names = 0
        self.entity_scans = 0

    def __enter__(self):
        self._name = self.model.getPhysicalName
        self._entities = self.model.getEntities

        def name(*args, **kwargs):
            self.names += 1
            return self._name(*args, **kwargs)

        def entities(*args, **kwargs):
            self.entity_scans += 1
            return self._entities(*args, **kwargs)

        self.model.getPhysicalName = name
        self.model.getEntities = entities
        return self

    def __exit__(self, *_exc):
        self.model.getPhysicalName = self._name
        self.model.getEntities = self._entities


def _query_counts(mesh_obj: Mesh, n: int) -> tuple[int, int, int]:
    with _ModelQueryCounter(mesh_obj.model_manager.model) as counter:
        mesh_obj._recover_labels_from_cad(_one_spec_per_cell(n))
    groups = len(mesh_obj.model_manager.model.getPhysicalGroups())
    return counter.names, counter.entity_scans, groups


def test_recovery_queries_the_model_a_bounded_number_of_times(tmp_path):
    """Query count grows with the group count, not with its square."""
    counts = {}
    for n in (3, 6):
        mesh_obj = _grid_scene(tmp_path, n)
        try:
            counts[n] = _query_counts(mesh_obj, n)
        finally:
            mesh_obj.model_manager.finalize()

    for n, (names, scans, groups) in counts.items():
        # One getPhysicalName per group, plus the top-names scan.
        assert names <= 2 * groups, f"{n}x{n}: {names} name queries for {groups} groups"
        # One entity scan per dimension, regardless of how many entities exist.
        assert scans == 4, f"{n}x{n}: {scans} entity scans"

    # Four times the entities must not mean an order of magnitude more queries.
    assert counts[6][0] < 8 * counts[3][0]


def test_recovery_survives_a_missing_model():
    """An uninitialized manager yields no entities rather than raising."""
    mesh_obj = Mesh(model=ModelManager())
    assert mesh_obj._recover_labels_from_cad({}) == ([], {})


def test_update_boundaries_accepts_a_shared_entity_snapshot():
    """Passing the snapshot gives what reading the model per call gives."""
    gmsh.initialize()
    try:
        gmsh.model.add("snapshot")
        box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()
        snapshot = {
            dim: {tag for _, tag in gmsh.model.getEntities(dim)} for dim in range(4)
        }

        from_model = _MeshEntity(
            index=0, model=gmsh.model, dimtags=[(3, box)], physical_name="A"
        )
        with _ModelQueryCounter(gmsh.model) as counter:
            expected = from_model.update_boundaries()
        assert counter.entity_scans == 4

        from_snapshot = _MeshEntity(
            index=0, model=gmsh.model, dimtags=[(3, box)], physical_name="A"
        )
        with _ModelQueryCounter(gmsh.model) as counter:
            actual = from_snapshot.update_boundaries(snapshot)
        assert counter.entity_scans == 0

        assert actual == expected
        assert from_snapshot.dimtags == from_model.dimtags
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
