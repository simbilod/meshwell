"""End-to-end: a structured cohort carves lower-priority overlapping cladding.

``classify_cohort_overlaps`` lets an unstructured entity overlap a cohort
when it has strictly lower priority (larger / ``None`` ``mesh_order``);
cad_occ then cuts it with the cohort's carve prisms. The auto-carved
result must match hand-carving the cladding, for polygonal and arc
(cylindrical-lateral) cohorts.
"""

from __future__ import annotations

import numpy as np
import pytest
import shapely

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.resolution import StructuredExtrusionResolutionSpec
from meshwell.structured.exceptions import StructuredVolumetricOverlapError

DOM = shapely.box(-1, -1, 5, 5)
BOX = shapely.box(1, 1, 3, 2)


def _sector(r0=4.0, r1=4.5, a0=45.0, a1=75.0, n=64) -> shapely.Polygon:
    t = np.radians(np.linspace(a0, a1, n))
    outer = np.c_[r1 * np.cos(t), r1 * np.sin(t)]
    inner = np.c_[r0 * np.cos(t), r0 * np.sin(t)][::-1]
    return shapely.Polygon(np.vstack([outer, inner]))


def _prism(poly, z0, z1, name, structured=False, mesh_order=None, **kw):
    return PolyPrism(
        polygons=poly,
        buffers={z0: 0.0, z1: 0.0},
        physical_name=name,
        structured=structured,
        mesh_order=mesh_order,
        **kw,
    )


def _mesh(entities):
    return generate_mesh(
        entities=entities,
        dim=3,
        output_mesh=None,
        default_characteristic_length=0.5,
        resolution_specs={"core": [StructuredExtrusionResolutionSpec(n_layers=2)]},
        identify_arcs=True,
    )


def _tet_volumes(pts, tets):
    a, b, c, d = (pts[tets[:, i]] for i in range(4))
    return np.abs(np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)) / 6.0


# Split each 3D cell type into tetrahedra (vertex index tuples).
_SPLITS = {
    "tetra": [(0, 1, 2, 3)],
    "pyramid": [(0, 1, 2, 4), (0, 2, 3, 4)],
    "wedge": [(0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5)],
    "hexahedron": [
        (0, 1, 3, 4),
        (1, 2, 3, 6),
        (1, 4, 5, 6),
        (3, 4, 6, 7),
        (1, 3, 4, 6),
    ],
}


def _group_volume(mesh, name) -> float:
    total = 0.0
    for block, cell_ids in zip(mesh.cells, mesh.cell_sets[name], strict=True):
        if block.type not in _SPLITS or len(cell_ids) == 0:
            continue
        cells = block.data[cell_ids]
        for split in _SPLITS[block.type]:
            total += _tet_volumes(mesh.points, cells[:, split]).sum()
    return total


def _groups(mesh) -> list[str]:
    return sorted(k for k in mesh.cell_sets_dict if not k.startswith("gmsh"))


def _cell_types(mesh) -> set[str]:
    return {c.type for c in mesh.cells if len(c.data)}


@pytest.mark.parametrize("core", [BOX, _sector()], ids=["box", "sector"])
def test_auto_carve_matches_hand_carve(core):
    base = [_prism(core, 0, 0.4, "core", True, 1), _prism(DOM, -1, 0, "sub", False, 5)]
    top = _prism(DOM, 0.4, 1.0, "clad", False, 5)
    hand = _mesh([*base, _prism(DOM.difference(core), 0, 0.4, "clad", False, 5), top])
    auto = _mesh([*base, _prism(DOM, 0, 0.4, "clad", False, 5), top])

    assert _groups(auto) == _groups(hand)
    assert "wedge" in _cell_types(auto)
    expected_clad = DOM.area * 1.0 - core.area * 0.4
    for name, expected in (("clad", expected_clad), ("core", core.area * 0.4)):
        assert _group_volume(auto, name) == pytest.approx(expected, rel=2e-3)
        assert _group_volume(auto, name) == pytest.approx(
            _group_volume(hand, name), rel=2e-3
        )


@pytest.mark.parametrize("core", [BOX, _sector()], ids=["box", "sector"])
def test_tall_cladding_with_none_mesh_order_is_carved(core):
    # One cladding spanning past the cohort in z, with mesh_order=None.
    mesh = _mesh([_prism(core, 0, 0.4, "core", True, 1), _prism(DOM, -1, 1, "clad")])
    assert {"core", "clad", "core___clad"} <= set(_groups(mesh))
    assert _group_volume(mesh, "clad") == pytest.approx(
        DOM.area * 2.0 - core.area * 0.4, rel=2e-3
    )


def test_stepped_cohort_is_carved():
    upper = shapely.box(1.5, 1.2, 2.5, 1.8)
    mesh = _mesh(
        [
            _prism(BOX, 0, 0.4, "core", True, 1),
            _prism(upper, 0.4, 0.8, "core", True, 1),
            _prism(DOM, -1, 1, "clad", False, 5),
        ]
    )
    assert _group_volume(mesh, "clad") == pytest.approx(
        DOM.area * 2.0 - (BOX.area + upper.area) * 0.4, rel=2e-3
    )
    assert _group_volume(mesh, "core") == pytest.approx(
        (BOX.area + upper.area) * 0.4, rel=2e-3
    )


@pytest.mark.parametrize("hole", [BOX, _sector()], ids=["box", "sector"])
def test_standalone_void_cohort_carves_cladding(hole):
    # A void that carves no structured slab emits no cohort sub-solid; the
    # leftover-void carve tool must still remove it from the cladding.
    mesh = _mesh(
        [
            _prism(hole, 0, 0.4, "core", True, 1, mesh_bool=False),
            _prism(DOM, -1, 1, "clad", False, 5),
        ]
    )
    assert "core" not in mesh.cell_sets
    assert _group_volume(mesh, "clad") == pytest.approx(
        DOM.area * 2.0 - hole.area * 0.4, rel=2e-3
    )


def test_void_straddling_structured_slab_carves_both():
    hole = shapely.box(2.5, 1.2, 3.5, 1.8)  # half inside BOX, half outside
    mesh = _mesh(
        [
            _prism(hole, 0, 0.4, "void", True, 1, mesh_bool=False),
            _prism(BOX, 0, 0.4, "core", True, 2),
            _prism(DOM, -1, 1, "clad", False, 5),
        ]
    )
    assert "void" not in mesh.cell_sets
    assert _group_volume(mesh, "core") == pytest.approx(
        BOX.difference(hole).area * 0.4, rel=2e-3
    )
    assert _group_volume(mesh, "clad") == pytest.approx(
        DOM.area * 2.0 - BOX.union(hole).area * 0.4, rel=2e-3
    )


@pytest.mark.parametrize("clad_order", [1, 0.5], ids=["equal", "higher"])
def test_equal_or_higher_priority_cladding_raises(clad_order):
    with pytest.raises(StructuredVolumetricOverlapError):
        _mesh(
            [
                _prism(BOX, 0, 0.4, "core", True, 1),
                _prism(DOM, -1, 1, "clad", False, clad_order),
            ]
        )


def _n_subpieces(entities) -> int:
    from meshwell.structured.pipeline import structured_pre_pass

    state = structured_pre_pass(entities, point_tolerance=1e-3)
    return sum(
        len(e.slab_meta) for e in state.entities_out if getattr(e, "is_cohort", False)
    )


def test_carved_outline_does_not_split_cohort_regions():
    # The cladding's own outline (x=2) crosses the cohort, but the cohort
    # carves it there, so it must not split the cohort into two regions.
    clad = shapely.box(2, 0, 5, 3)
    ents = [
        _prism(BOX, 0, 0.4, "core", True, 1),
        _prism(clad, 0, 0.4, "clad", False, 5),
    ]
    assert _n_subpieces(ents) == 1
    mesh = _mesh(ents)
    assert _group_volume(mesh, "core") == pytest.approx(BOX.area * 0.4, rel=2e-3)
    assert _group_volume(mesh, "clad") == pytest.approx(
        clad.difference(BOX).area * 0.4, rel=2e-3
    )


def test_carved_outline_on_lower_step_is_kept_as_interface():
    # Cladding z 0.4..1 overlaps the upper step. What survives the carve
    # sits on the lower slab's top for x > 2, so x=2 IS an interface there.
    upper = shapely.box(1.5, 1.2, 2.5, 1.8)
    clad = shapely.box(2, 0, 5, 3)
    ents = [
        _prism(BOX, 0, 0.4, "core", True, 1),
        _prism(upper, 0.4, 0.8, "core", True, 1),
        _prism(clad, 0.4, 1.0, "clad", False, 5),
    ]
    mesh = _mesh(ents)
    assert _group_volume(mesh, "core") == pytest.approx(
        (BOX.area + upper.area) * 0.4, rel=2e-3
    )
    assert _group_volume(mesh, "clad") == pytest.approx(
        clad.area * 0.6 - clad.intersection(upper).area * 0.4, rel=2e-3
    )
