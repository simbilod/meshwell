"""_group_slabs_into_vertical_stacks: groups stacked pieces, separates by gap."""

from __future__ import annotations

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.phantom import _group_slabs_into_vertical_stacks
from meshwell.structured.plan import build_plan


def _square(x0, y0, x1, y1):
    return shapely.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _polyprism(name, z0, z1, mesh_order):
    return PolyPrism(
        polygons=_square(0, 0, 1, 1),
        buffers={float(z0): 0.0, float(z1): 0.0},
        physical_name=name,
        mesh_order=mesh_order,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )


def test_vertical_stack_groups_touching_slabs_same_xy():
    """A (z=0..1) and B (z=1..2) same XY -> one stack of length 2."""
    plan = build_plan([_polyprism("A", 0, 1, 1), _polyprism("B", 1, 2, 2)])
    stacks = _group_slabs_into_vertical_stacks(plan)
    multi = [s for s in stacks if len(s) > 1]
    assert len(multi) == 1
    assert len(multi[0]) == 2
    zlos = [slab.zlo for slab, _piece_idx in multi[0]]
    assert zlos == sorted(zlos)


def test_gap_separates_stacks():
    """A (z=0..1) and C (z=10..11): gap >= _Z_TOL_VERT -> two separate singletons."""
    plan = build_plan([_polyprism("A", 0, 1, 1), _polyprism("C", 10, 11, 2)])
    stacks = _group_slabs_into_vertical_stacks(plan)
    assert all(len(s) == 1 for s in stacks)


def test_disjoint_xy_no_stack():
    """Two slabs same Z, different XY -> two singletons (no vertical stack)."""
    A = _polyprism("A", 0, 1, 1)
    B = PolyPrism(
        polygons=_square(5, 5, 6, 6),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="B",
        mesh_order=2,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )
    plan = build_plan([A, B])
    stacks = _group_slabs_into_vertical_stacks(plan)
    assert all(len(s) == 1 for s in stacks)
