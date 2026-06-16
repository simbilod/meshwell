from shapely.geometry import Polygon

from meshwell.structured.decompose import zinterval_footprint
from meshwell.structured.types import StructuredSlab


def slab(idx, poly, mesh_bool=True, mesh_order=1.0):
    return StructuredSlab(
        source_index=idx,
        footprint=poly,
        zlo=0.0,
        zhi=1.0,
        mesh_order=mesh_order,
        mesh_bool=mesh_bool,
        physical_name=("x",),
        identify_arcs=False,
        arc_tolerance=1e-3,
        min_arc_points=4,
    )


SQ_BIG = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
SQ_SMALL = Polygon([(2, 2), (5, 2), (5, 5), (2, 5)])


def test_single_slab_returns_its_footprint():
    fp = zinterval_footprint([slab(0, SQ_BIG)])
    assert fp.equals(SQ_BIG)


def test_two_slabs_union_when_same_mesh_order():
    fp = zinterval_footprint([slab(0, SQ_BIG), slab(1, SQ_SMALL)])
    assert fp.equals(SQ_BIG.union(SQ_SMALL))


def test_lower_mesh_order_carves_higher():
    # SQ_BIG mesh_order=2 (higher), SQ_SMALL mesh_order=1 (lower wins).
    fp = zinterval_footprint(
        [
            slab(0, SQ_BIG, mesh_order=2.0),
            slab(1, SQ_SMALL, mesh_order=1.0),
        ]
    )
    # Lower mesh_order keeps full footprint; higher gets (big - small).
    assert fp.equals(SQ_BIG)  # union of (SQ_SMALL) + (SQ_BIG - SQ_SMALL)


def test_void_subtracts():
    # mesh_bool=False should subtract from the footprint.
    fp = zinterval_footprint(
        [
            slab(0, SQ_BIG, mesh_order=1.0),
            slab(1, SQ_SMALL, mesh_order=2.0, mesh_bool=False),
        ]
    )
    # SQ_BIG processed first (lower mesh_order, keeps full).
    # SQ_SMALL has mesh_bool=False → subtract from accumulated.
    assert fp.equals(SQ_BIG.difference(SQ_SMALL))
