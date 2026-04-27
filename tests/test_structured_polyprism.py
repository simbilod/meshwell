"""Tests for structured-mode PolyPrism (gmsh-tutorial-t3-style layered extrusion)."""
from __future__ import annotations

import pytest
from shapely.geometry import Polygon


@pytest.fixture
def square_poly():
    return Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


def test_polyprism_without_n_layers_is_unstructured(square_poly):
    """Default PolyPrism path is untouched; same class returned."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured_polyprism import _StructuredPolyPrism

    pp = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="film",
    )
    assert type(pp) is PolyPrism
    assert not isinstance(pp, _StructuredPolyPrism)


def test_polyprism_with_n_layers_dispatches_to_structured(square_poly):
    """Passing n_layers triggers __new__ -> _StructuredPolyPrism instance.

    The user-facing class name is still ``PolyPrism``; isinstance(..., PolyPrism)
    must remain true.
    """
    from meshwell.polyprism import PolyPrism
    from meshwell.structured_polyprism import _StructuredPolyPrism

    pp = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[4],
        physical_name="film",
    )
    assert isinstance(pp, _StructuredPolyPrism)
    assert isinstance(pp, PolyPrism)
    assert pp.n_layers == [4]
    assert pp.recombine is False
    assert pp.physical_name == ("film",)


def test_structured_mode_requires_zero_buffers(square_poly):
    from meshwell.polyprism import PolyPrism

    with pytest.raises(ValueError, match="zero"):
        PolyPrism(
            polygons=square_poly,
            buffers={0.0: 0.0, 1.0: 0.1},  # nonzero buffer
            n_layers=[4],
        )


def test_structured_mode_requires_n_layers_length(square_poly):
    from meshwell.polyprism import PolyPrism

    with pytest.raises(ValueError, match="n_layers"):
        PolyPrism(
            polygons=square_poly,
            buffers={0.0: 0.0, 1.0: 0.0},
            n_layers=[4, 8],  # too many
        )

    with pytest.raises(ValueError, match="n_layers"):
        PolyPrism(
            polygons=square_poly,
            buffers={0.0: 0.0, 0.5: 0.0, 1.0: 0.0},
            n_layers=[4],  # too few
        )


def test_structured_mode_rejects_non_positive_layers(square_poly):
    from meshwell.polyprism import PolyPrism

    with pytest.raises(ValueError, match="n_layers"):
        PolyPrism(
            polygons=square_poly,
            buffers={0.0: 0.0, 1.0: 0.0},
            n_layers=[0],
        )


def test_structured_mode_rejects_non_increasing_z(square_poly):
    from meshwell.polyprism import PolyPrism

    with pytest.raises(ValueError, match="z"):
        PolyPrism(
            polygons=square_poly,
            buffers={1.0: 0.0, 0.0: 0.0},  # not strictly increasing in dict order
            n_layers=[4],
        )


def test_structured_mode_rejects_additive_or_subdivision(square_poly):
    """`additive=True` and `subdivision=` are unstructured-only knobs."""
    from meshwell.polyprism import PolyPrism

    with pytest.raises(ValueError, match=r"additive|subdivision"):
        PolyPrism(
            polygons=square_poly,
            buffers={0.0: 0.0, 1.0: 0.0},
            n_layers=[4],
            additive=True,
        )

    with pytest.raises(ValueError, match=r"additive|subdivision"):
        PolyPrism(
            polygons=square_poly,
            buffers={0.0: 0.0, 1.0: 0.0},
            n_layers=[4],
            subdivision=(2, 2, 1),
        )


def test_polyprism_init_direct_with_n_layers_raises(square_poly):
    """Bypassing __new__ and calling __init__ directly with n_layers must error.

    Guards against future from_dict/round-trip code paths constructing a
    bare PolyPrism with n_layers set, which would silently produce a
    non-structured prism.
    """
    from meshwell.polyprism import PolyPrism

    bare = object.__new__(PolyPrism)
    with pytest.raises(TypeError, match="n_layers"):
        PolyPrism.__init__(
            bare,
            polygons=square_poly,
            buffers={0.0: 0.0, 1.0: 0.0},
            n_layers=[4],
        )


def test_expand_slabs_single_interval(square_poly):
    from meshwell.polyprism import PolyPrism
    from meshwell.structured_polyprism import expand_slabs_for_entity

    sp = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[4],
        physical_name="film",
        mesh_order=2.0,
    )
    slabs = expand_slabs_for_entity(sp, source_index=7)
    assert len(slabs) == 1
    s = slabs[0]
    assert s.zlo == 0.0
    assert s.zhi == 1.0
    assert s.n_layers == 4
    assert s.physical_name == ("film",)
    assert s.source_index == 7
    assert s.mesh_order == 2.0
    # Footprint is a MultiPolygon wrapping the original square.
    from shapely.geometry import MultiPolygon

    assert isinstance(s.footprint, MultiPolygon)
    assert len(s.footprint.geoms) == 1
    assert s.footprint.geoms[0].equals(square_poly)


def test_expand_slabs_multi_interval(square_poly):
    from meshwell.polyprism import PolyPrism
    from meshwell.structured_polyprism import expand_slabs_for_entity

    sp = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 0.5: 0.0, 1.0: 0.0},
        n_layers=[8, 2],
        physical_name="film",
    )
    slabs = expand_slabs_for_entity(sp, source_index=0)
    assert len(slabs) == 2
    assert (slabs[0].zlo, slabs[0].zhi, slabs[0].n_layers) == (0.0, 0.5, 8)
    assert (slabs[1].zlo, slabs[1].zhi, slabs[1].n_layers) == (0.5, 1.0, 2)


def test_expand_slabs_handles_multipolygon_input():
    """If user passes a MultiPolygon, expand preserves all geoms."""
    from shapely.geometry import MultiPolygon, Polygon

    from meshwell.polyprism import PolyPrism
    from meshwell.structured_polyprism import expand_slabs_for_entity

    a = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    b = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
    mp = MultiPolygon([a, b])
    sp = PolyPrism(
        polygons=mp,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[3],
        physical_name="film",
    )
    slabs = expand_slabs_for_entity(sp, source_index=0)
    assert len(slabs) == 1
    assert isinstance(slabs[0].footprint, MultiPolygon)
    assert len(slabs[0].footprint.geoms) == 2


def test_resolve_disjoint_xy_identity(square_poly):
    from shapely.affinity import translate

    from meshwell.polyprism import PolyPrism
    from meshwell.structured_polyprism import resolve_structured_slabs

    sp1 = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[4],
        physical_name="A",
    )
    sp2 = PolyPrism(
        polygons=translate(square_poly, xoff=2.0),
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[3],
        physical_name="B",
    )
    slabs = resolve_structured_slabs([sp1, sp2])
    assert len(slabs) == 2
    names = {s.physical_name[0] for s in slabs}
    assert names == {"A", "B"}


def test_resolve_disjoint_z_identity(square_poly):
    from meshwell.polyprism import PolyPrism
    from meshwell.structured_polyprism import resolve_structured_slabs

    sp1 = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[4],
        physical_name="A",
    )
    sp2 = PolyPrism(
        polygons=square_poly,  # same xy
        buffers={2.0: 0.0, 3.0: 0.0},  # disjoint z
        n_layers=[3],
        physical_name="B",
    )
    slabs = resolve_structured_slabs([sp1, sp2])
    assert len(slabs) == 2
    z_ranges = {(s.zlo, s.zhi) for s in slabs}
    assert z_ranges == {(0.0, 1.0), (2.0, 3.0)}


def test_resolve_filters_non_structured_entities(square_poly):
    from meshwell.polyprism import PolyPrism
    from meshwell.structured_polyprism import resolve_structured_slabs

    class _Other:
        polygons = square_poly
        physical_name = ("foo",)
        mesh_order = None

    sp = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[4],
        physical_name="film",
    )
    slabs = resolve_structured_slabs([_Other(), sp, _Other()])
    assert len(slabs) == 1
    assert slabs[0].physical_name == ("film",)


def test_resolve_xy_overlap_same_z():
    from shapely.geometry import MultiPolygon, Polygon

    from meshwell.polyprism import PolyPrism
    from meshwell.structured_polyprism import resolve_structured_slabs

    big = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    small = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])  # inside big

    sp_big = PolyPrism(
        polygons=big,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[2],
        physical_name="bg",
        mesh_order=2.0,
    )
    sp_small = PolyPrism(
        polygons=small,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[5],
        physical_name="hole",
        mesh_order=1.0,
    )
    slabs = resolve_structured_slabs([sp_big, sp_small])
    by_name = {s.physical_name[0]: s for s in slabs}
    assert "hole" in by_name
    assert "bg" in by_name
    assert by_name["hole"].footprint.equals(MultiPolygon([small]))
    assert by_name["bg"].footprint.area == pytest.approx(
        big.area - small.area, rel=1e-9
    )


def test_resolve_xy_overlap_total_eats_low_priority(square_poly):
    from meshwell.polyprism import PolyPrism
    from meshwell.structured_polyprism import resolve_structured_slabs

    sp_loser = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[4],
        physical_name="loser",
        mesh_order=2.0,
    )
    sp_winner = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[3],
        physical_name="winner",
        mesh_order=1.0,
    )
    slabs = resolve_structured_slabs([sp_loser, sp_winner])
    assert len(slabs) == 1
    assert slabs[0].physical_name == ("winner",)


def test_resolve_partial_z_overlap_splits_low_priority():
    from shapely.geometry import Polygon

    from meshwell.polyprism import PolyPrism
    from meshwell.structured_polyprism import resolve_structured_slabs

    base = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    small = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])

    sp_lo = PolyPrism(
        polygons=base,
        buffers={0.0: 0.0, 3.0: 0.0},
        n_layers=[6],
        physical_name="lo",
        mesh_order=2.0,
    )
    sp_hi = PolyPrism(
        polygons=small,
        buffers={1.0: 0.0, 2.0: 0.0},
        n_layers=[2],
        physical_name="hi",
        mesh_order=1.0,
    )
    slabs = resolve_structured_slabs([sp_lo, sp_hi])
    by_name = sorted(slabs, key=lambda s: (s.physical_name, s.zlo))
    names_z = [(s.physical_name[0], s.zlo, s.zhi) for s in by_name]
    assert ("hi", 1.0, 2.0) in names_z
    assert ("lo", 0.0, 1.0) in names_z
    assert ("lo", 1.0, 2.0) in names_z
    assert ("lo", 2.0, 3.0) in names_z
    lo_slabs = [s for s in slabs if s.physical_name == ("lo",)]
    middle = next(s for s in lo_slabs if s.zlo == 1.0)
    full = next(s for s in lo_slabs if s.zlo == 0.0)
    assert middle.footprint.area == pytest.approx(base.area - small.area, rel=1e-9)
    assert full.footprint.area == pytest.approx(base.area, rel=1e-9)


def test_phantom_gmsh_extrudes_volume(square_poly):
    """The phantom's instanciate must produce one 3D dimtag."""
    from shapely.geometry import MultiPolygon

    from meshwell.model import ModelManager
    from meshwell.structured_polyprism import Slab, _StructuredPhantom

    slab = Slab(
        footprint=MultiPolygon([square_poly]),
        zlo=0.0,
        zhi=1.0,
        n_layers=4,
        recombine=False,
        physical_name=("film",),
        source_index=0,
    )
    phantom = _StructuredPhantom(slab)
    assert phantom.dimension == 3
    assert phantom.mesh_bool is False  # keep=False at top-dim
    assert phantom.physical_name == ("film",)

    # Drive instanciate() against a minimal gmsh model.
    mm = ModelManager(filename="t_phantom_gmsh")
    mm.ensure_initialized("t_phantom_gmsh")

    class _Mock:
        model_manager = mm

    try:
        dimtags = phantom.instanciate(_Mock())
        assert len(dimtags) == 1
        assert dimtags[0][0] == 3
    finally:
        mm.finalize()
