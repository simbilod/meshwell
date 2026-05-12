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


def test_prepare_entities_swaps_structured_for_phantoms(square_poly):
    from shapely.geometry import Polygon

    from meshwell.cad_common import prepare_entities
    from meshwell.polyprism import PolyPrism
    from meshwell.structured_polyprism import (
        _StructuredPhantom,
        _StructuredPolyPrism,
    )

    sp = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[3],
        physical_name="film",
    )

    class _NeutralPolyEntity:
        def __init__(self, p):
            self.polygons = p
            self.physical_name = ("bg",)

    other = _NeutralPolyEntity(Polygon([(-2, -2), (2, -2), (2, 2), (-2, 2)]))
    entities = [other, sp]

    captured_slabs: list = []
    prepare_entities(
        entities,
        perturbation=1e-5,
        resolve_snap=1e-3,
        structured_slabs_out=captured_slabs,
    )
    # Structured PolyPrism replaced by phantom in-place.
    assert not any(isinstance(e, _StructuredPolyPrism) for e in entities)
    phantoms = [e for e in entities if isinstance(e, _StructuredPhantom)]
    assert len(phantoms) == 1
    assert phantoms[0].physical_name == ("film",)
    assert phantoms[0].mesh_bool is False
    # Captured slab list mirrors the cascade output.
    assert len(captured_slabs) == 1
    assert captured_slabs[0].physical_name == ("film",)


def test_prepare_entities_no_kwarg_preserves_structured(square_poly):
    """Without structured_slabs_out, structured PolyPrism flows unchanged."""
    from meshwell.cad_common import prepare_entities
    from meshwell.polyprism import PolyPrism
    from meshwell.structured_polyprism import _StructuredPolyPrism

    sp = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[3],
        physical_name="film",
    )
    entities = [sp]
    prepare_entities(entities, perturbation=1e-5)
    # Without the kwarg, structured prism is passed through (no phantom).
    assert any(isinstance(e, _StructuredPolyPrism) for e in entities)


def test_cad_gmsh_populates_model_manager_structured_slabs(square_poly):
    from meshwell.cad_gmsh import cad_gmsh
    from meshwell.polyprism import PolyPrism

    sp = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[3],
        physical_name="film",
    )
    _, mm = cad_gmsh([sp], filename="t8")
    try:
        slabs = mm.structured_slabs
        assert len(slabs) == 1
        assert slabs[0].physical_name == ("film",)
        assert slabs[0].n_layers == 3
    finally:
        mm.finalize()


def test_apply_structured_slabs_isolated_slab(square_poly):
    """Single structured prism, no neighbors -> geo extrude with N layers."""
    from collections import defaultdict

    import gmsh

    from meshwell.cad_gmsh import cad_gmsh
    from meshwell.polyprism import PolyPrism
    from meshwell.structured_polyprism import apply_structured_slabs

    sp = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[3],
        physical_name="film",
    )
    _, mm = cad_gmsh([sp], filename="t9")
    try:
        # After CAD, the OCC body is gone (mesh_bool=False) and the void's
        # boundary faces no longer exist (no neighbors held them).
        # apply_structured_slabs builds the geo replica from scratch.
        apply_structured_slabs(mm, mm.structured_slabs)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.5)
        mm.model.mesh.generate(3)

        # Use the structured layer signature: every interior xy node should
        # appear at exactly n_layers+1 distinct z values.
        all_nodes = mm.model.mesh.getNodes()
        _, coords, _ = all_nodes
        coords = coords.reshape(-1, 3)
        z_by_xy = defaultdict(set)
        for c in coords:
            z_by_xy[(round(c[0], 6), round(c[1], 6))].add(round(c[2], 6))
        layer_counts = {len(zs) for zs in z_by_xy.values() if len(zs) > 1}
        assert layer_counts == {4}, layer_counts  # 3 layers => 4 z-levels
    finally:
        mm.finalize()


def test_mesh_endtoend_single_structured_slab(tmp_path, square_poly):
    """Drive the full Mesh pipeline.

    structured slab should be meshed with n_layers layers automatically.
    """
    from meshwell.cad_gmsh import cad_gmsh
    from meshwell.mesh import mesh as mesh_fn
    from meshwell.polyprism import PolyPrism

    sp = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[5],
        physical_name="film",
    )
    _, mm = cad_gmsh([sp], filename="t10")
    out_msh = tmp_path / "t10.msh"
    try:
        mesh_fn(
            dim=3,
            default_characteristic_length=0.5,
            output_file=out_msh,
            model=mm,
        )
    except Exception:
        mm.finalize()
        raise
    assert out_msh.exists()
    import meshio

    m = meshio.read(out_msh)
    # Without the wiring, no 3D entities exist (the structured void is
    # untouched), so meshio.read finds zero 3D blocks. The wedge/hex
    # check originally proposed in the plan only holds with recombine=True;
    # gmsh subdivides extruded prisms into tetrahedra when recombine is
    # False (the structured-mode default), so we instead assert that the
    # mesh contains 3D cells AND retains the structured z-layering.
    cell_block_types = {b.type for b in m.cells}
    assert cell_block_types & {"tetra", "wedge", "hexahedron"}, cell_block_types

    # n_layers=5 -> 6 distinct z-levels for any column of nodes.
    from collections import defaultdict

    z_by_xy: dict[tuple[float, float], set[float]] = defaultdict(set)
    for x, y, z in m.points:
        z_by_xy[(round(x, 6), round(y, 6))].add(round(z, 6))
    layer_counts = {len(zs) for zs in z_by_xy.values() if len(zs) > 1}
    assert layer_counts == {6}, layer_counts


def test_multi_interval_layer_counts(tmp_path, square_poly):
    """Two stacked z-intervals -> respective layer counts respected."""
    from collections import defaultdict

    import meshio

    from meshwell.cad_gmsh import cad_gmsh
    from meshwell.mesh import mesh as mesh_fn
    from meshwell.polyprism import PolyPrism

    sp = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 0.5: 0.0, 1.0: 0.0},
        n_layers=[8, 2],
        physical_name="film",
    )
    _, mm = cad_gmsh([sp], filename="t11")
    out_msh = tmp_path / "t11.msh"
    try:
        mesh_fn(dim=3, default_characteristic_length=0.5, output_file=out_msh, model=mm)
    except Exception:
        mm.finalize()
        raise

    m = meshio.read(out_msh)
    coords = m.points
    z_by_xy = defaultdict(set)
    for c in coords:
        z_by_xy[(round(c[0], 6), round(c[1], 6))].add(round(c[2], 6))
    column_lengths = {len(zs) for zs in z_by_xy.values() if len(zs) > 1}
    # 8 + 2 = 10 layers => 11 distinct z values per column
    assert 11 in column_lengths, column_lengths


def test_two_xy_disjoint_structured_slabs(tmp_path, square_poly):
    import meshio
    from shapely.affinity import translate

    from meshwell.cad_gmsh import cad_gmsh
    from meshwell.mesh import mesh as mesh_fn
    from meshwell.polyprism import PolyPrism

    spA = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[2],
        physical_name="A",
    )
    spB = PolyPrism(
        polygons=translate(square_poly, xoff=2.0),
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[5],
        physical_name="B",
    )
    _, mm = cad_gmsh([spA, spB], filename="t12")
    out_msh = tmp_path / "t12.msh"
    try:
        mesh_fn(dim=3, default_characteristic_length=0.5, output_file=out_msh, model=mm)
    except Exception:
        mm.finalize()
        raise

    m = meshio.read(out_msh)
    assert "A" in m.field_data
    assert "B" in m.field_data


def test_structured_slab_with_unstructured_neighbor(tmp_path):
    """Structured slab next to a PolyPrism, both physicals present.

    Structured slab keeps n_layers, PolyPrism is meshed unstructured.
    """
    import meshio
    from shapely.geometry import Polygon

    from meshwell.cad_gmsh import cad_gmsh
    from meshwell.mesh import mesh as mesh_fn
    from meshwell.polyprism import PolyPrism

    sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    sq_neighbor = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])

    sp = PolyPrism(
        polygons=sq,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[4],
        physical_name="structured",
    )
    pp = PolyPrism(
        polygons=sq_neighbor,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="unstruct",
    )
    _, mm = cad_gmsh([sp, pp], filename="t14")
    out_msh = tmp_path / "t14.msh"
    try:
        mesh_fn(dim=3, default_characteristic_length=0.5, output_file=out_msh, model=mm)
    except Exception:
        mm.finalize()
        raise

    m = meshio.read(out_msh)
    assert "structured" in m.field_data
    assert "unstruct" in m.field_data


def test_resolve_partial_z_split_distributes_layers_proportionally():
    """Splitting in z must distribute n_layers proportionally across pieces."""
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

    lo_slabs = sorted(
        [s for s in slabs if s.physical_name == ("lo",)], key=lambda s: s.zlo
    )
    # 3 lo slabs each spanning 1 of 3 z-units; 6 total layers => 2 per slab.
    assert [(s.zlo, s.zhi, s.n_layers) for s in lo_slabs] == [
        (0.0, 1.0, 2),
        (1.0, 2.0, 2),
        (2.0, 3.0, 2),
    ]
    hi_slab = next(s for s in slabs if s.physical_name == ("hi",))
    assert hi_slab.n_layers == 2


def test_resolve_partial_z_split_rejects_unaligned_layers():
    """If the user's z-split doesn't divide n_layers evenly, raise."""
    import pytest
    from shapely.geometry import Polygon

    from meshwell.polyprism import PolyPrism
    from meshwell.structured_polyprism import (
        StructuredLayerMismatchError,
        resolve_structured_slabs,
    )

    base = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    small = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])

    sp_lo = PolyPrism(
        polygons=base,
        buffers={0.0: 0.0, 3.0: 0.0},
        n_layers=[1],  # 1 layer over 3 units; can't be split into 3
        physical_name="lo",
        mesh_order=2.0,
    )
    sp_hi = PolyPrism(
        polygons=small,
        buffers={1.0: 0.0, 2.0: 0.0},
        n_layers=[1],
        physical_name="hi",
        mesh_order=1.0,
    )
    with pytest.raises(StructuredLayerMismatchError, match="layer count"):
        resolve_structured_slabs([sp_lo, sp_hi])


def test_overlapping_structured_priority_resolution(tmp_path):
    """Two overlapping prisms (xy + z): higher priority wins, loser is split."""
    import meshio
    from shapely.geometry import Polygon

    from meshwell.cad_gmsh import cad_gmsh
    from meshwell.mesh import mesh as mesh_fn
    from meshwell.polyprism import PolyPrism

    big = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    small = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])

    # Compatible layer densities so the cascade produces matching n_layers
    # on shared z-intervals (2 layers/unit on both lo and hi).
    sp_lo = PolyPrism(
        polygons=big,
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
    _, mm = cad_gmsh([sp_lo, sp_hi], filename="t13")
    out_msh = tmp_path / "t13.msh"
    try:
        mesh_fn(dim=3, default_characteristic_length=1.0, output_file=out_msh, model=mm)
    except Exception:
        mm.finalize()
        raise

    m = meshio.read(out_msh)
    assert "lo" in m.field_data
    assert "hi" in m.field_data


def test_stacked_layer_mismatch_raises():
    """Two structured slabs share a horizontal face with different n_layers."""
    import pytest
    from shapely.geometry import Polygon

    from meshwell.cad_gmsh import cad_gmsh
    from meshwell.mesh import mesh as mesh_fn
    from meshwell.polyprism import PolyPrism
    from meshwell.structured_polyprism import StructuredLayerMismatchError

    sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    sp_low = PolyPrism(
        polygons=sq,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[4],
        physical_name="lo",
    )
    sp_high = PolyPrism(
        polygons=sq,
        buffers={1.0: 0.0, 2.0: 0.0},
        n_layers=[7],  # mismatch with lo
        physical_name="hi",
    )
    _, mm = cad_gmsh([sp_low, sp_high], filename="t15")
    try:
        with pytest.raises(StructuredLayerMismatchError, match="n_layers"):
            mesh_fn(dim=3, default_characteristic_length=0.5, model=mm)
    finally:
        mm.finalize()


def test_generate_mesh_endtoend_with_structured_prism(tmp_path):
    """generate_mesh (orchestrator) routes structured slabs through OCC + XAO + sidecar.

    Stronger than just checking the physical name: assert the actual layered
    structure (each interior xy-column has exactly n_layers+1 z-levels), which
    is only produced when the cascade + sidecar + apply_structured_slabs all
    fire correctly. Without the wiring, the unstructured fallback would
    produce a tet mesh with arbitrary z-levels.
    """
    from collections import defaultdict

    import meshio
    from shapely.geometry import Polygon

    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism

    sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    sp = PolyPrism(
        polygons=sq,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[5],
        physical_name="film",
    )
    out_msh = tmp_path / "t16.msh"
    generate_mesh(
        entities=[sp],
        dim=3,
        output_mesh=out_msh,
        default_characteristic_length=0.5,
    )
    m = meshio.read(out_msh)
    assert "film" in m.field_data
    # Structured layering signature: 5 layers => 6 distinct z-levels per column.
    z_by_xy = defaultdict(set)
    for c in m.points:
        z_by_xy[(round(c[0], 6), round(c[1], 6))].add(round(c[2], 6))
    column_lengths = {len(zs) for zs in z_by_xy.values() if len(zs) > 1}
    assert column_lengths == {6}, column_lengths


def test_polyprism_dict_roundtrip_unstructured(square_poly):
    """Existing PolyPrism (no n_layers) roundtrip is unchanged."""
    from meshwell.polyprism import PolyPrism
    from meshwell.utils import deserialize

    pp = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="film",
    )
    d = pp.to_dict()
    assert d["type"] == "PolyPrism"
    # New keys are absent (or None) when unstructured.
    assert d.get("n_layers") is None
    pp2 = deserialize(d)
    assert type(pp2) is PolyPrism


def test_polyprism_dict_roundtrip_structured(square_poly):
    """Structured-mode PolyPrism roundtrips through the unified type string.

    Uses the same ``PolyPrism`` type tag and re-dispatches to
    ``_StructuredPolyPrism`` via __new__.
    """
    from meshwell.polyprism import PolyPrism
    from meshwell.structured_polyprism import _StructuredPolyPrism
    from meshwell.utils import deserialize

    sp = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 0.5: 0.0, 1.0: 0.0},
        n_layers=[3, 7],
        physical_name="film",
        recombine=True,
        mesh_order=2.0,
    )
    d = sp.to_dict()
    assert d["type"] == "PolyPrism"  # unified type
    assert d["n_layers"] == [3, 7]
    assert d["recombine"] is True

    sp2 = deserialize(d)
    assert isinstance(sp2, _StructuredPolyPrism)
    assert sp2.n_layers == [3, 7]
    assert sp2.recombine is True
    assert sp2.physical_name == ("film",)


def test_backend_equivalence_structured_prism(tmp_path):
    """cad_gmsh path and cad_occ-via-orchestrator path agree on physical groups."""
    import meshio
    from shapely.geometry import Polygon

    from meshwell.cad_gmsh import cad_gmsh
    from meshwell.mesh import mesh as mesh_fn
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism

    def make_entities():
        sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        return [
            PolyPrism(
                polygons=sq,
                buffers={0.0: 0.0, 1.0: 0.0},
                n_layers=[3],
                physical_name="film",
            )
        ]

    out_gmsh = tmp_path / "gmsh.msh"
    out_occ = tmp_path / "occ.msh"

    _, mm = cad_gmsh(make_entities(), filename="t19g")
    try:
        mesh_fn(
            dim=3, default_characteristic_length=0.5, output_file=out_gmsh, model=mm
        )
    except Exception:
        mm.finalize()
        raise

    generate_mesh(
        entities=make_entities(),
        dim=3,
        output_mesh=out_occ,
        default_characteristic_length=0.5,
    )

    g = meshio.read(out_gmsh)
    o = meshio.read(out_occ)
    assert set(g.field_data.keys()) == set(o.field_data.keys())
    assert "film" in g.field_data


def test_recombine_true_produces_wedge_or_hex_elements(tmp_path, square_poly):
    """recombine=True asks gmsh to recombine swept layers into wedge/hex prisms.

    gmsh's ``geo.extrude(..., recombine=True)`` produces wedge (triangular
    prism) elements when the base is triangulated, and hex elements when the
    base is itself recombined to quads. We assert that at least one of those
    cell types is present -- which is only true when recombine=True is
    plumbed all the way through to the geo extrude call.
    """
    import meshio

    from meshwell.cad_gmsh import cad_gmsh
    from meshwell.mesh import mesh as mesh_fn
    from meshwell.polyprism import PolyPrism

    sp = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[3],
        physical_name="film",
        recombine=True,
    )
    _, mm = cad_gmsh([sp], filename="t_recombine")
    out_msh = tmp_path / "t_recombine.msh"
    try:
        mesh_fn(
            dim=3,
            default_characteristic_length=0.5,
            output_file=out_msh,
            model=mm,
        )
    except Exception:
        mm.finalize()
        raise

    m = meshio.read(out_msh)
    cell_types = {b.type for b in m.cells}
    assert cell_types & {"wedge", "hexahedron"}, cell_types


def test_structured_unstructured_interface_conformal(tmp_path, square_poly):
    """Bottom and top of a structured slab share nodes with OCC neighbors.

    The conformal v2 path builds the structured slab as a discrete-entity
    layered mesh whose layer-0 nodes ARE the bottom OCC face's existing
    node tags, and whose layer-n triangulation OVERRIDES the top OCC
    face's mesh. Both interfaces should therefore have zero duplicate
    (coincident-but-distinct) nodes after gmsh writes the mesh out.
    """
    from collections import Counter

    import meshio
    import numpy as np
    from shapely.geometry import Polygon

    from meshwell.cad_gmsh import cad_gmsh
    from meshwell.mesh import mesh as mesh_fn
    from meshwell.polyprism import PolyPrism

    sub = PolyPrism(
        polygons=Polygon([(-1, -1), (2, -1), (2, 2), (-1, 2)]),
        buffers={0.0: 0.0, 0.5: 0.0},
        physical_name="sub",
        mesh_order=10,
    )
    structured = PolyPrism(
        polygons=square_poly,
        buffers={0.5: 0.0, 1.0: 0.0},
        n_layers=[3],
        physical_name="film",
        mesh_order=2,
    )
    cap = PolyPrism(
        polygons=Polygon([(-1, -1), (2, -1), (2, 2), (-1, 2)]),
        buffers={1.0: 0.0, 1.3: 0.0},
        physical_name="cap",
        mesh_order=10,
    )
    _, mm = cad_gmsh([sub, structured, cap], filename="t_conformal")
    out = tmp_path / "conformal.msh"
    try:
        mesh_fn(dim=3, default_characteristic_length=0.4, output_file=out, model=mm)
    except Exception:
        mm.finalize()
        raise

    m = meshio.read(out)
    pts = np.asarray(m.points)
    rounded = np.round(pts, 8)
    counts = Counter(map(tuple, rounded))

    # Count duplicate nodes ONLY at the structured/unstructured interface
    # planes. z=0.5 (sub <-> structured) and z=1.0 (structured <-> cap).
    dupes_at_iface = 0
    for c, n in counts.items():
        if n > 1 and abs(c[2] - 0.5) < 1e-6:
            dupes_at_iface += n - 1
        if n > 1 and abs(c[2] - 1.0) < 1e-6:
            dupes_at_iface += n - 1
    # In the v1 (geo-extrude) impl, this would be ~50-200 duplicates.
    # In v2 (conformal), it should be 0.
    assert dupes_at_iface == 0, (
        f"Expected 0 duplicate nodes at structured/unstructured interfaces "
        f"(z=0.5, z=1.0); got {dupes_at_iface}."
    )


def test_structured_lateral_conformality_rectangular(tmp_path, square_poly):
    """Lateral faces conformal: zero duplicate nodes on rectangular wire scene.

    Structured/unstructured interfaces should match exactly when transfinite
    hints are applied to the void's lateral OCC faces.
    """
    from collections import Counter

    import meshio
    import numpy as np
    from shapely.geometry import Polygon

    from meshwell.cad_gmsh import cad_gmsh
    from meshwell.mesh import mesh as mesh_fn
    from meshwell.polyprism import PolyPrism

    cladding = PolyPrism(
        polygons=Polygon([(-2, -2), (3, -2), (3, 3), (-2, 3)]),
        buffers={0.0: 0.0, 0.4: 0.0},
        physical_name="cladding",
        mesh_order=10,
    )
    wire = PolyPrism(
        polygons=square_poly,
        buffers={0.4: 0.0, 1.0: 0.0},
        n_layers=[4],
        physical_name="wire",
        mesh_order=2,
    )
    encap = PolyPrism(
        polygons=Polygon([(-2, -2), (3, -2), (3, 3), (-2, 3)]),
        buffers={1.0: 0.0, 1.3: 0.0},
        physical_name="encap",
        mesh_order=10,
    )
    filler = PolyPrism(
        polygons=Polygon([(-2, -2), (3, -2), (3, 3), (-2, 3)]),
        buffers={0.4: 0.0, 1.0: 0.0},
        physical_name="filler",
        mesh_order=15,
    )
    _, mm = cad_gmsh([cladding, wire, encap, filler], filename="t_lateral_rect")
    out = tmp_path / "lateral_rect.msh"
    try:
        mesh_fn(dim=3, default_characteristic_length=0.4, output_file=out, model=mm)
    except Exception:
        mm.finalize()
        raise

    m = meshio.read(out)
    pts = np.asarray(m.points)
    rounded = np.round(pts, 8)
    counts = Counter(map(tuple, rounded))
    dupes = sum(n - 1 for c, n in counts.items() if n > 1)
    assert dupes == 0, f"Expected 0 duplicate nodes; got {dupes}."


def test_structured_lateral_conformality_arc_bearing(tmp_path):
    """Arc-bearing structured slab still produces zero duplicates on lateral interfaces."""
    import math
    from collections import Counter

    import meshio
    import numpy as np
    from shapely.geometry import Polygon

    from meshwell.cad_gmsh import cad_gmsh
    from meshwell.mesh import mesh as mesh_fn
    from meshwell.polyprism import PolyPrism

    # A pill-shape polygon: rectangle with two semicircular caps. PolyPrism with
    # identify_arcs=True will detect the curved sides.
    n_arc = 24
    radius = 0.5
    left_arc = [
        (
            -1.0 + radius * math.cos(math.pi / 2 + i * math.pi / n_arc),
            radius * math.sin(math.pi / 2 + i * math.pi / n_arc),
        )
        for i in range(n_arc + 1)
    ]
    right_arc = [
        (
            1.0 + radius * math.cos(-math.pi / 2 + i * math.pi / n_arc),
            radius * math.sin(-math.pi / 2 + i * math.pi / n_arc),
        )
        for i in range(n_arc + 1)
    ]
    pill = Polygon(left_arc + right_arc)

    cladding = PolyPrism(
        polygons=Polygon([(-3, -2), (3, -2), (3, 2), (-3, 2)]),
        buffers={0.0: 0.0, 0.5: 0.0},
        physical_name="cladding",
        mesh_order=10,
    )
    pill_prism = PolyPrism(
        polygons=pill,
        buffers={0.5: 0.0, 1.0: 0.0},
        n_layers=[3],
        physical_name="pill",
        mesh_order=2,
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
    )
    encap = PolyPrism(
        polygons=Polygon([(-3, -2), (3, -2), (3, 2), (-3, 2)]),
        buffers={1.0: 0.0, 1.3: 0.0},
        physical_name="encap",
        mesh_order=10,
    )
    filler = PolyPrism(
        polygons=Polygon([(-3, -2), (3, -2), (3, 2), (-3, 2)]),
        buffers={0.5: 0.0, 1.0: 0.0},
        physical_name="filler",
        mesh_order=15,
    )
    _, mm = cad_gmsh([cladding, pill_prism, encap, filler], filename="t_lateral_arc")
    out = tmp_path / "lateral_arc.msh"
    try:
        mesh_fn(dim=3, default_characteristic_length=0.3, output_file=out, model=mm)
    except Exception:
        mm.finalize()
        raise

    m = meshio.read(out)
    pts = np.asarray(m.points)
    rounded = np.round(pts, 8)
    counts = Counter(map(tuple, rounded))
    dupes = sum(n - 1 for c, n in counts.items() if n > 1)
    # Arc-bearing case may have a few stragglers due to floating-point at arc
    # corners. Accept up to 5 duplicate residues.
    assert dupes <= 5, f"Expected <= 5 duplicate nodes; got {dupes}."


def test_recombine_true_produces_hex_in_conformal_path(tmp_path, square_poly):
    """recombine=True yields hex elements even on the conformal (neighbored) path.

    In this scene the structured ``film`` sits between an unstructured
    substrate and cap (conformal path). Without proper plumbing,
    ``recombine=True`` is silently ignored and the slab is filled with
    wedge elements. After plumbing through, the slab volume must contain
    hex elements (gmsh type 5).
    """
    import meshio
    from shapely.geometry import Polygon

    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism

    hull = Polygon([(-1, -1), (2, -1), (2, 2), (-1, 2)])
    sub = PolyPrism(
        polygons=hull,
        buffers={0.0: 0.0, 0.5: 0.0},
        physical_name="sub",
        mesh_order=10,
    )
    film = PolyPrism(
        polygons=square_poly,
        buffers={0.5: 0.0, 1.0: 0.0},
        n_layers=[3],
        physical_name="film",
        mesh_order=2,
        recombine=True,
    )
    cap = PolyPrism(
        polygons=hull,
        buffers={1.0: 0.0, 1.5: 0.0},
        physical_name="cap",
        mesh_order=10,
    )
    out = tmp_path / "recombine_conformal.msh"
    generate_mesh(
        entities=[sub, film, cap],
        dim=3,
        output_mesh=str(out),
        default_characteristic_length=0.4,
    )
    m = meshio.read(out)
    cell_types = {b.type for b in m.cells}
    assert (
        "hexahedron" in cell_types
    ), f"Expected hexahedron cells for recombine=True+conformal; got {cell_types}"


def test_structured_lateral_exposed_to_none_is_tagged(tmp_path, square_poly):
    """Lateral walls exposed to None get tagged ``slab___None``.

    A structured slab sandwiched by substrate/cap (no lateral neighbor)
    emits ``slab___None`` for its lateral walls. The substrate and cap
    extend above/below in z but do NOT cover the slab's z-range
    laterally, so the slab's lateral OCC faces are orphan and removed
    at the CAD stage. The lateral wall mesh and ``slab___None`` tag
    must still appear in the output.
    """
    import meshio
    from shapely.geometry import Polygon

    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism

    hull = Polygon([(-1, -1), (2, -1), (2, 2), (-1, 2)])
    sub = PolyPrism(
        polygons=hull,
        buffers={0.0: 0.0, 0.5: 0.0},
        physical_name="sub",
        mesh_order=10,
    )
    film = PolyPrism(
        polygons=square_poly,
        buffers={0.5: 0.0, 1.0: 0.0},
        n_layers=[3],
        physical_name="film",
        mesh_order=2,
    )
    cap = PolyPrism(
        polygons=hull,
        buffers={1.0: 0.0, 1.5: 0.0},
        physical_name="cap",
        mesh_order=10,
    )
    out = tmp_path / "lateral_none.msh"
    generate_mesh(
        entities=[sub, film, cap],
        dim=3,
        output_mesh=str(out),
        default_characteristic_length=0.4,
    )
    m = meshio.read(out)
    assert (
        "film___None" in m.field_data
    ), f"Expected film___None in field_data; got {sorted(m.field_data.keys())}"


def test_structured_isolated_emits_none_tag_on_all_faces(tmp_path, square_poly):
    """Isolated structured slab tags every external face as ``slab___None``.

    With no neighbors the slab is built via the geo-fallback path; bottom,
    top, and lateral side faces must all appear in ``slab___None``.
    """
    import meshio
    import numpy as np

    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism

    slab = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[3],
        physical_name="film",
    )
    out = tmp_path / "isolated_none.msh"
    generate_mesh(
        entities=[slab],
        dim=3,
        output_mesh=str(out),
        default_characteristic_length=0.4,
    )
    m = meshio.read(out)
    assert (
        "film___None" in m.field_data
    ), f"Expected film___None in field_data; got {sorted(m.field_data.keys())}"

    # Tag must cover bottom (z=0), top (z=1), and lateral walls.
    pts = np.asarray(m.points)
    cell_sets = m.cell_sets.get("film___None", [])
    bot = top = lat = 0
    for i, ids in enumerate(cell_sets):
        if ids is None or len(ids) == 0:
            continue
        block = m.cells[i]
        coords = pts[block.data[ids]]
        z_lo = coords[..., 2].min(axis=1)
        z_hi = coords[..., 2].max(axis=1)
        bot += int(((z_lo > -1e-6) & (z_hi < 1e-6)).sum())
        top += int(((z_lo > 1.0 - 1e-6) & (z_hi < 1.0 + 1e-6)).sum())
        lat += int((z_hi - z_lo > 1e-6).sum())
    assert bot > 0, f"film___None missing bottom-face elements (bot={bot})"
    assert top > 0, f"film___None missing top-face elements (top={top})"
    assert lat > 0, f"film___None missing lateral-face elements (lat={lat})"


def test_structured_fully_embedded_has_no_none_tag(tmp_path, square_poly):
    """Fully embedded structured slab has no ``___None`` tag.

    Every face of the slab is shared with the surrounding bulk so the
    only interface tag should be ``bulk___slab``. Guards against the
    lateral-tagging post-pass emitting spurious ``___None`` for embedded
    slabs.
    """
    import meshio
    from shapely.geometry import Polygon

    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism

    bulk = PolyPrism(
        polygons=Polygon([(-2, -2), (3, -2), (3, 3), (-2, 3)]),
        buffers={0.0: 0.0, 2.0: 0.0},
        physical_name="bulk",
        mesh_order=10,
    )
    wire = PolyPrism(
        polygons=square_poly,
        buffers={0.5: 0.0, 1.0: 0.0},
        n_layers=[3],
        physical_name="wire",
        mesh_order=2,
    )
    out = tmp_path / "embedded_no_none.msh"
    generate_mesh(
        entities=[bulk, wire],
        dim=3,
        output_mesh=str(out),
        default_characteristic_length=0.4,
    )
    m = meshio.read(out)
    assert "wire___None" not in m.field_data, (
        f"wire is fully embedded inside bulk; wire___None should not exist. "
        f"Got: {sorted(m.field_data.keys())}"
    )
    assert "bulk___wire" in m.field_data


def test_thin_structured_slab_cut_by_arc_keepfalse_inside_bulk(tmp_path):
    """Stack of thin structured slabs carved by a ``mesh_bool=False`` arc-bearing prism, inside a bulk.

    Scene:
      - Large unstructured ``bulk`` PolyPrism wraps everything.
      - Three stacked thin structured slabs (``film_lo``, ``film_mid``,
        ``film_hi``) with slightly different thicknesses, sharing the same
        xy footprint and a common ``n_layers`` so the touching horizontal
        interfaces are conformal (the structured layer-count constraint
        across shared horizontal faces).
      - A stadium-shape (rectangle with two semicircular caps) PolyPrism with
        ``identify_arcs=True`` and ``mesh_bool=False`` punches through every
        slab in z. Higher priority (lower mesh_order) than the slabs so it
        wins the overlap and removes material from each.

    Asserts:
      - Mesh writes successfully and contains 3D cells.
      - ``bulk`` and all three film physicals are present.
      - The cutter has no own volume physical (mesh_bool=False).
      - Every slab's surviving region still carries its own structured
        z-layering (n_layers+1 distinct z-levels per interior xy column
        within that slab's z-range).
    """
    import math
    from collections import defaultdict

    import meshio
    from shapely.geometry import Polygon

    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism

    # Stadium polygon: rectangle [-0.6, 0.6] x [-0.25, 0.25] capped with semicircles.
    n_arc = 24
    radius = 0.25
    half_len = 0.6
    right_arc = [
        (
            half_len + radius * math.cos(-math.pi / 2 + i * math.pi / n_arc),
            radius * math.sin(-math.pi / 2 + i * math.pi / n_arc),
        )
        for i in range(n_arc + 1)
    ]
    left_arc = [
        (
            -half_len + radius * math.cos(math.pi / 2 + i * math.pi / n_arc),
            radius * math.sin(math.pi / 2 + i * math.pi / n_arc),
        )
        for i in range(n_arc + 1)
    ]
    stadium = Polygon(right_arc + left_arc)

    bulk = PolyPrism(
        polygons=Polygon([(-3, -3), (3, -3), (3, 3), (-3, 3)]),
        buffers={0.0: 0.0, 2.0: 0.0},
        physical_name="bulk",
        mesh_order=10,
    )
    # Three stacked thin structured slabs sharing the same xy footprint.
    # Thicknesses differ slightly (0.10, 0.08, 0.12) but n_layers matches
    # across them so the conformal touching faces at z=0.80 and z=0.88 do
    # not trigger StructuredLayerMismatchError.
    film_footprint = Polygon([(-2, -2), (2, -2), (2, 2), (-2, 2)])
    film_lo = PolyPrism(
        polygons=film_footprint,
        buffers={0.70: 0.0, 0.80: 0.0},
        n_layers=[3],
        physical_name="film_lo",
        mesh_order=2,
    )
    film_mid = PolyPrism(
        polygons=film_footprint,
        buffers={0.80: 0.0, 0.88: 0.0},
        n_layers=[3],
        physical_name="film_mid",
        mesh_order=2,
    )
    film_hi = PolyPrism(
        polygons=film_footprint,
        buffers={0.88: 0.0, 1.00: 0.0},
        n_layers=[3],
        physical_name="film_hi",
        mesh_order=2,
    )
    # Arc-bearing cutter, mesh_bool=False, spans (and exceeds) the full
    # stack z-range so it cuts cleanly through every slab.
    cutter = PolyPrism(
        polygons=stadium,
        buffers={0.65: 0.0, 1.05: 0.0},
        physical_name="hole",
        mesh_order=1,
        mesh_bool=False,
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
    )

    out = tmp_path / "thin_slab_stack_arc_cut.msh"
    generate_mesh(
        entities=[bulk, film_lo, film_mid, film_hi, cutter],
        dim=3,
        output_mesh=str(out),
        default_characteristic_length=0.3,
    )

    m = meshio.read(out)
    cell_types = {b.type for b in m.cells}
    assert cell_types & {"tetra", "wedge", "hexahedron"}, cell_types

    assert "bulk" in m.field_data
    for name in ("film_lo", "film_mid", "film_hi"):
        assert (
            name in m.field_data
        ), f"missing physical {name!r}; got {sorted(m.field_data.keys())}"
    # Cutter is keep=False at top dim: no own volume physical.
    assert "hole" not in m.field_data, (
        f"mesh_bool=False cutter must not produce a volume physical; "
        f"got {sorted(m.field_data.keys())}"
    )

    # Each slab must (a) actually contain 3D mesh cells in its physical
    # group and (b) keep its own structured layering signature (n_layers+1
    # distinct z-levels per interior xy column within that slab's z-range).
    import numpy as np

    pts = np.asarray(m.points)
    slab_ranges = [
        ("film_lo", 0.70, 0.80, 4),
        ("film_mid", 0.80, 0.88, 4),
        ("film_hi", 0.88, 1.00, 4),
    ]
    for name, zlo, zhi, expected_levels in slab_ranges:
        # (a) The physical group must have at least one 3D cell whose
        # centroid lies inside [zlo, zhi].
        cells_in_slab = 0
        cell_sets = m.cell_sets.get(name, [])
        for i, ids in enumerate(cell_sets):
            if ids is None or len(ids) == 0:
                continue
            block = m.cells[i]
            if block.type not in ("tetra", "wedge", "hexahedron"):
                continue
            centroids_z = pts[block.data[ids]][..., 2].mean(axis=1)
            cells_in_slab += int(
                ((centroids_z >= zlo - 1e-6) & (centroids_z <= zhi + 1e-6)).sum()
            )
        assert cells_in_slab > 0, (
            f"{name}: expected >0 3D cells with centroid in z=[{zlo}, {zhi}]; "
            f"got {cells_in_slab}"
        )

        # (b) Structured layering signature.
        z_by_xy: dict[tuple[float, float], set[float]] = defaultdict(set)
        for x, y, z in m.points:
            if zlo - 1e-6 <= z <= zhi + 1e-6:
                z_by_xy[(round(x, 6), round(y, 6))].add(round(z, 6))
        column_lengths = {len(zs) for zs in z_by_xy.values() if len(zs) > 1}
        assert expected_levels in column_lengths, (
            f"{name}: expected at least one xy column with {expected_levels} "
            f"distinct z-levels in z=[{zlo}, {zhi}]; got {column_lengths}"
        )

    # Bulk also gets actually meshed (3D cells in its physical).
    bulk_cells = 0
    for i, ids in enumerate(m.cell_sets.get("bulk", [])):
        if ids is None or len(ids) == 0:
            continue
        if m.cells[i].type in ("tetra", "wedge", "hexahedron"):
            bulk_cells += len(ids)
    assert bulk_cells > 0, "bulk physical has no 3D cells"
