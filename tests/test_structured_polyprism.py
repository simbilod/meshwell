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
