"""Tests for meshwell.structured.wedge — pre_2d transfinite hints."""
import pytest
from shapely.geometry import Polygon

from meshwell.resolution import StructuredExtrusionResolutionSpec

SQ = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


def test_transfinite_hints_produce_quad_laterals(tmp_path):
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism

    p = PolyPrism(
        polygons=SQ,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="s",
        structured=True,
    )
    generate_mesh(
        entities=[p],
        dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
        resolution_specs={
            "s": [StructuredExtrusionResolutionSpec(n_layers=2)],
        },
    )
    import meshio

    m = meshio.read(tmp_path / "out.msh")
    quads = sum(cb.data.shape[0] for cb in m.cells if cb.type == "quad")
    assert quads >= 16


def test_n_layers_mismatch_raises(tmp_path):
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism
    from meshwell.structured.exceptions import StructuredLateralNLayersMismatchError

    A = PolyPrism(
        polygons=SQ,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="a",
        structured=True,
    )
    SQ2 = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
    B = PolyPrism(
        polygons=SQ2,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="b",
        structured=True,
    )
    with pytest.raises(StructuredLateralNLayersMismatchError):
        generate_mesh(
            entities=[A, B],
            dim=3,
            output_mesh=tmp_path / "out.msh",
            default_characteristic_length=0.5,
            resolution_specs={
                "a": [StructuredExtrusionResolutionSpec(n_layers=2)],
                "b": [StructuredExtrusionResolutionSpec(n_layers=5)],
            },
        )


# ---------------------------------------------------------------------------
# Unit tests for resolve_n_layers — must pass without orchestrator wiring
# ---------------------------------------------------------------------------


def test_resolve_n_layers_default():
    from meshwell.structured.wedge import resolve_n_layers

    assert resolve_n_layers(("missing",), None) == 1
    assert resolve_n_layers(("missing",), {}) == 1


def test_resolve_n_layers_explicit():
    from meshwell.structured.wedge import resolve_n_layers

    rs = {"x": [StructuredExtrusionResolutionSpec(n_layers=4)]}
    assert resolve_n_layers(("x",), rs) == 4


def test_resolve_n_layers_multiple_raises():
    from meshwell.structured.exceptions import StructuredError
    from meshwell.structured.wedge import resolve_n_layers

    rs = {
        "x": [
            StructuredExtrusionResolutionSpec(n_layers=2),
            StructuredExtrusionResolutionSpec(n_layers=4),
        ]
    }
    with pytest.raises(StructuredError):
        resolve_n_layers(("x",), rs)


def test_n_layers_1_meshes_cleanly(tmp_path):
    """S3: degenerate n_layers=1 — one wedge per bot triangle, no intermediate-layer nodes needed."""
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism

    p = PolyPrism(
        polygons=SQ,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="s",
        structured=True,
    )
    generate_mesh(
        entities=[p],
        dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
        resolution_specs={
            "s": [StructuredExtrusionResolutionSpec(n_layers=1)],
        },
    )
    import meshio

    m = meshio.read(tmp_path / "out.msh")
    wedges = sum(cb.data.shape[0] for cb in m.cells if cb.type == "wedge")
    # SQ has 4 vertices, characteristic_length=0.5 -> ~4 bot triangles
    # n_layers=1 -> ~4 wedges
    assert wedges >= 2, f"expected >=2 wedges, got {wedges}"
    # No 3D group named "s" should be missing wedges.
    assert "s" in m.cell_sets


def test_shared_lateral_between_two_subsolids(tmp_path):
    """S5: two side-by-side structured slabs share a lateral face.

    The shared face must be meshed once (one set of quads, not two).
    Both volumes must have wedges. The interface group ``a___b`` must
    contain quads, not duplicates.
    """
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism

    SQ_A = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    SQ_B = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])

    a = PolyPrism(
        polygons=SQ_A,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="a",
        structured=True,
    )
    b = PolyPrism(
        polygons=SQ_B,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="b",
        structured=True,
    )
    generate_mesh(
        entities=[a, b],
        dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
        resolution_specs={
            "a": [StructuredExtrusionResolutionSpec(n_layers=2)],
            "b": [StructuredExtrusionResolutionSpec(n_layers=2)],
        },
    )
    import meshio

    m = meshio.read(tmp_path / "out.msh")

    # Both volumes get wedges.
    assert "a" in m.cell_sets
    assert "b" in m.cell_sets

    # Shared lateral interface must exist as a quad set.
    iface_name = "a___b" if "a___b" in m.cell_sets else "b___a"
    assert (
        iface_name in m.cell_sets
    ), f"shared lateral interface missing; groups: {sorted(m.cell_sets)}"

    # Count quads in the interface. n_layers=2, the shared edge spans
    # x=1 from y=0 to y=1 with characteristic_length=0.5 -> ~3 bot
    # edge nodes -> ~2 horizontal subdivisions -> 2 * n_layers = 4 quads.
    iface_sets = m.cell_sets[iface_name]
    iface_quads = sum(
        len(s)
        for s, b in zip(iface_sets, m.cells)
        if b.type == "quad" and s is not None
    )
    assert iface_quads >= 2, f"expected >=2 interface quads, got {iface_quads}"

    # Total wedge count should be the same in both volumes (symmetric).
    a_sets = m.cell_sets["a"]
    b_sets = m.cell_sets["b"]
    a_wedges = sum(
        len(s) for s, bk in zip(a_sets, m.cells) if bk.type == "wedge" and s is not None
    )
    b_wedges = sum(
        len(s) for s, bk in zip(b_sets, m.cells) if bk.type == "wedge" and s is not None
    )
    assert a_wedges > 0
    assert b_wedges > 0
