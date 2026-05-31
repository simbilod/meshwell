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
