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
