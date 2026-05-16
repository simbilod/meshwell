"""API contract tests for ``PolyPrism(structured=True)``."""
from __future__ import annotations

import pytest
from shapely.geometry import Polygon


def _square() -> Polygon:
    return Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


def test_structured_flag_defaults_false():
    from meshwell.polyprism import PolyPrism

    p = PolyPrism(polygons=_square(), buffers={0.0: 0.0, 1.0: 0.0})
    assert p.structured is False


def test_structured_true_requires_resolution_spec():
    """structured=True without a StructuredExtrusionResolutionSpec raises."""
    from meshwell.polyprism import PolyPrism

    with pytest.raises(ValueError, match="StructuredExtrusionResolutionSpec"):
        PolyPrism(
            polygons=_square(),
            buffers={0.0: 0.0, 1.0: 0.0},
            structured=True,
        )


def test_structured_true_with_spec_succeeds():
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    spec = StructuredExtrusionResolutionSpec(n_layers=[3])
    p = PolyPrism(
        polygons=_square(),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[spec],
    )
    assert p.structured is True
    # Spec is preserved on the entity for the planner to retrieve.
    assert any(isinstance(r, StructuredExtrusionResolutionSpec) for r in p.resolutions)


def test_structured_true_rejects_tapered_buffers():
    """Non-uniform buffers raise StructuredBufferTaperError."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec
    from meshwell.structured.spec import StructuredBufferTaperError

    spec = StructuredExtrusionResolutionSpec(n_layers=[2])
    with pytest.raises(StructuredBufferTaperError, match="buffers"):
        PolyPrism(
            polygons=_square(),
            buffers={0.0: 0.0, 1.0: 0.1},
            structured=True,
            resolutions=[spec],
        )


def test_structured_true_n_layers_length_matches_z_intervals():
    """spec.n_layers length must equal len(buffers) - 1."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    spec = StructuredExtrusionResolutionSpec(n_layers=[3])
    with pytest.raises(ValueError, match="n_layers length"):
        PolyPrism(
            polygons=_square(),
            buffers={0.0: 0.0, 1.0: 0.0, 2.0: 0.0},
            structured=True,
            resolutions=[spec],
        )


def test_structured_true_rejects_multiple_specs():
    """structured=True with 2+ specs raises ValueError."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    spec_a = StructuredExtrusionResolutionSpec(n_layers=[3])
    spec_b = StructuredExtrusionResolutionSpec(n_layers=[3])
    with pytest.raises(ValueError, match="at most one"):
        PolyPrism(
            polygons=_square(),
            buffers={0.0: 0.0, 1.0: 0.0},
            structured=True,
            resolutions=[spec_a, spec_b],
        )


def test_spec_on_non_structured_entity_warns():
    """Attaching the spec without structured=True emits a warning and ignores it."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    spec = StructuredExtrusionResolutionSpec(n_layers=[3])
    with pytest.warns(UserWarning, match="structured=True"):
        p = PolyPrism(
            polygons=_square(),
            buffers={0.0: 0.0, 1.0: 0.0},
            structured=False,
            resolutions=[spec],
        )
    assert p.structured is False
