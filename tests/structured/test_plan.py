"""Tests for meshwell.structured.plan."""
from __future__ import annotations

from shapely.geometry import Polygon


def _square(x=0, y=0, w=1, h=1) -> Polygon:
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def _structured(polygon, buffers, n_layers, name, mesh_order=1.0):
    """Test helper: build a structured PolyPrism with a spec attached."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec

    return PolyPrism(
        polygons=polygon,
        buffers=buffers,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=n_layers)],
        physical_name=name,
        mesh_order=mesh_order,
    )


def test_gather_filters_structured_entities():
    from meshwell.polyprism import PolyPrism
    from meshwell.structured.plan import gather_structured_entities

    s = _structured(_square(), {0.0: 0.0, 1.0: 0.0}, [3], "s")
    u = PolyPrism(
        polygons=_square(2, 2),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="u",
    )

    pairs = gather_structured_entities([u, s])
    assert len(pairs) == 1
    entity, spec, source_idx = pairs[0]
    assert entity is s
    assert spec.n_layers == [3]
    assert source_idx == 1  # s was at index 1 in the input list


def test_gather_returns_empty_when_no_structured_entities():
    from meshwell.polyprism import PolyPrism
    from meshwell.structured.plan import gather_structured_entities

    u = PolyPrism(
        polygons=_square(),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="u",
    )
    assert gather_structured_entities([u]) == []


def test_gather_preserves_source_indices_across_mixed_list():
    from meshwell.polyprism import PolyPrism
    from meshwell.structured.plan import gather_structured_entities

    u1 = PolyPrism(polygons=_square(), buffers={0.0: 0.0, 1.0: 0.0}, physical_name="u1")
    s1 = _structured(_square(2, 0), {0.0: 0.0, 1.0: 0.0}, [3], "s1")
    u2 = PolyPrism(
        polygons=_square(4, 0), buffers={0.0: 0.0, 1.0: 0.0}, physical_name="u2"
    )
    s2 = _structured(_square(6, 0), {0.0: 0.0, 2.0: 0.0}, [4], "s2")

    pairs = gather_structured_entities([u1, s1, u2, s2])
    assert [p[2] for p in pairs] == [1, 3]
    assert [p[0].physical_name for p in pairs] == [("s1",), ("s2",)]
