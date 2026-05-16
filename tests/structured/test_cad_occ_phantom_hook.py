"""Tests for the cad_occ hook that lets the structured pipeline.

1. Push extra OCP shapes into the global BOP.
2. Receive the BOPAlgo_Builder after Perform() for history extraction.
"""
from __future__ import annotations

from shapely.geometry import Polygon


def _square() -> Polygon:
    return Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


def test_cad_occ_accepts_extra_occ_shapes_kwarg():
    """Smoke test: passing extra_occ_shapes=[] is a no-op vs no kwarg."""
    from meshwell.cad_occ import cad_occ
    from meshwell.polyprism import PolyPrism

    p = PolyPrism(
        polygons=_square(),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="p",
    )
    a = cad_occ([p])
    b = cad_occ([p], extra_occ_shapes=[])
    assert len(a) == len(b)


def test_cad_occ_callback_invoked_with_builder():
    """When passed, callback receives the BOPAlgo_Builder post-Perform."""
    from meshwell.cad_occ import cad_occ
    from meshwell.polyprism import PolyPrism

    captured: list = []

    def cb(builder):
        captured.append(builder)

    p = PolyPrism(
        polygons=_square(),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="p",
    )
    cad_occ([p], cad_occ_callback=cb)
    assert len(captured) == 1
    # The captured object should expose Modified() — the BOP history API.
    assert hasattr(captured[0], "Modified")
    assert hasattr(captured[0], "Generated")


def test_cad_occ_extra_shapes_participate_in_fragmentation():
    """An extra phantom shape should get fragmented against the entities."""
    from meshwell.cad_occ import cad_occ
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan
    from meshwell.structured.phantom import build_phantom_shapes

    p = PolyPrism(
        polygons=_square(),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="p",
    )
    plan = build_plan([p])
    phantom_result = build_phantom_shapes(plan)
    extra = [s.solid for s in phantom_result.shapes]

    captured: list = []
    cad_occ(
        [p],
        extra_occ_shapes=extra,
        cad_occ_callback=lambda b: captured.append(b),
    )
    # Callback was invoked; builder exposes Modified() for any of the extras.
    assert len(captured) == 1
    for s in extra:
        # Modified() returns a TopTools_ListOfShape; we just check the call
        # doesn't raise and the input was tracked.
        modified = captured[0].Modified(s)
        assert modified is not None
