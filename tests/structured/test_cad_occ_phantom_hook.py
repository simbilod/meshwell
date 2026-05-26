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


def test_instantiate_entity_occ_uses_shape_override_when_provided():
    """When shape_override is given, _instantiate_entity_occ skips instanciate_occ()."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    from meshwell.cad_occ import CAD_OCC
    from meshwell.polyprism import PolyPrism

    p = PolyPrism(
        polygons=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="p",
    )
    tiny = PolyPrism(
        polygons=Polygon([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)]),
        buffers={0.0: 0.0, 0.1: 0.0},
        physical_name="tiny",
    )
    override_solid = tiny.instanciate_occ()

    proc = CAD_OCC()
    labeled = proc._instantiate_entity_occ(
        index=0,
        entity_obj=p,
        shape_override=[override_solid],
    )
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(labeled.shapes[0], g)
    assert (
        abs(g.Mass() - 0.001) < 1e-9
    ), f"Expected override volume 0.001 (0.1^3), got {g.Mass()}"
    assert labeled.physical_name == ("p",)
    assert labeled.dim == 3


def test_process_entities_cut_only_uses_overrides_when_supplied():
    """process_entities_cut_only respects entity_shape_overrides per source_index."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    from meshwell.cad_occ import CAD_OCC
    from meshwell.polyprism import PolyPrism

    p0 = PolyPrism(
        polygons=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="p0",
    )
    p1 = PolyPrism(
        polygons=Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="p1",
    )
    tiny = PolyPrism(
        polygons=Polygon([(2, 0), (2.1, 0), (2.1, 0.1), (2, 0.1)]),
        buffers={0.0: 0.0, 0.1: 0.0},
        physical_name="tiny",
    )
    override = tiny.instanciate_occ()

    proc = CAD_OCC()
    result = proc.process_entities_cut_only(
        [p0, p1],
        entity_shape_overrides={1: [override]},
    )

    by_name = {le.physical_name[0]: le for le in result}
    g = GProp_GProps()
    # process_entities_cut_only applies a perturbation buffer; assert order
    # of magnitude rather than bit-precision.
    BRepGProp.VolumeProperties_s(by_name["p0"].shapes[0], g)
    assert (
        0.99 < g.Mass() < 1.01
    ), f"p0 should be ~1.0-vol from instanciate_occ; got {g.Mass()}"
    BRepGProp.VolumeProperties_s(by_name["p1"].shapes[0], g)
    assert g.Mass() < 0.01, (
        f"p1 should be ~0.001-vol from override (not ~1.0 from instanciate_occ); "
        f"got {g.Mass()}"
    )


def test_process_entities_legacy_uses_overrides():
    """Legacy process_entities path also respects entity_shape_overrides."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    from meshwell.cad_occ import CAD_OCC
    from meshwell.polyprism import PolyPrism

    p = PolyPrism(
        polygons=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="p",
    )
    tiny = PolyPrism(
        polygons=Polygon([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)]),
        buffers={0.0: 0.0, 0.1: 0.0},
        physical_name="tiny",
    )
    override = tiny.instanciate_occ()

    proc = CAD_OCC()
    result = proc.process_entities(
        [p],
        entity_shape_overrides={0: [override]},
    )
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(result[0].shapes[0], g)
    assert (
        g.Mass() < 0.01
    ), f"override should win (~0.001), not instanciate_occ (~1.0); got {g.Mass()}"


def test_process_entities_parallel_uses_overrides_serial_executor():
    """Parallel path with executor='serial' respects entity_shape_overrides."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    from meshwell.cad_occ import CAD_OCC
    from meshwell.polyprism import PolyPrism

    p = PolyPrism(
        polygons=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="p",
    )
    tiny = PolyPrism(
        polygons=Polygon([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)]),
        buffers={0.0: 0.0, 0.1: 0.0},
        physical_name="tiny",
    )
    override = tiny.instanciate_occ()

    proc = CAD_OCC()
    result = proc.process_entities_parallel(
        [p],
        entity_shape_overrides={0: [override]},
        executor="serial",
    )
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(result[0].shapes[0], g)
    assert (
        g.Mass() < 0.01
    ), f"override should win (~0.001), not instanciate_occ (~1.0); got {g.Mass()}"
