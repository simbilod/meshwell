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


def test_cad_occ_top_level_accepts_entity_shape_overrides():
    """cad_occ() entry point forwards entity_shape_overrides to the processor."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    from meshwell.cad_occ import cad_occ
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

    result = cad_occ([p], entity_shape_overrides={0: [override]})
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(result[0].shapes[0], g)
    assert (
        g.Mass() < 0.01
    ), f"override should win (~0.001), not instanciate_occ (~1.0); got {g.Mass()}"


def test_structured_entity_shapes_are_phantom_solids_after_cad_occ():
    """After cad_occ with entity_shape_overrides, each entity's shapes IsSame the phantom solids.

    For disjoint structured PolyPrisms, BOP fragmentation doesn't modify
    any input shape, so the entity's post-cad_occ shapes must be IsSame
    with the phantom solids that were passed in as overrides.
    """
    from meshwell.cad_occ import cad_occ
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec
    from meshwell.structured.phantom import (
        _group_phantom_solids_by_entity,
        build_phantom_shapes,
    )
    from meshwell.structured.plan import build_plan

    a = PolyPrism(
        polygons=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="A",
        mesh_order=1.0,
    )
    b = PolyPrism(
        polygons=Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="B",
        mesh_order=2.0,
    )
    entities = [a, b]
    plan = build_plan(entities)
    phantom_result = build_phantom_shapes(plan)
    overrides = _group_phantom_solids_by_entity(plan, phantom_result)

    occ_entities = cad_occ(entities, entity_shape_overrides=overrides)

    by_name = {le.physical_name[0]: le for le in occ_entities}
    expected_a_solids = overrides[0]
    expected_b_solids = overrides[1]
    assert len(by_name["A"].shapes) == len(expected_a_solids)
    for s, exp in zip(by_name["A"].shapes, expected_a_solids):
        assert s.IsSame(exp), "A's entity solid must IsSame the phantom solid"
    assert len(by_name["B"].shapes) == len(expected_b_solids)
    for s, exp in zip(by_name["B"].shapes, expected_b_solids):
        assert s.IsSame(exp), "B's entity solid must IsSame the phantom solid"


def test_no_sliver_solids_for_ring_quarter_cut():
    """Ring-segment quarter-cut scene must produce no sliver sub-solids per entity.

    Mirrors the failing stress-pattern scene: three half-annuli rotated 90
    degrees per layer that cut each other into quarter-rings. With
    entity_shape_overrides routing phantom solids as entity shapes, BOP
    cannot create slivers from parallel-construction mismatch (there is no
    parallel construction).
    """
    import math

    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    from meshwell.cad_occ import cad_occ
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec
    from meshwell.structured.phantom import (
        _group_phantom_solids_by_entity,
        build_phantom_shapes,
    )
    from meshwell.structured.plan import build_plan

    def _ring_segment(cx, cy, ri, ro, t0, t1, n=24):
        step = 2 * math.pi / n
        eps = 1e-9
        interior = [
            k * step
            for k in range(math.ceil(t0 / step), math.floor(t1 / step) + 1)
            if t0 + eps < k * step < t1 - eps
        ]
        angles = [t0, *interior, t1]
        outer = [(cx + ro * math.cos(a), cy + ro * math.sin(a)) for a in angles]
        inner = [
            (cx + ri * math.cos(a), cy + ri * math.sin(a)) for a in reversed(angles)
        ]
        return Polygon(outer + inner)

    def _ring(poly, zlo, zhi, name):
        return PolyPrism(
            polygons=poly,
            buffers={zlo: 0.0, zhi: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
            identify_arcs=True,
            min_arc_points=4,
            arc_tolerance=1e-3,
            physical_name=name,
        )

    L1 = _ring(_ring_segment(0, 0, 0.5, 1.0, 0.0, math.pi), 0.0, 1.0, "L1")
    L2 = _ring(
        _ring_segment(0, 0, 0.5, 1.0, math.pi / 2, 3 * math.pi / 2),
        1.0,
        2.0,
        "L2",
    )
    L3 = _ring(_ring_segment(0, 0, 0.5, 1.0, math.pi, 2 * math.pi), 2.0, 3.0, "L3")
    entities = [L1, L2, L3]
    plan = build_plan(entities)
    phantom_result = build_phantom_shapes(plan)
    overrides = _group_phantom_solids_by_entity(plan, phantom_result)
    occ_entities = cad_occ(entities, entity_shape_overrides=overrides)

    by_name = {le.physical_name[0]: le for le in occ_entities}
    g = GProp_GProps()
    quarter_ring_vol = (math.pi / 4) * (1.0**2 - 0.5**2) * 1.0
    min_vol = quarter_ring_vol * 0.01
    for name in ("L1", "L2", "L3"):
        ent = by_name[name]
        assert len(ent.shapes) == 2, (
            f"{name} has {len(ent.shapes)} shapes; expected 2 (one per "
            f"face_partition piece). Slivers from parallel-construction bug."
        )
        for i, s in enumerate(ent.shapes):
            BRepGProp.VolumeProperties_s(s, g)
            assert g.Mass() > min_vol, (
                f"{name}.shapes[{i}] vol={g.Mass():.4e} < {min_vol:.4e}; "
                f"likely a sliver artifact"
            )


def test_phantom_map_laterals_all_in_xao_compound_for_arc_cut_scene():
    """All PhantomMap output_laterals must IsSame match in the XAO compound.

    Uses the three-layer rotated half-annulus scene. After cad_occ +
    extract_phantom_map + _build_xao_compound, every lateral face the
    PhantomMap references must be findable by IsSame in the compound's
    face map. This is the invariant whose violation raised "PhantomMap
    lateral ... has no IsSame() match" before the fix.
    """
    import math

    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape

    from meshwell.cad_occ import cad_occ
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec
    from meshwell.structured.builder import _build_xao_compound
    from meshwell.structured.phantom import (
        _group_phantom_solids_by_entity,
        build_phantom_shapes,
        extract_phantom_map,
    )
    from meshwell.structured.plan import build_plan

    def _ring_segment(cx, cy, ri, ro, t0, t1, n=24):
        step = 2 * math.pi / n
        eps = 1e-9
        interior = [
            k * step
            for k in range(math.ceil(t0 / step), math.floor(t1 / step) + 1)
            if t0 + eps < k * step < t1 - eps
        ]
        angles = [t0, *interior, t1]
        outer = [(cx + ro * math.cos(a), cy + ro * math.sin(a)) for a in angles]
        inner = [
            (cx + ri * math.cos(a), cy + ri * math.sin(a)) for a in reversed(angles)
        ]
        return Polygon(outer + inner)

    def _ring(poly, zlo, zhi, name):
        return PolyPrism(
            polygons=poly,
            buffers={zlo: 0.0, zhi: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
            identify_arcs=True,
            min_arc_points=4,
            arc_tolerance=1e-3,
            physical_name=name,
        )

    L1 = _ring(_ring_segment(0, 0, 0.5, 1.0, 0.0, math.pi), 0.0, 1.0, "L1")
    L2 = _ring(
        _ring_segment(0, 0, 0.5, 1.0, math.pi / 2, 3 * math.pi / 2),
        1.0,
        2.0,
        "L2",
    )
    L3 = _ring(_ring_segment(0, 0, 0.5, 1.0, math.pi, 2 * math.pi), 2.0, 3.0, "L3")
    entities = [L1, L2, L3]
    plan = build_plan(entities)
    phantom_result = build_phantom_shapes(plan)
    overrides = _group_phantom_solids_by_entity(plan, phantom_result)

    captured: list = []
    occ_entities = cad_occ(
        entities,
        entity_shape_overrides=overrides,
        cad_occ_callback=lambda b: captured.append(b),
    )
    assert len(captured) == 1
    pmap = extract_phantom_map(phantom_result, captured[0])

    compound = _build_xao_compound(occ_entities)
    fmap = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(compound, TopAbs_FACE, fmap)

    missing = []
    for lat_key, faces in pmap.output_laterals.items():
        for i, f in enumerate(faces):
            if fmap.FindIndex(f) == 0:
                missing.append((lat_key, i))

    assert not missing, (
        f"{len(missing)} PhantomMap lateral face(s) have no IsSame match "
        f"in the XAO compound: {missing[:5]}..."
    )


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
