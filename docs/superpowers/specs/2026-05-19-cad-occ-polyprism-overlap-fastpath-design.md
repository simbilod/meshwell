# cad_occ: shapely + z fast-path for polyprism-vs-polyprism overlap test

**Status:** approved (2026-05-19), pending implementation
**Author:** simbilod (with Claude)
**Related:**
- [`meshwell/cad_occ.py`](../../../meshwell/cad_occ.py) — call site of `_shapes_actually_overlap`
- [`meshwell/polyprism.py`](../../../meshwell/polyprism.py) — source of the metadata
- Prior spec: [`2026-05-19-cad-occ-batched-compound-cut-design.md`](2026-05-19-cad-occ-batched-compound-cut-design.md) — promoted `_shapes_actually_overlap` to the critical path

## Summary

Add a polyprism-aware fast-path to the cut-cascade's overlap gate. When both entities expose 2D footprint + z-range metadata, decide the overlap question using shapely `dwithin` + a 1D interval test instead of OCC `BRepExtrema_DistShapeShape`. For axis-aligned extrusions (`PolyPrism` with `extrude=True`) the polygon + z-interval test is mathematically equivalent to the OCC distance computation and the OCC call is skipped entirely. For tapered polyprisms the test is conservative (fast-reject only) and the OCC call is preserved when the cheap test cannot prove the pair is disjoint.

## Motivation

The batched-compound cut change (commit `6d4e02b`) eliminated the per-tool BOP loop and made `_shapes_actually_overlap` part of the dominant inner-loop cost — it runs once per `(current entity sub-shape, candidate tool sub-shape)` pair and uses `BRepExtrema_DistShapeShape`, an O(face-count × face-count) OCC computation.

Most production scenes are built almost exclusively from `PolyPrism` entities. For those entities the geometry is fully described by a 2D shapely polygon plus two scalar z bounds — the OCC distance call is far more work than needed. A shapely intersection + scalar interval check is orders of magnitude cheaper, and for axis-aligned extrusions it is provably equivalent.

## Design

### New `OCCLabeledEntity` fields

Three optional fields are added; all default to `None`/`False` so existing code paths are unaffected:

```python
@dataclass
class OCCLabeledEntity:
    shapes: list[TopoDS_Shape]
    physical_name: tuple[str, ...]
    index: int
    keep: bool
    dim: int
    mesh_order: float | None = None
    overlap_footprint: shapely.Geometry | None = None  # 2D xy envelope
    overlap_zrange: tuple[float, float] | None = None  # (zmin, zmax)
    overlap_exact: bool = False  # True iff (footprint, zrange) is geometrically exact
```

`overlap_exact` distinguishes the two cases:

- `True`: the entity is an axis-aligned extrusion. The 3D shape is exactly `footprint × [zmin, zmax]`. A 2D shapely test + a 1D interval test together form a necessary AND sufficient overlap condition.
- `False`: the entity has a non-trivial z-dependent profile (tapered polyprism). `footprint` is the union of all buffered cross-sections (a conservative xy envelope). The combined test is only necessary, not sufficient — disjoint result is reliable, overlapping result needs OCC confirmation.

### Metadata source: `GeometryEntity.overlap_metadata()`

A new optional method on `GeometryEntity`, returning `None` by default:

```python
# meshwell/geometry_entity.py
def overlap_metadata(self) -> tuple[shapely.Geometry, tuple[float, float], bool] | None:
    """Return (xy_envelope, (zmin, zmax), is_exact) or None.

    Subclasses that have a cheap, prismatic representation override this
    to opt into the cad_occ fast-overlap path. `is_exact=True` asserts
    the entity is mathematically (footprint × z-interval); the cad_occ
    cut-prep gate may then skip the OCC distance check entirely.
    """
    return None
```

`PolyPrism` overrides it:

```python
# meshwell/polyprism.py
def overlap_metadata(self):
    if self.extrude:
        footprint = (
            shapely.unary_union(self.polygons)
            if isinstance(self.polygons, list)
            else self.polygons
        )
        return (footprint, (self.zmin, self.zmax), True)
    # Tapered case: footprint = union of buffered cross-sections,
    # zrange = full extent. Conservative bound: necessary but not sufficient.
    # NOTE: buffered_polygons is list[list[tuple[float, Polygon]]] -- one
    # inner list per input polygon. The implementation flattens both levels.
    all_polygons = []
    z_keys = set()
    for entry in self.buffered_polygons:
        for z, polygon in entry:
            all_polygons.append(polygon)
            z_keys.add(z)
    footprint = shapely.unary_union(all_polygons)
    return (footprint, (min(z_keys), max(z_keys)), False)
```

### `_instantiate_entity_occ` populates the fields

`CAD_OCC._instantiate_entity_occ` is the only place that builds `OCCLabeledEntity`. It calls `entity_obj.overlap_metadata()` and writes the three fields:

```python
md = entity_obj.overlap_metadata()
if md is not None:
    fp, zr, exact = md
    return OCCLabeledEntity(
        ...,
        overlap_footprint=fp,
        overlap_zrange=zr,
        overlap_exact=exact,
    )
return OCCLabeledEntity(...)
```

### Fast-path in the cut prep loop

A new private helper centralises the decision:

```python
def _polyprism_fast_overlap(
    self, a: OCCLabeledEntity, b: OCCLabeledEntity
) -> bool | None:
    """Cheap shapely+z test.

    Returns:
      - False: definitively disjoint (skip cut).
      - True:  definitively overlapping (skip OCC distance, take cut).
                Only returned when both sides are exact axis-aligned extrusions.
      - None:  cannot decide (no metadata on one side, or tapered envelope
                test passed and OCC must verify).
    """
    if a.overlap_footprint is None or b.overlap_footprint is None:
        return None
    az, bz = a.overlap_zrange, b.overlap_zrange
    z_dist = max(0.0, max(az[0], bz[0]) - min(az[1], bz[1]))
    if z_dist > self.cut_fuzzy_value:
        return False
    if not a.overlap_footprint.dwithin(b.overlap_footprint, self.cut_fuzzy_value):
        return False
    if a.overlap_exact and b.overlap_exact:
        return True
    return None
```

The existing call site in `process_entities_cut_only` becomes:

```python
for ts in prev.shapes:
    tb = self._shape_bbox(ts)
    if tb is None:
        continue
    if not any(self._bboxes_overlap(ob, tb) for ob in obj_bboxes):
        continue
    fast = self._polyprism_fast_overlap(labeled, prev)
    if fast is False:
        continue
    if fast is None:
        if not any(
            self._shapes_actually_overlap(s, ts) for s in labeled.shapes
        ):
            continue
    # fast is True OR OCC distance confirmed overlap
    tool_shapes.append(ts)
```

Note: the fast-path runs once per `(labeled, prev)` entity pair, not per `(s, ts)` sub-shape pair, because the entity-level envelope is invariant to the cut cascade (cuts can only remove material from the footprint, never extend it).

### Why `overlap_exact = True` permits skipping OCC

For two axis-aligned extrusions A = (foot_A, [z0_A, z1_A]) and B = (foot_B, [z0_B, z1_B]), the 3D shapes overlap (within fuzzy ε) iff:

- foot_A.distance(foot_B) ≤ ε **AND**
- max(0, max(z0_A, z0_B) − min(z1_A, z1_B)) ≤ ε

This is necessary and sufficient — the shapely + interval test reproduces what OCC would compute, modulo floating-point. Skipping the OCC call is therefore correctness-preserving and is the primary perf win.

The regression test [`tests/test_cad_occ_batched_compound_cut.py`](../../../tests/test_cad_occ_batched_compound_cut.py) covers the substrate-vs-N-bodies pattern where this skip fires repeatedly and end-to-end output must match the pre-change behaviour.

## Public API

- New optional method `GeometryEntity.overlap_metadata()` returning `None` by default. Backward compatible.
- `PolyPrism.overlap_metadata()` implementation. Backward compatible.
- Three new optional fields on `OCCLabeledEntity` with safe defaults. Backward compatible.

No changes to `cad_occ()` or `process_entities()` signatures.

## Configuration

No new config flags. The existing `cut_fuzzy_value` continues to govern the overlap threshold.

## Testing

### Correctness

A new test module `tests/test_cad_occ_polyprism_overlap_fastpath.py` covers four scenarios with hand-built two-entity scenes:

1. **Two axis-aligned prisms with overlapping AABBs but disjoint polygons** — fast-path returns `False`; OCC distance (run for verification) would also report disjoint.
2. **Two axis-aligned prisms tangent in xy** (touching face) — fast-path returns `True`; OCC distance ≤ cut_fuzzy_value.
3. **Two axis-aligned prisms with separated z-ranges** — fast-path returns `False` via the z-interval branch; OCC concurs.
4. **Tapered prism (extrude=False) vs axis-aligned prism with overlapping conservative envelope but disjoint actual cross-section** — fast-path returns `None`; OCC distance correctly resolves to disjoint.

Each scenario asserts that the cut output (post-`process_entities_cut_only`) is identical to the cut output with the fast-path disabled (use a flag or temporary monkeypatch).

### Regression

The existing [`tests/test_cad_occ_batched_compound_cut.py`](../../../tests/test_cad_occ_batched_compound_cut.py) must continue to pass — it covers substrate-vs-20-bodies and locks down the cut output. The dense scene is all polyprisms with `extrude=True`, so it exercises the fast-path heavily.

### Performance

Extend the existing `scripts/bench_cut_strategies_dense.py` with a `time_cut_prep` helper that times the overlap-gate loop in isolation (separate from the BOP itself). Report the n=20 numbers in the implementation commit message. Target: at least 1.3× speedup on the *full* cut cascade (the overlap gate is one component; the BOP itself is unaffected).

## Out of scope

- `PolySurface` (2D) fast-path — different cut code path; not exercised by the same gate.
- Cross-entity-class fast-paths (e.g. interface tags). They already return `None` from `overlap_metadata()` and fall through to OCC, unchanged.
- Caching of OCC distance results for mixed (polyprism × non-polyprism) pairs. Possible follow-up if profiling shows residual hotspot.

## Risks

- **Tapered fast-reject is too conservative.** If `unary_union` of buffered cross-sections is poorly behaved for an unusual tapered shape, the bounding envelope might be too tight and incorrectly reject a real overlap. Mitigation: scenario 4 in the test suite covers tapered behaviour; the fast-path returns `None` rather than `False` when `overlap_exact=False` and the envelope says "maybe", preserving OCC verification.
- **fuzzy interpretation drift.** `dwithin` uses Euclidean distance; OCC `BRepExtrema_DistShapeShape` does too. Treatments should match. If a future change uses a different distance metric on one side without updating the other, the gate decisions could diverge. Mitigation: spec section "Why exact" documents the equivalence so a future maintainer notices the dependency.
- **Metadata staleness.** `overlap_metadata()` is called once at instantiate-time. If anything later mutates the entity's polygons or z-range (currently nothing does), the cached metadata goes stale. Mitigation: PolyPrism polygons are set in `__init__` and not mutated; if that invariant breaks, the regression test would fail.
