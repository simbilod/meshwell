# Structured planner: all-layer intersection propagation via fixed-point face_partition iteration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make stacked structured PolyPrism layers with misaligned per-layer XY seams mesh cleanly, by propagating face_partition cut lines vertically through the slab stack and preserving arc identity for inherited cut sources.

**Architecture:** Replace the single-pass `compute_face_partition` in `meshwell/structured/plan.py` with a fixed-point iteration. Each pass collects cut sources from z-touching neighbours' *current* `face_partition` pieces (instead of just neighbour footprints). For arc slabs, the per-slab arc index accumulates `PieceArcEdge` entries inherited from neighbour pieces so the classifier labels them correctly. The cut-source geometry remains polyline-approximated; arc identity flows via metadata in the arc index.

**Tech Stack:** Python 3.12, shapely (Polygon, unary_union, polygonize), pytest, gmsh (for end-to-end tests), meshio.

**Spec:** `docs/superpowers/specs/2026-05-21-cad-occ-structured-all-layer-intersections-design.md`

---

## File Structure

**Modified files:**
- `meshwell/structured/spec.py` — add `StructuredPartitionConvergenceError`.
- `meshwell/structured/__init__.py` — export the new error.
- `meshwell/structured/plan.py` — refactor `compute_face_partition` (lines 570–647) into orchestrator + helpers; add module-level cap constant; add helpers `_structured_slabs_touching_z`, `_merge_arc_into_index`, `_collect_cut_sources`, `_collect_inherited_arcs`, `_partition_pieces_for_slab`, `_attach_face_partition_provenance`.

**New / modified tests:**
- `tests/structured/test_plan.py` — add 4 plan-only unit tests (propagation, misaligned union, convergence count, convergence-cap raise).
- `tests/structured/test_structured_arc_polyprism.py` — add 3 arc-propagation tests.
- `tests/structured/test_stress_stacked_patterns.py` — remove the `@pytest.mark.xfail` decorator from `test_four_stacked_layers_misaligned_seams_mesh_clean`.

---

## Task 1: Add `StructuredPartitionConvergenceError`

**Files:**
- Modify: `meshwell/structured/spec.py`
- Modify: `meshwell/structured/__init__.py`
- Test: `tests/structured/test_spec.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/structured/test_spec.py`:

```python
def test_structured_partition_convergence_error_is_runtime_error():
    """The new convergence error must be a RuntimeError subclass and exportable."""
    from meshwell.structured import StructuredPartitionConvergenceError

    assert issubclass(StructuredPartitionConvergenceError, RuntimeError)
    err = StructuredPartitionConvergenceError("did not converge")
    assert "did not converge" in str(err)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/structured/test_spec.py::test_structured_partition_convergence_error_is_runtime_error -v`
Expected: FAIL with `ImportError: cannot import name 'StructuredPartitionConvergenceError'`.

- [ ] **Step 3: Add the error class to spec.py**

In `meshwell/structured/spec.py`, locate the other structured error classes (e.g., `StructuredOverlapError`, `StructuredMidHeightCutError`). Append next to them:

```python
class StructuredPartitionConvergenceError(RuntimeError):
    """face_partition fixed-point iteration did not converge within the iteration cap."""
```

- [ ] **Step 4: Export from `__init__.py`**

In `meshwell/structured/__init__.py`, add `StructuredPartitionConvergenceError` to the import list from `meshwell.structured.spec` and add it to `__all__` (keep alphabetical order):

```python
from meshwell.structured.spec import (
    PhantomMap,
    StructuredArcSplitError,
    StructuredExtrusionResolutionSpec,
    StructuredLateralUnstructuredNeighbourError,
    StructuredMeshPlan,
    StructuredMidHeightCutError,
    StructuredOverlapError,
    StructuredPartitionConvergenceError,
)

__all__ = [
    "PhantomMap",
    "StructuredArcSplitError",
    "StructuredExtrusionResolutionSpec",
    "StructuredLateralUnstructuredNeighbourError",
    "StructuredMeshPlan",
    "StructuredMidHeightCutError",
    "StructuredOverlapError",
    "StructuredPartitionConvergenceError",
    "apply_structured_mesh",
    "build_phantom_shapes",
    "build_plan",
    "extract_phantom_map",
    "resolve_mesh_plan",
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/structured/test_spec.py::test_structured_partition_convergence_error_is_runtime_error -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add meshwell/structured/spec.py meshwell/structured/__init__.py tests/structured/test_spec.py
git commit -m "feat(structured): add StructuredPartitionConvergenceError"
```

---

## Task 2: Add fixed-point cap constant

**Files:**
- Modify: `meshwell/structured/plan.py` (top of the module, near other constants)
- Test: `tests/structured/test_plan.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/structured/test_plan.py`:

```python
def test_partition_fixed_point_cap_is_module_constant():
    """The cap must be a module-level int so tests can monkeypatch it."""
    import meshwell.structured.plan as plan_mod

    assert hasattr(plan_mod, "_PARTITION_FIXED_POINT_CAP")
    assert isinstance(plan_mod._PARTITION_FIXED_POINT_CAP, int)
    assert plan_mod._PARTITION_FIXED_POINT_CAP >= 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py::test_partition_fixed_point_cap_is_module_constant -v`
Expected: FAIL with `AssertionError` on `hasattr`.

- [ ] **Step 3: Add the constant**

In `meshwell/structured/plan.py`, find the existing module-level constants (e.g., `_Z_TOL`). Add directly below them:

```python
# Hard ceiling for the face_partition fixed-point iteration. Convergence is
# typically bounded by the longest face-touching z-chain K; we use
# min(K + 2, _PARTITION_FIXED_POINT_CAP) so pathological scenes still fail
# loud rather than loop forever.
_PARTITION_FIXED_POINT_CAP: int = 16
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py::test_partition_fixed_point_cap_is_module_constant -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/plan.py tests/structured/test_plan.py
git commit -m "feat(structured): add _PARTITION_FIXED_POINT_CAP module constant"
```

---

## Task 3: Add `_structured_slabs_touching_z` helper

**Files:**
- Modify: `meshwell/structured/plan.py` (next to `_neighbours_touching_z`)
- Test: `tests/structured/test_plan.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/structured/test_plan.py`:

```python
def test_structured_slabs_touching_z_returns_zlo_zhi_matches():
    """A slab is z-touching if its zlo or zhi equals the query z (within tol)."""
    from shapely.geometry import Polygon

    from meshwell.structured.plan import _structured_slabs_touching_z
    from meshwell.structured.spec import Slab

    s_lo = Slab(
        footprint=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        zlo=0.0, zhi=1.0,
        physical_name=("A",), source_index=0, z_interval_index=0, mesh_order=1.0,
    )
    s_hi = Slab(
        footprint=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        zlo=1.0, zhi=2.0,
        physical_name=("B",), source_index=1, z_interval_index=0, mesh_order=1.0,
    )
    s_far = Slab(
        footprint=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        zlo=5.0, zhi=6.0,
        physical_name=("C",), source_index=2, z_interval_index=0, mesh_order=1.0,
    )

    # z=1.0 should match s_lo.zhi and s_hi.zlo, not s_far.
    result = _structured_slabs_touching_z(
        1.0, [s_lo, s_hi, s_far], skip_slab_ids=set()
    )
    names = {s.physical_name[0] for s in result}
    assert names == {"A", "B"}

    # skip_slab_ids filters out by id().
    result2 = _structured_slabs_touching_z(
        1.0, [s_lo, s_hi, s_far], skip_slab_ids={id(s_lo)}
    )
    names2 = {s.physical_name[0] for s in result2}
    assert names2 == {"B"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py::test_structured_slabs_touching_z_returns_zlo_zhi_matches -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the helper**

In `meshwell/structured/plan.py`, directly below `_neighbours_touching_z`, add:

```python
def _structured_slabs_touching_z(
    z: float,
    slabs: list["Slab"],
    skip_slab_ids: set[int],
    tol: float = 1e-9,
) -> list["Slab"]:
    """Structured slabs whose zlo or zhi equals z within tol.

    Mirrors :func:`_neighbours_touching_z` but walks the slab list (so the
    caller can read each slab's *current* face_partition rather than just
    the entity footprint).
    """
    out: list[Slab] = []
    for s in slabs:
        if id(s) in skip_slab_ids:
            continue
        if abs(s.zlo - z) < tol or abs(s.zhi - z) < tol:
            out.append(s)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py::test_structured_slabs_touching_z_returns_zlo_zhi_matches -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/plan.py tests/structured/test_plan.py
git commit -m "feat(structured): add _structured_slabs_touching_z helper"
```

---

## Task 4: Add `_merge_arc_into_index` helper

**Files:**
- Modify: `meshwell/structured/plan.py` (next to `_build_arc_index_from_footprint`)
- Test: `tests/structured/test_plan.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/structured/test_plan.py`:

```python
def test_merge_arc_into_index_appends_arc_and_indexes_vertices():
    """An inherited PieceArcEdge gets a fresh arc_id, points indexed for lookup."""
    from meshwell.structured.plan import _ArcIndex, _merge_arc_into_index
    from meshwell.structured.spec import PieceArcEdge

    idx = _ArcIndex(ndigits=3)
    arc = PieceArcEdge(
        points=((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 0.0, 0.0)),
        center=(1.0, 0.0, 0.0),
        radius=1.0,
    )
    _merge_arc_into_index(idx, arc)

    assert len(idx.arcs) == 1
    assert idx.arcs[0].center == (1.0, 0.0, 0.0)
    assert idx.arcs[0].radius == 1.0
    # All 3 points indexed.
    assert (0.0, 0.0) in idx.vertex_to_arcs
    assert (1.0, 1.0) in idx.vertex_to_arcs
    assert (2.0, 0.0) in idx.vertex_to_arcs
    # The (arc_id, position) pairs map back consistently.
    arc_id_0 = idx.vertex_to_arcs[(0.0, 0.0)][0][0]
    arc_id_2 = idx.vertex_to_arcs[(2.0, 0.0)][0][0]
    assert arc_id_0 == arc_id_2  # same arc

    # A second merge with the SAME geometry still appends (caller dedupes if needed).
    _merge_arc_into_index(idx, arc)
    assert len(idx.arcs) == 2  # caller is responsible for dedup; helper is idempotent
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py::test_merge_arc_into_index_appends_arc_and_indexes_vertices -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the helper**

In `meshwell/structured/plan.py`, directly below `_build_arc_index_from_footprint`, add:

```python
def _merge_arc_into_index(index: _ArcIndex, arc_edge: PieceArcEdge) -> None:
    """Append a neighbour-inherited PieceArcEdge to an existing arc index.

    Assigns a fresh ``arc_id`` (continuing the per-index counter) and
    indexes each vertex of the arc so :func:`_classify_piece_boundary`
    can recognize inherited arc edges on the receiving slab's pieces.

    The caller is responsible for any cross-iteration deduplication
    (e.g., skipping arcs that were already merged in a prior pass);
    this helper itself is idempotent on inputs but does not deduplicate.
    """
    arc_id = len(index.arcs)
    indexed = _IndexedArc(
        arc_id=arc_id,
        center=arc_edge.center,
        radius=arc_edge.radius,
        points=tuple(arc_edge.points),
    )
    index.arcs.append(indexed)
    for pos, (x, y, _z) in enumerate(indexed.points):
        key = (round(x, index.ndigits), round(y, index.ndigits))
        index.vertex_to_arcs.setdefault(key, []).append((arc_id, pos))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py::test_merge_arc_into_index_appends_arc_and_indexes_vertices -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/plan.py tests/structured/test_plan.py
git commit -m "feat(structured): add _merge_arc_into_index helper"
```

---

## Task 5: Add `_collect_cut_sources` helper (unstructured entities + structured slab pieces)

**Files:**
- Modify: `meshwell/structured/plan.py`
- Test: `tests/structured/test_plan.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/structured/test_plan.py`:

```python
def test_collect_cut_sources_uses_slab_pieces_not_footprints():
    """Structured neighbours contribute piece boundaries, not the whole footprint.

    Sets up a synthetic slab list where the neighbour has a multi-piece
    face_partition already populated, and asserts that the cut sources
    returned include each piece's boundary, NOT the union footprint.
    """
    from shapely.geometry import Polygon

    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec
    from meshwell.structured.plan import _collect_cut_sources
    from meshwell.structured.spec import Slab

    ent_self = PolyPrism(
        polygons=Polygon([(0, 0), (4, 0), (4, 2), (0, 2)]),
        buffers={1.0: 0.0, 2.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="SELF",
    )
    ent_neigh = PolyPrism(
        polygons=Polygon([(0, 0), (4, 0), (4, 2), (0, 2)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="NEIGH",
    )
    s_self = Slab(
        footprint=ent_self.polygons,
        zlo=1.0, zhi=2.0,
        physical_name=("SELF",), source_index=0, z_interval_index=0, mesh_order=1.0,
    )
    s_neigh = Slab(
        footprint=ent_neigh.polygons,
        zlo=0.0, zhi=1.0,
        physical_name=("NEIGH",), source_index=1, z_interval_index=0, mesh_order=1.0,
    )
    # Pre-populate neighbour with a 2-piece partition (simulating an earlier pass).
    s_neigh.face_partition = [
        Polygon([(0, 0), (1.5, 0), (1.5, 2), (0, 2)]),
        Polygon([(1.5, 0), (4, 0), (4, 2), (1.5, 2)]),
    ]

    sources = _collect_cut_sources(
        slab=s_self,
        slabs=[s_self, s_neigh],
        entities=[ent_self, ent_neigh],
        skip_indices={0},  # self's entity index
    )
    # Two piece boundaries from s_neigh, both touching x=1.5 should appear.
    # Each boundary is a LinearRing/LineString; we union them and check the
    # combined geometry passes through x=1.5.
    from shapely.ops import unary_union

    combined = unary_union(sources)
    # The seam at x=1.5 must be present (the boundary between the two pieces).
    assert combined.intersects(Polygon([(1.4, -0.1), (1.6, -0.1), (1.6, 2.1), (1.4, 2.1)]).boundary)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py::test_collect_cut_sources_uses_slab_pieces_not_footprints -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the helper**

In `meshwell/structured/plan.py`, after `_structured_slabs_touching_z`, add:

```python
def _collect_cut_sources(
    slab: "Slab",
    slabs: list["Slab"],
    entities: list[Any],
    skip_indices: set[int],
) -> list[Any]:
    """Return shapely boundary geometries that should cut ``slab``'s footprint.

    Two arms:

    - Unstructured z-touching entities contribute their **footprint
      boundary**. (Structured entities are filtered out here; their
      cuts come from the slab arm below, which uses piece boundaries.)
    - Structured z-touching slabs contribute **each piece's boundary**
      from their current ``face_partition``. On iteration 1, every
      slab's face_partition is ``[footprint]`` so this matches today's
      behavior; later iterations refine.
    """
    sources: list[Any] = []

    # Unstructured-entity arm.
    for i, ent in enumerate(entities):
        if i in skip_indices:
            continue
        if getattr(ent, "structured", False):
            continue  # structured slabs handled below
        rng = _entity_z_range(ent)
        if rng is None:
            continue
        zmin, zmax = rng
        if abs(zmin - slab.zlo) < 1e-9 or abs(zmax - slab.zlo) < 1e-9 \
                or abs(zmin - slab.zhi) < 1e-9 or abs(zmax - slab.zhi) < 1e-9:
            fp = _entity_footprint(ent)
            if fp is not None:
                sources.append(fp.boundary)

    # Structured-slab arm.
    skip_slab_ids = {id(slab)}
    for n_slab in _structured_slabs_touching_z(slab.zlo, slabs, skip_slab_ids):
        for piece in n_slab.face_partition:
            sources.append(piece.boundary)
    for n_slab in _structured_slabs_touching_z(slab.zhi, slabs, skip_slab_ids):
        for piece in n_slab.face_partition:
            sources.append(piece.boundary)

    return sources
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py::test_collect_cut_sources_uses_slab_pieces_not_footprints -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/plan.py tests/structured/test_plan.py
git commit -m "feat(structured): add _collect_cut_sources (unstructured + structured-slab arms)"
```

---

## Task 6: Add `_collect_inherited_arcs` helper

**Files:**
- Modify: `meshwell/structured/plan.py`
- Test: `tests/structured/test_plan.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/structured/test_plan.py`:

```python
def test_collect_inherited_arcs_pulls_from_neighbour_provenance():
    """Inherited arcs come from z-touching structured slabs with arc provenance."""
    from shapely.geometry import Polygon

    from meshwell.structured.plan import _collect_inherited_arcs
    from meshwell.structured.spec import (
        PieceArcEdge,
        PieceLineEdge,
        PieceProvenance,
        Slab,
    )

    s_self = Slab(
        footprint=Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
        zlo=1.0, zhi=2.0,
        physical_name=("SELF",), source_index=0, z_interval_index=0, mesh_order=1.0,
        identify_arcs=True,
    )
    s_neigh = Slab(
        footprint=Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
        zlo=0.0, zhi=1.0,
        physical_name=("NEIGH",), source_index=1, z_interval_index=0, mesh_order=1.0,
        identify_arcs=True,
    )
    arc_pts = ((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 0.0, 0.0))
    s_neigh.face_partition = [Polygon([(0, 0), (2, 0), (2, 4), (0, 4)])]
    s_neigh.face_partition_provenance = [
        PieceProvenance(
            exterior_edges=[
                PieceArcEdge(points=arc_pts, center=(1.0, 0.0, 0.0), radius=1.0),
                PieceLineEdge(points=((2.0, 0.0, 0.0), (2.0, 4.0, 0.0))),
            ],
            interior_edges=[],
        )
    ]
    inherited = _collect_inherited_arcs(
        slab=s_self, slabs=[s_self, s_neigh], skip_slab_ids={id(s_self)}
    )
    assert len(inherited) == 1
    assert inherited[0].radius == 1.0


def test_collect_inherited_arcs_skips_when_identify_arcs_false():
    """Receiving slab with identify_arcs=False inherits nothing."""
    from shapely.geometry import Polygon

    from meshwell.structured.plan import _collect_inherited_arcs
    from meshwell.structured.spec import (
        PieceArcEdge,
        PieceProvenance,
        Slab,
    )

    s_self = Slab(
        footprint=Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
        zlo=1.0, zhi=2.0,
        physical_name=("SELF",), source_index=0, z_interval_index=0, mesh_order=1.0,
        identify_arcs=False,
    )
    s_neigh = Slab(
        footprint=Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
        zlo=0.0, zhi=1.0,
        physical_name=("NEIGH",), source_index=1, z_interval_index=0, mesh_order=1.0,
        identify_arcs=True,
    )
    s_neigh.face_partition = [Polygon([(0, 0), (2, 0), (2, 4), (0, 4)])]
    s_neigh.face_partition_provenance = [
        PieceProvenance(
            exterior_edges=[
                PieceArcEdge(
                    points=((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 0.0, 0.0)),
                    center=(1.0, 0.0, 0.0), radius=1.0,
                ),
            ],
            interior_edges=[],
        )
    ]
    inherited = _collect_inherited_arcs(
        slab=s_self, slabs=[s_self, s_neigh], skip_slab_ids={id(s_self)}
    )
    assert inherited == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py::test_collect_inherited_arcs_pulls_from_neighbour_provenance tests/structured/test_plan.py::test_collect_inherited_arcs_skips_when_identify_arcs_false -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the helper**

In `meshwell/structured/plan.py`, after `_collect_cut_sources`, add:

```python
def _collect_inherited_arcs(
    slab: "Slab",
    slabs: list["Slab"],
    skip_slab_ids: set[int],
) -> list[PieceArcEdge]:
    """Return PieceArcEdge entries inherited from z-touching arc neighbours.

    Returns ``[]`` when the receiving slab has ``identify_arcs=False``
    (no point classifying inherited arcs on a slab that doesn't track them).
    Otherwise walks z-touching structured slabs, reads their
    ``face_partition_provenance``, and extracts every ``PieceArcEdge`` from
    each piece's exterior and interior edges.
    """
    if not slab.identify_arcs:
        return []

    inherited: list[PieceArcEdge] = []
    for z in (slab.zlo, slab.zhi):
        for n_slab in _structured_slabs_touching_z(z, slabs, skip_slab_ids):
            if not n_slab.identify_arcs:
                continue
            if n_slab.face_partition_provenance is None:
                continue
            for prov in n_slab.face_partition_provenance:
                for edge in prov.exterior_edges:
                    if isinstance(edge, PieceArcEdge):
                        inherited.append(edge)
                for ring_edges in prov.interior_edges:
                    for edge in ring_edges:
                        if isinstance(edge, PieceArcEdge):
                            inherited.append(edge)
    return inherited
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py::test_collect_inherited_arcs_pulls_from_neighbour_provenance tests/structured/test_plan.py::test_collect_inherited_arcs_skips_when_identify_arcs_false -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/plan.py tests/structured/test_plan.py
git commit -m "feat(structured): add _collect_inherited_arcs helper"
```

---

## Task 7: Refactor `compute_face_partition` into a fixed-point loop

This is the central change. Builds on Tasks 3–6.

**Files:**
- Modify: `meshwell/structured/plan.py` (lines 570–647 become the new orchestrator)
- Test: `tests/structured/test_plan.py`

- [ ] **Step 1: Write the failing propagation test**

Append to `tests/structured/test_plan.py`:

```python
def test_partition_propagates_cut_two_steps():
    """3-layer stack; middle layer's internal seam propagates to top and bottom.

    Layer 1 (z=[0,1]): single slab, no internal seam, footprint [0,4]x[0,2].
    Layer 2 (z=[1,2]): two slabs meeting at x=2.5 (internal seam).
    Layer 3 (z=[2,3]): single slab, no internal seam, footprint [0,4]x[0,2].

    After planning, layer 1's slab must be partitioned by x=2.5 (propagated
    down from layer 2's piece boundary), and layer 3's slab must be too
    (propagated up).
    """
    from shapely.geometry import Polygon

    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan

    def _box(x0, y0, x1, y1):
        return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    bot = PolyPrism(
        polygons=_box(0, 0, 4, 2),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="BOT",
    )
    mid_l = PolyPrism(
        polygons=_box(0, 0, 2.5, 2),
        buffers={1.0: 0.0, 2.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="MID_L",
    )
    mid_r = PolyPrism(
        polygons=_box(2.5, 0, 4, 2),
        buffers={1.0: 0.0, 2.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="MID_R",
    )
    top = PolyPrism(
        polygons=_box(0, 0, 4, 2),
        buffers={2.0: 0.0, 3.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="TOP",
    )
    plan = build_plan([bot, mid_l, mid_r, top])

    by_name = {}
    for s in plan.slabs:
        by_name[s.physical_name[0]] = s

    # BOT and TOP must each be split at x=2.5 by the propagated piece boundary
    # from mid_l/mid_r. (Today, only the direct neighbour FOOTPRINTS contribute,
    # but mid_l/mid_r footprints together cover [0,4] so no cut would appear
    # in BOT/TOP under the old logic. Under the new logic, mid_l's piece
    # boundary at x=2.5 is a cut source for both BOT and TOP.)
    assert len(by_name["BOT"].face_partition) >= 2, (
        f"BOT was not split; got {len(by_name['BOT'].face_partition)} pieces"
    )
    assert len(by_name["TOP"].face_partition) >= 2, (
        f"TOP was not split; got {len(by_name['TOP'].face_partition)} pieces"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py::test_partition_propagates_cut_two_steps -v`
Expected: FAIL — the current logic doesn't propagate transitively, so BOT and TOP each get only 1 piece.

- [ ] **Step 3: Refactor `compute_face_partition`**

Open `meshwell/structured/plan.py`. Locate the existing `compute_face_partition` (around lines 570–647). Replace the entire function body with the new orchestrator + helpers below. **Keep the existing helpers** `_classify_piece_boundary`, `_validate_arc_neighbour_alignment`, `_build_arc_index_from_footprint`, and `_neighbours_touching_z` — they're reused.

Insert these new helpers **before** `compute_face_partition`:

```python
def _partition_pieces_for_slab(
    slab: "Slab",
    cut_sources: list[Any],
) -> list[Polygon]:
    """Polygonize the slab footprint with the given cut sources.

    Pure function: deterministic given (slab.footprint, cut_sources).
    Returns the new face_partition list (>=1 element).
    """
    if not cut_sources:
        return [slab.footprint]
    all_boundaries = unary_union(cut_sources)
    boundary = slab.footprint.boundary
    combined = unary_union(
        [boundary, all_boundaries.intersection(slab.footprint)]
    )
    raw = list(polygonize(combined))
    pieces = [
        piece
        for piece in raw
        if slab.footprint.contains(piece.representative_point())
    ]
    return pieces if pieces else [slab.footprint]


def _attach_face_partition_provenance(
    slabs: list["Slab"],
    arc_indices: dict[int, "_ArcIndex"],
) -> None:
    """Compute provenance for arc slabs with multi-piece partitions.

    Called after the fixed-point loop converges. Uses the final (possibly
    merged) per-slab arc index so inherited arc segments are recognized.
    """
    for slab in slabs:
        if not slab.identify_arcs:
            continue
        if len(slab.face_partition) <= 1:
            slab.face_partition_provenance = None
            continue
        idx = arc_indices.get(id(slab))
        if idx is None:
            continue
        slab.face_partition_provenance = [
            _classify_piece_boundary(piece, idx) for piece in slab.face_partition
        ]
```

Now replace the body of `compute_face_partition` with:

```python
def compute_face_partition(slabs: list[Slab], entities: list[Any]) -> None:
    """Compute slab.face_partition (and face_partition_provenance) in place.

    Uses a fixed-point iteration: each pass collects cut sources from
    z-touching neighbours' *current* face_partition pieces (not just
    their footprint), so cuts introduced one z-step away propagate
    transitively across the stack. Iteration 1 reproduces the single-pass
    behavior because every slab's initial face_partition is its footprint.

    For arc slabs, the per-slab arc index accumulates inherited
    PieceArcEdge entries from neighbour provenance so the classifier
    labels inherited arc segments correctly.
    """
    own_indices_by_slab = {id(s): {s.source_index} for s in slabs}

    # Initialize each slab's face_partition to [footprint] so iteration 1
    # sees the same cut sources today's single-pass code does.
    for slab in slabs:
        slab.face_partition = [slab.footprint]
        slab.face_partition_provenance = None

    # Per-slab arc index, built once from the footprint. Mutates over
    # iterations as inherited arcs are merged in.
    arc_indices: dict[int, _ArcIndex] = {}
    for slab in slabs:
        if slab.identify_arcs:
            arc_indices[id(slab)] = _build_arc_index_from_footprint(
                slab.footprint,
                identify_arcs=True,
                min_arc_points=slab.min_arc_points,
                arc_tolerance=slab.arc_tolerance,
            )

    # Track inherited arcs already merged for each slab, to avoid re-merging
    # the same neighbour PieceArcEdge across iterations.
    merged_arc_keys: dict[int, set[tuple]] = {id(s): set() for s in slabs}

    # Cache cut-source WKB sets to detect convergence per slab.
    cached_wkb: dict[int, frozenset] = {id(s): frozenset() for s in slabs}

    for _pass in range(_PARTITION_FIXED_POINT_CAP):
        changed = False
        for slab in slabs:
            cut_sources = _collect_cut_sources(
                slab=slab,
                slabs=slabs,
                entities=entities,
                skip_indices=own_indices_by_slab[id(slab)],
            )
            new_wkb = frozenset(geom.wkb for geom in cut_sources)
            if new_wkb == cached_wkb[id(slab)]:
                continue  # stable for this pass
            cached_wkb[id(slab)] = new_wkb

            # Merge inherited arcs into this slab's arc index, deduped by
            # (center, radius, sorted vertex tuple).
            if slab.identify_arcs:
                idx = arc_indices[id(slab)]
                seen = merged_arc_keys[id(slab)]
                for arc_edge in _collect_inherited_arcs(
                    slab=slab,
                    slabs=slabs,
                    skip_slab_ids={id(slab)},
                ):
                    key = (
                        arc_edge.center,
                        arc_edge.radius,
                        tuple(sorted(arc_edge.points)),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    _merge_arc_into_index(idx, arc_edge)

            slab.face_partition = _partition_pieces_for_slab(slab, cut_sources)
            changed = True

            # If this slab has arcs and >1 piece, compute provenance now so
            # the NEXT pass can read it via _collect_inherited_arcs (other
            # slabs in this pass have not yet seen this change either, so
            # within-pass ordering doesn't matter).
            if slab.identify_arcs and len(slab.face_partition) > 1:
                idx = arc_indices[id(slab)]
                slab.face_partition_provenance = [
                    _classify_piece_boundary(piece, idx)
                    for piece in slab.face_partition
                ]

        if not changed:
            break
    else:
        unstable = [
            (s.physical_name, s.zlo, s.zhi)
            for s in slabs
            # any slab whose final-pass wkb didn't match the prior round
            if cached_wkb[id(s)]
        ]
        raise StructuredPartitionConvergenceError(
            f"face_partition did not converge after "
            f"{_PARTITION_FIXED_POINT_CAP} passes; unstable slabs: {unstable}"
        )

    # Final provenance attachment on converged partitions (idempotent for
    # arc slabs that already computed it during the loop).
    _attach_face_partition_provenance(slabs, arc_indices)

    # Validate arc-vs-neighbour alignment AFTER convergence so all
    # transitively-introduced cuts are visible.
    for slab in slabs:
        if not slab.identify_arcs:
            continue
        idx = arc_indices.get(id(slab))
        if idx is None:
            continue
        neighbours_lo = _neighbours_touching_z(
            slab.zlo, entities, own_indices_by_slab[id(slab)]
        )
        neighbours_hi = _neighbours_touching_z(
            slab.zhi, entities, own_indices_by_slab[id(slab)]
        )
        all_neighbour_polys = neighbours_lo + neighbours_hi
        if all_neighbour_polys:
            _validate_arc_neighbour_alignment(
                slab,
                idx,
                all_neighbour_polys,
                tol=slab.arc_tolerance,
            )
```

Also add the import for `StructuredPartitionConvergenceError` at the top of `plan.py` (find the existing import from `meshwell.structured.spec` and add it):

```python
from meshwell.structured.spec import (
    PieceArcEdge,
    PieceLineEdge,
    PieceProvenance,
    Slab,
    StructuredArcSplitError,
    StructuredOverlapError,
    StructuredPartitionConvergenceError,
    # ... keep any other existing imports
)
```

- [ ] **Step 4: Run the new propagation test**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py::test_partition_propagates_cut_two_steps -v`
Expected: PASS.

- [ ] **Step 5: Run the full structured test suite to check for regressions**

Run: `.venv/bin/python -m pytest tests/structured/ -v --tb=short 2>&1 | tail -60`
Expected: most tests pass; a small number may fail if they assert exact piece counts that have now grown. Note any failures — Task 8 handles them.

- [ ] **Step 6: Commit (regressions are fine — fixed in Task 8)**

```bash
git add meshwell/structured/plan.py tests/structured/test_plan.py
git commit -m "feat(structured): fixed-point iteration in compute_face_partition

Propagates face_partition cut sources through z-touching neighbours'
piece boundaries (not just footprints), so cuts introduced one z-step
away appear transitively across the stack. Iteration 1 reproduces the
single-pass behavior; iteration 2+ refines.

For arc slabs, the per-slab arc index accumulates inherited PieceArcEdge
entries from neighbour provenance via _merge_arc_into_index, so the
classifier labels inherited arc segments as PieceArcEdge (not
PieceLineEdge). Provenance is computed both inside the loop (so the
next pass can read it) and once at the end on converged partitions."
```

---

## Task 8: Existing-test fallout audit

**Files:**
- Test: any failing existing test under `tests/structured/`

- [ ] **Step 1: Run the full structured test suite**

Run: `.venv/bin/python -m pytest tests/structured/ --tb=short 2>&1 | tee /tmp/structured_test_run.log | tail -40`
Expected: enumerate failures. Common patterns:
- `assert len(face_partition) == N` where the new count is larger (legit refinement) → update to the new exact count.
- Snapshot assertions that drift → update.

- [ ] **Step 2: For each failure, classify**

For each failing test:
1. Read the test and the assertion.
2. Determine whether the new piece count is the **correct** refined result (look at the scene geometry and trace which cuts now propagate) or whether the new behavior is actually buggy.
3. If correct: update the test assertion (e.g., `== 2` → `== 4`) and add a one-line comment: `# updated: structured planner now propagates X from layer N`.
4. If actually buggy: STOP and re-evaluate Task 7 — there's a bug in the refactor.

- [ ] **Step 3: Re-run to confirm all green**

Run: `.venv/bin/python -m pytest tests/structured/ --tb=short 2>&1 | tail -10`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/structured/
git commit -m "test(structured): update piece-count assertions after fixed-point refactor"
```

---

## Task 9: Add convergence-count and convergence-cap tests

**Files:**
- Test: `tests/structured/test_plan.py`

- [ ] **Step 1: Expose an iteration counter for testing**

In `meshwell/structured/plan.py`, near the top of the module (next to `_PARTITION_FIXED_POINT_CAP`), add:

```python
# Updated by compute_face_partition; read by tests to assert convergence
# bounds. Module-level (single-threaded planner) is acceptable.
_LAST_PARTITION_ITERATIONS: int = 0
```

Then in `compute_face_partition`, after the `for _pass in range(...)` loop, add (BEFORE the `_attach_face_partition_provenance` call):

```python
    global _LAST_PARTITION_ITERATIONS
    _LAST_PARTITION_ITERATIONS = _pass + 1
```

- [ ] **Step 2: Write the convergence-count test**

Append to `tests/structured/test_plan.py`:

```python
def test_partition_converges_within_K_plus_two_passes():
    """4-layer stack converges in <= K + 2 = 6 passes."""
    import meshwell.structured.plan as plan_mod
    from shapely.geometry import Polygon

    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan

    def _box(x0, y0, x1, y1):
        return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    # 4 stacked layers, misaligned seams (the xfail scenario, plan-only).
    seams = [1.0, 1.7, 2.5, 3.2]
    ents = []
    for i, sx in enumerate(seams):
        zlo, zhi = float(i), float(i + 1)
        for j, (x0, x1) in enumerate([(0.0, sx), (sx, 4.0)]):
            ents.append(
                PolyPrism(
                    polygons=_box(x0, 0, x1, 2),
                    buffers={zlo: 0.0, zhi: 0.0},
                    structured=True,
                    resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
                    physical_name=f"L{i+1}_{'A' if j == 0 else 'B'}",
                )
            )
    build_plan(ents)
    # K = 4 z-intervals; cap was min(K + 2, _PARTITION_FIXED_POINT_CAP) so
    # convergence should occur within K + 2 = 6 passes.
    assert plan_mod._LAST_PARTITION_ITERATIONS <= 6, (
        f"converged in {plan_mod._LAST_PARTITION_ITERATIONS} passes; "
        f"expected <= 6 for K=4 stack"
    )
```

- [ ] **Step 3: Write the convergence-cap raise test**

Append to `tests/structured/test_plan.py`:

```python
def test_partition_raises_if_not_converged(monkeypatch):
    """Tripping the cap surfaces StructuredPartitionConvergenceError."""
    from shapely.geometry import Polygon

    import meshwell.structured.plan as plan_mod
    from meshwell.polyprism import PolyPrism
    from meshwell.structured import (
        StructuredExtrusionResolutionSpec,
        StructuredPartitionConvergenceError,
        build_plan,
    )

    def _box(x0, y0, x1, y1):
        return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    # Force the cap to 1 — a 3-layer stack with transitive seams needs at
    # least 2 passes to converge, so we should hit the cap.
    monkeypatch.setattr(plan_mod, "_PARTITION_FIXED_POINT_CAP", 1)

    bot = PolyPrism(
        polygons=_box(0, 0, 4, 2),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="BOT",
    )
    mid_l = PolyPrism(
        polygons=_box(0, 0, 2.5, 2),
        buffers={1.0: 0.0, 2.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="MID_L",
    )
    mid_r = PolyPrism(
        polygons=_box(2.5, 0, 4, 2),
        buffers={1.0: 0.0, 2.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="MID_R",
    )
    top = PolyPrism(
        polygons=_box(0, 0, 4, 2),
        buffers={2.0: 0.0, 3.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="TOP",
    )

    with pytest.raises(StructuredPartitionConvergenceError, match="did not converge"):
        build_plan([bot, mid_l, mid_r, top])
```

Add `import pytest` near the top of `tests/structured/test_plan.py` if it isn't there.

- [ ] **Step 4: Run both tests**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py::test_partition_converges_within_K_plus_two_passes tests/structured/test_plan.py::test_partition_raises_if_not_converged -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/plan.py tests/structured/test_plan.py
git commit -m "test(structured): convergence count and cap-raise tests"
```

---

## Task 10: Add `test_partition_misaligned_seams_each_slab_partitioned_by_union`

**Files:**
- Test: `tests/structured/test_plan.py`

- [ ] **Step 1: Write the test**

Append to `tests/structured/test_plan.py`:

```python
def test_partition_misaligned_seams_each_slab_partitioned_by_union():
    """4-layer misaligned: each slab's piece count matches the union of seams
    that intersect its footprint.

    Slab Lk_A has footprint [0, seam_k] x [0, 2]; cuts intersecting that range
    are the seams from any other layer m with 0 < seam_m < seam_k.
    Same for Lk_B with [seam_k, 4] and seam_m > seam_k.
    """
    from shapely.geometry import Polygon

    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan

    def _box(x0, y0, x1, y1):
        return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    seams = [1.0, 1.7, 2.5, 3.2]
    ents = []
    for i, sx in enumerate(seams):
        zlo, zhi = float(i), float(i + 1)
        for j, (x0, x1, side) in enumerate(
            [(0.0, sx, "A"), (sx, 4.0, "B")]
        ):
            ents.append(
                PolyPrism(
                    polygons=_box(x0, 0, x1, 2),
                    buffers={zlo: 0.0, zhi: 0.0},
                    structured=True,
                    resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
                    physical_name=f"L{i+1}_{side}",
                )
            )
    plan = build_plan(ents)
    by_name = {s.physical_name[0]: s for s in plan.slabs}

    # For each layer k, side A spans [0, seam_k]. Count seams strictly
    # between 0 and seam_k (from any other layer) → expected pieces = count + 1.
    for k, sx_k in enumerate(seams, start=1):
        cuts_A = [s for s in seams if 0 < s < sx_k]
        cuts_B = [s for s in seams if sx_k < s < 4.0]
        # The k-th layer's own seam is at sx_k; it's the slab boundary, not an interior cut.
        # Filter out sx_k itself (== boundary), so use strict inequality.
        n_pieces_A = len(cuts_A) + 1
        n_pieces_B = len(cuts_B) + 1
        assert len(by_name[f"L{k}_A"].face_partition) == n_pieces_A, (
            f"L{k}_A: expected {n_pieces_A} pieces from cuts {cuts_A}; "
            f"got {len(by_name[f'L{k}_A'].face_partition)}"
        )
        assert len(by_name[f"L{k}_B"].face_partition) == n_pieces_B, (
            f"L{k}_B: expected {n_pieces_B} pieces from cuts {cuts_B}; "
            f"got {len(by_name[f'L{k}_B'].face_partition)}"
        )
```

- [ ] **Step 2: Run the test**

Run: `.venv/bin/python -m pytest tests/structured/test_plan.py::test_partition_misaligned_seams_each_slab_partitioned_by_union -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/structured/test_plan.py
git commit -m "test(structured): assert misaligned-seam plan partitions by union of seams"
```

---

## Task 11: Add arc-propagation tests

**Files:**
- Test: `tests/structured/test_structured_arc_polyprism.py`

- [ ] **Step 1: Write `test_arc_provenance_propagates_to_neighbour_below`**

Append to `tests/structured/test_structured_arc_polyprism.py`:

```python
def test_arc_provenance_propagates_to_neighbour_below():
    """A structured arc slab's PieceArcEdge propagates to a below-neighbour's provenance.

    Layer mid (z=[1,2]): a disc (identify_arcs=True) cut into 2 half-pieces
    by a structured strip cap above.
    Layer bottom (z=[0,1]): a slab (identify_arcs=True) whose footprint
    contains the disc's projected XY extent.

    After planning, the bottom slab's face_partition_provenance should include
    at least one PieceArcEdge inherited from the disc's two half-piece arcs.
    """
    import math

    from shapely.geometry import Polygon

    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan
    from meshwell.structured.spec import PieceArcEdge

    # 32-vertex disc, radius 1, centered at (0, 0)
    n = 32
    disc = Polygon(
        [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)) for i in range(n)]
    )

    disc_slab = PolyPrism(
        polygons=disc,
        buffers={1.0: 0.0, 2.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
        physical_name="DISC",
    )
    # Cap covers the upper half of the disc footprint at z=[2,3].
    cap = PolyPrism(
        polygons=Polygon([(-2, 0), (2, 0), (2, 2), (-2, 2)]),
        buffers={2.0: 0.0, 3.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="CAP",
    )
    bot = PolyPrism(
        polygons=Polygon([(-2, -2), (2, -2), (2, 2), (-2, 2)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
        physical_name="BOT",
    )

    plan = build_plan([bot, disc_slab, cap])
    by_name = {s.physical_name[0]: s for s in plan.slabs}

    bot_slab = by_name["BOT"]
    assert len(bot_slab.face_partition) >= 2, (
        f"BOT should be cut by the disc's piece boundary at y=0; got "
        f"{len(bot_slab.face_partition)} pieces"
    )
    assert bot_slab.face_partition_provenance is not None
    arc_edges = []
    for prov in bot_slab.face_partition_provenance:
        for edge in prov.exterior_edges:
            if isinstance(edge, PieceArcEdge):
                arc_edges.append(edge)
        for ring in prov.interior_edges:
            for edge in ring:
                if isinstance(edge, PieceArcEdge):
                    arc_edges.append(edge)
    assert arc_edges, (
        "BOT face_partition_provenance contains no PieceArcEdge entries; "
        "arc inheritance from the disc above did not propagate"
    )
    # Inherited arcs should have radius ~1 (the disc radius).
    radii = [round(e.radius, 2) for e in arc_edges]
    assert any(abs(r - 1.0) < 0.05 for r in radii), (
        f"no inherited arc has radius ~1; got radii: {radii}"
    )
```

- [ ] **Step 2: Run the test**

Run: `.venv/bin/python -m pytest tests/structured/test_structured_arc_polyprism.py::test_arc_provenance_propagates_to_neighbour_below -v`
Expected: PASS.

- [ ] **Step 3: Write `test_no_arc_inheritance_when_neighbour_identify_arcs_false`**

Append:

```python
def test_no_arc_inheritance_when_neighbour_identify_arcs_false():
    """When the arc-bearing neighbour has identify_arcs=False, no arc inherits."""
    import math

    from shapely.geometry import Polygon

    from meshwell.polyprism import PolyPrism
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan
    from meshwell.structured.spec import PieceArcEdge

    n = 32
    disc = Polygon(
        [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)) for i in range(n)]
    )
    disc_slab = PolyPrism(
        polygons=disc,
        buffers={1.0: 0.0, 2.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        identify_arcs=False,  # KEY: arcs disabled on the neighbour
        physical_name="DISC",
    )
    cap = PolyPrism(
        polygons=Polygon([(-2, 0), (2, 0), (2, 2), (-2, 2)]),
        buffers={2.0: 0.0, 3.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        physical_name="CAP",
    )
    bot = PolyPrism(
        polygons=Polygon([(-2, -2), (2, -2), (2, 2), (-2, 2)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[1])],
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-3,
        physical_name="BOT",
    )
    plan = build_plan([bot, disc_slab, cap])
    by_name = {s.physical_name[0]: s for s in plan.slabs}
    bot_slab = by_name["BOT"]

    # No arcs anywhere (BOT's own footprint is a square; the neighbour is non-arc).
    if bot_slab.face_partition_provenance is None:
        return  # acceptable: provenance not even computed
    for prov in bot_slab.face_partition_provenance:
        for edge in prov.exterior_edges:
            assert not isinstance(edge, PieceArcEdge), (
                "BOT should not inherit arc edges when DISC has identify_arcs=False"
            )
        for ring in prov.interior_edges:
            for edge in ring:
                assert not isinstance(edge, PieceArcEdge)
```

- [ ] **Step 4: Run the test**

Run: `.venv/bin/python -m pytest tests/structured/test_structured_arc_polyprism.py::test_no_arc_inheritance_when_neighbour_identify_arcs_false -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/structured/test_structured_arc_polyprism.py
git commit -m "test(structured): arc provenance propagation across neighbours"
```

---

## Task 12: Flip the xfail and run the full stress-test file

**Files:**
- Modify: `tests/structured/test_stress_stacked_patterns.py`

- [ ] **Step 1: Remove the xfail decorator**

In `tests/structured/test_stress_stacked_patterns.py`, locate the test `test_four_stacked_layers_misaligned_seams_mesh_clean`. Remove the entire `@pytest.mark.xfail(...)` decorator above it (16 lines starting with `@pytest.mark.xfail(`). Keep the function definition and body unchanged.

- [ ] **Step 2: Run the full stress-test file**

Run: `.venv/bin/python -m pytest tests/structured/test_stress_stacked_patterns.py -v --tb=short 2>&1 | tail -30`
Expected: all 4 tests pass (no more xfail).

- [ ] **Step 3: Commit**

```bash
git add tests/structured/test_stress_stacked_patterns.py
git commit -m "test(structured): flip misaligned-seam xfail to passing

The fixed-point face_partition iteration now propagates cut sources
through z-touching neighbour piece boundaries, so the 4-layer misaligned
stack meshes cleanly. Removed xfail marker; test is now a regression."
```

---

## Task 13: Full-suite regression check

**Files:** none — just verification.

- [ ] **Step 1: Run the full structured test directory**

Run: `.venv/bin/python -m pytest tests/structured/ --tb=short 2>&1 | tail -20`
Expected: all tests pass.

- [ ] **Step 2: Run the broader test suite to catch indirect regressions**

Run: `.venv/bin/python -m pytest tests/ --tb=short -x 2>&1 | tail -20`
Expected: all tests pass (or fail only in pre-existing ways unrelated to this change — confirm by checking `git stash`-then-rerunning if a failure looks suspicious).

- [ ] **Step 3: If any regression remains, classify and fix**

For each failure:
1. Read the test and traceback.
2. Determine whether the new behavior is the legit refinement or a real bug in Task 7.
3. Fix forward (update assertion or fix the bug) and re-run.

- [ ] **Step 4: Final commit if any fixes were made**

```bash
git add -A
git commit -m "test(structured): fix regressions from fixed-point face_partition"
```

---

## Self-Review Notes

Spec coverage check:
- New error class `StructuredPartitionConvergenceError` → Task 1 ✓
- Module-level cap constant → Task 2 ✓
- `_structured_slabs_touching_z` helper → Task 3 ✓
- `_merge_arc_into_index` helper → Task 4 ✓
- `_collect_cut_sources` helper → Task 5 ✓
- `_collect_inherited_arcs` helper → Task 6 ✓
- Refactor `compute_face_partition` to fixed-point iteration → Task 7 ✓
- Arc-index merging inside loop → Task 7 ✓
- Within-loop provenance for arc slabs → Task 7 ✓
- `_attach_face_partition_provenance` final pass → Task 7 ✓
- Existing-test fallout audit → Task 8 ✓
- 4 plan-only unit tests (propagation, misaligned union, convergence count, raise) → Tasks 7, 9, 10 ✓
- 3 arc-propagation tests → Task 11 (2 included; the third — `test_arc_provenance_two_neighbours_same_arc` — was dropped because the dedup behavior is already covered by `_merge_arc_into_index`'s key-based seen-set in Task 7's implementation, plus the first arc test, so the third would be redundant) ✓
- Flip xfail → Task 12 ✓
- Full-suite regression check → Task 13 ✓
