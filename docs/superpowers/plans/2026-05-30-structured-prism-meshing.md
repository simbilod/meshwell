# Structured Prism Meshing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add structured (wedge/prism) meshing to `meshwell.cad_occ` so a user can mark `PolyPrism` entities `structured=True` and get back wedge-element volumes with conformal interfaces to surrounding tet-meshed unstructured regions.

**Architecture:** Five-stage planner (`collect → cohort → decompose → build → wedge`) that builds one `TopoDS_Compound` per cohort containing N first-class sub-solids with shared internal TShapes. Bidirectional pre-cut in shapely at shared z-planes so BOP merges coincident TShapes rather than splitting faces. Wedge stamping per sub-solid in `pre_3d_hook`.

**Tech Stack:** Python 3.10+, shapely, OCP (OpenCASCADE Python bindings), gmsh, pytest.

**Spec:** [docs/superpowers/specs/2026-05-30-structured-prism-meshing-design.md](../specs/2026-05-30-structured-prism-meshing-design.md)

---

## File map

**Created:**
- `meshwell/structured/__init__.py` — public exports
- `meshwell/structured/exceptions.py` — custom error classes
- `meshwell/structured/types.py` — dataclasses (StructuredSlab, Cohort, SubPiece, SlabMeta)
- `meshwell/structured/collect.py` — Stage 1
- `meshwell/structured/cohort.py` — Stage 2
- `meshwell/structured/decompose.py` — Stage 3
- `meshwell/structured/build.py` — Stage 4
- `meshwell/structured/cohort_entity.py` — `_CohortEntity` wrapper for cad_occ
- `meshwell/structured/validators.py` — z-stack + shell-invariance
- `meshwell/structured/wedge.py` — Stage 5 (pre_2d + pre_3d hooks)
- 12 test files under `tests/structured/`

**Modified:**
- `meshwell/polyprism.py` — add `structured`, `n_layers` fields
- `meshwell/cad_occ.py` — unwrap compound at instantiation; expose post-BOP `Modified()` map
- `meshwell/mesh.py` — accept `pre_2d_hook`, `pre_3d_hook` parameters
- `meshwell/orchestrator.py` — wire structured pre/post passes and hooks

---

## Task 0: Module skeleton + custom exceptions

**Files:**
- Create: `meshwell/structured/__init__.py`
- Create: `meshwell/structured/exceptions.py`
- Test: `tests/structured/test_exceptions.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/structured/test_exceptions.py
from meshwell.structured.exceptions import (
    StructuredExtrudeRequiredError,
    StructuredEntityTypeError,
    StructuredZStackError,
    UnstructuredImprintRequiresPolyPrismError,
    SubPolygonAssemblyError,
    CohortNonManifoldError,
    CohortShellModifiedError,
    StructuredLateralNLayersMismatchError,
    StructuredTransfiniteRejectedError,
    WedgeCountMismatchError,
    WedgeBotNodeMismatchError,
)


def test_all_exceptions_are_structured_errors():
    """Every custom error inherits a common base for easy catching."""
    from meshwell.structured.exceptions import StructuredError
    for cls in [
        StructuredExtrudeRequiredError,
        StructuredEntityTypeError,
        StructuredZStackError,
        UnstructuredImprintRequiresPolyPrismError,
        SubPolygonAssemblyError,
        CohortNonManifoldError,
        CohortShellModifiedError,
        StructuredLateralNLayersMismatchError,
        StructuredTransfiniteRejectedError,
        WedgeCountMismatchError,
        WedgeBotNodeMismatchError,
    ]:
        assert issubclass(cls, StructuredError), cls.__name__


def test_zstack_error_carries_context():
    err = StructuredZStackError(entity_index=2, z=1.5, cohort_index=0)
    assert err.entity_index == 2
    assert err.z == 1.5
    assert err.cohort_index == 0
    assert "1.5" in str(err)
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/structured/test_exceptions.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# meshwell/structured/__init__.py
"""Structured prism meshing for meshwell.cad_occ."""
```

```python
# meshwell/structured/exceptions.py
"""All custom errors raised by the structured pipeline.

Each error inherits from StructuredError so callers can catch the
whole family with one except clause.
"""
from __future__ import annotations


class StructuredError(Exception):
    """Base class for all structured-pipeline errors."""


class StructuredExtrudeRequiredError(StructuredError):
    def __init__(self, entity_index: int):
        self.entity_index = entity_index
        super().__init__(
            f"PolyPrism #{entity_index} has structured=True but extrude=False; "
            "only constant-XY-footprint prisms are structured-eligible."
        )


class StructuredEntityTypeError(StructuredError):
    def __init__(self, entity_index: int, type_name: str):
        self.entity_index = entity_index
        self.type_name = type_name
        super().__init__(
            f"Entity #{entity_index} ({type_name}) has structured=True; "
            "only PolyPrism is supported as a structured entity."
        )


class StructuredZStackError(StructuredError):
    def __init__(self, entity_index: int, z: float, cohort_index: int):
        self.entity_index = entity_index
        self.z = z
        self.cohort_index = cohort_index
        super().__init__(
            f"Entity #{entity_index} has a z-boundary at z={z} falling "
            f"strictly inside cohort #{cohort_index} while sharing XY. "
            "v1 requires all entity z-boundaries to coincide with cohort "
            "z-planes; restructure your stack to make z-boundaries explicit."
        )


class UnstructuredImprintRequiresPolyPrismError(StructuredError):
    def __init__(self, entity_index: int, type_name: str, z: float):
        self.entity_index = entity_index
        self.type_name = type_name
        self.z = z
        super().__init__(
            f"Entity #{entity_index} ({type_name}) shares z-plane z={z} with "
            "a structured cohort but is not a PolyPrism(extrude=True); "
            "pre-cut requires a shapely polygon."
        )


class SubPolygonAssemblyError(StructuredError):
    def __init__(self, cohort_index: int, z_interval: tuple[float, float], reason: str):
        self.cohort_index = cohort_index
        self.z_interval = z_interval
        self.reason = reason
        super().__init__(
            f"Cohort #{cohort_index} sub-polygon assembly failed at "
            f"z={z_interval}: {reason}"
        )


class CohortNonManifoldError(StructuredError):
    def __init__(self, cohort_index: int, edge_count: int):
        self.cohort_index = cohort_index
        self.edge_count = edge_count
        super().__init__(
            f"Cohort #{cohort_index} sewn compound has {edge_count} "
            "non-manifold edges (planner bug — internal face sharing is wrong)."
        )


class CohortShellModifiedError(StructuredError):
    def __init__(
        self, slab_index: int, face_role: str, fragment_count: int
    ):
        self.slab_index = slab_index
        self.face_role = face_role
        self.fragment_count = fragment_count
        super().__init__(
            f"BOP modified pre-baked cohort shell face (slab #{slab_index}, "
            f"role={face_role}); post-BOP fragment count = {fragment_count}. "
            "Either Stage 3d's pre-cut decomposition was incomplete, or the "
            "fragment_fuzzy_value needs adjustment."
        )


class StructuredLateralNLayersMismatchError(StructuredError):
    def __init__(
        self,
        slab_a: int,
        slab_b: int,
        face_tag: int,
        n_layers_a: int,
        n_layers_b: int,
    ):
        self.slab_a = slab_a
        self.slab_b = slab_b
        self.face_tag = face_tag
        self.n_layers_a = n_layers_a
        self.n_layers_b = n_layers_b
        super().__init__(
            f"Lateral face #{face_tag} shared between structured slabs "
            f"#{slab_a} (n_layers={n_layers_a}) and #{slab_b} "
            f"(n_layers={n_layers_b}); n_layers must match."
        )


class StructuredTransfiniteRejectedError(StructuredError):
    def __init__(self, face_tag: int, slab_index: int, reason: str):
        self.face_tag = face_tag
        self.slab_index = slab_index
        self.reason = reason
        super().__init__(
            f"gmsh rejected transfinite hint on lateral face #{face_tag} "
            f"(slab #{slab_index}): {reason}"
        )


class WedgeCountMismatchError(StructuredError):
    def __init__(self, slab_index: int, expected: int, got: int):
        self.slab_index = slab_index
        self.expected = expected
        self.got = got
        super().__init__(
            f"Wedge stamp for slab #{slab_index}: expected {expected} "
            f"wedges (bot triangles × n_layers), emitted {got}."
        )


class WedgeBotNodeMismatchError(StructuredError):
    def __init__(self, slab_index: int, mismatched_count: int):
        self.slab_index = slab_index
        self.mismatched_count = mismatched_count
        super().__init__(
            f"Wedge stamp for slab #{slab_index}: {mismatched_count} bot "
            "vertices did not match the bot face mesh node tags."
        )
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/structured/test_exceptions.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add meshwell/structured/__init__.py meshwell/structured/exceptions.py tests/structured/test_exceptions.py
git commit -m "feat(structured): module skeleton + custom exception hierarchy"
```

---

## Task 1: PolyPrism gains `structured` and `n_layers` fields

**Files:**
- Modify: `meshwell/polyprism.py:29-98` (`__init__`)
- Test: `tests/structured/test_polyprism_structured_flag.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/structured/test_polyprism_structured_flag.py
import pytest
from shapely.geometry import Polygon

from meshwell.polyprism import PolyPrism
from meshwell.structured.exceptions import StructuredExtrudeRequiredError


SQUARE = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])


def test_default_is_unstructured():
    p = PolyPrism(polygons=SQUARE, buffers={0.0: 0.0, 1.0: 0.0}, physical_name="x")
    assert p.structured is False
    assert p.n_layers == 1


def test_structured_true_extrude_ok():
    p = PolyPrism(
        polygons=SQUARE,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="x",
        structured=True,
        n_layers=3,
    )
    assert p.structured is True
    assert p.n_layers == 3
    assert p.extrude is True
    assert p.identify_arcs is True  # default flips when structured


def test_structured_identify_arcs_user_override():
    p = PolyPrism(
        polygons=SQUARE,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="x",
        structured=True,
        identify_arcs=False,
    )
    assert p.identify_arcs is False


def test_structured_buffered_raises():
    with pytest.raises(StructuredExtrudeRequiredError):
        PolyPrism(
            polygons=SQUARE,
            buffers={0.0: 0.0, 1.0: 0.5},  # non-zero buffer → extrude=False
            physical_name="x",
            structured=True,
        )


def test_n_layers_validation():
    with pytest.raises(ValueError, match="n_layers"):
        PolyPrism(
            polygons=SQUARE,
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="x",
            structured=True,
            n_layers=0,
        )
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/structured/test_polyprism_structured_flag.py -v
```
Expected: FAIL — `structured` is not a kwarg of `PolyPrism.__init__`.

- [ ] **Step 3: Modify PolyPrism.__init__**

Add two parameters and validation. In `meshwell/polyprism.py`, locate the existing `__init__` signature and add `structured: bool = False, n_layers: int = 1` right after `identify_arcs`. Add validation immediately after `self.extrude` is set:

```python
# In PolyPrism.__init__, in the signature block:
        structured: bool = False,
        n_layers: int = 1,
```

```python
# In __init__ body, AFTER self.extrude is set (~line 78) and BEFORE
# self.identify_arcs is assigned:
        if structured:
            from meshwell.structured.exceptions import (
                StructuredExtrudeRequiredError,
            )
            if not self.extrude:
                raise StructuredExtrudeRequiredError(entity_index=-1)
            if n_layers < 1:
                raise ValueError(
                    f"n_layers must be >= 1 (got {n_layers})"
                )
            # Default identify_arcs flips True for structured (arcs matter
            # for lateral wall quad quality). User override wins.
            if identify_arcs is False and "identify_arcs" not in (
                # crude detection: leave alone if user set it explicitly.
                # Easier path: just trust the caller's value.
                {}
            ):
                identify_arcs = True
        self.structured = structured
        self.n_layers = n_layers
```

The `identify_arcs` default flip is awkward because `__init__` can't easily know if the caller passed `identify_arcs` explicitly. Cleaner: change the default of `identify_arcs` from `False` to `None`, then resolve it after we see `structured`. Apply this:

```python
# Signature (replace):
        identify_arcs: bool | None = None,
        ...
        structured: bool = False,
        n_layers: int = 1,
```

```python
# Body (replace the identify_arcs assignment + add structured validation):
        if structured:
            from meshwell.structured.exceptions import (
                StructuredExtrudeRequiredError,
            )
            if not self.extrude:
                raise StructuredExtrudeRequiredError(entity_index=-1)
            if n_layers < 1:
                raise ValueError(f"n_layers must be >= 1 (got {n_layers})")
        if identify_arcs is None:
            identify_arcs = True if structured else False
        self.structured = structured
        self.n_layers = n_layers
        self.identify_arcs = identify_arcs
```

Remove the old `self.identify_arcs = identify_arcs` line that already exists if it's now duplicated.

- [ ] **Step 4: Run test to verify it passes + existing tests still pass**

```
pytest tests/structured/test_polyprism_structured_flag.py -v
pytest tests/test_arc_extrusion.py tests/test_arc_identification.py -v
```
Expected: PASS on both.

- [ ] **Step 5: Commit**

```
git add meshwell/polyprism.py tests/structured/test_polyprism_structured_flag.py
git commit -m "feat(polyprism): add structured + n_layers fields with validation"
```

---

## Task 2: `types.py` — core dataclasses

**Files:**
- Create: `meshwell/structured/types.py`
- Test: `tests/structured/test_types.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/structured/test_types.py
from shapely.geometry import Polygon

from meshwell.structured.types import (
    StructuredSlab, Cohort, SubPiece, SlabMeta, ShapeKey,
)

SQ = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


def test_structured_slab_is_frozen():
    s = StructuredSlab(
        source_index=0, footprint=SQ, zlo=0.0, zhi=1.0,
        n_layers=2, mesh_order=1.0, mesh_bool=True,
        physical_name=("a",), identify_arcs=True,
        arc_tolerance=1e-3, min_arc_points=4,
    )
    assert s.zlo == 0.0
    import dataclasses
    assert dataclasses.is_dataclass(s)
    # frozen → assignment raises
    import pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.zlo = 5.0


def test_cohort_default_z_planes_sorted():
    s1 = StructuredSlab(0, SQ, 0.0, 1.0, 1, 1.0, True, ("a",), True, 1e-3, 4)
    s2 = StructuredSlab(1, SQ, 1.0, 2.0, 1, 1.0, True, ("b",), True, 1e-3, 4)
    c = Cohort(slabs=(s1, s2), z_planes=(0.0, 1.0, 2.0))
    assert c.z_planes == (0.0, 1.0, 2.0)
    assert c.zmin == 0.0
    assert c.zmax == 2.0


def test_subpiece_carries_source_indices():
    sp = SubPiece(
        cohort_index=0, z_interval=(0.0, 1.0),
        sub_polygon=SQ, source_slab_indices=(0, 3),
    )
    assert sp.source_slab_indices == (0, 3)


def test_shape_key_is_hashable():
    k = ShapeKey(tshape_id=12345, orientation=0)
    {k: "value"}  # noqa: B018 — must be hashable
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/structured/test_types.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the module**

```python
# meshwell/structured/types.py
"""Dataclasses shared across all structured-pipeline stages.

Kept in one module so importers don't have to know which stage owns
which type. All dataclasses are frozen — these records flow through
the pipeline immutably.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shapely.geometry import MultiPolygon, Polygon


@dataclass(frozen=True)
class ShapeKey:
    """Stable identity for a TopoDS_Shape used as a dict key.

    TShape pointer + orientation matches cad_occ._shape_key. We
    redeclare it here as a frozen dataclass so it's pickle-safe and
    type-checkable.
    """
    tshape_id: int
    orientation: int


@dataclass(frozen=True)
class StructuredSlab:
    """One z-interval of one structured PolyPrism.

    A single PolyPrism with N+1 buffer keys yields N StructuredSlab
    records, one per adjacent (zlo, zhi) pair.
    """
    source_index: int
    footprint: "Polygon | MultiPolygon"
    zlo: float
    zhi: float
    n_layers: int
    mesh_order: float | None
    mesh_bool: bool
    physical_name: tuple[str, ...]
    identify_arcs: bool
    arc_tolerance: float
    min_arc_points: int


@dataclass(frozen=True)
class Cohort:
    """Connected component of structured slabs (Union-Find)."""
    slabs: tuple[StructuredSlab, ...]
    z_planes: tuple[float, ...]   # sorted unique cohort z-boundaries

    @property
    def zmin(self) -> float:
        return self.z_planes[0]

    @property
    def zmax(self) -> float:
        return self.z_planes[-1]


@dataclass(frozen=True)
class SubPiece:
    """One (z-interval × sub-polygon) cell after decomposition.

    Each SubPiece becomes one TopoDS_Solid in the cohort compound.
    """
    cohort_index: int
    z_interval: tuple[float, float]
    sub_polygon: "Polygon"
    source_slab_indices: tuple[int, ...]


@dataclass(frozen=True)
class SlabMeta:
    """Per-sub-solid metadata used at meshing time.

    Lookup happens by post-BOP ShapeKey of the sub-solid in the
    OCCLabeledEntity's shapes list.
    """
    slab_index: int
    n_layers: int
    physical_name: tuple[str, ...]
    bot_face_key: ShapeKey
    top_face_key: ShapeKey
    lateral_face_keys: tuple[ShapeKey, ...]
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/structured/test_types.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add meshwell/structured/types.py tests/structured/test_types.py
git commit -m "feat(structured): core dataclasses (StructuredSlab, Cohort, SubPiece, SlabMeta, ShapeKey)"
```

---

---

## Task 3: `collect.py` — Stage 1

**Files:**
- Create: `meshwell/structured/collect.py`
- Test: `tests/structured/test_collect.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/structured/test_collect.py
import pytest
from shapely.geometry import Polygon

from meshwell.polyprism import PolyPrism
from meshwell.structured.collect import collect_structured_slabs
from meshwell.structured.exceptions import StructuredEntityTypeError


SQ = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


def make_prism(name, zlo=0.0, zhi=1.0, structured=False, n_layers=1, mesh_order=1.0):
    return PolyPrism(
        polygons=SQ, buffers={zlo: 0.0, zhi: 0.0},
        physical_name=name, structured=structured,
        n_layers=n_layers, mesh_order=mesh_order,
    )


def test_separates_structured_from_unstructured():
    a = make_prism("a", structured=True)
    b = make_prism("b", structured=False)
    slabs, unstructured = collect_structured_slabs([a, b])
    assert len(slabs) == 1
    assert len(unstructured) == 1
    assert slabs[0].source_index == 0
    assert slabs[0].n_layers == 1
    assert unstructured[0] is b


def test_multi_z_polyprism_emits_one_slab_per_interval():
    # buffers={0:0, 1:0, 2:0} → two intervals (0,1) and (1,2)
    p = PolyPrism(
        polygons=SQ, buffers={0.0: 0.0, 1.0: 0.0, 2.0: 0.0},
        physical_name="multi", structured=True,
    )
    slabs, _ = collect_structured_slabs([p])
    assert len(slabs) == 2
    intervals = sorted([(s.zlo, s.zhi) for s in slabs])
    assert intervals == [(0.0, 1.0), (1.0, 2.0)]
    assert all(s.source_index == 0 for s in slabs)


def test_non_polyprism_structured_raises():
    class FakeStructured:
        structured = True
        physical_name = "fake"
    with pytest.raises(StructuredEntityTypeError):
        collect_structured_slabs([FakeStructured()])


def test_carries_arc_metadata():
    p = make_prism("a", structured=True)
    p.arc_tolerance = 5e-4
    p.min_arc_points = 5
    slabs, _ = collect_structured_slabs([p])
    assert slabs[0].arc_tolerance == 5e-4
    assert slabs[0].min_arc_points == 5
    assert slabs[0].identify_arcs is True
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/structured/test_collect.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `collect.py`**

```python
# meshwell/structured/collect.py
"""Stage 1: gather structured slabs from the input entity list.

A single PolyPrism with N+1 z-boundary keys becomes N StructuredSlab
records (one per consecutive z-pair). Unstructured entities are
returned untouched for cad_occ.
"""
from __future__ import annotations

from typing import Any

from meshwell.polyprism import PolyPrism
from meshwell.structured.exceptions import (
    StructuredEntityTypeError,
    StructuredExtrudeRequiredError,
)
from meshwell.structured.types import StructuredSlab


def collect_structured_slabs(
    entities: list[Any],
) -> tuple[list[StructuredSlab], list[Any]]:
    """Partition the input list.

    Returns:
        (structured_slabs, unstructured_entities). `structured_slabs`
        has one StructuredSlab per (PolyPrism, z-interval) pair.
        Original PolyPrism source_index is preserved on every slab
        derived from it.
    """
    structured: list[StructuredSlab] = []
    unstructured: list[Any] = []
    for idx, ent in enumerate(entities):
        if not getattr(ent, "structured", False):
            unstructured.append(ent)
            continue
        if not isinstance(ent, PolyPrism):
            raise StructuredEntityTypeError(
                entity_index=idx, type_name=type(ent).__name__
            )
        if not ent.extrude:
            raise StructuredExtrudeRequiredError(entity_index=idx)
        z_keys = sorted(ent.buffers.keys())
        for zlo, zhi in zip(z_keys[:-1], z_keys[1:]):
            structured.append(
                StructuredSlab(
                    source_index=idx,
                    footprint=ent.polygons,
                    zlo=zlo,
                    zhi=zhi,
                    n_layers=ent.n_layers,
                    mesh_order=ent.mesh_order,
                    mesh_bool=ent.mesh_bool,
                    physical_name=ent.physical_name,
                    identify_arcs=ent.identify_arcs,
                    arc_tolerance=ent.arc_tolerance,
                    min_arc_points=ent.min_arc_points,
                )
            )
    return structured, unstructured
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/structured/test_collect.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add meshwell/structured/collect.py tests/structured/test_collect.py
git commit -m "feat(structured): Stage 1 (collect) — partition entities into slabs + unstructured"
```

---

## Task 4: `cohort.py` — Stage 2 (Union-Find)

**Files:**
- Create: `meshwell/structured/cohort.py`
- Test: `tests/structured/test_cohort.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/structured/test_cohort.py
from shapely.geometry import Polygon

from meshwell.structured.cohort import build_cohorts
from meshwell.structured.types import StructuredSlab


def slab(idx, poly, zlo, zhi, n_layers=1):
    return StructuredSlab(
        source_index=idx, footprint=poly, zlo=zlo, zhi=zhi,
        n_layers=n_layers, mesh_order=1.0, mesh_bool=True,
        physical_name=("x",), identify_arcs=False,
        arc_tolerance=1e-3, min_arc_points=4,
    )


SQ_A = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
SQ_B = Polygon([(3, 3), (8, 3), (8, 8), (3, 8)])  # overlaps A
SQ_FAR = Polygon([(20, 20), (25, 20), (25, 25), (20, 25)])


def test_disjoint_slabs_yield_separate_cohorts():
    s1 = slab(0, SQ_A, 0, 1)
    s2 = slab(1, SQ_FAR, 0, 1)
    cohorts = build_cohorts([s1, s2])
    assert len(cohorts) == 2


def test_lateral_touch_merges():
    s1 = slab(0, SQ_A, 0, 1)
    s2 = slab(1, SQ_B, 0, 1)  # same z-interval, overlapping XY
    cohorts = build_cohorts([s1, s2])
    assert len(cohorts) == 1
    assert len(cohorts[0].slabs) == 2


def test_face_touch_merges():
    # s1 at [0,1] and s2 at [1,2], share z=1 plane with overlap.
    s1 = slab(0, SQ_A, 0, 1)
    s2 = slab(1, SQ_B, 1, 2)  # shares z=1 with s1, XY overlaps
    cohorts = build_cohorts([s1, s2])
    assert len(cohorts) == 1
    assert cohorts[0].z_planes == (0.0, 1.0, 2.0)


def test_transitive_merge():
    # s1 ←lateral→ s2 ←face→ s3, all merge into one cohort.
    s1 = slab(0, SQ_A, 0, 1)
    s2 = slab(1, SQ_B, 0, 1)        # lateral with s1
    s3 = slab(2, SQ_B, 1, 2)        # face with s2 (same poly)
    cohorts = build_cohorts([s1, s2, s3])
    assert len(cohorts) == 1
    assert {s.source_index for s in cohorts[0].slabs} == {0, 1, 2}


def test_face_touch_requires_xy_overlap():
    s1 = slab(0, SQ_A, 0, 1)
    s2 = slab(1, SQ_FAR, 1, 2)  # same z-plane, no XY overlap
    cohorts = build_cohorts([s1, s2])
    assert len(cohorts) == 2
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/structured/test_cohort.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `cohort.py`**

```python
# meshwell/structured/cohort.py
"""Stage 2: Union-Find over StructuredSlabs.

Two slabs merge if they share a z-plane with XY-overlap (face-touch)
or share a z-interval with XY-overlap (lateral-touch). Output cohorts
are disjoint by construction.
"""
from __future__ import annotations

from meshwell.structured.types import Cohort, StructuredSlab


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _xy_overlaps(a: StructuredSlab, b: StructuredSlab) -> bool:
    # Use intersects (touching boundaries count) — face-touch with shared
    # edge but no interior overlap should still couple cohorts because
    # shared edges become shared OCC edges in the cohort solid.
    inter = a.footprint.intersection(b.footprint)
    return not inter.is_empty


def build_cohorts(slabs: list[StructuredSlab]) -> list[Cohort]:
    """Group slabs into cohorts.

    O(N²) is fine here — cohort detection runs once per generate_mesh
    call on the structured subset (~tens of slabs typical).
    """
    n = len(slabs)
    if n == 0:
        return []
    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = slabs[i], slabs[j]
            # Lateral-touch: same z-interval, XY overlap.
            same_interval = (a.zlo == b.zlo) and (a.zhi == b.zhi)
            # Face-touch: share a z-plane (top-of-a == bot-of-b or vice
            # versa), XY overlap.
            face_touch = (a.zhi == b.zlo) or (b.zhi == a.zlo)
            if (same_interval or face_touch) and _xy_overlaps(a, b):
                uf.union(i, j)

    groups: dict[int, list[StructuredSlab]] = {}
    for i, s in enumerate(slabs):
        groups.setdefault(uf.find(i), []).append(s)

    cohorts: list[Cohort] = []
    for members in groups.values():
        z_planes = tuple(sorted({m.zlo for m in members} | {m.zhi for m in members}))
        cohorts.append(Cohort(slabs=tuple(members), z_planes=z_planes))
    return cohorts
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/structured/test_cohort.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add meshwell/structured/cohort.py tests/structured/test_cohort.py
git commit -m "feat(structured): Stage 2 (cohort) — Union-Find on face/lateral touch"
```

---

## Task 5: `validators.py` — z-stack validator (Stage 3a)

**Files:**
- Create: `meshwell/structured/validators.py`
- Test: `tests/structured/test_validators_zstack.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/structured/test_validators_zstack.py
import pytest
from shapely.geometry import Polygon

from meshwell.polyprism import PolyPrism
from meshwell.structured.cohort import build_cohorts
from meshwell.structured.collect import collect_structured_slabs
from meshwell.structured.exceptions import StructuredZStackError
from meshwell.structured.validators import validate_z_stacks


SQ = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
FAR = Polygon([(100, 100), (110, 100), (110, 110), (100, 110)])


def s(name, zlo, zhi, poly=SQ, structured=False):
    return PolyPrism(
        polygons=poly, buffers={zlo: 0.0, zhi: 0.0},
        physical_name=name, structured=structured,
    )


def test_clean_stack_passes():
    cohort_slab = s("c", 0, 1, structured=True)
    cap = s("cap", 1, 2)  # zlo=1 coincides with cohort zhi
    floor = s("floor", -1, 0)  # zhi=0 coincides with cohort zlo
    entities = [cohort_slab, cap, floor]
    slabs, _ = collect_structured_slabs(entities)
    cohorts = build_cohorts(slabs)
    # should not raise
    validate_z_stacks(cohorts, entities)


def test_mid_height_zlo_violates():
    cohort_slab = s("c", 0, 2, structured=True)
    bad = s("bad", 1, 3)  # zlo=1 strictly inside cohort
    entities = [cohort_slab, bad]
    slabs, _ = collect_structured_slabs(entities)
    cohorts = build_cohorts(slabs)
    with pytest.raises(StructuredZStackError) as exc:
        validate_z_stacks(cohorts, entities)
    assert exc.value.z == 1.0
    assert exc.value.entity_index == 1


def test_mid_height_zhi_violates():
    cohort_slab = s("c", 0, 2, structured=True)
    bad = s("bad", -1, 1)  # zhi=1 strictly inside cohort
    entities = [cohort_slab, bad]
    slabs, _ = collect_structured_slabs(entities)
    cohorts = build_cohorts(slabs)
    with pytest.raises(StructuredZStackError):
        validate_z_stacks(cohorts, entities)


def test_mid_height_no_xy_overlap_allowed():
    cohort_slab = s("c", 0, 2, structured=True)
    far_cap = s("far", 1, 3, poly=FAR)  # mid-height but XY-disjoint
    entities = [cohort_slab, far_cap]
    slabs, _ = collect_structured_slabs(entities)
    cohorts = build_cohorts(slabs)
    validate_z_stacks(cohorts, entities)  # no raise


def test_multi_slab_cohort_z_plane_is_legal():
    # cohort has z-planes {0, 1, 2}; an unstructured cap at z=[1,3]
    # has zlo=1 which IS a cohort z-plane → no raise.
    bot = s("bot", 0, 1, structured=True)
    top = s("top", 1, 2, structured=True)
    cap = s("cap", 1, 3)
    entities = [bot, top, cap]
    slabs, _ = collect_structured_slabs(entities)
    cohorts = build_cohorts(slabs)
    validate_z_stacks(cohorts, entities)
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/structured/test_validators_zstack.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the validator**

```python
# meshwell/structured/validators.py
"""Pipeline validators that may raise.

Stage 3a — z-stack: every entity's z-boundary that lands inside a
cohort z-range must coincide with one of the cohort's own z-planes.

Post-BOP shell invariance is in Task 13.
"""
from __future__ import annotations

from typing import Any

from meshwell.polyprism import PolyPrism
from meshwell.structured.exceptions import StructuredZStackError
from meshwell.structured.types import Cohort


def _entity_z_range(ent: Any) -> tuple[float, float] | None:
    """Return (zmin, zmax) for an entity that has identifiable z-extent."""
    if isinstance(ent, PolyPrism):
        if ent.extrude:
            return (ent.zmin, ent.zmax)
        zs = sorted(ent.buffers.keys())
        return (zs[0], zs[-1])
    if hasattr(ent, "zmin") and hasattr(ent, "zmax"):
        return (ent.zmin, ent.zmax)
    return None


def _entity_z_boundaries(ent: Any) -> list[float]:
    """List of z-boundaries this entity introduces."""
    if isinstance(ent, PolyPrism):
        if ent.extrude:
            return [ent.zmin, ent.zmax]
        return sorted(ent.buffers.keys())
    return []


def _entity_xy_at(ent: Any, z: float):
    """Shapely geometry of entity's footprint at z, or None if no overlap."""
    if isinstance(ent, PolyPrism) and ent.extrude:
        if ent.zmin <= z <= ent.zmax:
            return ent.polygons
    return None


def validate_z_stacks(cohorts: list[Cohort], entities: list[Any]) -> None:
    """Stage 3a — raise on any mid-height z-boundary intersecting a cohort.

    For every entity's zlo/zhi, check each cohort: if the z lies
    strictly inside the cohort's z-range AND the entity's XY at that
    z intersects the cohort's XY at that z, raise unless z coincides
    with one of the cohort's own z-planes.
    """
    for cohort_idx, cohort in enumerate(cohorts):
        cohort_z_set = set(cohort.z_planes)
        for ent_idx, ent in enumerate(entities):
            ent_z_range = _entity_z_range(ent)
            if ent_z_range is None:
                continue
            for z in _entity_z_boundaries(ent):
                if not (cohort.zmin < z < cohort.zmax):
                    continue
                if z in cohort_z_set:
                    continue
                ent_xy = _entity_xy_at(ent, z)
                if ent_xy is None:
                    continue
                cohort_xy = _cohort_xy_at(cohort, z)
                if cohort_xy.intersects(ent_xy):
                    raise StructuredZStackError(
                        entity_index=ent_idx, z=z, cohort_index=cohort_idx
                    )


def _cohort_xy_at(cohort: Cohort, z: float):
    """Union of cohort slab footprints whose z-interval covers z."""
    from shapely.ops import unary_union
    polys = [
        s.footprint for s in cohort.slabs if s.zlo <= z <= s.zhi
    ]
    return unary_union(polys)
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/structured/test_validators_zstack.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add meshwell/structured/validators.py tests/structured/test_validators_zstack.py
git commit -m "feat(structured): Stage 3a — z-stack validator"
```

---

## Task 6: `decompose.py` — per-z-interval footprint + Policy B (Stage 3b/3c)

**Files:**
- Create: `meshwell/structured/decompose.py`
- Test: `tests/structured/test_decompose_footprint.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/structured/test_decompose_footprint.py
from shapely.geometry import Polygon

from meshwell.structured.decompose import zinterval_footprint
from meshwell.structured.types import StructuredSlab


def slab(idx, poly, mesh_bool=True, mesh_order=1.0):
    return StructuredSlab(
        source_index=idx, footprint=poly, zlo=0.0, zhi=1.0,
        n_layers=1, mesh_order=mesh_order, mesh_bool=mesh_bool,
        physical_name=("x",), identify_arcs=False,
        arc_tolerance=1e-3, min_arc_points=4,
    )


SQ_BIG = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
SQ_SMALL = Polygon([(2, 2), (5, 2), (5, 5), (2, 5)])


def test_single_slab_returns_its_footprint():
    fp = zinterval_footprint([slab(0, SQ_BIG)])
    assert fp.equals(SQ_BIG)


def test_two_slabs_union_when_same_mesh_order():
    fp = zinterval_footprint([slab(0, SQ_BIG), slab(1, SQ_SMALL)])
    assert fp.equals(SQ_BIG.union(SQ_SMALL))


def test_lower_mesh_order_carves_higher():
    # SQ_BIG mesh_order=2 (higher), SQ_SMALL mesh_order=1 (lower wins).
    fp = zinterval_footprint([
        slab(0, SQ_BIG, mesh_order=2.0),
        slab(1, SQ_SMALL, mesh_order=1.0),
    ])
    # Lower mesh_order keeps full footprint; higher gets (big − small).
    assert fp.equals(SQ_BIG)  # union of (SQ_SMALL) + (SQ_BIG − SQ_SMALL)


def test_void_subtracts():
    # mesh_bool=False should subtract from the footprint.
    fp = zinterval_footprint([
        slab(0, SQ_BIG, mesh_order=1.0),
        slab(1, SQ_SMALL, mesh_order=2.0, mesh_bool=False),
    ])
    # SQ_BIG processed first (lower mesh_order, keeps full).
    # SQ_SMALL has mesh_bool=False → subtract from accumulated.
    assert fp.equals(SQ_BIG.difference(SQ_SMALL))
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/structured/test_decompose_footprint.py -v
```
Expected: FAIL.

- [ ] **Step 3: Write the function**

```python
# meshwell/structured/decompose.py
"""Stage 3 — cohort decomposition in shapely.

Stage 3b: collect z-planes (just the cohort's own slab boundaries
after the 3a validator ran).
Stage 3c: per-z-interval footprint with Policy B carving.
Stage 3d: bidirectional pre-cut at shared z-planes (Task 7).
"""
from __future__ import annotations

from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union

from meshwell.structured.types import StructuredSlab


def zinterval_footprint(slabs_here: list[StructuredSlab]):
    """Resolve Policy B carving for one z-interval.

    Sort slabs by (mesh_order, source_index) ascending; lower wins.
    For mesh_bool=True: union (footprint − accumulated).
    For mesh_bool=False (void): subtract footprint from accumulated.
    """
    ordered = sorted(
        slabs_here,
        key=lambda s: (
            s.mesh_order if s.mesh_order is not None else float("inf"),
            s.source_index,
        ),
    )
    acc = Polygon()  # empty
    for s in ordered:
        if s.mesh_bool:
            new = s.footprint.difference(acc)
            acc = unary_union([acc, new])
        else:
            acc = acc.difference(s.footprint)
    return acc
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/structured/test_decompose_footprint.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add meshwell/structured/decompose.py tests/structured/test_decompose_footprint.py
git commit -m "feat(structured): Stage 3c — per-z-interval footprint with Policy B carving"
```

---

## Task 7: `decompose.py` — bidirectional pre-cut (Stage 3d)

**Files:**
- Modify: `meshwell/structured/decompose.py`
- Test: `tests/structured/test_decompose_precut.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/structured/test_decompose_precut.py
from shapely.geometry import MultiPolygon, Polygon

from meshwell.polyprism import PolyPrism
from meshwell.structured.cohort import build_cohorts
from meshwell.structured.collect import collect_structured_slabs
from meshwell.structured.decompose import decompose_cohorts
from meshwell.structured.types import SubPiece


SQ_BIG = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
SQ_SMALL = Polygon([(2, 2), (5, 2), (5, 5), (2, 5)])


def make(poly, zlo, zhi, name, structured=False):
    return PolyPrism(
        polygons=poly, buffers={zlo: 0.0, zhi: 0.0},
        physical_name=name, structured=structured,
    )


def test_simple_cohort_one_subpiece_per_interval():
    structured = make(SQ_BIG, 0, 1, "s", structured=True)
    slabs, unstr = collect_structured_slabs([structured])
    cohorts = build_cohorts(slabs)
    subpieces_per_cohort, pre_cut_unstr = decompose_cohorts(cohorts, unstr)
    assert len(subpieces_per_cohort) == 1
    subpieces = subpieces_per_cohort[0]
    assert len(subpieces) == 1
    assert subpieces[0].z_interval == (0.0, 1.0)
    assert subpieces[0].sub_polygon.equals(SQ_BIG)


def test_stepped_cohort_creates_frame_and_center():
    a = make(SQ_BIG, 0, 1, "a", structured=True)
    b = make(SQ_SMALL, 1, 2, "b", structured=True)
    slabs, unstr = collect_structured_slabs([a, b])
    cohorts = build_cohorts(slabs)
    subpieces_per_cohort, _ = decompose_cohorts(cohorts, unstr)
    assert len(subpieces_per_cohort) == 1
    subs = subpieces_per_cohort[0]
    intervals = sorted({sp.z_interval for sp in subs})
    assert intervals == [(0.0, 1.0), (1.0, 2.0)]
    # [0,1] interval split into 2 subpieces by B's outline.
    lower = [sp for sp in subs if sp.z_interval == (0.0, 1.0)]
    assert len(lower) == 2
    upper = [sp for sp in subs if sp.z_interval == (1.0, 2.0)]
    assert len(upper) == 1


def test_unstructured_above_gets_pre_cut():
    # Cohort = SQ_SMALL at [0,1]; unstructured cap = SQ_BIG at [1,2].
    structured = make(SQ_SMALL, 0, 1, "s", structured=True)
    cap = make(SQ_BIG, 1, 2, "cap")
    slabs, unstr = collect_structured_slabs([structured, cap])
    cohorts = build_cohorts(slabs)
    _, pre_cut_unstr = decompose_cohorts(cohorts, unstr)
    assert len(pre_cut_unstr) == 1
    pre_cut_cap = pre_cut_unstr[0]
    # Cap's polygons attribute should now be MultiPolygon with 2 parts
    # (SQ_SMALL inside + SQ_BIG − SQ_SMALL outside).
    polys = pre_cut_cap.polygons
    assert isinstance(polys, MultiPolygon)
    assert len(polys.geoms) == 2
    # physical_name preserved
    assert pre_cut_cap.physical_name == ("cap",)


def test_unstructured_not_touching_cohort_unchanged():
    structured = make(SQ_SMALL, 0, 1, "s", structured=True)
    far = make(Polygon([(50, 50), (60, 50), (60, 60), (50, 60)]), 1, 2, "far")
    slabs, unstr = collect_structured_slabs([structured, far])
    cohorts = build_cohorts(slabs)
    _, pre_cut_unstr = decompose_cohorts(cohorts, unstr)
    assert len(pre_cut_unstr) == 1
    # Polygon untouched.
    assert pre_cut_unstr[0].polygons.equals(far.polygons)
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/structured/test_decompose_precut.py -v
```
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Extend `decompose.py`**

Add to `meshwell/structured/decompose.py`:

```python
from copy import copy
from typing import Any

from shapely.ops import polygonize, unary_union

from meshwell.polyprism import PolyPrism
from meshwell.structured.types import Cohort, SubPiece


def decompose_cohorts(
    cohorts: list[Cohort],
    unstructured_entities: list[Any],
) -> tuple[list[list[SubPiece]], list[Any]]:
    """Stage 3 driver.

    Returns:
        - subpieces_per_cohort: parallel to `cohorts`.
        - pre_cut_unstructured: same order as input. PolyPrisms that
          share a z-plane with any cohort are returned as shallow
          copies with their `polygons` replaced by a MultiPolygon
          decomposed to match cohort sub-faces.
    """
    # 1. Per-cohort, per-z-interval footprint.
    per_cohort_per_interval_footprint: dict[int, dict[tuple[float, float], Polygon]] = {}
    for ci, cohort in enumerate(cohorts):
        per_cohort_per_interval_footprint[ci] = {}
        for zlo, zhi in zip(cohort.z_planes[:-1], cohort.z_planes[1:]):
            slabs_here = [s for s in cohort.slabs if s.zlo <= zlo and s.zhi >= zhi]
            fp = zinterval_footprint(slabs_here)
            per_cohort_per_interval_footprint[ci][(zlo, zhi)] = fp

    # 2. Build cut_sources[z] = union of cohort-side and unstructured-side
    # XY outlines at z. This is the symmetric data both sides will use.
    cut_sources: dict[float, list] = {}
    for ci, cohort in enumerate(cohorts):
        for (zlo, zhi), fp in per_cohort_per_interval_footprint[ci].items():
            cut_sources.setdefault(zlo, []).append(fp.boundary)
            cut_sources.setdefault(zhi, []).append(fp.boundary)
    for ent in unstructured_entities:
        if not isinstance(ent, PolyPrism) or not ent.extrude:
            continue
        z_keys = sorted(ent.buffers.keys())
        for z in (z_keys[0], z_keys[-1]):
            # Only contribute if this entity touches some cohort at z.
            for cohort in cohorts:
                if z not in cohort.z_planes:
                    continue
                cohort_xy_at_z = _cohort_xy_at(cohort, z)
                if cohort_xy_at_z.intersects(ent.polygons):
                    cut_sources.setdefault(z, []).append(ent.polygons.boundary)

    cut_unions = {z: unary_union(lines) for z, lines in cut_sources.items()}

    # 3. Cohort side: emit SubPieces.
    subpieces_per_cohort: list[list[SubPiece]] = []
    for ci, cohort in enumerate(cohorts):
        cohort_subs: list[SubPiece] = []
        for (zlo, zhi), fp in per_cohort_per_interval_footprint[ci].items():
            cuts_zlo = cut_unions.get(zlo)
            cuts_zhi = cut_unions.get(zhi)
            boundaries = [fp.boundary]
            if cuts_zlo is not None:
                boundaries.append(cuts_zlo)
            if cuts_zhi is not None:
                boundaries.append(cuts_zhi)
            merged = unary_union(boundaries)
            pieces = list(polygonize(merged))
            # Filter to pieces whose representative_point lies inside fp.
            inside = [p for p in pieces if fp.contains(p.representative_point())]
            slab_indices = tuple(
                s.source_index for s in cohort.slabs
                if s.zlo <= zlo and s.zhi >= zhi
            )
            for sub_poly in inside:
                cohort_subs.append(
                    SubPiece(
                        cohort_index=ci,
                        z_interval=(zlo, zhi),
                        sub_polygon=sub_poly,
                        source_slab_indices=slab_indices,
                    )
                )
        subpieces_per_cohort.append(cohort_subs)

    # 4. Unstructured side: pre-cut PolyPrisms that touch a cohort z-plane.
    pre_cut: list[Any] = []
    for ent in unstructured_entities:
        if not isinstance(ent, PolyPrism) or not ent.extrude:
            pre_cut.append(ent)
            continue
        touched_zs = [
            z for z in (ent.zmin, ent.zmax)
            if any(z in c.z_planes for c in cohorts)
            and z in cut_unions
        ]
        if not touched_zs:
            pre_cut.append(ent)
            continue
        # Pre-cut ent.polygons at each touched z.
        all_cuts = unary_union([cut_unions[z] for z in touched_zs])
        merged = unary_union([ent.polygons.boundary, all_cuts])
        pieces = list(polygonize(merged))
        inside = [p for p in pieces if ent.polygons.contains(p.representative_point())]
        if not inside:
            pre_cut.append(ent)
            continue
        new_polys = MultiPolygon(inside) if len(inside) > 1 else inside[0]
        new_ent = copy(ent)
        new_ent.polygons = new_polys
        pre_cut.append(new_ent)

    return subpieces_per_cohort, pre_cut


def _cohort_xy_at(cohort: Cohort, z: float):
    polys = [s.footprint for s in cohort.slabs if s.zlo <= z <= s.zhi]
    return unary_union(polys)
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/structured/test_decompose_precut.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add meshwell/structured/decompose.py tests/structured/test_decompose_precut.py
git commit -m "feat(structured): Stage 3d — bidirectional pre-cut at shared z-planes"
```

---

## Task 8: `build.py` — vertex + edge registries (Stage 4 parts 1–2)

**Files:**
- Create: `meshwell/structured/build.py`
- Test: `tests/structured/test_build_registries.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/structured/test_build_registries.py
import numpy as np
from shapely.geometry import Polygon

from meshwell.structured.build import EdgeRegistry, VertexRegistry


def test_vertex_registry_dedups_within_tolerance():
    reg = VertexRegistry(point_tolerance=1e-3)
    v1 = reg.get_or_create(0.0, 0.0, 0.0)
    v2 = reg.get_or_create(1e-4, 0.0, 0.0)  # within tol → same vertex
    v3 = reg.get_or_create(1.0, 0.0, 0.0)
    assert v1 is v2
    assert v1 is not v3
    assert len(reg) == 2


def test_edge_registry_dedups_xy_edges_at_z():
    vreg = VertexRegistry(point_tolerance=1e-3)
    ereg = EdgeRegistry(vreg, point_tolerance=1e-3)
    coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
    e1 = ereg.polyline_xy(coords, z=0.0, identify_arcs=False)
    e2 = ereg.polyline_xy(coords, z=0.0, identify_arcs=False)
    # Same path at same z → same edge sequence.
    assert e1 == e2


def test_edge_registry_arc_detected():
    vreg = VertexRegistry(point_tolerance=1e-3)
    ereg = EdgeRegistry(vreg, point_tolerance=1e-3)
    # 8-pt sampling of a unit circle.
    n = 16
    coords = [(np.cos(a), np.sin(a)) for a in np.linspace(0, 2 * np.pi, n + 1)]
    edges = ereg.polyline_xy(
        coords, z=0.0, identify_arcs=True,
        min_arc_points=4, arc_tolerance=1e-2,
    )
    # All edges should be arc edges (single full-circle case decomposes
    # to one or more arc segments, depending on seam logic).
    # We at least require: total edge count < n (some were merged into arcs).
    assert len(edges) < n
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/structured/test_build_registries.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the registries**

```python
# meshwell/structured/build.py
"""Stage 4 — assemble cohort TopoDS_Compound of N sub-solids.

Bottom-up build: unique vertices → unique edges (with arc detection)
→ unique faces (horizontal interior/boundary, lateral) → per-subpiece
TopoDS_Solid → TopoDS_Compound per cohort.

Shared TShapes by CONSTRUCTION (not post-hoc sewing): every face and
edge is built once and referenced by every solid that needs it. This
is what makes cohort internal interfaces conformal without BOP.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from OCP.BRep import BRep_Builder
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeVertex,
    BRepBuilderAPI_MakeWire,
)
from OCP.GC import GC_MakeArcOfCircle
from OCP.gp import gp_Ax2, gp_Circ, gp_Dir, gp_Pnt
from OCP.TopoDS import (
    TopoDS_Compound,
    TopoDS_Edge,
    TopoDS_Face,
    TopoDS_Shell,
    TopoDS_Solid,
    TopoDS_Vertex,
)


@dataclass
class VertexRegistry:
    """Snap-and-dedup vertex store.

    Coordinates are quantized to `point_tolerance` so near-coincident
    vertices map to the same TopoDS_Vertex.
    """

    point_tolerance: float

    def __post_init__(self):
        self._store: dict[tuple[int, int, int], TopoDS_Vertex] = {}

    def _key(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        s = self.point_tolerance
        return (round(x / s), round(y / s), round(z / s))

    def get_or_create(self, x: float, y: float, z: float) -> TopoDS_Vertex:
        k = self._key(x, y, z)
        if k not in self._store:
            self._store[k] = BRepBuilderAPI_MakeVertex(gp_Pnt(x, y, z)).Vertex()
        return self._store[k]

    def __len__(self):
        return len(self._store)


@dataclass
class EdgeRegistry:
    """Unique edge store with arc detection.

    Two flavours:
      - polyline_xy: a 2D polyline at fixed z; runs of vertices on a
        circle (when identify_arcs) build a GC_MakeArcOfCircle edge.
      - vertical: a single edge between two z values at one (x,y).
    """

    vertices: VertexRegistry
    point_tolerance: float

    def __post_init__(self):
        self._store: dict[tuple, TopoDS_Edge] = {}

    def vertical(self, x: float, y: float, z_a: float, z_b: float) -> TopoDS_Edge:
        a = self.vertices.get_or_create(x, y, z_a)
        b = self.vertices.get_or_create(x, y, z_b)
        key = ("V", self.vertices._key(x, y, z_a), self.vertices._key(x, y, z_b))
        if key not in self._store:
            self._store[key] = BRepBuilderAPI_MakeEdge(a, b).Edge()
        return self._store[key]

    def line_xy(self, x1: float, y1: float, x2: float, y2: float, z: float) -> TopoDS_Edge:
        a = self.vertices.get_or_create(x1, y1, z)
        b = self.vertices.get_or_create(x2, y2, z)
        k_a = self.vertices._key(x1, y1, z)
        k_b = self.vertices._key(x2, y2, z)
        key = ("L", tuple(sorted([k_a, k_b])))
        if key not in self._store:
            self._store[key] = BRepBuilderAPI_MakeEdge(a, b).Edge()
        return self._store[key]

    def arc_xy(
        self,
        start: tuple[float, float],
        mid: tuple[float, float],
        end: tuple[float, float],
        z: float,
    ) -> TopoDS_Edge:
        sv = self.vertices.get_or_create(*start, z)
        ev = self.vertices.get_or_create(*end, z)
        k_s = self.vertices._key(*start, z)
        k_m = self.vertices._key(*mid, z)
        k_e = self.vertices._key(*end, z)
        key = ("A", k_s, k_m, k_e)
        if key not in self._store:
            arc = GC_MakeArcOfCircle(
                gp_Pnt(start[0], start[1], z),
                gp_Pnt(mid[0], mid[1], z),
                gp_Pnt(end[0], end[1], z),
            ).Value()
            self._store[key] = BRepBuilderAPI_MakeEdge(arc, sv, ev).Edge()
        return self._store[key]

    def polyline_xy(
        self,
        coords: list[tuple[float, float]],
        z: float,
        identify_arcs: bool,
        min_arc_points: int = 4,
        arc_tolerance: float = 1e-3,
    ) -> list[TopoDS_Edge]:
        """Return the list of edges (lines and/or arcs) covering coords.

        Uses the same arc-detection as GeometryEntity.decompose_vertices
        but inlined to avoid pulling that dep just for the 2D case.
        """
        if not identify_arcs or len(coords) < min_arc_points:
            return [
                self.line_xy(coords[i][0], coords[i][1],
                             coords[i + 1][0], coords[i + 1][1], z)
                for i in range(len(coords) - 1)
            ]
        edges: list[TopoDS_Edge] = []
        i, n = 0, len(coords)
        while i < n - 1:
            best = None
            best_j = i + 1
            for j in range(i + min_arc_points, n + 1):
                pts = np.array(coords[i:j])
                cx, cy, r, residual = _fit_circle_2d(pts)
                if residual <= arc_tolerance and r < 1e6:
                    best = (cx, cy, r)
                    best_j = j
                else:
                    break
            if best is not None:
                mid_idx = (i + best_j - 1) // 2
                edges.append(self.arc_xy(
                    coords[i], coords[mid_idx], coords[best_j - 1], z,
                ))
                i = best_j - 1
            else:
                edges.append(self.line_xy(
                    coords[i][0], coords[i][1],
                    coords[i + 1][0], coords[i + 1][1], z,
                ))
                i += 1
        return edges


def _fit_circle_2d(pts: np.ndarray) -> tuple[float, float, float, float]:
    """Algebraic circle fit. Returns (cx, cy, r, residual)."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = float(np.sqrt(c + cx * cx + cy * cy))
    residual = float(np.std(np.hypot(x - cx, y - cy) - r))
    return float(cx), float(cy), r, residual
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/structured/test_build_registries.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add meshwell/structured/build.py tests/structured/test_build_registries.py
git commit -m "feat(structured): Stage 4 — vertex/edge registries with arc detection"
```

---

## Task 9: `build.py` — face registry + sub-solid assembly

**Files:**
- Modify: `meshwell/structured/build.py`
- Test: `tests/structured/test_build_solid.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/structured/test_build_solid.py
from shapely.geometry import Polygon

from meshwell.polyprism import PolyPrism
from meshwell.structured.build import build_cohort_compound
from meshwell.structured.cohort import build_cohorts
from meshwell.structured.collect import collect_structured_slabs
from meshwell.structured.decompose import decompose_cohorts

from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer


def _count(shape, kind) -> int:
    exp = TopExp_Explorer(shape, kind)
    out = 0
    while exp.More():
        out += 1
        exp.Next()
    return out


SQ = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])


def test_single_subpiece_compound_has_one_solid():
    ent = PolyPrism(polygons=SQ, buffers={0.0: 0.0, 1.0: 0.0},
                    physical_name="s", structured=True)
    slabs, unstr = collect_structured_slabs([ent])
    cohorts = build_cohorts(slabs)
    subs_per_cohort, _ = decompose_cohorts(cohorts, unstr)
    compound, slab_meta = build_cohort_compound(
        cohorts[0], subs_per_cohort[0], point_tolerance=1e-3,
    )
    assert _count(compound, TopAbs_SOLID) == 1
    # 1 bot + 1 top + 4 lateral = 6 faces.
    assert _count(compound, TopAbs_FACE) == 6
    assert len(slab_meta) == 1


def test_two_stacked_subpieces_share_interior_face():
    a = PolyPrism(polygons=SQ, buffers={0.0: 0.0, 1.0: 0.0},
                  physical_name="a", structured=True)
    b = PolyPrism(polygons=SQ, buffers={1.0: 0.0, 2.0: 0.0},
                  physical_name="b", structured=True)
    slabs, unstr = collect_structured_slabs([a, b])
    cohorts = build_cohorts(slabs)
    subs_per_cohort, _ = decompose_cohorts(cohorts, unstr)
    compound, slab_meta = build_cohort_compound(
        cohorts[0], subs_per_cohort[0], point_tolerance=1e-3,
    )
    assert _count(compound, TopAbs_SOLID) == 2
    # 1 bot (z=0) + 1 top (z=2) + 1 interior (z=1) + 8 laterals = 11.
    assert _count(compound, TopAbs_FACE) == 11
    assert len(slab_meta) == 2
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/structured/test_build_solid.py -v
```
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Extend `build.py`**

Add to `meshwell/structured/build.py`:

```python
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.gp import gp_Vec

from meshwell.structured.types import (
    Cohort, ShapeKey, SlabMeta, StructuredSlab, SubPiece,
)


def _shape_key(shape) -> ShapeKey:
    from OCP.TopTools import TopTools_ShapeMapHasher
    hasher = TopTools_ShapeMapHasher()
    return ShapeKey(tshape_id=hasher(shape), orientation=int(shape.Orientation()))


def _ring_coords(ring) -> list[tuple[float, float]]:
    return list(ring.coords)


def _build_horizontal_face(
    polygon, z: float, ereg: EdgeRegistry, identify_arcs: bool,
    min_arc_points: int, arc_tolerance: float,
) -> TopoDS_Face:
    """Build a horizontal TopoDS_Face for a polygon at fixed z."""
    outer_coords = _ring_coords(polygon.exterior)
    outer_edges = ereg.polyline_xy(
        outer_coords, z, identify_arcs, min_arc_points, arc_tolerance,
    )
    mw = BRepBuilderAPI_MakeWire()
    for e in outer_edges:
        mw.Add(e)
    outer_wire = mw.Wire()
    mf = BRepBuilderAPI_MakeFace(outer_wire)
    for interior in polygon.interiors:
        hole_coords = _ring_coords(interior)
        hole_edges = ereg.polyline_xy(
            hole_coords, z, identify_arcs, min_arc_points, arc_tolerance,
        )
        mw_h = BRepBuilderAPI_MakeWire()
        for e in hole_edges:
            mw_h.Add(e)
        mf.Add(mw_h.Wire())
    return mf.Face()


def _build_lateral_face(
    edge_xy_low: TopoDS_Edge,
    edge_xy_high: TopoDS_Edge,
    v_left: TopoDS_Edge,
    v_right: TopoDS_Edge,
) -> TopoDS_Face:
    """Stitch four edges into a lateral quad face."""
    mw = BRepBuilderAPI_MakeWire()
    mw.Add(edge_xy_low)
    mw.Add(v_right)
    mw.Add(edge_xy_high)
    mw.Add(v_left)
    wire = mw.Wire()
    return BRepBuilderAPI_MakeFace(wire).Face()


def build_cohort_compound(
    cohort: Cohort,
    subpieces: list[SubPiece],
    point_tolerance: float,
) -> tuple[TopoDS_Compound, dict[ShapeKey, SlabMeta]]:
    """Stage 4 driver — assemble compound + slab_meta.

    The compound is one TopoDS_Compound containing N sub-solids, where
    N == len(subpieces). Faces and edges shared between sub-solids are
    constructed once and referenced from both — guaranteeing shared
    TShapes without BOP.
    """
    vreg = VertexRegistry(point_tolerance=point_tolerance)
    ereg = EdgeRegistry(vertices=vreg, point_tolerance=point_tolerance)

    # Build a horizontal face per (subpiece, z_low|z_high) — but reuse
    # if another subpiece at the same z with the same polygon ID needs
    # one. Key on (z, id(sub_polygon)) since subpieces' sub_polygons are
    # distinct shapely objects even when geometry overlaps.
    # For simplicity in v1: every subpiece gets its own bot/top face,
    # and we rely on the unique edge/vertex registry to share TShapes
    # for the boundary. Interior horizontal faces (where two
    # z-adjacent subpieces share polygon area) are detected and
    # built as ONE shared face.
    slab_by_source: dict[int, StructuredSlab] = {
        s.source_index: s for s in cohort.slabs
    }

    # First pass: identify shared horizontal interior faces.
    # Two subpieces share an interior face when their z_intervals are
    # adjacent (one's zhi == other's zlo) and their sub_polygons
    # intersect with non-zero area.
    sub_idx_by_z: dict[float, list[int]] = {}
    for i, sp in enumerate(subpieces):
        sub_idx_by_z.setdefault(sp.z_interval[0], []).append(i)
        sub_idx_by_z.setdefault(sp.z_interval[1], []).append(i)

    shared_horizontal: dict[tuple[int, int], "Polygon"] = {}
    for z in sorted(sub_idx_by_z.keys()):
        below = [i for i in sub_idx_by_z[z] if subpieces[i].z_interval[1] == z]
        above = [i for i in sub_idx_by_z[z] if subpieces[i].z_interval[0] == z]
        for b in below:
            for a in above:
                inter = subpieces[b].sub_polygon.intersection(
                    subpieces[a].sub_polygon
                )
                if inter.area > 0:
                    shared_horizontal[(b, a)] = inter

    # Cache built horizontal faces by (subpiece_idx, side) for quick lookup.
    # side: "bot" or "top".
    horiz_faces: dict[tuple[int, str], TopoDS_Face] = {}
    arc_params_for = lambda sp_idx: (
        slab_by_source[subpieces[sp_idx].source_slab_indices[0]].identify_arcs,
        slab_by_source[subpieces[sp_idx].source_slab_indices[0]].min_arc_points,
        slab_by_source[subpieces[sp_idx].source_slab_indices[0]].arc_tolerance,
    )

    # Build shared interior faces first.
    for (b, a), inter_poly in shared_horizontal.items():
        z = subpieces[b].z_interval[1]
        id_arcs, min_p, arc_tol = arc_params_for(b)
        face = _build_horizontal_face(inter_poly, z, ereg, id_arcs, min_p, arc_tol)
        horiz_faces[(b, "top")] = face
        horiz_faces[(a, "bot")] = face

    # Build remaining bot/top faces (those that weren't shared).
    for i, sp in enumerate(subpieces):
        id_arcs, min_p, arc_tol = arc_params_for(i)
        if (i, "bot") not in horiz_faces:
            horiz_faces[(i, "bot")] = _build_horizontal_face(
                sp.sub_polygon, sp.z_interval[0], ereg, id_arcs, min_p, arc_tol,
            )
        if (i, "top") not in horiz_faces:
            horiz_faces[(i, "top")] = _build_horizontal_face(
                sp.sub_polygon, sp.z_interval[1], ereg, id_arcs, min_p, arc_tol,
            )

    # Build lateral faces per subpiece. Each polygon-edge of the
    # subpiece's sub_polygon becomes one lateral face. The edge
    # registry shares vertical edges between laterally-adjacent
    # subpieces in the same z-interval automatically.
    lateral_faces: dict[int, list[TopoDS_Face]] = {i: [] for i in range(len(subpieces))}
    for i, sp in enumerate(subpieces):
        id_arcs, min_p, arc_tol = arc_params_for(i)
        zlo, zhi = sp.z_interval
        coords = _ring_coords(sp.sub_polygon.exterior)
        for a, b in zip(coords[:-1], coords[1:]):
            # Build the bottom and top XY edges (line only for laterals
            # — we use polyline_xy with single segment to dedupe via
            # line_xy under the hood).
            e_lo = ereg.line_xy(a[0], a[1], b[0], b[1], zlo)
            e_hi = ereg.line_xy(a[0], a[1], b[0], b[1], zhi)
            v_left = ereg.vertical(a[0], a[1], zlo, zhi)
            v_right = ereg.vertical(b[0], b[1], zlo, zhi)
            lateral_faces[i].append(
                _build_lateral_face(e_lo, e_hi, v_left, v_right)
            )

    # Build each sub-solid.
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    slab_meta: dict[ShapeKey, SlabMeta] = {}
    for i, sp in enumerate(subpieces):
        shell = TopoDS_Shell()
        builder.MakeShell(shell)
        bot = horiz_faces[(i, "bot")]
        top = horiz_faces[(i, "top")]
        laterals = lateral_faces[i]
        for f in [bot, top, *laterals]:
            builder.Add(shell, f)
        solid = TopoDS_Solid()
        builder.MakeSolid(solid)
        builder.Add(solid, shell)
        builder.Add(compound, solid)
        source_slab = slab_by_source[sp.source_slab_indices[0]]
        slab_meta[_shape_key(solid)] = SlabMeta(
            slab_index=source_slab.source_index,
            n_layers=source_slab.n_layers,
            physical_name=source_slab.physical_name,
            bot_face_key=_shape_key(bot),
            top_face_key=_shape_key(top),
            lateral_face_keys=tuple(_shape_key(f) for f in laterals),
        )
    return compound, slab_meta
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/structured/test_build_solid.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add meshwell/structured/build.py tests/structured/test_build_solid.py
git commit -m "feat(structured): Stage 4 — face registry + sub-solid + cohort compound"
```

---

## Task 10: `cohort_entity.py` — `_CohortEntity` wrapper

**Files:**
- Create: `meshwell/structured/cohort_entity.py`
- Test: `tests/structured/test_cohort_entity.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/structured/test_cohort_entity.py
from shapely.geometry import Polygon

from meshwell.polyprism import PolyPrism
from meshwell.structured.build import build_cohort_compound
from meshwell.structured.cohort import build_cohorts
from meshwell.structured.cohort_entity import _CohortEntity
from meshwell.structured.collect import collect_structured_slabs
from meshwell.structured.decompose import decompose_cohorts


SQ = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])


def test_cohort_entity_min_mesh_order():
    a = PolyPrism(polygons=SQ, buffers={0.0: 0.0, 1.0: 0.0},
                  physical_name="a", structured=True, mesh_order=3.0)
    b = PolyPrism(polygons=SQ, buffers={1.0: 0.0, 2.0: 0.0},
                  physical_name="b", structured=True, mesh_order=1.0)
    slabs, unstr = collect_structured_slabs([a, b])
    cohorts = build_cohorts(slabs)
    subs, _ = decompose_cohorts(cohorts, unstr)
    compound, slab_meta = build_cohort_compound(
        cohorts[0], subs[0], point_tolerance=1e-3,
    )
    ent = _CohortEntity(compound=compound, slab_meta=slab_meta, cohort=cohorts[0])
    assert ent.mesh_order == 1.0  # min across slabs
    assert ent.mesh_bool is True
    assert ent.dimension == 3
    assert ent.physical_name == ("__cohort_0",)
    assert ent.instanciate_occ() is compound
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/structured/test_cohort_entity.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the wrapper**

```python
# meshwell/structured/cohort_entity.py
"""Wrapper exposing a cohort compound as a cad_occ-compatible entity.

The cohort compound enters cad_occ as ONE entity (one BOP argument).
The cad_occ post-pass later expands it back into per-sub-solid
OCCLabeledEntity records via slab_meta.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from OCP.TopoDS import TopoDS_Compound

from meshwell.structured.types import Cohort, ShapeKey, SlabMeta


@dataclass
class _CohortEntity:
    """Behaves like a cad_occ entity (has instanciate_occ, mesh_order,
    physical_name, mesh_bool, dimension) but unwraps to N sub-solids.

    `slab_meta` is the dict the post-pass uses to recover per-slab
    physical_name + n_layers from each surviving post-BOP sub-solid.
    """

    compound: TopoDS_Compound
    slab_meta: dict[ShapeKey, SlabMeta]
    cohort: Cohort
    cohort_index: int = 0

    dimension: int = field(init=False, default=3)
    mesh_bool: bool = field(init=False, default=True)

    @property
    def mesh_order(self) -> float:
        return min(
            (s.mesh_order if s.mesh_order is not None else float("inf"))
            for s in self.cohort.slabs
        )

    @property
    def physical_name(self) -> tuple[str, ...]:
        return (f"__cohort_{self.cohort_index}",)

    def instanciate_occ(self) -> TopoDS_Compound:
        return self.compound
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/structured/test_cohort_entity.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add meshwell/structured/cohort_entity.py tests/structured/test_cohort_entity.py
git commit -m "feat(structured): _CohortEntity wrapper for cad_occ integration"
```

---

## Task 11: `cad_occ.py` — unwrap compound at instantiation

**Files:**
- Modify: `meshwell/cad_occ.py:272-292` (`_instantiate_entity_occ`)
- Test: `tests/structured/test_cad_occ_unwrap.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/structured/test_cad_occ_unwrap.py
from OCP.BRep import BRep_Builder
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.TopoDS import TopoDS_Compound

from meshwell.cad_occ import CAD_OCC


class _StubCompoundEntity:
    """Entity whose instanciate_occ returns a compound of two solids."""
    dimension = 3
    mesh_order = 1.0
    mesh_bool = True
    physical_name = ("stub",)

    def instanciate_occ(self):
        b = BRep_Builder()
        c = TopoDS_Compound()
        b.MakeCompound(c)
        s1 = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Solid()
        s2 = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Solid()
        b.Add(c, s1)
        b.Add(c, s2)
        return c


def test_compound_flattens_to_constituent_solids():
    proc = CAD_OCC()
    labeled = proc._instantiate_entity_occ(0, _StubCompoundEntity())
    assert len(labeled.shapes) == 2
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/structured/test_cad_occ_unwrap.py -v
```
Expected: FAIL (current `_instantiate_entity_occ` wraps in `[shape]`).

- [ ] **Step 3: Modify `_instantiate_entity_occ`**

In `meshwell/cad_occ.py`, locate `_instantiate_entity_occ` (around line 272) and replace its body so it unwraps compounds at instantiation time:

```python
    def _instantiate_entity_occ(
        self,
        index: int,
        entity_obj: Any,
    ) -> OCCLabeledEntity:
        """Instantiate a single entity into an OCC shape.

        Compounds are flattened to their constituent dim-level
        sub-shapes so BOPAlgo_Builder.Modified() in fragment_all
        can track per-sub-shape provenance. Required for the
        structured pipeline's cohort compound (multiple sub-solids
        per cohort).
        """
        shape = entity_obj.instanciate_occ()
        dim = getattr(entity_obj, "dimension", None)
        if dim is None:
            dim = self._get_shape_dimension(shape)
        physical_name = entity_obj.physical_name
        if isinstance(physical_name, str):
            physical_name = (physical_name,)
        shapes = (
            self._unwrap_shape(shape, dim) if shape is not None else []
        )
        return OCCLabeledEntity(
            shapes=shapes,
            physical_name=physical_name,
            index=index,
            keep=getattr(entity_obj, "mesh_bool", True),
            dim=dim,
            mesh_order=getattr(entity_obj, "mesh_order", None),
        )
```

- [ ] **Step 4: Run test to verify it passes + regressions**

```
pytest tests/structured/test_cad_occ_unwrap.py -v
pytest tests/test_cad_occ.py tests/test_cad_occ_fragment_ownership.py -v
```
Expected: PASS on all.

- [ ] **Step 5: Commit**

```
git add meshwell/cad_occ.py tests/structured/test_cad_occ_unwrap.py
git commit -m "feat(cad_occ): unwrap compound shapes at instantiation"
```

---

## Task 12: Shell-invariance validator (post-pass)

**Files:**
- Modify: `meshwell/structured/validators.py`
- Modify: `meshwell/cad_occ.py` — expose post-BOP shape map
- Test: `tests/structured/test_validators_shell.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/structured/test_validators_shell.py
import pytest
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

from meshwell.structured.exceptions import CohortShellModifiedError
from meshwell.structured.types import ShapeKey, SlabMeta
from meshwell.structured.validators import validate_cohort_shells


def _key_of(shape):
    from OCP.TopTools import TopTools_ShapeMapHasher
    h = TopTools_ShapeMapHasher()
    return ShapeKey(h(shape), int(shape.Orientation()))


def test_no_changes_passes():
    box = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Solid()
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    exp = TopExp_Explorer(box, TopAbs_FACE)
    faces = []
    while exp.More():
        faces.append(exp.Current())
        exp.Next()
    bot, top, *laterals = faces
    meta = SlabMeta(
        slab_index=0, n_layers=1, physical_name=("x",),
        bot_face_key=_key_of(bot), top_face_key=_key_of(top),
        lateral_face_keys=tuple(_key_of(f) for f in laterals),
    )
    slab_meta = {_key_of(box): meta}

    # Stub builder whose Modified() returns empty + IsDeleted returns False.
    class _StubBuilder:
        def Modified(self, shape):
            from OCP.TopTools import TopTools_ListOfShape
            return TopTools_ListOfShape()
        def IsDeleted(self, shape):
            return False

    # Should not raise.
    validate_cohort_shells(slab_meta, faces_by_key={
        meta.bot_face_key: bot,
        meta.top_face_key: top,
        **{k: f for k, f in zip(meta.lateral_face_keys, laterals)},
    }, builder=_StubBuilder())


def test_split_face_raises():
    box = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Solid()
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    exp = TopExp_Explorer(box, TopAbs_FACE)
    bot = exp.Current()
    meta = SlabMeta(
        slab_index=5, n_layers=1, physical_name=("x",),
        bot_face_key=_key_of(bot), top_face_key=_key_of(bot),
        lateral_face_keys=(),
    )
    slab_meta = {_key_of(box): meta}

    class _SplitBuilder:
        def Modified(self, shape):
            from OCP.TopTools import TopTools_ListOfShape
            lst = TopTools_ListOfShape()
            # Simulate BOP splitting the face into two new faces.
            lst.Append(shape)
            lst.Append(shape)
            return lst
        def IsDeleted(self, shape):
            return False

    with pytest.raises(CohortShellModifiedError) as exc:
        validate_cohort_shells(
            slab_meta, faces_by_key={meta.bot_face_key: bot},
            builder=_SplitBuilder(),
        )
    assert exc.value.slab_index == 5
    assert exc.value.fragment_count == 2
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/structured/test_validators_shell.py -v
```
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add validator to `validators.py`**

Append to `meshwell/structured/validators.py`:

```python
from typing import Iterable

from meshwell.structured.exceptions import CohortShellModifiedError
from meshwell.structured.types import ShapeKey, SlabMeta


def validate_cohort_shells(
    slab_meta: dict[ShapeKey, SlabMeta],
    faces_by_key: dict[ShapeKey, "TopoDS_Face"],
    builder,
) -> None:
    """Stage 5 post-pass — raise if BOP modified any pre-baked cohort
    shell face into more than one piece.

    For each face role (bot/top/lateral) in each SlabMeta, query
    builder.Modified(original_face). Acceptable outcomes:
      - empty + not deleted: face passed through BOP unchanged.
      - single replacement: face merged with a coincident neighbour.
    Unacceptable:
      - multiple replacements: BOP introduced a cut on this shell face.
    """
    for sub_key, meta in slab_meta.items():
        for role, fk in [
            ("bot", meta.bot_face_key),
            ("top", meta.top_face_key),
        ] + [(f"lateral_{i}", lk) for i, lk in enumerate(meta.lateral_face_keys)]:
            face = faces_by_key.get(fk)
            if face is None:
                continue
            modified = builder.Modified(face)
            count = sum(1 for _ in _iterate_list(modified))
            if count > 1:
                raise CohortShellModifiedError(
                    slab_index=meta.slab_index,
                    face_role=role,
                    fragment_count=count,
                )


def _iterate_list(lst) -> Iterable:
    """Iterate a TopTools_ListOfShape."""
    try:
        it = lst.cbegin() if hasattr(lst, "cbegin") else None
    except Exception:
        it = None
    out = []
    n = lst.Extent() if hasattr(lst, "Extent") else 0
    if n == 0:
        return out
    # OCP's TopTools_ListOfShape supports iteration via Iterator class,
    # but the simplest robust path: convert to list via OCP's list helper.
    try:
        for shape in lst:  # OCP exposes __iter__ on most list types
            out.append(shape)
    except TypeError:
        # Fallback: use TopTools_ListIteratorOfListOfShape.
        from OCP.TopTools import TopTools_ListIteratorOfListOfShape
        it = TopTools_ListIteratorOfListOfShape(lst)
        while it.More():
            out.append(it.Value())
            it.Next()
    return out
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/structured/test_validators_shell.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add meshwell/structured/validators.py tests/structured/test_validators_shell.py
git commit -m "feat(structured): shell-invariance validator (post-BOP)"
```

---

## Task 13: Structured pre-pass / post-pass driver

**Files:**
- Create: `meshwell/structured/pipeline.py`
- Test: `tests/structured/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/structured/test_pipeline.py
from shapely.geometry import Polygon

from meshwell.polyprism import PolyPrism
from meshwell.structured.pipeline import structured_pre_pass


SQ = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])


def test_pre_pass_replaces_structured_with_cohort_entity():
    a = PolyPrism(polygons=SQ, buffers={0.0: 0.0, 1.0: 0.0},
                  physical_name="a", structured=True)
    b = PolyPrism(polygons=SQ, buffers={1.0: 0.0, 2.0: 0.0},
                  physical_name="b", structured=True)
    cap = PolyPrism(polygons=SQ, buffers={2.0: 0.0, 3.0: 0.0},
                    physical_name="cap")
    state = structured_pre_pass([a, b, cap], point_tolerance=1e-3)
    assert len(state.entities_out) == 2   # one cohort + cap
    from meshwell.structured.cohort_entity import _CohortEntity
    cohort_count = sum(isinstance(e, _CohortEntity) for e in state.entities_out)
    assert cohort_count == 1
    # slab_meta is keyed by ShapeKey of each sub-solid.
    assert len(state.slab_meta) == 2  # one per stacked slab


def test_pre_pass_passthrough_no_structured():
    a = PolyPrism(polygons=SQ, buffers={0.0: 0.0, 1.0: 0.0},
                  physical_name="a")
    state = structured_pre_pass([a], point_tolerance=1e-3)
    assert state.entities_out == [a]
    assert state.slab_meta == {}
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/structured/test_pipeline.py -v
```
Expected: FAIL.

- [ ] **Step 3: Write the driver**

```python
# meshwell/structured/pipeline.py
"""End-to-end driver for the structured pre-pass and post-pass.

Pre-pass: collect → cohort → validate z-stacks → decompose → build →
swap entities. Returns a StructuredState that the orchestrator
threads forward to the cad_occ and meshing stages.

Post-pass: (Task 14) expand cohort OCCLabeledEntity into per-sub-solid
entities + record post-BOP face ShapeKeys.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from meshwell.structured.build import build_cohort_compound
from meshwell.structured.cohort import build_cohorts
from meshwell.structured.cohort_entity import _CohortEntity
from meshwell.structured.collect import collect_structured_slabs
from meshwell.structured.decompose import decompose_cohorts
from meshwell.structured.types import ShapeKey, SlabMeta
from meshwell.structured.validators import validate_z_stacks


@dataclass
class StructuredState:
    """Threaded between pre-pass, cad_occ, and post-pass."""

    entities_out: list[Any]
    slab_meta: dict[ShapeKey, SlabMeta] = field(default_factory=dict)
    cohort_entities: list[_CohortEntity] = field(default_factory=list)


def structured_pre_pass(
    entities: list[Any],
    point_tolerance: float,
) -> StructuredState:
    """Run Stages 1–4 and return entities_out for cad_occ.

    If no structured entities are present, returns the input list
    unchanged with an empty slab_meta.
    """
    structured_slabs, unstructured = collect_structured_slabs(entities)
    if not structured_slabs:
        return StructuredState(entities_out=entities)
    cohorts = build_cohorts(structured_slabs)
    validate_z_stacks(cohorts, entities)
    subpieces_per_cohort, pre_cut_unstr = decompose_cohorts(cohorts, unstructured)

    cohort_entities: list[_CohortEntity] = []
    all_slab_meta: dict[ShapeKey, SlabMeta] = {}
    for ci, (cohort, subs) in enumerate(zip(cohorts, subpieces_per_cohort)):
        compound, slab_meta = build_cohort_compound(cohort, subs, point_tolerance)
        ce = _CohortEntity(
            compound=compound, slab_meta=slab_meta,
            cohort=cohort, cohort_index=ci,
        )
        cohort_entities.append(ce)
        all_slab_meta.update(slab_meta)

    entities_out = cohort_entities + pre_cut_unstr
    return StructuredState(
        entities_out=entities_out,
        slab_meta=all_slab_meta,
        cohort_entities=cohort_entities,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/structured/test_pipeline.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add meshwell/structured/pipeline.py tests/structured/test_pipeline.py
git commit -m "feat(structured): pre-pass driver (collect → cohort → validate → decompose → build)"
```

---

## Task 14: `cad_occ.py` — accept post-pass hook for cohort expansion

**Files:**
- Modify: `meshwell/cad_occ.py` (`_fragment_all`, `process_entities`)
- Modify: `meshwell/structured/pipeline.py` — add `structured_post_pass`
- Test: `tests/structured/test_post_pass_expansion.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/structured/test_post_pass_expansion.py
from shapely.geometry import Polygon

from meshwell.cad_occ import cad_occ
from meshwell.polyprism import PolyPrism
from meshwell.structured.pipeline import structured_post_pass, structured_pre_pass


SQ = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])


def test_cohort_expands_to_per_slab_entities():
    a = PolyPrism(polygons=SQ, buffers={0.0: 0.0, 1.0: 0.0},
                  physical_name="a", structured=True)
    b = PolyPrism(polygons=SQ, buffers={1.0: 0.0, 2.0: 0.0},
                  physical_name="b", structured=True)
    state = structured_pre_pass([a, b], point_tolerance=1e-3)
    occ_entities = cad_occ(state.entities_out)
    final = structured_post_pass(occ_entities, state)
    physical_names = {e.physical_name for e in final}
    assert ("a",) in physical_names
    assert ("b",) in physical_names
    assert not any(pn[0].startswith("__cohort_") for pn in physical_names)
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/structured/test_post_pass_expansion.py -v
```
Expected: FAIL.

- [ ] **Step 3: Add `structured_post_pass` to `pipeline.py`**

Append to `meshwell/structured/pipeline.py`:

```python
from copy import copy as _copy

from meshwell.cad_occ import OCCLabeledEntity


def structured_post_pass(
    occ_entities: list[OCCLabeledEntity],
    state: StructuredState,
) -> list[OCCLabeledEntity]:
    """Expand every cohort OCCLabeledEntity into per-sub-solid entities.

    Matches each surviving post-BOP shape to its slab_meta entry by
    ShapeKey (cad_occ already preserved sub-solid TShapes via the
    fragment piece-ownership pass). One OCCLabeledEntity per
    sub-solid, carrying the source slab's physical_name and a
    synthetic index.
    """
    from meshwell.structured.build import _shape_key

    expanded: list[OCCLabeledEntity] = []
    next_index = max((e.index for e in occ_entities), default=-1) + 1
    cohort_pnames = {ce.physical_name for ce in state.cohort_entities}
    for ent in occ_entities:
        if ent.physical_name not in cohort_pnames:
            expanded.append(ent)
            continue
        for shape in ent.shapes:
            key = _shape_key(shape)
            meta = state.slab_meta.get(key)
            if meta is None:
                # Sub-solid was modified by BOP; we still represent it.
                expanded.append(_copy_with(ent, [shape], next_index))
                next_index += 1
                continue
            sub_ent = OCCLabeledEntity(
                shapes=[shape],
                physical_name=meta.physical_name,
                index=next_index,
                keep=True,
                dim=3,
                mesh_order=ent.mesh_order,
            )
            expanded.append(sub_ent)
            next_index += 1
    return expanded


def _copy_with(ent: OCCLabeledEntity, shapes, idx: int) -> OCCLabeledEntity:
    return OCCLabeledEntity(
        shapes=list(shapes),
        physical_name=ent.physical_name,
        index=idx,
        keep=ent.keep,
        dim=ent.dim,
        mesh_order=ent.mesh_order,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/structured/test_post_pass_expansion.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add meshwell/structured/pipeline.py tests/structured/test_post_pass_expansion.py
git commit -m "feat(structured): post-pass expansion of cohort entity into per-slab entities"
```

---

## Task 15: `mesh.py` — accept `pre_2d_hook` + `pre_3d_hook` parameters

**Files:**
- Modify: `meshwell/mesh.py` (`Mesh.process_geometry` + `mesh()`)
- Test: `tests/structured/test_mesh_hooks.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/structured/test_mesh_hooks.py
from shapely.geometry import Polygon

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism


def test_pre_2d_and_pre_3d_hooks_called(tmp_path):
    SQ = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    p = PolyPrism(polygons=SQ, buffers={0.0: 0.0, 1.0: 0.0}, physical_name="x")
    calls: dict[str, int] = {"pre_2d": 0, "pre_3d": 0}

    def hook_2d():
        calls["pre_2d"] += 1

    def hook_3d():
        calls["pre_3d"] += 1

    generate_mesh(
        entities=[p],
        dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
        pre_2d_hook=hook_2d,
        pre_3d_hook=hook_3d,
    )
    assert calls["pre_2d"] >= 1
    assert calls["pre_3d"] >= 1
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/structured/test_mesh_hooks.py -v
```
Expected: FAIL — `pre_2d_hook` not recognized.

- [ ] **Step 3: Add hook parameters**

Locate `Mesh.process_geometry` in `meshwell/mesh.py`. Add `pre_2d_hook` and `pre_3d_hook` to its signature (defaults `None`). Find the calls to `gmsh.model.mesh.generate(2)` and `gmsh.model.mesh.generate(3)` and invoke the hooks just before each, if not None:

```python
# Inside Mesh.process_geometry, before generate(2):
if pre_2d_hook is not None:
    pre_2d_hook()
gmsh.model.mesh.generate(2)
# ...
if pre_3d_hook is not None:
    pre_3d_hook()
gmsh.model.mesh.generate(3)
```

Add the same params to the module-level `mesh()` function and forward them to `mesh_generator.process_geometry(...)`.

Add them to `meshwell/orchestrator.py::generate_mesh` signature too (kwargs are forwarded via `**mesh_kwargs` already, so this works without explicit declaration — verify by running the test).

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/structured/test_mesh_hooks.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add meshwell/mesh.py tests/structured/test_mesh_hooks.py
git commit -m "feat(mesh): pre_2d_hook + pre_3d_hook callable parameters"
```

---

## Task 16: `wedge.py` — `pre_2d_hook` (transfinite hints)

**Files:**
- Create: `meshwell/structured/wedge.py`
- Test: `tests/structured/test_wedge_pre2d.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/structured/test_wedge_pre2d.py
import pytest
from shapely.geometry import Polygon

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.structured.exceptions import StructuredLateralNLayersMismatchError


SQ = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


def test_transfinite_hints_produce_quad_laterals(tmp_path):
    p = PolyPrism(polygons=SQ, buffers={0.0: 0.0, 1.0: 0.0},
                  physical_name="s", structured=True, n_layers=2)
    out = generate_mesh(
        entities=[p], dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
    )
    # Lateral faces should be quadrangular meshes (4 quads per wall
    # x 4 walls = 16 quads for n_layers=2 and one segment per wall).
    import meshio
    m = meshio.read(tmp_path / "out.msh")
    quads = sum(cb.data.shape[0] for cb in m.cells if cb.type == "quad")
    assert quads >= 16


def test_n_layers_mismatch_raises(tmp_path):
    # Two laterally-touching structured slabs with different n_layers.
    A = PolyPrism(polygons=SQ, buffers={0.0: 0.0, 1.0: 0.0},
                  physical_name="a", structured=True, n_layers=2)
    SQ2 = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
    B = PolyPrism(polygons=SQ2, buffers={0.0: 0.0, 1.0: 0.0},
                  physical_name="b", structured=True, n_layers=5)
    with pytest.raises(StructuredLateralNLayersMismatchError):
        generate_mesh(
            entities=[A, B], dim=3,
            output_mesh=tmp_path / "out.msh",
            default_characteristic_length=0.5,
        )
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/structured/test_wedge_pre2d.py -v
```
Expected: FAIL — wedge.py doesn't exist; structured entities don't go through wedge hooks yet.

- [ ] **Step 3: Write `wedge.py` pre-2d portion**

```python
# meshwell/structured/wedge.py
"""Stage 5 — gmsh meshing hooks for structured cohorts.

pre_2d_hook (apply_lateral_transfinite_hints): sets transfinite curve
counts on vertical lateral edges and transfinite surface hints on
lateral faces of every cohort sub-solid. Raises on n_layers mismatch
or unsupported lateral topology.

pre_3d_hook (stamp_wedges, Task 17): per cohort sub-solid, copies bot
triangulation to top and emits wedge elements.
"""
from __future__ import annotations

from collections import defaultdict

import gmsh

from meshwell.structured.exceptions import (
    StructuredLateralNLayersMismatchError,
    StructuredTransfiniteRejectedError,
)
from meshwell.structured.types import ShapeKey, SlabMeta


def apply_lateral_transfinite_hints(
    slab_meta: dict[ShapeKey, SlabMeta],
    face_tag_by_key: dict[ShapeKey, int],
) -> None:
    """For each cohort sub-solid lateral face: enforce n_layers and
    apply gmsh transfinite + setTransfiniteSurface hints.

    Raise on:
      - Shared lateral face with mismatched n_layers.
      - Lateral face with multi-wire boundary or != 4 boundary edges.
    """
    # Group: face_tag -> list[(slab_index, n_layers)] for shared-lateral
    # n_layers consistency.
    owners_per_face: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for sub_key, meta in slab_meta.items():
        for fk in meta.lateral_face_keys:
            tag = face_tag_by_key.get(fk)
            if tag is None:
                continue
            owners_per_face[tag].append((meta.slab_index, meta.n_layers))

    for face_tag, owners in owners_per_face.items():
        # n_layers must agree across all owners of this face.
        n_layers_set = {n for _, n in owners}
        if len(n_layers_set) > 1:
            (sa, na), (sb, nb) = owners[0], owners[1]
            raise StructuredLateralNLayersMismatchError(
                slab_a=sa, slab_b=sb, face_tag=face_tag,
                n_layers_a=na, n_layers_b=nb,
            )
        n_layers = owners[0][1]

        # Get the boundary 1D edges of this face.
        edges = gmsh.model.getBoundary(
            [(2, face_tag)], oriented=False, recursive=False
        )
        if len(edges) != 4:
            raise StructuredTransfiniteRejectedError(
                face_tag=face_tag, slab_index=owners[0][0],
                reason=f"expected 4 boundary edges, got {len(edges)}",
            )
        # Identify vertical edges (endpoints differ in z).
        for dim, etag in edges:
            assert dim == 1
            ev = gmsh.model.getBoundary(
                [(1, etag)], oriented=False, recursive=False
            )
            zs = []
            for vd, vt in ev:
                _, _, z = gmsh.model.getValue(0, vt, [])[0:3]
                zs.append(z)
            if len(zs) == 2 and abs(zs[0] - zs[1]) > 1e-9:
                gmsh.model.mesh.setTransfiniteCurve(etag, n_layers + 1)
        gmsh.model.mesh.setTransfiniteSurface(face_tag)
```

Wire this into the cad_occ/mesh integration in Task 18.

- [ ] **Step 4: Don't run yet — depends on Task 18**

Mark test as expected-fail for now; remove the xfail after Task 18.

- [ ] **Step 5: Commit**

```
git add meshwell/structured/wedge.py tests/structured/test_wedge_pre2d.py
git commit -m "feat(structured): pre_2d transfinite hints (wedge.py partial)"
```

---

## Task 17: `wedge.py` — `pre_3d_hook` (wedge stamping)

**Files:**
- Modify: `meshwell/structured/wedge.py`
- Test: `tests/structured/test_wedge_pre3d.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/structured/test_wedge_pre3d.py
from shapely.geometry import Polygon

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism


SQ = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


def test_wedge_count_matches_bot_triangles_times_n_layers(tmp_path):
    p = PolyPrism(polygons=SQ, buffers={0.0: 0.0, 1.0: 0.0},
                  physical_name="s", structured=True, n_layers=3)
    generate_mesh(
        entities=[p], dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
    )
    import meshio
    m = meshio.read(tmp_path / "out.msh")
    wedges = sum(cb.data.shape[0] for cb in m.cells if cb.type == "wedge")
    tets = sum(cb.data.shape[0] for cb in m.cells if cb.type == "tetra")
    assert wedges > 0
    assert tets == 0  # entire volume is structured


def test_stacked_cohort_wedges_conformal(tmp_path):
    a = PolyPrism(polygons=SQ, buffers={0.0: 0.0, 1.0: 0.0},
                  physical_name="a", structured=True, n_layers=2)
    b = PolyPrism(polygons=SQ, buffers={1.0: 0.0, 2.0: 0.0},
                  physical_name="b", structured=True, n_layers=2)
    generate_mesh(
        entities=[a, b], dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
    )
    import meshio
    m = meshio.read(tmp_path / "out.msh")
    # Check no duplicate nodes (conformal interior interface).
    import numpy as np
    pts = m.points
    _, counts = np.unique(np.round(pts, 6), axis=0, return_counts=True)
    assert counts.max() == 1, "duplicate node positions → non-conformal mesh"
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/structured/test_wedge_pre3d.py -v
```
Expected: FAIL.

- [ ] **Step 3: Add `stamp_wedges` to `wedge.py`**

Append to `meshwell/structured/wedge.py`:

```python
import numpy as np

from meshwell.structured.exceptions import (
    WedgeBotNodeMismatchError, WedgeCountMismatchError,
)


def stamp_wedges(
    slab_meta: dict[ShapeKey, SlabMeta],
    face_tag_by_key: dict[ShapeKey, int],
    sub_solid_tag_by_key: dict[ShapeKey, int],
    point_tolerance: float = 1e-3,
) -> None:
    """For each cohort sub-solid: read bot tri mesh, stamp on top,
    emit n_layers wedges per bot triangle into the sub-solid's 3D tag.

    Iterates sub-solids in z_lo ascending order so shared bot/top
    faces are stamped from below before being read from above.
    """
    # Order sub-solids by zlo via their bot face z.
    order: list[tuple[float, ShapeKey, SlabMeta]] = []
    for k, meta in slab_meta.items():
        bot_tag = face_tag_by_key.get(meta.bot_face_key)
        if bot_tag is None:
            continue
        _, _, z = _face_centroid(bot_tag)
        order.append((z, k, meta))
    order.sort(key=lambda t: t[0])

    for _, sub_key, meta in order:
        bot_tag = face_tag_by_key[meta.bot_face_key]
        top_tag = face_tag_by_key[meta.top_face_key]
        vol_tag = sub_solid_tag_by_key[sub_key]
        _stamp_one(bot_tag, top_tag, vol_tag, meta, point_tolerance)


def _face_centroid(face_tag: int) -> tuple[float, float, float]:
    node_tags, coord, _ = gmsh.model.mesh.getNodes(2, face_tag, includeBoundary=True)
    pts = np.array(coord).reshape(-1, 3)
    if len(pts) == 0:
        # Fall back to OCC face centroid via bbox.
        bbox = gmsh.model.getBoundingBox(2, face_tag)
        return ((bbox[0] + bbox[3]) / 2, (bbox[1] + bbox[4]) / 2, (bbox[2] + bbox[5]) / 2)
    return tuple(pts.mean(axis=0))


def _stamp_one(
    bot_tag: int, top_tag: int, vol_tag: int,
    meta: SlabMeta, point_tolerance: float,
) -> None:
    # 1) Read bot triangulation.
    elem_types, elem_tags, node_tags = gmsh.model.mesh.getElements(2, bot_tag)
    if 2 not in elem_types:  # type 2 = triangle
        return
    tri_idx = list(elem_types).index(2)
    tris = np.array(node_tags[tri_idx]).reshape(-1, 3)
    bot_node_tags, bot_coord, _ = gmsh.model.mesh.getNodes(
        2, bot_tag, includeBoundary=True
    )
    bot_pts = np.array(bot_coord).reshape(-1, 3)
    bot_tag_by_idx = {t: i for i, t in enumerate(bot_node_tags)}
    bot_z = bot_pts[:, 2].mean()

    # 2) Determine top z from top face bbox.
    bbox = gmsh.model.getBoundingBox(2, top_tag)
    top_z = bbox[5]
    dz = (top_z - bot_z) / meta.n_layers

    # 3) Clear top face mesh & rebuild from bot.
    gmsh.model.mesh.clear([(2, top_tag)])
    # Existing top boundary nodes (from lateral mesh) should be reused.
    existing_top_nodes, existing_top_coord, _ = gmsh.model.mesh.getNodes(
        2, top_tag, includeBoundary=True
    )
    existing_top_pts = np.array(existing_top_coord).reshape(-1, 3) if len(existing_top_coord) else np.zeros((0, 3))

    # bot_node_tag -> top_node_tag map.
    bot_to_top: dict[int, int] = {}
    mismatched = 0
    for bnt, bpt in zip(bot_node_tags, bot_pts):
        target_xy = np.array([bpt[0], bpt[1], top_z])
        if len(existing_top_pts):
            d = np.linalg.norm(existing_top_pts[:, :2] - target_xy[:2], axis=1)
            if d.min() < point_tolerance:
                bot_to_top[bnt] = int(existing_top_nodes[int(np.argmin(d))])
                continue
        # Allocate new node on top face.
        new_tag = gmsh.model.mesh.addNodes(
            2, top_tag, [], [float(bpt[0]), float(bpt[1]), float(top_z)]
        )
        # gmsh.model.mesh.addNodes assigns auto tags when [] passed.
        # Fetch the freshly added node tag via getMaxNodeTag - this
        # API varies by gmsh version; safer: use addNode singular.
        new_tag = gmsh.model.mesh.getMaxNodeTag() + 1
        gmsh.model.mesh.addNodes(
            2, top_tag, [new_tag],
            [float(bpt[0]), float(bpt[1]), float(top_z)],
        )
        bot_to_top[bnt] = int(new_tag)

    # Re-stamp triangulation on top face for symmetry.
    gmsh.model.mesh.addElementsByType(
        top_tag, 2,
        [],  # auto tags
        [bot_to_top[t] for tri in tris for t in tri],
    )

    # 4) Intermediate layer nodes (for n_layers > 1).
    layer_node_maps: list[dict[int, int]] = [bot_tag_by_idx_to_tag := dict(zip(range(len(bot_node_tags)), bot_node_tags))]
    if meta.n_layers > 1:
        for layer in range(1, meta.n_layers):
            z_layer = bot_z + dz * layer
            this_map: dict[int, int] = {}
            for i, bpt in enumerate(bot_pts):
                tag = gmsh.model.mesh.getMaxNodeTag() + 1
                gmsh.model.mesh.addNodes(
                    3, vol_tag, [tag],
                    [float(bpt[0]), float(bpt[1]), float(z_layer)],
                )
                this_map[i] = tag
            layer_node_maps.append(this_map)
        layer_node_maps.append({i: bot_to_top[bot_node_tags[i]] for i in range(len(bot_node_tags))})
    else:
        layer_node_maps.append({i: bot_to_top[bot_node_tags[i]] for i in range(len(bot_node_tags))})

    # 5) Emit wedges. gmsh element type 6 = prism (6-node wedge).
    wedge_node_tags: list[int] = []
    expected = 0
    for layer in range(meta.n_layers):
        bot_map = {i: layer_node_maps[layer][i] for i in range(len(bot_node_tags))}
        top_map = {i: layer_node_maps[layer + 1][i] for i in range(len(bot_node_tags))}
        bot_idx_by_tag = {t: i for i, t in enumerate(bot_node_tags)}
        for tri in tris:
            b0, b1, b2 = (bot_idx_by_tag[t] for t in tri)
            wedge_node_tags.extend([
                bot_map[b0], bot_map[b1], bot_map[b2],
                top_map[b0], top_map[b1], top_map[b2],
            ])
            expected += 1
    gmsh.model.mesh.addElementsByType(vol_tag, 6, [], wedge_node_tags)
    emitted = len(wedge_node_tags) // 6
    if emitted != expected:
        raise WedgeCountMismatchError(
            slab_index=meta.slab_index, expected=expected, got=emitted,
        )
    if mismatched:
        raise WedgeBotNodeMismatchError(
            slab_index=meta.slab_index, mismatched_count=mismatched,
        )
```

- [ ] **Step 4: Don't run yet — depends on Task 18 for orchestration**

- [ ] **Step 5: Commit**

```
git add meshwell/structured/wedge.py tests/structured/test_wedge_pre3d.py
git commit -m "feat(structured): pre_3d wedge stamping per cohort sub-solid"
```

---

## Task 18: Wire structured pipeline into `orchestrator.py`

**Files:**
- Modify: `meshwell/orchestrator.py`
- Test: re-run tests from Tasks 16 + 17

- [ ] **Step 1: Modify `orchestrator.generate_mesh`**

Replace the body of `meshwell/orchestrator.py::generate_mesh`'s "Stage 1" section with the structured pre-pass + post-pass wiring:

```python
from meshwell.structured.pipeline import structured_post_pass, structured_pre_pass
from meshwell.structured.wedge import (
    apply_lateral_transfinite_hints, stamp_wedges,
)
```

```python
# Replace existing Stage 1 + Stage 2 in generate_mesh:
entities = deserialize(entities, registry=registry)

cad_kwargs: dict[str, Any] = {}
# ... (existing arg forwarding unchanged) ...

# Structured pre-pass: replace structured entities with cohort
# compounds; record slab_meta for the wedge hooks.
state = structured_pre_pass(
    entities, point_tolerance=cad_kwargs.get("point_tolerance", 1e-3),
)

occ_entities_raw = cad_occ(state.entities_out, **cad_kwargs)

# Post-pass: expand cohort entities into per-slab entities for XAO.
occ_entities = structured_post_pass(occ_entities_raw, state)

# --- Stage 2: XAO emit + gmsh load ---
# ... (unchanged) ...

# Build the gmsh-side face/sub-solid tag maps once XAO is loaded.
face_tag_by_key: dict = {}
sub_solid_tag_by_key: dict = {}
if state.slab_meta:
    from meshwell.structured.build import _shape_key
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer, TopExp
    # Build face/solid lookup by ShapeKey via the model's TopoDS_Compound.
    # ModelManager exposes the post-load compound (verify the API path).
    compound = mm.get_loaded_compound()
    face_to_tag = _build_tag_lookup(mm, dim=2)
    solid_to_tag = _build_tag_lookup(mm, dim=3)
    # Match ShapeKeys.
    exp = TopExp_Explorer(compound, TopAbs_FACE)
    while exp.More():
        f = exp.Current()
        k = _shape_key(f)
        if f in face_to_tag:
            face_tag_by_key[k] = face_to_tag[f]
        exp.Next()
    exp = TopExp_Explorer(compound, TopAbs_SOLID)
    while exp.More():
        s = exp.Current()
        k = _shape_key(s)
        if s in solid_to_tag:
            sub_solid_tag_by_key[k] = solid_to_tag[s]
        exp.Next()


def _pre_2d():
    if state.slab_meta:
        apply_lateral_transfinite_hints(state.slab_meta, face_tag_by_key)
    if user_pre_2d:
        user_pre_2d()


def _pre_3d():
    if state.slab_meta:
        stamp_wedges(state.slab_meta, face_tag_by_key, sub_solid_tag_by_key)
    if user_pre_3d:
        user_pre_3d()


user_pre_2d = mesh_kwargs.pop("pre_2d_hook", None)
user_pre_3d = mesh_kwargs.pop("pre_3d_hook", None)

return mesh(
    dim=dim,
    model=mm,
    output_file=Path(output_mesh) if output_mesh else None,
    pre_2d_hook=_pre_2d,
    pre_3d_hook=_pre_3d,
    **mesh_kwargs,
)
```

You'll need to implement `mm.get_loaded_compound()` and `_build_tag_lookup(mm, dim)` — these are small wrappers over gmsh's `getEntities(dim)` and the post-load OCC compound stored on `ModelManager`. If `ModelManager` does not yet expose the compound, add a simple accessor in `meshwell/model.py` (read the BREP from disk via `BRepTools.Read_s`, or capture it at XAO load time).

- [ ] **Step 2: Run the wedge tests**

```
pytest tests/structured/test_wedge_pre2d.py tests/structured/test_wedge_pre3d.py -v
```
Expected: PASS.

- [ ] **Step 3: Commit**

```
git add meshwell/orchestrator.py meshwell/model.py
git commit -m "feat(orchestrator): wire structured pre/post-pass and wedge hooks"
```

---

## Task 19: Integration test — single structured slab end-to-end

**Files:**
- Create: `tests/structured/test_e2e_single.py`

- [ ] **Step 1: Write the test**

```python
# tests/structured/test_e2e_single.py
from shapely.geometry import Polygon

import meshio
from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism


SQ = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


def test_single_slab_yields_wedges_only(tmp_path):
    p = PolyPrism(polygons=SQ, buffers={0.0: 0.0, 1.0: 0.0},
                  physical_name="slab", structured=True, n_layers=2)
    generate_mesh([p], dim=3, output_mesh=tmp_path / "out.msh",
                  default_characteristic_length=0.5)
    m = meshio.read(tmp_path / "out.msh")
    wedge_count = sum(cb.data.shape[0] for cb in m.cells if cb.type == "wedge")
    tet_count = sum(cb.data.shape[0] for cb in m.cells if cb.type == "tetra")
    assert wedge_count > 0
    assert tet_count == 0
    assert "slab" in {n for grp in m.cell_sets for n in [grp]}
```

- [ ] **Step 2: Run**

```
pytest tests/structured/test_e2e_single.py -v
```
Expected: PASS.

- [ ] **Step 3: Commit**

```
git add tests/structured/test_e2e_single.py
git commit -m "test(structured): end-to-end single structured slab"
```

---

## Task 20: Integration test — stacked cohort with unstructured cap

**Files:**
- Create: `tests/structured/test_e2e_stacked_with_cap.py`

- [ ] **Step 1: Write the test**

```python
# tests/structured/test_e2e_stacked_with_cap.py
from shapely.geometry import Polygon

import meshio
from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism


SQ_BIG = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
SQ_SMALL = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])


def test_stacked_with_unstructured_cap_conformal(tmp_path):
    a = PolyPrism(polygons=SQ_BIG, buffers={0.0: 0.0, 1.0: 0.0},
                  physical_name="a", structured=True, n_layers=2)
    b = PolyPrism(polygons=SQ_SMALL, buffers={1.0: 0.0, 2.0: 0.0},
                  physical_name="b", structured=True, n_layers=2)
    cap = PolyPrism(polygons=SQ_BIG, buffers={2.0: 0.0, 3.0: 0.0},
                    physical_name="cap")
    generate_mesh([a, b, cap], dim=3, output_mesh=tmp_path / "out.msh",
                  default_characteristic_length=0.5)
    m = meshio.read(tmp_path / "out.msh")
    wedge_count = sum(cb.data.shape[0] for cb in m.cells if cb.type == "wedge")
    tet_count = sum(cb.data.shape[0] for cb in m.cells if cb.type == "tetra")
    assert wedge_count > 0
    assert tet_count > 0
    # No duplicate nodes — proxy for conformality.
    import numpy as np
    pts = np.round(m.points, 5)
    assert len(np.unique(pts, axis=0)) == len(pts)
```

- [ ] **Step 2: Run**

```
pytest tests/structured/test_e2e_stacked_with_cap.py -v
```
Expected: PASS.

- [ ] **Step 3: Commit**

```
git add tests/structured/test_e2e_stacked_with_cap.py
git commit -m "test(structured): e2e stacked cohort with unstructured cap"
```

---

## Task 21: Integration test — mesh_order overlap carving

**Files:**
- Create: `tests/structured/test_e2e_overlap_carving.py`

- [ ] **Step 1: Write the test**

```python
# tests/structured/test_e2e_overlap_carving.py
from shapely.geometry import Polygon

import meshio
from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism


SQ_BIG = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
SQ_SMALL = Polygon([(2, 2), (4, 2), (4, 4), (2, 4)])


def test_lower_mesh_order_void_carves_structured(tmp_path):
    big = PolyPrism(polygons=SQ_BIG, buffers={0.0: 0.0, 1.0: 0.0},
                    physical_name="bg", structured=True, mesh_order=2.0)
    void = PolyPrism(polygons=SQ_SMALL, buffers={0.0: 0.0, 1.0: 0.0},
                     physical_name="void", structured=True,
                     mesh_order=1.0, mesh_bool=False)
    generate_mesh([big, void], dim=3, output_mesh=tmp_path / "out.msh",
                  default_characteristic_length=0.5)
    m = meshio.read(tmp_path / "out.msh")
    # void's volume should not be present in physical groups.
    wedge_count = sum(cb.data.shape[0] for cb in m.cells if cb.type == "wedge")
    assert wedge_count > 0
```

- [ ] **Step 2: Run**

```
pytest tests/structured/test_e2e_overlap_carving.py -v
```
Expected: PASS.

- [ ] **Step 3: Commit**

```
git add tests/structured/test_e2e_overlap_carving.py
git commit -m "test(structured): e2e mesh_order Policy B carving with void"
```

---

## Task 22: STRESS TEST — complex cohort scene

**Files:**
- Create: `tests/structured/test_stress_complex_scene.py`

This is the user-requested stress test: a cohort with multiple sublevels, each with multiple polygons of varied shapes (including arcs), some with `mesh_bool=False`, with complex unstructured material above and below.

- [ ] **Step 1: Write the test**

```python
# tests/structured/test_stress_complex_scene.py
"""Stress tests for complex structured scenes.

Validates the v1 pipeline on inputs that exercise every planner
branch:
  - multi-level cohort
  - multiple polygons per level
  - arcs (full and partial)
  - mesh_bool=False voids
  - unstructured neighbours above and below with similar complexity
  - mesh_order carving across structured slabs
"""
from __future__ import annotations

import numpy as np
import meshio
import pytest
from shapely.geometry import MultiPolygon, Polygon

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism


def _circle(cx: float, cy: float, r: float, n: int = 48) -> Polygon:
    angles = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    return Polygon([(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles])


def _annulus(cx: float, cy: float, r_out: float, r_in: float, n: int = 48) -> Polygon:
    outer = _circle(cx, cy, r_out, n)
    inner = _circle(cx, cy, r_in, n)
    return Polygon(outer.exterior.coords, holes=[inner.exterior.coords])


def _rect(x1, y1, x2, y2) -> Polygon:
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


# ---------- cohort layer A: z=[0, 1] ----------
SQUARE_A = _rect(-5, -5, 5, 5)
CIRCLE_A = _circle(0, 8, 2)
RECT_HOLE_A = Polygon(
    _rect(-9, -9, -3, -3).exterior.coords,
    holes=[_rect(-7, -7, -5, -5).exterior.coords],
)

# ---------- cohort layer B: z=[1, 2] ----------
CIRCLE_B = _circle(0, 0, 3)
ANNULUS_B = _annulus(0, 8, 2.5, 1.2)

# ---------- cohort layer C: z=[2, 3] ----------
HEX_C = Polygon([
    (2 * np.cos(a), 2 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 7)[:-1]
])
VOID_C = _circle(0, 0, 0.5)  # mesh_bool=False inside HEX_C

# ---------- unstructured below: z=[-2, 0] ----------
BIG_BASE = _rect(-15, -15, 15, 15)
HOLE_BASE = _circle(0, 0, 1.0)

# ---------- unstructured above: z=[3, 5] ----------
BIG_CAP = _rect(-15, -15, 15, 15)
CAP_ARCH = _circle(3, 3, 2)  # carves part of the cap (lower mesh_order)


@pytest.fixture
def complex_scene_entities() -> list:
    return [
        # Cohort layer A
        PolyPrism(SQUARE_A, {0.0: 0.0, 1.0: 0.0}, physical_name="A_square",
                  structured=True, n_layers=2, mesh_order=3.0),
        PolyPrism(CIRCLE_A, {0.0: 0.0, 1.0: 0.0}, physical_name="A_circle",
                  structured=True, n_layers=2, mesh_order=3.0,
                  identify_arcs=True),
        PolyPrism(RECT_HOLE_A, {0.0: 0.0, 1.0: 0.0}, physical_name="A_recth",
                  structured=True, n_layers=2, mesh_order=3.0),
        # Cohort layer B
        PolyPrism(CIRCLE_B, {1.0: 0.0, 2.0: 0.0}, physical_name="B_circle",
                  structured=True, n_layers=2, mesh_order=3.0,
                  identify_arcs=True),
        PolyPrism(ANNULUS_B, {1.0: 0.0, 2.0: 0.0}, physical_name="B_annulus",
                  structured=True, n_layers=2, mesh_order=3.0,
                  identify_arcs=True),
        # Cohort layer C with void
        PolyPrism(HEX_C, {2.0: 0.0, 3.0: 0.0}, physical_name="C_hex",
                  structured=True, n_layers=2, mesh_order=3.0),
        PolyPrism(VOID_C, {2.0: 0.0, 3.0: 0.0}, physical_name="C_void",
                  structured=True, n_layers=2, mesh_order=1.0, mesh_bool=False),
        # Unstructured below
        PolyPrism(Polygon(BIG_BASE.exterior.coords,
                          holes=[HOLE_BASE.exterior.coords]),
                  {-2.0: 0.0, 0.0: 0.0}, physical_name="base",
                  mesh_order=5.0),
        # Unstructured above
        PolyPrism(BIG_CAP, {3.0: 0.0, 5.0: 0.0}, physical_name="cap",
                  mesh_order=5.0),
        PolyPrism(CAP_ARCH, {3.0: 0.0, 5.0: 0.0}, physical_name="cap_arch",
                  mesh_order=2.0, identify_arcs=True),
    ]


def test_complex_scene_meshes_without_error(complex_scene_entities, tmp_path):
    generate_mesh(
        complex_scene_entities, dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.8,
    )
    m = meshio.read(tmp_path / "out.msh")
    wedge_count = sum(cb.data.shape[0] for cb in m.cells if cb.type == "wedge")
    tet_count = sum(cb.data.shape[0] for cb in m.cells if cb.type == "tetra")
    assert wedge_count > 0, "expected structured wedge elements"
    assert tet_count > 0, "expected tet elements in unstructured caps"


def test_complex_scene_all_physical_groups_present(
    complex_scene_entities, tmp_path
):
    generate_mesh(
        complex_scene_entities, dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.8,
    )
    m = meshio.read(tmp_path / "out.msh")
    expected = {
        "A_square", "A_circle", "A_recth",
        "B_circle", "B_annulus",
        "C_hex",
        "base", "cap", "cap_arch",
    }
    field = set(m.cell_sets.keys())
    missing = expected - field
    assert not missing, f"physical groups missing from mesh: {missing}"
    # Void should NOT appear as a 3D physical group.
    assert "C_void" not in field


def test_complex_scene_node_uniqueness(complex_scene_entities, tmp_path):
    """Proxy conformality check — no duplicated nodes after dedup."""
    generate_mesh(
        complex_scene_entities, dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.8,
    )
    m = meshio.read(tmp_path / "out.msh")
    pts = np.round(m.points, 5)
    unique = np.unique(pts, axis=0)
    assert len(unique) == len(pts), (
        f"{len(pts) - len(unique)} duplicate node positions "
        "indicate non-conformal interfaces."
    )
```

- [ ] **Step 2: Run**

```
pytest tests/structured/test_stress_complex_scene.py -v
```
Expected: all three tests PASS.

- [ ] **Step 3: Commit**

```
git add tests/structured/test_stress_complex_scene.py
git commit -m "test(structured): stress test — complex multi-level cohort + unstructured caps"
```

---

## Task 23: STRESS TEST — error case coverage

**Files:**
- Create: `tests/structured/test_stress_errors.py`

- [ ] **Step 1: Write the test**

```python
# tests/structured/test_stress_errors.py
"""Negative stress tests: every planner failure mode raises with the
correct exception class and context.
"""
import pytest
from shapely.geometry import Polygon

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.structured.exceptions import (
    CohortShellModifiedError,
    StructuredEntityTypeError,
    StructuredExtrudeRequiredError,
    StructuredLateralNLayersMismatchError,
    StructuredZStackError,
)


SQ = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
SQ2 = Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])  # touches SQ on x=10
FAR = Polygon([(100, 100), (110, 100), (110, 110), (100, 110)])


def test_structured_on_buffered_raises():
    with pytest.raises(StructuredExtrudeRequiredError):
        PolyPrism(SQ, {0.0: 0.0, 1.0: 0.5}, physical_name="x", structured=True)


def test_non_polyprism_structured_raises(tmp_path):
    class Fake:
        structured = True
        physical_name = "fake"
        instanciate_occ = lambda self: None
    with pytest.raises(StructuredEntityTypeError):
        generate_mesh([Fake()], dim=3, output_mesh=tmp_path / "x.msh",
                      default_characteristic_length=1.0)


def test_zstack_violation_raises(tmp_path):
    s = PolyPrism(SQ, {0.0: 0.0, 2.0: 0.0}, physical_name="s", structured=True)
    bad = PolyPrism(SQ, {1.0: 0.0, 3.0: 0.0}, physical_name="bad")  # zlo=1 mid-cohort
    with pytest.raises(StructuredZStackError):
        generate_mesh([s, bad], dim=3, output_mesh=tmp_path / "x.msh",
                      default_characteristic_length=1.0)


def test_n_layers_mismatch_lateral_touch_raises(tmp_path):
    a = PolyPrism(SQ, {0.0: 0.0, 1.0: 0.0}, physical_name="a",
                  structured=True, n_layers=2)
    b = PolyPrism(SQ2, {0.0: 0.0, 1.0: 0.0}, physical_name="b",
                  structured=True, n_layers=5)
    with pytest.raises(StructuredLateralNLayersMismatchError):
        generate_mesh([a, b], dim=3, output_mesh=tmp_path / "x.msh",
                      default_characteristic_length=1.0)
```

- [ ] **Step 2: Run**

```
pytest tests/structured/test_stress_errors.py -v
```
Expected: all PASS (every raise hit).

- [ ] **Step 3: Commit**

```
git add tests/structured/test_stress_errors.py
git commit -m "test(structured): negative stress tests — all failure modes raise"
```

---

## Task 24: Public API + module exports

**Files:**
- Modify: `meshwell/structured/__init__.py`
- Test: `tests/structured/test_public_api.py`

- [ ] **Step 1: Write the test**

```python
# tests/structured/test_public_api.py
def test_public_imports():
    from meshwell.structured import (
        StructuredError,
        StructuredZStackError,
        WedgeCountMismatchError,
    )
    assert StructuredError is not None
    assert issubclass(StructuredZStackError, StructuredError)
    assert issubclass(WedgeCountMismatchError, StructuredError)
```

- [ ] **Step 2: Run** — expected FAIL.

- [ ] **Step 3: Update `__init__.py`**

```python
# meshwell/structured/__init__.py
"""Structured prism meshing for meshwell.cad_occ.

User-facing: set ``structured=True`` and ``n_layers=N`` on a PolyPrism
to have its volume meshed with wedge elements. Surrounding unstructured
regions remain tet-meshed; interfaces are conformal by construction.
"""
from meshwell.structured.exceptions import (
    CohortNonManifoldError,
    CohortShellModifiedError,
    StructuredEntityTypeError,
    StructuredError,
    StructuredExtrudeRequiredError,
    StructuredLateralNLayersMismatchError,
    StructuredTransfiniteRejectedError,
    StructuredZStackError,
    SubPolygonAssemblyError,
    UnstructuredImprintRequiresPolyPrismError,
    WedgeBotNodeMismatchError,
    WedgeCountMismatchError,
)

__all__ = [
    "CohortNonManifoldError",
    "CohortShellModifiedError",
    "StructuredEntityTypeError",
    "StructuredError",
    "StructuredExtrudeRequiredError",
    "StructuredLateralNLayersMismatchError",
    "StructuredTransfiniteRejectedError",
    "StructuredZStackError",
    "SubPolygonAssemblyError",
    "UnstructuredImprintRequiresPolyPrismError",
    "WedgeBotNodeMismatchError",
    "WedgeCountMismatchError",
]
```

- [ ] **Step 4: Run** — expected PASS.

- [ ] **Step 5: Commit**

```
git add meshwell/structured/__init__.py tests/structured/test_public_api.py
git commit -m "feat(structured): public API exports"
```

---

## Final verification

After all tasks complete:

- [ ] Run the full structured test suite

```
pytest tests/structured/ -v
```
Expected: 100% PASS.

- [ ] Run the full meshwell test suite for regressions

```
pytest tests/ -v --ignore=tests/structured/
```
Expected: no regressions vs. main.

- [ ] Final commit if any fix-ups

```
git add -A
git commit -m "test: final pass — full suite green"
```
