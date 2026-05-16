# Structured PolyPrism Clean Rewrite — Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the CAD-stage phantom shape construction (Layer A) and the BOP-history-based vertex/edge/face correspondence map (Layer B) as pure-OCP modules — testable in isolation, with no `cad_occ` integration yet. End state: given a `StructuredPlan` from Phase 1, we can build OCP sub-prisms, run a `BOPAlgo_Builder` against them + arbitrary neighbour shapes, and recover the `PhantomMap` that the mesh stage (Phase 3) will consume.

**Architecture:** `meshwell/structured/phantom.py` exports two functions: `build_phantom_shapes(plan) → PhantomBuildResult` (constructs one OCP `TopoDS_Solid` per partition piece, plus the input-tag bookkeeping) and `extract_phantom_map(build_result, builder) → PhantomMap` (walks `algo.Modified()` / `algo.Generated()` / `algo.IsDeleted()` to populate the post-BOP map). `spec.py` gains the key dataclasses (`FaceKey`, `EdgeKey`, `VertexKey`, `LateralKey`, `PhantomMap`, `PhantomBuildResult`, `PhantomShape`). Tests use OCP directly with handcrafted small scenes; no gmsh, no `cad_occ`.

**Tech Stack:** Python 3.12, OCP (the Anthropic-CAD-friendly Python bindings to OCCT), shapely 2.x (for the partition polygons), pytest. Pydantic and gmsh are explicitly NOT involved at this layer.

**Spec reference:** `docs/superpowers/specs/2026-05-15-structured-polyprism-clean-design.md`, Layer A (sub-prism construction) and Layer B (BOP history map).

**Phase 1 reference:** `docs/superpowers/plans/2026-05-15-structured-polyprism-clean-phase1.md` (provides `StructuredPlan`, `Slab`, `face_partition`).

---

## File Structure

**Modify:**
- `meshwell/structured/spec.py` — add 7 new dataclasses: `FaceKey`, `EdgeKey`, `VertexKey`, `LateralKey`, `PhantomShape`, `PhantomBuildResult`, `PhantomMap`. (`Side = Literal["bot", "top"]` alias is also added.)
- `meshwell/structured/__init__.py` — re-export `build_phantom_shapes`, `extract_phantom_map`, `PhantomMap`.

**Create:**
- `meshwell/structured/phantom.py` — the new module (~250-350 LOC at completion).
- `tests/structured/test_phantom_keys.py` — dataclass tests (~50 LOC).
- `tests/structured/test_phantom_shapes.py` — shape-construction tests, direct OCP (~200 LOC).
- `tests/structured/test_phantom_history.py` — BOP-history extraction tests, direct OCP (~250 LOC).
- `tests/structured/_occ_helpers.py` — shared private OCP fixtures (small box, stick, plane intersection helpers) (~80 LOC).

**Untouched in Phase 2:**
- `meshwell/cad_occ.py`, `meshwell/orchestrator.py`, `meshwell/mesh.py` — integration happens in Phase 3.
- `meshwell/structured/plan.py` — Phase 1 is the data source; we only consume.
- `meshwell/polyprism.py` — no further changes.

---

## Tech notes (read this before any task)

### OCP topology iteration patterns

Walking sub-topology (vertices of a face, edges of a solid, etc.) uses `TopExp_Explorer`:

```python
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX

def _iter_subshapes(parent, sub_type):
    exp = TopExp_Explorer(parent, sub_type)
    while exp.More():
        yield exp.Current()
        exp.Next()
```

OCP `TopoDS_*` objects are not hashable by Python identity in a useful way; use `TopTools_ShapeMapHasher` (already used at `meshwell/cad_occ.py:50`) or pin shapes into a `dict[int, TopoDS_Shape]` keyed by a counter assigned at construction time. This plan uses the counter approach because the input shapes are stable for the lifetime of the build.

### Recording input tags

"Tags" in this module are **Python-side integers** we assign at construction. They are not gmsh tags. They serve as keys into a dict so we can ask, after BOP: "what did the shape originally numbered N become?"

### History API on `BOPAlgo_Builder` / `BRepAlgoAPI_BuilderAlgo`

The builder exposes:
- `Modified(in_shape) → TopTools_ListOfShape` — list of output shapes that *replaced* the input (e.g. an input face cut into pieces).
- `Generated(in_shape) → TopTools_ListOfShape` — list of *newly created* shapes on the input (e.g. a vertex created on an edge by a cut).
- `IsDeleted(in_shape) → bool` — True iff the input shape no longer exists in the output.

To iterate a `TopTools_ListOfShape`:

```python
from OCP.TopTools import TopTools_ListOfShape, TopTools_ListIteratorOfListOfShape

def _list_of_shape_to_list(lst):
    it = TopTools_ListIteratorOfListOfShape(lst)
    out = []
    while it.More():
        out.append(it.Value())
        it.Next()
    return out
```

### Building a `TopoDS_Face` from a shapely Polygon at fixed z

```python
from OCP.gp import gp_Pnt
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace

def _make_face_from_polygon(poly, z):
    # Outer boundary
    outer_coords = list(poly.exterior.coords)
    # Drop the last point if it duplicates the first (shapely closes rings).
    if outer_coords[0] == outer_coords[-1]:
        outer_coords = outer_coords[:-1]
    wire_builder = BRepBuilderAPI_MakeWire()
    for i in range(len(outer_coords)):
        p1 = gp_Pnt(*outer_coords[i], z)
        p2 = gp_Pnt(*outer_coords[(i + 1) % len(outer_coords)], z)
        edge = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
        wire_builder.Add(edge)
    outer_wire = wire_builder.Wire()
    face_builder = BRepBuilderAPI_MakeFace(outer_wire)
    # Holes (interior rings) via Add(inner_wire reversed).
    for ring in poly.interiors:
        inner_coords = list(ring.coords)
        if inner_coords[0] == inner_coords[-1]:
            inner_coords = inner_coords[:-1]
        inner_wire_builder = BRepBuilderAPI_MakeWire()
        for i in range(len(inner_coords)):
            p1 = gp_Pnt(*inner_coords[i], z)
            p2 = gp_Pnt(*inner_coords[(i + 1) % len(inner_coords)], z)
            edge = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
            inner_wire_builder.Add(edge)
        face_builder.Add(inner_wire_builder.Wire())
    return face_builder.Face()
```

**Orientation note:** shapely polygon orientation isn't guaranteed (per Phase 1 explore report). Use `shapely.geometry.polygon.orient(poly, sign=1.0)` to force CCW exterior and CW interior rings before face construction — this matches OCC convention.

### Building a sub-prism

```python
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.gp import gp_Vec

def _make_prism(face, height):
    return BRepPrimAPI_MakePrism(face, gp_Vec(0.0, 0.0, height)).Shape()
```

Returns a `TopoDS_Solid`. Walk its sub-faces via `TopExp_Explorer(solid, TopAbs_FACE)`; the resulting faces are the bottom face (= input `face`), the top face, and N lateral faces.

### Identifying which face of the prism is which after extrude

After `BRepPrimAPI_MakePrism`, the solid has:
- 1 bottom face (the input face, unchanged TopoDS identity — same OCC tag as the input)
- 1 top face (newly created at `z = zlo + height`)
- N lateral faces (one per outer boundary segment + per inner boundary segment of the input)

The `BRepPrimAPI_MakePrism` object exposes:
- `.FirstShape()` → bottom face
- `.LastShape()` → top face
- `.GeneratedShape(in_edge)` → the lateral face created from a given input boundary edge

Use these methods, NOT geometric matching, to identify which sub-face is which. Build the lateral-face lookup at construction time by iterating the input face's wires and asking the prism for each edge's generated lateral face.

### Test fixtures

`tests/structured/_occ_helpers.py` exposes:

```python
def make_box(x0, y0, z0, dx, dy, dz):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    return BRepPrimAPI_MakeBox(gp_Pnt(x0, y0, z0), gp_Pnt(x0+dx, y0+dy, z0+dz)).Shape()

def make_stick(x0, y0, z_lo, z_hi, dx, dy):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    return BRepPrimAPI_MakeBox(gp_Pnt(x0, y0, z_lo), gp_Pnt(x0+dx, y0+dy, z_hi)).Shape()

def count_subshapes(shape, sub_type):
    from OCP.TopExp import TopExp_Explorer
    exp = TopExp_Explorer(shape, sub_type)
    n = 0
    while exp.More():
        n += 1
        exp.Next()
    return n
```

---

## Task 1: Add Phase-2 dataclasses to `spec.py`

**Files:**
- Modify: `meshwell/structured/spec.py`
- Test: `tests/structured/test_phantom_keys.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/structured/test_phantom_keys.py`:

```python
"""Tests for Phase-2 key dataclasses in meshwell.structured.spec."""
from __future__ import annotations

import pytest


def test_face_key_fields():
    from meshwell.structured.spec import FaceKey

    k = FaceKey(slab_index=2, side="top", piece_index=0)
    assert k.slab_index == 2
    assert k.side == "top"
    assert k.piece_index == 0


def test_face_key_hashable():
    from meshwell.structured.spec import FaceKey

    k1 = FaceKey(slab_index=2, side="top", piece_index=0)
    k2 = FaceKey(slab_index=2, side="top", piece_index=0)
    assert k1 == k2
    assert hash(k1) == hash(k2)
    assert {k1: "a"}[k2] == "a"  # usable as dict key


def test_edge_key_includes_edge_index():
    from meshwell.structured.spec import EdgeKey

    k = EdgeKey(slab_index=0, side="bot", piece_index=3, edge_index=2)
    assert k.edge_index == 2


def test_vertex_key_includes_corner_index():
    from meshwell.structured.spec import VertexKey

    k = VertexKey(slab_index=0, side="bot", piece_index=0, corner_index=4)
    assert k.corner_index == 4


def test_lateral_key_fields():
    from meshwell.structured.spec import LateralKey

    k = LateralKey(slab_index=1, outer_edge_index=2)
    assert k.slab_index == 1
    assert k.outer_edge_index == 2


def test_side_only_accepts_bot_or_top_runtime():
    """Side is a typing Literal; runtime acceptance of any str is fine — type checker enforces it."""
    from meshwell.structured.spec import FaceKey

    # Pydantic would reject; plain dataclass does not. Just confirm we can construct.
    k = FaceKey(slab_index=0, side="bot", piece_index=0)
    assert k.side == "bot"


def test_phantom_map_defaults_to_empty_dicts():
    from meshwell.structured.spec import PhantomMap

    m = PhantomMap()
    assert m.output_faces == {}
    assert m.output_edges == {}
    assert m.output_vertices == {}
    assert m.output_laterals == {}


def test_phantom_shape_holds_solid_and_keys():
    """PhantomShape carries the OCP solid + the input-tag bookkeeping for one piece."""
    from meshwell.structured.spec import (
        EdgeKey,
        FaceKey,
        PhantomShape,
        VertexKey,
    )

    # We don't need a real TopoDS here — use a sentinel object.
    sentinel = object()
    s = PhantomShape(
        slab_index=0,
        piece_index=2,
        solid=sentinel,
        input_faces_by_key={
            FaceKey(0, "bot", 2): "fake_face_tag_id_a",
            FaceKey(0, "top", 2): "fake_face_tag_id_b",
        },
        input_edges_by_key={
            EdgeKey(0, "bot", 2, 0): "fake_edge_tag_id_a",
        },
        input_vertices_by_key={
            VertexKey(0, "bot", 2, 0): "fake_vertex_tag_id_a",
        },
        input_laterals_by_outer_edge={0: "fake_lateral_a"},
    )
    assert s.solid is sentinel
    assert FaceKey(0, "bot", 2) in s.input_faces_by_key


def test_phantom_build_result_aggregates_shapes():
    from meshwell.structured.spec import PhantomBuildResult, PhantomShape

    s = PhantomShape(
        slab_index=0,
        piece_index=0,
        solid=object(),
        input_faces_by_key={},
        input_edges_by_key={},
        input_vertices_by_key={},
        input_laterals_by_outer_edge={},
    )
    r = PhantomBuildResult(shapes=(s,))
    assert r.shapes == (s,)
```

- [ ] **Step 2: Run; expect ImportError**

Run: `.venv/bin/python -m pytest tests/structured/test_phantom_keys.py -v`

Expected: all tests fail with `ImportError: cannot import name 'FaceKey'` (or similar) from `meshwell.structured.spec`.

- [ ] **Step 3: Add the dataclasses to `meshwell/structured/spec.py`**

Append (after the existing `StructuredPlan` definition):

```python
from typing import Any, Literal

Side = Literal["bot", "top"]


@dataclass(frozen=True)
class FaceKey:
    """Identifies an input face by slab/side/piece. Survives BOP because it indexes
    by piece identity, not by OCC tag.
    """
    slab_index: int
    side: Side
    piece_index: int


@dataclass(frozen=True)
class EdgeKey:
    """Identifies an input boundary edge on a piece face."""
    slab_index: int
    side: Side
    piece_index: int
    edge_index: int


@dataclass(frozen=True)
class VertexKey:
    """Identifies an input boundary corner on a piece face."""
    slab_index: int
    side: Side
    piece_index: int
    corner_index: int


@dataclass(frozen=True)
class LateralKey:
    """Identifies an input lateral face on a slab. ``outer_edge_index``
    indexes into the slab's union-footprint outer boundary.
    """
    slab_index: int
    outer_edge_index: int


@dataclass
class PhantomShape:
    """One partition piece's input OCC bookkeeping.

    ``solid`` is the TopoDS_Solid produced by ``BRepPrimAPI_MakePrism``.
    The four ``input_*`` dicts map our Phase-2 key types to the
    corresponding input OCC sub-shapes (TopoDS_Face / TopoDS_Edge /
    TopoDS_Vertex), captured at construction time so we can ask BOP
    history what they became.

    The values are OCC ``TopoDS_*`` objects (declared ``Any`` to avoid
    importing OCP at type-check time when callers may not have it).
    """
    slab_index: int
    piece_index: int
    solid: Any
    input_faces_by_key: dict[FaceKey, Any]
    input_edges_by_key: dict[EdgeKey, Any]
    input_vertices_by_key: dict[VertexKey, Any]
    # Lateral faces are not piece-scoped on the union footprint — they're
    # slab-scoped. But each piece may own zero or more lateral faces
    # (those that coincide with an outer-boundary edge segment). The dict
    # is keyed by the outer_edge_index they map to.
    input_laterals_by_outer_edge: dict[int, Any]


@dataclass(frozen=True)
class PhantomBuildResult:
    """Output of ``build_phantom_shapes(plan)``.

    Contains one ``PhantomShape`` per (slab, piece) pair, in deterministic
    order (slab_index ascending, then piece_index ascending).
    """
    shapes: tuple[PhantomShape, ...]


@dataclass
class PhantomMap:
    """Post-BOP correspondence map.

    Each input element maps to a *list* of output OCC tags because
    BOP may split a single input into many output shapes (e.g. a
    neighbour cut a piece's top face into 3 sub-faces).
    """
    output_faces: dict[FaceKey, list[Any]] = field(default_factory=dict)
    output_edges: dict[EdgeKey, list[Any]] = field(default_factory=dict)
    output_vertices: dict[VertexKey, list[Any]] = field(default_factory=dict)
    output_laterals: dict[LateralKey, list[Any]] = field(default_factory=dict)
```

- [ ] **Step 4: Run; expect all 8 tests pass**

Run: `.venv/bin/python -m pytest tests/structured/test_phantom_keys.py -v`

Expected: 8 PASSES. The existing 30 tests in `tests/structured/` should also still pass.

Run: `.venv/bin/python -m pytest tests/structured/ -v` → expect 38/38 PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/spec.py tests/structured/test_phantom_keys.py
git commit -m "$(cat <<'EOF'
feat(structured): Phase-2 key dataclasses (FaceKey/EdgeKey/VertexKey/LateralKey + PhantomShape/Map)

Adds the 7 dataclasses Layer B will use to bookkeep OCC input tags
through the global BOP. Keys are frozen (hashable, usable as dict
keys); PhantomShape and PhantomMap are mutable since builders populate
them in place. PhantomMap value-lists are list[TopoDS_*] because
BOP Modified() can split one input into many outputs.

No phantom.py module yet — that lands in Task 2-6.
EOF
)"
```

---

## Task 2: Helper module `tests/structured/_occ_helpers.py`

**Files:**
- Create: `tests/structured/_occ_helpers.py`
- Test: a small `tests/structured/test_occ_helpers.py` to make sure the helpers themselves work (we'll lean on them in Tasks 3-6 — they need to be solid).

- [ ] **Step 1: Write the failing test**

Create `tests/structured/test_occ_helpers.py`:

```python
"""Smoke test for the OCC test helpers."""
from __future__ import annotations


def test_make_box_returns_solid_with_expected_topology():
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX

    from tests.structured._occ_helpers import count_subshapes, make_box

    box = make_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    assert count_subshapes(box, TopAbs_SOLID) == 1
    assert count_subshapes(box, TopAbs_FACE) == 6
    assert count_subshapes(box, TopAbs_EDGE) == 12
    assert count_subshapes(box, TopAbs_VERTEX) == 8


def test_make_stick_returns_taller_box():
    from OCP.TopAbs import TopAbs_FACE

    from tests.structured._occ_helpers import count_subshapes, make_stick

    stick = make_stick(0.4, 0.4, -0.1, 1.1, 0.2, 0.2)
    assert count_subshapes(stick, TopAbs_FACE) == 6


def test_list_of_shape_iteration():
    """_list_of_shape_to_list converts OCP iterators to plain Python lists."""
    from OCP.TopTools import TopTools_ListOfShape

    from tests.structured._occ_helpers import _list_of_shape_to_list

    empty = TopTools_ListOfShape()
    assert _list_of_shape_to_list(empty) == []
```

- [ ] **Step 2: Run; expect ImportError**

Run: `.venv/bin/python -m pytest tests/structured/test_occ_helpers.py -v`

Expected: tests fail with `ModuleNotFoundError`.

- [ ] **Step 3: Create `tests/structured/_occ_helpers.py`**

```python
"""Shared OCP test fixtures for the structured-polyprism test suite.

Underscore prefix marks this as private to tests/structured/.
"""
from __future__ import annotations

from typing import Any


def make_box(x0: float, y0: float, z0: float, dx: float, dy: float, dz: float) -> Any:
    """Return a TopoDS_Solid for an axis-aligned box."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    return BRepPrimAPI_MakeBox(gp_Pnt(x0, y0, z0), gp_Pnt(x0 + dx, y0 + dy, z0 + dz)).Solid()


def make_stick(x0: float, y0: float, z_lo: float, z_hi: float, dx: float, dy: float) -> Any:
    """Convenience: a tall thin box for through-cut tests."""
    return make_box(x0, y0, z_lo, dx, dy, z_hi - z_lo)


def count_subshapes(shape: Any, sub_type: Any) -> int:
    """Count distinct sub-shapes of ``sub_type`` (TopAbs_FACE etc.) in shape."""
    from OCP.TopExp import TopExp_Explorer

    exp = TopExp_Explorer(shape, sub_type)
    n = 0
    while exp.More():
        n += 1
        exp.Next()
    return n


def _list_of_shape_to_list(lst: Any) -> list[Any]:
    """Convert an OCP TopTools_ListOfShape to a plain Python list of TopoDS_*."""
    from OCP.TopTools import TopTools_ListIteratorOfListOfShape

    it = TopTools_ListIteratorOfListOfShape(lst)
    out: list[Any] = []
    while it.More():
        out.append(it.Value())
        it.Next()
    return out
```

Also create `tests/structured/__init__.py` if missing — but it already exists from Phase 1. Just confirm with `ls tests/structured/__init__.py`.

- [ ] **Step 4: Run; expect 3 PASSES**

Run: `.venv/bin/python -m pytest tests/structured/test_occ_helpers.py -v`

Expected: 3 PASSES.

- [ ] **Step 5: Commit**

```bash
git add tests/structured/_occ_helpers.py tests/structured/test_occ_helpers.py
git commit -m "$(cat <<'EOF'
test(structured): shared OCP fixtures (_occ_helpers) for Phase-2 phantom tests

Adds make_box, make_stick, count_subshapes, and a
_list_of_shape_to_list converter that Phase-2 phantom tests will reuse.
The helpers wrap OCP imports defensively so this module imports
without OCP at collection time when needed.
EOF
)"
```

---

## Task 3: `phantom._make_face_from_polygon` — shapely → OCC face

**Files:**
- Create: `meshwell/structured/phantom.py`
- Test: `tests/structured/test_phantom_shapes.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/structured/test_phantom_shapes.py`:

```python
"""Tests for meshwell.structured.phantom shape construction."""
from __future__ import annotations

import pytest
from shapely.geometry import Polygon


def _unit_square() -> Polygon:
    return Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


def _square_with_hole() -> Polygon:
    outer = [(0, 0), (4, 0), (4, 4), (0, 4)]
    hole = [(1, 1), (2, 1), (2, 2), (1, 2)]
    return Polygon(outer, [hole])


def test_make_face_returns_topods_face_at_z():
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_VERTEX

    from meshwell.structured.phantom import _make_face_from_polygon
    from tests.structured._occ_helpers import count_subshapes

    face = _make_face_from_polygon(_unit_square(), z=2.5)
    # A square face has 4 edges and 4 vertices.
    assert count_subshapes(face, TopAbs_EDGE) == 4
    assert count_subshapes(face, TopAbs_VERTEX) == 8  # each edge has 2 vertices (shared)


def test_make_face_with_hole_has_two_wires():
    from OCP.TopAbs import TopAbs_WIRE

    from meshwell.structured.phantom import _make_face_from_polygon
    from tests.structured._occ_helpers import count_subshapes

    face = _make_face_from_polygon(_square_with_hole(), z=0.0)
    # 1 outer wire + 1 inner wire.
    assert count_subshapes(face, TopAbs_WIRE) == 2


def test_make_face_canonicalizes_orientation():
    """A CW-ordered shapely polygon (reversed) still produces a valid face."""
    from meshwell.structured.phantom import _make_face_from_polygon

    cw = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])  # CW exterior
    face = _make_face_from_polygon(cw, z=0.0)
    assert face is not None  # didn't crash; the canonicalize step flipped it


def test_make_face_z_is_respected():
    """The face sits at the requested z plane."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface

    from meshwell.structured.phantom import _make_face_from_polygon

    face = _make_face_from_polygon(_unit_square(), z=7.0)
    # Sample the surface at (u,v)=(0.5, 0.5).
    surf = BRepAdaptor_Surface(face)
    u_mid = 0.5 * (surf.FirstUParameter() + surf.LastUParameter())
    v_mid = 0.5 * (surf.FirstVParameter() + surf.LastVParameter())
    pnt = surf.Value(u_mid, v_mid)
    assert pnt.Z() == pytest.approx(7.0)
```

- [ ] **Step 2: Run; expect ImportError**

Run: `.venv/bin/python -m pytest tests/structured/test_phantom_shapes.py -v`

Expected: `ModuleNotFoundError: No module named 'meshwell.structured.phantom'`.

- [ ] **Step 3: Create `meshwell/structured/phantom.py` with the helper**

```python
"""Phase-2: CAD-stage phantom shape construction + BOP-history-based PhantomMap.

The two public entry points are:

- :func:`build_phantom_shapes` — turn a ``StructuredPlan`` into
  per-piece OCP sub-prisms, recording input OCC tags into
  ``PhantomBuildResult``.
- :func:`extract_phantom_map` — given a post-Perform
  ``BOPAlgo_Builder`` (or any builder exposing the Modified() /
  Generated() / IsDeleted() history API), walk the recorded input
  tags to produce the ``PhantomMap``.

Phase 2 does not integrate with ``cad_occ`` (that's Phase 3). All
tests here use OCP directly with handcrafted scenes.
"""
from __future__ import annotations

from typing import Any

from shapely.geometry import Polygon
from shapely.geometry.polygon import orient


def _make_face_from_polygon(polygon: Polygon, z: float) -> Any:
    """Build a planar TopoDS_Face at the given z from a shapely Polygon.

    Handles interior holes (rings) by adding each as a reversed wire to
    the face builder. Forces CCW exterior + CW interior orientation to
    match OCC convention.
    """
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakeWire,
    )
    from OCP.gp import gp_Pnt

    # Canonicalize orientation: shapely's `orient` returns a polygon with
    # CCW exterior and CW interiors when sign=1.0, matching OCC convention.
    poly = orient(polygon, sign=1.0)

    def _wire_from_coords(coords: list[tuple[float, float]]) -> Any:
        if coords[0] == coords[-1]:
            coords = coords[:-1]
        wire_builder = BRepBuilderAPI_MakeWire()
        for i in range(len(coords)):
            p1 = gp_Pnt(coords[i][0], coords[i][1], z)
            p2 = gp_Pnt(coords[(i + 1) % len(coords)][0], coords[(i + 1) % len(coords)][1], z)
            edge = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
            wire_builder.Add(edge)
        return wire_builder.Wire()

    outer_wire = _wire_from_coords(list(poly.exterior.coords))
    face_builder = BRepBuilderAPI_MakeFace(outer_wire)
    for ring in poly.interiors:
        inner_wire = _wire_from_coords(list(ring.coords))
        face_builder.Add(inner_wire)
    return face_builder.Face()
```

- [ ] **Step 4: Run; expect 4 PASSES**

Run: `.venv/bin/python -m pytest tests/structured/test_phantom_shapes.py -v`

Expected: 4 PASSES.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/phantom.py tests/structured/test_phantom_shapes.py
git commit -m "$(cat <<'EOF'
feat(structured): phantom._make_face_from_polygon (shapely -> OCP face)

Builds a planar TopoDS_Face at fixed z from a shapely Polygon,
handling interior holes and forcing CCW-exterior/CW-interior
orientation per OCC convention. First building block for Phase-2
phantom sub-prism construction.
EOF
)"
```

---

## Task 4: `phantom._build_sub_prism` — single piece with input-tag bookkeeping

**Files:**
- Modify: `meshwell/structured/phantom.py`
- Modify: `tests/structured/test_phantom_shapes.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/structured/test_phantom_shapes.py`:

```python
def test_build_sub_prism_returns_solid_with_expected_topology():
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID

    from meshwell.structured.phantom import _build_sub_prism
    from tests.structured._occ_helpers import count_subshapes

    out = _build_sub_prism(_unit_square(), zlo=0.0, zhi=1.0)
    assert count_subshapes(out.solid, TopAbs_SOLID) == 1
    # 1 bottom + 1 top + 4 lateral = 6 faces.
    assert count_subshapes(out.solid, TopAbs_FACE) == 6


def test_build_sub_prism_records_bottom_and_top_face_keys():
    """The returned record knows which face is bottom and which is top, by key."""
    from meshwell.structured.phantom import _build_sub_prism
    from meshwell.structured.spec import FaceKey

    out = _build_sub_prism(_unit_square(), zlo=0.0, zhi=1.0, slab_index=2, piece_index=3)
    assert FaceKey(slab_index=2, side="bot", piece_index=3) in out.input_faces_by_key
    assert FaceKey(slab_index=2, side="top", piece_index=3) in out.input_faces_by_key


def test_build_sub_prism_records_lateral_faces_per_outer_edge():
    """One lateral face per outer-edge segment, indexed by edge_index."""
    from meshwell.structured.phantom import _build_sub_prism

    out = _build_sub_prism(_unit_square(), zlo=0.0, zhi=1.0)
    # Unit square has 4 outer edges -> 4 lateral faces.
    assert len(out.input_laterals_by_outer_edge) == 4
    assert set(out.input_laterals_by_outer_edge.keys()) == {0, 1, 2, 3}


def test_build_sub_prism_with_hole_records_extra_lateral_faces():
    """A face with a hole produces lateral faces for both outer and inner edges."""
    from meshwell.structured.phantom import _build_sub_prism

    out = _build_sub_prism(_square_with_hole(), zlo=0.0, zhi=1.0)
    # 4 outer + 4 inner = 8 lateral faces total, but we only key the
    # outer ones (Layer A's outer-only contract).
    assert len(out.input_laterals_by_outer_edge) == 4


def test_build_sub_prism_records_bottom_edge_keys():
    """Bottom edge keys cover all bottom face boundary segments."""
    from meshwell.structured.phantom import _build_sub_prism
    from meshwell.structured.spec import EdgeKey

    out = _build_sub_prism(_unit_square(), zlo=0.0, zhi=1.0, slab_index=0, piece_index=0)
    bot_edges = {
        k for k in out.input_edges_by_key if k.side == "bot"
    }
    # 4 outer edges on a square.
    assert len(bot_edges) == 4
    # All have piece_index=0.
    assert all(k.piece_index == 0 for k in bot_edges)


def test_build_sub_prism_records_bottom_vertex_keys():
    from meshwell.structured.phantom import _build_sub_prism
    from meshwell.structured.spec import VertexKey

    out = _build_sub_prism(_unit_square(), zlo=0.0, zhi=1.0, slab_index=0, piece_index=0)
    bot_verts = {
        k for k in out.input_vertices_by_key if k.side == "bot"
    }
    assert len(bot_verts) == 4
```

- [ ] **Step 2: Run; expect ImportError on `_build_sub_prism`**

Run: `.venv/bin/python -m pytest tests/structured/test_phantom_shapes.py -v -k build_sub_prism`

Expected: 6 tests fail with `ImportError`.

- [ ] **Step 3: Implement `_build_sub_prism`**

Append to `meshwell/structured/phantom.py`:

```python
from shapely.geometry.polygon import orient

from meshwell.structured.spec import (
    EdgeKey,
    FaceKey,
    PhantomShape,
    VertexKey,
)


def _build_sub_prism(
    piece: Polygon,
    zlo: float,
    zhi: float,
    slab_index: int = 0,
    piece_index: int = 0,
) -> PhantomShape:
    """Build one OCP sub-prism for a single partition piece.

    Returns a :class:`PhantomShape` carrying:

    - The TopoDS_Solid produced by extruding the piece face from zlo to zhi.
    - The input OCC tags for bottom face, top face, outer-edge edges,
      outer-edge vertices, and lateral faces — keyed by our Phase-2 key
      types so the post-BOP map can index them.

    Inner-ring edges/vertices are NOT keyed (Layer A's outer-only
    contract: lateral OCC faces are 4-corner on the outer boundary; hole
    boundaries are not in the structured pipeline's correspondence map).
    """
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer
    from OCP.gp import gp_Vec

    height = zhi - zlo
    poly = orient(piece, sign=1.0)
    bottom_face = _make_face_from_polygon(poly, z=zlo)
    prism_builder = BRepPrimAPI_MakePrism(bottom_face, gp_Vec(0.0, 0.0, height))
    solid = prism_builder.Shape()
    top_face = prism_builder.LastShape()

    # Record bottom + top faces.
    input_faces: dict[FaceKey, Any] = {
        FaceKey(slab_index, "bot", piece_index): bottom_face,
        FaceKey(slab_index, "top", piece_index): top_face,
    }

    # Iterate the bottom face's OUTER wire (exterior only — we skip holes
    # for the outer-only contract). We use the polygon's exterior coords
    # to know the edge order: edge i goes from coord i to coord i+1.
    outer_coords = list(poly.exterior.coords)
    if outer_coords[0] == outer_coords[-1]:
        outer_coords = outer_coords[:-1]
    n_outer = len(outer_coords)

    # Collect bottom-face outer edges + their lateral-face counterparts.
    # We walk TopExp_Explorer on the bottom face to enumerate all its
    # edges, then filter to the outer N (skipping any inner-ring edges).
    bot_edges_all = []
    edge_explorer = TopExp_Explorer(bottom_face, TopAbs_EDGE)
    while edge_explorer.More():
        bot_edges_all.append(edge_explorer.Current())
        edge_explorer.Next()
    # The first n_outer edges (by traversal order) belong to the outer
    # wire. We trust BRepBuilderAPI_MakeFace's wire-addition order
    # (outer first, then interiors).
    bot_outer_edges = bot_edges_all[:n_outer]

    input_edges: dict[EdgeKey, Any] = {}
    input_laterals: dict[int, Any] = {}
    for edge_i, bot_edge in enumerate(bot_outer_edges):
        input_edges[EdgeKey(slab_index, "bot", piece_index, edge_i)] = bot_edge
        # The prism builder knows which lateral face was generated from
        # this input edge.
        lateral_face = prism_builder.GeneratedShape(bot_edge)
        input_laterals[edge_i] = lateral_face

    # Top face's outer edges, in the same order (the prism generates one
    # top edge per bottom outer edge, accessible via .Generated() on the
    # bottom edge's vertices). For simplicity we enumerate top-face edges
    # via TopExp_Explorer and trust the same outer-first order, then map
    # 1:1 to bottom edges by index.
    top_edges_all = []
    edge_explorer = TopExp_Explorer(top_face, TopAbs_EDGE)
    while edge_explorer.More():
        top_edges_all.append(edge_explorer.Current())
        edge_explorer.Next()
    top_outer_edges = top_edges_all[:n_outer]
    for edge_i, top_edge in enumerate(top_outer_edges):
        input_edges[EdgeKey(slab_index, "top", piece_index, edge_i)] = top_edge

    # Vertices: walk the bottom face's outer-wire vertices in order. Same
    # for top face.
    bot_verts_all = []
    vert_explorer = TopExp_Explorer(bottom_face, TopAbs_VERTEX)
    while vert_explorer.More():
        bot_verts_all.append(vert_explorer.Current())
        vert_explorer.Next()
    # TopExp_Explorer over a face's vertices yields 2 per edge (start, end)
    # so we'll see 2*n_outer entries for the outer wire, possibly with
    # duplicates. Deduplicate by IsSame, then take the first n_outer.
    seen: list[Any] = []
    for v in bot_verts_all:
        if not any(v.IsSame(s) for s in seen):
            seen.append(v)
    bot_outer_verts = seen[:n_outer]

    input_vertices: dict[VertexKey, Any] = {}
    for corner_i, bot_vert in enumerate(bot_outer_verts):
        input_vertices[VertexKey(slab_index, "bot", piece_index, corner_i)] = bot_vert

    # Top vertices: same approach on the top face.
    top_verts_all = []
    vert_explorer = TopExp_Explorer(top_face, TopAbs_VERTEX)
    while vert_explorer.More():
        top_verts_all.append(vert_explorer.Current())
        vert_explorer.Next()
    seen = []
    for v in top_verts_all:
        if not any(v.IsSame(s) for s in seen):
            seen.append(v)
    top_outer_verts = seen[:n_outer]
    for corner_i, top_vert in enumerate(top_outer_verts):
        input_vertices[VertexKey(slab_index, "top", piece_index, corner_i)] = top_vert

    return PhantomShape(
        slab_index=slab_index,
        piece_index=piece_index,
        solid=solid,
        input_faces_by_key=input_faces,
        input_edges_by_key=input_edges,
        input_vertices_by_key=input_vertices,
        input_laterals_by_outer_edge=input_laterals,
    )
```

- [ ] **Step 4: Run; expect 6 PASSES on build_sub_prism tests**

Run: `.venv/bin/python -m pytest tests/structured/test_phantom_shapes.py -v -k build_sub_prism`

Expected: 6 PASSES. Then run the full file: `.venv/bin/python -m pytest tests/structured/test_phantom_shapes.py -v` → 10 PASSES total.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/phantom.py tests/structured/test_phantom_shapes.py
git commit -m "$(cat <<'EOF'
feat(structured): phantom._build_sub_prism with input-tag bookkeeping

For one partition piece, build a TopoDS_Solid via BRepPrimAPI_MakePrism
and record the input OCC tags for: bottom face, top face, outer-edge
edges (bottom + top), outer-edge vertices (bottom + top), and lateral
faces (one per outer edge). All keyed by Phase-2 FaceKey/EdgeKey/
VertexKey/(outer_edge_index) so the post-BOP map can index them.

Inner-ring (hole) edges/vertices/laterals are NOT keyed — Layer A's
outer-only contract.
EOF
)"
```

---

## Task 5: `phantom.build_phantom_shapes(plan)` — aggregate all slabs

**Files:**
- Modify: `meshwell/structured/phantom.py`
- Modify: `tests/structured/test_phantom_shapes.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/structured/test_phantom_shapes.py`:

```python
def test_build_phantom_shapes_empty_plan_returns_empty_result():
    from meshwell.structured.phantom import build_phantom_shapes
    from meshwell.structured.spec import StructuredPlan

    plan = StructuredPlan(slabs=(), z_planes=(), overlaps=())
    result = build_phantom_shapes(plan)
    assert result.shapes == ()


def test_build_phantom_shapes_one_slab_one_piece():
    """Single slab with a one-piece partition yields one PhantomShape."""
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan
    from meshwell.structured.phantom import build_phantom_shapes
    from meshwell.polyprism import PolyPrism

    s = PolyPrism(
        polygons=_unit_square(),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[3])],
        physical_name="s",
    )
    plan = build_plan([s])
    result = build_phantom_shapes(plan)
    assert len(result.shapes) == 1
    assert result.shapes[0].slab_index == 0
    assert result.shapes[0].piece_index == 0


def test_build_phantom_shapes_multi_piece_partition():
    """A slab with a 2-piece face_partition yields 2 PhantomShapes."""
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan
    from meshwell.structured.phantom import build_phantom_shapes
    from meshwell.polyprism import PolyPrism

    # 4x4 structured square; a non-structured neighbour on top covers half.
    s = PolyPrism(
        polygons=Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[3])],
        physical_name="s",
    )
    n = PolyPrism(
        polygons=Polygon([(0, 0), (2, 0), (2, 4), (0, 4)]),
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="n",
    )
    plan = build_plan([s, n])
    # face_partition should have 2 pieces (covered + uncovered halves).
    assert len(plan.slabs[0].face_partition) == 2

    result = build_phantom_shapes(plan)
    assert len(result.shapes) == 2
    assert {s.slab_index for s in result.shapes} == {0}
    assert sorted(s.piece_index for s in result.shapes) == [0, 1]


def test_build_phantom_shapes_is_deterministic_ordering():
    """Output ordering is (slab_index, piece_index) ascending."""
    from meshwell.structured import StructuredExtrusionResolutionSpec, build_plan
    from meshwell.structured.phantom import build_phantom_shapes
    from meshwell.polyprism import PolyPrism

    # Two disjoint slabs.
    s0 = PolyPrism(
        polygons=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="s0",
    )
    s1 = PolyPrism(
        polygons=Polygon([(10, 0), (11, 0), (11, 1), (10, 1)]),
        buffers={0.0: 0.0, 1.0: 0.0},
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
        physical_name="s1",
    )
    plan = build_plan([s0, s1])
    result = build_phantom_shapes(plan)
    indices = [(s.slab_index, s.piece_index) for s in result.shapes]
    assert indices == sorted(indices)
```

- [ ] **Step 2: Run; expect 4 ImportErrors on build_phantom_shapes**

Run: `.venv/bin/python -m pytest tests/structured/test_phantom_shapes.py -v -k build_phantom_shapes`

- [ ] **Step 3: Implement `build_phantom_shapes`**

Append to `meshwell/structured/phantom.py`:

```python
from meshwell.structured.spec import (
    PhantomBuildResult,
    StructuredPlan,
)


def build_phantom_shapes(plan: StructuredPlan) -> PhantomBuildResult:
    """For each slab, build one OCP sub-prism per partition piece.

    Returns a :class:`PhantomBuildResult` with shapes in
    (slab_index, piece_index) ascending order for deterministic
    downstream processing.
    """
    shapes: list[PhantomShape] = []
    for slab_index, slab in enumerate(plan.slabs):
        if not slab.face_partition:
            # Defensive: a planner-produced StructuredPlan should always
            # have face_partition populated (Phase 1 guarantees at least
            # one piece). Skip silently if not.
            continue
        for piece_index, piece in enumerate(slab.face_partition):
            shapes.append(
                _build_sub_prism(
                    piece=piece,
                    zlo=slab.zlo,
                    zhi=slab.zhi,
                    slab_index=slab_index,
                    piece_index=piece_index,
                )
            )
    return PhantomBuildResult(shapes=tuple(shapes))
```

- [ ] **Step 4: Run; expect 4 PASSES**

Run: `.venv/bin/python -m pytest tests/structured/test_phantom_shapes.py -v -k build_phantom_shapes`

Expected: 4 PASSES. Then full file: `.venv/bin/python -m pytest tests/structured/test_phantom_shapes.py -v` → 14 PASSES.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/phantom.py tests/structured/test_phantom_shapes.py
git commit -m "$(cat <<'EOF'
feat(structured): phantom.build_phantom_shapes(plan) -> PhantomBuildResult

Aggregates _build_sub_prism over every (slab, piece) pair in a
StructuredPlan, returning shapes in (slab_index, piece_index)
ascending order for deterministic downstream processing.

End of Phase-2 Layer A (shape construction). Layer B (PhantomMap from
BOP history) lands in Tasks 6-7.
EOF
)"
```

---

## Task 6: `phantom.extract_phantom_map(build_result, builder)` — Layer B

**Files:**
- Modify: `meshwell/structured/phantom.py`
- Create: `tests/structured/test_phantom_history.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/structured/test_phantom_history.py`:

```python
"""Tests for meshwell.structured.phantom.extract_phantom_map.

Tests construct a small BOP scene with OCP directly:
  - One phantom sub-prism (a 1x1x1 box).
  - One neighbour (a 0.3x0.3 stick passing through, or a half-cover top box).
We run BOPAlgo_Builder, then call extract_phantom_map, then assert the
output map's structure.
"""
from __future__ import annotations

import pytest
from shapely.geometry import Polygon


def _square(x=0, y=0, w=1, h=1) -> Polygon:
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def _run_bop(*shapes):
    """Run BOPAlgo_Builder on the given input shapes; return the builder."""
    from OCP.BOPAlgo import BOPAlgo_Builder

    builder = BOPAlgo_Builder()
    for s in shapes:
        builder.AddArgument(s)
    builder.Perform()
    return builder


def test_extract_phantom_map_no_neighbours_identity():
    """With no neighbour shape, every input maps to itself (one-element list)."""
    from meshwell.structured.phantom import _build_sub_prism, extract_phantom_map
    from meshwell.structured.spec import (
        FaceKey,
        PhantomBuildResult,
    )

    shape = _build_sub_prism(_square(), zlo=0.0, zhi=1.0, slab_index=0, piece_index=0)
    build_result = PhantomBuildResult(shapes=(shape,))
    builder = _run_bop(shape.solid)

    pmap = extract_phantom_map(build_result, builder)
    # Every input face should map to exactly one output (itself).
    assert all(len(v) == 1 for v in pmap.output_faces.values())
    assert FaceKey(0, "bot", 0) in pmap.output_faces
    assert FaceKey(0, "top", 0) in pmap.output_faces


def test_extract_phantom_map_neighbour_cuts_top_face():
    """A neighbour box overlapping the phantom's top half-plane splits the top face."""
    from meshwell.structured.phantom import _build_sub_prism, extract_phantom_map
    from meshwell.structured.spec import FaceKey, PhantomBuildResult
    from tests.structured._occ_helpers import make_box

    # Phantom box from z=0 to z=1 with footprint [0,4]x[0,4].
    shape = _build_sub_prism(
        _square(0, 0, 4, 4), zlo=0.0, zhi=1.0, slab_index=0, piece_index=0
    )
    # Neighbour box sits z=[1,2] over half the phantom: covers x in [0,2], y in [0,4].
    # Its bottom face at z=1 partially-covers the phantom's top face at z=1.
    neighbour = make_box(0, 0, 1, 2, 4, 1)
    build_result = PhantomBuildResult(shapes=(shape,))
    builder = _run_bop(shape.solid, neighbour)

    pmap = extract_phantom_map(build_result, builder)
    top_key = FaceKey(0, "top", 0)
    # The top face was cut into 2 pieces by the neighbour bottom-face boundary.
    assert len(pmap.output_faces[top_key]) >= 2, (
        f"Expected the top face to split, got {len(pmap.output_faces[top_key])} pieces"
    )
    # The bottom face was untouched.
    bot_key = FaceKey(0, "bot", 0)
    assert len(pmap.output_faces[bot_key]) == 1


def test_extract_phantom_map_lateral_face_through_cut():
    """A stick passing through the phantom cuts its lateral faces."""
    from meshwell.structured.phantom import _build_sub_prism, extract_phantom_map
    from meshwell.structured.spec import LateralKey, PhantomBuildResult
    from tests.structured._occ_helpers import make_stick

    shape = _build_sub_prism(
        _square(0, 0, 4, 4), zlo=0.0, zhi=1.0, slab_index=0, piece_index=0
    )
    # Stick passes through the phantom from z=-1 to z=2, intersecting one
    # lateral face (say the x=4 face) — no, the stick is INSIDE the phantom.
    # Let's overlap a stick that crosses the y=4 lateral face: x in [1,3], y in [3.5,4.5].
    stick = make_stick(1.0, 3.5, -0.5, 1.5, 2.0, 1.0)
    build_result = PhantomBuildResult(shapes=(shape,))
    builder = _run_bop(shape.solid, stick)

    pmap = extract_phantom_map(build_result, builder)
    # At least one lateral output_laterals entry should have more than 1
    # output face (the cut one). We don't assert WHICH one, just that the
    # mechanism works.
    cut_counts = [len(v) for v in pmap.output_laterals.values()]
    assert any(c > 1 for c in cut_counts), (
        f"Expected at least one lateral face to be cut, got counts {cut_counts}"
    )


def test_extract_phantom_map_records_all_input_keys():
    """Every key in the PhantomShape's input_*_by_key should appear in the map."""
    from meshwell.structured.phantom import _build_sub_prism, extract_phantom_map
    from meshwell.structured.spec import PhantomBuildResult

    shape = _build_sub_prism(_square(), zlo=0.0, zhi=1.0, slab_index=0, piece_index=0)
    builder = _run_bop(shape.solid)
    pmap = extract_phantom_map(PhantomBuildResult(shapes=(shape,)), builder)

    assert set(pmap.output_faces.keys()) == set(shape.input_faces_by_key.keys())
    assert set(pmap.output_edges.keys()) == set(shape.input_edges_by_key.keys())
    assert set(pmap.output_vertices.keys()) == set(shape.input_vertices_by_key.keys())
    # Laterals are keyed by LateralKey in the map (NOT by raw outer_edge_index).
    from meshwell.structured.spec import LateralKey
    expected_lateral_keys = {
        LateralKey(slab_index=0, outer_edge_index=i)
        for i in shape.input_laterals_by_outer_edge
    }
    assert set(pmap.output_laterals.keys()) == expected_lateral_keys
```

- [ ] **Step 2: Run; expect 4 ImportError on extract_phantom_map**

Run: `.venv/bin/python -m pytest tests/structured/test_phantom_history.py -v`

Expected: 4 fail with `ImportError: cannot import name 'extract_phantom_map'`.

- [ ] **Step 3: Implement `extract_phantom_map`**

Append to `meshwell/structured/phantom.py`:

```python
from meshwell.structured.spec import (
    LateralKey,
    PhantomMap,
)


def _modified_or_unchanged(builder: Any, input_shape: Any) -> list[Any]:
    """Return list of output shapes for input_shape.

    Mirrors the cad_occ.py pattern (line 350-354): if Modified() is empty
    AND the shape is not deleted, the shape passed through unchanged
    (input == output, one element). Otherwise Modified() gives the
    actual successor list.
    """
    from OCP.TopTools import TopTools_ListIteratorOfListOfShape

    modified = builder.Modified(input_shape)
    if modified.IsEmpty():
        if builder.IsDeleted(input_shape):
            return []
        return [input_shape]
    out: list[Any] = []
    it = TopTools_ListIteratorOfListOfShape(modified)
    while it.More():
        out.append(it.Value())
        it.Next()
    return out


def extract_phantom_map(
    build_result: PhantomBuildResult,
    builder: Any,
) -> PhantomMap:
    """Walk the post-Perform BOP history to build the PhantomMap.

    For every input OCC tag recorded in ``build_result``, ask the
    ``builder`` (a ``BOPAlgo_Builder`` or any object exposing
    ``Modified(shape)`` / ``IsDeleted(shape)``) what the input became
    in the output.

    Args:
        build_result: From :func:`build_phantom_shapes`.
        builder: Post-Perform BOP builder.

    Returns:
        :class:`PhantomMap` with all four output_*_by_key dicts
        populated. Each value is a list because a single input can
        split into many outputs.
    """
    pmap = PhantomMap()
    for shape in build_result.shapes:
        for face_key, in_face in shape.input_faces_by_key.items():
            pmap.output_faces[face_key] = _modified_or_unchanged(builder, in_face)
        for edge_key, in_edge in shape.input_edges_by_key.items():
            pmap.output_edges[edge_key] = _modified_or_unchanged(builder, in_edge)
        for vert_key, in_vert in shape.input_vertices_by_key.items():
            pmap.output_vertices[vert_key] = _modified_or_unchanged(builder, in_vert)
        for outer_edge_idx, in_lateral in shape.input_laterals_by_outer_edge.items():
            lateral_key = LateralKey(
                slab_index=shape.slab_index, outer_edge_index=outer_edge_idx
            )
            pmap.output_laterals[lateral_key] = _modified_or_unchanged(
                builder, in_lateral
            )
    return pmap
```

- [ ] **Step 4: Run; expect 4 PASSES**

Run: `.venv/bin/python -m pytest tests/structured/test_phantom_history.py -v`

Expected: 4 PASSES. The first test (no-neighbour identity) is the smoke test. The next two verify face / lateral splits. The fourth verifies all keys are present.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/phantom.py tests/structured/test_phantom_history.py
git commit -m "$(cat <<'EOF'
feat(structured): phantom.extract_phantom_map - Layer B BOP history map

Given a post-Perform BOPAlgo_Builder (or any compatible builder) and
the PhantomBuildResult holding input OCC tags, walk
Modified()/IsDeleted() for every input to produce the PhantomMap.

Each map value is list[TopoDS_*] because BOP can split a single input
into many outputs (e.g. a neighbour cuts a top face into N pieces).
Mirrors the cad_occ.py:350-354 pattern for handling empty Modified()
results.

End of Phase-2 Layer B. Tests directly build OCP scenes; cad_occ
integration is Phase 3.
EOF
)"
```

---

## Task 7: Mid-height cut detection via `Generated()` on lateral faces

**Files:**
- Modify: `meshwell/structured/phantom.py`
- Modify: `meshwell/structured/spec.py` (add `lateral_has_midheight_cut: dict[LateralKey, bool]` to PhantomMap)
- Modify: `tests/structured/test_phantom_history.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/structured/test_phantom_history.py`:

```python
def test_lateral_no_midheight_cut_default():
    """A phantom with no neighbours should have no mid-height cuts."""
    from meshwell.structured.phantom import _build_sub_prism, extract_phantom_map
    from meshwell.structured.spec import PhantomBuildResult

    shape = _build_sub_prism(_square(), zlo=0.0, zhi=1.0, slab_index=0, piece_index=0)
    builder = _run_bop(shape.solid)
    pmap = extract_phantom_map(PhantomBuildResult(shapes=(shape,)), builder)
    # All four laterals should have no mid-height cut.
    assert all(not v for v in pmap.lateral_has_midheight_cut.values())


def test_lateral_midheight_cut_detected_from_partial_neighbour():
    """A neighbour that touches only part of a lateral face is a mid-height cut."""
    from meshwell.structured.phantom import _build_sub_prism, extract_phantom_map
    from meshwell.structured.spec import PhantomBuildResult
    from tests.structured._occ_helpers import make_box

    # Phantom z=[0,2], 4x4 footprint.
    shape = _build_sub_prism(
        _square(0, 0, 4, 4), zlo=0.0, zhi=2.0, slab_index=0, piece_index=0
    )
    # Neighbour box: protrudes into the y=4 lateral face but only at z in [0.5, 1.5].
    # This puts new vertices at z=0.5 and z=1.5 on the lateral face -> mid-height cut.
    neighbour = make_box(1.0, 3.5, 0.5, 2.0, 1.0, 1.0)
    builder = _run_bop(shape.solid, neighbour)
    pmap = extract_phantom_map(PhantomBuildResult(shapes=(shape,)), builder)

    # At least one lateral should have a mid-height cut.
    assert any(pmap.lateral_has_midheight_cut.values()), (
        f"Expected mid-height cut on a lateral, got "
        f"{dict(pmap.lateral_has_midheight_cut)}"
    )
```

- [ ] **Step 2: Run; expect AttributeError on lateral_has_midheight_cut**

Run: `.venv/bin/python -m pytest tests/structured/test_phantom_history.py -v -k midheight`

Expected: tests fail with `AttributeError: 'PhantomMap' object has no attribute 'lateral_has_midheight_cut'`.

- [ ] **Step 3: Add the field to `PhantomMap`**

Edit `meshwell/structured/spec.py`:

In the `PhantomMap` dataclass, append:

```python
    # Per-lateral flag: True iff BOP introduced a new vertex on the
    # lateral face with z strictly between zlo and zhi (a "mid-height
    # cut"). Phase 3's builder uses this to decide which lateral faces
    # to exclude from transfinite hints.
    lateral_has_midheight_cut: dict[LateralKey, bool] = field(default_factory=dict)
```

- [ ] **Step 4: Detect mid-height cuts in `extract_phantom_map`**

In `meshwell/structured/phantom.py`, modify `extract_phantom_map` so that after populating `output_laterals`, it computes the mid-height-cut flag. Add this block after the lateral loop, inside `extract_phantom_map`:

```python
        # Mid-height cut detection: for each lateral face, ask BOP what
        # vertices were Generated() on it. If any generated vertex has
        # z strictly between zlo and zhi (within a small tolerance), the
        # lateral face was cut mid-height.
        slab_zlo, slab_zhi = _slab_z_range_for_shape(build_result, shape)
        for outer_edge_idx, in_lateral in shape.input_laterals_by_outer_edge.items():
            lateral_key = LateralKey(
                slab_index=shape.slab_index, outer_edge_index=outer_edge_idx
            )
            pmap.lateral_has_midheight_cut[lateral_key] = _has_midheight_vertex(
                builder, in_lateral, slab_zlo, slab_zhi
            )
```

Then add the helper functions at module scope:

```python
_MIDHEIGHT_TOL = 1e-7


def _slab_z_range_for_shape(
    build_result: PhantomBuildResult, shape: PhantomShape
) -> tuple[float, float]:
    """Recover (zlo, zhi) for a shape from the build_result.

    PhantomShape doesn't store zlo/zhi directly; we infer it from the
    bottom face's z-coordinate (any vertex of input_faces_by_key with
    side='bot') and the top face's z-coordinate.
    """
    from OCP.BRep import BRep_Tool

    bot_key = FaceKey(slab_index=shape.slab_index, side="bot", piece_index=shape.piece_index)
    top_key = FaceKey(slab_index=shape.slab_index, side="top", piece_index=shape.piece_index)
    bot_face = shape.input_faces_by_key[bot_key]
    top_face = shape.input_faces_by_key[top_key]
    # Get any vertex from each face and read its z.
    from OCP.TopAbs import TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer

    def _any_z(face: Any) -> float:
        exp = TopExp_Explorer(face, TopAbs_VERTEX)
        v = exp.Current()
        return BRep_Tool.Pnt_s(v).Z()

    return _any_z(bot_face), _any_z(top_face)


def _has_midheight_vertex(
    builder: Any, lateral_face: Any, zlo: float, zhi: float
) -> bool:
    """True if BOP.Generated(lateral_face) produced any vertex with zlo < z < zhi."""
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_VERTEX
    from OCP.TopTools import TopTools_ListIteratorOfListOfShape

    generated = builder.Generated(lateral_face)
    if generated.IsEmpty():
        return False
    it = TopTools_ListIteratorOfListOfShape(generated)
    while it.More():
        sub = it.Value()
        if sub.ShapeType() == TopAbs_VERTEX:
            z = BRep_Tool.Pnt_s(sub).Z()
            if zlo + _MIDHEIGHT_TOL < z < zhi - _MIDHEIGHT_TOL:
                return True
        it.Next()
    return False
```

- [ ] **Step 5: Run; expect 2 PASSES**

Run: `.venv/bin/python -m pytest tests/structured/test_phantom_history.py -v -k midheight`

Expected: 2 PASSES. Then run the full file: `.venv/bin/python -m pytest tests/structured/test_phantom_history.py -v` → 6 PASSES.

- [ ] **Step 6: Update the test that checks all map keys are present**

The earlier test `test_extract_phantom_map_records_all_input_keys` now needs to also check `lateral_has_midheight_cut` has the same keys as `output_laterals`. Edit that test:

```python
    assert set(pmap.lateral_has_midheight_cut.keys()) == expected_lateral_keys
```

(Add this assertion at the end of the existing test body.)

Re-run: `.venv/bin/python -m pytest tests/structured/test_phantom_history.py -v` → 6 PASSES.

- [ ] **Step 7: Commit**

```bash
git add meshwell/structured/spec.py meshwell/structured/phantom.py tests/structured/test_phantom_history.py
git commit -m "$(cat <<'EOF'
feat(structured): mid-height cut detection via Generated() on laterals

extract_phantom_map now populates PhantomMap.lateral_has_midheight_cut
by walking BOPAlgo_Builder.Generated() on each input lateral face: any
generated vertex with z strictly between zlo and zhi flags a
mid-height cut. Phase 3's builder will use this flag to exclude such
lateral faces from transfinite hints.

PhantomMap dataclass gets the new dict field; existing tests updated
to verify the key set matches output_laterals.
EOF
)"
```

---

## Task 8: Update `meshwell/structured/__init__.py` exports

**Files:**
- Modify: `meshwell/structured/__init__.py`
- Test: existing `tests/structured/test_package_smoke.py` (already lives there from Phase 1)

- [ ] **Step 1: Add a failing assertion in the smoke test**

Append to `tests/structured/test_package_smoke.py`:

```python
def test_phase2_public_exports():
    """Phase 2 adds build_phantom_shapes / extract_phantom_map / PhantomMap."""
    from meshwell.structured import (
        PhantomMap,
        build_phantom_shapes,
        extract_phantom_map,
    )

    assert PhantomMap is not None
    assert build_phantom_shapes is not None
    assert extract_phantom_map is not None
```

- [ ] **Step 2: Run; expect ImportError**

Run: `.venv/bin/python -m pytest tests/structured/test_package_smoke.py -v -k phase2`

Expected: fails with `ImportError`.

- [ ] **Step 3: Update `meshwell/structured/__init__.py`**

```python
"""Clean structured-polyprism pipeline.

Public surface:

- :class:`StructuredExtrusionResolutionSpec` -- attach to a
  ``PolyPrism(structured=True)`` to specify per-z-interval layer counts.
- :func:`build_plan` -- planner entry point: validates structured
  entities, returns a frozen ``StructuredPlan`` for the CAD + mesh stages.
- :func:`build_phantom_shapes` -- CAD-stage Layer A: build one OCP
  sub-prism per partition piece.
- :func:`extract_phantom_map` -- CAD-stage Layer B: walk
  ``BOPAlgo_Builder`` history to produce the ``PhantomMap`` consumed
  by the mesh stage.
- :class:`PhantomMap` -- the post-BOP correspondence map.
"""
from __future__ import annotations

from meshwell.structured.phantom import build_phantom_shapes, extract_phantom_map
from meshwell.structured.plan import build_plan
from meshwell.structured.spec import PhantomMap, StructuredExtrusionResolutionSpec

__all__ = [
    "PhantomMap",
    "StructuredExtrusionResolutionSpec",
    "build_phantom_shapes",
    "build_plan",
    "extract_phantom_map",
]
```

- [ ] **Step 4: Run smoke tests**

Run: `.venv/bin/python -m pytest tests/structured/test_package_smoke.py -v`

Expected: all 3 PASSES (2 original + 1 new).

Run full suite: `.venv/bin/python -m pytest tests/structured/ -v`

Expected: all PASS (~50 tests now).

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/__init__.py tests/structured/test_package_smoke.py
git commit -m "$(cat <<'EOF'
feat(structured): re-export Phase-2 phantom API from package __init__

Adds PhantomMap, build_phantom_shapes, extract_phantom_map to the
public structured.* namespace alongside the Phase-1 entries.

Phase 2 ships: CAD-stage Layer A (piece-by-piece sub-prism
construction) and Layer B (BOP-history-based correspondence map). All
testable in isolation with direct OCP. Integration with cad_occ is
Phase 3, alongside the mesh-stage builder.
EOF
)"
```

---

## Self-Review Checklist

**1. Spec coverage** (from `2026-05-15-structured-polyprism-clean-design.md`):

| Spec section | Phase 2 task | Status |
|---|---|---|
| Layer A — piece-by-piece sub-prism, no fuse | Tasks 4, 5 | ✓ |
| Layer A — outer-only lateral faces | Task 4 | ✓ |
| Layer A — mid-height cut detection | Task 7 | ✓ |
| Layer B — input OCC tag bookkeeping | Tasks 1, 4 | ✓ |
| Layer B — Modified()/Generated()/IsDeleted() walk | Tasks 6, 7 | ✓ |
| Layer B — PhantomMap with list-valued post-BOP dicts | Tasks 1, 6 | ✓ |
| Phantom volumes marked for non-recursive removal | **Phase 3** | deferred (cad_occ integration) |
| cad_occ integration (entry hook to receive PhantomMap) | **Phase 3** | deferred |
| Internal-seam faces marked "no auto-mesh" | **Phase 3** | deferred (mesh stage) |
| Layer C — mesh stage owns top mesh | **Phase 3** | deferred |
| StructuredMeshPlan / removeDuplicateNodes | **Phase 3** | deferred |

**2. Placeholder scan:** none. Every step has executable code or commands.

**3. Type consistency:**
- `FaceKey(slab_index, side, piece_index)` — same field order across Tasks 1, 4, 6, 7. ✓
- `PhantomShape.input_faces_by_key` is `dict[FaceKey, Any]` — consumed by `extract_phantom_map` iterating `shape.input_faces_by_key.items()`. ✓
- `PhantomMap.output_faces` is `dict[FaceKey, list[Any]]` — every test asserts on `list[Any]` semantics. ✓
- `PhantomMap.lateral_has_midheight_cut` is `dict[LateralKey, bool]` — keyed identically to `output_laterals`. ✓
- `extract_phantom_map(build_result, builder)` signature matches every test's call site. ✓

**4. Ambiguity check:**
- Task 4's wire-iteration order assumption ("outer wire is first") relies on `BRepBuilderAPI_MakeFace`'s contract (outer added first via constructor, interiors added later via `Add()`). This is documented OCC behaviour but should be asserted in code via the explicit `outer_coords` count (which the task does: `bot_outer_edges = bot_edges_all[:n_outer]`). Acceptable.
- Task 7's `_slab_z_range_for_shape` uses `BRep_Tool.Pnt_s` (static method); in OCP, the standard call is `BRep_Tool.Pnt_s(vertex) -> gp_Pnt`. If OCP exposes this differently in the installed version, the implementer will catch it via the failing test.
- Internal-seam-face "no auto-mesh" marking is deferred to Phase 3 (it's a mesh-stage concern that hooks into gmsh, not OCP).

---

## Out of scope for Phase 2

- `meshwell/structured/builder.py` — Phase 3.
- `meshwell/structured/logging.py` — Phase 3.
- `meshwell/cad_occ.py` modifications — Phase 3 (we need an entry hook for the structured pipeline to register phantom inputs into the global BOP).
- Orchestrator wiring — Phase 3.
- Internal-seam mesh suppression — Phase 3.
- `StructuredMeshPlan` (mesh-stage parameter resolution from specs) — Phase 3.
- `removeDuplicateNodes` global cleanup — Phase 3.
- Arc-provenance face-partition migration from `feat/structured` — also Phase 3 once builder needs it.
- End-to-end mesh tests with gmsh — Phase 3.
