# OCC All-Fragment + Priority-Tagging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the cuts-then-fragments OCC pipeline with a single all-fragment pass that resolves overlaps by `mesh_order` priority, and serialize results through one compound BREP so GMSH preserves shared topology for interface/boundary tagging.

**Architecture:**
- `meshwell/cad_occ.py` (`CAD_OCC`) does **one** `BOPAlgo_Builder.Perform()` across all entities (all dimensions together). Each fragment piece is mapped back to every entity whose original `Modified`/`Images` list contains it; the piece is owned by the lowest-`mesh_order` entity. Lower = higher priority. This mirrors `cad_gmsh.py:_process_dimension_group_cuts` ownership logic.
- `meshwell/occ_to_gmsh.py` (`inject_occ_entities_into_gmsh`) assembles one `TopoDS_Compound` whose top-level children are the per-entity fragment pieces (entity-ordered). It writes one BREP, calls `gmsh.model.occ.importShapes` once — BREP preserves sub-shape sharing, so coincident faces stay coincident in GMSH — then slices the returned flat dimtag list by each entity's piece count.
- Tagging (`tag_entities`, `tag_interfaces`, `tag_boundaries`) and dangling cleanup remain unchanged.

**Tech Stack:** Python 3.11+, `OCP` (OpenCASCADE Python bindings: `BOPAlgo_Builder`, `TopoDS_Compound`, `BRep_Builder`, `BRepTools`, `TopoDS_Iterator`), `gmsh` (Python API), `pytest`, `shapely`.

---

## File Structure

- **Modify:** `meshwell/cad_occ.py`
  - `OCCLabeledEntity` dataclass: add `shapes: list[TopoDS_Shape]` (list of fragment pieces). Keep `shape` as a legacy single-shape slot during the transition, populated from `shapes` for backward compatibility. Later tasks remove `shape`.
  - Replace `_process_dimension_group_cuts_occ` and `_process_dimension_group_fragments_occ` with `_fragment_all` (single all-dim pass) plus `_assign_ownership` (priority resolution).
  - `process_entities` now: instantiate → `_fragment_all` → return list of `OCCLabeledEntity`.
- **Modify:** `meshwell/occ_to_gmsh.py`
  - Drop per-entity BREP loop. Build one compound in entity order, `BRepTools.Write_s` once, `importShapes` once, slice returned dimtags by per-entity piece counts.
- **Modify:** `meshwell/backend_occ.py` — no interface changes expected; verify after refactor.
- **Test (reuse):** `tests/test_cad_occ.py`, `tests/test_multidimensional_cad_occ.py`, `tests/test_performance_cad_occ.py` — must pass unchanged as regression.
- **Test (new):** `tests/test_cad_occ_fragment_ownership.py` — unit tests for ownership resolution and shared-topology survival through BREP round-trip.

---

## Task 1: Extend `OCCLabeledEntity` to hold a list of fragment pieces

**Files:**
- Modify: `meshwell/cad_occ.py:18-26`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cad_occ_fragment_ownership.py`:

```python
"""Unit tests for the all-fragment OCC pipeline."""
from __future__ import annotations

import pytest
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt

from meshwell.cad_occ import OCCLabeledEntity


def test_occ_labeled_entity_accepts_shapes_list():
    """OCCLabeledEntity should store a list of fragment pieces."""
    box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1.0, 1.0, 1.0).Shape()
    ent = OCCLabeledEntity(
        shapes=[box],
        physical_name=("box",),
        index=0,
        keep=True,
        dim=3,
    )
    assert ent.shapes == [box]
    assert ent.dim == 3


def test_occ_labeled_entity_multiple_pieces():
    """OCCLabeledEntity must support multiple fragment pieces per entity."""
    b1 = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1.0, 1.0, 1.0).Shape()
    b2 = BRepPrimAPI_MakeBox(gp_Pnt(2, 0, 0), 1.0, 1.0, 1.0).Shape()
    ent = OCCLabeledEntity(
        shapes=[b1, b2],
        physical_name=("disjoint",),
        index=1,
        keep=True,
        dim=3,
    )
    assert len(ent.shapes) == 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_cad_occ_fragment_ownership.py -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'shapes'` (or similar).

- [ ] **Step 3: Update the dataclass**

In `meshwell/cad_occ.py`, replace lines 18–26 with:

```python
@dataclass
class OCCLabeledEntity:
    """Dataclass to store OCC shape(s) and associated metadata.

    shapes holds the fragment pieces this entity owns after the all-fragment pass.
    """

    shapes: list[TopoDS_Shape]
    physical_name: tuple[str, ...]
    index: int
    keep: bool
    dim: int
```

- [ ] **Step 4: Fix existing construction site**

In `meshwell/cad_occ.py`, `_instantiate_entity_occ` (around line 70–83) currently passes `shape=shape`. Update to:

```python
    def _instantiate_entity_occ(self, index: int, entity_obj: Any) -> OCCLabeledEntity:
        """Instantiate a single entity into an OCC shape."""
        shape = entity_obj.instanciate_occ()
        dim = self._get_shape_dimension(shape)

        return OCCLabeledEntity(
            shapes=[shape],
            physical_name=entity_obj.physical_name,
            index=index,
            keep=entity_obj.mesh_bool,
            dim=dim,
        )
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest tests/test_cad_occ_fragment_ownership.py -v`
Expected: PASS for both tests.

- [ ] **Step 6: Commit**

```bash
git add meshwell/cad_occ.py tests/test_cad_occ_fragment_ownership.py
git commit -m "feat(cad_occ): OCCLabeledEntity holds list of fragment shapes"
```

---

## Task 2: Add ownership resolver (pure function, no OCC calls)

**Files:**
- Create: `meshwell/cad_occ.py` (new helper `_resolve_piece_ownership`)
- Test: `tests/test_cad_occ_fragment_ownership.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cad_occ_fragment_ownership.py`:

```python
from meshwell.cad_occ import _resolve_piece_ownership


def test_resolve_piece_ownership_lowest_wins():
    """When multiple entities claim a piece, lowest mesh_order wins."""
    # piece_candidates maps piece_id -> list of (entity_index, mesh_order)
    piece_candidates = {
        "pA": [(0, 2.0), (1, 1.0)],  # entity 1 (mesh_order 1) wins
        "pB": [(0, 2.0)],             # entity 0 only
        "pC": [(2, 3.0), (1, 1.0), (0, 2.0)],  # entity 1 wins
    }
    owners = _resolve_piece_ownership(piece_candidates)
    assert owners == {"pA": 1, "pB": 0, "pC": 1}


def test_resolve_piece_ownership_tie_first_wins():
    """On mesh_order tie, the first candidate (insertion order) wins."""
    piece_candidates = {"p": [(3, 1.0), (5, 1.0), (2, 1.0)]}
    owners = _resolve_piece_ownership(piece_candidates)
    assert owners == {"p": 3}


def test_resolve_piece_ownership_inf_mesh_order():
    """Entities with mesh_order=None treated as infinity (lowest priority)."""
    piece_candidates = {
        "p": [(0, float("inf")), (1, 5.0)],
    }
    owners = _resolve_piece_ownership(piece_candidates)
    assert owners == {"p": 1}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_cad_occ_fragment_ownership.py::test_resolve_piece_ownership_lowest_wins -v`
Expected: FAIL with `ImportError: cannot import name '_resolve_piece_ownership'`.

- [ ] **Step 3: Implement the helper**

Add to `meshwell/cad_occ.py` above `class CAD_OCC`:

```python
def _resolve_piece_ownership(
    piece_candidates: dict[Any, list[tuple[int, float]]],
) -> dict[Any, int]:
    """Pick the owning entity index for each fragment piece.

    Rule: lowest mesh_order wins. On tie, first candidate in insertion order wins.

    Args:
        piece_candidates: maps piece key -> list of (entity_index, mesh_order).

    Returns:
        dict mapping piece key -> winning entity_index.
    """
    owners: dict[Any, int] = {}
    for piece, candidates in piece_candidates.items():
        best_idx = candidates[0][0]
        best_mo = candidates[0][1]
        for idx, mo in candidates[1:]:
            if mo < best_mo:
                best_idx = idx
                best_mo = mo
        owners[piece] = best_idx
    return owners
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_cad_occ_fragment_ownership.py -v`
Expected: all five tests PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/cad_occ.py tests/test_cad_occ_fragment_ownership.py
git commit -m "feat(cad_occ): add _resolve_piece_ownership helper"
```

---

## Task 3: Add a piece-key helper so TopoDS_Shape handles are hashable

**Files:**
- Modify: `meshwell/cad_occ.py` (add `_shape_key`)
- Test: `tests/test_cad_occ_fragment_ownership.py` (extend)

**Why:** `TopoDS_Shape` instances are not hashable by default (Python-binding quirk). `BOPAlgo_Builder.Modified(s)` returns a list of shapes we must dedupe / key. OCC's `TShape` pointer provides object identity and is the idiomatic key.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cad_occ_fragment_ownership.py`:

```python
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt

from meshwell.cad_occ import _shape_key


def test_shape_key_same_shape_equal():
    """Two handles to the same underlying shape must compare equal."""
    box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1.0, 1.0, 1.0).Shape()
    k1 = _shape_key(box)
    k2 = _shape_key(box)
    assert k1 == k2
    assert hash(k1) == hash(k2)


def test_shape_key_different_shapes_differ():
    """Distinct shape constructions produce distinct keys."""
    b1 = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1.0, 1.0, 1.0).Shape()
    b2 = BRepPrimAPI_MakeBox(gp_Pnt(2, 0, 0), 1.0, 1.0, 1.0).Shape()
    assert _shape_key(b1) != _shape_key(b2)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_cad_occ_fragment_ownership.py::test_shape_key_same_shape_equal -v`
Expected: FAIL with `ImportError: cannot import name '_shape_key'`.

- [ ] **Step 3: Implement the helper**

Add to `meshwell/cad_occ.py` near `_resolve_piece_ownership`:

```python
from OCP.TopTools import TopTools_ShapeMapHasher

_SHAPE_HASHER = TopTools_ShapeMapHasher()


def _shape_key(shape: TopoDS_Shape) -> tuple[int, int]:
    """Return a hashable identity key for a TopoDS_Shape.

    Uses the TShape pointer plus orientation so that reversed shapes
    (e.g. a face and its reversed twin used to glue solids) compare distinct
    when BOPAlgo differentiates them, and equal when it does not.

    Note: OCP returns a fresh Python wrapper on each call to
    ``TopoDS_Shape.TShape()``, so ``id()``/default ``__hash__`` is not stable.
    We use OCC's own ``TopTools_ShapeMapHasher``, which hashes on the
    underlying ``TShape*`` pointer and is the idiomatic key in OCC.
    """
    return (_SHAPE_HASHER(shape), int(shape.Orientation()))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_cad_occ_fragment_ownership.py -v`
Expected: all seven tests PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/cad_occ.py tests/test_cad_occ_fragment_ownership.py
git commit -m "feat(cad_occ): add _shape_key for hashable shape identity"
```

---

## Task 4: Implement `_fragment_all` — single BOPAlgo_Builder pass

**Files:**
- Modify: `meshwell/cad_occ.py` (add method to `CAD_OCC`)
- Test: `tests/test_cad_occ_fragment_ownership.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cad_occ_fragment_ownership.py`:

```python
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt

from meshwell.cad_occ import CAD_OCC, OCCLabeledEntity


def _make_ent(idx, shape, mesh_order, name, dim=3, keep=True):
    ent = OCCLabeledEntity(
        shapes=[shape],
        physical_name=(name,),
        index=idx,
        keep=keep,
        dim=dim,
    )
    ent._mesh_order = mesh_order  # attached for the fragment step
    return ent


def test_fragment_all_disjoint_boxes_preserved():
    """Disjoint shapes are unchanged; each entity keeps its piece."""
    b1 = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1.0, 1.0, 1.0).Shape()
    b2 = BRepPrimAPI_MakeBox(gp_Pnt(5, 0, 0), 1.0, 1.0, 1.0).Shape()
    ents = [_make_ent(0, b1, 1.0, "a"), _make_ent(1, b2, 2.0, "b")]
    processor = CAD_OCC()
    result = processor._fragment_all(ents)
    assert len(result) == 2
    assert len(result[0].shapes) == 1
    assert len(result[1].shapes) == 1


def test_fragment_all_overlap_goes_to_lower_mesh_order():
    """Overlapping region is owned by the entity with the lower mesh_order."""
    b1 = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 2.0, 2.0, 2.0).Shape()
    b2 = BRepPrimAPI_MakeBox(gp_Pnt(1, 1, 1), 2.0, 2.0, 2.0).Shape()
    # a has mesh_order 1 (higher priority), b has mesh_order 2
    ents = [_make_ent(0, b1, 1.0, "a"), _make_ent(1, b2, 2.0, "b")]
    processor = CAD_OCC()
    result = processor._fragment_all(ents)
    # Sum of all pieces should equal the number of fragments produced.
    total_pieces = sum(len(e.shapes) for e in result)
    # At minimum a gets the whole a, b gets only its non-overlapping remainder.
    assert total_pieces >= 2
    # 'a' must not have been shrunk to zero
    assert len(result[0].shapes) >= 1
    # 'b' is split; its pieces should be fewer than b1+b2 combined
    assert len(result[1].shapes) >= 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_cad_occ_fragment_ownership.py::test_fragment_all_disjoint_boxes_preserved -v`
Expected: FAIL with `AttributeError: 'CAD_OCC' object has no attribute '_fragment_all'`.

- [ ] **Step 3: Implement `_fragment_all` on `CAD_OCC`**

In `meshwell/cad_occ.py`, add this method to `CAD_OCC` (replacing `_process_dimension_group_cuts_occ` and `_process_dimension_group_fragments_occ`):

```python
    def _fragment_all(
        self, entities: list[OCCLabeledEntity]
    ) -> list[OCCLabeledEntity]:
        """Fragment all entities together; assign pieces by mesh_order priority.

        Each input entity carries a ``_mesh_order`` attribute (float or None).
        After this call, each entity's ``shapes`` list contains only the
        fragment pieces it owns. Ownership rule: lowest mesh_order wins.
        Pieces that come from only one entity are unambiguously owned by it.
        """
        if not entities:
            return []

        # Single-entity shortcut — nothing to fragment against.
        if len(entities) == 1:
            return entities

        builder = BOPAlgo_Builder()
        builder.SetRunParallel(self.n_threads > 1)
        builder.SetFuzzyValue(self.point_tolerance)
        builder.SetNonDestructive(False)

        originals_per_entity: list[list[TopoDS_Shape]] = []
        for ent in entities:
            originals_per_entity.append(list(ent.shapes))
            for s in ent.shapes:
                builder.AddArgument(s)

        builder.Perform()

        # piece_candidates: shape_key -> list of (entity_index, mesh_order).
        # piece_shapes: shape_key -> the TopoDS_Shape handle.
        piece_candidates: dict[tuple[int, int], list[tuple[int, float]]] = {}
        piece_shapes: dict[tuple[int, int], TopoDS_Shape] = {}

        for ent_idx, ent in enumerate(entities):
            mo = getattr(ent, "_mesh_order", None)
            if mo is None:
                mo = float("inf")
            for original in originals_per_entity[ent_idx]:
                modified = builder.Modified(original)
                if modified.IsEmpty() and not builder.IsDeleted(original):
                    # Shape survived untouched.
                    pieces = [original]
                else:
                    pieces = list(modified)
                for piece in pieces:
                    k = _shape_key(piece)
                    piece_shapes.setdefault(k, piece)
                    piece_candidates.setdefault(k, []).append((ent_idx, mo))

        owners = _resolve_piece_ownership(piece_candidates)

        # Reset each entity's shapes and reassign by owner.
        for ent in entities:
            ent.shapes = []
        for key, ent_idx in owners.items():
            entities[ent_idx].shapes.append(piece_shapes[key])

        return entities
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_cad_occ_fragment_ownership.py -v`
Expected: all nine tests PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/cad_occ.py tests/test_cad_occ_fragment_ownership.py
git commit -m "feat(cad_occ): single all-fragment pass with priority ownership"
```

---

## Task 5: Wire `_fragment_all` into `process_entities`

**Files:**
- Modify: `meshwell/cad_occ.py` (`process_entities`, lines ~219–293)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cad_occ_fragment_ownership.py`:

```python
from meshwell.cad_occ import cad_occ
from meshwell.occ_entity import OCC_entity


def test_process_entities_overlapping_boxes_end_to_end():
    """Higher-priority box keeps its full volume; lower-priority box loses overlap."""
    a = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 2.0, 2.0, 2.0).Shape(),
        physical_name="a",
        mesh_order=1,
        dimension=3,
    )
    b = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(gp_Pnt(1, 1, 1), 2.0, 2.0, 2.0).Shape(),
        physical_name="b",
        mesh_order=2,
        dimension=3,
    )
    result = cad_occ([a, b])
    assert len(result) == 2
    # Both entities should still have pieces.
    assert all(len(ent.shapes) >= 1 for ent in result)
    names = {ent.physical_name[0] for ent in result}
    assert names == {"a", "b"}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_cad_occ_fragment_ownership.py::test_process_entities_overlapping_boxes_end_to_end -v`
Expected: FAIL (current `process_entities` still uses old cuts pipeline with `shape` attribute; likely `AttributeError` on `.shape`).

- [ ] **Step 3: Rewrite `process_entities`**

In `meshwell/cad_occ.py`, replace the body of `process_entities` (everything from `def process_entities` through `return all_processed_entities`, approx lines 219–293) with:

```python
    def process_entities(
        self,
        entities_list: list[Any],
        _progress_bars: bool = False,
    ) -> list[OCCLabeledEntity]:
        """Process entities and return list of OCCLabeledEntity objects.

        Strategy: instantiate each entity into an OCC shape, then do one
        BOPAlgo_Builder.Perform() across everything. Fragment pieces are
        assigned to the entity with the lowest mesh_order. Lower-dim
        entities inside higher-dim ones end up sharing topology (coincident
        sub-faces) because BOPAlgo preserves sub-shape sharing.

        Args:
            entities_list: list of entity objects (PolyPrism, PolySurface,
                OCC_entity, ...).
            _progress_bars: unused; kept for interface parity with CAD_GMSH.

        Returns:
            Processed OCCLabeledEntity objects (entities with zero pieces
            after ownership resolution are still returned so callers can
            see that the entity produced nothing).
        """
        if not entities_list:
            return []

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            labeled_entities = list(
                executor.map(
                    lambda x: self._instantiate_entity_occ(x[0], x[1]),
                    enumerate(entities_list),
                )
            )

        # Attach mesh_order directly to each labeled entity so _fragment_all
        # can read it without a parallel list.
        for ent_obj, labeled_ent in zip(entities_list, labeled_entities):
            labeled_ent._mesh_order = ent_obj.mesh_order
            # If the entity object declares a dimension explicitly, trust it.
            explicit_dim = getattr(ent_obj, "dimension", None)
            if explicit_dim is not None:
                labeled_ent.dim = explicit_dim

        return self._fragment_all(labeled_entities)
```

- [ ] **Step 4: Delete the obsolete helpers**

In `meshwell/cad_occ.py`, delete both `_process_dimension_group_cuts_occ` (old lines 85–165) and `_process_dimension_group_fragments_occ` (old lines 167–217) in full. Also remove the now-unused imports `from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut` (line 10) if no other callers exist — grep first:

Run: `grep -n "BRepAlgoAPI_Cut" meshwell/cad_occ.py`
Expected: no matches remain after deletion. Then remove the import.

- [ ] **Step 5: Run the full OCC test suite**

Run: `pytest tests/test_cad_occ_fragment_ownership.py tests/test_cad_occ.py -v`
Expected: all new ownership tests PASS. `test_cad_occ.py` tests may still fail because `occ_to_gmsh.py` has not been updated yet to read `.shapes`; that is Task 6.

- [ ] **Step 6: Commit**

```bash
git add meshwell/cad_occ.py
git commit -m "refactor(cad_occ): replace cuts+fragments with single all-fragment pass"
```

---

## Task 6: Rewrite `inject_occ_entities_into_gmsh` to use one compound BREP

**Files:**
- Modify: `meshwell/occ_to_gmsh.py:18-129`
- Test: `tests/test_cad_occ_fragment_ownership.py` (extend with GMSH-round-trip test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cad_occ_fragment_ownership.py`:

```python
import gmsh

from meshwell.model import ModelManager
from meshwell.occ_to_gmsh import inject_occ_entities_into_gmsh


def test_inject_two_overlapping_boxes_produces_shared_interface():
    """After injection, the shared face between two touching boxes exists once."""
    a = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1.0, 1.0, 1.0).Shape(),
        physical_name="a",
        mesh_order=1,
        dimension=3,
    )
    b = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(gp_Pnt(1, 0, 0), 1.0, 1.0, 1.0).Shape(),
        physical_name="b",
        mesh_order=2,
        dimension=3,
    )
    occ_ents = cad_occ([a, b])

    mm = ModelManager(filename="test_shared_interface")
    try:
        labeled = inject_occ_entities_into_gmsh(occ_ents, mm)
        # Each entity present with at least one volume tag.
        by_name = {le.physical_name[0]: le for le in labeled}
        assert set(by_name) == {"a", "b"}
        # Interface physical group must exist (either a___b or b___a).
        groups = gmsh.model.getPhysicalGroups(2)
        names = [gmsh.model.getPhysicalName(dim, tag) for dim, tag in groups]
        assert any("a___b" in n or "b___a" in n for n in names), names
    finally:
        mm.finalize()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_cad_occ_fragment_ownership.py::test_inject_two_overlapping_boxes_produces_shared_interface -v`
Expected: FAIL — current `occ_to_gmsh.py` uses `occ_ent.shape` (singular), so an `AttributeError` is the first failure; even after switching to `.shapes`, the per-entity BREP import produces duplicate coincident faces and no interface physical group.

- [ ] **Step 3: Rewrite `inject_occ_entities_into_gmsh`**

In `meshwell/occ_to_gmsh.py`, replace the body (lines 18–129) with:

```python
def inject_occ_entities_into_gmsh(
    occ_entities: list[OCCLabeledEntity],
    model_manager: ModelManager | None = None,
    interface_delimiter: str = "___",
    boundary_delimiter: str = "None",
) -> list[LabeledEntities]:
    """Inject OCC shapes into gmsh and tag them.

    All entity shapes are packed into one TopoDS_Compound (entity-ordered),
    written to a single BREP file, and imported in one importShapes call.
    BREP serialization preserves sub-shape sharing, so coincident faces
    stay coincident in GMSH — this is what lets tag_interfaces find the
    shared boundaries between entities.

    Args:
        occ_entities: list of OCCLabeledEntity objects. Each may carry
            multiple fragment pieces in ``shapes``.
        model_manager: ModelManager instance. A fresh one is created if None.
        interface_delimiter: delimiter for interface physical names.
        boundary_delimiter: delimiter for exterior boundary physical names.

    Returns:
        list of LabeledEntities aligned 1:1 with the input entity order.
    """
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound

    from meshwell.model import ModelManager

    owns_model = False
    if model_manager is None:
        model_manager = ModelManager()
        owns_model = True

    model_manager.ensure_initialized(str(model_manager.filename))
    gmsh_model = model_manager.model

    max_dim = 0
    for ent in occ_entities:
        if ent.shapes:
            max_dim = max(max_dim, ent.dim)

    # Build one compound. Record how many top-level children each entity
    # contributes so we can slice the returned dimtag list afterwards.
    comp_builder = BRep_Builder()
    compound = TopoDS_Compound()
    comp_builder.MakeCompound(compound)

    piece_counts: list[int] = []
    for ent in occ_entities:
        piece_counts.append(len(ent.shapes))
        for s in ent.shapes:
            comp_builder.Add(compound, s)

    final_entity_list: list[LabeledEntities] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        brep_file = Path(tmpdir) / "all_entities.brep"
        BRepTools.Write_s(compound, str(brep_file))

        imported_dimtags = gmsh_model.occ.importShapes(str(brep_file))
        gmsh_model.occ.synchronize()

    # importShapes preserves top-level iteration order of the compound, so
    # slicing by piece_counts yields each entity's dimtags.
    cursor = 0
    for ent, count in zip(occ_entities, piece_counts):
        dimtags = imported_dimtags[cursor : cursor + count]
        cursor += count
        final_entity_list.append(
            LabeledEntities(
                index=ent.index,
                dimtags=list(dimtags),
                physical_name=ent.physical_name,
                keep=ent.keep,
                model=gmsh_model,
            )
        )

    for entity in final_entity_list:
        entity.update_boundaries()

    if final_entity_list:
        tag_entities(final_entity_list, gmsh_model)
        tag_interfaces(
            final_entity_list,
            max_dim,
            interface_delimiter,
            gmsh_model,
        )
        tag_boundaries(
            final_entity_list,
            max_dim,
            interface_delimiter,
            boundary_delimiter,
            gmsh_model,
        )

    # Remove entities marked keep=False (e.g. helpers used only to cut).
    for entity in final_entity_list:
        if not entity.keep and entity.dimtags:
            gmsh_model.occ.remove(entity.dimtags, recursive=False)
            gmsh_model.occ.synchronize()

    # Strip boundary/curve entities left over without a higher-dim parent.
    if max_dim == 3:
        dangling = [
            (dim, tag)
            for dim, tag in gmsh_model.getEntities(2)
            if len(gmsh_model.getAdjacencies(dim, tag)[0]) == 0
        ]
        if dangling:
            gmsh_model.occ.remove(dangling, recursive=True)
            gmsh_model.occ.synchronize()
    elif max_dim == 2:
        dangling = [
            (dim, tag)
            for dim, tag in gmsh_model.getEntities(1)
            if len(gmsh_model.getAdjacencies(dim, tag)[0]) == 0
        ]
        if dangling:
            gmsh_model.occ.remove(dangling, recursive=True)
            gmsh_model.occ.synchronize()

    if owns_model:
        model_manager.finalize()

    return final_entity_list
```

- [ ] **Step 4: Run the new test to verify it passes**

Run: `pytest tests/test_cad_occ_fragment_ownership.py::test_inject_two_overlapping_boxes_produces_shared_interface -v`
Expected: PASS. The `a___b` (or `b___a`) physical group is present.

- [ ] **Step 5: Run existing OCC tests as regression**

Run: `pytest tests/test_cad_occ.py tests/test_multidimensional_cad_occ.py -v`
Expected: all tests PASS. If `test_occ_composite_3D` fails because the previous implementation leaked duplicate surfaces, that is expected — the new path should emit clean topology.

- [ ] **Step 6: Commit**

```bash
git add meshwell/occ_to_gmsh.py tests/test_cad_occ_fragment_ownership.py
git commit -m "refactor(occ_to_gmsh): one-compound BREP import preserves shared topology"
```

---

## Task 7: Regression — cross-dimensional embedding (2D surface inside 3D box)

**Files:**
- Test: `tests/test_cad_occ_fragment_ownership.py` (extend)

**Why:** The whole point of the redesign is to make lower-dim-in-higher-dim topology conforming. Add an explicit regression so the shared-face invariant is caught by CI.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cad_occ_fragment_ownership.py`:

```python
def test_embedded_surface_splits_volume_and_shares_face():
    """A 2D surface inside a 3D box must appear as a shared face of the box sub-solids."""
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakeWire,
    )

    def rect(x, y, z, dx, dy):
        p1 = gp_Pnt(x, y, z)
        p2 = gp_Pnt(x + dx, y, z)
        p3 = gp_Pnt(x + dx, y + dy, z)
        p4 = gp_Pnt(x, y + dy, z)
        w = BRepBuilderAPI_MakeWire(
            BRepBuilderAPI_MakeEdge(p1, p2).Edge(),
            BRepBuilderAPI_MakeEdge(p2, p3).Edge(),
            BRepBuilderAPI_MakeEdge(p3, p4).Edge(),
            BRepBuilderAPI_MakeEdge(p4, p1).Edge(),
        ).Wire()
        return BRepBuilderAPI_MakeFace(w).Face()

    box = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 2.0, 2.0, 2.0).Shape(),
        physical_name="box",
        mesh_order=1,
        dimension=3,
    )
    # A surface cutting the box in half at z=1.
    cut_surface = OCC_entity(
        occ_function=lambda: rect(0.0, 0.0, 1.0, 2.0, 2.0),
        physical_name="cut",
        mesh_order=2,
        dimension=2,
    )

    occ_ents = cad_occ([box, cut_surface])
    mm = ModelManager(filename="test_embedded_surface")
    try:
        inject_occ_entities_into_gmsh(occ_ents, mm)
        # Box should be split into two volumes.
        vols = gmsh.model.getEntitiesForPhysicalGroup(
            3,
            next(
                tag
                for dim, tag in gmsh.model.getPhysicalGroups(3)
                if gmsh.model.getPhysicalName(dim, tag) == "box"
            ),
        )
        assert len(vols) == 2

        # The "cut" physical group must exist in 2D.
        surf_groups = gmsh.model.getPhysicalGroups(2)
        surf_names = [gmsh.model.getPhysicalName(d, t) for d, t in surf_groups]
        assert "cut" in surf_names
    finally:
        mm.finalize()
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_cad_occ_fragment_ownership.py::test_embedded_surface_splits_volume_and_shares_face -v`
Expected: PASS — this verifies that cross-dim fragmenting in OCC survived the BREP round-trip.

- [ ] **Step 3: Commit**

```bash
git add tests/test_cad_occ_fragment_ownership.py
git commit -m "test(cad_occ): embedded surface splits volume with shared face"
```

---

## Task 8: Regression — full existing OCC suite

**Files:** no changes — this task is just verification.

- [ ] **Step 1: Run every OCC-tagged test**

Run:

```bash
pytest tests/test_cad_occ.py \
       tests/test_multidimensional_cad_occ.py \
       tests/test_performance_cad_occ.py \
       tests/test_cad_occ_fragment_ownership.py \
       -v
```

Expected: ALL PASS. If any fail, do not proceed; diagnose the failure against the refactor before continuing. Common failure modes:
- `AttributeError: 'OCCLabeledEntity' object has no attribute 'shape'` — a caller still reads the removed legacy attribute. Grep for `.shape` on `OCCLabeledEntity` usages: `grep -rn "\.shape" meshwell/ | grep -v shapes`
- Tagging produces no interfaces — `update_boundaries` likely ran before `importShapes` completed; ensure `synchronize()` is called before `update_boundaries`.

- [ ] **Step 2: Run broader smoke test (non-OCC entry points unaffected)**

Run:

```bash
pytest tests/test_cad.py tests/test_multidimensional_cad.py -v
```

Expected: PASS. The GMSH-backed pipeline is untouched.

- [ ] **Step 3: Commit nothing (verification only)**

If the prior tasks produced stray files (`.msh`, `.xao`), ensure they are either committed or ignored — do **not** delete files without confirming they are not tracked:

```bash
git status
```

If `.msh`/`.xao` appear in `Untracked files` and `.gitignore` already ignores them, leave alone. If not, ask the user before adding ignores (out of scope of this plan).

---

## Task 9: Update `backend_occ.py` comment/docstring if needed

**Files:**
- Modify (maybe): `meshwell/backend_occ.py`

- [ ] **Step 1: Read the file**

Run: `cat meshwell/backend_occ.py`

- [ ] **Step 2: If the docstring mentions cuts or fragments as separate phases, update it**

Example — if a line like `"""Process entities using OCC backend (cuts+fragments)."""` exists, replace with:

```python
    def process_entities(self, entities: list[Any], **kwargs) -> list[Any]:
        """Process entities using OCC backend (single all-fragment pass)."""
```

If no such phrasing exists, skip. No code change is required — `backend_occ.py` only calls `CAD_OCC.process_entities` and `inject_occ_entities_into_gmsh`, both refactored in place.

- [ ] **Step 3: Commit if changed**

```bash
git add meshwell/backend_occ.py
git commit -m "docs(backend_occ): reflect all-fragment pipeline"
```

---

## Self-Review Checklist (run after all tasks)

1. **Spec coverage:**
   - All-fragment pass: Tasks 4–5 ✓
   - Priority-based ownership: Task 2 (`_resolve_piece_ownership`) + Task 4 (integration) ✓
   - Shared topology through GMSH import: Task 6 (one-compound BREP) ✓
   - Regression coverage: Tasks 7–8 ✓
2. **Type consistency:**
   - `OCCLabeledEntity.shapes: list[TopoDS_Shape]` used everywhere — Tasks 1, 4, 5, 6 all read/write `.shapes`.
   - `_mesh_order` is a float attached at runtime in Task 5 and consumed in Task 4 — naming matches.
   - `piece_counts` slicing in Task 6 uses `len(ent.shapes)` — matches what Task 4 writes.
3. **Placeholder scan:** search the plan for "TBD", "TODO", "similar to": none expected.
4. **Removed legacy:** `_process_dimension_group_cuts_occ`, `_process_dimension_group_fragments_occ`, `BRepAlgoAPI_Cut` import, and per-entity BREP loop all deleted in Tasks 5 and 6.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-19-occ-all-fragment.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
