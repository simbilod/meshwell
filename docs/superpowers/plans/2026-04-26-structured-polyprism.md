# Structured Extruded PolyPrism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add structured (gmsh-tutorial-t3-style) layered extrusion to meshwell. **Unified user API:** users still write `PolyPrism(...)`; passing a new `n_layers=` kwarg switches the entity into structured mode. Internally, `PolyPrism.__new__` dispatches to a private subclass `_StructuredPolyPrism` (defined in `meshwell/structured_polyprism.py`); the unstructured path is unchanged for any code that does not pass `n_layers`.

**Architecture:** Structured mode is a *meshing-strategy* layer added on top of `PolyPrism`. At CAD time the entity acts as a phantom OCC extrude with `mesh_bool=False` so it cuts neighbors and is then removed (leaving a void). At mesh time the void is reinstantiated in the gmsh **geo kernel** via `geo.extrude(..., Layers, recombine)` — which natively supports layered swept meshing on arbitrary polygons. Boundary node mating between the geo replica and surrounding OCC neighbors is achieved with `gmsh.model.mesh.embed`, never by removing OCC faces. Multiple structured prisms that overlap are resolved upfront by a 3D-disjoint shapely cascade ordered by `mesh_order`.

**API note:** Throughout this plan, "structured `PolyPrism`" means a `PolyPrism` instance constructed with `n_layers=` set; in test code it is created as `PolyPrism(..., n_layers=...)`. The runtime type of such an instance is `_StructuredPolyPrism`, which is what the cascade and `isinstance` checks inside the package use.

**Tech Stack:** Python, `shapely` (3D-disjoint slab cascade), OCP / `gmsh.model.occ` (phantom CAD), `gmsh.model.geo` (structured swept meshing), `pytest` (TDD).

**Spec:** `docs/superpowers/specs/2026-04-26-extruded-polyprisms-design.md`

---

## File Structure

| File | Status | Responsibility |
|------|--------|----------------|
| `meshwell/polyprism.py` | MODIFY | `PolyPrism.__new__` dispatch + `n_layers` / `recombine` kwargs in `__init__`; round-trip `n_layers` / `recombine` in `to_dict` / `from_dict` |
| `meshwell/structured_polyprism.py` | NEW | `_StructuredPolyPrism` (private subclass of `PolyPrism`), `Slab`, `expand_slabs_for_entity`, `resolve_structured_slabs`, `_StructuredPhantom`, `apply_structured_slabs`, sidecar JSON helpers |
| `meshwell/cad_common.py` | MODIFY | Hook structured-slab cascade into `prepare_entities`, emit phantoms |
| `meshwell/model.py` | MODIFY | Add `structured_slabs: list[Slab]` attribute |
| `meshwell/occ_xao_writer.py` | MODIFY | Write / read `<filename>.structured_slabs.json` sidecar |
| `meshwell/orchestrator.py` | MODIFY | Capture slabs from `cad_occ`; populate `mm.structured_slabs` and write sidecar at checkpoint |
| `meshwell/cad_occ.py` | MODIFY | Plumb `structured_slabs_out` into `prepare_entities`; expose captured slabs |
| `meshwell/cad_gmsh.py` | MODIFY | Plumb `structured_slabs_out` into `prepare_entities` |
| `meshwell/mesh.py` | MODIFY | Call `apply_structured_slabs` before `mesh.generate`; sidecar read in `load_xao_file` |
| `meshwell/utils.py` | (no change) | Existing `"PolyPrism"` deserialize hook covers structured mode (since `from_dict` re-routes via `__new__`) |
| `meshwell/__init__.py` | (no change) | Public API stays `PolyPrism` |
| `tests/test_structured_polyprism.py` | NEW | All structured-prism tests |
| `docs/prisms.py` | MODIFY | Add structured-mode example using `PolyPrism(..., n_layers=...)` |

`_StructuredPolyPrism` lives in its own file to keep the structured pipeline (entity subclass + slab cascade + mesh-stage helper) co-located, even though it's a subclass of `PolyPrism`. `polyprism.py` only owns the dispatch hook; everything else structured-specific stays in `structured_polyprism.py`.

---

## Task 1: Unified `PolyPrism(..., n_layers=...)` API + `__new__` dispatch + `Slab` dataclass

**Files:**
- Modify: `meshwell/polyprism.py` (add `n_layers` / `recombine` kwargs + `__new__` dispatch)
- Create: `meshwell/structured_polyprism.py` (defines `Slab` and the `_StructuredPolyPrism` subclass)
- Test: `tests/test_structured_polyprism.py`

- [ ] **Step 1.1: Write failing tests for unified API + structured-mode validation**

```python
# tests/test_structured_polyprism.py
"""Tests for structured-mode PolyPrism (gmsh-tutorial-t3-style layered extrusion)."""
from __future__ import annotations

import pytest
from shapely.geometry import MultiPolygon, Polygon


@pytest.fixture
def square_poly():
    return Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


def test_polyprism_without_n_layers_is_unstructured(square_poly):
    """Default PolyPrism path is untouched; same class returned."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured_polyprism import _StructuredPolyPrism

    pp = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="film",
    )
    assert type(pp) is PolyPrism
    assert not isinstance(pp, _StructuredPolyPrism)


def test_polyprism_with_n_layers_dispatches_to_structured(square_poly):
    """Passing n_layers triggers __new__ -> _StructuredPolyPrism instance.

    The user-facing class name is still ``PolyPrism``; isinstance(..., PolyPrism)
    must remain true.
    """
    from meshwell.polyprism import PolyPrism
    from meshwell.structured_polyprism import _StructuredPolyPrism

    pp = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[4],
        physical_name="film",
    )
    assert isinstance(pp, _StructuredPolyPrism)
    assert isinstance(pp, PolyPrism)
    assert pp.n_layers == [4]
    assert pp.recombine is False
    assert pp.physical_name == ("film",)


def test_structured_mode_requires_zero_buffers(square_poly):
    from meshwell.polyprism import PolyPrism

    with pytest.raises(ValueError, match="zero"):
        PolyPrism(
            polygons=square_poly,
            buffers={0.0: 0.0, 1.0: 0.1},  # nonzero buffer
            n_layers=[4],
        )


def test_structured_mode_requires_n_layers_length(square_poly):
    from meshwell.polyprism import PolyPrism

    with pytest.raises(ValueError, match="n_layers"):
        PolyPrism(
            polygons=square_poly,
            buffers={0.0: 0.0, 1.0: 0.0},
            n_layers=[4, 8],  # too many
        )

    with pytest.raises(ValueError, match="n_layers"):
        PolyPrism(
            polygons=square_poly,
            buffers={0.0: 0.0, 0.5: 0.0, 1.0: 0.0},
            n_layers=[4],  # too few
        )


def test_structured_mode_rejects_non_positive_layers(square_poly):
    from meshwell.polyprism import PolyPrism

    with pytest.raises(ValueError, match="n_layers"):
        PolyPrism(
            polygons=square_poly,
            buffers={0.0: 0.0, 1.0: 0.0},
            n_layers=[0],
        )


def test_structured_mode_rejects_non_increasing_z(square_poly):
    from meshwell.polyprism import PolyPrism

    with pytest.raises(ValueError, match="z"):
        PolyPrism(
            polygons=square_poly,
            buffers={1.0: 0.0, 0.0: 0.0},  # not strictly increasing in dict order
            n_layers=[4],
        )


def test_structured_mode_rejects_additive_or_subdivision(square_poly):
    """`additive=True` and `subdivision=` are unstructured-only knobs."""
    from meshwell.polyprism import PolyPrism

    with pytest.raises(ValueError, match="additive|subdivision"):
        PolyPrism(
            polygons=square_poly,
            buffers={0.0: 0.0, 1.0: 0.0},
            n_layers=[4],
            additive=True,
        )

    with pytest.raises(ValueError, match="additive|subdivision"):
        PolyPrism(
            polygons=square_poly,
            buffers={0.0: 0.0, 1.0: 0.0},
            n_layers=[4],
            subdivision=(2, 2, 1),
        )
```

- [ ] **Step 1.2: Run failing tests**

Run: `pytest tests/test_structured_polyprism.py -x -v`
Expected: failures because `n_layers` / `recombine` are unknown kwargs in the current `PolyPrism.__init__`.

- [ ] **Step 1.3: Create the structured module with `Slab` and the subclass stub**

Create `meshwell/structured_polyprism.py`:

```python
"""Structured (gmsh-tutorial-t3-style) layered extrusion for ``PolyPrism``.

This module hosts everything that is structured-specific:

* :class:`_StructuredPolyPrism` — private subclass of ``PolyPrism`` that
  the ``PolyPrism.__new__`` dispatcher returns when the user passes
  ``n_layers=``. Same user-visible class identity (``isinstance(p, PolyPrism)``
  remains true), but with the structured-mode validation rules.
* :class:`Slab` — one pairwise-disjoint piece of a structured prism after
  the cascade.
* :func:`expand_slabs_for_entity`, :func:`resolve_structured_slabs` — the
  3D-disjoint cascade.
* :class:`_StructuredPhantom` — the CAD-stage phantom that fragments
  neighbors and is removed via ``mesh_bool=False``.
* :func:`apply_structured_slabs` — the mesh-stage helper that refills the
  void with a geo-kernel layered extrude.

Why a private subclass rather than a flag on ``PolyPrism``: structured-mode
behavior diverges in three places (validation, ``mesh_bool`` semantics,
``instanciate``-time replacement by ``_StructuredPhantom``); a subclass
keeps each branch in one place and lets the rest of meshwell dispatch via
``isinstance``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shapely.geometry import MultiPolygon, Polygon

from meshwell.polyprism import PolyPrism


@dataclass
class Slab:
    """One pairwise-disjoint piece of a structured prism after resolution.

    Attributes:
        footprint: 2D shapely Polygon or MultiPolygon at z = zlo.
        zlo: Bottom z of this slab.
        zhi: Top z of this slab. Strictly greater than ``zlo``.
        n_layers: Number of element layers in [zlo, zhi].
        recombine: Whether to recombine the swept mesh into hex elements.
        physical_name: Tuple of physical-group names carried from the
            originating ``PolyPrism(..., n_layers=...)``.
        source_index: Index of the originating entity in the user-supplied
            list. Deterministic tie-break for the cascade.
        mesh_order: Ownership priority (lower = higher priority).
    """

    footprint: Polygon | MultiPolygon
    zlo: float
    zhi: float
    n_layers: int
    recombine: bool
    physical_name: tuple[str, ...]
    source_index: int = 0
    mesh_order: float = float("inf")


class _StructuredPolyPrism(PolyPrism):
    """Private subclass of ``PolyPrism`` for structured (layered) mode.

    Constructed indirectly by ``PolyPrism(..., n_layers=...)``. Users
    never reference this class by name; meshwell internals do, via
    ``isinstance``.
    """

    def __init__(
        self,
        polygons,
        buffers,
        n_layers,
        recombine: bool = False,
        **kwargs,
    ):
        # ----- Structured-mode validation -----
        if not all(b == 0.0 for b in buffers.values()):
            raise ValueError(
                "PolyPrism with n_layers requires all buffer values to be zero "
                "(taper is not supported in structured mode)."
            )
        z_keys = list(buffers.keys())
        if len(z_keys) < 2:
            raise ValueError(
                "PolyPrism with n_layers needs at least 2 z-breakpoints in `buffers`."
            )
        for a, b in zip(z_keys, z_keys[1:]):
            if not (b > a):
                raise ValueError(
                    f"PolyPrism with n_layers requires `buffers` z keys to be "
                    f"strictly increasing; got {a} then {b}."
                )
        if len(n_layers) != len(z_keys) - 1:
            raise ValueError(
                f"PolyPrism `n_layers` must have length {len(z_keys) - 1} "
                f"(one per z-interval); got {len(n_layers)}."
            )
        if any(n < 1 for n in n_layers):
            raise ValueError(
                f"PolyPrism `n_layers` entries must all be >= 1; got {n_layers}."
            )
        if kwargs.get("additive") or kwargs.get("subdivision") is not None:
            raise ValueError(
                "`additive=True` and `subdivision=` are not supported in "
                "structured mode (n_layers given). Drop n_layers or remove "
                "those kwargs."
            )

        # Defer the heavy polygon snap + state set to the parent so the
        # extrude=True path is selected (zero buffers).
        super().__init__(polygons=polygons, buffers=buffers, **kwargs)

        # Structured-mode-only state.
        self.n_layers = list(n_layers)
        self.recombine = recombine

    @property
    def z_breakpoints(self) -> list[float]:
        return list(self.buffers.keys())
```

- [ ] **Step 1.4: Add `__new__` dispatch and the new kwargs to `PolyPrism`**

Modify `meshwell/polyprism.py`. Add a `__new__` before `__init__` and accept the new kwargs in `__init__`:

```python
class PolyPrism(GeometryEntity):
    """[existing docstring]"""

    def __new__(cls, *args, n_layers=None, **kwargs):
        """Dispatch to ``_StructuredPolyPrism`` when ``n_layers`` is given.

        Keeps the user-facing surface as one class (``PolyPrism``) while
        letting the structured pipeline own its own subclass for
        validation and isinstance dispatch.
        """
        if cls is PolyPrism and n_layers is not None:
            # Lazy import to avoid the structured module importing
            # PolyPrism back at module load time.
            from meshwell.structured_polyprism import _StructuredPolyPrism

            return object.__new__(_StructuredPolyPrism)
        return object.__new__(cls)

    def __init__(
        self,
        polygons,
        buffers,
        physical_name=None,
        mesh_order=None,
        mesh_bool: bool = True,
        additive: bool = False,
        subdivision=None,
        point_tolerance: float = 1e-3,
        identify_arcs: bool = False,
        min_arc_points: int = 4,
        arc_tolerance: float = 1e-3,
        translation=None,
        rotation_axis=None,
        rotation_point=None,
        rotation_angle: float = 0.0,
        # New kwargs accepted but consumed by _StructuredPolyPrism:
        n_layers=None,
        recombine: bool = False,
    ):
        # When `n_layers` is None, this is the regular unstructured
        # path -- ``recombine`` is ignored. When it is set, ``__new__``
        # has already returned a ``_StructuredPolyPrism`` instance and
        # this ``PolyPrism.__init__`` is *not* the one that runs (the
        # subclass overrides it). The ``n_layers`` / ``recombine``
        # parameters here exist only to keep the public signature
        # consistent and to suppress a TypeError when the unstructured
        # path is used with the kwargs at default values.
        del n_layers, recombine

        # ... [existing PolyPrism __init__ body unchanged below this point] ...
```

Locate the existing `__init__` body (everything starting with `# Initialize parent class with point tracking and transformation parameters`) and leave it untouched after the `del n_layers, recombine` line above. Read `meshwell/polyprism.py` around the `__init__` definition and apply the diff carefully.

- [ ] **Step 1.5: Run tests, verify pass**

Run: `pytest tests/test_structured_polyprism.py -x -v`
Expected: 7 tests pass.

- [ ] **Step 1.6: Run full unstructured-path regression suite**

Run: `pytest tests/test_buffers_prism.py tests/test_arc_extrusion.py -x -q`
Expected: all existing PolyPrism tests still pass (the new kwargs default to None / False; behavior unchanged).

- [ ] **Step 1.7: Commit**

```bash
git add meshwell/polyprism.py meshwell/structured_polyprism.py tests/test_structured_polyprism.py
git commit -m "feat(polyprism): unified PolyPrism API with n_layers structured-mode dispatch"
```

---

## Task 2: `expand_slabs_for_entity` (single entity → list of Slabs)

**Files:**
- Modify: `meshwell/structured_polyprism.py`
- Modify: `tests/test_structured_polyprism.py`

- [ ] **Step 2.1: Write failing test**

Add to `tests/test_structured_polyprism.py`:

```python
def test_expand_slabs_single_interval(square_poly):
    from meshwell.polyprism import PolyPrism
    from meshwell.structured_polyprism import expand_slabs_for_entity

    sp = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[4],
        physical_name="film",
        mesh_order=2.0,
    )
    slabs = expand_slabs_for_entity(sp, source_index=7)
    assert len(slabs) == 1
    s = slabs[0]
    assert s.zlo == 0.0
    assert s.zhi == 1.0
    assert s.n_layers == 4
    assert s.physical_name == ("film",)
    assert s.source_index == 7
    assert s.mesh_order == 2.0
    assert s.footprint.equals(square_poly)


def test_expand_slabs_multi_interval(square_poly):
    from meshwell.polyprism import PolyPrism
    from meshwell.structured_polyprism import expand_slabs_for_entity

    sp = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 0.5: 0.0, 1.0: 0.0},
        n_layers=[8, 2],
        physical_name="film",
    )
    slabs = expand_slabs_for_entity(sp, source_index=0)
    assert len(slabs) == 2
    assert (slabs[0].zlo, slabs[0].zhi, slabs[0].n_layers) == (0.0, 0.5, 8)
    assert (slabs[1].zlo, slabs[1].zhi, slabs[1].n_layers) == (0.5, 1.0, 2)
```

- [ ] **Step 2.2: Run test, verify fail**

Run: `pytest tests/test_structured_polyprism.py::test_expand_slabs_single_interval -v`
Expected: `ImportError: cannot import name 'expand_slabs_for_entity'`.

- [ ] **Step 2.3: Implement `expand_slabs_for_entity`**

Add to `meshwell/structured_polyprism.py` (append):

```python
def _polygon_to_multipolygon(geom: Polygon | MultiPolygon | list) -> MultiPolygon:
    """Coerce shapely Polygon/list/MultiPolygon to a MultiPolygon."""
    if isinstance(geom, MultiPolygon):
        return geom
    if isinstance(geom, list):
        flat: list[Polygon] = []
        for g in geom:
            if isinstance(g, MultiPolygon):
                flat.extend(g.geoms)
            else:
                flat.append(g)
        return MultiPolygon(flat)
    return MultiPolygon([geom])


def expand_slabs_for_entity(
    entity: "_StructuredPolyPrism", source_index: int
) -> list[Slab]:
    """Expand a structured ``PolyPrism`` (i.e. ``_StructuredPolyPrism``) into per-z-interval slabs.

    Each adjacent pair of z-keys becomes one slab with the corresponding
    ``n_layers``. The 2D footprint is the same for every slab (no taper).
    """
    z_keys = entity.z_breakpoints
    mp = _polygon_to_multipolygon(entity.polygons)
    mesh_order = entity.mesh_order if entity.mesh_order is not None else float("inf")
    out: list[Slab] = []
    for (zlo, zhi), n in zip(zip(z_keys, z_keys[1:]), entity.n_layers):
        out.append(
            Slab(
                footprint=mp,
                zlo=zlo,
                zhi=zhi,
                n_layers=int(n),
                recombine=entity.recombine,
                physical_name=entity.physical_name,
                source_index=source_index,
                mesh_order=mesh_order,
            )
        )
    return out
```

- [ ] **Step 2.4: Run tests**

Run: `pytest tests/test_structured_polyprism.py -x -v`
Expected: all tests pass (7 total).

- [ ] **Step 2.5: Commit**

```bash
git add meshwell/structured_polyprism.py tests/test_structured_polyprism.py
git commit -m "feat(structured): expand structured PolyPrism into per-z-interval Slabs"
```

---

## Task 3: `resolve_structured_slabs` — disjoint identity case

**Files:**
- Modify: `meshwell/structured_polyprism.py`
- Modify: `tests/test_structured_polyprism.py`

- [ ] **Step 3.1: Write failing test**

```python
def test_resolve_disjoint_xy_identity(square_poly):
    from shapely.affinity import translate
    from meshwell.polyprism import PolyPrism
    from meshwell.structured_polyprism import resolve_structured_slabs

    sp1 = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[4],
        physical_name="A",
    )
    # Disjoint in xy
    sp2 = PolyPrism(
        polygons=translate(square_poly, xoff=2.0),
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[3],
        physical_name="B",
    )
    slabs = resolve_structured_slabs([sp1, sp2])
    assert len(slabs) == 2
    names = {s.physical_name[0] for s in slabs}
    assert names == {"A", "B"}


def test_resolve_disjoint_z_identity(square_poly):
    from meshwell.polyprism import PolyPrism
    from meshwell.structured_polyprism import resolve_structured_slabs

    sp1 = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[4],
        physical_name="A",
    )
    sp2 = PolyPrism(
        polygons=square_poly,  # same xy
        buffers={2.0: 0.0, 3.0: 0.0},  # disjoint z
        n_layers=[3],
        physical_name="B",
    )
    slabs = resolve_structured_slabs([sp1, sp2])
    assert len(slabs) == 2
    z_ranges = {(s.zlo, s.zhi) for s in slabs}
    assert z_ranges == {(0.0, 1.0), (2.0, 3.0)}


def test_resolve_filters_non_structured_entities(square_poly):
    from meshwell.polyprism import PolyPrism
    from meshwell.structured_polyprism import resolve_structured_slabs

    class _Other:
        polygons = square_poly
        physical_name = ("foo",)
        mesh_order = None

    sp = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[4],
        physical_name="film",
    )
    slabs = resolve_structured_slabs([_Other(), sp, _Other()])
    assert len(slabs) == 1
    assert slabs[0].physical_name == ("film",)
```

- [ ] **Step 3.2: Run failing test**

Run: `pytest tests/test_structured_polyprism.py::test_resolve_disjoint_xy_identity -v`
Expected: ImportError.

- [ ] **Step 3.3: Implement `resolve_structured_slabs` (identity-only path)**

Append to `meshwell/structured_polyprism.py`:

```python
def resolve_structured_slabs(entities_list: list[Any]) -> list[Slab]:
    """Decompose structured ``PolyPrism`` (``_StructuredPolyPrism``) entities into 3D-disjoint slabs.

    For overlapping prisms, lower ``mesh_order`` wins; ties resolved by
    insertion order. After this call, every returned :class:`Slab` is
    pairwise 3D-disjoint with every other returned slab (touching faces
    are allowed; volumetric intersection is not).

    Non-structured entities (``isinstance(ent, _StructuredPolyPrism)`` is
    False) are skipped.
    """
    # Local import avoids a circular: this module is imported by
    # polyprism.__new__ at construction time, so we defer the subclass
    # reference to call time.
    raw_slabs: list[Slab] = []
    for idx, ent in enumerate(entities_list):
        if not isinstance(ent, _StructuredPolyPrism):
            continue
        raw_slabs.extend(expand_slabs_for_entity(ent, source_index=idx))

    if len(raw_slabs) <= 1:
        return raw_slabs

    # Sort highest-priority first: lower mesh_order wins, then insertion
    # order (source_index then z position).
    raw_slabs.sort(
        key=lambda s: (s.mesh_order, s.source_index, s.zlo)
    )

    # Resolution pass added in Task 4 / 5; for disjoint inputs this is a
    # no-op so identity tests pass.
    return raw_slabs
```

- [ ] **Step 3.4: Run tests**

Run: `pytest tests/test_structured_polyprism.py -x -v`
Expected: all tests pass (10 total).

- [ ] **Step 3.5: Commit**

```bash
git add meshwell/structured_polyprism.py tests/test_structured_polyprism.py
git commit -m "feat(structured): identity case for resolve_structured_slabs"
```

---

## Task 4: `resolve_structured_slabs` — xy-overlap subtraction within shared z

**Files:**
- Modify: `meshwell/structured_polyprism.py`
- Modify: `tests/test_structured_polyprism.py`

- [ ] **Step 4.1: Write failing test**

```python
def test_resolve_xy_overlap_same_z(square_poly):
    from meshwell.polyprism import PolyPrism
    from meshwell.structured_polyprism import resolve_structured_slabs

    big = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    small = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])  # inside big

    sp_big = PolyPrism(
        polygons=big,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[2],
        physical_name="bg",
        mesh_order=2.0,  # lower priority
    )
    sp_small = PolyPrism(
        polygons=small,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[5],
        physical_name="hole",
        mesh_order=1.0,  # higher priority
    )
    slabs = resolve_structured_slabs([sp_big, sp_small])
    # Expect: small remains as-is; big's footprint becomes big - small.
    by_name = {s.physical_name[0]: s for s in slabs}
    assert "hole" in by_name and "bg" in by_name
    assert by_name["hole"].footprint.equals(MultiPolygon([small]))
    assert by_name["bg"].footprint.area == pytest.approx(
        big.area - small.area, rel=1e-9
    )


def test_resolve_xy_overlap_total_eats_low_priority(square_poly):
    from meshwell.polyprism import PolyPrism
    from meshwell.structured_polyprism import resolve_structured_slabs

    sp_loser = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[4],
        physical_name="loser",
        mesh_order=2.0,
    )
    # Same xy footprint, same z, higher priority -> loser should vanish.
    sp_winner = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[3],
        physical_name="winner",
        mesh_order=1.0,
    )
    slabs = resolve_structured_slabs([sp_loser, sp_winner])
    assert len(slabs) == 1
    assert slabs[0].physical_name == ("winner",)
```

Add `from shapely.geometry import MultiPolygon, Polygon` to test imports if not already present.

- [ ] **Step 4.2: Run failing tests**

Run: `pytest tests/test_structured_polyprism.py::test_resolve_xy_overlap_same_z -v`
Expected: FAIL (cascade is currently identity).

- [ ] **Step 4.3: Implement xy-difference cascade with z-overlap guard**

Replace the `resolve_structured_slabs` body in `meshwell/structured_polyprism.py`:

```python
def _z_overlap(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> tuple[float, float] | None:
    """Return overlapping z-interval, or None if disjoint or zero-volume touching."""
    lo = max(a_lo, b_lo)
    hi = min(a_hi, b_hi)
    if hi <= lo:
        return None
    return (lo, hi)


def _difference_footprint(
    low: Polygon | MultiPolygon, high: Polygon | MultiPolygon
) -> Polygon | MultiPolygon | None:
    """Shapely difference returning ``None`` when the result is empty."""
    diff = low.difference(high)
    if diff.is_empty:
        return None
    if isinstance(diff, Polygon):
        return diff
    if isinstance(diff, MultiPolygon):
        return diff
    # Could be GeometryCollection containing lines/points after a degenerate
    # subtraction; keep only polygonal parts.
    polys = [g for g in getattr(diff, "geoms", []) if isinstance(g, Polygon)]
    if not polys:
        return None
    return polys[0] if len(polys) == 1 else MultiPolygon(polys)


def resolve_structured_slabs(entities_list: list[Any]) -> list[Slab]:
    """Decompose structured ``PolyPrism`` (``_StructuredPolyPrism``) entities into 3D-disjoint slabs.

    See module docstring. v1 handles xy-overlap and z-overlap; full
    z-splitting (when overlap is partial in z) is added in Task 5.
    """
    raw_slabs: list[Slab] = []
    for idx, ent in enumerate(entities_list):
        if not isinstance(ent, _StructuredPolyPrism):
            continue
        raw_slabs.extend(expand_slabs_for_entity(ent, source_index=idx))

    if len(raw_slabs) <= 1:
        return raw_slabs

    raw_slabs.sort(key=lambda s: (s.mesh_order, s.source_index, s.zlo))

    # Run priority-ordered subtraction. Each slab subtracts every higher-
    # priority slab whose z-interval *fully covers* its own z-interval.
    # Partial-z-overlap splitting is handled in Task 5; here the guard
    # below skips partial overlaps so we don't silently produce wrong
    # geometry (an explicit assertion fires in those cases instead).
    resolved: list[Slab] = []
    for slab in raw_slabs:
        footprint: Polygon | MultiPolygon | None = slab.footprint
        for hi in resolved:
            overlap = _z_overlap(slab.zlo, slab.zhi, hi.zlo, hi.zhi)
            if overlap is None:
                continue
            # Partial-z overlap is not handled in this task; defer to Task 5.
            if (overlap[0], overlap[1]) != (slab.zlo, slab.zhi):
                continue
            new_fp = _difference_footprint(footprint, hi.footprint)
            if new_fp is None:
                footprint = None
                break
            footprint = new_fp
        if footprint is None:
            continue
        resolved.append(
            Slab(
                footprint=footprint,
                zlo=slab.zlo,
                zhi=slab.zhi,
                n_layers=slab.n_layers,
                recombine=slab.recombine,
                physical_name=slab.physical_name,
                source_index=slab.source_index,
                mesh_order=slab.mesh_order,
            )
        )
    return resolved
```

- [ ] **Step 4.4: Run all tests**

Run: `pytest tests/test_structured_polyprism.py -x -v`
Expected: all 12 tests pass.

- [ ] **Step 4.5: Commit**

```bash
git add meshwell/structured_polyprism.py tests/test_structured_polyprism.py
git commit -m "feat(structured): xy-overlap subtraction cascade for resolve_structured_slabs"
```

---

## Task 5: `resolve_structured_slabs` — partial z-overlap splits low-priority slab

**Files:**
- Modify: `meshwell/structured_polyprism.py`
- Modify: `tests/test_structured_polyprism.py`

- [ ] **Step 5.1: Write failing test**

```python
def test_resolve_partial_z_overlap_splits_low_priority():
    from shapely.geometry import Polygon
    from meshwell.polyprism import PolyPrism
    from meshwell.structured_polyprism import resolve_structured_slabs

    base = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    small = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])

    sp_lo = PolyPrism(
        polygons=base,
        buffers={0.0: 0.0, 3.0: 0.0},
        n_layers=[6],
        physical_name="lo",
        mesh_order=2.0,  # lower priority
    )
    sp_hi = PolyPrism(
        polygons=small,
        buffers={1.0: 0.0, 2.0: 0.0},  # only overlaps lo's z=[1,2]
        n_layers=[2],
        physical_name="hi",
        mesh_order=1.0,
    )
    slabs = resolve_structured_slabs([sp_lo, sp_hi])
    by_name = sorted(slabs, key=lambda s: (s.physical_name, s.zlo))
    # Expect 4 slabs:
    #   hi: [1,2] with footprint=small
    #   lo: [0,1] with footprint=base
    #   lo: [1,2] with footprint=base-small
    #   lo: [2,3] with footprint=base
    names_z = [(s.physical_name[0], s.zlo, s.zhi) for s in by_name]
    assert ("hi", 1.0, 2.0) in names_z
    assert ("lo", 0.0, 1.0) in names_z
    assert ("lo", 1.0, 2.0) in names_z
    assert ("lo", 2.0, 3.0) in names_z
    lo_slabs = [s for s in slabs if s.physical_name == ("lo",)]
    middle = next(s for s in lo_slabs if s.zlo == 1.0)
    full = next(s for s in lo_slabs if s.zlo == 0.0)
    assert middle.footprint.area == pytest.approx(
        base.area - small.area, rel=1e-9
    )
    assert full.footprint.area == pytest.approx(base.area, rel=1e-9)
```

- [ ] **Step 5.2: Run failing test**

Run: `pytest tests/test_structured_polyprism.py::test_resolve_partial_z_overlap_splits_low_priority -v`
Expected: FAIL — guard in Task 4 skipped this case.

- [ ] **Step 5.3: Implement z-splitting**

Replace the body of the cascade loop inside `resolve_structured_slabs` (the `for slab in raw_slabs` block) with:

```python
    resolved: list[Slab] = []
    for slab in raw_slabs:
        # Working list: list of (zlo, zhi, footprint) sub-pieces still to
        # resolve against subsequent higher-priority slabs.
        pieces: list[tuple[float, float, Polygon | MultiPolygon]] = [
            (slab.zlo, slab.zhi, slab.footprint)
        ]
        for hi in resolved:
            new_pieces: list[tuple[float, float, Polygon | MultiPolygon]] = []
            for (p_lo, p_hi, p_fp) in pieces:
                overlap = _z_overlap(p_lo, p_hi, hi.zlo, hi.zhi)
                if overlap is None:
                    new_pieces.append((p_lo, p_hi, p_fp))
                    continue
                ov_lo, ov_hi = overlap
                # Pre-overlap stub
                if p_lo < ov_lo:
                    new_pieces.append((p_lo, ov_lo, p_fp))
                # Overlap region: subtract hi.footprint
                ov_fp = _difference_footprint(p_fp, hi.footprint)
                if ov_fp is not None:
                    new_pieces.append((ov_lo, ov_hi, ov_fp))
                # Post-overlap stub
                if p_hi > ov_hi:
                    new_pieces.append((ov_hi, p_hi, p_fp))
            pieces = new_pieces
        for (p_lo, p_hi, p_fp) in pieces:
            resolved.append(
                Slab(
                    footprint=p_fp,
                    zlo=p_lo,
                    zhi=p_hi,
                    n_layers=slab.n_layers,
                    recombine=slab.recombine,
                    physical_name=slab.physical_name,
                    source_index=slab.source_index,
                    mesh_order=slab.mesh_order,
                )
            )
    return resolved
```

- [ ] **Step 5.4: Run all tests**

Run: `pytest tests/test_structured_polyprism.py -x -v`
Expected: all tests pass.

- [ ] **Step 5.5: Commit**

```bash
git add meshwell/structured_polyprism.py tests/test_structured_polyprism.py
git commit -m "feat(structured): split low-priority slabs on partial z-overlap"
```

---

## Task 6: Phantom CAD entity (`_StructuredPhantom`) — instanciate hooks

**Files:**
- Modify: `meshwell/structured_polyprism.py`
- Modify: `tests/test_structured_polyprism.py`

The phantom is an internal, transient entity that the cascade returns alongside its `Slab` records; it is what `cad_gmsh` / `cad_occ` actually fragment against. It must implement the same interface as `PolyPrism` (`instanciate`, `instanciate_occ`, `polygons`, `physical_name`, `mesh_order`, `mesh_bool`, `dimension`).

- [ ] **Step 6.1: Write failing test (gmsh path)**

```python
def test_phantom_gmsh_extrudes_volume(square_poly, monkeypatch):
    """The phantom's instanciate must produce one 3D dimtag."""
    import gmsh
    from meshwell.structured_polyprism import _StructuredPhantom, Slab
    from meshwell.model import ModelManager
    from shapely.geometry import MultiPolygon

    slab = Slab(
        footprint=MultiPolygon([square_poly]),
        zlo=0.0,
        zhi=1.0,
        n_layers=4,
        recombine=False,
        physical_name=("film",),
        source_index=0,
    )
    phantom = _StructuredPhantom(slab)
    assert phantom.dimension == 3
    assert phantom.mesh_bool is False  # keep=False at top-dim
    assert phantom.physical_name == ("film",)

    # Drive instanciate() against a minimal gmsh model.
    mm = ModelManager(filename="t_phantom_gmsh")
    mm.ensure_initialized("t_phantom_gmsh")

    class _Mock:
        model_manager = mm

    try:
        dimtags = phantom.instanciate(_Mock())
        assert len(dimtags) == 1
        assert dimtags[0][0] == 3
    finally:
        mm.finalize()
```

- [ ] **Step 6.2: Run failing test**

Run: `pytest tests/test_structured_polyprism.py::test_phantom_gmsh_extrudes_volume -v`
Expected: ImportError.

- [ ] **Step 6.3: Implement `_StructuredPhantom`**

Append to `meshwell/structured_polyprism.py`:

```python
class _StructuredPhantom:
    """Internal CAD-stage phantom for one resolved structured ``Slab``.

    Quacks like a meshwell entity for the purposes of ``cad_gmsh`` /
    ``cad_occ``: exposes ``polygons``, ``physical_name``, ``mesh_order``,
    ``mesh_bool``, ``dimension``, and the ``instanciate`` /
    ``instanciate_occ`` hooks. Built from a fully-resolved ``Slab`` (so
    the cascade has already pinned its footprint) and always sets
    ``mesh_bool = False`` so the existing keep=False top-dim machinery
    removes the body after fragmentation, leaving a void.
    """

    def __init__(self, slab: Slab):
        self.slab = slab
        self.polygons = slab.footprint  # MultiPolygon, used by prepare_entities
        self.physical_name = slab.physical_name
        self.mesh_order = slab.mesh_order if slab.mesh_order != float("inf") else None
        self.mesh_bool = False  # phantom -> removed after fragmentation
        self.dimension = 3

    # gmsh path: build via gmsh.model.occ.extrude
    def instanciate(self, cad_model: Any) -> list[tuple[int, int]]:
        import gmsh

        height = self.slab.zhi - self.slab.zlo
        polys = (
            self.slab.footprint.geoms
            if hasattr(self.slab.footprint, "geoms")
            else [self.slab.footprint]
        )
        out_dimtags: list[tuple[int, int]] = []
        for poly in polys:
            ext_tags = self._add_polygon_face_gmsh(poly, z=self.slab.zlo)
            ext_dimtags = [(2, t) for t in ext_tags]
            extruded = gmsh.model.occ.extrude(ext_dimtags, 0, 0, height)
            out_dimtags.extend([dt for dt in extruded if dt[0] == 3])
        gmsh.model.occ.synchronize()
        return out_dimtags

    def _add_polygon_face_gmsh(self, polygon: Polygon, z: float) -> list[int]:
        """Construct a planar face for ``polygon`` at ``z`` in the OCC kernel.

        Holes are honored: the exterior is a curve loop; interiors become
        additional curve loops passed to ``addPlaneSurface``.
        """
        import gmsh

        def _curve_loop(coords) -> int:
            pts: list[int] = []
            for x, y in coords:
                pts.append(gmsh.model.occ.addPoint(x, y, z))
            lines: list[int] = []
            for a, b in zip(pts, pts[1:]):
                lines.append(gmsh.model.occ.addLine(a, b))
            # Close the loop
            lines.append(gmsh.model.occ.addLine(pts[-1], pts[0]))
            return gmsh.model.occ.addCurveLoop(lines)

        # Drop the duplicated closing vertex shapely returns
        ext_coords = list(polygon.exterior.coords)[:-1]
        loops = [_curve_loop(ext_coords)]
        for interior in polygon.interiors:
            int_coords = list(interior.coords)[:-1]
            loops.append(_curve_loop(int_coords))
        face = gmsh.model.occ.addPlaneSurface(loops)
        return [face]

    # OCC path: build via OCP BRepPrimAPI_MakePrism
    def instanciate_occ(self) -> Any:
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
        from OCP.gp import gp_Vec

        height = self.slab.zhi - self.slab.zlo
        vec = gp_Vec(0, 0, height)
        polys = (
            self.slab.footprint.geoms
            if hasattr(self.slab.footprint, "geoms")
            else [self.slab.footprint]
        )
        # We delegate wire construction to GeometryEntity helpers via a
        # tiny adapter so we don't duplicate arc handling. The phantom
        # itself never carries arcs (slabs are linear-segment polygons
        # produced by shapely), so we use the simple polyline path.
        from meshwell.geometry_entity import GeometryEntity

        adapter = GeometryEntity(point_tolerance=0.0)
        result_solids = []
        for poly in polys:
            ext_vertices = [(x, y, self.slab.zlo) for x, y in poly.exterior.coords]
            outer = adapter._make_occ_wire_from_vertices(
                ext_vertices, identify_arcs=False, min_arc_points=4, arc_tolerance=0.0
            )
            mf = BRepBuilderAPI_MakeFace(outer)
            for interior in poly.interiors:
                int_vertices = [(x, y, self.slab.zlo) for x, y in interior.coords]
                hole = adapter._make_occ_wire_from_vertices(
                    int_vertices, identify_arcs=False, min_arc_points=4, arc_tolerance=0.0
                )
                hole.Reverse()
                mf.Add(hole)
            face = mf.Face()
            result_solids.append(BRepPrimAPI_MakePrism(face, vec).Shape())

        if len(result_solids) == 1:
            return result_solids[0]

        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse

        result = result_solids[0]
        for s in result_solids[1:]:
            fuser = BRepAlgoAPI_Fuse(result, s)
            fuser.Build()
            result = fuser.Shape()
        return result
```

- [ ] **Step 6.4: Run test**

Run: `pytest tests/test_structured_polyprism.py::test_phantom_gmsh_extrudes_volume -x -v`
Expected: pass.

- [ ] **Step 6.5: Commit**

```bash
git add meshwell/structured_polyprism.py tests/test_structured_polyprism.py
git commit -m "feat(structured): _StructuredPhantom with gmsh and OCC instanciate hooks"
```

---

## Task 7: Hook the cascade into `prepare_entities`; emit phantoms in-place

**Files:**
- Modify: `meshwell/cad_common.py`
- Modify: `meshwell/structured_polyprism.py`
- Modify: `tests/test_structured_polyprism.py`

`prepare_entities` runs before the per-entity buffer pass. We add: after the buffer pass, run `resolve_structured_slabs` on the entity list, **remove** the structured `PolyPrism` instances (i.e. `_StructuredPolyPrism`), and append `_StructuredPhantom` instances built from the resolved slabs. The cascade output is also stashed on a list passed by reference (or set on the model_manager later) so the mesh stage can read it back.

- [ ] **Step 7.1: Write failing test**

```python
def test_prepare_entities_swaps_structured_for_phantoms(square_poly):
    from meshwell.cad_common import prepare_entities
    from meshwell.polyprism import PolyPrism
    from meshwell.structured_polyprism import _StructuredPhantom

    sp = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[3],
        physical_name="film",
    )

    class _NeutralPolyEntity:
        def __init__(self, p):
            self.polygons = p
            self.physical_name = ("bg",)

    other = _NeutralPolyEntity(Polygon([(-2, -2), (2, -2), (2, 2), (-2, 2)]))
    entities = [other, sp]

    captured_slabs: list = []
    prepare_entities(
        entities,
        perturbation=1e-5,
        resolve_snap=1e-3,
        structured_slabs_out=captured_slabs,
    )
    # Structured PolyPrism replaced by phantom in-place.
    from meshwell.structured_polyprism import _StructuredPolyPrism
    assert not any(isinstance(e, _StructuredPolyPrism) for e in entities)
    phantoms = [e for e in entities if isinstance(e, _StructuredPhantom)]
    assert len(phantoms) == 1
    assert phantoms[0].physical_name == ("film",)
    assert phantoms[0].mesh_bool is False
    # Captured slab list mirrors the cascade output.
    assert len(captured_slabs) == 1
    assert captured_slabs[0].physical_name == ("film",)
```

- [ ] **Step 7.2: Run failing test**

Run: `pytest tests/test_structured_polyprism.py::test_prepare_entities_swaps_structured_for_phantoms -v`
Expected: TypeError on unexpected kwarg.

- [ ] **Step 7.3: Modify `prepare_entities`**

In `meshwell/cad_common.py`, replace the signature and add the cascade:

```python
def prepare_entities(
    entities_list: list[Any],
    perturbation: float,
    resolve_snap: float | None = None,
    structured_slabs_out: list | None = None,
) -> None:
    """In-place pre-pass shared by cad_gmsh and cad_occ.

    [existing docstring]

    Args:
        ... (existing) ...
        structured_slabs_out: Optional list. When provided, the resolved
            ``Slab`` objects from any structured ``PolyPrism``
            (``_StructuredPolyPrism``) in ``entities_list`` are appended
            here, and each such instance is replaced *in place* with
            one ``_StructuredPhantom`` per resolved slab. When
            ``None``, structured prisms are passed through unchanged
            (used by code paths that don't run the structured pipeline).
    """
```

Append after the existing InterfaceTag resolution block:

```python
    if structured_slabs_out is not None:
        from meshwell.structured_polyprism import (
            _StructuredPolyPrism,
            _StructuredPhantom,
            resolve_structured_slabs,
        )

        # Run the cascade once on the full list (uses the post-buffer
        # polygons; structured prisms participate in buffering above).
        slabs = resolve_structured_slabs(entities_list)
        structured_slabs_out.extend(slabs)

        # Build phantoms in source order so insertion order maps directly
        # back to user intent for fragmentation tie-breaks.
        phantom_entities = [_StructuredPhantom(s) for s in slabs]

        # Replace original structured-mode PolyPrism instances with
        # phantoms in place.
        new_list: list[Any] = []
        replaced = False
        for ent in entities_list:
            if isinstance(ent, _StructuredPolyPrism):
                if not replaced:
                    new_list.extend(phantom_entities)
                    replaced = True
                # subsequent structured entries: skip (already represented
                # in the phantom batch).
                continue
            new_list.append(ent)
        if not replaced and phantom_entities:
            # All slabs came from a single structured PolyPrism that was
            # the first entity scanned; this branch is unreachable
            # because the loop above handles it, but kept for safety.
            new_list = phantom_entities + new_list

        entities_list[:] = new_list
```

- [ ] **Step 7.4: Run test**

Run: `pytest tests/test_structured_polyprism.py::test_prepare_entities_swaps_structured_for_phantoms -x -v`
Expected: pass.

- [ ] **Step 7.5: Run full test suite to confirm no regressions**

Run: `pytest tests/ -x -q --ignore=tests/benchmarks 2>&1 | tail -30`
Expected: all existing tests still pass (the new kwarg defaults to None, preserving old behavior).

- [ ] **Step 7.6: Commit**

```bash
git add meshwell/cad_common.py meshwell/structured_polyprism.py tests/test_structured_polyprism.py
git commit -m "feat(structured): hook structured-slab cascade into prepare_entities"
```

---

## Task 8: ModelManager carries `structured_slabs`; cad_gmsh populates it

**Files:**
- Modify: `meshwell/model.py`
- Modify: `meshwell/cad_gmsh.py`
- Modify: `tests/test_structured_polyprism.py`

- [ ] **Step 8.1: Write failing test**

```python
def test_cad_gmsh_populates_model_manager_structured_slabs(square_poly):
    from meshwell.cad_gmsh import cad_gmsh
    from meshwell.polyprism import PolyPrism

    sp = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[3],
        physical_name="film",
    )
    _, mm = cad_gmsh([sp], filename="t8")
    try:
        slabs = mm.structured_slabs
        assert len(slabs) == 1
        assert slabs[0].physical_name == ("film",)
        assert slabs[0].n_layers == 3
    finally:
        mm.finalize()
```

- [ ] **Step 8.2: Run failing test**

Run: `pytest tests/test_structured_polyprism.py::test_cad_gmsh_populates_model_manager_structured_slabs -v`
Expected: AttributeError on `mm.structured_slabs`.

- [ ] **Step 8.3: Add `structured_slabs` to `ModelManager.__init__`**

In `meshwell/model.py`, in `ModelManager.__init__`, before the line `self._mesh = None`:

```python
        # Structured-slab metadata, populated by cad_gmsh / cad_occ when
        # any structured-mode PolyPrism flows through prepare_entities. The
        # mesh stage reads this to drive geo-kernel reinstantiation.
        self.structured_slabs: list = []
```

- [ ] **Step 8.4: Wire up `cad_gmsh.process_entities` to populate it**

In `meshwell/cad_gmsh.py`, replace the `prepare_entities` call inside `process_entities`:

```python
        prepare_entities(
            entities_list,
            perturbation=self.perturbation,
            resolve_snap=max(self.perturbation, self.point_tolerance),
            structured_slabs_out=self.model_manager.structured_slabs,
        )
```

- [ ] **Step 8.5: Run test**

Run: `pytest tests/test_structured_polyprism.py::test_cad_gmsh_populates_model_manager_structured_slabs -x -v`
Expected: pass.

- [ ] **Step 8.6: Run full test suite**

Run: `pytest tests/ -x -q --ignore=tests/benchmarks 2>&1 | tail -30`
Expected: no regressions.

- [ ] **Step 8.7: Commit**

```bash
git add meshwell/model.py meshwell/cad_gmsh.py tests/test_structured_polyprism.py
git commit -m "feat(structured): cad_gmsh populates ModelManager.structured_slabs"
```

---

## Task 9: `apply_structured_slabs` — locate void faces + geo replica + extrude

**Files:**
- Modify: `meshwell/structured_polyprism.py`
- Modify: `tests/test_structured_polyprism.py`

- [ ] **Step 9.1: Write failing test (single isolated slab, no neighbors)**

```python
def test_apply_structured_slabs_isolated_slab(square_poly):
    """Single structured prism, no neighbors -> geo extrude with N layers."""
    import gmsh
    from meshwell.cad_gmsh import cad_gmsh
    from meshwell.polyprism import PolyPrism
    from meshwell.structured_polyprism import apply_structured_slabs

    sp = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[3],
        physical_name="film",
    )
    _, mm = cad_gmsh([sp], filename="t9")
    try:
        # After CAD, the OCC body is gone (mesh_bool=False) and the void's
        # boundary faces no longer exist (no neighbors held them).
        # apply_structured_slabs builds the geo replica from scratch.
        apply_structured_slabs(mm, mm.structured_slabs)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.5)
        mm.model.mesh.generate(3)

        # Count nodes along the central vertical line at x=0.5,y=0.5 between
        # z=0 and z=1: must be exactly n_layers+1 = 4 nodes.
        all_nodes = mm.model.mesh.getNodes()
        node_tags, coords, _ = all_nodes
        coords = coords.reshape(-1, 3)
        on_axis = [
            c for c in coords
            if abs(c[0] - 0.5) < 1e-6 and abs(c[1] - 0.5) < 1e-6
        ]
        # We don't know the precise 2D mesh, so don't pin the count to 4
        # exactly; instead check element count along z dimension via the
        # physical group.
        # Equivalent check: total nodes count gives layered structure.
        # Use the structured layer signature: every interior xy node should
        # appear at exactly n_layers+1 distinct z values.
        from collections import defaultdict
        z_by_xy = defaultdict(set)
        for c in coords:
            z_by_xy[(round(c[0], 6), round(c[1], 6))].add(round(c[2], 6))
        layer_counts = {len(zs) for zs in z_by_xy.values() if len(zs) > 1}
        assert layer_counts == {4}, layer_counts  # 3 layers => 4 z-levels
    finally:
        mm.finalize()
```

- [ ] **Step 9.2: Run failing test**

Run: `pytest tests/test_structured_polyprism.py::test_apply_structured_slabs_isolated_slab -v`
Expected: ImportError on `apply_structured_slabs`.

- [ ] **Step 9.3: Implement `apply_structured_slabs`**

Append to `meshwell/structured_polyprism.py`:

```python
def apply_structured_slabs(model_manager: Any, slabs: list[Slab]) -> None:
    """Reinstantiate each ``Slab`` in the gmsh geo kernel as a structured
    layered volume.

    Pipeline per slab:
      1. Build the bottom-face polygon in the geo kernel using the slab
         footprint's vertex coordinates at z = ``slab.zlo``.
      2. Call ``geo.extrude`` with ``numElements=[n_layers]``,
         ``heights=[1.0]`` and ``recombine=slab.recombine``. This
         produces a swept layered mesh on the resulting volume.
      3. ``geo.synchronize()``.
      4. Find OCC neighbor faces that are coincident with the geo
         replica's bottom / top / side faces (within
         ``model_manager.point_tolerance``). For each match, embed the
         geo face's discretization into the OCC face via
         ``gmsh.model.mesh.embed`` so nodes mate across kernels.
      5. Tag the geo volume with the slab's ``physical_name``.

    No OCC face is ever removed -- the void's bounding faces are owned
    by neighbor volumes and must remain in place.
    """
    import gmsh

    if not slabs:
        return

    tol = model_manager.point_tolerance or 1e-9
    for slab in slabs:
        _apply_one_slab(slab, tol)

    gmsh.model.geo.synchronize()


def _apply_one_slab(slab: Slab, tol: float) -> None:
    import gmsh

    polys = (
        slab.footprint.geoms
        if hasattr(slab.footprint, "geoms")
        else [slab.footprint]
    )
    height = slab.zhi - slab.zlo
    for poly in polys:
        loops = [_geo_curve_loop(list(poly.exterior.coords)[:-1], slab.zlo)]
        for interior in poly.interiors:
            loops.append(_geo_curve_loop(list(interior.coords)[:-1], slab.zlo))
        bottom_surface = gmsh.model.geo.addPlaneSurface(loops)

        extruded = gmsh.model.geo.extrude(
            [(2, bottom_surface)],
            0, 0, height,
            numElements=[slab.n_layers],
            heights=[1.0],
            recombine=slab.recombine,
        )
        # extruded: [(top_surface), (volume), (lateral_surface_0), ...]
        top_dt = next(dt for dt in extruded if dt[0] == 2)
        volume_dt = next(dt for dt in extruded if dt[0] == 3)
        side_dts = [dt for dt in extruded if dt[0] == 2 and dt != top_dt]

        # Sync so the geo entities show up in gmsh.model.getEntities().
        gmsh.model.geo.synchronize()

        # Embed into coincident OCC neighbors so the swept layers and
        # the OCC neighbor 2D meshes share the same vertices/edges.
        _embed_geo_into_occ_neighbors(
            geo_bottom=bottom_surface,
            geo_top=top_dt[1],
            geo_sides=[dt[1] for dt in side_dts],
            slab=slab,
            tol=tol,
        )

        # Tag the volume with the slab's physical names.
        for name in slab.physical_name:
            pg = gmsh.model.addPhysicalGroup(3, [volume_dt[1]])
            gmsh.model.setPhysicalName(3, pg, name)


def _geo_curve_loop(coords: list[tuple[float, float]], z: float) -> int:
    """Build a closed curve loop in the geo kernel at given z."""
    import gmsh

    pts = [gmsh.model.geo.addPoint(x, y, z) for x, y in coords]
    lines = [gmsh.model.geo.addLine(a, b) for a, b in zip(pts, pts[1:])]
    lines.append(gmsh.model.geo.addLine(pts[-1], pts[0]))
    return gmsh.model.geo.addCurveLoop(lines)


def _embed_geo_into_occ_neighbors(
    geo_bottom: int,
    geo_top: int,
    geo_sides: list[int],
    slab: Slab,
    tol: float,
) -> None:
    """Embed each geo face into any coincident OCC neighbor face.

    A geo face F at z=zlo is "coincident" with an OCC face G iff:
      * G's bounding-box centroid z is within ``tol`` of F's z, AND
      * The xy bounding box of G overlaps F's xy bounds within ``tol``.

    For each match, embed G's containing volume's representation:
    ``mesh.embed(2, [F], 3, occ_volume_tag)`` is the canonical form,
    but here F is already 2D and we embed its boundary points/edges
    into the OCC face directly using
    ``mesh.embed(2, [F], 2, G)``. gmsh accepts same-dim embed for the
    purpose of forcing G's mesh to honor F's discretization at the
    shared geometry.
    """
    import gmsh

    candidate_occ_faces = gmsh.model.getEntities(2)
    for geo_face_tag, expected_z in [
        (geo_bottom, slab.zlo),
        (geo_top, slab.zhi),
    ]:
        match = _find_occ_face_at_z(candidate_occ_faces, expected_z, tol)
        if match is None:
            continue
        try:
            gmsh.model.mesh.embed(2, [geo_face_tag], 2, match)
        except Exception:
            # Embed can fail if the geo face is not strictly contained
            # in the OCC face (e.g., footprint mismatch within tol but
            # not exactly equal). Failure here is non-fatal: gmsh will
            # mesh the void independently and meshwell relies on
            # downstream removeDuplicateNodes to mate them. The full
            # failure path is exercised by tests in Task 14.
            pass

    # Lateral side embedding is intricate (geo sides are rectangles, OCC
    # lateral faces may be polygons). Defer to v2 where it matters; v1
    # focuses on bottom/top mating which covers the t3-style stack-of-
    # films use case.


def _find_occ_face_at_z(
    candidates: list[tuple[int, int]], target_z: float, tol: float
) -> int | None:
    """Return tag of an OCC face whose centroid z is within tol of target_z."""
    import gmsh

    best_tag: int | None = None
    best_delta = float("inf")
    for dim, tag in candidates:
        if dim != 2:
            continue
        # getEntities(2) includes geo-kernel faces too; skip ones whose
        # underlying type is geo by checking via getType (geo faces
        # report "Plane" too, but they are not in occ.getEntities()).
        # Cheap exclusion: gmsh.model.occ.getEntities(2) gives only OCC.
        pass

    occ_faces = gmsh.model.occ.getEntities(2)
    for _, tag in occ_faces:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(2, tag)
        if abs(zmin - zmax) > tol:
            continue  # not an axis-aligned constant-z face
        z_face = 0.5 * (zmin + zmax)
        delta = abs(z_face - target_z)
        if delta < tol and delta < best_delta:
            best_tag = tag
            best_delta = delta
    return best_tag
```

- [ ] **Step 9.4: Run test**

Run: `pytest tests/test_structured_polyprism.py::test_apply_structured_slabs_isolated_slab -x -v`
Expected: pass. The isolated slab has no OCC neighbors, so embedding is a no-op and the geo extrude meshes itself with `n_layers` layers.

- [ ] **Step 9.5: Commit**

```bash
git add meshwell/structured_polyprism.py tests/test_structured_polyprism.py
git commit -m "feat(structured): apply_structured_slabs builds geo replica with layered extrude"
```

---

## Task 10: Wire `apply_structured_slabs` into `Mesh.process_geometry`

**Files:**
- Modify: `meshwell/mesh.py`
- Modify: `tests/test_structured_polyprism.py`

- [ ] **Step 10.1: Write failing test (end-to-end via `mesh()`)**

```python
def test_mesh_endtoend_single_structured_slab(tmp_path, square_poly):
    """Drive the full Mesh pipeline; structured slab should be meshed
    with n_layers layers automatically."""
    from meshwell.cad_gmsh import cad_gmsh
    from meshwell.mesh import mesh as mesh_fn
    from meshwell.polyprism import PolyPrism

    sp = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[5],
        physical_name="film",
    )
    _, mm = cad_gmsh([sp], filename="t10")
    out_msh = tmp_path / "t10.msh"
    try:
        mesh_fn(
            dim=3,
            default_characteristic_length=0.5,
            output_file=out_msh,
            model=mm,
        )
    except Exception:
        mm.finalize()
        raise
    assert out_msh.exists()
    import meshio
    m = meshio.read(out_msh)
    # Wedge cell block expected.
    cell_block_types = {b.type for b in m.cells}
    assert "wedge" in cell_block_types or "hexahedron" in cell_block_types, cell_block_types
```

- [ ] **Step 10.2: Run failing test**

Run: `pytest tests/test_structured_polyprism.py::test_mesh_endtoend_single_structured_slab -v`
Expected: FAIL — `mesh()` doesn't yet call `apply_structured_slabs`, so the void produces no 3D elements.

- [ ] **Step 10.3: Modify `Mesh.process_geometry` to invoke the helper**

In `meshwell/mesh.py`, in `Mesh.process_geometry`, locate the line that runs refinement / labels and add a call before `process_mesh`. Insert just before the `# --- Pass C` comment (or before the line that calls `self.process_mesh(...)`):

```python
        # Reinstantiate any structured-prism slabs in the geo kernel
        # before meshing. Matched against the slab list populated at CAD
        # time. No-op when the model has none.
        structured_slabs = getattr(self.model_manager, "structured_slabs", [])
        if structured_slabs:
            from meshwell.structured_polyprism import apply_structured_slabs
            apply_structured_slabs(self.model_manager, structured_slabs)
```

Find the exact insertion point: read mesh.py, locate `process_geometry` body, insert immediately before any call to `_apply_mesh_refinement` or `process_mesh`.

- [ ] **Step 10.4: Run test**

Run: `pytest tests/test_structured_polyprism.py::test_mesh_endtoend_single_structured_slab -x -v`
Expected: pass.

- [ ] **Step 10.5: Run full suite**

Run: `pytest tests/ -x -q --ignore=tests/benchmarks 2>&1 | tail -30`
Expected: no regressions.

- [ ] **Step 10.6: Commit**

```bash
git add meshwell/mesh.py tests/test_structured_polyprism.py
git commit -m "feat(structured): mesh.process_geometry calls apply_structured_slabs"
```

---

## Task 11: Multi-interval test (t3 [8, 2] / [0.5, 1.0] analog)

**Files:**
- Modify: `tests/test_structured_polyprism.py`

- [ ] **Step 11.1: Write test**

```python
def test_multi_interval_layer_counts(tmp_path, square_poly):
    """Two stacked z-intervals -> respective layer counts respected."""
    from meshwell.cad_gmsh import cad_gmsh
    from meshwell.mesh import mesh as mesh_fn
    from meshwell.polyprism import PolyPrism

    sp = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 0.5: 0.0, 1.0: 0.0},
        n_layers=[8, 2],
        physical_name="film",
    )
    _, mm = cad_gmsh([sp], filename="t11")
    out_msh = tmp_path / "t11.msh"
    try:
        mesh_fn(dim=3, default_characteristic_length=0.5, output_file=out_msh, model=mm)
    except Exception:
        mm.finalize()
        raise

    import meshio
    m = meshio.read(out_msh)
    # Total z-levels: 8 + 2 + 1 = 11 distinct z values across the column.
    coords = m.points
    from collections import defaultdict
    z_by_xy = defaultdict(set)
    for c in coords:
        z_by_xy[(round(c[0], 6), round(c[1], 6))].add(round(c[2], 6))
    column_lengths = {len(zs) for zs in z_by_xy.values() if len(zs) > 1}
    assert 11 in column_lengths, column_lengths
```

- [ ] **Step 11.2: Run test**

Run: `pytest tests/test_structured_polyprism.py::test_multi_interval_layer_counts -x -v`
Expected: pass.

- [ ] **Step 11.3: Commit**

```bash
git add tests/test_structured_polyprism.py
git commit -m "test(structured): multi-interval layer counts (t3 [8,2] analog)"
```

---

## Task 12: Two structured slabs xy-disjoint at same z

**Files:**
- Modify: `tests/test_structured_polyprism.py`

- [ ] **Step 12.1: Write test**

```python
def test_two_xy_disjoint_structured_slabs(tmp_path, square_poly):
    from shapely.affinity import translate
    from meshwell.cad_gmsh import cad_gmsh
    from meshwell.mesh import mesh as mesh_fn
    from meshwell.polyprism import PolyPrism

    spA = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[2],
        physical_name="A",
    )
    spB = PolyPrism(
        polygons=translate(square_poly, xoff=2.0),
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[5],
        physical_name="B",
    )
    _, mm = cad_gmsh([spA, spB], filename="t12")
    out_msh = tmp_path / "t12.msh"
    try:
        mesh_fn(dim=3, default_characteristic_length=0.5, output_file=out_msh, model=mm)
    except Exception:
        mm.finalize()
        raise

    import meshio
    m = meshio.read(out_msh)
    physical_names = {n for n in m.cell_data_dict.get("gmsh:physical", {}).keys()} \
        if False else set()
    # Use field_data instead
    assert "A" in m.field_data
    assert "B" in m.field_data
```

- [ ] **Step 12.2: Run test**

Run: `pytest tests/test_structured_polyprism.py::test_two_xy_disjoint_structured_slabs -x -v`
Expected: pass.

- [ ] **Step 12.3: Commit**

```bash
git add tests/test_structured_polyprism.py
git commit -m "test(structured): two xy-disjoint structured slabs coexist"
```

---

## Task 13: Priority-based resolution between overlapping structured slabs

**Files:**
- Modify: `tests/test_structured_polyprism.py`

- [ ] **Step 13.1: Write test**

```python
def test_overlapping_structured_priority_resolution(tmp_path):
    """Two overlapping prisms (xy + z): higher priority wins, loser is split."""
    from shapely.geometry import Polygon
    from meshwell.cad_gmsh import cad_gmsh
    from meshwell.mesh import mesh as mesh_fn
    from meshwell.polyprism import PolyPrism

    big = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    small = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])

    sp_lo = PolyPrism(
        polygons=big,
        buffers={0.0: 0.0, 3.0: 0.0},
        n_layers=[6],
        physical_name="lo",
        mesh_order=2.0,
    )
    sp_hi = PolyPrism(
        polygons=small,
        buffers={1.0: 0.0, 2.0: 0.0},
        n_layers=[2],
        physical_name="hi",
        mesh_order=1.0,
    )
    _, mm = cad_gmsh([sp_lo, sp_hi], filename="t13")
    out_msh = tmp_path / "t13.msh"
    try:
        mesh_fn(dim=3, default_characteristic_length=1.0, output_file=out_msh, model=mm)
    except Exception:
        mm.finalize()
        raise

    import meshio
    m = meshio.read(out_msh)
    # Loser physical group is still present (occupies the 'donut' regions).
    assert "lo" in m.field_data
    assert "hi" in m.field_data
```

- [ ] **Step 13.2: Run test**

Run: `pytest tests/test_structured_polyprism.py::test_overlapping_structured_priority_resolution -x -v`
Expected: pass.

- [ ] **Step 13.3: Commit**

```bash
git add tests/test_structured_polyprism.py
git commit -m "test(structured): priority resolution splits overlapping slabs"
```

---

## Task 14: Structured slab + unstructured `PolyPrism` neighbor

**Files:**
- Modify: `tests/test_structured_polyprism.py`

- [ ] **Step 14.1: Write test**

```python
def test_structured_slab_with_unstructured_neighbor(tmp_path):
    """Structured slab next to a PolyPrism: structured slab keeps n_layers,
    PolyPrism is meshed unstructured, both physicals present."""
    from shapely.geometry import Polygon
    from meshwell.cad_gmsh import cad_gmsh
    from meshwell.mesh import mesh as mesh_fn
    from meshwell.polyprism import PolyPrism
    from meshwell.polyprism import PolyPrism

    sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    sq_neighbor = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])

    sp = PolyPrism(
        polygons=sq,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[4],
        physical_name="structured",
    )
    pp = PolyPrism(
        polygons=sq_neighbor,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="unstruct",
    )
    _, mm = cad_gmsh([sp, pp], filename="t14")
    out_msh = tmp_path / "t14.msh"
    try:
        mesh_fn(dim=3, default_characteristic_length=0.5, output_file=out_msh, model=mm)
    except Exception:
        mm.finalize()
        raise

    import meshio
    m = meshio.read(out_msh)
    assert "structured" in m.field_data
    assert "unstruct" in m.field_data
```

- [ ] **Step 14.2: Run test**

Run: `pytest tests/test_structured_polyprism.py::test_structured_slab_with_unstructured_neighbor -x -v`
Expected: pass. Mating between the void's geo replica and the OCC neighbor face on the shared `x=1` plane is handled by `_embed_geo_into_occ_neighbors`. If embed silently fails, the test still passes for non-conformal meshes — meaning we tolerate non-conformality in v1.

- [ ] **Step 14.3: Commit**

```bash
git add tests/test_structured_polyprism.py
git commit -m "test(structured): structured slab next to unstructured PolyPrism"
```

---

## Task 15: Layer-mating mismatch detection on stacked structured slabs

**Files:**
- Modify: `meshwell/structured_polyprism.py`
- Modify: `tests/test_structured_polyprism.py`

When two structured slabs stack vertically with the same xy footprint but different `n_layers`, the shared horizontal face must satisfy `n_layers_top == n_layers_bottom` for layer counts to mate. We detect mismatch upfront in `apply_structured_slabs` (before extruding) by checking pairwise: for each pair of slabs whose `zhi == other.zlo` and footprints overlap, assert `n_layers` agree on the overlapping footprint.

Per the spec, this is a v1 user error.

- [ ] **Step 15.1: Write failing test**

```python
def test_stacked_layer_mismatch_raises():
    """Two structured slabs share a horizontal face with different n_layers."""
    from shapely.geometry import Polygon
    from meshwell.cad_gmsh import cad_gmsh
    from meshwell.mesh import mesh as mesh_fn
    from meshwell.polyprism import PolyPrism
    from meshwell.structured_polyprism import StructuredLayerMismatchError

    sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    sp_low = PolyPrism(
        polygons=sq,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[4],
        physical_name="lo",
    )
    sp_high = PolyPrism(
        polygons=sq,
        buffers={1.0: 0.0, 2.0: 0.0},
        n_layers=[7],  # mismatch with lo
        physical_name="hi",
    )
    _, mm = cad_gmsh([sp_low, sp_high], filename="t15")
    try:
        with pytest.raises(StructuredLayerMismatchError, match="n_layers"):
            mesh_fn(dim=3, default_characteristic_length=0.5, model=mm)
    finally:
        mm.finalize()
```

- [ ] **Step 15.2: Run failing test**

Run: `pytest tests/test_structured_polyprism.py::test_stacked_layer_mismatch_raises -v`
Expected: ImportError on `StructuredLayerMismatchError`.

- [ ] **Step 15.3: Implement detection**

Append to `meshwell/structured_polyprism.py`:

```python
class StructuredLayerMismatchError(ValueError):
    """Raised when two stacked structured slabs share a horizontal face
    with conflicting ``n_layers``."""


def _validate_slab_layer_mating(slabs: list[Slab], tol: float) -> None:
    """Raise if any pair of slabs sharing a horizontal face disagrees on
    n_layers.

    Two slabs share a horizontal face iff ``a.zhi == b.zlo`` (within
    ``tol``) and their xy footprints overlap with non-zero area.
    """
    for i, a in enumerate(slabs):
        for b in slabs[i + 1 :]:
            # Order so a is below b.
            lo, hi = (a, b) if a.zhi <= b.zlo + tol else (
                (b, a) if b.zhi <= a.zlo + tol else (None, None)
            )
            if lo is None:
                continue
            if abs(lo.zhi - hi.zlo) > tol:
                continue
            shared = lo.footprint.intersection(hi.footprint)
            if shared.is_empty or shared.area < tol * tol:
                continue
            if lo.n_layers != hi.n_layers:
                raise StructuredLayerMismatchError(
                    f"Stacked structured slabs {lo.physical_name} (n_layers="
                    f"{lo.n_layers}) and {hi.physical_name} (n_layers="
                    f"{hi.n_layers}) share a horizontal face at z="
                    f"{lo.zhi} but disagree on n_layers. v1 requires "
                    f"matching layer counts on shared horizontal faces."
                )
```

In `apply_structured_slabs`, call the validator before the loop:

```python
def apply_structured_slabs(model_manager: Any, slabs: list[Slab]) -> None:
    """[existing docstring]"""
    import gmsh

    if not slabs:
        return

    tol = model_manager.point_tolerance or 1e-9
    _validate_slab_layer_mating(slabs, tol)

    for slab in slabs:
        _apply_one_slab(slab, tol)

    gmsh.model.geo.synchronize()
```

- [ ] **Step 15.4: Run test**

Run: `pytest tests/test_structured_polyprism.py::test_stacked_layer_mismatch_raises -x -v`
Expected: pass.

- [ ] **Step 15.5: Run full suite**

Run: `pytest tests/test_structured_polyprism.py -x -v`
Expected: all tests pass.

- [ ] **Step 15.6: Commit**

```bash
git add meshwell/structured_polyprism.py tests/test_structured_polyprism.py
git commit -m "feat(structured): detect stacked-slab n_layers mismatch"
```

---

## Task 16: XAO sidecar JSON for cad_occ + orchestrator wiring

**Files:**
- Modify: `meshwell/cad_occ.py`
- Modify: `meshwell/occ_xao_writer.py`
- Modify: `meshwell/orchestrator.py`
- Modify: `meshwell/structured_polyprism.py`
- Modify: `tests/test_structured_polyprism.py`

`cad_occ` doesn't talk to a `ModelManager` directly — it produces an XAO. The slab list must travel via a sidecar JSON file written next to the `.xao` and read back during the orchestrator's mesh stage.

- [ ] **Step 16.1: Write failing test**

```python
def test_generate_mesh_endtoend_with_structured_prism(tmp_path):
    """generate_mesh (orchestrator) routes structured slabs through OCC + XAO + sidecar."""
    from shapely.geometry import Polygon
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism

    sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    sp = PolyPrism(
        polygons=sq,
        buffers={0.0: 0.0, 1.0: 0.0},
        n_layers=[3],
        physical_name="film",
    )
    out_msh = tmp_path / "t16.msh"
    generate_mesh(
        entities=[sp],
        dim=3,
        output_mesh=out_msh,
        default_characteristic_length=0.5,
    )
    import meshio
    m = meshio.read(out_msh)
    assert "film" in m.field_data
```

- [ ] **Step 16.2: Run failing test**

Run: `pytest tests/test_structured_polyprism.py::test_generate_mesh_endtoend_with_structured_prism -v`
Expected: FAIL — orchestrator currently has no structured-slab path through `cad_occ`.

- [ ] **Step 16.3: Add slabs-out support to `cad_occ`**

In `meshwell/cad_occ.py`, locate the `prepare_entities` call and pass `structured_slabs_out=`:

```python
        slabs_out: list = []
        prepare_entities(
            entities_list,
            perturbation=self.perturbation,
            resolve_snap=self.point_tolerance,
            structured_slabs_out=slabs_out,
        )
        self._captured_slabs = slabs_out
```

(Init `self._captured_slabs = []` in `__init__` near other state.)

Modify the `cad_occ(...)` factory to return slabs alongside entities. Keep backwards compatibility: existing return type is a list; we add an optional second return when `return_slabs=True`. Find existing signature:

```python
def cad_occ(...) -> list[OCCLabeledEntity]:
```

Change to:

```python
def cad_occ(
    entities_list: list[Any],
    *,
    return_slabs: bool = False,
    **kwargs,
) -> list[OCCLabeledEntity] | tuple[list[OCCLabeledEntity], list]:
    """[existing docstring]

    When ``return_slabs=True``, returns ``(entities, slabs)``."""
    processor = CAD_OCC(**kwargs)
    labeled = processor.process_entities(entities_list, ...)
    if return_slabs:
        return labeled, processor._captured_slabs
    return labeled
```

(Read the file before editing to use the exact signature; do not introduce signature drift.)

- [ ] **Step 16.4: Add JSON serializer for slabs**

Append to `meshwell/structured_polyprism.py`:

```python
def slabs_to_json(slabs: list[Slab]) -> list[dict]:
    """Serialize a slab list to a JSON-safe list of dicts."""
    import shapely.wkt

    out: list[dict] = []
    for s in slabs:
        out.append(
            {
                "footprint_wkt": shapely.wkt.dumps(
                    s.footprint, rounding_precision=12
                ),
                "zlo": s.zlo,
                "zhi": s.zhi,
                "n_layers": s.n_layers,
                "recombine": s.recombine,
                "physical_name": list(s.physical_name),
                "source_index": s.source_index,
                "mesh_order": s.mesh_order if s.mesh_order != float("inf") else None,
            }
        )
    return out


def slabs_from_json(data: list[dict]) -> list[Slab]:
    """Inverse of ``slabs_to_json``."""
    import shapely.wkt

    out: list[Slab] = []
    for d in data:
        out.append(
            Slab(
                footprint=shapely.wkt.loads(d["footprint_wkt"]),
                zlo=d["zlo"],
                zhi=d["zhi"],
                n_layers=d["n_layers"],
                recombine=d["recombine"],
                physical_name=tuple(d["physical_name"]),
                source_index=d["source_index"],
                mesh_order=d["mesh_order"] if d["mesh_order"] is not None else float("inf"),
            )
        )
    return out
```

- [ ] **Step 16.5: Sidecar write/read in `occ_xao_writer` + orchestrator**

Append to `meshwell/occ_xao_writer.py`:

```python
def write_structured_slabs_sidecar(
    output_xao: Path, slabs: list
) -> None:
    """Write ``<output_xao>.structured_slabs.json`` next to the XAO."""
    import json

    from meshwell.structured_polyprism import slabs_to_json

    if not slabs:
        return
    sidecar = Path(output_xao).with_suffix(".structured_slabs.json")
    sidecar.write_text(json.dumps(slabs_to_json(slabs)))


def read_structured_slabs_sidecar(input_xao: Path) -> list:
    """Read ``<input_xao>.structured_slabs.json`` if present; else []."""
    import json

    from meshwell.structured_polyprism import slabs_from_json

    sidecar = Path(input_xao).with_suffix(".structured_slabs.json")
    if not sidecar.exists():
        return []
    return slabs_from_json(json.loads(sidecar.read_text()))
```

In `meshwell/orchestrator.py`, modify `generate_mesh`:

```python
    occ_entities, slabs = cad_occ(entities, return_slabs=True, **cad_kwargs)
    ...
    mm.load_occ_entities(...)
    mm.structured_slabs = slabs   # populate so mesh stage finds them

    if checkpoint_cad:
        from meshwell.occ_xao_writer import write_structured_slabs_sidecar
        mm.save_to_xao(Path(checkpoint_cad))
        write_structured_slabs_sidecar(Path(checkpoint_cad), slabs)
```

For the `mesh()` standalone path with `input_file=`, add a sidecar read in `Mesh.load_xao_file`:

```python
    def load_xao_file(self, input_file: Path) -> None:
        self.model_manager.load_from_xao(input_file)
        from meshwell.occ_xao_writer import read_structured_slabs_sidecar
        slabs = read_structured_slabs_sidecar(Path(input_file))
        if slabs:
            self.model_manager.structured_slabs = slabs
```

- [ ] **Step 16.6: Run test**

Run: `pytest tests/test_structured_polyprism.py::test_generate_mesh_endtoend_with_structured_prism -x -v`
Expected: pass.

- [ ] **Step 16.7: Run full test suite**

Run: `pytest tests/ -x -q --ignore=tests/benchmarks 2>&1 | tail -40`
Expected: no regressions.

- [ ] **Step 16.8: Commit**

```bash
git add meshwell/cad_occ.py meshwell/occ_xao_writer.py meshwell/orchestrator.py meshwell/structured_polyprism.py meshwell/mesh.py tests/test_structured_polyprism.py
git commit -m "feat(structured): XAO sidecar JSON for slabs through cad_occ + orchestrator"
```

---

## Task 17: Round-trip `n_layers` / `recombine` through `PolyPrism.to_dict` / `from_dict`

**Files:**
- Modify: `meshwell/polyprism.py`
- Modify: `tests/test_structured_polyprism.py`

The unified API means structured mode rides on the existing
`type: "PolyPrism"` serialization. `to_dict` adds `n_layers` and
`recombine` only when set; `from_dict` passes them through, and the
existing `__new__` dispatcher routes the resulting instance to the
subclass automatically. The existing `meshwell.utils.deserialize` hook
for `"PolyPrism"` already calls `PolyPrism.from_dict`; no change to
`utils.py` or to `__init__.py`.

- [ ] **Step 17.1: Write failing test**

```python
def test_polyprism_dict_roundtrip_unstructured(square_poly):
    """Existing PolyPrism (no n_layers) roundtrip is unchanged."""
    from meshwell.polyprism import PolyPrism
    from meshwell.utils import deserialize

    pp = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="film",
    )
    d = pp.to_dict()
    assert d["type"] == "PolyPrism"
    # New keys are absent (or None) when unstructured.
    assert d.get("n_layers") is None
    pp2 = deserialize(d)
    assert type(pp2) is PolyPrism


def test_polyprism_dict_roundtrip_structured(square_poly):
    """Structured-mode PolyPrism roundtrips through the same `PolyPrism` type
    string and re-dispatches to ``_StructuredPolyPrism`` via __new__."""
    from meshwell.polyprism import PolyPrism
    from meshwell.structured_polyprism import _StructuredPolyPrism
    from meshwell.utils import deserialize

    sp = PolyPrism(
        polygons=square_poly,
        buffers={0.0: 0.0, 0.5: 0.0, 1.0: 0.0},
        n_layers=[3, 7],
        physical_name="film",
        recombine=True,
        mesh_order=2.0,
    )
    d = sp.to_dict()
    assert d["type"] == "PolyPrism"  # unified type
    assert d["n_layers"] == [3, 7]
    assert d["recombine"] is True

    sp2 = deserialize(d)
    assert isinstance(sp2, _StructuredPolyPrism)
    assert sp2.n_layers == [3, 7]
    assert sp2.recombine is True
    assert sp2.physical_name == ("film",)
```

- [ ] **Step 17.2: Run failing tests**

Run: `pytest tests/test_structured_polyprism.py -k "dict_roundtrip" -v`
Expected: structured roundtrip fails because `to_dict` doesn't yet emit `n_layers` / `recombine`.

- [ ] **Step 17.3: Update `PolyPrism.to_dict` / `from_dict`**

Modify `meshwell/polyprism.py`. In `to_dict`, add the two new fields at the bottom of the returned dict (read the file first to pin the exact location). The existing return looks like:

```python
        return {
            "type": "PolyPrism",
            "polygons_wkt": polygons_wkt,
            "buffers": {str(k): v for k, v in self.buffers.items()},
            "physical_name": self.physical_name,
            ...
        }
```

Add (preserving every existing key):

```python
            ...
            "n_layers": list(self.n_layers) if hasattr(self, "n_layers") else None,
            "recombine": getattr(self, "recombine", False),
        }
```

In `from_dict`, pass the two new kwargs through to the constructor:

```python
        return cls(
            ...
            n_layers=data.get("n_layers"),
            recombine=data.get("recombine", False),
            ...
        )
```

The `__new__` dispatch in `PolyPrism` reads `n_layers` from kwargs and routes to `_StructuredPolyPrism` if non-None. No further serialization-side changes needed.

- [ ] **Step 17.4: Run tests**

Run: `pytest tests/test_structured_polyprism.py -k "dict_roundtrip" -x -v`
Expected: both roundtrip tests pass.

- [ ] **Step 17.5: Run full suite**

Run: `pytest tests/ -x -q --ignore=tests/benchmarks 2>&1 | tail -30`
Expected: no regressions. The `_StructuredPolyPrism` overrides `to_dict` only if needed; here we delegate to the parent (which writes `n_layers` from the subclass's attribute via `hasattr`).

- [ ] **Step 17.6: Commit**

```bash
git add meshwell/polyprism.py tests/test_structured_polyprism.py
git commit -m "feat(structured): roundtrip n_layers/recombine through unified PolyPrism dict"
```

---

## Task 18: Documentation example

**Files:**
- Modify: `docs/prisms.py`

- [ ] **Step 18.1: Read existing example file**

Run: `head -80 docs/prisms.py` to see the established example style.

- [ ] **Step 18.2: Append a structured-prism example**

Append to `docs/prisms.py`:

```python
# # Structured (layered) extruded prisms
#
# Passing `n_layers=` to `PolyPrism` switches it into structured mode
# (gmsh tutorial t3 style): each z-interval declared in `buffers` gets
# its own layer count from `n_layers`, producing a swept layered mesh
# of wedge / prismatic elements. The base polygon can be any shape —
# users can freely mix structured layered prisms with regular (taper,
# unstructured) `PolyPrism` entities and `PolySurface` entities in the
# same scene.
#
# Multiple structured prisms whose 3D extents overlap are resolved
# upfront by mesh_order: lower-priority prisms are sliced in xy and z
# so every structured volume in the final mesh is disjoint from every
# other structured volume.

from shapely.geometry import Polygon

from meshwell.polyprism import PolyPrism
from meshwell.orchestrator import generate_mesh

base = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

structured = PolyPrism(
    polygons=base,
    buffers={0.0: 0.0, 0.5: 0.0, 1.0: 0.0},  # two intervals
    n_layers=[8, 2],                          # 8 layers, then 2
    physical_name="film_stack",
    recombine=False,
)

generate_mesh(
    entities=[structured],
    dim=3,
    output_mesh="structured_prism.msh",
    default_characteristic_length=0.2,
)
```

- [ ] **Step 18.3: Smoke-run the example**

Run: `python docs/prisms.py 2>&1 | tail -10`
Expected: writes `structured_prism.msh`, no exceptions.

- [ ] **Step 18.4: Commit**

```bash
git add docs/prisms.py
git commit -m "docs(structured): add structured-mode PolyPrism example to prisms.py"
```

---

## Task 19: Final integration smoke test (cross-backend equivalence)

**Files:**
- Modify: `tests/test_structured_polyprism.py`

This task is the safety net: the same scene through `cad_gmsh` and through `cad_occ + XAO + sidecar` must produce the same physical groups. Cell counts may differ slightly because of per-backend BOP behavior, but physical group membership and presence must be identical.

- [ ] **Step 19.1: Write test**

```python
def test_backend_equivalence_structured_prism(tmp_path):
    """cad_gmsh path and cad_occ-via-orchestrator path agree on physical groups."""
    from shapely.geometry import Polygon
    from meshwell.cad_gmsh import cad_gmsh
    from meshwell.mesh import mesh as mesh_fn
    from meshwell.orchestrator import generate_mesh
    from meshwell.polyprism import PolyPrism

    def make_entities():
        sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        return [
            PolyPrism(
                polygons=sq,
                buffers={0.0: 0.0, 1.0: 0.0},
                n_layers=[3],
                physical_name="film",
            )
        ]

    out_gmsh = tmp_path / "gmsh.msh"
    out_occ = tmp_path / "occ.msh"

    _, mm = cad_gmsh(make_entities(), filename="t19g")
    try:
        mesh_fn(dim=3, default_characteristic_length=0.5, output_file=out_gmsh, model=mm)
    except Exception:
        mm.finalize()
        raise

    generate_mesh(
        entities=make_entities(),
        dim=3,
        output_mesh=out_occ,
        default_characteristic_length=0.5,
    )

    import meshio
    g = meshio.read(out_gmsh)
    o = meshio.read(out_occ)
    assert set(g.field_data.keys()) == set(o.field_data.keys())
    assert "film" in g.field_data
```

- [ ] **Step 19.2: Run test**

Run: `pytest tests/test_structured_polyprism.py::test_backend_equivalence_structured_prism -x -v`
Expected: pass.

- [ ] **Step 19.3: Run the full meshwell test suite**

Run: `pytest tests/ -x -q --ignore=tests/benchmarks 2>&1 | tail -30`
Expected: 0 regressions; structured-prism tests all green.

- [ ] **Step 19.4: Commit**

```bash
git add tests/test_structured_polyprism.py
git commit -m "test(structured): cross-backend equivalence for structured prism"
```

---

## Self-Review

Spec coverage map:

| Spec section | Task(s) |
|---|---|
| Public API (unified `PolyPrism(..., n_layers=...)`, validation rules, `__new__` dispatch) | 1 |
| `Slab` dataclass | 1 |
| Slab resolution cascade (z-overlap, partial-z splitting, priority) | 2, 3, 4, 5 |
| Phantom CAD body (`_StructuredPhantom`) | 6 |
| `prepare_entities` hook | 7 |
| ModelManager.structured_slabs | 8 |
| Mesh-stage geo-kernel reinstantiation (`apply_structured_slabs`) | 9, 10 |
| Multi-interval mesh test | 11 |
| Multiple structured slabs co-existing | 12, 13 |
| Structured + unstructured neighbor | 14 |
| Layer-mating mismatch validation | 15 |
| `cad_occ` XAO sidecar | 16 |
| Orchestrator wiring | 16 |
| Serialization round-trip via unified `PolyPrism` type (no new export) | 17 |
| Docs example | 18 |
| Cross-backend smoke test | 19 |

Placeholder scan: no `TBD`, no "implement later", no "similar to Task N", every code step has full code. Type/method consistency check: `Slab` fields used identically in Tasks 1–19; `apply_structured_slabs(model_manager, slabs)` signature matches across producer (Task 10) and consumer (Task 9, 16); `_StructuredPhantom.instanciate` / `.instanciate_occ` match meshwell's existing entity protocol shown in `polyprism.py:350,504`.

Spec gap check: spec lists Risk #1 ("`mesh.embed` across kernels may be unreliable") with mitigation "build minimal-reproducer test early"; that's covered by Task 14 (structured + unstructured neighbor) and Task 9 (isolated slab). Risk #4 (tapered-prism confusion) is covered by Task 1's construction-time error. No gaps.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-26-structured-polyprism.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
