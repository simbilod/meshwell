# Structured Sweeps (2D bands) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Anisotropic structured "bands" of cells (tensor grids) grown from interfaces/boundaries/polylines, conformal with unstructured triangles, per `docs/superpowers/specs/2026-09-06-meshwell-structured-sweeps-design.md`.

**Architecture:** A CAD-side declaration (`StructuredSweep`) is clipped in shapely and imprinted into the post-BOP OCC model as a final fragment pass; synthetic `__sweep|…` physical groups written into the XAO make the mesh stage fully discovery-based (works identically for `generate_mesh` and separate CAD→mesh steps). A 2D stamping kernel freezes seam-curve nodes and hand-stamps triangle-pairs/quads, then `Mesh.MeshOnlyEmpty=1` lets gmsh fill only unstructured surfaces.

**Tech Stack:** Python, shapely 2.x, OCP (OpenCASCADE), gmsh Python API, pydantic (v1-style `class Config`), pytest via `uv run pytest`.

## Global Constraints

- Repo: `meshwell` (this repo). All paths relative to repo root.
- Read the spec first: `docs/superpowers/specs/2026-09-06-meshwell-structured-sweeps-design.md`.
- Synthetic name formats (exact): faces `__sweep|<name>|<side>|<i>`, source curves `__sweepsrc|<name>`. `|` is the parse separator; user-facing names must not contain `|` (validate).
- Normal convention: "left of travel" = +90° rotation of the tangent, matching meshwell's material-left-of-travel arc convention.
- `thickness`/`normal` dict keys depend on attachment: interface `a___b` → subset of `{a, b}`; boundary `a___None` → exactly `{a}`; PolyLine → subset of `{"left", "right"}`.
- Phase 1: attachment curves must be straight (2 points after merge+simplify); curved sources raise `SweepCurvedSourceError`.
- All new exceptions live in `meshwell/structured/exceptions.py` and subclass its existing `StructuredMeshingError` base (check the actual base class name in that file first; if it has no common base, subclass `ValueError` like the existing ones).
- Follow existing code style: `from __future__ import annotations`, module-level `logger = logging.getLogger(__name__)`, Google-style docstrings.
- Run the full suite `uv run pytest -x -q` before the final commit of Tasks 5–8; earlier tasks run their own test file only.
- Commit after every green test cycle. Message style: `feat(sweep): …`, `test(sweep): …`.

---

### Task 1: `Graded`, `StructuredSweepResolutionSpec`, alias, and `resolve_normal_offsets`

**Files:**
- Modify: `meshwell/resolution.py` (append after `StructuredExtrusionResolutionSpec`, then refactor it)
- Modify: `meshwell/structured/exceptions.py` (append)
- Test: `tests/test_sweep_spec.py` (create)

**Interfaces:**
- Consumes: existing `ResolutionSpec` pydantic base in `meshwell/resolution.py`.
- Produces (used by Tasks 5–7):
  - `Graded(h0: float, ratio: float)` — pydantic model.
  - `StructuredSweepResolutionSpec(tangential: float | list[float] | None, normal: dict[str, int | Graded | list[float]], element_type: Literal["triangle","quad"])` with no-op `apply()`.
  - `StructuredExtrusionResolutionSpec(n_layers=N)` still works exactly as today (wedge pipeline reads `.n_layers`).
  - `resolve_normal_offsets(normal_spec, thickness: float, atol: float) -> np.ndarray` — increasing array from `0.0` to `thickness`.
  - `SweepNormalExtentError` exception.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sweep_spec.py
import numpy as np
import pytest

from meshwell.resolution import (
    Graded,
    StructuredExtrusionResolutionSpec,
    StructuredSweepResolutionSpec,
    resolve_normal_offsets,
)
from meshwell.structured.exceptions import SweepNormalExtentError


def test_resolve_int_uniform():
    out = resolve_normal_offsets(4, thickness=1.0, atol=1e-9)
    np.testing.assert_allclose(out, [0.0, 0.25, 0.5, 0.75, 1.0])


def test_resolve_graded_fills_thickness():
    out = resolve_normal_offsets(Graded(h0=0.1, ratio=2.0), thickness=1.0, atol=1e-9)
    assert out[0] == 0.0
    assert out[-1] == pytest.approx(1.0)
    assert np.all(np.diff(out) > 0)
    # first cell is exactly h0
    assert out[1] == pytest.approx(0.1)


def test_resolve_graded_ratio_one_is_uniform():
    out = resolve_normal_offsets(Graded(h0=0.25, ratio=1.0), thickness=1.0, atol=1e-9)
    np.testing.assert_allclose(out, [0.0, 0.25, 0.5, 0.75, 1.0])


def test_resolve_explicit_array_validated():
    out = resolve_normal_offsets([0.0, 0.2, 1.0], thickness=1.0, atol=1e-9)
    np.testing.assert_allclose(out, [0.0, 0.2, 1.0])
    with pytest.raises(SweepNormalExtentError):
        resolve_normal_offsets([0.0, 0.2, 0.9], thickness=1.0, atol=1e-9)
    with pytest.raises(SweepNormalExtentError):
        resolve_normal_offsets([0.1, 0.2, 1.0], thickness=1.0, atol=1e-9)


def test_sweep_spec_fields_and_noop_apply():
    spec = StructuredSweepResolutionSpec(
        tangential=0.05,
        normal={"well_1": Graded(h0=1e-3, ratio=1.3), "sch": 3},
        element_type="quad",
    )
    assert spec.apply() is None  # no-op


def test_graded_validation():
    with pytest.raises(Exception):
        Graded(h0=-1.0, ratio=1.3)
    with pytest.raises(Exception):
        Graded(h0=1e-3, ratio=0.5)


def test_extrusion_spec_alias_unchanged():
    spec = StructuredExtrusionResolutionSpec(n_layers=3)
    assert spec.n_layers == 3
    assert spec.apply() is None
    # subclass relationship lets shared machinery treat both uniformly
    assert isinstance(spec, StructuredSweepResolutionSpec)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sweep_spec.py -q`
Expected: FAIL with `ImportError: cannot import name 'Graded'`

- [ ] **Step 3: Implement**

In `meshwell/structured/exceptions.py`, first read the file to find the existing base-class pattern, then append (adjusting the base class to match):

```python
class SweepNormalExtentError(ValueError):
    """Explicit normal offset array does not span [0, thickness]."""

    def __init__(self, offsets, thickness: float):
        super().__init__(
            f"Explicit normal offsets must start at 0.0 and end at the sweep "
            f"thickness {thickness!r}; got first={offsets[0]!r}, last={offsets[-1]!r}. "
            "Use Graded(h0, ratio) or an int layer count to avoid restating thickness."
        )
```

In `meshwell/resolution.py`, add near the top-level classes (keep `numpy as np` import that already exists):

```python
class Graded(BaseModel):
    """Geometric grading for a structured sweep's normal direction.

    Cell k has size ``h0 * ratio**k``; cells are emitted until the sweep
    thickness (known only to the CAD stage) is filled, with the last
    cell adjusted to land exactly on the thickness. Carries NO thickness
    on purpose — see the structured-sweeps design doc.
    """

    h0: float = Field(gt=0)
    ratio: float = Field(ge=1)


class StructuredSweepResolutionSpec(ResolutionSpec):
    """Discretization of a structured sweep (2D band today, 3D later).

    Consumed by the sweep stamping kernel, not by gmsh size fields,
    hence the no-op ``apply()``. Keyed in ``resolution_specs`` by the
    ``StructuredSweep.name`` it discretizes.
    """

    apply_to: Literal["surfaces"] = "surfaces"
    tangential: float | list[float] | None = None
    normal: dict[str, int | Graded | list[float]] = Field(default_factory=dict)
    element_type: Literal["triangle", "quad"] = "triangle"

    class Config:
        arbitrary_types_allowed = True

    def apply(self, **_kwargs) -> None:
        """No-op: consumed by the sweep stamping kernel."""


def resolve_normal_offsets(normal_spec, thickness: float, atol: float) -> "np.ndarray":
    """Resolve a per-side normal spec into offsets [0, ..., thickness].

    ``int n`` -> n uniform layers. ``Graded`` -> geometric cells, last
    cell adjusted to land on thickness. Explicit array -> validated to
    span [0, thickness] within ``atol`` (SweepNormalExtentError).
    """
    from meshwell.structured.exceptions import SweepNormalExtentError

    if isinstance(normal_spec, int):
        return np.linspace(0.0, thickness, normal_spec + 1)
    if isinstance(normal_spec, Graded):
        offsets = [0.0]
        h = normal_spec.h0
        while offsets[-1] + h < thickness - atol:
            offsets.append(offsets[-1] + h)
            h *= normal_spec.ratio
        offsets.append(thickness)
        # merge a sliver last cell into its neighbour for quality
        if len(offsets) >= 3 and (offsets[-1] - offsets[-2]) < 0.5 * (
            offsets[-2] - offsets[-3]
        ):
            del offsets[-2]
        return np.asarray(offsets)
    offsets = np.asarray(normal_spec, dtype=float)
    if abs(offsets[0]) > atol or abs(offsets[-1] - thickness) > atol:
        raise SweepNormalExtentError(offsets, thickness)
    return offsets
```

Then refactor the existing `StructuredExtrusionResolutionSpec` (keep its
docstring, keep the `n_layers` field so `meshwell/structured/wedge.py::resolve_n_layers`
is untouched) to:

```python
class StructuredExtrusionResolutionSpec(StructuredSweepResolutionSpec):
    """Number of z-layers per structured slab for wedge stamping.

    Deprecated alias of ``StructuredSweepResolutionSpec(normal=n_layers)``;
    kept because the 3D wedge kernel reads ``n_layers`` directly.
    """

    apply_to: Literal["volumes"] = "volumes"  # type: ignore[assignment]
    n_layers: int = Field(default=1, ge=1)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_sweep_spec.py -q` — Expected: PASS.
Also run the wedge regression subset: `uv run pytest tests/ -q -k "structured"` — Expected: no new failures (the alias must not break wedge tests).

- [ ] **Step 5: Commit**

```bash
git add meshwell/resolution.py meshwell/structured/exceptions.py tests/test_sweep_spec.py
git commit -m "feat(sweep): Graded + StructuredSweepResolutionSpec; extrusion spec becomes alias"
```

---

### Task 2: `StructuredSweep` declaration class

**Files:**
- Create: `meshwell/structured/sweep.py`
- Modify: `meshwell/structured/exceptions.py` (append)
- Test: `tests/test_sweep_declaration.py` (create)

**Interfaces:**
- Produces (used by Tasks 5–6):
  - `StructuredSweep(name: str, on: str, thickness: dict[str, float])`
  - `.attachment_kind` property → `"interface" | "boundary" | "polyline"` (parsed from `on`: contains `___` with second part `"None"` → boundary; contains `___` otherwise → interface; else polyline).
  - `.sides()` → `list[str]` (thickness keys, validated).
  - `.to_dict() -> dict` / `StructuredSweep.from_dict(d) -> StructuredSweep`.
  - `SweepKeyError` exception (invalid thickness keys for the attachment kind).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sweep_declaration.py
import pytest

from meshwell.structured.exceptions import SweepKeyError
from meshwell.structured.sweep import StructuredSweep


def test_interface_keys():
    s = StructuredSweep(name="qw", on="a___b", thickness={"a": 0.1, "b": 0.2})
    assert s.attachment_kind == "interface"
    assert set(s.sides()) == {"a", "b"}
    with pytest.raises(SweepKeyError):
        StructuredSweep(name="qw", on="a___b", thickness={"c": 0.1})


def test_boundary_keys():
    s = StructuredSweep(name="bot", on="a___None", thickness={"a": 0.1})
    assert s.attachment_kind == "boundary"
    with pytest.raises(SweepKeyError):
        StructuredSweep(name="bot", on="a___None", thickness={"a": 0.1, "None": 0.1})


def test_polyline_keys():
    s = StructuredSweep(name="jn", on="junction_line", thickness={"left": 0.1})
    assert s.attachment_kind == "polyline"
    with pytest.raises(SweepKeyError):
        StructuredSweep(name="jn", on="junction_line", thickness={"a": 0.1})


def test_name_validation():
    with pytest.raises(ValueError):
        StructuredSweep(name="bad|name", on="a___b", thickness={"a": 0.1})
    with pytest.raises(ValueError):
        StructuredSweep(name="qw", on="a___b", thickness={"a": -0.1})


def test_roundtrip_serialization():
    s = StructuredSweep(name="qw", on="a___b", thickness={"a": 0.1})
    d = s.to_dict()
    assert d["type"] == "StructuredSweep"
    s2 = StructuredSweep.from_dict(d)
    assert s2.name == s.name and s2.on == s.on and s2.thickness == s.thickness
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_sweep_declaration.py -q` — Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `meshwell/structured/sweep.py`**

```python
"""StructuredSweep: CAD-side declaration of a structured band.

Geometry only (attachment + per-side thickness). Discretization lives in
StructuredSweepResolutionSpec, paired by ``name`` — see the design doc.
"""
from __future__ import annotations

from meshwell.structured.exceptions import SweepKeyError

_INTERFACE_DELIMITER = "___"
_BOUNDARY_SUFFIX = "None"
_POLYLINE_KEYS = frozenset({"left", "right"})


class StructuredSweep:
    """Declares a structured band grown from a dim-(N-1) physical name."""

    def __init__(self, name: str, on: str, thickness: dict[str, float]):
        if "|" in name or "|" in on:
            raise ValueError(f"'|' not allowed in sweep name/attachment: {name!r}, {on!r}")
        if not thickness:
            raise ValueError(f"Sweep {name!r}: thickness dict must not be empty")
        for side, t in thickness.items():
            if not t > 0:
                raise ValueError(f"Sweep {name!r}: thickness[{side!r}] must be > 0, got {t!r}")
        self.name = name
        self.on = on
        self.thickness = dict(thickness)
        self._validate_keys()

    @property
    def attachment_kind(self) -> str:
        parts = self.on.split(_INTERFACE_DELIMITER)
        if len(parts) == 2:
            return "boundary" if parts[1] == _BOUNDARY_SUFFIX else "interface"
        return "polyline"

    def _admissible_keys(self) -> frozenset[str]:
        kind = self.attachment_kind
        parts = self.on.split(_INTERFACE_DELIMITER)
        if kind == "interface":
            return frozenset(parts)
        if kind == "boundary":
            return frozenset({parts[0]})
        return _POLYLINE_KEYS

    def _validate_keys(self) -> None:
        admissible = self._admissible_keys()
        bad = set(self.thickness) - admissible
        if bad:
            raise SweepKeyError(self.name, self.attachment_kind, bad, admissible)

    def sides(self) -> list[str]:
        return list(self.thickness)

    def to_dict(self) -> dict:
        return {
            "type": "StructuredSweep",
            "name": self.name,
            "on": self.on,
            "thickness": self.thickness,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StructuredSweep":
        return cls(name=data["name"], on=data["on"], thickness=data["thickness"])
```

Append to `meshwell/structured/exceptions.py`:

```python
class SweepKeyError(ValueError):
    """thickness/normal keys invalid for the sweep's attachment kind."""

    def __init__(self, name, kind, bad_keys, admissible):
        super().__init__(
            f"Sweep {name!r} ({kind} attachment): invalid side keys {sorted(bad_keys)}; "
            f"admissible keys are {sorted(admissible)}."
        )
```

- [ ] **Step 4: Run** `uv run pytest tests/test_sweep_declaration.py -q` — PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/sweep.py meshwell/structured/exceptions.py tests/test_sweep_declaration.py
git commit -m "feat(sweep): StructuredSweep declaration with attachment-dependent key validation"
```

---

### Task 3: shapely-side region polygons and attachment resolution

**Files:**
- Create: `meshwell/structured/sweep_cad.py`
- Modify: `meshwell/structured/exceptions.py` (append)
- Test: `tests/test_sweep_cad_resolve.py` (create)

**Interfaces:**
- Consumes: `StructuredSweep` (Task 2); meshwell 2D entities exposing `.polygons` (list of shapely Polygons), `.physical_name` (tuple), `.mesh_order` (float), `.mesh_bool`, `.dimension`; `PolyLine` exposing `.linestrings`.
- Produces (used by Tasks 4–5):
  - `final_region_polygons(entities) -> dict[str, shapely Polygon|MultiPolygon]` — per physical name after mesh_order precedence (lower number wins overlaps).
  - `resolve_attachment(sweep, entities, region_polys, point_tolerance) -> tuple[np.ndarray, np.ndarray]` — (p0, p1) endpoints of the straight attachment segment; for polylines, preserves user coordinate order.
  - `side_normal(p0, p1, side, sweep, region_polys, point_tolerance) -> np.ndarray` — unit normal pointing into that side.
  - Exceptions: `SweepAttachmentNotFoundError`, `SweepCurvedSourceError`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sweep_cad_resolve.py
import numpy as np
import pytest
import shapely

from meshwell.polyline import PolyLine
from meshwell.polysurface import PolySurface
from meshwell.structured.exceptions import (
    SweepAttachmentNotFoundError,
    SweepCurvedSourceError,
)
from meshwell.structured.sweep import StructuredSweep
from meshwell.structured.sweep_cad import (
    final_region_polygons,
    resolve_attachment,
    side_normal,
)


def _stack():
    lower = PolySurface(polygons=shapely.box(0, 0, 4, 1), physical_name="lower", mesh_order=2)
    upper = PolySurface(polygons=shapely.box(0, 1, 4, 2), physical_name="upper", mesh_order=1)
    return [lower, upper]


def test_final_region_polygons_mesh_order_precedence():
    # overlapping boxes: mesh_order 1 wins the overlap strip
    a = PolySurface(polygons=shapely.box(0, 0, 2, 2), physical_name="a", mesh_order=2)
    b = PolySurface(polygons=shapely.box(1, 0, 3, 2), physical_name="b", mesh_order=1)
    regions = final_region_polygons([a, b])
    assert regions["b"].area == pytest.approx(4.0)
    assert regions["a"].area == pytest.approx(2.0)  # lost the overlap


def test_resolve_interface_attachment():
    entities = _stack()
    sweep = StructuredSweep(name="s", on="lower___upper", thickness={"upper": 0.5})
    regions = final_region_polygons(entities)
    p0, p1 = resolve_attachment(sweep, entities, regions, point_tolerance=1e-6)
    ys = {p0[1], p1[1]}
    assert ys == {pytest.approx(1.0)}
    assert {min(p0[0], p1[0]), max(p0[0], p1[0])} == {0.0, 4.0}


def test_resolve_boundary_attachment():
    entities = _stack()
    sweep = StructuredSweep(name="s", on="lower___None", thickness={"lower": 0.5})
    regions = final_region_polygons(entities)
    # lower's hull boundary is 3 sides (bottom + two laterals) -> not a single
    # straight segment -> curved-source error is the phase-1 contract
    with pytest.raises(SweepCurvedSourceError):
        resolve_attachment(sweep, entities, regions, point_tolerance=1e-6)


def test_resolve_boundary_attachment_straight():
    # single region: its bottom edge selected via a PolyLine-free trick is not
    # possible; use a wide flat region whose hull IS a rectangle -> still 4 sides.
    # Straight boundary attachment therefore uses an embedded PolyLine in
    # practice; this test pins the error message mentions PolyLine.
    entities = _stack()
    sweep = StructuredSweep(name="s", on="lower___None", thickness={"lower": 0.5})
    regions = final_region_polygons(entities)
    with pytest.raises(SweepCurvedSourceError, match="PolyLine"):
        resolve_attachment(sweep, entities, regions, point_tolerance=1e-6)


def test_resolve_polyline_attachment_preserves_orientation():
    entities = _stack()
    pl = PolyLine(
        linestrings=shapely.LineString([(3.0, 0.5), (1.0, 0.5)]),
        physical_name="jline",
    )
    entities.append(pl)
    sweep = StructuredSweep(name="s", on="jline", thickness={"left": 0.2})
    regions = final_region_polygons(entities)
    p0, p1 = resolve_attachment(sweep, entities, regions, point_tolerance=1e-6)
    np.testing.assert_allclose(p0, [3.0, 0.5])
    np.testing.assert_allclose(p1, [1.0, 0.5])
    # travel direction is -x, so "left" is -y
    n = side_normal(p0, p1, "left", sweep, regions, point_tolerance=1e-6)
    np.testing.assert_allclose(n, [0.0, -1.0], atol=1e-12)


def test_missing_attachment_raises():
    entities = _stack()
    sweep = StructuredSweep(name="s", on="lower___nosuch", thickness={"lower": 0.5})
    regions = final_region_polygons(entities)
    with pytest.raises(SweepAttachmentNotFoundError):
        resolve_attachment(sweep, entities, regions, point_tolerance=1e-6)


def test_interface_side_normals_point_into_regions():
    entities = _stack()
    sweep = StructuredSweep(name="s", on="lower___upper", thickness={"upper": 0.5, "lower": 0.3})
    regions = final_region_polygons(entities)
    p0, p1 = resolve_attachment(sweep, entities, regions, point_tolerance=1e-6)
    n_up = side_normal(p0, p1, "upper", sweep, regions, point_tolerance=1e-6)
    n_lo = side_normal(p0, p1, "lower", sweep, regions, point_tolerance=1e-6)
    assert n_up[1] == pytest.approx(1.0)
    assert n_lo[1] == pytest.approx(-1.0)


def test_curved_polyline_rejected():
    entities = _stack()
    pl = PolyLine(
        linestrings=shapely.LineString([(1, 0.5), (2, 0.7), (3, 0.5)]),
        physical_name="curvy",
    )
    entities.append(pl)
    sweep = StructuredSweep(name="s", on="curvy", thickness={"left": 0.1})
    regions = final_region_polygons(entities)
    with pytest.raises(SweepCurvedSourceError):
        resolve_attachment(sweep, entities, regions, point_tolerance=1e-6)
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_sweep_cad_resolve.py -q` → ImportError.

- [ ] **Step 3: Implement in `meshwell/structured/sweep_cad.py`**

```python
"""CAD-stage sweep pass: region resolution, attachment, clip, imprint."""
from __future__ import annotations

import logging

import numpy as np
import shapely
from shapely.geometry import LineString, Point

from meshwell.structured.exceptions import (
    SweepAttachmentNotFoundError,
    SweepCurvedSourceError,
)

logger = logging.getLogger(__name__)

_DELIM = "___"


def final_region_polygons(entities) -> dict:
    """Final per-physical-name 2D region polygons after mesh_order precedence.

    Lower mesh_order overrides higher (meshwell convention). Entities with
    mesh_order None sort last, in list order (mirrors cad ordering).
    """
    surf = [e for e in entities if getattr(e, "dimension", None) == 2 and e.mesh_bool]
    order = sorted(
        range(len(surf)),
        key=lambda i: (surf[i].mesh_order is None, surf[i].mesh_order, i),
    )
    taken = None
    out: dict = {}
    for i in order:
        ent = surf[i]
        poly = shapely.unary_union(ent.polygons)
        if taken is not None:
            poly = poly.difference(taken)
        taken = poly if taken is None else shapely.unary_union([taken, poly])
        for name in ent.physical_name or ():
            out[name] = poly if name not in out else shapely.unary_union([out[name], poly])
    return out


def _as_straight_segment(geom, point_tolerance: float, context: str):
    """Merge + simplify a linear geometry; require a single straight segment."""
    merged = shapely.line_merge(geom) if geom.geom_type != "LineString" else geom
    if merged.geom_type != "LineString" or merged.is_empty:
        raise SweepCurvedSourceError(context, merged.geom_type)
    simple = merged.simplify(point_tolerance)
    coords = list(simple.coords)
    if len(coords) != 2:
        raise SweepCurvedSourceError(context, f"{len(coords)}-point polyline")
    return np.asarray(coords[0]), np.asarray(coords[1])


def resolve_attachment(sweep, entities, region_polys, point_tolerance: float):
    """Return (p0, p1) endpoints of the sweep's straight attachment segment."""
    kind = sweep.attachment_kind
    if kind == "polyline":
        for ent in entities:
            if getattr(ent, "dimension", None) == 1 and sweep.on in (ent.physical_name or ()):
                lines = ent.linestrings
                line = lines[0] if isinstance(lines, list) else line_from(lines)
                coords = list((lines[0] if isinstance(lines, list) else lines).coords)
                if len(coords) != 2:
                    # allow collinear multi-point lines
                    simple = LineString(coords).simplify(point_tolerance)
                    coords = list(simple.coords)
                if len(coords) != 2:
                    raise SweepCurvedSourceError(sweep.on, f"{len(coords)}-point polyline")
                return np.asarray(coords[0]), np.asarray(coords[1])
        raise SweepAttachmentNotFoundError(sweep.name, sweep.on)

    a, b = sweep.on.split(_DELIM)
    if a not in region_polys:
        raise SweepAttachmentNotFoundError(sweep.name, sweep.on)
    if kind == "interface":
        if b not in region_polys:
            raise SweepAttachmentNotFoundError(sweep.name, sweep.on)
        shared = region_polys[a].boundary.intersection(region_polys[b].boundary)
    else:  # boundary
        hull = shapely.unary_union(list(region_polys.values()))
        shared = region_polys[a].boundary.intersection(hull.boundary)
    if shared.is_empty:
        raise SweepAttachmentNotFoundError(sweep.name, sweep.on)
    return _as_straight_segment(shared, point_tolerance, sweep.on)


def side_normal(p0, p1, side: str, sweep, region_polys, point_tolerance: float):
    """Unit normal from the segment into the given side."""
    t = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    t = t / np.linalg.norm(t)
    left = np.array([-t[1], t[0]])  # +90 deg: material-left-of-travel
    if side == "left":
        return left
    if side == "right":
        return -left
    mid = (np.asarray(p0) + np.asarray(p1)) / 2.0
    eps = sweep.thickness[side] * 1e-3 + point_tolerance
    if region_polys[side].contains(Point(*(mid + left * eps))):
        return left
    if region_polys[side].contains(Point(*(mid - left * eps))):
        return -left
    raise SweepAttachmentNotFoundError(sweep.name, f"{sweep.on} (side {side} not adjacent)")
```

Note for the implementer: the `line_from` reference above is a plan typo
guard — delete that dead line and keep only the `coords`-based path; the
tests define correctness. Append the two exceptions:

```python
class SweepAttachmentNotFoundError(ValueError):
    """The sweep's ``on`` name doesn't resolve to geometry in the model."""

    def __init__(self, name, on):
        super().__init__(f"Sweep {name!r}: attachment {on!r} not found/adjacent in model.")


class SweepCurvedSourceError(ValueError):
    """Attachment resolved to a non-straight source (phase 3 territory)."""

    def __init__(self, on, got):
        super().__init__(
            f"Sweep attachment {on!r} is not a single straight segment (got {got}). "
            "Curved/multi-segment sources are not supported in phase 1; for a "
            "straight subset of a boundary, attach to an embedded PolyLine instead."
        )
```

- [ ] **Step 4: Run** `uv run pytest tests/test_sweep_cad_resolve.py -q` — PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/sweep_cad.py meshwell/structured/exceptions.py tests/test_sweep_cad_resolve.py
git commit -m "feat(sweep): region resolution and straight-attachment lookup"
```

---

### Task 4: clip algorithm (deficit-shadow subtraction)

**Files:**
- Modify: `meshwell/structured/sweep_cad.py` (append)
- Modify: `meshwell/structured/exceptions.py` (append `SweepOverlapError`)
- Test: `tests/test_sweep_clip.py` (create)

**Interfaces:**
- Produces (used by Task 5):
  - `clip_sweep_side(p0, p1, n_dir, thickness, region_poly, point_tolerance) -> list[tuple[float, float]]` — kept tangential intervals (arclength from p0), full-normal-extent rule.
  - `sweep_rectangles(p0, p1, n_dir, thickness, intervals) -> list[shapely.Polygon]`
  - `SweepOverlapError`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sweep_clip.py
import numpy as np
import pytest
import shapely

from meshwell.structured.sweep_cad import clip_sweep_side, sweep_rectangles

P0 = np.array([0.0, 1.0])
P1 = np.array([4.0, 1.0])
UP = np.array([0.0, 1.0])


def test_clean_band_keeps_full_interval():
    region = shapely.box(0, 1, 4, 2)
    assert clip_sweep_side(P0, P1, UP, 0.5, region, 1e-6) == [(0.0, pytest.approx(4.0))]


def test_corner_notch_blocks_its_shadow():
    # region loses a notch x in [3, 4], y in [1, 1.3]: band (thickness .5)
    # can't reach full extent there -> entire [3,4] shadow dropped
    region = shapely.box(0, 1, 4, 2).difference(shapely.box(3, 1, 4, 1.3))
    kept = clip_sweep_side(P0, P1, UP, 0.5, region, 1e-6)
    assert len(kept) == 1
    lo, hi = kept[0]
    assert lo == pytest.approx(0.0) and hi == pytest.approx(3.0)


def test_mid_obstacle_splits_interval():
    region = shapely.box(0, 1, 4, 2).difference(shapely.box(1.5, 1, 2.5, 2))
    kept = clip_sweep_side(P0, P1, UP, 0.5, region, 1e-6)
    assert [pytest.approx(v) for iv in kept for v in iv] == [0.0, 1.5, 2.5, 4.0]


def test_band_thicker_than_region_drops_everything():
    region = shapely.box(0, 1, 4, 1.2)
    assert clip_sweep_side(P0, P1, UP, 0.5, region, 1e-6) == []


def test_sweep_rectangles():
    rects = sweep_rectangles(P0, P1, UP, 0.5, [(0.0, 1.5), (2.5, 4.0)])
    assert len(rects) == 2
    assert rects[0].bounds == (0.0, 1.0, 1.5, 1.5)
    assert rects[1].bounds == (2.5, 1.0, 4.0, 1.5)
```

- [ ] **Step 2: Run to verify failure** — ImportError.

- [ ] **Step 3: Implement (append to `sweep_cad.py`)**

```python
def clip_sweep_side(p0, p1, n_dir, thickness, region_poly, point_tolerance):
    """Kept tangential intervals where the FULL normal extent fits in region.

    Any deficit piece (rect minus region) blocks its entire tangential
    shadow — per the design's full-normal-extent rule.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    t_hat = (p1 - p0) / np.linalg.norm(p1 - p0)
    length = float(np.linalg.norm(p1 - p0))
    rect = shapely.Polygon(
        [p0, p1, p1 + n_dir * thickness, p0 + n_dir * thickness]
    )
    deficit = rect.difference(region_poly.buffer(point_tolerance))
    kept = [(0.0, length)]
    pieces = getattr(deficit, "geoms", [deficit]) if not deficit.is_empty else []
    for piece in pieces:
        if piece.area <= (10 * point_tolerance) ** 2:
            continue  # tolerance sliver, not a real deficit
        ts = [float(np.dot(np.asarray(c) - p0, t_hat)) for c in piece.exterior.coords]
        blo, bhi = min(ts), max(ts)
        nxt = []
        for lo, hi in kept:
            if bhi <= lo or blo >= hi:
                nxt.append((lo, hi))
                continue
            if blo > lo:
                nxt.append((lo, blo))
            if bhi < hi:
                nxt.append((bhi, hi))
        kept = nxt
    return [(lo, hi) for lo, hi in kept if hi - lo > 10 * point_tolerance]


def sweep_rectangles(p0, p1, n_dir, thickness, intervals):
    """Shapely rectangles for the kept intervals, in the (t, n) frame."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    t_hat = (p1 - p0) / np.linalg.norm(p1 - p0)
    out = []
    for lo, hi in intervals:
        a = p0 + t_hat * lo
        b = p0 + t_hat * hi
        out.append(shapely.Polygon([a, b, b + n_dir * thickness, a + n_dir * thickness]))
    return out
```

Append the overlap exception (consumed in Task 5):

```python
class SweepOverlapError(ValueError):
    """Two sweep footprints overlap (unsupported in phase 1)."""

    def __init__(self, name_a, name_b):
        super().__init__(
            f"Sweep footprints of {name_a!r} and {name_b!r} overlap; "
            "overlapping sweeps are not supported in phase 1."
        )
```

- [ ] **Step 4: Run** `uv run pytest tests/test_sweep_clip.py -q` — PASS.

- [ ] **Step 5: Commit**

```bash
git add meshwell/structured/sweep_cad.py meshwell/structured/exceptions.py tests/test_sweep_clip.py
git commit -m "feat(sweep): full-normal-extent clip via deficit-shadow subtraction"
```

---

### Task 5: imprint pass, XAO synthetics, orchestrator `sweeps=` param

**Files:**
- Modify: `meshwell/structured/sweep_cad.py` (append `sweep_imprint_pass`)
- Modify: `meshwell/occ_xao_writer.py` (`_is_purely_synthetic`: also treat names starting with `__sweep` as synthetic-only)
- Modify: `meshwell/orchestrator.py` (add `sweeps` parameter; call imprint after `structured_post_pass`)
- Test: `tests/test_sweep_imprint.py` (create)

**Interfaces:**
- Consumes: Tasks 2–4 functions; `OCCLabeledEntity(shapes, physical_name, index, keep, dim, mesh_order)` from `meshwell/cad_occ.py`; `_shape_key` from `meshwell/cad_occ.py`.
- Produces (used by Tasks 6–7 via the XAO only):
  - `sweep_imprint_pass(occ_entities: list[OCCLabeledEntity], sweeps: list[StructuredSweep], entities: list, point_tolerance: float) -> list[OCCLabeledEntity]`
  - Physical groups in the written XAO: `__sweep|<name>|<side>|<i>` (dim 2, one face each) and `__sweepsrc|<name>` (dim 1, all source curve fragments).
  - `generate_mesh(..., sweeps=[StructuredSweep(...)], ...)` accepted (list of objects or their dicts).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sweep_imprint.py
import gmsh
import pytest
import shapely

from meshwell.orchestrator import generate_mesh
from meshwell.polysurface import PolySurface
from meshwell.resolution import Graded, StructuredSweepResolutionSpec
from meshwell.structured.sweep import StructuredSweep


def _entities():
    lower = PolySurface(polygons=shapely.box(0, 0, 4, 1), physical_name="lower", mesh_order=2)
    upper = PolySurface(polygons=shapely.box(0, 1, 4, 2), physical_name="upper", mesh_order=1)
    return [lower, upper]


def test_imprint_writes_sweep_groups(tmp_path):
    xao = tmp_path / "model.xao"
    generate_mesh(
        entities=_entities(),
        sweeps=[StructuredSweep(name="qw", on="lower___upper", thickness={"upper": 0.4})],
        dim=2,
        checkpoint_cad=xao,
        output_mesh=str(tmp_path / "out.msh"),
        default_characteristic_length=0.5,
        resolution_specs={
            "qw": [StructuredSweepResolutionSpec(
                tangential=0.5, normal={"upper": Graded(h0=0.05, ratio=1.5)})],
        },
    )
    gmsh.initialize()
    try:
        gmsh.merge(str(xao))
        names = {
            gmsh.model.getPhysicalName(d, t)
            for d, t in gmsh.model.getPhysicalGroups()
        }
    finally:
        gmsh.finalize()
    assert "__sweep|qw|upper|0" in names
    assert "__sweepsrc|qw" in names
    assert "lower___upper" in names       # real interface preserved
    assert "lower" in names and "upper" in names


def test_sweep_groups_stripped_from_msh(tmp_path):
    import meshio

    out = tmp_path / "out.msh"
    generate_mesh(
        entities=_entities(),
        sweeps=[StructuredSweep(name="qw", on="lower___upper", thickness={"upper": 0.4})],
        dim=2,
        output_mesh=str(out),
        default_characteristic_length=0.5,
        resolution_specs={
            "qw": [StructuredSweepResolutionSpec(tangential=0.5, normal={"upper": 2})],
        },
    )
    m = meshio.read(out)
    assert not any(k.startswith("__sweep") for k in m.cell_sets)
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_sweep_imprint.py -q` → TypeError (unexpected kwarg `sweeps`).

- [ ] **Step 3: Implement `sweep_imprint_pass` (append to `sweep_cad.py`)**

```python
def sweep_imprint_pass(occ_entities, sweeps, entities, point_tolerance):
    """Clip + imprint every sweep; emit synthetic __sweep/__sweepsrc entities.

    A second, sweeps-only BOP fragment over all dim-2 entity shapes with the
    clipped sweep rectangles as tool faces ("highest mesh order, last").
    Sub-faces inherit their entity's physical name (shapes are replaced by
    their Modified() pieces in place); pieces inside a sweep rectangle
    additionally get a synthetic dim-2 annotator entity, and edges of those
    pieces lying on the source segment get one dim-1 __sweepsrc entity.
    """
    from OCP.BOPAlgo import BOPAlgo_Builder
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakeWire,
    )
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.gp import gp_Pnt
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_ShapeEnum
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    from meshwell.cad_occ import OCCLabeledEntity

    if not sweeps:
        return occ_entities

    region_polys = final_region_polygons(entities)

    # ---- resolve every sweep side into rectangles -----------------------
    resolved = []  # (sweep, side, rect_polygon, p0, p1, n_dir)
    all_rects = []
    for sweep in sweeps:
        p0, p1 = resolve_attachment(sweep, entities, region_polys, point_tolerance)
        for side, thick in sweep.thickness.items():
            n_dir = side_normal(p0, p1, side, sweep, region_polys, point_tolerance)
            target = (
                region_polys[side]
                if side in region_polys
                else _polyline_target_region(p0, p1, n_dir, region_polys)
            )
            intervals = clip_sweep_side(p0, p1, n_dir, thick, target, point_tolerance)
            for rect in sweep_rectangles(p0, p1, n_dir, thick, intervals):
                for other_sweep, other_side, other_rect, *_ in resolved:
                    if rect.intersection(other_rect).area > (10 * point_tolerance) ** 2:
                        raise SweepOverlapError(sweep.name, other_sweep.name)
                resolved.append((sweep, side, rect, p0, p1, n_dir))
                all_rects.append(rect)

    if not resolved:
        return occ_entities

    # ---- tool faces -----------------------------------------------------
    def _face_from_polygon(poly):
        wire = BRepBuilderAPI_MakeWire()
        coords = list(poly.exterior.coords)[:-1]
        for a, b in zip(coords, coords[1:] + coords[:1]):
            edge = BRepBuilderAPI_MakeEdge(
                gp_Pnt(a[0], a[1], 0.0), gp_Pnt(b[0], b[1], 0.0)
            ).Edge()
            wire.Add(edge)
        return BRepBuilderAPI_MakeFace(wire.Wire()).Face()

    tools = [_face_from_polygon(r) for r in all_rects]

    # ---- fragment -------------------------------------------------------
    builder = BOPAlgo_Builder()
    originals = []  # (entity_index_in_list, shape)
    for ei, ent in enumerate(occ_entities):
        if ent.dim != 2:
            continue
        for shape in ent.shapes:
            builder.AddArgument(shape)
            originals.append((ei, shape))
    for tool in tools:
        builder.AddArgument(tool)
    builder.SetFuzzyValue(point_tolerance)
    builder.Perform()

    def _pieces(shape):
        mods = builder.Modified(shape)
        if mods.Size() == 0:
            return [shape]
        return list(mods)

    # replace each dim-2 entity's shapes by their pieces
    for ei, ent in enumerate(occ_entities):
        if ent.dim != 2:
            continue
        new_shapes = []
        for shape in ent.shapes:
            for piece in _pieces(shape):
                # unwrap: keep faces only
                if piece.ShapeType() == TopAbs_ShapeEnum.TopAbs_FACE:
                    new_shapes.append(piece)
                else:
                    exp = TopExp_Explorer(piece, TopAbs_FACE)
                    while exp.More():
                        new_shapes.append(exp.Current())
                        exp.Next()
        ent.shapes = new_shapes

    # ---- synthetic annotators ------------------------------------------
    def _face_centroid(face):
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, props)
        p = props.CentreOfMass()
        return np.array([p.X(), p.Y()])

    def _edge_midpoint(edge):
        props = GProp_GProps()
        BRepGProp.LinearProperties_s(edge, props)
        p = props.CentreOfMass()
        return np.array([p.X(), p.Y()])

    next_index = max((e.index for e in occ_entities), default=-1) + 1
    out = list(occ_entities)
    per_sweep_counter: dict = {}
    for sweep, side, rect, p0, p1, n_dir in resolved:
        i = per_sweep_counter.get((sweep.name, side), 0)
        source = LineString([p0, p1])
        band_faces = []
        for ent in occ_entities:
            if ent.dim != 2 or not ent.keep:
                continue
            for face in ent.shapes:
                c = _face_centroid(face)
                if rect.contains(Point(*c)):
                    band_faces.append(face)
        for face in band_faces:
            out.append(
                OCCLabeledEntity(
                    shapes=[face],
                    physical_name=(f"__sweep|{sweep.name}|{side}|{i}",),
                    index=next_index,
                    keep=True,
                    dim=2,
                    mesh_order=None,
                )
            )
            next_index += 1
            i += 1
            # source edges of this face
            src_edges = []
            exp = TopExp_Explorer(face, TopAbs_EDGE)
            while exp.More():
                edge = TopoDS.Edge_s(exp.Current())
                if source.distance(Point(*_edge_midpoint(edge))) < 10 * point_tolerance:
                    src_edges.append(edge)
                exp.Next()
            if src_edges:
                out.append(
                    OCCLabeledEntity(
                        shapes=src_edges,
                        physical_name=(f"__sweepsrc|{sweep.name}",),
                        index=next_index,
                        keep=True,
                        dim=1,
                        mesh_order=None,
                    )
                )
                next_index += 1
        per_sweep_counter[(sweep.name, side)] = i
    return out


def _polyline_target_region(p0, p1, n_dir, region_polys):
    """For left/right sides: the region containing a probe point off the line."""
    mid = (np.asarray(p0) + np.asarray(p1)) / 2.0
    seg = float(np.linalg.norm(np.asarray(p1) - np.asarray(p0)))
    probe = Point(*(mid + n_dir * seg * 1e-4))
    for poly in region_polys.values():
        if poly.contains(probe):
            return poly
    raise SweepAttachmentNotFoundError("<polyline sweep>", f"no region contains {probe.wkt}")
```

- [ ] **Step 4: Wire the orchestrator**

In `meshwell/orchestrator.py::generate_mesh`, add parameter `sweeps: list | None = None` to the signature; after the `occ_entities = structured_post_pass(occ_entities_raw, state)` line insert:

```python
    if sweeps:
        from meshwell.structured.sweep import StructuredSweep
        from meshwell.structured.sweep_cad import sweep_imprint_pass

        sweeps = [
            StructuredSweep.from_dict(s) if isinstance(s, dict) else s for s in sweeps
        ]
        occ_entities = sweep_imprint_pass(
            occ_entities, sweeps, entities, point_tolerance
        )
```

- [ ] **Step 5: XAO writer synthetics**

In `meshwell/occ_xao_writer.py::_is_purely_synthetic`, find the return
statement checking `startswith("__cohort_")` (read the function body; the
docstring is around line 208) and extend it to also match `"__sweep"`:

```python
    return all(n.startswith(("__cohort_", "__sweep")) for n in ent.physical_name)
```

(`"__sweep"` covers both `__sweep|` and `__sweepsrc|`.)

- [ ] **Step 6: Strip synthetics from .msh output**

In `meshwell/orchestrator.py::_strip_synthetic_physical_groups`, change the
`gname.startswith("__cohort_")` check to `gname.startswith(("__cohort_", "__sweep"))`,
and in `generate_mesh` make the strip hook run whenever sweeps OR cohorts exist
(the current call is inside `_structured_pre_2d` guarded by `state.slab_meta`;
add `or sweeps` to the `has_structured` wiring so the hook is installed, and
guard `freeze_lateral_mesh` separately on `state.slab_meta`). NOTE: Task 6 adds
the same strip to the standalone `mesh()` path; keep the helper importable
(move `_strip_synthetic_physical_groups` to `meshwell/structured/sweep2d.py`
in Task 6 if orchestrator import from mesh.py would be circular — decide
there, and leave a re-export in orchestrator for backward compatibility).

- [ ] **Step 7: Run** `uv run pytest tests/test_sweep_imprint.py -q` — PASS
(the mesh generated here still uses gmsh's default triangulation of the band
faces — stamping arrives in Task 7; only group presence is asserted).
Then `uv run pytest -x -q` — no regressions.

- [ ] **Step 8: Commit**

```bash
git add meshwell/structured/sweep_cad.py meshwell/orchestrator.py meshwell/occ_xao_writer.py tests/test_sweep_imprint.py
git commit -m "feat(sweep): clip+imprint CAD pass with __sweep synthetic XAO groups"
```

---

### Task 6: mesh-stage discovery + pairing validation

**Files:**
- Create: `meshwell/structured/sweep2d.py` (discovery half)
- Modify: `meshwell/mesh.py` (`process_geometry`: discover, validate pairing, compose hook)
- Modify: `meshwell/structured/exceptions.py` (append `SweepPairingError`)
- Test: `tests/test_sweep_discovery.py` (create)

**Interfaces:**
- Consumes: loaded gmsh model containing `__sweep|…` / `__sweepsrc|…` groups (Task 5); `StructuredSweepResolutionSpec` (Task 1).
- Produces (used by Task 7):
  - `discover_sweeps() -> dict[str, dict]` — `{name: {"faces": {(side, i): face_tag}, "src_curves": set[int]}}`; empty dict when no sweep groups.
  - `validate_sweep_pairing(discovered, resolution_specs) -> dict[str, StructuredSweepResolutionSpec]` — raises `SweepPairingError` in both unpaired directions; excludes `StructuredExtrusionResolutionSpec` instances (they pair with cohorts, not sweeps).
  - `make_sweep_pre_2d_hook(discovered, specs_by_name, point_tolerance) -> Callable[[], None]` — Task 6 delivers a stub that only strips `__sweep*` groups; Task 7 fills in stamping.
  - `mesh(...)`/`generate_mesh(...)` transparently run the hook when sweep groups exist.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sweep_discovery.py
import pytest
import shapely

from meshwell.mesh import mesh
from meshwell.orchestrator import generate_mesh
from meshwell.polysurface import PolySurface
from meshwell.resolution import StructuredSweepResolutionSpec
from meshwell.structured.exceptions import SweepPairingError
from meshwell.structured.sweep import StructuredSweep


def _entities():
    lower = PolySurface(polygons=shapely.box(0, 0, 4, 1), physical_name="lower", mesh_order=2)
    upper = PolySurface(polygons=shapely.box(0, 1, 4, 2), physical_name="upper", mesh_order=1)
    return [lower, upper]


def _cad(tmp_path):
    xao = tmp_path / "model.xao"
    generate_mesh(
        entities=_entities(),
        sweeps=[StructuredSweep(name="qw", on="lower___upper", thickness={"upper": 0.4})],
        dim=2,
        checkpoint_cad=xao,
        output_mesh=str(tmp_path / "cadstep.msh"),
        default_characteristic_length=0.5,
        resolution_specs={"qw": [StructuredSweepResolutionSpec(tangential=0.5, normal={"upper": 2})]},
    )
    return xao


def test_unpaired_group_raises(tmp_path):
    xao = _cad(tmp_path)
    with pytest.raises(SweepPairingError, match="qw"):
        mesh(
            dim=2,
            input_file=xao,
            output_file=str(tmp_path / "out.msh"),
            default_characteristic_length=0.5,
            resolution_specs={},
        )


def test_unpaired_spec_raises(tmp_path):
    xao = _cad(tmp_path)
    with pytest.raises(SweepPairingError, match="ghost"):
        mesh(
            dim=2,
            input_file=xao,
            output_file=str(tmp_path / "out.msh"),
            default_characteristic_length=0.5,
            resolution_specs={
                "qw": [StructuredSweepResolutionSpec(tangential=0.5, normal={"upper": 2})],
                "ghost": [StructuredSweepResolutionSpec(tangential=0.5, normal={"upper": 2})],
            },
        )


def test_paired_separate_steps_mesh_succeeds(tmp_path):
    xao = _cad(tmp_path)
    m = mesh(
        dim=2,
        input_file=xao,
        output_file=str(tmp_path / "out.msh"),
        default_characteristic_length=0.5,
        resolution_specs={
            "qw": [StructuredSweepResolutionSpec(tangential=0.5, normal={"upper": 2})],
        },
    )
    assert m is not None
```

- [ ] **Step 2: Run to verify failure** — ImportError / no `SweepPairingError`.

- [ ] **Step 3: Implement `meshwell/structured/sweep2d.py` (discovery half)**

```python
"""Mesh-stage structured-sweep kernel: discovery + stamping (2D)."""
from __future__ import annotations

import logging
from collections import defaultdict

import gmsh
import numpy as np

from meshwell.resolution import (
    StructuredExtrusionResolutionSpec,
    StructuredSweepResolutionSpec,
    resolve_normal_offsets,
)
from meshwell.structured.exceptions import SweepPairingError

logger = logging.getLogger(__name__)

FACE_PREFIX = "__sweep|"
SRC_PREFIX = "__sweepsrc|"


def discover_sweeps() -> dict[str, dict]:
    """Scan loaded gmsh physical groups for sweep synthetics."""
    faces: dict[str, dict] = defaultdict(dict)
    srcs: dict[str, set] = defaultdict(set)
    for dim, gtag in gmsh.model.getPhysicalGroups():
        gname = gmsh.model.getPhysicalName(dim, gtag)
        if dim == 2 and gname.startswith(FACE_PREFIX):
            _, name, side, idx = gname.split("|")
            for tag in gmsh.model.getEntitiesForPhysicalGroup(dim, gtag):
                faces[name][(side, int(idx))] = int(tag)
        elif dim == 1 and gname.startswith(SRC_PREFIX):
            _, name = gname.split("|")
            srcs[name].update(
                int(t) for t in gmsh.model.getEntitiesForPhysicalGroup(dim, gtag)
            )
    return {
        name: {"faces": f, "src_curves": srcs.get(name, set())}
        for name, f in faces.items()
    }


def _is_sweep_spec(spec) -> bool:
    return isinstance(spec, StructuredSweepResolutionSpec) and not isinstance(
        spec, StructuredExtrusionResolutionSpec
    )


def validate_sweep_pairing(discovered, resolution_specs):
    """Hard-error on unpaired sweep groups or unpaired sweep specs."""
    specs_by_name = {}
    for key, specs in (resolution_specs or {}).items():
        for spec in specs:
            if _is_sweep_spec(spec):
                specs_by_name[key] = spec
    missing = set(discovered) - set(specs_by_name)
    if missing:
        raise SweepPairingError(
            f"Sweep group(s) {sorted(missing)} present in CAD but no "
            "StructuredSweepResolutionSpec provided under that name."
        )
    orphans = set(specs_by_name) - set(discovered)
    if orphans:
        raise SweepPairingError(
            f"StructuredSweepResolutionSpec(s) {sorted(orphans)} have no "
            "matching __sweep group in the loaded CAD."
        )
    return specs_by_name


def strip_sweep_groups() -> None:
    """Remove __sweep*/__cohort_* bookkeeping groups before .msh write."""
    import contextlib

    to_remove, names = [], []
    for dim, gtag in gmsh.model.getPhysicalGroups():
        gname = gmsh.model.getPhysicalName(dim, gtag)
        if gname.startswith(("__sweep", "__cohort_")):
            to_remove.append((dim, gtag))
            names.append(gname)
    if to_remove:
        gmsh.model.removePhysicalGroups(to_remove)
        for gname in names:
            with contextlib.suppress(Exception):
                gmsh.model.removePhysicalName(gname)


def make_sweep_pre_2d_hook(discovered, specs_by_name, point_tolerance):
    """Return the pre-generate(2) stamping hook (stamping added in Task 7)."""

    def _hook() -> None:
        _stamp_all(discovered, specs_by_name, point_tolerance)
        strip_sweep_groups()

    return _hook


def _stamp_all(discovered, specs_by_name, point_tolerance) -> None:
    """Stamp every sweep face. Implemented in the stamping task."""
    # Task 7 replaces this body; the discovery task lands a no-op so the
    # pairing/discovery wiring is testable independently.
    logger.info("sweep stamping: %d sweep(s) discovered", len(discovered))
```

Append the exception:

```python
class SweepPairingError(ValueError):
    """__sweep groups and StructuredSweepResolutionSpecs must pair 1:1."""
```

- [ ] **Step 4: Wire into `meshwell/mesh.py::process_geometry`**

Locate where `process_geometry` has recovered labels and is about to call
`process_mesh` (read the body; it ends by calling `self.process_mesh(...)` with
the hook kwargs). Insert immediately before that call:

```python
        from meshwell.structured.sweep2d import (
            discover_sweeps,
            make_sweep_pre_2d_hook,
            validate_sweep_pairing,
        )

        discovered = discover_sweeps()
        if discovered:
            specs_by_name = validate_sweep_pairing(discovered, resolution_specs)
            sweep_hook = make_sweep_pre_2d_hook(
                discovered, specs_by_name, self.point_tolerance or 1e-3
            )
            user_pre_2d = pre_2d_hook

            def pre_2d_hook() -> None:  # noqa: F811 - deliberate rebind
                if user_pre_2d is not None:
                    user_pre_2d()
                sweep_hook()
```

(The orchestrator path also flows through `process_geometry`, so
`generate_mesh` needs no additional mesh-side sweep code — this is the
discovery-based single implementation from the spec.)

- [ ] **Step 5: Run** `uv run pytest tests/test_sweep_discovery.py -q` — PASS. Then `uv run pytest tests/test_sweep_imprint.py -q` — the Task 5 test must now ALSO pass with the hook active (specs were already provided there).

- [ ] **Step 6: Commit**

```bash
git add meshwell/structured/sweep2d.py meshwell/mesh.py meshwell/structured/exceptions.py tests/test_sweep_discovery.py
git commit -m "feat(sweep): discovery-based mesh stage with 1:1 pairing validation"
```

---### Task 7: 2D stamping kernel

**Files:**
- Modify: `meshwell/structured/sweep2d.py` (replace `_stamp_all` stub; add helpers)
- Modify: `meshwell/structured/exceptions.py` (append `SweepSeamMismatchError`, `SweepSplitCoordinateError`)
- Test: `tests/test_sweep_stamping.py` (create)

**Interfaces:**
- Consumes: `discovered` dict + `specs_by_name` (Task 6), `resolve_normal_offsets` (Task 1).
- Produces: band faces meshed with exact tensor-product nodes and right-triangle pairs (or quads); `Mesh.MeshOnlyEmpty=1` set so gmsh only meshes the rest.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sweep_stamping.py
import numpy as np
import pytest
import shapely

from meshwell.orchestrator import generate_mesh
from meshwell.polysurface import PolySurface
from meshwell.resolution import StructuredSweepResolutionSpec
from meshwell.structured.sweep import StructuredSweep


def _run(tmp_path, element_type="triangle", tangential=1.0, normal=2, thickness=0.4):
    return generate_mesh(
        entities=[
            PolySurface(polygons=shapely.box(0, 0, 4, 1), physical_name="lower", mesh_order=2),
            PolySurface(polygons=shapely.box(0, 1, 4, 2), physical_name="upper", mesh_order=1),
        ],
        sweeps=[StructuredSweep(name="qw", on="lower___upper", thickness={"upper": thickness})],
        dim=2,
        output_mesh=str(tmp_path / "out.msh"),
        default_characteristic_length=0.5,
        resolution_specs={
            "qw": [StructuredSweepResolutionSpec(
                tangential=tangential, normal={"upper": normal}, element_type=element_type)],
        },
    )


def _band_nodes(m, thickness=0.4):
    pts = m.points[:, :2]
    return pts[(pts[:, 1] >= 1.0 - 1e-9) & (pts[:, 1] <= 1.0 + thickness + 1e-9)]


def test_band_nodes_are_exact_tensor_grid(tmp_path):
    m = _run(tmp_path, tangential=1.0, normal=2, thickness=0.4)
    band = _band_nodes(m)
    xs = np.unique(np.round(band[:, 0], 9))
    ys = np.unique(np.round(band[:, 1], 9))
    np.testing.assert_allclose(xs, [0.0, 1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(ys, [1.0, 1.2, 1.4])
    # every grid point exists exactly once
    assert len(band) == len(xs) * len(ys)


def test_band_cells_are_right_triangles(tmp_path):
    m = _run(tmp_path)
    tri = next(cb.data for cb in m.cells if cb.type == "triangle")
    pts = m.points[:, :2]
    # collect triangles fully inside the band
    for conn in tri:
        p = pts[conn]
        if p[:, 1].min() >= 1.0 - 1e-9 and p[:, 1].max() <= 1.4 + 1e-9:
            # right triangle: one vertex has a 90 deg angle
            v = [p[(i + 1) % 3] - p[i] for i in range(3)]
            dots = [abs(np.dot(v[i], -v[(i - 1) % 3])) for i in range(3)]
            assert min(dots) == pytest.approx(0.0, abs=1e-9)


def test_quad_variant(tmp_path):
    m = _run(tmp_path, element_type="quad")
    quads = sum(cb.data.shape[0] for cb in m.cells if cb.type == "quad")
    assert quads == 4 * 2  # 4 tangential cells x 2 normal layers


def test_interface_groups_survive(tmp_path):
    m = _run(tmp_path)
    assert "lower___upper" in m.cell_sets
    assert "lower" in m.cell_sets and "upper" in m.cell_sets
    assert not any(k.startswith("__sweep") for k in m.cell_sets)


def test_conformal_no_duplicate_nodes(tmp_path):
    m = _run(tmp_path)
    pts = np.round(m.points[:, :2], 9)
    assert len(np.unique(pts, axis=0)) == len(pts)
```

- [ ] **Step 2: Run to verify failure** — grid assertions fail (band still unstructured-triangulated by gmsh).

- [ ] **Step 3: Implement stamping (replace `_stamp_all` in `sweep2d.py`)**

Algorithm per sweep (all faces of all sides share one source frame):

```python
def _stamp_all(discovered, specs_by_name, point_tolerance) -> None:
    gmsh.option.setNumber("Mesh.MeshOnlyEmpty", 1)
    stamped_curves: set[int] = set()
    for name, groups in discovered.items():
        spec = specs_by_name[name]
        frame = _sweep_frame(groups["src_curves"])
        for (side, _idx), face_tag in sorted(groups["faces"].items()):
            _stamp_face(
                face_tag, side, spec, frame, groups, stamped_curves, point_tolerance
            )
    # 1D pass for all remaining (unstamped, non-sweep) curves
    gmsh.model.mesh.generate(1)


def _sweep_frame(src_curve_tags):
    """(origin, t_hat, n_hat) from the union of source curve endpoints.

    t_hat points from the min-projection endpoint to the max; n_hat is
    +90 deg (left of travel). Sign of n per face resolved in _stamp_face.
    """
    pts = []
    for ctag in src_curve_tags:
        for _dim, ptag in gmsh.model.getBoundary([(1, ctag)], oriented=False):
            xyz = gmsh.model.getValue(0, abs(ptag), [])
            pts.append(np.array(xyz[:2]))
    pts = np.array(pts)
    # direction: principal axis of the point cloud
    d = pts.max(axis=0) - pts.min(axis=0)
    t_hat = d / np.linalg.norm(d)
    proj = pts @ t_hat
    origin = pts[int(np.argmin(proj))]
    n_hat = np.array([-t_hat[1], t_hat[0]])
    return origin, t_hat, n_hat
```

Face stamping — this is the heart; implement exactly this structure:

```python
def _stamp_face(face_tag, side, spec, frame, groups, stamped_curves, tol):
    origin, t_hat, n_hat = frame
    curves = [
        (abs(t),)
        for _d, t in gmsh.model.getBoundary([(2, face_tag)], oriented=False, recursive=False)
    ]
    curves = sorted({c[0] for c in curves})

    def _tn(xy):
        v = np.asarray(xy[:2]) - origin
        return float(v @ t_hat), float(v @ n_hat)

    def _curve_endpoints(ctag):
        out = []
        for _d, ptag in gmsh.model.getBoundary([(1, ctag)], oriented=False):
            out.append(np.array(gmsh.model.getValue(0, abs(ptag), [])[:2]))
        return out

    # face corners in (t, n)
    all_tn = [ _tn(p) for c in curves for p in _curve_endpoints(c) ]
    t0f, t1f = min(t for t, _ in all_tn), max(t for t, _ in all_tn)
    n_vals = sorted({round(n, 12) for _, n in all_tn})
    n_lo, n_hi = n_vals[0], n_vals[-1]
    # source seam is at the n value closest to 0 among src curves of this face
    src_here = [c for c in curves if c in groups["src_curves"]]
    if src_here:
        n_src = _tn(_curve_endpoints(src_here[0])[0])[1]
    else:
        n_src = n_lo if abs(n_lo) < abs(n_hi) else n_hi
    n_far = n_hi if n_src == n_lo else n_lo
    thickness = abs(n_far - n_src)
    sign = 1.0 if n_far > n_src else -1.0

    offsets = resolve_normal_offsets(
        spec.normal[side] if isinstance(spec.normal, dict) else spec.normal,
        thickness,
        tol,
    )
    ts = _tangential_coords(spec.tangential, t0f, t1f, groups, face_tag, tol)

    # grid of physical points: shape (len(ts), len(offsets))
    grid_xy = np.array(
        [
            [origin + t * t_hat + (n_src + sign * off) * n_hat for off in offsets]
            for t in ts
        ]
    )

    # 1) stamp boundary curves (once each, shared with neighbours)
    for ctag in curves:
        if ctag in stamped_curves:
            continue
        stamped_curves.add(ctag)
        e0, e1 = _curve_endpoints(ctag)
        pts = _grid_points_on_segment(grid_xy, e0, e1, tol)
        _stamp_curve(ctag, pts, tol)

    # 2) interior nodes
    node_tag = gmsh.model.mesh.getMaxNodeTag() + 1
    tag_grid = np.zeros(grid_xy.shape[:2], dtype=np.int64)
    interior_tags, interior_coords = [], []
    boundary_lookup = _boundary_node_lookup(curves, tol)
    for it in range(len(ts)):
        for ik in range(len(offsets)):
            key = _round_key(grid_xy[it, ik], tol)
            if key in boundary_lookup:
                tag_grid[it, ik] = boundary_lookup[key]
            else:
                tag_grid[it, ik] = node_tag
                interior_tags.append(node_tag)
                interior_coords += [grid_xy[it, ik, 0], grid_xy[it, ik, 1], 0.0]
                node_tag += 1
    if interior_tags:
        gmsh.model.mesh.addNodes(2, face_tag, interior_tags, interior_coords)

    # 3) elements
    tris, quads = [], []
    for it in range(len(ts) - 1):
        for ik in range(len(offsets) - 1):
            q = [
                tag_grid[it, ik], tag_grid[it + 1, ik],
                tag_grid[it + 1, ik + 1], tag_grid[it, ik + 1],
            ]
            # enforce CCW in xy using the physical coords
            a, b, c = grid_xy[it, ik], grid_xy[it + 1, ik], grid_xy[it + 1, ik + 1]
            if np.cross(b - a, c - b) < 0:
                q = q[::-1]
            if spec.element_type == "quad":
                quads += q
            else:
                tris += [q[0], q[1], q[2], q[0], q[2], q[3]]
    if quads:
        gmsh.model.mesh.addElementsByType(face_tag, 3, [], quads)
    if tris:
        gmsh.model.mesh.addElementsByType(face_tag, 2, [], tris)
```

Helpers (complete implementations):

```python
def _round_key(xy, tol):
    q = max(tol, 1e-12)
    return (round(float(xy[0]) / q), round(float(xy[1]) / q))


def _boundary_node_lookup(curves, tol):
    lookup = {}
    for ctag in curves:
        tags, coords, _ = gmsh.model.mesh.getNodes(1, ctag, includeBoundary=True)
        for i, t in enumerate(tags):
            lookup[_round_key(coords[3 * i : 3 * i + 2], tol)] = int(t)
    return lookup


def _grid_points_on_segment(grid_xy, e0, e1, tol):
    """Ordered grid points lying on segment e0-e1 (inclusive)."""
    seg = np.asarray(e1) - np.asarray(e0)
    length = np.linalg.norm(seg)
    d = seg / length
    flat = grid_xy.reshape(-1, 2)
    on = []
    for p in flat:
        v = p - e0
        s = float(v @ d)
        if -tol <= s <= length + tol and abs(float(np.cross(d, v))) <= 10 * tol:
            on.append((s, p))
    on.sort(key=lambda x: x[0])
    return [p for _s, p in on]


def _ensure_point_node(ptag):
    tags, _coords, _ = gmsh.model.mesh.getNodes(0, ptag)
    if len(tags):
        return int(tags[0])
    xyz = gmsh.model.getValue(0, ptag, [])
    new = gmsh.model.mesh.getMaxNodeTag() + 1
    gmsh.model.mesh.addNodes(0, ptag, [new], list(xyz))
    return new


def _stamp_curve(ctag, pts, tol):
    """Stamp ordered nodes+line elements on a curve. pts includes endpoints."""
    if len(pts) < 2:
        return
    end_nodes = {}
    for _d, ptag in gmsh.model.getBoundary([(1, ctag)], oriented=False):
        xyz = gmsh.model.getValue(0, abs(ptag), [])
        end_nodes[_round_key(xyz[:2], tol)] = _ensure_point_node(abs(ptag))
    seq = []
    interior_tags, interior_coords = [], []
    next_tag = gmsh.model.mesh.getMaxNodeTag() + 1
    for p in pts:
        key = _round_key(p, tol)
        if key in end_nodes:
            seq.append(end_nodes[key])
        else:
            seq.append(next_tag)
            interior_tags.append(next_tag)
            interior_coords += [float(p[0]), float(p[1]), 0.0]
            next_tag += 1
    if interior_tags:
        gmsh.model.mesh.addNodes(1, ctag, interior_tags, interior_coords)
    conn = []
    for a, b in zip(seq[:-1], seq[1:]):
        conn += [int(a), int(b)]
    gmsh.model.mesh.addElementsByType(ctag, 1, [], conn)


def _tangential_coords(tangential, t0f, t1f, groups, face_tag, tol):
    """Tangential grid for one face; membership rule for interior splits.

    Global sweep extent [T0, T1] = min/max t over all faces of the sweep
    (compute once per sweep and cache on `groups["_extent"]`). Face ends
    strictly inside (T0, T1) are BOP splits -> must be grid members
    (SweepSplitCoordinateError otherwise). Ends at the global extent are
    clip/attachment ends -> auto-inserted.
    """
    from meshwell.structured.exceptions import SweepSplitCoordinateError

    T0, T1 = groups["_extent"]

    if tangential is None:
        raise SweepPairingError("StructuredSweepResolutionSpec.tangential is required in 2D")
    if isinstance(tangential, (int, float)):
        h = float(tangential)
        k0 = int(np.floor((t0f - T0) / h + 0.5))
        candidates = [T0 + k * h for k in range(k0, int((T1 - T0) / h) + 2)]
        inside = [t for t in candidates if t0f - tol <= t <= t1f + tol]
    else:
        arr = [T0 + t for t in np.asarray(tangential, dtype=float)]
        inside = [t for t in arr if t0f - tol <= t <= t1f + tol]
    for end, is_global in ((t0f, abs(t0f - T0) <= tol), (t1f, abs(t1f - T1) <= tol)):
        if not any(abs(t - end) <= tol for t in inside):
            if is_global:
                inside.append(end)  # clip end: auto-insert
            else:
                raise SweepSplitCoordinateError(end, face_tag)
    ts = sorted(set(round(t, 12) for t in inside))
    return [t for t in ts if t0f - tol <= t <= t1f + tol]
```

In `_stamp_all`, before the per-face loop, compute the extent cache:

```python
        all_tn_ends = []
        for (_side, _i), ftag in groups["faces"].items():
            for _d, ct in gmsh.model.getBoundary([(2, ftag)], oriented=False, recursive=False):
                for _dd, pt in gmsh.model.getBoundary([(1, abs(ct))], oriented=False):
                    xy = np.array(gmsh.model.getValue(0, abs(pt), [])[:2])
                    all_tn_ends.append(float((xy - frame[0]) @ frame[1]))
        groups["_extent"] = (min(all_tn_ends), max(all_tn_ends))
```

Append the two exceptions:

```python
class SweepSeamMismatchError(ValueError):
    """Adjacent sweeps disagree on shared-seam node coordinates."""


class SweepSplitCoordinateError(ValueError):
    """A BOP split point on a sweep edge is not a tangential grid member."""

    def __init__(self, t, face_tag):
        super().__init__(
            f"Sweep face {face_tag}: fragment boundary at tangential coordinate "
            f"{t!r} is not a member of the tangential grid. Add this coordinate "
            "to the explicit tangential array (BOP splits mark user geometry the "
            "grid must align with)."
        )
```

Seam-mismatch detection comes free from `_stamp_curve`'s `stamped_curves`
guard plus a check: when a curve is already stamped, verify the would-be
points match the existing nodes (`_boundary_node_lookup` on that curve; if
any `_round_key` of the new points is missing → raise
`SweepSeamMismatchError(f"curve {ctag}")`). Add that check in `_stamp_face`
step 1's `continue` branch.

- [ ] **Step 4: Run** `uv run pytest tests/test_sweep_stamping.py -q` — PASS. Iterate here: this task has the most gmsh-API friction (node/element bookkeeping); the tests define done.

- [ ] **Step 5: Run full suite** `uv run pytest -x -q` — no regressions.

- [ ] **Step 6: Commit**

```bash
git add meshwell/structured/sweep2d.py meshwell/structured/exceptions.py tests/test_sweep_stamping.py
git commit -m "feat(sweep): 2D tensor-grid stamping kernel (tri/quad) with frozen seams"
```

---

### Task 8: integration tests (ridge split, PolyLine two-sided, clip fallback, separate steps)

**Files:**
- Test: `tests/test_sweep_integration.py` (create)
- Modify: `meshwell/structured/sweep2d.py` / `sweep_cad.py` only as needed to make these pass.

**Interfaces:** consumes everything above; produces no new API.

- [ ] **Step 1: Write the tests**

```python
# tests/test_sweep_integration.py
import numpy as np
import pytest
import shapely

from meshwell.mesh import mesh
from meshwell.orchestrator import generate_mesh
from meshwell.polyline import PolyLine
from meshwell.polysurface import PolySurface
from meshwell.resolution import Graded, StructuredSweepResolutionSpec
from meshwell.structured.exceptions import SweepSplitCoordinateError
from meshwell.structured.sweep import StructuredSweep


def test_ridge_split_edges(tmp_path):
    """A ridge on top splits the band's top seam into 3 curves; explicit
    tangential array containing the ridge corners (x=1.5, 2.5) succeeds."""
    entities = [
        PolySurface(polygons=shapely.box(0, 0, 4, 1), physical_name="layer", mesh_order=3),
        PolySurface(polygons=shapely.box(1.5, 1, 2.5, 2), physical_name="ridge", mesh_order=1),
        PolySurface(polygons=shapely.box(0, 1, 4, 2), physical_name="clad", mesh_order=2),
    ]
    m = generate_mesh(
        entities=entities,
        sweeps=[StructuredSweep(name="top", on="layer___clad", thickness={"layer": 0.3})],
        dim=2,
        output_mesh=str(tmp_path / "ridge.msh"),
        default_characteristic_length=0.5,
        resolution_specs={
            "top": [StructuredSweepResolutionSpec(
                tangential=[0.0, 0.75, 1.5, 2.0, 2.5, 3.25, 4.0],
                normal={"layer": 3})],
        },
    )
    band = m.points[(m.points[:, 1] >= 0.7 - 1e-9) & (m.points[:, 1] <= 1.0 + 1e-9), 0]
    assert {1.5, 2.5} <= set(np.round(np.unique(band), 9))


def test_ridge_split_without_member_coordinate_raises(tmp_path):
    entities = [
        PolySurface(polygons=shapely.box(0, 0, 4, 1), physical_name="layer", mesh_order=3),
        PolySurface(polygons=shapely.box(1.5, 1, 2.5, 2), physical_name="ridge", mesh_order=1),
        PolySurface(polygons=shapely.box(0, 1, 4, 2), physical_name="clad", mesh_order=2),
    ]
    with pytest.raises(SweepSplitCoordinateError):
        generate_mesh(
            entities=entities,
            sweeps=[StructuredSweep(name="top", on="layer___clad", thickness={"layer": 0.3})],
            dim=2,
            output_mesh=str(tmp_path / "ridge2.msh"),
            default_characteristic_length=0.5,
            resolution_specs={
                "top": [StructuredSweepResolutionSpec(tangential=1.0, normal={"layer": 3})],
            },
        )


def test_polyline_two_sided_different_grading(tmp_path):
    entities = [
        PolySurface(polygons=shapely.box(0, 0, 4, 2), physical_name="bulk", mesh_order=1),
        PolyLine(linestrings=shapely.LineString([(0.0, 1.0), (4.0, 1.0)]), physical_name="jn"),
    ]
    m = generate_mesh(
        entities=entities,
        sweeps=[StructuredSweep(name="j", on="jn", thickness={"left": 0.4, "right": 0.2})],
        dim=2,
        output_mesh=str(tmp_path / "jn.msh"),
        default_characteristic_length=0.5,
        resolution_specs={
            "j": [StructuredSweepResolutionSpec(
                tangential=1.0,
                normal={"left": Graded(h0=0.05, ratio=2.0), "right": 2})],
        },
    )
    ys = np.round(np.unique(m.points[:, 1]), 9)
    # right side (below, travel +x -> right is -y): uniform 2 layers of 0.1
    assert {0.8, 0.9, 1.0} <= set(ys)
    # left side (above): first cell exactly h0
    assert 1.05 in set(ys)


def test_clip_corner_falls_back_to_unstructured(tmp_path):
    """Band would exit its region near x in [3,4] (notched region):
    that shadow has no band; the mesh still generates and is conformal."""
    entities = [
        PolySurface(polygons=shapely.box(0, 0, 4, 1), physical_name="lower", mesh_order=2),
        PolySurface(
            polygons=shapely.box(0, 1, 4, 2).difference(shapely.box(3, 1, 4, 1.2)),
            physical_name="upper", mesh_order=1),
    ]
    m = generate_mesh(
        entities=entities,
        sweeps=[StructuredSweep(name="qw", on="lower___upper", thickness={"upper": 0.4})],
        dim=2,
        output_mesh=str(tmp_path / "clip.msh"),
        default_characteristic_length=0.3,
        resolution_specs={
            "qw": [StructuredSweepResolutionSpec(tangential=0.5, normal={"upper": 2})],
        },
    )
    pts = np.round(m.points[:, :2], 9)
    assert len(np.unique(pts, axis=0)) == len(pts)  # conformal, no dups


def test_separate_steps_equivalent(tmp_path):
    ents = [
        PolySurface(polygons=shapely.box(0, 0, 4, 1), physical_name="lower", mesh_order=2),
        PolySurface(polygons=shapely.box(0, 1, 4, 2), physical_name="upper", mesh_order=1),
    ]
    specs = {"qw": [StructuredSweepResolutionSpec(tangential=1.0, normal={"upper": 2})]}
    sweeps = [StructuredSweep(name="qw", on="lower___upper", thickness={"upper": 0.4})]
    m1 = generate_mesh(
        entities=ents, sweeps=sweeps, dim=2, checkpoint_cad=tmp_path / "m.xao",
        output_mesh=str(tmp_path / "one.msh"),
        default_characteristic_length=0.5, resolution_specs=specs,
    )
    m2 = mesh(
        dim=2, input_file=tmp_path / "m.xao", output_file=str(tmp_path / "two.msh"),
        default_characteristic_length=0.5, resolution_specs=specs,
    )
    b1 = np.unique(np.round(m1.points[(m1.points[:, 1] >= 1 - 1e-9) & (m1.points[:, 1] <= 1.4 + 1e-9), :2], 9), axis=0)
    b2 = np.unique(np.round(m2.points[(m2.points[:, 1] >= 1 - 1e-9) & (m2.points[:, 1] <= 1.4 + 1e-9), :2], 9), axis=0)
    np.testing.assert_array_equal(b1, b2)
```

- [ ] **Step 2: Run** `uv run pytest tests/test_sweep_integration.py -q`; fix implementation bugs these surface (expected: seam classification and clip/split-end distinctions). The tests define done.

- [ ] **Step 3: Full suite** `uv run pytest -x -q` — green.

- [ ] **Step 4: Commit**

```bash
git add tests/test_sweep_integration.py meshwell/structured/
git commit -m "test(sweep): integration coverage - ridge splits, two-sided polyline, clip fallback, separate steps"
```

---

### Task 9: docs notebook + laser plan text updates

**Files:**
- Create: `docs/24_structured_sweeps.py` (percent-format notebook like `docs/23_structured.py`)
- Modify: `../plans/02a-meshwell-structured-bands.md` (laser repo — NOT a git repo at that level; just edit the file)

**Interfaces:** none produced; documentation only.

- [ ] **Step 1: Write `docs/24_structured_sweeps.py`**

Mirror the structure of `docs/23_structured.py` (markdown cells + runnable
code). Content: (a) one-paragraph motivation (anisotropic admissible bands,
link to 23 for the 3D sibling); (b) the two-layer worked example from
`tests/test_sweep_stamping.py::_run` with `plot2D` visualization; (c) a
section showing `Graded` vs explicit arrays vs `int`; (d) the quad variant;
(e) the separate CAD→mesh workflow from
`tests/test_sweep_integration.py::test_separate_steps_equivalent`; (f) a
short "errors you will meet" section listing `SweepPairingError`,
`SweepSplitCoordinateError`, `SweepCurvedSourceError` with one-line causes.
Use real runnable code taken from the tests (adjust paths to tempfile or
local output names as the other docs do).

- [ ] **Step 2: Verify the notebook executes**

Run: `uv run python docs/24_structured_sweeps.py` — Expected: exits 0, writes its example mesh.

- [ ] **Step 3: Update the laser plan file**

In `../plans/02a-meshwell-structured-bands.md` apply the three revisions
listed at the end of the design spec (growth-ratio DSL now supported; band
is a `StructuredSweep` declaration + spec, not an entity; quad output now a
2D option). Keep the physics/phase content untouched; add a line under
"Phase 1" pointing to the meshwell spec and plan documents.

- [ ] **Step 4: Commit**

```bash
git add docs/24_structured_sweeps.py
git commit -m "docs(sweep): structured sweeps notebook"
```

---

## Self-review notes (already applied)

- Spec coverage: API (Tasks 1–2), clip+imprint (Tasks 4–5), discovery +
  pairing (Task 6), stamping incl. split-edge/clip-end rules (Task 7),
  quad option (Tasks 1, 7), separate-steps contract (Tasks 5, 6, 8),
  seam-mismatch check (Task 7), plan-02a updates (Task 9). The
  admissibility validator is explicitly phase 2 (not planned here).
- The `StructuredExtrusionResolutionSpec` alias keeps `n_layers` so
  `meshwell/structured/wedge.py` is untouched; wedge regression run in
  Task 1 Step 4 guards this.
- Known soft spots the worker should expect to iterate on with the tests:
  gmsh node/element tag bookkeeping in Task 7 (`addNodes` signatures,
  `getNodes(includeBoundary=True)` behavior), OCC `Modified()` piece
  unwrapping in Task 5, and `BOPAlgo_Builder` API details (mirror usage
  in `meshwell/cad_occ.py`).
