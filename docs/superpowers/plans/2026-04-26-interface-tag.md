# InterfaceTag Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `InterfaceTag` entity to `cad_gmsh` that names interfaces between buffered polygon entities by snapping to the boundary that wins the cut cascade — without introducing new geometry of its own.

**Architecture:** A new `meshwell/interface_tag.py` module exposes `InterfaceTag(linestrings, zmin, zmax, ...)`. `cad_gmsh.process_entities` is restructured into three passes: (A) buffer polygon entities in shapely, (B) resolve each `InterfaceTag` against the buffered polygons (walking targets in `mesh_order` so the winner's boundary is taken), (C) the existing instantiate/cut/fragment/tag loop. `InterfaceTag.instanciate` extrudes the resolved shapely linestrings vertically into 2D surfaces that the rest of the pipeline handles unmodified.

**Tech Stack:** Python 3.11+, shapely, gmsh (Python bindings), pytest.

**Spec:** [docs/superpowers/specs/2026-04-26-interface-tag-design.md](../specs/2026-04-26-interface-tag-design.md)

---

## File map

- **Create:** `meshwell/interface_tag.py` — `InterfaceTag` class + `_flatten_to_linestrings` helper.
- **Modify:** `meshwell/cad_gmsh.py` — add `perturbation` property; restructure `process_entities` into three passes; recognize `InterfaceTag` in pass B.
- **Create:** `tests/test_interface_tag.py` — all behaviour tests.

---

## Task 1: Introduce `perturbation` property on `CAD_GMSH`

A pure refactor. No behaviour change. Makes the buffer scale a named, single-source-of-truth attribute that pass A and `InterfaceTag.resolve` can both reference.

**Files:**
- Modify: `meshwell/cad_gmsh.py` (`CAD_GMSH` class and `process_entities`)

- [ ] **Step 1.1: Add the `perturbation` property**

In `meshwell/cad_gmsh.py`, after the `__init__` method of `CAD_GMSH` (before `_instantiate_entity`), add:

```python
    @property
    def perturbation(self) -> float:
        """Outward shapely buffer applied to polygon entities before
        sequential cuts. Same value used as the default snap distance
        for :class:`meshwell.interface_tag.InterfaceTag`. Currently
        ``2 * point_tolerance``; do not change without coordinated
        validation on complex scenes."""
        return 2 * self.point_tolerance
```

- [ ] **Step 1.2: Replace inline `2 * self.point_tolerance` with `self.perturbation`**

In `meshwell/cad_gmsh.py`, locate the two occurrences inside `process_entities` (in the buffer block around lines 428-436):

```python
                if isinstance(ent.polygons, list):
                    ent.polygons = [
                        p.buffer(2 * self.point_tolerance, join_style=2).intersection(
                            global_bbox
                        )
                        for p in ent.polygons
                    ]
                else:
                    ent.polygons = ent.polygons.buffer(
                        2 * self.point_tolerance, join_style=2
                    ).intersection(global_bbox)
```

Replace `2 * self.point_tolerance` with `self.perturbation` in both spots:

```python
                if isinstance(ent.polygons, list):
                    ent.polygons = [
                        p.buffer(self.perturbation, join_style=2).intersection(
                            global_bbox
                        )
                        for p in ent.polygons
                    ]
                else:
                    ent.polygons = ent.polygons.buffer(
                        self.perturbation, join_style=2
                    ).intersection(global_bbox)
```

- [ ] **Step 1.3: Run the full `cad_gmsh` test suite to confirm no regression**

Run: `pytest tests/test_cad_gmsh.py -v`
Expected: same pass/fail outcome as before this task. (At time of writing, `test_cad_gmsh_mesh_order_lower_wins_in_overlap` is already failing on this branch due to buffer-induced area inflation; that failure is unrelated and must remain identical.)

- [ ] **Step 1.4: Commit**

```bash
git add meshwell/cad_gmsh.py
git commit -m "refactor(cad_gmsh): expose buffer scale as CAD_GMSH.perturbation property"
```

---

## Task 2: `InterfaceTag` skeleton

Create the module and the class with constructor only. `resolve` and `instanciate` raise `NotImplementedError`. This locks in the public API and lets us write tests against it before any implementation exists.

**Files:**
- Create: `meshwell/interface_tag.py`

- [ ] **Step 2.1: Create the module**

Create `meshwell/interface_tag.py` with the following content:

```python
"""InterfaceTag: name an existing interface between buffered polygon entities.

Unlike :class:`meshwell.polysurface.PolySurface` or a `gmsh_entity` plane,
:class:`InterfaceTag` does not introduce new geometry into the model. It
declares a *nominal* trace where the user expects an interface to lie, and
at fragment time it resolves itself onto the boundary of the polygon entity
that wins the cad_gmsh cut cascade in that region. This avoids the sliver
slabs that result from co-positioning a `gmsh_entity` plane with an
asymmetrically-buffered ``PolyPrism`` edge.

Use ``gmsh_entity`` (or ``PolySurface``) when you need a NEW internal cut.
Use ``InterfaceTag`` when you only want to name an interface that already
exists between two polygon entities.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import shapely
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
)
from shapely.ops import unary_union

import gmsh
from meshwell.geometry_entity import GeometryEntity

if TYPE_CHECKING:
    from OCP.TopoDS import TopoDS_Shape


def _flatten_to_linestrings(geoms) -> list[LineString]:
    """Collect plain LineStrings from any nesting of LineString /
    MultiLineString / GeometryCollection. Empty geometries and Points
    (degenerate intersections) are dropped."""
    out: list[LineString] = []
    for g in geoms:
        if g.is_empty:
            continue
        if isinstance(g, LineString):
            out.append(g)
        elif isinstance(g, MultiLineString):
            out.extend(list(g.geoms))
        elif isinstance(g, GeometryCollection):
            out.extend(_flatten_to_linestrings(list(g.geoms)))
        # Points and Polygons are silently dropped (degenerate or N/A).
    return out


class InterfaceTag(GeometryEntity):
    """Snap-to-boundary interface tag for ``cad_gmsh``.

    Attributes:
        linestrings: list of shapely ``LineString`` giving the nominal XY
            trace of the interface.
        zmin: lower z-extent of the resulting vertical 2D surface.
        zmax: upper z-extent of the resulting vertical 2D surface.
        physical_name: name of the physical group this entity will own.
        targets: explicit list of polygon-entity physical names to snap
            to. ``None`` means "any polygon-bearing entity in the scene".
        snap_distance: how far the nominal trace is allowed to be from a
            target boundary. ``None`` means "inherit the cad processor's
            ``perturbation``".
        mesh_order: ownership priority on overlapping pieces.
        mesh_bool: whether to keep the resulting surfaces in the mesh.
    """

    def __init__(
        self,
        linestrings: LineString | list[LineString] | MultiLineString,
        zmin: float,
        zmax: float,
        physical_name: str | tuple[str, ...] | None = None,
        targets: list[str] | None = None,
        snap_distance: float | None = None,
        mesh_order: float | None = None,
        mesh_bool: bool = True,
        point_tolerance: float = 1e-3,
    ):
        super().__init__(point_tolerance=point_tolerance)

        # Normalize linestrings to a flat list[LineString].
        if isinstance(linestrings, list):
            normalized: list[LineString] = []
            for item in linestrings:
                if isinstance(item, MultiLineString):
                    normalized.extend(list(item.geoms))
                else:
                    normalized.append(item)
            self.linestrings = normalized
        elif isinstance(linestrings, MultiLineString):
            self.linestrings = list(linestrings.geoms)
        else:
            self.linestrings = [linestrings]

        # Snap input to the tolerance grid for determinism.
        if point_tolerance > 0:
            self.linestrings = [
                shapely.set_precision(
                    ls, grid_size=point_tolerance, mode="pointwise"
                )
                for ls in self.linestrings
            ]

        self.zmin = float(zmin)
        self.zmax = float(zmax)
        if isinstance(physical_name, str):
            self.physical_name = (physical_name,)
        else:
            self.physical_name = physical_name
        self.targets = list(targets) if targets is not None else None
        self.snap_distance = snap_distance
        self.mesh_order = mesh_order
        self.mesh_bool = mesh_bool
        self.dimension = 2

        # Populated by :meth:`resolve` before :meth:`instanciate` runs.
        self.resolved_linestrings: list[LineString] = []

    def resolve(self, polygon_ents: dict[str, Any], default_snap: float) -> None:
        """Compute the snapped trace from buffered polygon entities. Must be
        called before :meth:`instanciate`. See spec for algorithm."""
        raise NotImplementedError("InterfaceTag.resolve not yet implemented")

    def instanciate(
        self,
        cad_model: Any | None = None,  # noqa: ARG002
    ) -> list[tuple[int, int]]:
        """Build vertical 2D surfaces from ``self.resolved_linestrings``."""
        raise NotImplementedError("InterfaceTag.instanciate not yet implemented")

    def instanciate_occ(self) -> "TopoDS_Shape":
        raise NotImplementedError(
            "InterfaceTag is gmsh-only for v1; use the cad_gmsh backend."
        )
```

- [ ] **Step 2.2: Verify the module imports**

Run: `python -c "from meshwell.interface_tag import InterfaceTag, _flatten_to_linestrings; print('ok')"`
Expected: prints `ok`. (No actual InterfaceTag construction.)

- [ ] **Step 2.3: Commit**

```bash
git add meshwell/interface_tag.py
git commit -m "feat(interface_tag): add InterfaceTag skeleton with NotImplementedError stubs"
```

---

## Task 3: Implement `resolve()` (TDD)

Write a focused unit test that exercises only the shapely side (no gmsh), then implement.

**Files:**
- Modify: `meshwell/interface_tag.py`
- Create: `tests/test_interface_tag.py`

- [ ] **Step 3.1: Create the test file with the resolve unit test**

Create `tests/test_interface_tag.py`:

```python
"""Tests for :class:`meshwell.interface_tag.InterfaceTag`."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import shapely
from shapely.geometry import LineString

from meshwell.interface_tag import InterfaceTag


@dataclass
class _FakePolyEntity:
    """Minimal stand-in for a polygon-bearing entity used to unit-test
    :meth:`InterfaceTag.resolve` without spinning up gmsh."""

    polygons: object  # shapely Polygon | list[Polygon]
    mesh_order: float | None = None


def test_resolve_picks_winning_boundary_in_abutting_prisms():
    """A=mesh_order 1 (winner), B=mesh_order 2. Both buffered outward by
    pert. Nominal trace at x=5. Expect a single resolved segment on A's
    right boundary at x = 5 + pert. B's left boundary at x = 5 - pert is
    inside A's claimed body and must NOT appear."""
    pert = 1e-3
    a_poly = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]).buffer(
        pert, join_style=2
    )
    b_poly = shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)]).buffer(
        pert, join_style=2
    )

    tag = InterfaceTag(
        linestrings=LineString([(5, 0), (5, 5)]),
        zmin=0,
        zmax=1,
        physical_name="iface",
        targets=None,
    )
    tag.resolve(
        polygon_ents={
            "A": _FakePolyEntity(polygons=a_poly, mesh_order=1),
            "B": _FakePolyEntity(polygons=b_poly, mesh_order=2),
        },
        default_snap=2 * 1e-3,
    )

    assert len(tag.resolved_linestrings) == 1
    seg = tag.resolved_linestrings[0]
    # The resolved segment must lie at x = 5 + pert (A's buffered right edge).
    xs = {round(x, 6) for x, _ in seg.coords}
    assert xs == {round(5 + pert, 6)}, (xs, seg)


def test_resolve_warns_when_no_match():
    """A nominal trace far from any target produces no segments and warns."""
    pert = 1e-3
    a_poly = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]).buffer(
        pert, join_style=2
    )

    tag = InterfaceTag(
        linestrings=LineString([(100, 0), (100, 5)]),
        zmin=0,
        zmax=1,
        physical_name="iface",
        targets=None,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tag.resolve(
            polygon_ents={"A": _FakePolyEntity(polygons=a_poly, mesh_order=1)},
            default_snap=2 * 1e-3,
        )

    assert tag.resolved_linestrings == []
    assert any("resolved to no segments" in str(w.message) for w in caught), caught


def test_resolve_picks_up_hole_boundary():
    """A target polygon with an interior hole. InterfaceTag traces the hole
    and must snap to the inner ring."""
    pert = 1e-3
    outer = shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    hole = shapely.Polygon([(4, 4), (6, 4), (6, 6), (4, 6)])
    donut = outer.difference(hole).buffer(pert, join_style=2)

    tag = InterfaceTag(
        linestrings=LineString([(4, 4), (6, 4), (6, 6), (4, 6), (4, 4)]),
        zmin=0,
        zmax=1,
        physical_name="hole_iface",
        targets=None,
    )
    tag.resolve(
        polygon_ents={"O": _FakePolyEntity(polygons=donut, mesh_order=1)},
        default_snap=2 * 1e-3,
    )

    assert len(tag.resolved_linestrings) >= 1
    total_len = sum(ls.length for ls in tag.resolved_linestrings)
    # The hole's perimeter is 8; expect resolved length ~8 (within buffer slop).
    assert 7.5 < total_len < 8.5, total_len
```

- [ ] **Step 3.2: Run the test and confirm it fails**

Run: `pytest tests/test_interface_tag.py -v`
Expected: 3 failures, all with `NotImplementedError: InterfaceTag.resolve not yet implemented`.

- [ ] **Step 3.3: Implement `resolve()` in `meshwell/interface_tag.py`**

Replace the stub `resolve` method with:

```python
    def resolve(self, polygon_ents: dict[str, Any], default_snap: float) -> None:
        """Compute the snapped trace from buffered polygon entities.

        The cad_gmsh cut cascade processes targets lowest-``mesh_order``
        first; the winner keeps its full buffered shape. We mirror that
        ordering here: we walk targets in ascending ``mesh_order`` and
        each one *claims* its body's spatial region, so subsequent (higher
        mesh_order) targets only see the strip outside that claim. The
        boundary that ends up tagged is therefore exactly the boundary
        that survives the cut.
        """
        snap = self.snap_distance if self.snap_distance is not None else default_snap

        if self.targets is not None:
            # Skip names not present in the scene rather than KeyError-ing.
            targets = [polygon_ents[n] for n in self.targets if n in polygon_ents]
        else:
            targets = list(polygon_ents.values())

        targets.sort(
            key=lambda t: t.mesh_order
            if t.mesh_order is not None
            else float("inf")
        )

        nominal_strip = unary_union(
            [ls.buffer(snap, join_style=2) for ls in self.linestrings]
        )

        snapped: list = []
        remaining = nominal_strip
        for tgt in targets:
            polys = (
                tgt.polygons
                if isinstance(tgt.polygons, list)
                else [tgt.polygons]
            )
            for poly in polys:
                hit = poly.boundary.intersection(remaining)
                if not hit.is_empty:
                    snapped.append(hit)
            for poly in polys:
                remaining = remaining.difference(poly)

        self.resolved_linestrings = _flatten_to_linestrings(snapped)
        if not self.resolved_linestrings:
            warnings.warn(
                f"InterfaceTag {self.physical_name} resolved to no segments",
                stacklevel=2,
            )
```

- [ ] **Step 3.4: Run the resolve tests to confirm they pass**

Run: `pytest tests/test_interface_tag.py -v`
Expected: 3 passed.

- [ ] **Step 3.5: Commit**

```bash
git add meshwell/interface_tag.py tests/test_interface_tag.py
git commit -m "feat(interface_tag): implement mesh_order-aware resolve()"
```

---

## Task 4: Implement `instanciate()`

`instanciate` runs after `resolve` has populated `self.resolved_linestrings`. It builds vertical 2D surfaces by adding gmsh points at `zmin`, joining them into a wire per linestring, and extruding by `(0, 0, zmax - zmin)`.

**Files:**
- Modify: `meshwell/interface_tag.py`
- Modify: `tests/test_interface_tag.py`

- [ ] **Step 4.1: Add a gmsh-driven instanciate test**

Append to `tests/test_interface_tag.py`:

```python
import gmsh

from meshwell.model import ModelManager


def test_instanciate_builds_2d_vertical_surfaces():
    """Given two pre-set resolved_linestrings, instanciate returns 2D
    dimtags whose bounding box matches z = [zmin, zmax]."""
    mm = ModelManager(filename="test_interface_tag_instanciate")
    try:
        mm.ensure_initialized("test_interface_tag_instanciate")

        tag = InterfaceTag(
            linestrings=LineString([(0, 0), (5, 0)]),  # placeholder
            zmin=0.0,
            zmax=2.0,
            physical_name="iface",
        )
        # Bypass resolve(): inject the trace directly.
        tag.resolved_linestrings = [
            LineString([(0, 0), (5, 0)]),
            LineString([(0, 5), (5, 5)]),
        ]

        dimtags = tag.instanciate()
        assert len(dimtags) == 2, dimtags
        assert all(d == 2 for d, _ in dimtags), dimtags

        gmsh.model.occ.synchronize()
        for d, t in dimtags:
            xmin, ymin, zmin_b, xmax, ymax, zmax_b = (
                gmsh.model.getBoundingBox(d, t)
            )
            assert abs(zmin_b - 0.0) < 1e-9, (zmin_b, t)
            assert abs(zmax_b - 2.0) < 1e-9, (zmax_b, t)
    finally:
        mm.finalize()
```

- [ ] **Step 4.2: Run the test and confirm it fails**

Run: `pytest tests/test_interface_tag.py::test_instanciate_builds_2d_vertical_surfaces -v`
Expected: FAIL with `NotImplementedError: InterfaceTag.instanciate not yet implemented`.

- [ ] **Step 4.3: Implement `instanciate()`**

Replace the stub `instanciate` method in `meshwell/interface_tag.py` with:

```python
    def instanciate(
        self,
        cad_model: Any | None = None,  # noqa: ARG002
    ) -> list[tuple[int, int]]:
        """Build vertical 2D surfaces from ``self.resolved_linestrings``.

        Each linestring is laid out at ``z = zmin`` as a chain of gmsh
        OCC lines, wrapped in a wire, and extruded by ``zmax - zmin`` to
        produce the 2D vertical surface. The 2D surface dimtags are
        collected and returned; 1D end-caps from the extrude are
        discarded.
        """
        dimtags: list[tuple[int, int]] = []
        dz = self.zmax - self.zmin
        if dz == 0.0:
            return dimtags

        for ls in self.resolved_linestrings:
            coords = list(ls.coords)
            if len(coords) < 2:
                continue

            point_tags = [
                self._add_point_with_tolerance(x, y, self.zmin)
                for x, y in coords
            ]
            line_tags: list[int] = []
            for p1, p2 in zip(point_tags[:-1], point_tags[1:], strict=False):
                lt = self._add_line_with_cache(p1, p2)
                if lt != 0:
                    line_tags.append(lt)
            if not line_tags:
                continue

            wire_tag = gmsh.model.occ.addWire(line_tags)
            extruded = gmsh.model.occ.extrude(
                [(1, wire_tag)], 0.0, 0.0, dz
            )
            for d, t in extruded:
                if d == 2:
                    dimtags.append((d, t))

        gmsh.model.occ.synchronize()
        return dimtags
```

- [ ] **Step 4.4: Run the instanciate test to confirm it passes**

Run: `pytest tests/test_interface_tag.py::test_instanciate_builds_2d_vertical_surfaces -v`
Expected: PASS.

- [ ] **Step 4.5: Run all interface_tag tests**

Run: `pytest tests/test_interface_tag.py -v`
Expected: 4 passed.

- [ ] **Step 4.6: Commit**

```bash
git add meshwell/interface_tag.py tests/test_interface_tag.py
git commit -m "feat(interface_tag): implement instanciate via wire extrusion"
```

---

## Task 5: Wire `InterfaceTag` into `cad_gmsh.process_entities`

Restructure `process_entities` into three explicit passes so InterfaceTag's resolve runs after polygons are buffered but before instantiation.

**Files:**
- Modify: `meshwell/cad_gmsh.py` (`process_entities`)

- [ ] **Step 5.1: Add the InterfaceTag import at the top of `cad_gmsh.py`**

Add (with the other from-imports near the top of the file):

```python
from meshwell.interface_tag import InterfaceTag
```

- [ ] **Step 5.2: Restructure `process_entities` into three passes**

Locate the current body of `process_entities` (after the bbox computation block but before the existing loop). Replace the structure so it reads as below. The key changes: (a) buffer pass A runs OVER all polygon entities first as a separate loop; (b) new pass B resolves InterfaceTags; (c) the existing instantiate+cut loop becomes pass C and now skips `InterfaceTag` entities because they were already given resolved geometry (their `.polygons` attribute does not exist, so the buffer block is skipped; their `instanciate` consumes `resolved_linestrings`).

The full replacement for the body of `process_entities` after the `ensure_initialized` call is:

```python
        # ----- Pass A: buffer all polygon-bearing entities (shapely only) -----
        from shapely.geometry import box

        xmin, ymin, xmax, ymax = (
            float("inf"),
            float("inf"),
            float("-inf"),
            float("-inf"),
        )
        for ent in entities_list:
            if hasattr(ent, "polygons"):
                polys = (
                    ent.polygons if isinstance(ent.polygons, list) else [ent.polygons]
                )
                for p in polys:
                    b = p.bounds
                    xmin = min(xmin, b[0])
                    ymin = min(ymin, b[1])
                    xmax = max(xmax, b[2])
                    ymax = max(ymax, b[3])
        global_bbox = box(xmin, ymin, xmax, ymax)
        print(f"Global bounding box for clipping: {global_bbox.bounds}")

        for ent in entities_list:
            if not hasattr(ent, "polygons"):
                continue
            if isinstance(ent.polygons, list):
                ent.polygons = [
                    p.buffer(self.perturbation, join_style=2).intersection(
                        global_bbox
                    )
                    for p in ent.polygons
                ]
            else:
                ent.polygons = ent.polygons.buffer(
                    self.perturbation, join_style=2
                ).intersection(global_bbox)

        # ----- Pass B: resolve each InterfaceTag against the buffered polygons -----
        polygon_ents: dict[str, Any] = {}
        for ent in entities_list:
            if not hasattr(ent, "polygons"):
                continue
            name = ent.physical_name
            if isinstance(name, tuple):
                name = name[0]
            polygon_ents[name] = ent

        for ent in entities_list:
            if isinstance(ent, InterfaceTag):
                ent.resolve(polygon_ents, default_snap=self.perturbation)

        # ----- Pass C: existing mesh_order sort + instantiate + sequential cut -----
        indexed_entities = [(ent, i) for i, ent in enumerate(entities_list)]
        indexed_entities.sort(
            key=lambda x: (
                x[0].mesh_order if x[0].mesh_order is not None else float("inf"),
                x[1],
            )
        )

        instantiated_entities = []

        for ent, orig_idx in indexed_entities:
            # Instantiate (polygon entities have already been buffered above;
            # InterfaceTag entities will read resolved_linestrings).
            labeled_ent = self._instantiate_entity(orig_idx, ent)

            # Cut with all previously instantiated entities of the same dim.
            if labeled_ent.dimtags:
                all_tool_dimtags = []
                for prev_ent in instantiated_entities:
                    if prev_ent.dim == labeled_ent.dim and prev_ent.dimtags:
                        all_tool_dimtags.extend(prev_ent.dimtags)

                if all_tool_dimtags:
                    try:
                        out_dimtags, _ = gmsh.model.occ.cut(
                            labeled_ent.dimtags,
                            all_tool_dimtags,
                            removeObject=True,
                            removeTool=False,
                        )
                        self.model_manager.sync_model()
                        labeled_ent.dimtags = out_dimtags
                    except Exception as e:
                        print(f"Warning: Cut failed for entity {orig_idx}: {e}")

            instantiated_entities.append(labeled_ent)

        # Final global fragment + tag + cleanup.
        labeled = self._fragment_all(instantiated_entities, progress_bars=progress_bars)
        self._tag_entities(labeled, interface_delimiter, boundary_delimiter)
        self._remove_keep_false_top_dim(labeled)
        self.model_manager.sync_model()

        return labeled
```

- [ ] **Step 5.3: Run the existing cad_gmsh suite to confirm no regression**

Run: `pytest tests/test_cad_gmsh.py -v`
Expected: same pass/fail outcome as after Task 1.3 (the pre-existing
`test_cad_gmsh_mesh_order_lower_wins_in_overlap` failure is unchanged; all
other tests pass).

- [ ] **Step 5.4: Commit**

```bash
git add meshwell/cad_gmsh.py
git commit -m "feat(cad_gmsh): three-pass process_entities, integrate InterfaceTag"
```

---

## Task 6: End-to-end test — winning boundary in 2 abutting prisms

**Files:**
- Modify: `tests/test_interface_tag.py`

- [ ] **Step 6.1: Add the end-to-end test**

Append to `tests/test_interface_tag.py`:

```python
from meshwell.cad_gmsh import cad_gmsh, strip_suffix
from meshwell.polyprism import PolyPrism


def _physical_names() -> list[tuple[int, str]]:
    return [
        (d, gmsh.model.getPhysicalName(d, t))
        for d, t in gmsh.model.getPhysicalGroups()
    ]


def test_e2e_interface_tag_resolves_to_winning_boundary():
    """Two abutting prisms (A wins, B loses) plus a single InterfaceTag at
    the nominal interface x=5. After cad_gmsh: exactly one face is tagged
    `iface`, and A is not internally split."""
    A = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    B = shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)])
    buffers = {0.0: 0.0, 1.0: 0.0}
    try:
        labeled, mm = cad_gmsh(
            [
                PolyPrism(polygons=A, buffers=buffers, physical_name="A", mesh_order=1),
                PolyPrism(polygons=B, buffers=buffers, physical_name="B", mesh_order=2),
                InterfaceTag(
                    linestrings=LineString([(5, 0), (5, 5)]),
                    zmin=0.0,
                    zmax=1.0,
                    physical_name="iface",
                    mesh_order=3,
                ),
            ]
        )

        names = {n for _, n in _physical_names()}
        assert {"A", "B", "iface", "A___B"} <= names

        # The iface physical group must exist at dim 2 with at least one face.
        iface_pg = next(
            t for d, t in gmsh.model.getPhysicalGroups(dim=2)
            if gmsh.model.getPhysicalName(d, t) == "iface"
        )
        iface_faces = gmsh.model.getEntitiesForPhysicalGroup(2, iface_pg)
        assert len(iface_faces) >= 1, iface_faces

        # A must remain a single 3D piece (no internal cut by InterfaceTag).
        a_ent = next(
            e for e in labeled if strip_suffix(e.physical_name[0]) == "A"
        )
        assert sum(1 for d, _ in a_ent.dimtags if d == 3) == 1, a_ent.dimtags
    finally:
        mm.finalize()
```

- [ ] **Step 6.2: Run and confirm pass**

Run: `pytest tests/test_interface_tag.py::test_e2e_interface_tag_resolves_to_winning_boundary -v`
Expected: PASS.

- [ ] **Step 6.3: Commit**

```bash
git add tests/test_interface_tag.py
git commit -m "test(interface_tag): e2e winning-boundary in abutting prisms"
```

---

## Task 7: End-to-end test — `targets=None` with three abutting prisms

**Files:**
- Modify: `tests/test_interface_tag.py`

- [ ] **Step 7.1: Add the test**

Append to `tests/test_interface_tag.py`:

```python
def test_e2e_targets_none_picks_winners_only():
    """Three abutting prisms A/B/C (mesh_orders 1/2/3) with interfaces at
    x=2 and x=5. A single InterfaceTag with linestring spanning both,
    targets=None. Expect both interfaces tagged (one face each on the
    winner side), and no spurious internal cuts in any prism."""
    A = shapely.Polygon([(0, 0), (2, 0), (2, 5), (0, 5)])
    B = shapely.Polygon([(2, 0), (5, 0), (5, 5), (2, 5)])
    C = shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)])
    buffers = {0.0: 0.0, 1.0: 0.0}
    try:
        labeled, mm = cad_gmsh(
            [
                PolyPrism(polygons=A, buffers=buffers, physical_name="A", mesh_order=1),
                PolyPrism(polygons=B, buffers=buffers, physical_name="B", mesh_order=2),
                PolyPrism(polygons=C, buffers=buffers, physical_name="C", mesh_order=3),
                InterfaceTag(
                    linestrings=LineString([(2, 2.5), (5, 2.5)]),
                    zmin=0.0,
                    zmax=1.0,
                    physical_name="iface",
                    mesh_order=4,
                ),
            ]
        )

        # Each prism stays as exactly one 3D piece.
        for nm in ("A", "B", "C"):
            ent = next(
                e for e in labeled if strip_suffix(e.physical_name[0]) == nm
            )
            n_3d = sum(1 for d, _ in ent.dimtags if d == 3)
            assert n_3d == 1, (nm, ent.dimtags)

        # iface tagged at dim 2 with at least 2 faces (one per real interface).
        iface_pg = next(
            t for d, t in gmsh.model.getPhysicalGroups(dim=2)
            if gmsh.model.getPhysicalName(d, t) == "iface"
        )
        iface_faces = gmsh.model.getEntitiesForPhysicalGroup(2, iface_pg)
        assert len(iface_faces) >= 2, iface_faces
    finally:
        mm.finalize()
```

- [ ] **Step 7.2: Run and confirm pass**

Run: `pytest tests/test_interface_tag.py::test_e2e_targets_none_picks_winners_only -v`
Expected: PASS.

- [ ] **Step 7.3: Commit**

```bash
git add tests/test_interface_tag.py
git commit -m "test(interface_tag): e2e targets=None across three prisms"
```

---

## Task 8: End-to-end test — explicit `targets` subset

**Files:**
- Modify: `tests/test_interface_tag.py`

- [ ] **Step 8.1: Add the test**

Append to `tests/test_interface_tag.py`:

```python
def test_e2e_targets_explicit_subset():
    """Same 3-prism scene, but targets=['A','B']. Only the A/B interface is
    tagged; the B/C interface is not. The C entity must remain unsplit."""
    A = shapely.Polygon([(0, 0), (2, 0), (2, 5), (0, 5)])
    B = shapely.Polygon([(2, 0), (5, 0), (5, 5), (2, 5)])
    C = shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)])
    buffers = {0.0: 0.0, 1.0: 0.0}
    try:
        labeled, mm = cad_gmsh(
            [
                PolyPrism(polygons=A, buffers=buffers, physical_name="A", mesh_order=1),
                PolyPrism(polygons=B, buffers=buffers, physical_name="B", mesh_order=2),
                PolyPrism(polygons=C, buffers=buffers, physical_name="C", mesh_order=3),
                InterfaceTag(
                    linestrings=LineString([(2, 2.5), (5, 2.5)]),
                    zmin=0.0,
                    zmax=1.0,
                    physical_name="iface_AB",
                    targets=["A", "B"],
                    mesh_order=4,
                ),
            ]
        )

        iface_pg = next(
            t for d, t in gmsh.model.getPhysicalGroups(dim=2)
            if gmsh.model.getPhysicalName(d, t) == "iface_AB"
        )
        iface_faces = gmsh.model.getEntitiesForPhysicalGroup(2, iface_pg)

        # The tagged interface must be at x ≈ 2 + pert (A's right edge),
        # not at x ≈ 5. Inspect each tagged face's bounding box.
        for d, t in [(2, ft) for ft in iface_faces]:
            xmin, _, _, xmax, _, _ = gmsh.model.getBoundingBox(d, t)
            x_center = 0.5 * (xmin + xmax)
            assert 1.5 < x_center < 3.5, (
                f"unexpected face at x_center={x_center}, expected near x=2"
            )
    finally:
        mm.finalize()
```

- [ ] **Step 8.2: Run and confirm pass**

Run: `pytest tests/test_interface_tag.py::test_e2e_targets_explicit_subset -v`
Expected: PASS.

- [ ] **Step 8.3: Commit**

```bash
git add tests/test_interface_tag.py
git commit -m "test(interface_tag): e2e explicit targets subset"
```

---

## Task 9: End-to-end test — donut / hole boundary

**Files:**
- Modify: `tests/test_interface_tag.py`

- [ ] **Step 9.1: Add the test**

Append to `tests/test_interface_tag.py`:

```python
def test_e2e_picks_up_hole_boundary():
    """Outer prism with a square hole, inner prism filling the hole.
    InterfaceTag traces the hole. After cad_gmsh, the hole's perimeter
    must be tagged with iface, and at least one face exists in the group."""
    outer = shapely.Polygon(
        shell=[(0, 0), (10, 0), (10, 10), (0, 10)],
        holes=[[(4, 4), (6, 4), (6, 6), (4, 6)]],
    )
    inner = shapely.Polygon([(4, 4), (6, 4), (6, 6), (4, 6)])
    buffers = {0.0: 0.0, 1.0: 0.0}
    try:
        _, mm = cad_gmsh(
            [
                PolyPrism(
                    polygons=outer, buffers=buffers,
                    physical_name="O", mesh_order=2,
                ),
                PolyPrism(
                    polygons=inner, buffers=buffers,
                    physical_name="I", mesh_order=1,
                ),
                InterfaceTag(
                    linestrings=LineString(
                        [(4, 4), (6, 4), (6, 6), (4, 6), (4, 4)]
                    ),
                    zmin=0.0,
                    zmax=1.0,
                    physical_name="iface",
                    mesh_order=3,
                ),
            ]
        )

        iface_pg = next(
            t for d, t in gmsh.model.getPhysicalGroups(dim=2)
            if gmsh.model.getPhysicalName(d, t) == "iface"
        )
        iface_faces = gmsh.model.getEntitiesForPhysicalGroup(2, iface_pg)
        assert len(iface_faces) >= 1, iface_faces
    finally:
        mm.finalize()
```

- [ ] **Step 9.2: Run and confirm pass**

Run: `pytest tests/test_interface_tag.py::test_e2e_picks_up_hole_boundary -v`
Expected: PASS.

- [ ] **Step 9.3: Commit**

```bash
git add tests/test_interface_tag.py
git commit -m "test(interface_tag): e2e tag picks up hole boundary"
```

---

## Task 10: End-to-end test — no match emits warning, no crash

**Files:**
- Modify: `tests/test_interface_tag.py`

- [ ] **Step 10.1: Add the test**

Append to `tests/test_interface_tag.py`:

```python
from meshwell.mesh import mesh as mesh_func


def test_e2e_no_match_warns_and_skips():
    """A tag whose linestring is far from all targets emits a warning,
    creates no physical group, and downstream meshing still succeeds."""
    A = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    buffers = {0.0: 0.0, 1.0: 0.0}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, mm = cad_gmsh(
                [
                    PolyPrism(
                        polygons=A, buffers=buffers,
                        physical_name="A", mesh_order=1,
                    ),
                    InterfaceTag(
                        linestrings=LineString([(100, 0), (100, 5)]),
                        zmin=0.0,
                        zmax=1.0,
                        physical_name="iface_far",
                        mesh_order=2,
                    ),
                ]
            )
        assert any(
            "resolved to no segments" in str(w.message) for w in caught
        ), [str(w.message) for w in caught]

        names = {n for _, n in _physical_names()}
        assert "iface_far" not in names

        m = mesh_func(dim=3, model=mm, default_characteristic_length=2.0, verbosity=0)
        assert any(c.type == "tetra" and len(c.data) for c in m.cells)
    finally:
        mm.finalize()
```

- [ ] **Step 10.2: Run and confirm pass**

Run: `pytest tests/test_interface_tag.py::test_e2e_no_match_warns_and_skips -v`
Expected: PASS.

- [ ] **Step 10.3: Commit**

```bash
git add tests/test_interface_tag.py
git commit -m "test(interface_tag): e2e no-match warns + skips cleanly"
```

---

## Task 11: End-to-end test — `gmsh_entity` targets are silently skipped

**Files:**
- Modify: `tests/test_interface_tag.py`

- [ ] **Step 11.1: Add the test**

Append to `tests/test_interface_tag.py`:

```python
from meshwell.occ_entity import OCC_entity
from tests.test_occ_helpers import _occ_box


def test_e2e_gmsh_entity_targets_skipped():
    """A scene mixing a PolyPrism and an OCC_entity. InterfaceTag with
    targets=None resolves only against the PolyPrism (the OCC_entity has
    no .polygons and is silently skipped)."""
    A_poly = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    buffers = {0.0: 0.0, 1.0: 0.0}
    try:
        _, mm = cad_gmsh(
            [
                PolyPrism(
                    polygons=A_poly, buffers=buffers,
                    physical_name="A", mesh_order=1,
                ),
                OCC_entity(
                    occ_function=_occ_box(x=6, y=0, z=0, dx=2, dy=5, dz=1),
                    physical_name="occ_box",
                    mesh_order=2,
                    dimension=3,
                ),
                InterfaceTag(
                    linestrings=LineString([(5, 0), (5, 5)]),
                    zmin=0.0,
                    zmax=1.0,
                    physical_name="iface",
                    mesh_order=3,
                ),
            ]
        )

        iface_pg = next(
            t for d, t in gmsh.model.getPhysicalGroups(dim=2)
            if gmsh.model.getPhysicalName(d, t) == "iface"
        )
        iface_faces = gmsh.model.getEntitiesForPhysicalGroup(2, iface_pg)
        # There must be exactly one tagged face — A's right boundary.
        # The OCC box at x=6 must NOT contribute.
        assert len(iface_faces) == 1, iface_faces
        d, t = 2, iface_faces[0]
        xmin, _, _, xmax, _, _ = gmsh.model.getBoundingBox(d, t)
        x_center = 0.5 * (xmin + xmax)
        assert 4.5 < x_center < 5.5, x_center
    finally:
        mm.finalize()
```

- [ ] **Step 11.2: Run and confirm pass**

Run: `pytest tests/test_interface_tag.py::test_e2e_gmsh_entity_targets_skipped -v`
Expected: PASS.

- [ ] **Step 11.3: Commit**

```bash
git add tests/test_interface_tag.py
git commit -m "test(interface_tag): e2e gmsh_entity targets are skipped"
```

---

## Task 12: End-to-end test — z-extent of the resulting surface

**Files:**
- Modify: `tests/test_interface_tag.py`

- [ ] **Step 12.1: Add the test**

Append to `tests/test_interface_tag.py`:

```python
def test_e2e_extrudes_to_correct_z_range():
    """A single PolyPrism z=[0,1] with an InterfaceTag zmin=0, zmax=1
    along its right boundary. Expect the resulting tagged face's bbox to
    span z = [0, 1] (within tolerance)."""
    A = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    buffers = {0.0: 0.0, 1.0: 0.0}
    try:
        _, mm = cad_gmsh(
            [
                PolyPrism(
                    polygons=A, buffers=buffers,
                    physical_name="A", mesh_order=1,
                ),
                InterfaceTag(
                    linestrings=LineString([(5, 0), (5, 5)]),
                    zmin=0.0,
                    zmax=1.0,
                    physical_name="iface",
                    mesh_order=2,
                ),
            ]
        )

        iface_pg = next(
            t for d, t in gmsh.model.getPhysicalGroups(dim=2)
            if gmsh.model.getPhysicalName(d, t) == "iface"
        )
        iface_faces = gmsh.model.getEntitiesForPhysicalGroup(2, iface_pg)
        assert len(iface_faces) >= 1, iface_faces
        for t in iface_faces:
            _, _, zmin_b, _, _, zmax_b = gmsh.model.getBoundingBox(2, t)
            assert abs(zmin_b - 0.0) < 1e-6, zmin_b
            assert abs(zmax_b - 1.0) < 1e-6, zmax_b
    finally:
        mm.finalize()
```

- [ ] **Step 12.2: Run and confirm pass**

Run: `pytest tests/test_interface_tag.py::test_e2e_extrudes_to_correct_z_range -v`
Expected: PASS.

- [ ] **Step 12.3: Commit**

```bash
git add tests/test_interface_tag.py
git commit -m "test(interface_tag): e2e extrusion z-range correctness"
```

---

## Task 13: Final regression sweep

**Files:** none — verification only.

- [ ] **Step 13.1: Run the full meshwell test suite**

Run: `pytest tests/test_interface_tag.py tests/test_cad_gmsh.py -v`
Expected: all `test_interface_tag.py` tests pass; `test_cad_gmsh.py` retains the same pass/fail outcome documented at Task 1.3 (`test_cad_gmsh_mesh_order_lower_wins_in_overlap` remains the sole pre-existing failure).

- [ ] **Step 13.2: If any previously-passing test now fails, stop and investigate**

Do NOT mark this step complete until either (a) every previously-passing test still passes, or (b) the failure is traced to a real issue with this work and a follow-up task is added to the plan to fix it.
