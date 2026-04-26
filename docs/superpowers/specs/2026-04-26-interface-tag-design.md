# InterfaceTag entity for `cad_gmsh`

**Status:** approved spec, ready for implementation plan
**Date:** 2026-04-26
**Backend scope:** `cad_gmsh` only (gmsh-native CAD pipeline)

## Problem

`cad_gmsh.process_entities` buffers every polygon-bearing entity outward by
`2 * point_tolerance` before fragmenting. This is the only reliable fix we have
for BOPAlgo failing to merge almost-coincident faces on complex geometries
(simple test cases never reproduced the failure; only large scenes did). The
buffer is therefore load-bearing and stays.

The buffer is **asymmetric**: only entities that expose a `.polygons` attribute
get inflated. `gmsh_entity` shapes (e.g. a vertical plane built directly via
the gmsh API) do not. A user who places a vertical plane to coincide *exactly*
with a `PolyPrism` edge for tagging purposes hits this mismatch:

- The `PolyPrism` right face moves from `x = 5` to `x = 5 + perturbation`.
- The `gmsh_entity` plane stays at `x = 5`.
- The plane is now *inside* the prism by `perturbation`.
- The final fragment splits the prism along the plane → an unwanted sliver
  slab plus an extra physical group.

There is no way to keep both the bridging-via-buffer behaviour and the
exact-coincidence-of-mixed-entities behaviour with the current model.

## Solution

Add a new entity `InterfaceTag` whose **only** job is to name an interface
that already exists between two (or more) buffered polygon entities. It does
not introduce new geometry. Instead, at fragment time it resolves its own
geometry by looking at where its nominal trace falls along the *post-buffer*
boundaries of its targets. The resolved trace is then extruded vertically and
fed into the existing fragment + tag pipeline.

This is **additive**: `gmsh_entity` is still the right tool for cuts that
introduce new internal features. `InterfaceTag` is the right tool for naming
interfaces between polygon entities. Documentation must say so.

The buffer is **not** modified by this work. Tolerance shrinkage is a separate
investigation.

## API

New module `meshwell/interface_tag.py`. New class `InterfaceTag`, subclassing
`GeometryEntity` (for naming and transform conventions).

```python
class InterfaceTag(GeometryEntity):
    def __init__(
        self,
        linestrings: LineString | list[LineString] | MultiLineString,
        zmin: float,
        zmax: float,
        physical_name: str | tuple[str, ...] | None = None,
        targets: list[str] | None = None,        # None = any polygon entity
        snap_distance: float | None = None,      # None = inherit perturbation
        mesh_order: float | None = None,
        mesh_bool: bool = True,
        point_tolerance: float = 1e-3,
    ): ...
```

Semantics:

- `linestrings`: the user's nominal XY trace of the interface.
- `zmin`, `zmax`: vertical extent of the resulting 2D surface in 3D.
- `targets`: names of polygon entities to snap to. `None` = any entity in the
  scene with a `.polygons` attribute.
- `snap_distance`: how far the nominal trace is allowed to be from a target
  boundary. Defaults to the cad processor's perturbation
  (`2 * point_tolerance` today).
- `physical_name` / `mesh_order` / `mesh_bool`: same role as on every other
  entity.

## Pipeline integration

A small naming change makes the buffer scale explicit. `CAD_GMSH` gains a
read-only `self.perturbation` property that returns `2 * self.point_tolerance`
today. The existing inline `2 * self.point_tolerance` in `process_entities`
is replaced by `self.perturbation`. No public-API change. This name is what
the InterfaceTag pipeline references.

`cad_gmsh.CAD_GMSH.process_entities` is restructured into three passes:

```python
def process_entities(self, entities_list, ...):
    # Pass A: buffer all polygon-bearing entities (shapely only, no gmsh).
    for ent in entities_list:
        if hasattr(ent, "polygons"):
            ent.polygons = _buffer(ent.polygons, self.perturbation)

    # Pass B: resolve each InterfaceTag against the buffered polygons.
    polygon_ents = {
        strip_suffix(e.physical_name[0]): e
        for e in entities_list if hasattr(e, "polygons")
    }
    for ent in entities_list:
        if isinstance(ent, InterfaceTag):
            ent.resolve(polygon_ents, default_snap=self.perturbation)

    # Pass C: existing mesh_order sort, per-entity instantiate + sequential
    # cut, final global fragment, _tag_entities, _remove_keep_false_top_dim.
    # InterfaceTag's resolved 2D vertical surfaces participate in this pass
    # exactly like any other lower-dim entity.
```

The cut/fragment/tag steps in Pass C are unchanged. `InterfaceTag` is
lower-dim relative to the top-dim PolyPrisms, so it falls through the
existing "lower-dim entities are always tagged" branch in `_tag_entities`.

## `InterfaceTag.resolve()` algorithm

The resolve step replicates the cad_gmsh cut cascade in shapely directly,
then takes boundaries of the cut polygons. Iterate targets in ascending
`mesh_order` (winners first). Each target's polygon has all higher-priority
targets' polygons subtracted from it, mirroring the cad_gmsh cut step. The
resulting cut-polygon boundaries are exactly what would survive in gmsh.
Intersecting those boundaries with the user's nominal strip (built with
flat caps so the strip does not extend past the linestring endpoints)
gives the snapped trace. Coincident contributions from neighbouring
targets — both contribute the same shared face — collapse via a final
`unary_union`.

```python
def resolve(self, polygon_ents, default_snap):
    snap = self.snap_distance if self.snap_distance is not None else default_snap

    targets = ([polygon_ents[n] for n in self.targets if n in polygon_ents]
               if self.targets is not None
               else list(polygon_ents.values()))

    targets.sort(
        key=lambda t: t.mesh_order if t.mesh_order is not None else float("inf")
    )

    # Flat caps: don't extend past linestring endpoints, so corner
    # artifacts where the strip crosses lateral prism faces don't pollute
    # the resolved trace.
    nominal_strip = unary_union(
        [ls.buffer(snap, join_style=2, cap_style="flat")
         for ls in self.linestrings]
    )

    # Replicate the cad_gmsh sequential cut cascade in shapely.
    cut_polys = []
    for tgt in targets:
        polys = tgt.polygons if isinstance(tgt.polygons, list) else [tgt.polygons]
        tgt_geom = unary_union(polys)
        for prev in cut_polys:
            tgt_geom = tgt_geom.difference(prev)
        cut_polys.append(tgt_geom)

    snapped = []
    for cp in cut_polys:
        if cp.is_empty:
            continue
        hit = cp.boundary.intersection(nominal_strip)
        if not hit.is_empty:
            snapped.append(hit)

    if snapped:
        merged = unary_union(snapped)
        self.resolved_linestrings = _flatten_to_linestrings([merged])
    else:
        self.resolved_linestrings = []

    if not self.resolved_linestrings:
        warnings.warn(
            f"InterfaceTag {self.physical_name} resolved to no segments"
        )
```

`_flatten_to_linestrings` accepts any combination of `LineString`,
`MultiLineString`, and `GeometryCollection` and returns a flat
`list[LineString]`. Degenerate `Point` intersections (nominal trace crossing
a boundary at a single point) are dropped.

`poly.boundary` includes interior holes, so InterfaceTag automatically picks
up donut-style hole edges.

### Walkthrough — abutting prisms

A `mesh_order=1`, B `mesh_order=2`, both buffered by `pert`.
Nominal LineString at `x=5`, `snap=2*point_tolerance`, flat caps.

1. nominal strip = `[5-2pt, 5+2pt] × [0, 5]` (flat caps clip y to the
   linestring's y range).
2. Cut cascade: `A_cut = A` (winner), `B_cut = B - A = [5+pert, 10+pert] × [...]`.
3. `A_cut.boundary ∩ strip` → vertical line at `x = 5+pert` (A's right
   edge in strip).
4. `B_cut.boundary ∩ strip` → vertical line at `x = 5+pert` (B_cut's
   new left edge, exactly identical to A's right since B was cut by A).
5. `unary_union(snapped)` → single line at `x = 5+pert`.

Result: single resolved segment at `x = 5+pert`, exactly the post-cut
shared face. B's pre-cut left at `x = 5-pert` never appears (it was
removed by the difference in step 2). No internal cut in A.

## `instanciate(cad_model)`

For each `LineString` in `self.resolved_linestrings`:

1. Create gmsh OCC points for each `(x, y, zmin)` vertex.
2. Create line segments between consecutive points (and a closing segment if
   the linestring is closed).
3. Build a wire from those segments.
4. Extrude the wire vertically by `(0, 0, zmax - zmin)` via
   `gmsh.model.occ.extrude`. The 2D surface dimtags returned by `extrude`
   are the entity's geometry.
5. Append the 2D dimtags to the entity's dimtag list.

The resulting 2D vertical surfaces are then handled by the existing pipeline
exactly like any `PolySurface` would be.

`instanciate_occ` raises `NotImplementedError` — InterfaceTag is gmsh-only
for v1.

## Tests

New file `tests/test_interface_tag.py`:

1. **`test_interface_tag_resolves_to_winning_boundary`** — A
   (`mesh_order=1`), B (`mesh_order=2`) abutting at `x=5`. Nominal LineString
   at `x=5`. Assert: exactly one face in the `interface_tag` physical group;
   its area ≈ winning face area; A is not internally split.

2. **`test_interface_tag_targets_none_picks_winners_only`** — Three abutting
   prisms (A, B, C) at `mesh_orders` 1/2/3, interfaces at `x=2` and `x=5`.
   InterfaceTag with linestring spanning both, `targets=None`. Assert: two
   resolved segments at `x=2+pert` and `x=5+pert`. No spurious internal cuts
   in any prism.

3. **`test_interface_tag_targets_explicit_subset`** — Same 3-prism scene,
   `targets=["A", "B"]`. Assert: only the A/B segment is tagged; B/C is not.

4. **`test_interface_tag_picks_up_hole_boundary`** — Outer prism with a
   circular hole, inner prism filling the hole. InterfaceTag traces the hole.
   Assert: resolves to the inner ring, gets a physical group with the right
   curved area.

5. **`test_interface_tag_no_match_warns_and_skips`** — InterfaceTag at
   `x=100` (off all targets). Assert: warning emitted, no physical group
   created, no crash, downstream meshing still succeeds.

6. **`test_interface_tag_with_gmsh_entity_target_skipped`** — Scene mixing
   PolyPrisms and a `gmsh_entity` plane. InterfaceTag with `targets=None`.
   Assert: `gmsh_entity` is skipped (no `.polygons`); only PolyPrism
   boundaries are considered.

7. **`test_interface_tag_extrudes_to_correct_z_range`** — Single PolyPrism
   `z ∈ [0, 1]`, InterfaceTag with `zmin=0, zmax=1`. Assert: resolved 2D
   surface bbox matches in z.

The existing `tests/test_cad_gmsh.py` suite must still pass (specifically
the test added in this branch for partial 2D-on-3D-face overlap).

## Out of scope (v1)

- `cad_occ` backend — `instanciate_occ` raises `NotImplementedError` with a
  clear message pointing users at the gmsh backend.
- 2D scenes (no z-extent) — InterfaceTag requires `zmin`/`zmax`. A 2D
  variant (a `Point`-tag analogue) is future work.
- Non-polygon targets (`PolyLine`, etc.) — silently skipped (treated like
  `gmsh_entity`).
- Tolerance ladder revision — explicitly deferred per maintainer request.
