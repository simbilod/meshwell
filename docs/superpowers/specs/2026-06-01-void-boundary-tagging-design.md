# Void boundary tagging — design

**Date:** 2026-06-01
**Branch:** `feat/structured_discrete`
**Builds on:** [2026-05-30-structured-prism-meshing-design.md](2026-05-30-structured-prism-meshing-design.md)

## Goal

When a structured PolyPrism with `mesh_bool=False` ("void") carves a
hole in a surrounding structured slab, the resulting boundary faces
should be tagged with the void's `physical_name`, matching the way
meshwell already tags `keep=False` helpers in the non-structured
path (`neighbour___helper` interface groups).

Concretely, for a `bg` square with a `hole` disc void carved through:

- The cylindrical wall between bg and the carved void → `bg___hole`
- The disc-shaped patch at z=void.zmax where the void meets a cap
  above → `cap___hole`
- The disc-shaped patch at z=void.zmin where the void meets a base
  below → `base___hole`
- Free void boundaries (no neighbour) → no group at all (consistent
  with how top-dim keep=False helpers behave in the non-structured
  path)

## Why the current pipeline doesn't produce these tags

The structured planner *consumes* voids in `decompose.py`. In
`zinterval_footprint`, void slabs (mesh_bool=False) subtract their
footprint from the accumulator. The `_owner_slab` containment check
in `decompose_cohorts` then returns `None` for any sub-piece whose
representative point lies inside a void's footprint, and that
sub-piece is silently dropped.

The void's region therefore produces *no* `SubPiece`, *no* OCC
sub-solid, and *no* `OCCLabeledEntity`. The void never reaches the
XAO writer's keep=False machinery, so its physical_name is never
used for interface naming.

## Approach: voids as first-class keep=False sub-solids

Stop dropping void sub-pieces. Emit them like any other slab. Build
sub-solids for them in the cohort compound. Mark them `keep=False`
when the post-pass synthesizes per-sub-solid `OCCLabeledEntity`
records. Skip them in wedge stamping.

After that, *no* new tagging machinery is needed — the XAO writer's
existing keep=False handling produces the desired
`neighbour___void` tags from shared TShapes between the kept
neighbour and the void sub-solid.

## Architecture summary

```
decompose.py
  zinterval_footprint  — return (sub_polygons, source_slabs) pairs
                          from Policy B resolution, not a single
                          carved footprint. Voids contribute their own
                          (footprint, void_slab) pair.

  _owner_slab          — return the void's source_index for points
                          owned by voids (instead of None).

  decompose_cohorts    — emit SubPiece for void sub-pieces. They look
                          identical to solid sub-pieces structurally;
                          the difference (mesh_bool=False on the
                          source slab) is carried by
                          source_slab_indices.

types.py
  SlabMeta             — add `keep: bool`.

build.py
  build_cohort_compound — already builds one sub-solid per SubPiece;
                          inherits the void changes for free. Populate
                          `keep` in slab_meta from the source slab's
                          mesh_bool.

pipeline.py
  structured_post_pass — when expanding cohort entity to per-sub-solid
                          OCCLabeledEntities, set keep = source slab's
                          mesh_bool. Void sub-solids get keep=False
                          and the void's user-facing physical_name.

wedge.py
  apply_lateral_transfinite_hints — when picking n_layers from a
                          shared lateral face's owners, consider only
                          keep=True owners. n_layers mismatches between
                          keep=False owners and anything else are
                          ignored. (Voids don't dictate vertical
                          resolution.)

  stamp_wedges          — skip sub-solids where slab_meta.keep == False.
                          No wedge emission, no physical group
                          registration.

occ_xao_writer.py
  unchanged             — existing keep=False handling does the work.
```

## Data flow (worked example — case 4, sandwiched void)

Scene:
- `base` (unstructured, z=[-1,0])
- `bg` (structured square, z=[0,1], mesh_order=2)
- `hole` (structured disc, z=[0,1], mesh_order=1, mesh_bool=False)
- `cap` (unstructured, z=[1,2])

```
collect       structured_slabs = [bg_slab, hole_slab]
              hole_slab.mesh_bool = False

cohort        bg_slab + hole_slab → one cohort (lateral-touch:
              hole footprint ⊂ bg footprint at z=[0,1])
              cohort.z_planes = (0.0, 1.0)

validate      validate_z_stacks: pass
              validate_no_volumetric_cohort_overlap: pass
              validate_arc_consistency: pass
              validate_void_mesh_order (NEW): pass

decompose     zinterval_footprint at z=[0,1] returns:
                [(annular_ring, bg_slab),
                 (disc,         hole_slab)]
              cut sources at z=0: bg + hole + base boundaries
              cut sources at z=1: bg + hole + cap boundaries

              SubPieces:
                SubPiece(annular_ring, source=(bg_slab.source_index,))
                SubPiece(disc,         source=(hole_slab.source_index,))

              Bidirectional pre-cut: base and cap polygons get split at
              z=0 and z=1 into {annular_ring, disc, outside} sub-prisms.

build         Cohort compound = 2 sub-solids:
                solid_sub  (annular_ring prism)
                void_sub   (disc prism)
              Inner-cylindrical lateral TShape shared between them
              via the EdgeRegistry, same as any laterally-adjacent pair.

              slab_meta:
                solid_sub  → SlabMeta(physical_name=("bg",),
                                       keep=True, ...)
                void_sub   → SlabMeta(physical_name=("hole",),
                                       keep=False, ...)

cad_occ       Cohort compound goes in as one BOP argument.
              base and cap (pre-cut) fragment cleanly against the
              cohort at z=0 / z=1 because boundaries match by
              construction (bidirectional pre-cut + unified arc
              detection).
              Shell-invariance validator confirms cohort shell faces
              survive BOP.

post-pass     Cohort entity expands to 2 OCCLabeledEntity records:
                bg-entity    (dim=3, keep=True,  name=("bg",))
                hole-entity  (dim=3, keep=False, name=("hole",))

XAO write     hole-entity is top-dim keep=False:
                its body is NOT serialized to BREP
                its face TShapes ARE used to name interfaces
              Pairs emitted via shared TShapes:
                inner cylinder         → bg___hole
                disc at z=0            → base___hole
                disc at z=1            → cap___hole
              Plus standard kept-vs-kept interfaces:
                annular_ring at z=0    → bg___base
                annular_ring at z=1    → bg___cap

gmsh          Sees BREP without the void's body or its bot/top faces
              (the disc patches at z=0 and z=1 ARE in BREP because
              they're shared with kept base/cap pre-cut sub-faces).

pre_2d        apply_lateral_transfinite_hints: inner cylindrical face
              has two owners (bg, hole). n_layers picked from bg
              (keep=True). hole's keep=False status excludes it from
              the mismatch check.

pre_3d        stamp_wedges: iterate slab_meta; skip void_sub (keep=False).
              bg sub-solid gets wedges stamped normally.

mesh out      .msh contains:
                "bg" 3D group (wedges)
                "base" 3D group (tets)
                "cap"  3D group (tets)
                "bg___hole"   inner cylinder quads
                "base___hole" disc triangles at z=0
                "cap___hole"  disc triangles at z=1
                "bg___base", "bg___cap" annular interfaces
                "bg___None" outer walls
                "base___None", "cap___None" outer walls
                No "hole" 3D group, no "hole___None" group
```

## Edge cases

**Handled by the design without new code:**

| Case | Handling |
|---|---|
| Void with no neighbour (case 1) | keep=False → body excluded from BREP → no exposed face, no `___None` tag |
| Void spanning stacked cohort (case 5) | Void slab participates in Union-Find by face-touch; sub-pieces emit per z-interval; lateral TShape shared with each solid sub-piece independently |
| Void below cohort slab (case 6) | Void's top at z=1 shares TShape with upper sub-piece's disc sub-face (from pre-cut) → `upper___hole` |
| Two overlapping voids | Policy B on the void-vs-void pair: lower mo wins. Higher mo void contributes nothing (its area is in the lower mo void's footprint, so it gets the dropped-as-already-claimed treatment from `_owner_slab`). |
| Void XY-disjoint from any solid | Forms a void-only cohort. Sub-solid is fully keep=False → no faces in mesh. Valid but yields no output. |

**Handled by new validation:**

- **Void without `mesh_order`** (defaults to `None` → `float("inf")` in Policy B sort). With voids emitted as sub-pieces, the void going last in sort order would result in it carving solids that already ran — which is a confusing semantic. **Raise** `StructuredVoidMeshOrderRequiredError` at planner stage if any `mesh_bool=False` slab has `mesh_order=None`.

**Carried by existing infrastructure:**

- Arc-vs-polyline at void/neighbour shared z-planes uses the same bidirectional pre-cut + unified arc-detection path that cohort-vs-neighbour uses today. The same caveats (the `feat/structured-clean4` arc-merge corner case) apply.

**Out of scope for v1:**

- Tagging void-vs-void interfaces. Current XAO writer behaviour skips pairs where both sides are keep=False ([`occ_xao_writer.py:273`](../../../meshwell/occ_xao_writer.py#L273)). Stay consistent.
- Voids without `structured=True`. Use the existing non-structured `keep=False` path for those.

## Failure modes

| Stage | Error | Trigger |
|---|---|---|
| collect | `StructuredVoidMeshOrderRequiredError` (NEW) | void slab with `mesh_order=None` |

Existing failure modes
([structured spec](2026-05-30-structured-prism-meshing-design.md))
all continue to apply unchanged.

## API surface changes

No public API change. Users continue to write:

```python
solid = PolyPrism(big_polygon, {0.0: 0.0, 1.0: 0.0},
                  physical_name="bg",
                  structured=True, mesh_order=2.0)
hole = PolyPrism(small_polygon, {0.0: 0.0, 1.0: 0.0},
                 physical_name="hole",
                 structured=True, mesh_order=1.0, mesh_bool=False)

generate_mesh([solid, hole], dim=3, ...)
```

After this change, the user's mesh will contain `bg___hole` interface
groups (instead of `bg___None` covering the carved boundary as it
does today, or no group at all where the current pipeline drops the
void region entirely).

## Test plan

Reintroduce a `tests/structured/test_void_tagging.py` with the 7
positive cases the user already specified, plus three new tests:

```
test_void_inside_single_structured_slab          # case 1 — lateral only
test_void_below_unstructured_cap                  # case 2 — top→cap
test_void_above_unstructured_base                 # case 3 — bot→base
test_void_sandwiched_between_unstructured         # case 4 — all three
test_void_through_stacked_cohort                  # case 5 — lateral on both
test_void_below_structured_cohort_slab            # case 6 — top→upper slab
test_void_square_no_arcs                          # case 7 — polyline void
test_void_with_arc_neighbour_pre_cut              # NEW — arc-vs-pre-cut
test_two_overlapping_voids_policy_b               # NEW — void-vs-void Policy B
test_void_no_mesh_order_raises                    # NEW — validator
```

Each positive test asserts:
- The expected interface group(s) exist (e.g. `bg___hole`).
- The void's `physical_name` is NOT a 3D physical group.
- The void's `physical_name` is NOT a `___None` boundary group.
- For cases with both keep=True and keep=False sides on a lateral face,
  the wedge mesh in the keep=True volume is non-empty.

The 86 currently-passing structured tests must continue to pass.
`demo_curves.py` and `demo_structured.py` must produce mesh files
that are byte-identical (or at minimum, structurally identical with
the same wedge counts) before and after this change for scenes that
contain no voids.

## Implementation order suggestion

Mechanical changes first, then end-to-end:

1. Add `keep` to `SlabMeta` (no behaviour change).
2. Rewrite `zinterval_footprint` and `_owner_slab` to emit void sub-pieces.
3. Plumb `keep` through `build_cohort_compound` and `structured_post_pass`.
4. Add the `mesh_bool=False ⟹ mesh_order required` validator.
5. Update `apply_lateral_transfinite_hints` to ignore keep=False owners
   for the n_layers consistency check.
6. Update `stamp_wedges` to skip keep=False sub-solids.
7. Add the 10 tests; verify the 86 existing tests still pass.
