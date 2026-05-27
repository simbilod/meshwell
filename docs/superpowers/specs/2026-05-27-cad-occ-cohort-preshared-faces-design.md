# Pre-shared TopoDS_Face between vertically-stacked sub-prisms

**Status:** design
**Date:** 2026-05-27
**Owner:** simbilod
**Supersedes:** `2026-05-27-cad-occ-structured-cohort-compound-bop-design.md`

## Problem

cad_occ's `_fragment_all` ([cad_occ.py:724-805](../../../meshwell/cad_occ.py#L724-L805)) calls `BOPAlgo_Builder` over all entity shapes — including every structured sub-prism as a separate argument. The structured planner already conformed pieces within a cohort (connected z-component), but BOPAlgo doesn't know that, so its pave-filler still computes pairwise intersections between sub-prisms that are guaranteed to mate cleanly.

The first design pivot attempted to bundle each cohort into a `TopoDS_Compound` and add the compound as one BOPAlgo argument. **That design failed validation** ([tests/test_cad_occ_cohort_sewing.py#L65-L100](../../../tests/test_cad_occ_cohort_sewing.py#L65-L100), test 2 in the prior spec): `BOPAlgo_Builder.Modified(child)` returns empty for sub-shapes of a compound argument, breaking the per-piece history that downstream ownership resolution relies on.

## Pivot: pre-share TopoDS_Face at construction time

When `BRepPrimAPI_MakePrism(bottom_face, gp_Vec(0,0,h))` is called, the resulting prism's `LastShape()` returns the *top* face — a fresh `TopoDS_Face` whose TShape is internal to that prism. Reusing that same `TopoDS_Face` as the bottom face for the next prism (e.g., the slab vertically above) produces two solids that **genuinely share the interface face's TShape**. BOPAlgo sees them as separate arguments and tracks per-argument history correctly, but its pave-filler can recognize the shared TShape and skip the heavy intersection at the interface.

This was validated by three smoke tests committed in `tests/test_cad_occ_cohort_sewing.py`:

1. `test_prism_top_can_be_reused_as_next_prism_bottom` — the mechanism produces shared TShape identity.
2. `test_bopalgo_modified_works_per_solid_with_shared_interface_face` — `Modified()` works per-argument even when arguments share an internal face.
3. `test_shared_interface_survives_bop_with_inert_neighbor` — the shared interface TShape survives a BOP pass.

(The file is misnamed `_sewing` for historical reasons; rename is included in the implementation plan.)

## Goal

For each cohort, build vertically-stacked sub-prisms such that adjacent prisms share their interface `TopoDS_Face`. No changes to cad_occ. No metadata plumbing into `OCCLabeledEntity`. The optimization comes from BOPAlgo recognizing shared TShapes between independent arguments.

**Success criterion:** on a representative production scene dominated by vertically-stacked structured slabs, `_fragment_all` wall time drops measurably (target: ≥3×) with bit-identical mesh ownership vs. baseline.

## Non-goals (this phase)

- **Pre-sharing lateral faces** between sub-prisms in the same z-interval. The mechanism (`prism.LastShape()` for tops) doesn't apply to laterals. Lateral interfaces are out of scope; they'll continue to be unified by BOPAlgo as today. Revisit in Phase 2 if measured speedup is insufficient.
- **Sewing**, **compound arguments**, **per-cohort BOP calls** — all rejected during exploration.
- Changes to `compute_cutters`, per-entity cut, same-name fuse, or downstream stages.
- Changes to the gmsh backend.

## Key insight

The structured planner produces each cohort's `face_partition` once — vertically adjacent slabs in the same cohort that occupy the same XY footprint piece reference geometrically identical Shapely Polygons (the investigator confirmed this in the design map: "vertically adjacent slabs with coincident XY get the same partition polygon, hence the same bottom-face blueprint"). Today the phantom builder still constructs each slab's bottom face independently — using `BRepBuilderAPI_Transform(Copy=True)` to give each consumer a fresh TShape ([phantom.py:464-466](../../../meshwell/structured/phantom.py#L464-L466)). That `Copy=True` is intentional today — it gives per-slab edge/vertex distinctness — but it forecloses pre-sharing.

The new design **inverts this** for the specific case of vertical-stack interfaces:

1. Group cohort slabs by `(component_index, piece_polygon_identity)` → vertical stacks of pieces.
2. For each stack, sort by `zlo` and build bottom-up:
   - First slab in the stack: build bottom face normally.
   - Each subsequent slab: use the previous slab's `prism.LastShape()` as its bottom face.
3. `BRepPrimAPI_MakePrism` then generates a new top face per slab; only the *bottom* face (= previous's top) is shared.

Per-slab edge/vertex distinctness is **preserved** for laterals (each prism builder generates its own laterals from its own bottom edges) and for top faces. Only the shared interface face — and its boundary edges/vertices — have shared TShape identity, which is exactly what we want.

## Design

### Slab metadata

**`Slab.component_index: int`** — new frozen field on [spec.py:251-286](../../../meshwell/structured/spec.py#L251-L286). Populated in `build_plan` from `_connected_z_components` ([plan.py:286](../../../meshwell/structured/plan.py#L286)), which is already called by `build_stack_arrangements`. The component index is currently transient; we persist it.

This is the only metadata change. No change to `PhantomShape`, `OCCLabeledEntity`, `_group_phantom_solids_by_entity`, or any cad_occ structure.

### Phantom builder change

**New helper in `phantom.py`:** `_group_slabs_into_vertical_stacks(plan)` returns a structure suitable for ordered bottom-up traversal. Concretely:

```python
def _group_slabs_into_vertical_stacks(
    plan: StructuredPlan,
) -> list[list[tuple[Slab, int]]]:
    """Return list of vertical stacks; each stack is a list of (slab, piece_index)
    pairs in ascending zlo order. A "stack" is a set of pieces whose
    geometry is identical (same polygon key) within one cohort, such that
    sub-prism i+1's bottom face is the same XY at z = sub-prism i's zhi.

    Singleton stacks (no vertical neighbor) are included too — they just
    don't get sharing applied.
    """
```

**Grouping key:** `(slab.component_index, _polygon_face_cache_key(slab.face_partition[piece_idx], ...))`. The polygon cache key is the same one the existing face cache uses ([phantom.py:830](../../../meshwell/structured/phantom.py#L830)), so polygon identity is consistent with the existing caching logic.

**Z-touching check:** two adjacent slabs in the same `(cohort, polygon_key)` group form a vertical stack iff `abs(slab[i].zhi - slab[i+1].zlo) < _Z_TOL`. If there's a gap, the lower stack ends and a new one starts above.

**Modified `build_phantom_shapes`:**

1. Compute stacks via `_group_slabs_into_vertical_stacks(plan)`.
2. For each stack, walk bottom-up. Maintain `prev_top_face: TopoDS_Face | None`.
3. For each `(slab, piece_index)`:
   - Call `_build_sub_prism(...)` with new optional `bottom_face_override=prev_top_face`.
   - Capture the resulting prism's `LastShape()` and assign to `prev_top_face` for the next iteration.
4. Slabs/pieces NOT in any multi-element stack go through the existing build path unchanged.

**`_build_sub_prism` change:** add an optional `bottom_face_override: TopoDS_Face | None = None` parameter. When provided, skip bottom-face construction (the cache lookup, `_face_at_z`, `_make_face_from_polygon_with_arcs`) and use the override directly. The existing logic for building edges/vertices via `TopExp_Explorer` on the bottom face continues unchanged — it just sees the shared face's TShape, which is fine (PhantomShape.input_edges_by_key entries will share TShapes with the previous slab's input_edges_by_key for the top side; investigator confirmed this is safe — the consumer `extract_phantom_map` iterates by key and calls `Modified()` per key, which returns the correct successor list regardless).

### Validation gate — existing smoke tests

`tests/test_cad_occ_cohort_sewing.py` (rename to `tests/test_cad_occ_cohort_preshared_faces.py` in the plan) already proves the mechanism works. No changes needed.

### End-to-end parity test (new)

`tests/test_phantom_preshared_faces.py`: build a small synthetic cohort of three vertically-stacked structured polyprisms with one unstructured neighbor; run the full pipeline twice with a kill-switch toggle that disables the pre-sharing path; assert:

1. Identical entity piece counts.
2. Identical physical names per entity.
3. Identical interface tagging output (shared face counts between entity pairs).

The kill-switch is a module-level constant `phantom._PRESHARE_VERTICAL_FACES = True` or an env var the test flips. It's there only for parity testing; default stays `True`.

### Correctness invariants

**Preserved:**
- Per-slab `input_faces_by_key` mapping (two PhantomShapes can map to the same TShape — investigator confirmed this is safe; `extract_phantom_map` iterates per key, calling `builder.Modified()` independently per key).
- Per-slab lateral face generation via `prism_builder.Generated(bot_edge).First()` — each prism builder produces its own laterals from its own bottom-face edges (which, post-share, may have shared TShapes with the slab above, but that's irrelevant — Generated() operates on the builder's internal state, not on TShape identity).
- BOPAlgo per-argument `Modified()` history.
- Downstream interface tagging via shared TShape identity ([occ_xao_writer.py:129-213](../../../meshwell/occ_xao_writer.py#L129-L213)) — now actually *helped* by pre-sharing, since shared TShapes are present at construction time instead of having to be unified by BOP.

**Risks and mitigations:**
- `_build_sub_prism` shared-bottom path must produce the same `input_edges_by_key` traversal order as the non-shared path, else downstream edge tagging would change. Mitigation: the edge traversal code (`TopExp_Explorer` on bottom-face wire) is unchanged; only the face object's TShape differs.
- For slabs at the bottom of their stack (no `prev_top_face`), behavior must be identical to today. Mitigation: pass `bottom_face_override=None` → existing code path.
- A cohort that has been changed by `compute_face_partition` after `_connected_z_components` could have polygon_keys that don't match across slabs. Mitigation: in the plan, run `_group_slabs_into_vertical_stacks` *after* all planner stages, working from the final `face_partition`.
- `BRepPrimAPI_MakePrism` may produce a top face whose edges' TShapes differ in subtle orientation from a freshly-constructed face at that z. Mitigation: smoke test 3 already verified post-BOP shared-face survival; if downstream sees an orientation mismatch, the parity test will catch it.

## Deferred — Phase 2: Lateral face pre-sharing

Lateral interfaces between sub-prisms in the same z-interval cannot use the `prism.LastShape()` trick — laterals are generated per-prism by `prism_builder.Generated(bot_edge)`. Pre-sharing them would require either (a) building a single shared lateral face explicitly and using `BRep_Builder` to assemble each sub-prism's solid from explicit faces, or (b) some other shared-edge mechanism that coordinates `Generated()` across builders.

Defer until Phase 1 measurement shows whether the remaining lateral cost still dominates. If Phase 1 alone delivers ≥3×, Phase 2 may never be needed.

## Out of scope (separate work)

- `compute_cutters` overlap test for sub-prisms still runs N². Cheaper than `_fragment_all`; defer.
- Backporting cohort metadata to non-structured paths.
- Parallel phantom building.
