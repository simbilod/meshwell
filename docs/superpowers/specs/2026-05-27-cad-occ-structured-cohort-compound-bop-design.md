# Structured-cohort compound BOP for cad_occ

**Status:** SUPERSEDED 2026-05-27 by `2026-05-27-cad-occ-cohort-preshared-faces-design.md`
**Date:** 2026-05-27
**Owner:** simbilod

> **Superseded note:** The smoke test in this spec ([Section 4](#validation-gate--smoke-test))
> revealed that `BOPAlgo_Builder.Modified(child)` returns empty for sub-shapes of a compound
> argument — i.e., per-piece history is not queryable when the argument is a compound. The
> sewing+compound mechanism described here cannot be made to work. The design pivoted to
> pre-sharing `TopoDS_Face` objects at construction time in the phantom builder (via
> `prism.LastShape()` reuse between vertically-stacked sub-prisms). See the replacement spec.
> Smoke-test code in `tests/test_cad_occ_cohort_sewing.py` was rewritten in place to validate
> the new design before this header was added.

## Problem

On production scenes with many structured slabs, the cad_occ backend spends most of its wall time in `_fragment_all` ([cad_occ.py:724-805](../../../meshwell/cad_occ.py#L724-L805)). The structured planner already decomposes each slab into sub-prisms with pre-conformed interfaces, but cad_occ still adds every sub-prism as a separate argument to a single `BOPAlgo_Builder`. BOPAlgo then performs pairwise intersection work between sub-prisms that the planner has already guaranteed are non-overlapping and conforming.

The per-entity cut loop ([cad_occ.py:1057-1146](../../../meshwell/cad_occ.py#L1057-L1146)) already short-circuits for structured-overridden entities. The remaining cost is in the global fragment pass.

## Goal

Skip pairwise BOPAlgo work between sub-prisms that the structured planner already conformed, while still letting BOPAlgo fragment correctly at boundaries between different cohorts and between cohorts and unstructured entities. Preserve all downstream invariants (ownership resolution, interface tagging, physical-group assignment).

**Success criterion:** on a representative production scene with ≥3 structured cohorts of ≥10 sub-prisms each, `_fragment_all` wall time drops measurably (target: ≥3×) with bit-identical mesh ownership vs. baseline.

## Non-goals

- Optimizing `compute_cutters` overlap checks for structured entities (separate issue; cheaper than `_fragment_all`).
- Changes to the gmsh backend.
- Parallel BOP execution strategies.
- Changes to per-entity cut, same-name fuse, tolerance clamping, or unwrap stages.

## Key insight

A "cohort" is a **connected z-component** of slabs (computed by `_connected_z_components` at [plan.py:1712](../../../meshwell/structured/plan.py#L1712)). Two slabs are in the same component iff they share a z-face or share the same z-interval as lateral neighbors. All slabs in a component are co-decomposed via a single `StackArrangement` and their pieces are guaranteed mutually conforming.

A cohort may span multiple entities (multiple `source_index` values) when several structured entities share a z-stack. Pieces *within* one cohort are conformed to each other; pieces *across* cohorts are not.

Critically, the phantom builder produces sub-prisms whose interface faces are **geometrically coincident but topologically distinct** ([phantom.py:557-559](../../../meshwell/structured/phantom.py#L557-L559), [phantom.py:476](../../../meshwell/structured/phantom.py#L476) — explicitly uses `Copy=True`). BOPAlgo's current role is partly to *unify* these into shared `TShape` identity. Downstream interface tagging ([occ_xao_writer.py:129-213](../../../meshwell/occ_xao_writer.py#L129-L213), [builder.py:810-851](../../../meshwell/builder.py#L810-L851)) relies on shared TShape identity to identify shared faces. Any design that skips BOPAlgo on intra-cohort interfaces must provide an alternative unification mechanism.

## Design

### Cohort metadata plumbing

**`Slab.component_index: int`** — new frozen field. Populated in the planner where `_connected_z_components(slabs)` already runs ([plan.py:1712](../../../meshwell/structured/plan.py#L1712)). Instead of discarding the component map after `StackArrangement` construction, write it back onto each `Slab` before the plan freezes.

**`PhantomShape.component_index: int`** — new field on [spec.py:363](../../../meshwell/structured/spec.py#L363). The phantom builder already knows the source slab; the value is `slab.component_index`.

**`_group_phantom_solids_by_entity` return type change** ([phantom.py:787-815](../../../meshwell/structured/phantom.py#L787-L815)):
```python
dict[int, list[tuple[Any, int]]]  # entity_index -> [(shape, component_index)]
```
Empty-list semantics for fully-carved entities is preserved as `[]`.

**`OCCLabeledEntity.shape_cohorts: list[int | None]`** — new parallel list on [cad_occ.py:62-86](../../../meshwell/cad_occ.py#L62-L86), 1:1 with `shapes`. `None` means "not part of any cohort" (unstructured entities, non-overridden entities). Populated at the override-installation point in `_instantiate_entity_occ`.

### `_fragment_all` change

In [cad_occ.py:742-766](../../../meshwell/cad_occ.py#L742-L766), replace the flat argument-add loop with a cohort-aware path:

1. Iterate `(ent_idx, shape, cohort)` triples. Bucket shapes where `cohort is not None` into `cohort_buckets: dict[int, list[(int, TopoDS_Shape)]]`. Non-cohort shapes are added as individual arguments as today.
2. For each cohort with ≥2 shapes:
   a. Build a `TopoDS_Compound` from the cohort's shapes via `TopoDS_Builder().MakeCompound + Add`.
   b. Run `BRepBuilderAPI_Sewing(self.fragment_fuzzy_value)` → `.Load(compound)` → `.Perform()` → `.SewedShape()`. The sewn result has shared TShapes at internal interfaces.
   c. For each original sub-prism, record its post-sewing equivalent via `sewing.ModifiedSubShape(orig)` (fall back to `orig` if no modification). This becomes the entry in `originals_per_entity` for that entity.
   d. `builder.AddArgument(sewn_shape)` — one argument per cohort.
3. Cohorts with exactly 1 shape skip sewing (nothing to unify) and add the shape directly.
4. Singleton-cohort and non-cohort shapes are added per the existing path.

Downstream piece collection ([cad_occ.py:787-796](../../../meshwell/cad_occ.py#L787-L796)) is unchanged in structure but iterates the (possibly-post-sewn) shapes rather than the originals. Ownership resolution is unchanged.

### Validation gate — smoke test

`tests/cad_occ/test_bopalgo_cohort_sewing.py` — must pass before any `_fragment_all` change ships:

1. **Sewing unifies coincident faces.** Build two independent `TopoDS_Solid`s (e.g., two boxes) that share a coincident face. Wrap in a compound, sew with `fragment_fuzzy_value` tolerance. Assert: after sewing, the two solids' face lists contain a face with shared TShape identity (verify via `TopTools_ShapeMapHasher`).
2. **`BOPAlgo_Builder.Modified()` on sewn compound children.** Build three solids A, B, C where A and B share a face (cohort) and C overlaps A but not B. Sew A+B into compound K. Add K and C as `BOPAlgo_Builder` arguments and `Perform()`. Assert: `Modified(A)` returns pieces consistent with `A ∩ C` and `A \ C`; `Modified(B)` returns `[B]` (or empty-and-not-deleted); the A-B interface face survives unchanged.
3. **Downstream tagging parity.** For a small synthetic scene (two adjacent structured slabs in one cohort + one unstructured neighbor), run the full pipeline twice: (a) with the cohort-compound path, (b) with the legacy flat-argument path. Assert: identical interface-group output (same physical names, same boundary tags, same TShape sharing pattern).

If any assertion fails, design is wrong; do not ship.

### Correctness invariants

**Preserved:**
- Mesh-order-based ownership resolution ([cad_occ.py:798-803](../../../meshwell/cad_occ.py#L798-L803)) — operates on pieces, not arguments.
- Per-entity cut already skips overridden entities — no change.
- `entity_shape_overrides` empty-list semantics for fully-carved entities — preserved.
- Same-name fuse and downstream stages — untouched.
- Interface tagging via shared TShape identity — preserved because sewing produces shared TShapes intra-cohort and BOPAlgo continues to produce shared TShapes inter-cohort.

**Risks and mitigations:**
- `component_index` must be deterministic across planner runs. Mitigation: `_connected_z_components` already operates on a sorted slab list; this is stable.
- A cohort containing shapes from a *non-overridden* entity would short-circuit cuts that should still run. Mitigation: assert in `_fragment_all` that every shape with `cohort is not None` belongs to an entity that was in `entity_shape_overrides`.
- Sewing tolerance differs from BOP fuzzy value semantics. Mitigation: use the same `self.fragment_fuzzy_value` to keep behavior coherent; smoke test (3) verifies parity with legacy path.
- Sewing may alter face orientation. Mitigation: verify in smoke test (1) that orientation is preserved (BOPAlgo is orientation-sensitive).
- `sewing.ModifiedSubShape` may return `IsNull()` for unmodified shapes. Mitigation: fall back to the original shape in that case.

## Deferred — Option (2): Pre-share faces at planner level

Modify the phantom builder so adjacent sub-prisms reuse a single `TopoDS_Face` at their interface via explicit `BRep_Builder.Add`, eliminating the sewing pass entirely. This would:

- Remove the per-cohort sewing cost.
- Eliminate one tolerance-dependent stage.
- Allow the cohort-compound to be passed directly to BOPAlgo with shared TShapes from construction time.

It requires non-trivial planner refactoring: the phantom builder must track and reuse face objects across sub-prism construction, which interacts with the existing `Copy=True` design intent ([phantom.py:465-466](../../../meshwell/structured/phantom.py#L465-L466)) that gives each consumer a fresh TShape "for downstream per-slab keying." That intent would need re-examination.

Revisit if (a) sewing cost still dominates after this change, or (b) sewing tolerance produces brittleness in production scenes.

## Out of scope (separate work)

- `compute_cutters` overlap test for sub-prisms ([cad_occ.py:156-229](../../../meshwell/cad_occ.py#L156-L229)) still runs N². Cheaper than `_fragment_all`; defer.
- Parallelization of cohort sewing itself.
- Backporting cohort metadata to non-structured paths.
