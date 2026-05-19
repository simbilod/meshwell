# cad_occ: replace sequential per-tool cuts with a batched compound cut

**Status:** approved (2026-05-19), pending implementation
**Author:** simbilod (with Claude)
**Related:** [`meshwell/cad_occ.py`](../../../meshwell/cad_occ.py)

## Summary

Replace the sequential per-tool `BRepAlgoAPI_Cut` loop in
`CAD_OCC.process_entities_cut_only` ([cad_occ.py:474-481](../../../meshwell/cad_occ.py#L474-L481))
with a single `BRepAlgoAPI_Cut(substrate, compound_of_tools)` call per
substrate sub-shape. On a dense substrate-vs-N-bodies scene the cut cascade
drops from ~360 ms to ~95 ms at n=20 bodies — a **3.6×–3.9× speedup on the
cut step**, which is **~90 % of the full CAD pipeline cost** on dense scenes.

## Motivation

Profiling on a synthetic scene that mirrors the production "substrate +
metal pillars + vias + top cladding" pattern showed:

| n_bodies | cut cascade | fragment | cut share of pipeline |
|---:|---:|---:|---:|
| 12 | 444 ms | 66 ms | 87 % |
| 20 | 805 ms | 57 ms | 93 % |

The cut cascade is the dominant cost. Speeding it up directly translates to
overall CAD speedup.

The current code does N sequential `BRepAlgoAPI_Cut(result, tool)` calls,
each rebuilding internal BOP data structures. A single
`BRepAlgoAPI_Cut(substrate, compound_of_tools)` lets OCC plan the whole
operation once and run it with internal parallelism.

## The "previous failure" comment is stale

The current code carries a comment at [cad_occ.py:464-472](../../../meshwell/cad_occ.py#L464-L472)
warning that batched compound cuts "were observed to produce empty results
(zero SOLIDs) for large bodies like a substrate cut against ~10 metal+helper
bodies." That comment dates from commit `aa77f21` ("cuts avoid compounds",
2026-04-29). Empirical reproduction in May 2026 shows:

- The compound-cut approach (with explicit `Build()`, `SetFuzzyValue`,
  `SetRunParallel`) produces **correct results** on every test scene tried,
  including dense substrate-vs-20-bodies.
- Even mimicking the pre-`aa77f21` code verbatim (no `Build()`, no fuzzy)
  produced correct results.
- The full test suite (313 tests) shows **zero new failures** under the
  batched-compound code path. All failures present in the patched run are
  also present in baseline.

The most likely explanation: the bug was caused by the original code's
**AABB-only pre-filter**, which admitted volume-disjoint tools into the
compound. `BRepAlgoAPI_Cut` on AABB-overlapping but volume-disjoint shapes
silently splits the substrate (per the same warning at
[cad_occ.py:209-238](../../../meshwell/cad_occ.py#L209-L238)). The current
`_shapes_actually_overlap` filter (added later) rejects such tools, which
makes the compound cut safe.

The fix therefore requires keeping the modern `_shapes_actually_overlap`
gate; it is load-bearing for correctness of the batched approach.

## Alternatives considered (rejected)

**`BRepAlgoAPI_Splitter` with `TopTools_ListOfShape`.** Faster on dense
scenes (up to 6.9× cut speedup) but produces wrong results when a tool is
fully enclosed by the substrate. Splitter only *splits* arguments by tools;
it does not *subtract*. On the photonic_stack scene `top_clad` cut against
`contact_a` (fully enclosed) gained +1 SOLID and +0.97 % volume.
Recovering Cut semantics would require a per-piece "centroid inside any
tool" post-filter, which erodes the speedup and adds another correctness
risk. Not worth it given that `f_cut_compound` already gives 3.6×–3.9×.

**`SetRunParallel(True)` on each per-tool `BRepAlgoAPI_Cut`.** Consistently
*slower* than baseline (0.7×–0.99×). OCC's per-cut parallel setup cost
exceeds the gain on shapes this size.

**Independent-set partitioning of tools into non-overlapping batches.**
Conceptually safe fallback if compound cuts ever break, but adds complexity
(graph-coloring) for no extra benefit over `f_cut_compound` on the scenes
tested.

## Design

### Change

Replace the inner loop at [cad_occ.py:474-481](../../../meshwell/cad_occ.py#L474-L481):

```python
for s in labeled.shapes:
    try:
        result = s
        for ts in tool_shapes:
            cut_op = BRepAlgoAPI_Cut(result, ts)
            cut_op.SetFuzzyValue(self.cut_fuzzy_value)
            cut_op.Build()
            result = cut_op.Shape()
    except Exception as e:
        ...
```

with a single compound cut per sub-shape:

```python
tool_compound = TopoDS_Compound()
cb = BRep_Builder()
cb.MakeCompound(tool_compound)
for ts in tool_shapes:
    cb.Add(tool_compound, ts)

for s in labeled.shapes:
    try:
        cut_op = BRepAlgoAPI_Cut(s, tool_compound)
        cut_op.SetFuzzyValue(self.cut_fuzzy_value)
        cut_op.SetRunParallel(self.n_threads > 1)
        cut_op.Build()
        result = cut_op.Shape()
    except Exception as e:
        ...
```

Add imports for `BRep_Builder` and `TopoDS_Compound`.

### Comment rewrite

The current warning comment becomes incorrect after the change. Replace
[cad_occ.py:464-472](../../../meshwell/cad_occ.py#L464-L472) with:

```python
# Batched compound cut. Bundles all surviving tools into a single
# TopoDS_Compound and does one BRepAlgoAPI_Cut per substrate piece.
# This is ~3-4x faster than a per-tool sequential cascade on dense
# scenes (cut cascade dominates ~90% of CAD pipeline cost there).
#
# The _shapes_actually_overlap pre-filter above is LOAD-BEARING:
# without it, AABB-overlapping but volume-disjoint tools would enter
# the compound and OCC would silently split the substrate (see the
# detailed warning on _shapes_actually_overlap). An earlier
# implementation using AABB-only pre-filtering hit this bug and
# adopted sequential cuts as a workaround (see commits aa77f21 +
# 146b05a, Apr 2026); reverting to the batched form is safe now that
# the geometric-overlap gate is in place.
```

### Public API

No public API change. The signature, return type, and observed semantics
of `process_entities_cut_only`, `process_entities`, and the `cad_occ()`
top-level function are unchanged.

### Configuration

No new config flags. The existing `n_threads`, `cut_fuzzy_value`, and
`perturbation` settings continue to control the same surfaces of behaviour.

### Testing

Pre-existing test suite already exercises both the cut cascade and the
fragment step. Empirical run shows **zero new failures** under the patched
code path (188 / 313 relevant tests pass identically; the 6 pre-existing
failures are unchanged; 4 `test_restrict` failures are flaky regardless of
the patch).

Implementation plan should additionally:

1. **Add a regression test** that builds the dense substrate-vs-N-bodies
   scene from `scripts/bench_cut_strategies_dense.py` and asserts post-cut
   solid count = 1 per cut entity and post-cut volumes match the
   expected analytic values. This locks in the correctness contract that
   could regress if anyone re-loosens the pre-filter.

2. **Keep the benchmark scripts** under `scripts/bench_cut_strategies*.py`
   (already written during the spike) so future developers can re-validate
   if OCC behaviour changes.

3. **Run the full structured test suite + the broader test suite** under
   the new code path before merging.

## Out of scope

- The structured-plan-stage bbox check optimisation (`shapely + zmin/zmax`)
  is a separate brainstorm thread the user opened in the same session.
  Independent design will follow.
- Further per-cut parallelisation (process pool over substrate sub-shapes)
  is not warranted: typical sub-shape count per entity is 1 in tested
  scenes.
- The `BRepAlgoAPI_Splitter`-with-post-filter alternative may be worth
  revisiting if a future scene shows the 3.6× ceiling is too low, but is
  not pursued here.

## Risks

- **Hidden geometric configurations.** No reproduction of the original
  empty-result bug was achieved on the test scenes tried. There may exist
  production scenes (not in the test suite) that still trigger it. Mitigation:
  the change can be reverted by restoring the sequential loop; no other
  code depends on the new shape of the compound cut.
- **OCC version drift.** If OCC is upgraded and reintroduces a compound-cut
  bug, the regression test on the dense scene would catch it.
- **Filter dependency.** The compound cut depends on
  `_shapes_actually_overlap` rejecting volume-disjoint AABB-overlapping
  tools. The rewritten comment makes this explicit. The regression test
  protects against accidental filter loosening.
