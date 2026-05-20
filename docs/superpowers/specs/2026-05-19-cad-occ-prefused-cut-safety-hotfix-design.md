# cad_occ: switch cut cascade from compound to prefused (tangent-tool safety hotfix)

**Status:** approved (2026-05-19), pending implementation
**Author:** simbilod (with Claude)
**Related:**
- Discovery bench: [`2026-05-19-cad-cut-strategy-sweep-bench-design.md`](2026-05-19-cad-cut-strategy-sweep-bench-design.md)
- Bench output: `docs/distributed_example_work/bench_cut_strategy_sweep_2026-05-19.txt`
- Prior optimization being replaced: [`2026-05-19-cad-occ-batched-compound-cut-design.md`](2026-05-19-cad-occ-batched-compound-cut-design.md)
- Deferred follow-up: cross-entity AABB-DAG parallelism (no spec yet; planned)

## Summary

Replace the `BRepAlgoAPI_Cut(s, Compound(tools))` block in `CAD_OCC.process_entities_cut_only` with a prefused variant: `fused = BOPAlgo_BOP(Fuse, tools); Cut(s, fused)`. Eliminates a silent correctness bug — surfaced by the cut-strategy sweep bench — where `Cut(s, Compound)` corrupts the substrate (splits into multiple SOLIDs, or vanishes entirely) whenever two tools in the compound share a face or edge (tangent tools). Sequential and prefused are correct in all tested configurations.

The hotfix accepts a small perf regression (≈20–30 % slower than compound on disjoint scenes) in exchange for universal correctness. The deferred Layer 2 spec will recover this gap via cross-entity parallelism rather than by reverting to the unsafe compound form.

## The bug we're fixing

The cut-strategy sweep bench at `scripts/bench_cut_strategy_sweep.py` reproduces the failure across multiple substrate × tool × N configurations. Two representative cells:

```
=== substrate=square tool=square overlap=tangent ===
  N            sequential                      compound                      prefused
  2             25.04ms*     22.58ms  WRONG(s=2,v=400.0)                      28.00ms
 10            177.17ms     218.59ms  WRONG(s=1,v=0.0)                       113.76ms*

=== substrate=disc tool=disc overlap=tangent ===
  N            sequential                      compound                      prefused
  2            145.53ms*    285.58ms  WRONG(s=2,v=312.2)                     155.68ms
 20           3777.00ms   14783.20ms  WRONG(s=11,v=312.2)                    1552.27ms*
```

`compound` either splits the substrate into multiple SOLIDs (`s=2..11`, signifying face-topology corruption while material is uncut) or vanishes it (`s=1, v=0.0`, the original "empty result" pattern). The `_shapes_actually_overlap` pre-filter does NOT reject tangent tool pairs — their distance is 0, within `cut_fuzzy_value`. So tangent tools enter the compound in production.

The bug is latent today because:
- The existing regression test (`tests/test_cad_occ_batched_compound_cut.py`) uses pillars on a 2.5× pitch (disjoint).
- The photonic_stack scene has disc (r=1.5) inside ring inner-radius (r=2.5) — disc and ring don't touch; bus, contact_a, top_clad are mutually non-tangent.

Any production scene with abutting metals, concentric arc-bearing entities sharing an arc segment, or two contact pads flush against each other would trip this bug silently.

## Design

### Change

Replace the cut block currently at [cad_occ.py:485-509](../../../meshwell/cad_occ.py#L485-L509):

```python
tool_compound = TopoDS_Compound()
cb = BRep_Builder()
cb.MakeCompound(tool_compound)
for ts in tool_shapes:
    cb.Add(tool_compound, ts)

new_shapes: list[TopoDS_Shape] = []
for s in labeled.shapes:
    result = s
    try:
        cut_op = BRepAlgoAPI_Cut(s, tool_compound)
        cut_op.SetFuzzyValue(self.cut_fuzzy_value)
        cut_op.SetRunParallel(self.n_threads > 1)
        cut_op.Build()
        result = cut_op.Shape()
    except Exception as e:
        ...
```

with a prefused variant:

```python
# Prefuse all surviving tools into one shape, then a single Cut per
# substrate sub-shape. Replaces the previous batched compound cut,
# which corrupts the substrate when two tools share a face/edge
# (tangent). See docs/superpowers/specs/2026-05-19-cad-occ-prefused-
# cut-safety-hotfix-design.md and scripts/bench_cut_strategy_sweep.py.
if len(tool_shapes) == 1:
    fused_tools = tool_shapes[0]
else:
    fuse_op = BOPAlgo_BOP()
    fuse_op.SetOperation(BOPAlgo_Operation.BOPAlgo_FUSE)
    fuse_op.AddArgument(tool_shapes[0])
    for ts in tool_shapes[1:]:
        fuse_op.AddTool(ts)
    fuse_op.SetFuzzyValue(self.cut_fuzzy_value)
    fuse_op.SetRunParallel(self.n_threads > 1)
    fuse_op.Perform()
    fused_tools = fuse_op.Shape()

new_shapes: list[TopoDS_Shape] = []
for s in labeled.shapes:
    result = s
    try:
        cut_op = BRepAlgoAPI_Cut(s, fused_tools)
        cut_op.SetFuzzyValue(self.cut_fuzzy_value)
        cut_op.SetRunParallel(self.n_threads > 1)
        cut_op.Build()
        result = cut_op.Shape()
    except Exception as e:  # pragma: no cover -- defensive
        print(
            f"Warning: BRepAlgoAPI_Cut failed for entity "
            f"{orig_idx}: {e}"
        )
    if result is not None:
        new_shapes.extend(self._unwrap_shape(result, labeled.dim))
labeled.shapes = new_shapes
```

Imports update: add `from OCP.BOPAlgo import BOPAlgo_BOP, BOPAlgo_Operation` (currently only `BOPAlgo_Builder` is imported). Drop the now-unused `BRep_Builder` and `TopoDS_Compound` imports — the compound construction is gone.

### Why prefused works where compound fails

`BRepAlgoAPI_Cut` walks each face of the tools to subtract it from the substrate. When tools sit in a `TopoDS_Compound`, OCC sees them as independent sub-shapes; it intersects substrate vs each tool face. For tangent tools, two adjacent tool faces sit on the same plane with the same boundary edge — OCC's pair-wise BOP planner gets confused (the edge belongs to two faces simultaneously) and either splits the substrate along that phantom edge (`s > 1`) or eliminates the whole substrate (`v = 0`).

`BOPAlgo_BOP` with Fuse operation runs the general-fuse algorithm on the tools first, which **merges the shared boundary** between tangent tools into a single edge owned by one solid. The resulting fused solid has clean topology: one outer boundary, no internal duplicate edges. The subsequent Cut sees one tool with well-defined faces. No confusion, no corruption.

For disjoint tools, the fuse is geometrically a no-op but still incurs the BOPAlgo setup cost — that's where the 20-30% regression comes from. For tangent tools, the fuse pays for itself many times over (it's actively required for correctness).

### N=1 fast-path

When there's exactly one tool, prefusing is pure overhead. The code short-circuits to `fused_tools = tool_shapes[0]` and proceeds directly to the Cut. Bench confirms this matches the sequential strategy timing at N=1.

### Public API

No change. `cad_occ()`, `process_entities()`, `process_entities_cut_only()` signatures unchanged. The `_polyprism_fast_overlap` and `_shapes_actually_overlap` gates are unchanged and remain load-bearing for filtering volume-disjoint tools out of the fuse input.

### Configuration

No new config flags. The existing `cut_fuzzy_value` is applied to both the fuse and the cut (same value for both — they both should agree on what counts as touching).

## Testing

### Regression

The existing [`tests/test_cad_occ_batched_compound_cut.py`](../../../tests/test_cad_occ_batched_compound_cut.py) must continue to pass — its 4 tests cover the disjoint-pillars scene. Expected: identical SOLID counts and volumes within 0.5% (the prefused strategy is correctness-equivalent to compound for disjoint cases; bench data confirms).

### New tangent-tool safety test

Add `tests/test_cad_occ_prefused_cut_tangent_safety.py` with three parametrized scenes proving the bug is fixed:

1. **Two tangent square tools on a square substrate** (N=2 reproduction of the bench failure). Assert substrate stays exactly 1 SOLID with vol strictly less than the uncut substrate vol (i.e. material was actually removed) and vol > half the uncut substrate vol (i.e. it didn't vanish).
2. **Ten tangent square tools on a square substrate** (N=10 reproduction of the empty-result variant). Assert substrate stays exactly 1 SOLID, vol > half of uncut substrate vol.
3. **Three tangent disc tools on a disc substrate** (N=3 tangent disc-vs-disc, where compound produced s=3,v=312.2 — substrate split with no material removed). Assert substrate stays exactly 1 SOLID with vol strictly less than the uncut substrate vol.

The "exactly 1 SOLID" check catches the split-substrate failure; the "vol > half" check catches the vanish-substrate failure; the "vol strictly less than uncut" check catches the bug variant where the split happens without any material being removed.

Each scene constructs the entities directly and runs `CAD_OCC().process_entities_cut_only()`. The substrate's `physical_name` matches the entity defined in the test; volume and solid-count are extracted via the same `_solid_count` / `_volume` helpers already used in the existing regression test.

### Perf evidence

Re-run the cut-strategy sweep bench unmodified — it always tests both strategies on its own copies, independent of production code. Then run the existing dense bench to record the new production wall-clock; expect substrate-vs-20-bodies cut cascade time to increase from ~90 ms to ~115 ms (still ~3× the original sequential ~340 ms).

The implementation commit message should include before/after numbers for the regression test wall-clock and the dense bench n=20 cut time.

## Out of scope (deferred to Layer 2)

- Cross-entity AABB-DAG parallelism via `ThreadPoolExecutor`. The Layer 2 brainstorm will design how to recover the ≈20% prefused regression on disjoint-heavy scenes by running independent entities' cuts in parallel.
- Tangent-tool detection + selective compound (a "fast path that drops to prefused only when needed"). This would require a tangency check, which is its own non-trivial OCC computation — the bench shows prefused's overhead is small enough that the gating isn't worth its complexity.
- Switching back to compound under an opt-in flag. Anyone needing the extra speed today can install the future Layer 2.

## Risks

- **The 20-30 % regression is unacceptable on a critical scene.** Unknown — no production scene benchmarks exist beyond the dense substrate test. Mitigation: dense bench shows the regression is bounded; if a real workload trips an unacceptable slowdown the user can request Layer 2 priority.
- **`BOPAlgo_BOP` with `Fuse` operation behaves differently than expected on edge cases** (e.g., a single tool wrapped in fuse is a no-op but the N=1 fast-path already sidesteps that). The new tangent-safety test guards against this; the existing regression test guards the disjoint case.
- **Other code paths that read `_shapes_actually_overlap` semantics change implicitly.** They don't — the filter still rejects volume-disjoint pairs and admits tangent ones; the change is downstream of the filter.
