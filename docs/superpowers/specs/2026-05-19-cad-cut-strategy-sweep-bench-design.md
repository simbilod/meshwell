# cad_occ: Cut Strategy Sweep Benchmark

**Status:** approved (2026-05-19), pending implementation
**Author:** simbilod (with Claude)
**Related:**
- Prior optimization that motivated this bench: [`2026-05-19-cad-occ-batched-compound-cut-design.md`](2026-05-19-cad-occ-batched-compound-cut-design.md)
- Six additional batching candidates flagged for follow-up: see the explore-agent report in conversation history (PolyPrism/PolySurface interior cuts and fuse cascades, polyline fuse cascade, gmsh-backend polyprism interior cuts)

## Summary

A perf-investigation script that sweeps `BRepAlgoAPI_Cut` batching strategies (`sequential`, `compound`, `prefused`) across four geometric axes (tool count, substrate complexity, tool complexity, tool-tool overlap pattern) and prints a verdict table. The output answers: **for each (substrate, tool, overlap) regime, which strategy is fastest at which N?** The verdict directly informs how each of the six remaining batching candidates should be implemented.

## Motivation

Our prior cut-cascade change replaced a sequential per-tool `BRepAlgoAPI_Cut` loop with a single batched `Cut(substrate, Compound(tools))` and delivered a 3.6× speedup on the dense substrate-vs-N-bodies scene. The win was concentrated in the "many simple tools vs one large substrate" regime, which dominates that bench. We have six remaining candidate sites for similar batching, but they sit in different geometric regimes:

- PolyPrism / PolySurface interior cuts: one face vs a handful of small holes
- PolyPrism / PolyLine / PolySurface fuse cascades: small N over varied face counts

Three open questions:
- Is there a small-N crossover where `compound` loses to `sequential` due to setup overhead?
- For tools that share faces (arc-bearing prisms, tangent tools), does pre-fusing tools win over compound?
- Does the answer change with substrate complexity (4-vert square vs 128-vert annulus)?

We need data, not intuition, before implementing the 6 candidates with the wrong strategy.

## Design

### Strategies compared

| Label | Implementation sketch | Hypothesised win regime |
|---|---|---|
| `sequential` | `for ts in tools: result = Cut(result, ts).Shape()` (with `Build` + `SetFuzzyValue`) | very small N; substrate-light tool-heavy work where setup amortisation doesn't matter |
| `compound` | `Cut(s, TopoDS_Compound(tools))` (current production code) | moderate-to-large N with substrate-dominated work |
| `prefused` | `fused = BOPAlgo_Builder(args=tools, op=Fuse).Shape(); Cut(s, fused)` — one N-way Fuse pass, then one Cut. (For N=1, `prefused` falls back to `Cut(s, tools[0])` since there is nothing to fuse.) | tools that share faces/edges (arcs, tangent boundaries) where Fuse collapses shared topology |

`BRepAlgoAPI_Splitter` and per-cut `SetRunParallel(True)` are deliberately excluded — both already characterised as wrong/harmful by the prior spike.

### Sampling axes

A full 4-axis grid is ~100 cells. The bench samples ~72 cells targeting the regions where the answer is likely to be non-obvious:

| Axis | Values | Rationale |
|---|---|---|
| `N` (tool count per substrate) | 1, 2, 3, 5, 10, 20 | Finds the crossover N where `compound` overtakes `sequential`. |
| `substrate_complexity` | square (4 verts), disc (32 verts), annulus (128 verts) | Covers "simple substrate / complex substrate" axis; annulus is the worst-case face count we care about. |
| `tool_complexity` | square (4 verts), disc (32 verts) | Tools are typically simpler than substrates in production. Two values are enough to detect a trend. |
| `tool_tool_overlap` | `disjoint`, `tangent` | Disjoint = pillars in a grid. Tangent = tools sharing a face/edge (e.g. abutting pillars). `overlapping` (volume-overlapping) is intentionally skipped because the upstream `_shapes_actually_overlap` gate rejects volume-disjoint AABB-overlapping pairs and admits tangent pairs, so tangent is the operationally-relevant edge case. |

Total cells: 6 × 3 × 2 × 2 = 72. Each cell runs 3 strategies × 3 timing repetitions = 9 measurements. Estimated wall-clock at typical per-op costs of 50–500 ms: 5–10 minutes.

### Scene construction

A `build_sweep_scene(n_tools, substrate_complexity, tool_complexity, tool_tool_overlap) -> (substrate_shape, list[tool_shapes])` factory produces the inputs for one cell. The substrate is always a single axis-aligned extrusion at z=[-1, 0] sized large enough to contain the tools. Tools are placed at z=[1.0, 1.5] in a regular grid; `disjoint` uses a cell pitch large enough for an inter-tool gap, `tangent` uses a cell pitch equal to the tool width so tools share faces.

Tool placement uses the same metal-pillar layout as `scripts/bench_cut_strategies_dense.py::build_dense_scene` but with parameterised polygon shapes:

- `square`: 4-vertex unit square primitive
- `disc`: 32-side regular polygon (matches typical photonic device discretisation)
- `annulus`: 128-vertex outer ring + 128-vertex inner ring (worst-case complexity in our test fixtures)

### Measurement

For each cell × strategy:

1. Build the scene fresh (avoids `prepare_entities` in-place mutation issues).
2. Instantiate the substrate and tool OCC shapes via the entity layer (mirrors production code paths).
3. Time the strategy's cut block in isolation (no fragment pass) using `time.perf_counter()`.
4. Repeat 3 times; report the minimum (least machine-noise-affected).
5. Verify correctness: post-cut substrate has exactly 1 SOLID and volume within 0.5% of the analytic expected. Mark the cell as `WRONG` if not.

### Output

Plain-text tables, one per `(substrate, tool, overlap)` cell, columns = strategies, rows = N, values = milliseconds with the fastest strategy in each row marked with `*`. Then one summary table at the end ranking strategies by win-count per regime.

Example skeleton (numbers are illustrative):

```
=== substrate=disc(32) tool=square(4) overlap=disjoint ===
  N  sequential  compound   prefused
  1       4.2ms*     5.1ms      6.8ms
  3      12.5ms      9.2ms*    14.1ms
 10      45.1ms     18.3ms*    22.9ms

=== Summary: wins per regime (sample shape; cell values = win-count) ===
                     sequential  compound   prefused
substrate-light            3         4         1
substrate-heavy            0        12         0
tools-tangent              0         5         7
```

(The numbers in the summary are illustrative — actual values are filled in by the bench run.)

No CSV/JSON export — this is a one-shot design question, not a perf-tracking dashboard. If we later need a perf-regression gate the data and structure can be lifted.

### File layout

| File | Action | Responsibility |
|---|---|---|
| `scripts/bench_cut_strategy_sweep.py` | Create | Sweep harness: scene builder, strategy runners, measurement loop, reporter. ~250 lines. |
| `scripts/bench_cut_strategies.py` | Reuse | Imports `cut_baseline`, `cut_f_cut_compound`, and the volume/solid-count helpers. |
| `scripts/bench_cut_strategies_dense.py` | Untouched | Stays as-is — focused dense-scene comparison, used as basis of the regression test. |

The `prefused` strategy is implemented in the new file (not in the existing `bench_cut_strategies.py`), since this is the first time we're introducing it.

### Public API surface affected

None. This is a scripts/-only addition that produces console output. No changes to `meshwell/`, no changes to existing tests, no changes to existing bench scripts. The verdict it produces is read by humans (and by future spec docs that cite it) to inform the implementation strategy for the 6 remaining batching candidates.

## Out of scope

- Implementing any of the 6 batching candidates. The bench informs those decisions; each candidate is a separate spec.
- Full-pipeline (instantiate + cut + fragment) timing. Already established that cut is ~90% of CAD pipeline cost on dense scenes.
- Memory / footprint measurement.
- Setting up a CI perf-regression gate.
- Cross-machine portability. Numbers are local; the rankings are what matter.

## Risks

- **Per-op cost too small to measure reliably.** At N=1 with a 4-vert square the cut may run in <1 ms — Python timing overhead dominates. Mitigation: 3-repetition `min` aggregation. If still noisy, expand to 5 repetitions or skip the N=1 cell explicitly.
- **Pre-fusing tools that don't actually overlap is wasted work.** Expected. The point is to measure how wasted — if `prefused` is meaningfully slower than `compound` on disjoint tools, the verdict is "don't pre-fuse unless tools share topology."
- **The strategic-sampling axes might miss the cell where the answer flips.** Mitigation: if any verdict row looks suspiciously close, run additional manual samples in that region. The bench is a script meant to be edited and re-run, not a one-shot artefact.
- **OCC version differences.** Results are local-machine, local-OCC. Documented as a caveat in the bench's docstring. The qualitative rule-of-thumb derived from the bench should be robust across reasonable OCC versions.
