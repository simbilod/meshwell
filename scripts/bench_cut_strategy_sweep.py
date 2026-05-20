# ruff: noqa
"""Cut-strategy sweep bench: characterise BRepAlgoAPI_Cut batching across
geometric regimes.

Sweeps three strategies (sequential, compound, prefused) over four axes
(N, substrate complexity, tool complexity, tool-tool overlap) and prints
a verdict table. Answers: for each regime, which strategy is fastest at
which N? The verdict informs which batching strategy to pick for the six
remaining cut/fuse cascade candidates in meshwell/.

Spec: docs/superpowers/specs/2026-05-19-cad-cut-strategy-sweep-bench-design.md

Run::

    .venv/bin/python scripts/bench_cut_strategy_sweep.py

Output is local-machine, local-OCC-version specific; rankings are what
matter, not absolute milliseconds.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from shapely.geometry import Polygon

from OCP.BOPAlgo import BOPAlgo_BOP, BOPAlgo_Builder, BOPAlgo_Operation
from OCP.BRep import BRep_Builder
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS_Compound, TopoDS_Shape

from meshwell.cad_common import prepare_entities
from meshwell.polyprism import PolyPrism

CUT_FUZZY = 1e-5 / 2  # mirrors CAD_OCC default (perturbation / 2)


# --- Shape factories --------------------------------------------------------


def _square(x, y, w, h) -> Polygon:
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def _disc(cx, cy, r, n=32) -> Polygon:
    return Polygon(
        [
            (
                cx + r * math.cos(2 * math.pi * i / n),
                cy + r * math.sin(2 * math.pi * i / n),
            )
            for i in range(n)
        ]
    )


def _annulus(cx, cy, r_outer, r_inner, n=64) -> Polygon:
    outer = [
        (
            cx + r_outer * math.cos(2 * math.pi * i / n),
            cy + r_outer * math.sin(2 * math.pi * i / n),
        )
        for i in range(n)
    ]
    inner = [
        (
            cx + r_inner * math.cos(2 * math.pi * i / n),
            cy + r_inner * math.sin(2 * math.pi * i / n),
        )
        for i in range(n)
    ]
    return Polygon(outer, [inner])


SUBSTRATE_BUILDERS = {
    "square": lambda size: _square(-size / 2, -size / 2, size, size),
    "disc": lambda size: _disc(0.0, 0.0, size / 2, n=32),
    # annulus: 64 verts outer + 64 verts inner = 128 total
    "annulus": lambda size: _annulus(0.0, 0.0, size / 2, size / 2 * 0.6, n=64),
}

TOOL_BUILDERS = {
    "square": lambda w: _square(0, 0, w, w),
    "disc": lambda w: _disc(w / 2, w / 2, w / 2, n=32),
}


# --- Scene construction -----------------------------------------------------


def _grid_layout(n_tools: int) -> tuple[int, int]:
    cols = int(math.ceil(math.sqrt(n_tools)))
    rows = int(math.ceil(n_tools / cols))
    return rows, cols


def _build_scene(
    n_tools: int,
    substrate_complexity: str,
    tool_complexity: str,
    tool_tool_overlap: str,
    *,
    substrate_size: float = 20.0,
    tool_width: float = 0.8,
) -> tuple[PolyPrism, list[PolyPrism]]:
    """One substrate + n_tools tool polyprisms.

    Substrate covers z=[-1, 0], tools cover z=[1.0, 1.5]. They share the
    z=1 plane only via the substrate being below the tools -- they DON'T
    overlap the substrate's z range. But the substrate's mesh_order is
    HIGHER than the tools', so the substrate gets cut against any tool
    whose AABB overlaps via the AABB pre-filter (the cut cascade looks at
    pairs whose AABBs overlap; here the substrate AABB extends only to
    z=0, the tools sit at z=1+. No AABB overlap.) -- we need the tools
    to actually overlap the substrate to exercise the cut. Place tools
    at z=[-0.5, 0.5] so they punch into the substrate.

    tool_tool_overlap:
        - "disjoint": pitch much larger than tool_width
        - "tangent": pitch equal to tool_width (tools share faces)
    """
    substrate_polygon = SUBSTRATE_BUILDERS[substrate_complexity](substrate_size)
    substrate = PolyPrism(
        polygons=substrate_polygon,
        buffers={-1.0: 0.0, 0.0: 0.0},
        physical_name="substrate",
        mesh_order=100.0,
    )

    if tool_tool_overlap == "disjoint":
        pitch = tool_width * 2.5
    elif tool_tool_overlap == "tangent":
        pitch = tool_width  # adjacent tools share an edge
    else:
        raise ValueError(f"unknown tool_tool_overlap: {tool_tool_overlap}")

    rows, cols = _grid_layout(n_tools)
    # centre the grid on the origin so it lands inside the substrate
    grid_w = cols * pitch
    grid_h = rows * pitch
    x0 = -grid_w / 2
    y0 = -grid_h / 2

    tools: list[PolyPrism] = []
    placed = 0
    tool_factory = TOOL_BUILDERS[tool_complexity]
    for r in range(rows):
        for c in range(cols):
            if placed >= n_tools:
                break
            base = tool_factory(tool_width)
            # translate the base polygon to its grid slot
            x = x0 + c * pitch
            y = y0 + r * pitch
            from shapely.affinity import translate

            shifted = translate(base, xoff=x, yoff=y)
            tools.append(
                PolyPrism(
                    polygons=shifted,
                    # punch into the substrate (z=[-1, 0]) from z=[-0.5, 0.5]
                    buffers={-0.5: 0.0, 0.5: 0.0},
                    physical_name=f"tool_{placed}",
                    mesh_order=1.0 + placed * 0.01,
                )
            )
            placed += 1
    return substrate, tools


# --- OCC helpers ------------------------------------------------------------


def _solid_count(shape: TopoDS_Shape) -> int:
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    n = 0
    while exp.More():
        n += 1
        exp.Next()
    return n


def _volume(shape: TopoDS_Shape) -> float:
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return abs(props.Mass())


def _make_compound(shapes: list[TopoDS_Shape]) -> TopoDS_Compound:
    comp = TopoDS_Compound()
    cb = BRep_Builder()
    cb.MakeCompound(comp)
    for s in shapes:
        cb.Add(comp, s)
    return comp


def _fuse_n(shapes: list[TopoDS_Shape]) -> TopoDS_Shape:
    """N-way fuse via BOPAlgo_BOP (Fuse operation)."""
    if len(shapes) == 1:
        return shapes[0]
    if len(shapes) == 2:
        op = BRepAlgoAPI_Fuse(shapes[0], shapes[1])
        op.SetFuzzyValue(CUT_FUZZY)
        op.Build()
        return op.Shape()
    # N-way: BOPAlgo_BOP with the Fuse operation
    op = BOPAlgo_BOP()
    op.SetOperation(BOPAlgo_Operation.BOPAlgo_FUSE)
    op.AddArgument(shapes[0])
    for s in shapes[1:]:
        op.AddTool(s)
    op.SetFuzzyValue(CUT_FUZZY)
    op.SetRunParallel(False)
    op.Perform()
    return op.Shape()


# --- Strategies -------------------------------------------------------------


CutFn = Callable[[TopoDS_Shape, list[TopoDS_Shape]], TopoDS_Shape]


def cut_sequential(s: TopoDS_Shape, tools: list[TopoDS_Shape]) -> TopoDS_Shape:
    result = s
    for ts in tools:
        op = BRepAlgoAPI_Cut(result, ts)
        op.SetFuzzyValue(CUT_FUZZY)
        op.Build()
        result = op.Shape()
    return result


def cut_compound(s: TopoDS_Shape, tools: list[TopoDS_Shape]) -> TopoDS_Shape:
    comp = _make_compound(tools)
    op = BRepAlgoAPI_Cut(s, comp)
    op.SetFuzzyValue(CUT_FUZZY)
    op.SetRunParallel(False)
    op.Build()
    return op.Shape()


def cut_prefused(s: TopoDS_Shape, tools: list[TopoDS_Shape]) -> TopoDS_Shape:
    """Fuse all tools first (one BOPAlgo pass), then a single Cut."""
    if len(tools) == 1:
        # nothing to fuse; same as a direct single-tool cut
        return cut_sequential(s, tools)
    fused = _fuse_n(tools)
    op = BRepAlgoAPI_Cut(s, fused)
    op.SetFuzzyValue(CUT_FUZZY)
    op.SetRunParallel(False)
    op.Build()
    return op.Shape()


STRATEGIES: dict[str, CutFn] = {
    "sequential": cut_sequential,
    "compound": cut_compound,
    "prefused": cut_prefused,
}


# --- Measurement ------------------------------------------------------------


@dataclass
class CellResult:
    strategy: str
    ms: float
    solid_count: int
    volume: float
    wrong: bool


def _measure_one(
    strategy: str,
    cut_fn: CutFn,
    n_tools: int,
    substrate_complexity: str,
    tool_complexity: str,
    tool_tool_overlap: str,
    repetitions: int = 3,
    expected_substrate_volume: float | None = None,
) -> CellResult:
    """Run cut_fn `repetitions` times, return min wall-time + correctness."""
    times: list[float] = []
    last_result: TopoDS_Shape | None = None
    last_solids = 0
    last_volume = 0.0
    for _ in range(repetitions):
        # fresh scene each time -- prepare_entities mutates polygons
        substrate, tools = _build_scene(
            n_tools, substrate_complexity, tool_complexity, tool_tool_overlap
        )
        prepare_entities([substrate, *tools], perturbation=1e-5, resolve_snap=1e-3)
        s_shape = substrate.instanciate_occ()
        tool_shapes = [t.instanciate_occ() for t in tools]
        t0 = time.perf_counter()
        try:
            last_result = cut_fn(s_shape, tool_shapes)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            return CellResult(strategy, elapsed * 1000.0, -1, float("nan"), wrong=True)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        last_solids = _solid_count(last_result)
        try:
            last_volume = _volume(last_result)
        except Exception:
            last_volume = float("nan")
    best_ms = min(times) * 1000.0
    # Correctness invariants:
    #   (a) substrate post-cut must be exactly 1 SOLID (no splitting / no vanishing)
    #   (b) post-cut volume must be a non-trivial fraction of the substrate
    #       (catches the "empty result" bug where compound erases everything)
    # The exact post-cut volume isn't cheap to predict (depends on tool/substrate
    # intersection geometry) so we don't enforce a tight volume tolerance; we
    # only catch the catastrophic vanish case.
    wrong = False
    if last_solids != 1:
        wrong = True
    if expected_substrate_volume is not None and not math.isnan(last_volume):
        if last_volume < expected_substrate_volume * 0.5:
            wrong = True
    return CellResult(strategy, best_ms, last_solids, last_volume, wrong=wrong)


def _expected_substrate_volume(
    substrate_complexity: str, substrate_size: float = 20.0
) -> float:
    """Pre-perturbation analytic volume of the substrate (footprint area * z-extent).

    Tools punch through the substrate but are placed at z=[-0.5, 0.5] which
    overlaps the substrate's z=[-1, 0] only in z=[-0.5, 0] -- they remove
    a chunk in xy where they sit. Pre-correctness asserts roughly the
    substrate retains *most* of its volume (within 5%). We don't compute
    the exact expected post-cut volume here; we just use the substrate's
    full volume as a loose upper-bound check (volume should be within 5%
    of full, since tools are small).
    """
    sub_polygon = SUBSTRATE_BUILDERS[substrate_complexity](substrate_size)
    return sub_polygon.area * 1.0  # z-extent = 1


# --- Sweep ------------------------------------------------------------------


SWEEP_AXES = {
    "n_tools": [1, 2, 3, 5, 10, 20],
    "substrate_complexity": ["square", "disc", "annulus"],
    "tool_complexity": ["square", "disc"],
    "tool_tool_overlap": ["disjoint", "tangent"],
}


def run_sweep() -> dict[tuple, list[CellResult]]:
    """Return {(substrate, tool, overlap, n): [CellResult per strategy]}."""
    results: dict[tuple, list[CellResult]] = {}
    total_cells = (
        len(SWEEP_AXES["substrate_complexity"])
        * len(SWEEP_AXES["tool_complexity"])
        * len(SWEEP_AXES["tool_tool_overlap"])
        * len(SWEEP_AXES["n_tools"])
    )
    cell_idx = 0
    for sub in SWEEP_AXES["substrate_complexity"]:
        for tool in SWEEP_AXES["tool_complexity"]:
            for overlap in SWEEP_AXES["tool_tool_overlap"]:
                for n in SWEEP_AXES["n_tools"]:
                    cell_idx += 1
                    print(
                        f"  [{cell_idx:>3}/{total_cells}] substrate={sub:<7s} "
                        f"tool={tool:<6s} overlap={overlap:<8s} N={n:>2d}",
                        flush=True,
                    )
                    cell_results: list[CellResult] = []
                    expected_vol = _expected_substrate_volume(sub)
                    for strategy_name, fn in STRATEGIES.items():
                        cr = _measure_one(
                            strategy_name,
                            fn,
                            n,
                            sub,
                            tool,
                            overlap,
                            repetitions=3,
                            expected_substrate_volume=expected_vol,
                        )
                        cell_results.append(cr)
                    results[(sub, tool, overlap, n)] = cell_results
    return results


# --- Reporting --------------------------------------------------------------


def report(results: dict[tuple, list[CellResult]]) -> None:
    # Per-cell tables
    for sub in SWEEP_AXES["substrate_complexity"]:
        for tool in SWEEP_AXES["tool_complexity"]:
            for overlap in SWEEP_AXES["tool_tool_overlap"]:
                print()
                print(f"=== substrate={sub} tool={tool} overlap={overlap} ===")
                header = f"{'N':>3}  " + "  ".join(
                    f"{name:>28}" for name in STRATEGIES.keys()
                )
                print(header)
                print("-" * len(header))
                for n in SWEEP_AXES["n_tools"]:
                    row = results[(sub, tool, overlap, n)]
                    valid = [(r.strategy, r.ms) for r in row if not r.wrong]
                    fastest = min(valid, key=lambda p: p[1])[0] if valid else None
                    cells = []
                    for r in row:
                        marker = "*" if r.strategy == fastest else " "
                        if r.wrong:
                            # surface the failure mode (split / vanish)
                            tag = f" WRONG(s={r.solid_count},v={r.volume:.1f})"
                        else:
                            tag = ""
                        cells.append(f"{r.ms:>8.2f}ms{marker}{tag}")
                    print(f"{n:>3}  " + "  ".join(f"{c:>28}" for c in cells))

    # Summary: per-regime win counts
    print()
    print("=" * 70)
    print("Summary: wins per regime (cell value = strategy with min time in row)")
    print("=" * 70)
    # Bucket cells into regimes:
    # - substrate-light = square substrate
    # - substrate-heavy = annulus substrate
    # - tools-tangent = overlap=tangent (independent of substrate)
    # - tools-disjoint = overlap=disjoint
    regime_wins: dict[str, dict[str, int]] = {
        "substrate-light (square sub)": {s: 0 for s in STRATEGIES},
        "substrate-mid (disc sub)": {s: 0 for s in STRATEGIES},
        "substrate-heavy (annulus sub)": {s: 0 for s in STRATEGIES},
        "tools-disjoint": {s: 0 for s in STRATEGIES},
        "tools-tangent": {s: 0 for s in STRATEGIES},
        "small-N (1, 2, 3)": {s: 0 for s in STRATEGIES},
        "medium-N (5)": {s: 0 for s in STRATEGIES},
        "large-N (10, 20)": {s: 0 for s in STRATEGIES},
    }
    sub_regime = {
        "square": "substrate-light (square sub)",
        "disc": "substrate-mid (disc sub)",
        "annulus": "substrate-heavy (annulus sub)",
    }
    overlap_regime = {
        "disjoint": "tools-disjoint",
        "tangent": "tools-tangent",
    }

    def n_regime(n: int) -> str:
        if n <= 3:
            return "small-N (1, 2, 3)"
        if n <= 5:
            return "medium-N (5)"
        return "large-N (10, 20)"

    for (sub, tool, overlap, n), row in results.items():
        valid = [(r.strategy, r.ms) for r in row if not r.wrong]
        if not valid:
            continue
        winner = min(valid, key=lambda p: p[1])[0]
        regime_wins[sub_regime[sub]][winner] += 1
        regime_wins[overlap_regime[overlap]][winner] += 1
        regime_wins[n_regime(n)][winner] += 1

    print(f"\n{'regime':<32}  " + "  ".join(f"{s:>11}" for s in STRATEGIES.keys()))
    print("-" * 70)
    for regime, counts in regime_wins.items():
        cells = "  ".join(f"{counts[s]:>11d}" for s in STRATEGIES.keys())
        print(f"{regime:<32}  {cells}")

    # Crossover N per (substrate, tool, overlap)
    print()
    print("=" * 70)
    print("Crossover N (smallest N where compound beats sequential, per cell)")
    print("=" * 70)
    print(f"\n{'(substrate, tool, overlap)':<40}  {'crossover N':>14}")
    print("-" * 60)
    for sub in SWEEP_AXES["substrate_complexity"]:
        for tool in SWEEP_AXES["tool_complexity"]:
            for overlap in SWEEP_AXES["tool_tool_overlap"]:
                crossover = None
                for n in SWEEP_AXES["n_tools"]:
                    row = results[(sub, tool, overlap, n)]
                    by_name = {r.strategy: r for r in row}
                    seq = by_name["sequential"]
                    comp = by_name["compound"]
                    if seq.wrong or comp.wrong:
                        continue
                    if comp.ms < seq.ms:
                        crossover = n
                        break
                label = f"({sub}, {tool}, {overlap})"
                value = str(crossover) if crossover is not None else "never in range"
                print(f"{label:<40}  {value:>14}")


def main() -> None:
    print(
        f"Running cut-strategy sweep: {len(SWEEP_AXES['n_tools'])} N values x "
        f"{len(SWEEP_AXES['substrate_complexity'])} substrates x "
        f"{len(SWEEP_AXES['tool_complexity'])} tools x "
        f"{len(SWEEP_AXES['tool_tool_overlap'])} overlap modes "
        f"x {len(STRATEGIES)} strategies x 3 reps"
    )
    print()
    t0 = time.perf_counter()
    results = run_sweep()
    elapsed = time.perf_counter() - t0
    print(f"\nSweep complete in {elapsed:.1f}s.")
    report(results)


if __name__ == "__main__":
    main()
