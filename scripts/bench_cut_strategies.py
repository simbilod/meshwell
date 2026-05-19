# ruff: noqa
"""Benchmark spike: compare BRepAlgo cut strategies on the photonic_stack scene.

Mirrors :meth:`meshwell.cad_occ.CAD_OCC.process_entities_cut_only` but
swaps the inner cut step for four strategies and records timing +
correctness (SOLID count, total volume) for each entity that gets cut.

Strategies
----------
1. ``baseline``        : current sequential ``BRepAlgoAPI_Cut(result, ts)`` loop.
2. ``a2_runparallel``  : same loop, with ``cut_op.SetRunParallel(True)`` per call.
3. ``a1_splitter``     : ``BRepAlgoAPI_Splitter`` with all tools in a
                         ``TopTools_ListOfShape`` Arguments + Tools, parallel.
4. ``f_cut_compound``  : single ``BRepAlgoAPI_Cut(s, compound_of_tools)``
                         (the previously-reported failure mode).

Run::

    .venv/bin/python scripts/bench_cut_strategies.py
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from os import cpu_count
from typing import Any

from shapely.geometry import Polygon

from OCP.BRep import BRep_Builder
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Splitter
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS_Compound, TopoDS_Shape
from OCP.TopTools import TopTools_ListOfShape

from meshwell.cad_common import prepare_entities
from meshwell.cad_occ import CAD_OCC, OCCLabeledEntity
from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec


# --- Scene -------------------------------------------------------------------


def _square(x: float, y: float, w: float, h: float) -> Polygon:
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def _disc(cx: float, cy: float, r: float, n: int = 32) -> Polygon:
    return Polygon(
        [
            (
                cx + r * math.cos(2 * math.pi * i / n),
                cy + r * math.sin(2 * math.pi * i / n),
            )
            for i in range(n)
        ]
    )


def _annulus(cx, cy, r_outer, r_inner, n=32) -> Polygon:
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


def build_photonic_stack_entities() -> list[Any]:
    """Same entity set as ``scripts/demo_structured.py::scene_photonic_stack``."""
    big_xy = _square(-5, -5, 10, 10)
    return [
        PolyPrism(
            polygons=big_xy,
            buffers={-1.0: 0.0, 0.0: 0.0},
            physical_name="substrate",
            mesh_order=10.0,
        ),
        PolyPrism(
            polygons=big_xy,
            buffers={0.0: 0.0, 0.5: 0.0},
            physical_name="box",
            mesh_order=9.0,
        ),
        PolyPrism(
            polygons=_disc(0, 0, 1.5, n=32),
            buffers={0.5: 0.0, 1.0: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
            identify_arcs=True,
            physical_name="disc",
            mesh_order=1.0,
        ),
        PolyPrism(
            polygons=_annulus(0, 0, r_outer=3.0, r_inner=2.5, n=32),
            buffers={0.5: 0.0, 1.0: 0.0},
            structured=True,
            resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
            identify_arcs=True,
            physical_name="ring",
            mesh_order=1.0,
        ),
        PolyPrism(
            polygons=_square(-5, -4, 10, 0.8),
            buffers={0.5: 0.0, 1.0: 0.0},
            physical_name="bus",
            mesh_order=5.0,
        ),
        PolyPrism(
            polygons=_square(-3, -4, 6, 0.8),
            buffers={1.0: 0.0, 1.3: 0.0},
            physical_name="contact_a",
            mesh_order=3.0,
        ),
        PolyPrism(
            polygons=big_xy,
            buffers={1.0: 0.0, 2.5: 0.0},
            physical_name="top_clad",
            mesh_order=11.0,
        ),
    ]


# --- Measurement helpers -----------------------------------------------------


def shape_volume(shape: TopoDS_Shape) -> float:
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return abs(props.Mass())


def shape_solid_count(shape: TopoDS_Shape) -> int:
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    n = 0
    while exp.More():
        n += 1
        exp.Next()
    return n


def make_compound(shapes: list[TopoDS_Shape]) -> TopoDS_Compound:
    comp = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(comp)
    for s in shapes:
        builder.Add(comp, s)
    return comp


# --- Cut strategies ----------------------------------------------------------


CutFn = Callable[[TopoDS_Shape, list[TopoDS_Shape], float], TopoDS_Shape]


def cut_baseline(
    s: TopoDS_Shape, tools: list[TopoDS_Shape], fuzzy: float
) -> TopoDS_Shape:
    result = s
    for ts in tools:
        op = BRepAlgoAPI_Cut(result, ts)
        op.SetFuzzyValue(fuzzy)
        op.Build()
        result = op.Shape()
    return result


def cut_a2_runparallel(
    s: TopoDS_Shape, tools: list[TopoDS_Shape], fuzzy: float
) -> TopoDS_Shape:
    result = s
    for ts in tools:
        op = BRepAlgoAPI_Cut(result, ts)
        op.SetFuzzyValue(fuzzy)
        op.SetRunParallel(True)
        op.Build()
        result = op.Shape()
    return result


def cut_a1_splitter(
    s: TopoDS_Shape, tools: list[TopoDS_Shape], fuzzy: float
) -> TopoDS_Shape:
    args = TopTools_ListOfShape()
    args.Append(s)
    tlist = TopTools_ListOfShape()
    for ts in tools:
        tlist.Append(ts)
    op = BRepAlgoAPI_Splitter()
    op.SetArguments(args)
    op.SetTools(tlist)
    op.SetFuzzyValue(fuzzy)
    op.SetRunParallel(True)
    op.Build()
    # Splitter returns the full GF result (substrate parts + tool parts).
    # Keep only the SOLIDs that belong to (i.e. were modified from) the
    # substrate, which is exactly the post-cut substrate. The Splitter
    # docs say its result is the GF result minus pieces inside Tools.
    return op.Shape()


def cut_f_cut_compound(
    s: TopoDS_Shape, tools: list[TopoDS_Shape], fuzzy: float
) -> TopoDS_Shape:
    comp = make_compound(tools)
    op = BRepAlgoAPI_Cut(s, comp)
    op.SetFuzzyValue(fuzzy)
    op.SetRunParallel(True)
    op.Build()
    return op.Shape()


STRATEGIES: dict[str, CutFn] = {
    "baseline": cut_baseline,
    "a2_runparallel": cut_a2_runparallel,
    "a1_splitter": cut_a1_splitter,
    "f_cut_compound": cut_f_cut_compound,
}


# --- Bench harness -----------------------------------------------------------


@dataclass
class PerEntity:
    name: str
    n_tools: int
    n_subshapes: int
    elapsed: float
    solid_count: int
    total_volume: float


@dataclass
class StrategyResult:
    name: str
    per_entity: list[PerEntity] = field(default_factory=list)
    total_time: float = 0.0

    def add(self, p: PerEntity) -> None:
        self.per_entity.append(p)
        self.total_time += p.elapsed


def run_strategy(
    strategy_name: str,
    cut_fn: CutFn,
    entities_list: list[Any],
    point_tolerance: float = 1e-3,
    perturbation: float = 1e-5,
) -> StrategyResult:
    """Run the same instantiate+cut loop as process_entities_cut_only but use cut_fn."""
    processor = CAD_OCC(
        point_tolerance=point_tolerance,
        perturbation=perturbation,
    )

    indexed = list(enumerate(entities_list))
    indexed.sort(
        key=lambda p: (
            p[1].mesh_order if p[1].mesh_order is not None else float("inf"),
            p[0],
        )
    )

    result = StrategyResult(name=strategy_name)
    instantiated: list[OCCLabeledEntity | None] = [None] * len(entities_list)
    for orig_idx, ent in indexed:
        labeled = processor._instantiate_entity_occ(orig_idx, ent)

        obj_bboxes = [
            b for s in labeled.shapes if (b := processor._shape_bbox(s)) is not None
        ]
        tool_shapes: list[TopoDS_Shape] = []
        l_ord = labeled.mesh_order if labeled.mesh_order is not None else float("inf")
        for prev in instantiated:
            if prev is None or prev.dim != labeled.dim:
                continue
            p_ord = prev.mesh_order if prev.mesh_order is not None else float("inf")
            if p_ord >= l_ord:
                continue
            for ts in prev.shapes:
                tb = processor._shape_bbox(ts)
                if tb is None:
                    continue
                if not any(processor._bboxes_overlap(ob, tb) for ob in obj_bboxes):
                    continue
                if not any(
                    processor._shapes_actually_overlap(s, ts) for s in labeled.shapes
                ):
                    continue
                tool_shapes.append(ts)

        elapsed = 0.0
        solid_count = 0
        total_volume = 0.0
        if tool_shapes and labeled.shapes:
            t0 = time.perf_counter()
            new_shapes: list[TopoDS_Shape] = []
            try:
                for s in labeled.shapes:
                    cut_result = cut_fn(s, tool_shapes, processor.cut_fuzzy_value)
                    if cut_result is not None:
                        new_shapes.extend(
                            processor._unwrap_shape(cut_result, labeled.dim)
                        )
            except Exception as e:  # noqa: BLE001 - want diagnostic
                elapsed = time.perf_counter() - t0
                print(f"  [{strategy_name}] {labeled.physical_name} EXCEPTION: {e}")
                instantiated[orig_idx] = labeled
                result.add(
                    PerEntity(
                        name=str(labeled.physical_name),
                        n_tools=len(tool_shapes),
                        n_subshapes=len(labeled.shapes),
                        elapsed=elapsed,
                        solid_count=-1,
                        total_volume=float("nan"),
                    )
                )
                continue
            elapsed = time.perf_counter() - t0
            for ns in new_shapes:
                solid_count += shape_solid_count(ns)
                try:
                    total_volume += shape_volume(ns)
                except Exception:  # noqa: BLE001
                    total_volume = float("nan")
                    break
            labeled.shapes = new_shapes
        else:
            for s in labeled.shapes:
                solid_count += shape_solid_count(s)
                try:
                    total_volume += shape_volume(s)
                except Exception:
                    pass

        result.add(
            PerEntity(
                name=str(labeled.physical_name),
                n_tools=len(tool_shapes),
                n_subshapes=len(labeled.shapes),
                elapsed=elapsed,
                solid_count=solid_count,
                total_volume=total_volume,
            )
        )
        instantiated[orig_idx] = labeled

    return result


def report(results: list[StrategyResult]) -> None:
    print("\n" + "=" * 100)
    print("Per-entity comparison (only entities that had tools to cut against)")
    print("=" * 100)

    entity_names = [p.name for p in results[0].per_entity]
    header = f"{'entity':<14} {'n_tools':>7} {'n_sub':>5} " + " ".join(
        f"{r.name:>30}" for r in results
    )
    print(header)
    print("-" * len(header))

    for i, name in enumerate(entity_names):
        row_tools = results[0].per_entity[i].n_tools
        row_sub = results[0].per_entity[i].n_subshapes
        if row_tools == 0:
            continue
        cells = []
        for r in results:
            p = r.per_entity[i]
            cells.append(
                f"{p.elapsed*1000:7.1f}ms s={p.solid_count:2d} v={p.total_volume:9.4f}"
            )
        print(
            f"{name:<14} {row_tools:>7} {row_sub:>5} "
            + " ".join(f"{c:>30}" for c in cells)
        )

    print("\n" + "=" * 100)
    print("Strategy totals (sum of per-entity cut time)")
    print("=" * 100)
    base = results[0].total_time
    for r in results:
        speedup = base / r.total_time if r.total_time > 0 else float("inf")
        print(
            f"  {r.name:<20} total_cut_time={r.total_time*1000:8.1f} ms  speedup_vs_baseline={speedup:5.2f}x"
        )

    print("\n" + "=" * 100)
    print(
        "Volume-equivalence check vs baseline (per entity, only entities that got cut)"
    )
    print("=" * 100)
    baseline = {p.name: p for p in results[0].per_entity}
    for r in results[1:]:
        print(f"\n  Strategy: {r.name}")
        for p in r.per_entity:
            bp = baseline[p.name]
            if bp.n_tools == 0:
                continue
            if math.isnan(p.total_volume):
                print(f"    {p.name:<14}  FAILED / NaN")
                continue
            dv = p.total_volume - bp.total_volume
            rel = dv / bp.total_volume if bp.total_volume else 0.0
            ds = p.solid_count - bp.solid_count
            flag = "OK " if abs(rel) < 1e-6 and ds == 0 else "DIFF"
            print(
                f"    {p.name:<14}  {flag}  d_vol={dv:+.4e} ({rel*100:+.4f}%)  "
                f"d_solids={ds:+d}  (baseline solids={bp.solid_count}, vol={bp.total_volume:.4f})"
            )


def main() -> None:
    entities = build_photonic_stack_entities()
    # prepare_entities mutates in place; do it once. Then re-instantiate per strategy
    # (each instanciate_occ call rebuilds OCC shapes from the prepared polygons).
    prepare_entities(entities, perturbation=1e-5, resolve_snap=1e-3)

    print(f"CPUs available: {cpu_count()}")
    print(f"Scene: photonic_stack ({len(entities)} entities)")
    print(f"Running {len(STRATEGIES)} strategies...\n")

    results: list[StrategyResult] = []
    for name, fn in STRATEGIES.items():
        print(f"  -> {name}")
        results.append(run_strategy(name, fn, entities))

    report(results)


if __name__ == "__main__":
    main()
