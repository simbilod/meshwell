# ruff: noqa
"""Reproduce the original compound-cut failure and time the full pipeline.

Adds three additional strategies vs bench_cut_strategies_dense.py:

- ``f_legacy_no_build`` - mimics the pre-aa77f21 code: ``BRepAlgoAPI_Cut(s,
  compound)`` followed immediately by ``.Shape()`` with NO explicit ``Build()``
  call and NO fuzzy/parallel settings. Uses an AABB-only pre-filter (no
  ``_shapes_actually_overlap``) -- both differences against the modern code.
- ``f_legacy_with_build`` - AABB-only pre-filter, but WITH ``cut_op.Build()``
  and ``SetFuzzyValue``. Isolates whether the missing Build() was the bug.
- Full-pipeline timing -- runs ``CAD_OCC.process_entities()`` end-to-end so
  we can see the cut cascade's share of total CAD time.
"""
from __future__ import annotations

import math
import time
from typing import Any

from shapely.geometry import Polygon

from OCP.BRep import BRep_Builder
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.TopoDS import TopoDS_Compound, TopoDS_Shape

from meshwell.cad_common import prepare_entities
from meshwell.cad_occ import CAD_OCC, OCCLabeledEntity

import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from bench_cut_strategies import (  # type: ignore
    shape_volume,
    shape_solid_count,
    make_compound,
    cut_baseline,
    cut_a1_splitter,
    cut_f_cut_compound,
    StrategyResult,
    PerEntity,
)
from bench_cut_strategies_dense import build_dense_scene


def cut_f_legacy_no_build(
    s: TopoDS_Shape, tools: list[TopoDS_Shape], fuzzy: float
) -> TopoDS_Shape:
    """Pre-aa77f21 broken pattern: constructor + .Shape() with no .Build()."""
    comp = make_compound(tools)
    cut_op = BRepAlgoAPI_Cut(s, comp)
    return cut_op.Shape()


def cut_f_legacy_with_build(
    s: TopoDS_Shape, tools: list[TopoDS_Shape], fuzzy: float
) -> TopoDS_Shape:
    """AABB-only era but with proper Build() + Fuzzy."""
    comp = make_compound(tools)
    op = BRepAlgoAPI_Cut(s, comp)
    op.SetFuzzyValue(fuzzy)
    op.Build()
    return op.Shape()


def run_strategy_with_filter(
    strategy_name: str,
    cut_fn,
    entities_list: list[Any],
    use_actual_overlap_filter: bool,
    point_tolerance: float = 1e-3,
    perturbation: float = 1e-5,
) -> StrategyResult:
    """Same as run_strategy but lets you turn the _shapes_actually_overlap filter on/off."""
    processor = CAD_OCC(point_tolerance=point_tolerance, perturbation=perturbation)
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
                if use_actual_overlap_filter:
                    if not any(
                        processor._shapes_actually_overlap(s, ts)
                        for s in labeled.shapes
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
                    cr = cut_fn(s, tool_shapes, processor.cut_fuzzy_value)
                    if cr is not None:
                        new_shapes.extend(processor._unwrap_shape(cr, labeled.dim))
            except Exception as e:
                elapsed = time.perf_counter() - t0
                print(f"  [{strategy_name}] {labeled.physical_name} EXCEPTION: {e}")
                instantiated[orig_idx] = labeled
                result.add(
                    PerEntity(
                        str(labeled.physical_name),
                        len(tool_shapes),
                        len(labeled.shapes),
                        elapsed,
                        -1,
                        float("nan"),
                    )
                )
                continue
            elapsed = time.perf_counter() - t0
            for ns in new_shapes:
                solid_count += shape_solid_count(ns)
                try:
                    total_volume += shape_volume(ns)
                except Exception:
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
                str(labeled.physical_name),
                len(tool_shapes),
                len(labeled.shapes),
                elapsed,
                solid_count,
                total_volume,
            )
        )
        instantiated[orig_idx] = labeled
    return result


def short_report(
    results: list[StrategyResult], baseline_name: str = "baseline"
) -> None:
    print(
        f"\n{'strategy':<30} {'total_cut_ms':>14} {'speedup':>8}  per-entity (vol/solids)"
    )
    print("-" * 110)
    base = next(r for r in results if r.name == baseline_name).total_time
    bp_by_name = {
        p.name: p
        for p in next(r for r in results if r.name == baseline_name).per_entity
    }
    for r in results:
        sp = base / r.total_time if r.total_time > 0 else float("inf")
        cells = []
        for p in r.per_entity:
            if p.n_tools == 0:
                continue
            bp = bp_by_name[p.name]
            tag = "OK"
            if math.isnan(p.total_volume):
                tag = "NaN"
            elif bp.total_volume == 0:
                tag = "?"
            elif (
                abs((p.total_volume - bp.total_volume) / bp.total_volume) > 1e-6
                or p.solid_count != bp.solid_count
            ):
                tag = f"DIFF(v={p.total_volume:.4f},s={p.solid_count})"
            cells.append(f"{p.name.strip(chr(39)+'(),'):<10}={tag}")
        print(
            f"{r.name:<30} {r.total_time*1000:>14.1f} {sp:>7.2f}x  {'  '.join(cells)}"
        )


def time_full_pipeline(entities_list: list[Any], label: str) -> None:
    print(f"\n[{label}] Full-pipeline timing (instantiate + cut + fragment):")
    processor = CAD_OCC(point_tolerance=1e-3, perturbation=1e-5)
    # Re-prepare a fresh entity list (prepare_entities mutates in place)
    t_cut0 = time.perf_counter()
    cut_only = processor.process_entities_cut_only(entities_list, progress_bars=False)
    t_cut = time.perf_counter() - t_cut0
    t_frag0 = time.perf_counter()
    processor._fragment_all(cut_only, progress_bars=False)
    t_frag = time.perf_counter() - t_frag0
    print(f"  cut-cascade : {t_cut*1000:7.1f} ms")
    print(f"  fragment    : {t_frag*1000:7.1f} ms")
    print(
        f"  TOTAL       : {(t_cut+t_frag)*1000:7.1f} ms   (cut share = {100*t_cut/(t_cut+t_frag):.1f}%)"
    )


def main() -> None:
    for n in (12, 20):
        print("\n" + "#" * 100)
        print(f"# n_bodies={n}")
        print("#" * 100)

        # cleaner: rebuild for each
        results = []
        for name, fn, mode in [
            ("baseline (modern filter)", cut_baseline, True),
            ("f_cut_compound (modern filter)", cut_f_cut_compound, True),
            ("f_cut_compound (AABB-only filter)", cut_f_cut_compound, False),
            ("f_legacy_no_build (AABB-only filter)", cut_f_legacy_no_build, False),
            ("f_legacy_with_build (AABB-only filter)", cut_f_legacy_with_build, False),
            ("a1_splitter (modern filter)", cut_a1_splitter, True),
        ]:
            ents = build_dense_scene(n_bodies=n)
            prepare_entities(ents, perturbation=1e-5, resolve_snap=1e-3)
            print(f"  -> {name}")
            results.append(run_strategy_with_filter(name, fn, ents, mode))
        short_report(results, baseline_name="baseline (modern filter)")

        # --- full pipeline timing (instantiate + cut + fragment) using stock processor ---
        ents = build_dense_scene(n_bodies=n)
        time_full_pipeline(ents, f"n_bodies={n}")


if __name__ == "__main__":
    main()
