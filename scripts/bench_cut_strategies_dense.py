# ruff: noqa
"""Denser benchmark: substrate cut against many small bodies + full-pipeline timing.

Tries to reproduce the previously-reported BRepAlgoAPI_Cut(compound) empty-result
failure on a scene that more closely matches the original 'substrate vs ~10
metal+helper bodies' case mentioned in cad_occ.py:467-472.

Also reports the cost of the full pipeline (cut + fragment) so we can put the
cut-strategy speedups in context.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from shapely.geometry import Polygon

from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Splitter
from OCP.BRep import BRep_Builder
from OCP.TopoDS import TopoDS_Compound, TopoDS_Shape
from OCP.TopTools import TopTools_ListOfShape

from meshwell.cad_common import prepare_entities
from meshwell.cad_occ import CAD_OCC

import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from bench_cut_strategies import (  # type: ignore
    run_strategy,
    report,
    cut_baseline,
    cut_a2_runparallel,
    cut_a1_splitter,
    cut_f_cut_compound,
    shape_volume,
    shape_solid_count,
)
from meshwell.polyprism import PolyPrism


def _square(x, y, w, h):
    return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def build_dense_scene(n_bodies: int = 12) -> list[Any]:
    """Substrate at z=[-1,0], box=[0,0.5], plus n_bodies small metal pillars at z=[1,1.5].

    The substrate cuts against the metal pillars + their helper-y siblings,
    matching the original substrate-vs-10-bodies pattern.
    """
    big = _square(-10, -10, 20, 20)
    ents = [
        PolyPrism(
            polygons=big,
            buffers={-1.0: 0.0, 0.0: 0.0},
            physical_name="substrate",
            mesh_order=100.0,
        ),
        PolyPrism(
            polygons=big,
            buffers={0.0: 0.0, 0.5: 0.0},
            physical_name="box",
            mesh_order=90.0,
        ),
        PolyPrism(
            polygons=big,
            buffers={1.5: 0.0, 3.0: 0.0},
            physical_name="top_clad",
            mesh_order=99.0,
        ),
    ]
    # n_bodies metal pillars in a grid, with TWO bodies (metal + via) per slot
    # to mimic the 'metal + helper' structure
    cols = int(math.ceil(math.sqrt(n_bodies)))
    rows = int(math.ceil(n_bodies / cols))
    cell = 16.0 / cols
    placed = 0
    for r in range(rows):
        for c in range(cols):
            if placed >= n_bodies:
                break
            x = -8 + c * cell + cell * 0.25
            y = -8 + r * cell + cell * 0.25
            w = cell * 0.4
            ents.append(
                PolyPrism(
                    polygons=_square(x, y, w, w),
                    buffers={1.0: 0.0, 1.5: 0.0},
                    physical_name=f"metal_{placed}",
                    mesh_order=1.0 + placed * 0.01,
                )
            )
            # 'helper' via beside it
            ents.append(
                PolyPrism(
                    polygons=_square(x + w * 1.1, y, w * 0.5, w),
                    buffers={0.5: 0.0, 1.0: 0.0},
                    physical_name=f"via_{placed}",
                    mesh_order=2.0 + placed * 0.01,
                )
            )
            placed += 1
    return ents


STRATEGIES: dict[str, Callable] = {
    "baseline": cut_baseline,
    "a2_runparallel": cut_a2_runparallel,
    "a1_splitter": cut_a1_splitter,
    "f_cut_compound": cut_f_cut_compound,
}


def main() -> None:
    for n in (6, 12, 20):
        print("\n" + "#" * 100)
        print(
            f"# Dense scene: n_bodies={n}  (substrate cut against ~{n} metal+via pairs)"
        )
        print("#" * 100)
        entities = build_dense_scene(n_bodies=n)
        prepare_entities(entities, perturbation=1e-5, resolve_snap=1e-3)
        results = []
        for name, fn in STRATEGIES.items():
            print(f"  -> {name}")
            results.append(run_strategy(name, fn, entities))
        report(results)


if __name__ == "__main__":
    main()
