"""Benchmark Remesher.get_current_mesh_sizes and plot2D on random Delaunay meshes.

Pass ``--ref`` to also time the implementation at another git revision,
loaded straight from ``git show`` so no second checkout is needed.

Run::

    python scripts/benchmark_remesh_plot.py
    python scripts/benchmark_remesh_plot.py --ref main
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import meshio
import numpy as np
from scipy.spatial import Delaunay

import meshwell.remesh
import meshwell.visualization

REPO_ROOT = Path(__file__).resolve().parents[1]

# (dimension, number of random points)
SIZE_CASES = [(2, 50_000), (2, 200_000), (3, 20_000), (3, 100_000)]
PLOT_POINTS = [2_000, 10_000]


def _load_at_ref(ref: str, relpath: str, tmpdir: Path) -> ModuleType:
    source = subprocess.run(  # noqa: S603
        ["git", "show", f"{ref}:{relpath}"],  # noqa: S607
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    path = tmpdir / relpath.replace("/", "_")
    path.write_text(source)
    name = f"_ref_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses looks the module up in sys.modules while it executes
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _timed(func: Callable[[], object], repeat: int) -> float:
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - start)
    return best


def bench_sizes(
    impls: dict[str, ModuleType], rng: np.random.Generator, repeat: int
) -> None:
    """Time get_current_mesh_sizes per implementation, checking they agree."""
    print("get_current_mesh_sizes")
    for dim, n_points in SIZE_CASES:
        points = rng.random((n_points, dim))
        cells = Delaunay(points).simplices.astype(np.int64)
        vxyz = np.zeros((n_points, 3))
        vxyz[:, :dim] = points

        results, times = {}, {}
        for label, module in impls.items():
            remesher = module.Remesher(n_threads=1, filename=f"bench_{label}")
            remesher.vxyz = vxyz
            remesher.triangles = cells
            results[label] = remesher.get_current_mesh_sizes()
            times[label] = _timed(remesher.get_current_mesh_sizes, repeat)

        reference = next(iter(results.values()))
        if not all(np.allclose(reference, r) for r in results.values()):
            raise RuntimeError("implementations disagree on mesh sizes")
        kind = "tris" if dim == 2 else "tets"
        row = "  ".join(f"{label} {t:.3f}s" for label, t in times.items())
        print(f"  {dim}D {len(cells):>7} {kind}  {row}")


def bench_plot(
    impls: dict[str, ModuleType], rng: np.random.Generator, repeat: int
) -> None:
    """Time plot2D including the canvas draw, where most artist cost lands."""
    print("plot2D fill + draw")
    for n_points in PLOT_POINTS:
        points = rng.random((n_points, 2))
        triangles = Delaunay(points).simplices
        groups = (points[triangles].mean(axis=1)[:, 0] > 0.5).astype(int) + 1
        mesh = meshio.Mesh(
            np.c_[points, np.zeros(n_points)],
            [("triangle", triangles)],
            cell_data={"gmsh:physical": [groups]},
            field_data={"left": np.array([1, 2]), "right": np.array([2, 2])},
        )

        times = {}
        for label, module in impls.items():

            def draw(module: ModuleType = module, mesh: meshio.Mesh = mesh) -> None:
                plt.close("all")
                module.plot2D(mesh, ignore_lines=True)
                plt.gcf().canvas.draw()

            times[label] = _timed(draw, repeat)
        plt.close("all")

        row = "  ".join(f"{label} {t:.2f}s" for label, t in times.items())
        print(f"  {len(triangles):>6} tris  {row}")


def main() -> None:
    """Parse arguments and run both benchmarks."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", help="git revision to compare against, e.g. main")
    parser.add_argument("--repeat", type=int, default=3, help="best of N runs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        remesh_impls: dict[str, ModuleType] = {}
        plot_impls: dict[str, ModuleType] = {}
        if args.ref:
            remesh_impls[args.ref] = _load_at_ref(
                args.ref, "meshwell/remesh.py", Path(tmp)
            )
            plot_impls[args.ref] = _load_at_ref(
                args.ref, "meshwell/visualization.py", Path(tmp)
            )
        remesh_impls["current"] = meshwell.remesh
        plot_impls["current"] = meshwell.visualization

        rng = np.random.default_rng(args.seed)
        bench_sizes(remesh_impls, rng, args.repeat)
        bench_plot(plot_impls, rng, args.repeat)


if __name__ == "__main__":
    main()
