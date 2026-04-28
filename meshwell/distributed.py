"""Distributed domain-decomposition pipeline for meshwell.

Splits an input scene into per-subdomain CAD + mesh jobs (file-based bundles)
and stitches the resulting .msh files into one final mesh with conformal
seams. See docs/superpowers/specs/2026-04-28-distributed-domain-decomposition-design.md.
"""
from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from shapely.geometry import LineString, Polygon


@dataclass
class Slab:
    """One interface or junction subdomain in the plan.

    For interface slabs, ``between`` has length 2 (the two volume IDs).
    For junctions, ``between`` has length >= 3.
    """

    id: str
    polygon: Polygon
    between: list[str]
    cut_polylines: list[LineString]
    width: float


@dataclass
class VolumeRegion:
    """One volume subdomain in the plan."""

    id: str
    polygon: Polygon
    neighbors: list[str] = field(default_factory=list)


@dataclass
class SubdomainPlan:
    """Output of build_subdomain_plan."""

    volumes: list[VolumeRegion]
    interfaces: list[Slab]
    junctions: list[Slab]
    physical_names_seen: list[str]
    perturbation: float
    point_tolerance: float
    interface_delimiter: str = "___"
    boundary_delimiter: str = "None"


class Executor(Protocol):
    """Protocol for distributed-meshing executors.

    The default implementation is :class:`SubprocessExecutor`. Users can
    plug in Slurm/Ray/k8s adapters by implementing this protocol.
    """

    def submit(self, job_dir: Path) -> Future:
        """Submit a job bundle directory for execution; return a Future."""
        ...


class SubprocessExecutor:
    """Default Executor: runs ``meshwell run-job <job_dir>`` via ProcessPoolExecutor."""

    def __init__(self, max_workers: int | None = None):
        self._pool = ProcessPoolExecutor(max_workers=max_workers)

    def submit(self, job_dir: Path) -> Future:
        """Submit a job bundle directory; returns a Future of the worker's result dict."""
        return self._pool.submit(_run_job_in_subprocess, str(job_dir))


def _run_job_in_subprocess(job_dir_str: str) -> dict:
    import subprocess

    result = subprocess.run(  # noqa: S603
        ["meshwell", "run-job", job_dir_str],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "job_dir": job_dir_str,
    }


def subdomains_from_grid(
    bbox: tuple[float, float, float, float],
    nx: int,
    ny: int,
) -> list[Polygon]:
    """Helper: emit a regular nx-by-ny grid of subdomain polygons."""
    raise NotImplementedError("Task 7")


def build_subdomain_plan(
    subdomains: list[Polygon],
    entities: list[Any],
    interface_width,
    perturbation: float,
    point_tolerance: float,
) -> SubdomainPlan:
    """Build the volume + interface + junction plan from user-supplied subdomains."""
    raise NotImplementedError("Tasks 11-14")


def run_job(job_dir: Path) -> None:
    """Worker entrypoint: read a job bundle and dispatch to generate_mesh."""
    raise NotImplementedError("Task 16")


def cli_main() -> None:
    """`meshwell run-job <dir>` CLI entrypoint."""
    raise NotImplementedError("Task 17")


def generate_mesh_distributed(
    entities: list[Any],
    subdomains: list[Polygon],
    output_mesh: Path | str,
    work_dir: Path | str,
    interface_width,
    executor: Executor | None = None,
    keep_bundles: bool = False,
    registry: dict[str, Any] | None = None,
    **mesh_kwargs,
) -> Any:
    """Top-level distributed-meshing entrypoint."""
    raise NotImplementedError("Task 21")
