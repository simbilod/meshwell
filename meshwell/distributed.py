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

import shapely
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


def _clip_entity_to_polygon(entity: Any, mask: Polygon) -> Any | None:
    """Return a copy of ``entity`` whose ``.polygons`` are intersected with ``mask``.

    Returns None if the intersection is empty (entity drops out of this
    subdomain). Preserves physical_name, mesh_order, mesh_bool, additive,
    resolutions, and all transformation parameters.

    Supports PolyPrism / PolySurface (anything with a ``.polygons`` attr).
    OCC_entity is not supported and raises NotImplementedError; per spec
    risk R7, it must be fully contained in a single subdomain (validated
    at plan time).
    """
    from copy import deepcopy

    if not hasattr(entity, "polygons"):
        raise NotImplementedError(
            f"_clip_entity_to_polygon does not support entity type "
            f"{type(entity).__name__}"
        )

    polys = entity.polygons
    if isinstance(polys, list):
        clipped_list = [p.intersection(mask) for p in polys]
        clipped_list = [p for p in clipped_list if not p.is_empty]
        if not clipped_list:
            return None
        new_polys = clipped_list
    else:
        c = polys.intersection(mask)
        if c.is_empty:
            return None
        new_polys = c

    new = deepcopy(entity)
    new.polygons = new_polys

    # PolyPrism non-extrude path keeps a precomputed buffered_polygons list;
    # recompute it for the clipped footprint so downstream CAD sees the
    # right per-z buffer shapes.
    if (
        hasattr(new, "buffers")
        and hasattr(new, "extrude")
        and not new.extrude
        and hasattr(new, "_get_buffered_polygons")
    ):
        # _get_buffered_polygons expects an iterable of polygons
        polys_iter = new_polys if isinstance(new_polys, list) else [new_polys]
        new.buffered_polygons = new._get_buffered_polygons(polys_iter, new.buffers)

    return new


def _resolution_only_proxy(entity: Any) -> Any:
    """Wrap ``entity`` so it contributes no geometry but keeps its resolutions.

    Used by phase-1 bundles for entities that sit near (but do not intersect)
    a slab whose ResolutionSpecs may still affect the seam mesh sizing. The
    proxy's ``instanciate_occ`` returns an empty TopoDS_Compound so no CAD
    fragment is generated; the resolutions list rides along and gets
    consumed by the worker's resolver.
    """
    from copy import deepcopy

    proxy = deepcopy(entity)
    proxy.mesh_bool = False

    def _empty_occ_shape(_self=proxy):
        from OCP.BRep import BRep_Builder
        from OCP.TopoDS import TopoDS_Compound

        cb = BRep_Builder()
        c = TopoDS_Compound()
        cb.MakeCompound(c)
        return c

    proxy.instanciate_occ = _empty_occ_shape
    # Also stub the gmsh-backend instanciate so the worker doesn't try to
    # build geometry there either.
    proxy.instanciate = lambda _cad_model: []
    return proxy


def _flatten_to_linestrings(geom) -> list[LineString]:
    """Flatten a possibly-mixed shapely geometry into its LineString components."""
    from shapely.geometry import GeometryCollection, MultiLineString

    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        out: list[LineString] = []
        for g in geom.geoms:
            out.extend(_flatten_to_linestrings(g))
        return out
    return []


def subdomains_from_grid(
    bbox: tuple[float, float, float, float],
    nx: int,
    ny: int,
) -> list[Polygon]:
    """Helper: emit a regular nx-by-ny grid of subdomain polygons."""
    from shapely.geometry import box as _box

    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be >= 1")
    xmin, ymin, xmax, ymax = bbox
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"invalid bbox: {bbox}")
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    return [
        _box(
            xmin + i * dx,
            ymin + j * dy,
            xmin + (i + 1) * dx,
            ymin + (j + 1) * dy,
        )
        for i in range(nx)
        for j in range(ny)
    ]


def build_subdomain_plan(
    subdomains: list[Polygon],
    entities: list[Any],
    interface_width,
    perturbation: float,
    point_tolerance: float,
) -> SubdomainPlan:
    """Build the volume + interface + junction plan from user-supplied subdomains."""
    if not subdomains:
        raise ValueError("subdomains must be non-empty")
    for i, sd in enumerate(subdomains):
        if not sd.is_valid:
            raise ValueError(f"subdomain {i} is not valid: {sd.wkt}")

    # Coverage validation: every polygon-bearing entity must lie within
    # `point_tolerance` of the subdomain union. Fail fast before doing any work.
    union_sd = shapely.unary_union(subdomains)
    union_ent = shapely.unary_union(
        [
            p
            for ent in entities
            if hasattr(ent, "polygons")
            for p in (
                ent.polygons if isinstance(ent.polygons, list) else [ent.polygons]
            )
        ]
    )
    if not union_ent.is_empty:
        missing = union_ent.difference(union_sd.buffer(point_tolerance))
        if not missing.is_empty:
            raise ValueError(
                f"not covered: entity polygons not covered by subdomain union: "
                f"{missing.wkt[:200]}"
            )

    volumes = [
        VolumeRegion(id=f"volume_{i:04d}", polygon=sd, neighbors=[])
        for i, sd in enumerate(subdomains)
    ]
    physical_names_seen = sorted(
        {
            n if isinstance(n, str) else n[0]
            for ent in entities
            if hasattr(ent, "physical_name") and ent.physical_name
            for n in (
                (ent.physical_name,)
                if isinstance(ent.physical_name, str)
                else ent.physical_name
            )
        }
    )

    # ---- Interfaces: pairwise slabs between adjacent subdomains ----
    interfaces: list[Slab] = []
    iface_idx = 0
    if isinstance(interface_width, (int, float)):

        def width_for(_i: int, _j: int):
            return interface_width

    else:

        def width_for(i: int, j: int):
            return interface_width.get((min(i, j), max(i, j))) or interface_width.get(
                (max(i, j), min(i, j))
            )

    for i in range(len(subdomains)):
        for j in range(i + 1, len(subdomains)):
            shared = subdomains[i].boundary.intersection(subdomains[j].boundary)
            if shared.is_empty:
                continue
            polylines = _flatten_to_linestrings(shared)
            polylines = [ls for ls in polylines if ls.length > point_tolerance]
            if not polylines:
                continue
            w = width_for(i, j)
            if w is None or w <= 0:
                raise ValueError(f"interface_width missing for pair ({i}, {j})")
            # Flat caps so the slab doesn't bulge past the polyline endpoints —
            # bounds match the test's expectation of (xmin, ymin, xmax, ymax)
            # tight against the shared boundary segment.
            slab_polys = [ls.buffer(w / 2, cap_style=2) for ls in polylines]
            slab = (
                slab_polys[0]
                if len(slab_polys) == 1
                else shapely.unary_union(slab_polys)
            )
            interfaces.append(
                Slab(
                    id=f"interface_{iface_idx:04d}",
                    polygon=slab,
                    between=[f"volume_{i:04d}", f"volume_{j:04d}"],
                    cut_polylines=polylines,
                    width=w,
                )
            )
            volumes[i].neighbors.append(f"volume_{j:04d}")
            volumes[j].neighbors.append(f"volume_{i:04d}")
            iface_idx += 1

    # ---- Junctions: points where >= 3 subdomains meet ----
    from shapely.geometry import Point

    point_to_volumes: dict[tuple[float, float], set[str]] = {}
    for i, sd in enumerate(subdomains):
        # Quantize boundary vertices to point_tolerance grid
        coords: list[tuple[float, float]] = []
        rings = [sd.exterior, *list(sd.interiors)]
        for ring in rings:
            for x, y in ring.coords:
                key = (
                    round(x / point_tolerance) * point_tolerance,
                    round(y / point_tolerance) * point_tolerance,
                )
                coords.append(key)
        for k in set(coords):
            point_to_volumes.setdefault(k, set()).add(f"volume_{i:04d}")

    junctions: list[Slab] = []
    j_idx = 0
    junction_radius = max(
        (s.width for s in interfaces),
        default=interface_width if isinstance(interface_width, (int, float)) else 0,
    )
    for (x, y), vols in point_to_volumes.items():
        if len(vols) < 3:
            continue
        jpoint = Point(x, y)
        touching = [
            s
            for s in interfaces
            if any(jpoint.distance(pl) < point_tolerance for pl in s.cut_polylines)
        ]
        cut_lines: list[LineString] = []
        for s in touching:
            cut_lines.extend(s.cut_polylines)
        junctions.append(
            Slab(
                id=f"junction_{j_idx:04d}",
                polygon=jpoint.buffer(junction_radius),
                between=sorted(vols),
                cut_polylines=cut_lines,
                width=junction_radius,
            )
        )
        j_idx += 1

    return SubdomainPlan(
        volumes=volumes,
        interfaces=interfaces,
        junctions=junctions,
        physical_names_seen=list(physical_names_seen),
        perturbation=perturbation,
        point_tolerance=point_tolerance,
    )


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
