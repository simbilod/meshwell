"""Distributed domain-decomposition pipeline for meshwell (v2, single-phase).

Splits an input scene into per-subdomain CAD + mesh jobs (file-based bundles)
and stitches the resulting .msh files into one final mesh. Workers emit
``_hashed_physical_tags=True`` so all per-subdomain .msh files agree on the
integer tag of each physical-group name (a prerequisite for the gmsh.merge
based stitch in Task 5).

See docs/superpowers/specs/2026-04-28-distributed-domain-decomposition-design.md
and the empirical merge spike at tests/test_merge_spike.py.
"""
from __future__ import annotations

import hashlib
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import shapely
from shapely.geometry import Polygon

_TAG_SPACE = 1_000_000  # safe gap from gmsh's auto-tag range


def _name_to_tag(name: str, dim: int) -> int:
    """Deterministic positive int tag from (dim, name).

    Same name + dim across processes / runs / files produces the same
    integer, so independently-written .msh files all agree on which
    tag represents 'silicon' at dim=3. After ``gmsh.merge``, entities
    from different files contributing to the same (dim, tag) get
    auto-unioned by gmsh under the shared name — no post-merge
    consolidation pass needed.
    """
    h = hashlib.sha1(f"{dim}:{name}".encode(), usedforsecurity=False).digest()
    n = int.from_bytes(h[:4], "big") % _TAG_SPACE
    return max(1, n)


@dataclass
class VolumeRegion:
    """One volume subdomain in the plan."""

    id: str
    polygon: Polygon
    neighbors: list[str] = field(default_factory=list)


@dataclass
class SubdomainPlan:
    """Output of build_subdomain_plan (v2: volume regions only)."""

    volumes: list[VolumeRegion]
    physical_names_seen: list[str]
    perturbation: float
    point_tolerance: float


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


class InProcessExecutor:
    """Synchronous executor: runs each job in the calling process.

    Used by tests and for debugging — bypasses the subprocess + CLI hop.
    The returned :class:`Future` is already resolved when ``submit`` returns.
    """

    def submit(self, job_dir: Path) -> Future:
        """Run the job synchronously and return an already-resolved Future."""
        f: Future = Future()
        try:
            run_job(Path(job_dir))
            f.set_result({"returncode": 0, "job_dir": str(job_dir)})
        except Exception as e:
            f.set_exception(e)
        return f


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


def _clip_entity_to_polygon(
    entity: Any,
    mask: Polygon,
    point_tolerance: float = 1e-3,
    perturbation: float = 0.0,
) -> Any | None:
    """Return a copy of ``entity`` whose ``.polygons`` are intersected with ``mask``.

    Returns None if the intersection is empty (entity drops out of this
    subdomain). Preserves physical_name, mesh_order, mesh_bool, additive,
    resolutions, and all transformation parameters.

    Supports PolyPrism / PolySurface (anything with a ``.polygons`` attr).
    OCC_entity is not supported and raises NotImplementedError; per spec
    risk R7, it must be fully contained in a single subdomain (validated
    at plan time).

    ``point_tolerance`` is used as an area threshold: any clipped polygon
    whose area is below ``point_tolerance ** 2`` is discarded as a
    sub-tolerance sliver (typically arising from the master-side
    perturbation buffer crossing subdomain boundaries). These crumbs
    would otherwise snap to degenerate polygons under PolyPrism's
    ``set_precision(grid_size=point_tolerance)`` and erase legitimate
    materials in the receiving subdomain.

    Args:
        entity: PolyPrism / PolySurface (anything with a ``.polygons`` attr).
        mask: subdomain polygon to clip against.
        point_tolerance: area threshold for sliver removal (see above).
        perturbation: Inward erosion of the mask before intersection.
            Equal to the master-side polygon buffer (default 1e-5 in
            generate_mesh_distributed). Prevents adjacent subdomains'
            buffer halos from leaking across the cut. Set to 0 to disable.
    """
    from copy import deepcopy

    if not hasattr(entity, "polygons"):
        raise NotImplementedError(
            f"_clip_entity_to_polygon does not support entity type "
            f"{type(entity).__name__}"
        )

    area_threshold = point_tolerance * point_tolerance

    effective_mask = mask if perturbation == 0 else mask.buffer(-perturbation)
    if effective_mask.is_empty:
        return None

    polys = entity.polygons
    if isinstance(polys, list):
        clipped_list = [p.intersection(effective_mask) for p in polys]
        clipped_list = [
            p for p in clipped_list if not p.is_empty and p.area >= area_threshold
        ]
        if not clipped_list:
            return None
        new_polys = clipped_list
    else:
        c = polys.intersection(effective_mask)
        if c.is_empty or c.area < area_threshold:
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
    perturbation: float,
    point_tolerance: float,
) -> SubdomainPlan:
    """Build a plan with one VolumeRegion per subdomain polygon.

    Validates: subdomains non-empty + valid; their union covers the
    entity-polygon union within ``point_tolerance``.

    v2: no interface or junction subdomains. Workers mesh independently
    and the stitch step (``stitch_meshes``) uses ``gmsh.merge`` with
    hashed physical-group tags to consolidate entities by name. The
    ``neighbors`` field on each :class:`VolumeRegion` stays empty since
    we no longer compute pairwise adjacency for seam-bundle planning.
    """
    if not subdomains:
        raise ValueError("subdomains must be non-empty")
    for i, sd in enumerate(subdomains):
        if not sd.is_valid:
            raise ValueError(f"subdomain {i} is not valid: {sd.wkt}")

    # Coverage validation: every polygon-bearing entity must lie within
    # ``point_tolerance`` of the subdomain union. Fail fast before doing
    # any work.
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

    return SubdomainPlan(
        volumes=volumes,
        physical_names_seen=list(physical_names_seen),
        perturbation=perturbation,
        point_tolerance=point_tolerance,
    )


def write_bundles(
    work_dir: Path,
    plan: SubdomainPlan,
    entities: list[Any],
    mesh_kwargs: dict,
) -> None:
    """Write the bundle directory tree for ``plan`` under ``work_dir``.

    Emits ``manifest.json`` plus a per-volume directory under ``jobs/<id>/``
    containing ``job.json``, ``entities.json``, ``subdomain.wkt``, and
    ``mesh_kwargs.json``. Each volume bundle holds the entities clipped
    to its subdomain footprint.
    """
    import json

    work_dir = Path(work_dir)
    (work_dir / "jobs").mkdir(parents=True, exist_ok=True)

    subdomains_blob: dict[str, dict] = {
        v.id: {"polygon_wkt": v.polygon.wkt, "neighbors": v.neighbors}
        for v in plan.volumes
    }

    manifest = {
        "version": 2,
        "perturbation": plan.perturbation,
        "point_tolerance": plan.point_tolerance,
        "physical_names_seen": plan.physical_names_seen,
        "subdomains": subdomains_blob,
        "phase_order": [[v.id for v in plan.volumes]],
    }
    (work_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for v in plan.volumes:
        _write_volume_bundle(
            work_dir / "jobs" / v.id,
            v,
            entities,
            mesh_kwargs,
            point_tolerance=plan.point_tolerance,
            perturbation=plan.perturbation,
        )


def _serialize_entities(entities: list[Any]) -> list[dict]:
    return [e.to_dict() for e in entities if hasattr(e, "to_dict")]


def _write_volume_bundle(
    job_dir: Path,
    vol: VolumeRegion,
    entities: list[Any],
    mesh_kwargs: dict,
    point_tolerance: float = 1e-3,
    perturbation: float = 0.0,
) -> None:
    import json

    job_dir.mkdir(parents=True, exist_ok=True)
    clipped = []
    for ent in entities:
        c = _clip_entity_to_polygon(
            ent,
            vol.polygon,
            point_tolerance=point_tolerance,
            perturbation=perturbation,
        )
        if c is not None:
            clipped.append(c)
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "id": vol.id,
                "role": "volume",
                "dim": 3,
                "neighbors": vol.neighbors,
                "manifest_ref": "../../manifest.json",
            },
            indent=2,
        )
    )
    (job_dir / "entities.json").write_text(
        json.dumps(_serialize_entities(clipped), indent=2)
    )
    (job_dir / "subdomain.wkt").write_text(vol.polygon.wkt)
    (job_dir / "mesh_kwargs.json").write_text(
        json.dumps(mesh_kwargs, indent=2, default=str)
    )


def run_job(job_dir: Path) -> None:
    """Worker entrypoint: read a volume job bundle and dispatch to ``generate_mesh``.

    Loads ``job.json`` + ``manifest.json`` (resolved relative to ``job_dir``
    via the bundle's ``manifest_ref``), deserializes ``entities.json`` and
    ``mesh_kwargs.json``, then invokes
    :func:`meshwell.orchestrator.generate_mesh` with ``_pre_buffered=True``,
    ``_global_physical_names=manifest["physical_names_seen"]``, and
    ``_hashed_physical_tags=True``. The hashed-tag flag makes the worker
    emit deterministic, name-derived integer tags so that
    :func:`stitch_meshes` can rely on stable cross-file tag IDs when it
    calls ``gmsh.merge``.

    Always writes ``result.json``; re-raises on failure so subprocess
    executors see a non-zero exit.
    """
    import json
    import time

    from meshwell.orchestrator import generate_mesh
    from meshwell.utils import deserialize

    job_dir = Path(job_dir)
    job = json.loads((job_dir / "job.json").read_text())
    manifest_ref = job["manifest_ref"]
    manifest_path = (job_dir / Path(manifest_ref)).resolve()
    manifest = json.loads(manifest_path.read_text())
    entities = deserialize(json.loads((job_dir / "entities.json").read_text()))
    mesh_kwargs = json.loads((job_dir / "mesh_kwargs.json").read_text())

    extra: dict[str, Any] = {
        "_pre_buffered": True,
        "_global_physical_names": manifest["physical_names_seen"],
        "_hashed_physical_tags": True,
    }

    t0 = time.time()
    try:
        generate_mesh(
            entities=entities,
            dim=job["dim"],
            output_mesh=job_dir / "result.msh",
            **mesh_kwargs,
            **extra,
        )
        status = "ok"
        err = None
    except Exception as e:
        status = "error"
        err = repr(e)

    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "status": status,
                "error": err,
                "elapsed_s": time.time() - t0,
                "id": job["id"],
                "role": job["role"],
            },
            indent=2,
        )
    )
    if status != "ok":
        raise RuntimeError(f"run_job failed for {job['id']}: {err}")


def run_plan(work_dir: Path, plan: SubdomainPlan, executor: Executor) -> None:
    """Single-phase scheduler: submit all volume bundles, wait, raise on any failure."""
    work_dir = Path(work_dir)
    futures = {v.id: executor.submit(work_dir / "jobs" / v.id) for v in plan.volumes}
    failures: list[tuple[str, str]] = []
    for vid, f in futures.items():
        try:
            res = f.result()
            if isinstance(res, dict) and res.get("returncode", 0) != 0:
                failures.append((vid, res.get("stderr", "")))
        except Exception as e:
            failures.append((vid, repr(e)))
    if failures:
        raise RuntimeError(f"Job failures: {failures}")


def stitch_meshes(
    work_dir: Path,
    plan: SubdomainPlan,
    output_mesh: Path,
) -> None:
    """Concatenate all volume_*.msh files into one unified mesh.

    Uses meshio so that physical-group names from each file's field_data
    are preserved and consolidated by name. Tag IDs are remapped per-file
    to avoid collisions; consolidated groups are re-numbered with fresh
    monotonic tags.

    The previous implementation used ``gmsh.merge`` + ``getPhysicalName``,
    which collides physical-group tag IDs across per-volume .msh files —
    only the first file's name survives for any given (dim, tag), losing
    every subsequent material name. meshio gives us per-file ``field_data``
    keyed by name, which we can consolidate cleanly.

    Steps:
      1. Read each volume_*.msh with meshio.
      2. Track each file's (name -> (tag, dim)) mapping for re-tagging.
      3. Concatenate points and cells across files, offsetting node tags
         per-file to avoid collisions.
      4. Build a unified field_data: {name: [new_tag, dim]} where each
         unique (name, dim) combination gets one consolidated tag.
      5. Build unified cell_data['gmsh:physical'] using the new tags.
      6. Dedup duplicate nodes by coordinate within point_tolerance / 2.
      7. Write to output_mesh as gmsh22 (gmsh4 writer drops field_data).
    """
    import meshio
    import numpy as np

    work_dir = Path(work_dir)
    output_mesh = Path(output_mesh)

    files = []
    for v in plan.volumes:
        path = work_dir / "jobs" / v.id / "result.msh"
        if not path.exists():
            continue
        m = meshio.read(path)
        files.append((v.id, m))

    if not files:
        raise RuntimeError(f"No volume meshes found under {work_dir}/jobs/")

    # Concatenate points; track per-file node-index offsets.
    all_points = []
    point_offsets = []
    cur_offset = 0
    for _vid, m in files:
        all_points.append(m.points)
        point_offsets.append(cur_offset)
        cur_offset += m.points.shape[0]
    points_concat = np.vstack(all_points) if all_points else np.zeros((0, 3))

    # Build the unified field_data: each (name, dim) -> one new tag
    # (per-dim monotonic).
    name_dim_to_new_tag: dict[tuple[str, int], int] = {}
    next_tag_per_dim: dict[int, int] = {}
    for _vid, m in files:
        for name, arr in (m.field_data or {}).items():
            dim = int(arr[1])
            key = (name, dim)
            if key not in name_dim_to_new_tag:
                next_tag_per_dim.setdefault(dim, 0)
                next_tag_per_dim[dim] += 1
                name_dim_to_new_tag[key] = next_tag_per_dim[dim]

    field_data = {
        name: np.array([new_tag, dim])
        for (name, dim), new_tag in name_dim_to_new_tag.items()
    }

    # Concatenate cells; remap (dim, old_tag) -> new tag using each
    # file's own field_data lookup table.
    type_to_dim = {
        "vertex": 0,
        "line": 1,
        "line3": 1,
        "triangle": 2,
        "triangle6": 2,
        "quad": 2,
        "quad9": 2,
        "tetra": 3,
        "tetra10": 3,
        "hexahedron": 3,
        "wedge": 3,
        "pyramid": 3,
    }

    cell_blocks: list[meshio.CellBlock] = []
    cell_data_phys: list[np.ndarray] = []

    for file_idx, (_vid, m) in enumerate(files):
        offset = point_offsets[file_idx]
        per_file_oldtag_to_new: dict[tuple[int, int], int] = {}
        for name, arr in (m.field_data or {}).items():
            tag, dim = int(arr[0]), int(arr[1])
            per_file_oldtag_to_new[(dim, tag)] = name_dim_to_new_tag[(name, dim)]

        gmsh_phys = m.cell_data.get("gmsh:physical") if m.cell_data else None
        for block_idx, block in enumerate(m.cells):
            new_data = block.data + offset
            cell_blocks.append(meshio.CellBlock(block.type, new_data))

            dim = type_to_dim.get(block.type, -1)
            old_phys = (
                gmsh_phys[block_idx]
                if gmsh_phys is not None and block_idx < len(gmsh_phys)
                else None
            )
            if old_phys is None:
                cell_data_phys.append(np.zeros(len(new_data), dtype=np.int32))
            else:
                remapped = np.array(
                    [per_file_oldtag_to_new.get((dim, int(t)), 0) for t in old_phys],
                    dtype=np.int32,
                )
                cell_data_phys.append(remapped)

    # Dedup nodes by coordinate (within point_tolerance / 2). Bucketed:
    # round coords to tolerance, group identical keys.
    tol = max(plan.point_tolerance / 2.0, 1e-12)
    quantized = np.round(points_concat / tol).astype(np.int64)
    keys = [tuple(row) for row in quantized]
    canonical: dict[tuple, int] = {}
    remap = np.empty(points_concat.shape[0], dtype=np.int64)
    new_points: list = []
    for i, k in enumerate(keys):
        if k not in canonical:
            canonical[k] = len(new_points)
            new_points.append(points_concat[i])
        remap[i] = canonical[k]
    points_dedup = np.array(new_points) if new_points else np.zeros((0, 3))

    new_cell_blocks = [
        meshio.CellBlock(block.type, remap[block.data]) for block in cell_blocks
    ]

    # gmsh22 because meshio's gmsh4 writer drops field_data.
    out = meshio.Mesh(
        points=points_dedup,
        cells=new_cell_blocks,
        cell_data={"gmsh:physical": cell_data_phys},
        field_data=field_data,
    )
    meshio.write(output_mesh, out, file_format="gmsh22")


def cli_main() -> None:
    """``meshwell`` CLI entrypoint.

    Subcommands:
        run-job <job_dir>: invoke :func:`run_job` on a single bundle.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="meshwell")
    sub = parser.add_subparsers(dest="cmd", required=True)
    rj = sub.add_parser("run-job", help="Execute a single distributed job bundle")
    rj.add_argument("job_dir", type=Path)
    args = parser.parse_args(sys.argv[1:])
    if args.cmd == "run-job":
        run_job(args.job_dir)


def generate_mesh_distributed(
    entities: list[Any],
    subdomains: list[Polygon],
    output_mesh: Path | str,
    work_dir: Path | str,
    executor: Executor | None = None,
    keep_bundles: bool = False,
    registry: dict[str, Any] | None = None,
    **mesh_kwargs,
) -> None:
    """Distributed mesh generation: clip -> mesh in parallel -> stitch.

    Pipeline:

      1. ``deserialize`` — accept either already-instantiated entities or
         their dict form (round-tripped via :func:`meshwell.utils.deserialize`).
      2. Master-side ``prepare_entities`` — apply the global perturbation
         buffer ONCE here, on the full pre-clip entity list. Workers receive
         ``_pre_buffered=True`` and skip a second buffer pass.
      3. ``build_subdomain_plan`` — derive volume subdomains and validate
         coverage.
      4. ``write_bundles`` — emit per-job bundle directories under ``work_dir``.
      5. ``run_plan`` — submit every volume bundle through the supplied
         :class:`Executor` (defaults to :class:`SubprocessExecutor`); raise
         on any failure. Workers mesh independently with hashed physical-
         group tags so cross-file tag IDs agree by name.
      6. ``stitch_meshes`` — ``gmsh.merge`` each result.msh + dedup duplicate
         nodes at shared faces; write ``output_mesh``.
      7. Optional cleanup — when ``keep_bundles`` is False, ``shutil.rmtree``
         the working directory.

    No ``interface_width`` parameter: workers mesh fully independently and
    the stitch uses ``gmsh.merge`` with hashed tags. Adjacent tiles must
    share characteristic length on their common face for conformal seams
    (verified by the spike at tests/test_merge_spike.py).
    """
    import shutil

    from meshwell.cad_common import prepare_entities
    from meshwell.utils import deserialize

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if executor is None:
        executor = SubprocessExecutor()

    entities = deserialize(entities, registry=registry)

    perturbation = mesh_kwargs.get("perturbation", 1e-5)
    point_tolerance = mesh_kwargs.get("point_tolerance", 1e-3)

    # Apply the global perturbation buffer ONCE here. Workers must NOT
    # re-buffer; that's enforced inside run_job which always passes
    # _pre_buffered=True.
    prepare_entities(entities, perturbation=perturbation)

    plan = build_subdomain_plan(
        subdomains=subdomains,
        entities=entities,
        perturbation=perturbation,
        point_tolerance=point_tolerance,
    )
    write_bundles(work_dir, plan, entities, mesh_kwargs=mesh_kwargs)
    run_plan(work_dir, plan, executor=executor)
    stitch_meshes(work_dir, plan, output_mesh=Path(output_mesh))

    if not keep_bundles:
        shutil.rmtree(work_dir, ignore_errors=True)
