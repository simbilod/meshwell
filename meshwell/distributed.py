"""Distributed domain-decomposition pipeline for meshwell (v2, single-phase).

Splits an input scene into per-subdomain CAD + mesh jobs (file-based bundles)
and stitches the resulting .msh files into one final mesh via gmsh.merge.
Workers emit ``_hashed_physical_tags=True`` so all per-subdomain .msh files
agree on which integer tag represents each physical-group name; gmsh.merge
then auto-unions entities with matching (dim, tag) under the shared name and
``removeDuplicateNodes`` welds coincident nodes at shared faces.

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
    neighbors: list[tuple[str, str]] = field(default_factory=list)
    # neighbors: list of (neighbour_volume_id, shared_boundary_wkt) pairs.
    # Empty list means the volume is isolated; populated only for actual seams.


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
    # Compute pairwise shared boundaries between subdomains. The shared
    # boundary geometry is what the worker uses to identify which faces
    # of its result.msh sit on a subdomain seam (as opposed to a true
    # outer-domain edge).
    for i in range(len(subdomains)):
        for j in range(i + 1, len(subdomains)):
            shared = subdomains[i].boundary.intersection(subdomains[j].boundary)
            if shared.is_empty or shared.length < point_tolerance:
                continue
            volumes[i].neighbors.append((volumes[j].id, shared.wkt))
            volumes[j].neighbors.append((volumes[i].id, shared.wkt))

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
        v.id: {
            "polygon_wkt": v.polygon.wkt,
            "neighbors": [
                {"id": nb_id, "shared_boundary_wkt": wkt} for nb_id, wkt in v.neighbors
            ],
        }
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
                "neighbors": [
                    {"id": nb_id, "shared_boundary_wkt": wkt}
                    for nb_id, wkt in vol.neighbors
                ],
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


def _bbox_on_geometry(bb, seam_geom, tol: float) -> bool:
    """Return True iff the OCC face's xy projection LIES ON the seam line.

    bb is gmsh's 6-tuple (xmin, ymin, zmin, xmax, ymax, zmax). The seam
    geometry is a 1D LineString (or MultiLineString) in xy. A face lies
    on the seam if its xy bbox is contained within the seam line buffered
    by tol — i.e., the face is a vertical wall whose horizontal projection
    is a subset of the seam line.

    The previous looser check (face_xy.intersects(seam_geom)) misclassified
    floor/ceiling/lateral-outer faces that TOUCH the seam line at an edge.
    """
    from shapely.geometry import box

    xmin, ymin, _zmin, xmax, ymax, _zmax = bb
    face_xy = box(xmin, ymin, xmax, ymax)
    return face_xy.within(seam_geom.buffer(tol))


def _retag_subdomain_seams(
    neighbours: list[tuple[str, str]],
    own_id: str,
    point_tolerance: float,
) -> None:
    """Rename ``<material>___None`` faces that lie on subdomain seams.

    For each existing physical group ending in ``___None``, partition its
    face entities by which subdomain seam (if any) they lie on:

      * face on the seam shared with neighbour N: regroup as
        ``<material>___seam_<min(own,N)>_<max(own,N)>``
      * face not on any seam (true outer boundary): stays in the
        existing ``<material>___None`` group

    Run on the worker's gmsh model AFTER the standard CAD tagger has
    already produced the ``<material>___None`` groups but BEFORE the
    final write. Uses :func:`_name_to_tag` for the new ``___seam_i_j``
    groups so they consolidate by ``gmsh.merge`` in the master.
    """
    import shapely.wkt

    import gmsh

    nb_lines = [(nb_id, shapely.wkt.loads(wkt)) for nb_id, wkt in neighbours]
    own_index = int(own_id.split("_")[-1])

    for _d, tag in list(gmsh.model.getPhysicalGroups(2)):
        name = gmsh.model.getPhysicalName(2, tag)
        if not name.endswith("___None"):
            continue
        material = name[: -len("___None")]
        ents = list(gmsh.model.getEntitiesForPhysicalGroup(2, tag))

        outer: list[int] = []
        seam_buckets: dict[str, list[int]] = {}

        for ent in ents:
            bb = gmsh.model.getBoundingBox(2, int(ent))
            assigned = False
            for nb_id, nb_geom in nb_lines:
                if _bbox_on_geometry(bb, nb_geom, point_tolerance):
                    seam_buckets.setdefault(nb_id, []).append(int(ent))
                    assigned = True
                    break
            if not assigned:
                outer.append(int(ent))

        gmsh.model.removePhysicalGroups([(2, tag)])
        if outer:
            gmsh.model.addPhysicalGroup(
                2,
                outer,
                tag=_name_to_tag(name, 2),
                name=name,
            )
        for nb_id, seam_ents in seam_buckets.items():
            nb_index = int(nb_id.split("_")[-1])
            lo, hi = sorted([own_index, nb_index])
            seam_name = f"{material}___seam_{lo:04d}_{hi:04d}"
            gmsh.model.addPhysicalGroup(
                2,
                seam_ents,
                tag=_name_to_tag(seam_name, 2),
                name=seam_name,
            )


def _retag_subdomain_seams_in_msh(
    msh_path: Path,
    neighbours: list[tuple[str, str]],
    own_id: str,
    point_tolerance: float,
) -> None:
    """Open ``msh_path`` in a fresh gmsh session, retag subdomain seams, rewrite.

    Runs as a standalone post-processing pass on the worker's
    ``result.msh`` so we don't have to thread through the existing
    :func:`generate_mesh` flow.
    """
    import gmsh

    msh_path = Path(msh_path)
    owns = not gmsh.is_initialized()
    if owns:
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("retag")
        gmsh.merge(str(msh_path))
        _retag_subdomain_seams(neighbours, own_id, point_tolerance)
        gmsh.write(str(msh_path))
        gmsh.model.remove()
    finally:
        if owns:
            gmsh.finalize()


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

    neighbours = [
        (nb["id"], nb["shared_boundary_wkt"]) for nb in job.get("neighbors", [])
    ]

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
        if neighbours:
            _retag_subdomain_seams_in_msh(
                job_dir / "result.msh",
                neighbours=neighbours,
                own_id=job["id"],
                point_tolerance=manifest.get("point_tolerance", 1e-3),
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


def _consolidate_seam_groups() -> None:
    """Fold per-tile ``___seam_i_j`` groups into v1 ``A___B`` (or drop).

    Pairs coincident triangle elements (same node-tag set after
    ``removeDuplicateNodes`` welded coincident nodes) across the two
    tiles' ``<material>___seam_<i>_<j>`` physical groups. For each
    paired triangle:

      * same material on both sides: drop — invisible interior face.
      * different materials A != B: collect into a new dim-2 discrete
        entity tagged ``min(A,B)___max(A,B)``.

    Why operate at element granularity: ``gmsh.merge`` consolidates
    entities by tag, so two geometrically-different per-tile faces
    that happened to share a CAD tag end up unioned into ONE merged
    entity. An entity-level pairing therefore can't tell which
    triangles came from the actual seam vs. from the over-detected
    outer wall (the upstream ``_retag_subdomain_seams`` bbox check
    flags any wall whose bbox kisses the seam line, including
    floor/ceiling/sidewall faces). At the triangle level, only the
    actual coincident faces show up as duplicate-node-set pairs.

    The resulting ``A___B`` interface is materialised as a discrete
    surface entity carrying ONE copy of each duplicated triangle (the
    other copy stays on its original entity, which keeps its
    ``<material>___None`` tag where appropriate). All ``___seam_i_j``
    physical groups are dropped after pairing, regardless of outcome.
    """
    import re

    import gmsh

    pat = re.compile(r"^(.+)___seam_(\d+)_(\d+)$")

    # 1) Inventory ___seam_i_j physical groups and their entities.
    seam_groups: list[tuple[int, str, tuple[int, int]]] = []  # (tag, mat, ij)
    seam_entity_to_groups: dict[int, list[tuple[str, tuple[int, int]]]] = {}
    for _d, tag in gmsh.model.getPhysicalGroups(2):
        name = gmsh.model.getPhysicalName(2, tag)
        m = pat.match(name)
        if not m:
            continue
        material, i, j = m.group(1), int(m.group(2)), int(m.group(3))
        seam_groups.append((tag, material, (i, j)))
        for ent in gmsh.model.getEntitiesForPhysicalGroup(2, tag):
            seam_entity_to_groups.setdefault(int(ent), []).append((material, (i, j)))

    if not seam_groups:
        return

    # 2) Walk every triangle on every seam-tagged entity; key by node-set.
    #    For each triangle, record (entity, elem_tag, node-tags-tuple) and
    #    the material(s) the owning entity was tagged with at each seam.
    TRI_TYPE = 2  # gmsh element type for 3-node triangles
    tri_owners: dict[frozenset[int], list[tuple[int, int, tuple[int, int, int]]]] = {}
    for ent in seam_entity_to_groups:
        try:
            types, tags, node_tags = gmsh.model.mesh.getElements(2, ent)
        except Exception:  # noqa: S112
            # Empty entities (no elements) raise; skip them.
            continue
        for tp, tg_arr, nt_arr in zip(types, tags, node_tags):
            if tp != TRI_TYPE:
                continue
            for k in range(len(tg_arr)):
                a, b, c = (
                    int(nt_arr[3 * k]),
                    int(nt_arr[3 * k + 1]),
                    int(nt_arr[3 * k + 2]),
                )
                key = frozenset((a, b, c))
                tri_owners.setdefault(key, []).append((ent, int(tg_arr[k]), (a, b, c)))

    # 3) For each pair of coincident triangles, infer the (material_a,
    #    material_b) cross-tile relationship at the relevant seam.
    #    Same material -> drop. Different -> collect for A___B group.
    new_group_tris: dict[str, list[tuple[tuple[int, int, int], int]]] = {}
    # value entries: (node-tags-tuple, source-entity)

    for owners in tri_owners.values():
        if len(owners) < 2:
            continue
        # Pick the first two distinct-entity owners. (For meshwell-
        # generated tiles, exactly two coincident triangles is the norm.)
        seen_ents: list[tuple[int, int, tuple[int, int, int]]] = []
        for o in owners:
            if not seen_ents or o[0] != seen_ents[-1][0]:
                seen_ents.append(o)
            if len(seen_ents) == 2:
                break
        if len(seen_ents) < 2:
            continue
        ent_a, _tag_a, nt_a = seen_ents[0]
        ent_b, _tag_b, _nt_b = seen_ents[1]

        # Find a (i, j) seam common to both entities; the matching
        # materials at that seam are our pair.
        groups_a = seam_entity_to_groups.get(ent_a, [])
        groups_b = seam_entity_to_groups.get(ent_b, [])
        common_seams = {ij for _, ij in groups_a} & {ij for _, ij in groups_b}
        if not common_seams:
            continue
        for ij in common_seams:
            mats_a = sorted({m for m, jj in groups_a if jj == ij})
            mats_b = sorted({m for m, jj in groups_b if jj == ij})
            # Pick representative materials for each side. If multiple,
            # prefer one that's unique to that side (to avoid
            # symmetric same-material noise from the upstream over-
            # detector).
            mat_a = next((m for m in mats_a if m not in mats_b), mats_a[0])
            mat_b = next((m for m in mats_b if m not in mats_a), mats_b[0])
            if mat_a == mat_b:
                # invisible interior face — drop.
                break
            final_name = "___".join(sorted([mat_a, mat_b]))
            new_group_tris.setdefault(final_name, []).append((nt_a, ent_a))
            break

    # 4) Drop all ___seam_i_j physical groups (entities themselves stay,
    #    and so do their ___None tags where set by the worker).
    import contextlib

    for tag, _mat, _ij in seam_groups:
        with contextlib.suppress(Exception):
            gmsh.model.removePhysicalGroups([(2, tag)])

    # 5) Emit each A___B as a fresh discrete dim-2 entity carrying the
    #    paired triangles; assign it to a new physical group.
    for final_name, tris in new_group_tris.items():
        if not tris:
            continue
        disc = gmsh.model.addDiscreteEntity(2)
        flat_nodes: list[int] = []
        for nt, _src in tris:
            flat_nodes.extend(nt)
        # Empty elementTags → gmsh assigns fresh tags.
        gmsh.model.mesh.addElementsByType(disc, TRI_TYPE, [], flat_nodes)
        gmsh.model.addPhysicalGroup(
            2,
            [disc],
            tag=_name_to_tag(final_name, 2),
            name=final_name,
        )


def stitch_meshes(
    work_dir: Path,
    plan: SubdomainPlan,
    output_mesh: Path,
) -> None:
    """Stitch per-volume .msh files via ``gmsh.merge``.

    Each worker .msh was written with hashed physical-group tags
    (``_hashed_physical_tags=True``), so e.g. 'silicon' has the same
    integer tag in every per-volume file. ``gmsh.merge`` auto-unions
    entities that share a (dim, tag) under the common name, and
    ``removeDuplicateNodes`` welds coincident nodes at shared faces
    (conformal-by-construction when adjacent tiles share characteristic
    length, per the empirical spike at tests/test_merge_spike.py).

    After dedup, :func:`_consolidate_seam_groups` folds the per-tile
    ``<material>___seam_<i>_<j>`` markings into the v1 convention:
    different materials become ``A___B``; same material both sides
    drops out as an invisible interior face.
    """
    import gmsh

    work_dir = Path(work_dir)
    output_mesh = Path(output_mesh)

    owns_gmsh = not gmsh.is_initialized()
    if owns_gmsh:
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("stitched")
        for v in plan.volumes:
            path = work_dir / "jobs" / v.id / "result.msh"
            if path.exists():
                gmsh.merge(str(path))
        gmsh.option.setNumber("Geometry.Tolerance", plan.point_tolerance / 2)
        gmsh.model.mesh.removeDuplicateNodes()
        _consolidate_seam_groups()
        gmsh.write(str(output_mesh))
    finally:
        if owns_gmsh:
            gmsh.finalize()


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
