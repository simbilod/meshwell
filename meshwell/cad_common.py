"""Backend-agnostic shapely pre-pass for cad_gmsh and cad_occ.

Both backends call ``prepare_entities`` at the start of their own
``process_entities`` to:

1. Compute a global bounding box of polygon-bearing entities (slightly
   inflated so the buffer step doesn't clip beyond the user's intent).
2. Buffer each polygon-bearing entity outward by ``perturbation``,
   relaxing shapely's precision model first so sub-tolerance buffers
   actually take effect.
3. Resolve each :class:`meshwell.interface_tag.InterfaceTag` against
   the freshly-buffered polygon entities.

After the call, polygon entities have buffered ``polygons`` and
InterfaceTags have populated ``resolved_linestrings`` -- both ready
for backend-specific instantiation.
"""
from __future__ import annotations

from typing import Any

import shapely
from shapely.geometry import MultiPolygon, Polygon, box

from meshwell.interface_tag import InterfaceTag


def normalize_mesh_order(mo: float | None) -> float:
    """Map ``None`` mesh_order to +inf so unset entities sort last.

    Shared by cad_gmsh and cad_occ at every sort key / cut-priority
    comparison site so "unset mesh_order" has one meaning across both
    backends.
    """
    return float("inf") if mo is None else mo


def resolve_piece_ownership(
    piece_candidates: dict[Any, list[tuple[int, float]]],
) -> dict[Any, int]:
    """Pick the owning entity index for each fragment piece.

    Rule: lowest ``mesh_order`` wins; first candidate in insertion order
    wins on tie. Single source of truth for cad_gmsh and cad_occ, whose
    fragment-then-resolve pipelines both build a
    ``piece -> [(entity_index, mesh_order), ...]`` candidate map and call
    this to invert it into ``piece -> owning entity_index``.
    """
    owners: dict[Any, int] = {}
    for piece, candidates in piece_candidates.items():
        best_idx = candidates[0][0]
        best_mo = candidates[0][1]
        for idx, mo in candidates[1:]:
            if mo < best_mo:
                best_idx = idx
                best_mo = mo
        owners[piece] = best_idx
    return owners


def prepare_entities(
    entities_list: list[Any],
    perturbation: float,
    resolve_snap: float | None = None,
) -> None:
    """In-place pre-pass shared by cad_gmsh and cad_occ.

    Mutates polygon entities and InterfaceTags. Must NOT be called
    twice on the same list -- the second buffer would compound.

    Args:
        entities_list: List of entities to process.
        perturbation: Outward shapely buffer applied to polygon entities.
        resolve_snap: Snap distance passed to InterfaceTag.resolve().
            Defaults to ``perturbation`` when ``None``. cad_gmsh passes
            ``max(perturbation, point_tolerance)`` so the resolved strip
            is wide enough for non-degenerate panels.
    """
    if not entities_list:
        return

    # ----- Pass A: buffer all polygon-bearing entities (shapely only) -----
    xmin, ymin, xmax, ymax = (
        float("inf"),
        float("inf"),
        float("-inf"),
        float("-inf"),
    )
    for ent in entities_list:
        if hasattr(ent, "polygons"):
            polys = ent.polygons if isinstance(ent.polygons, list) else [ent.polygons]
            for p in polys:
                b = p.bounds
                xmin = min(xmin, b[0])
                ymin = min(ymin, b[1])
                xmax = max(xmax, b[2])
                ymax = max(ymax, b[3])

    if xmin == float("inf"):
        # No polygon-bearing entities; nothing to buffer or resolve.
        return

    # Bbox inflation so the clip doesn't trim the buffer halo at the
    # scene exterior. join_style=2 (mitre) below can extend a sharp
    # convex corner's offset well past 1x perturbation: mitre length is
    # perturbation / sin(interior_angle / 2), capped by shapely's default
    # mitre_limit=5 at 5x perturbation. Inflating by only 1x perturbation
    # (as before) would shave mitre tips sharper than ~asin(1)=90 degrees
    # wide back down to the scene bbox. Inflate by the full 5x so any
    # mitre tip up to the cap survives the intersection below.
    inflation = 5 * perturbation
    global_bbox = box(
        xmin - inflation,
        ymin - inflation,
        xmax + inflation,
        ymax + inflation,
    )

    # Sub-tolerance buffering requires relaxing the shapely precision
    # model installed by entity constructors (set_precision at
    # point_tolerance). Without this re-set, polygon.buffer(d) with
    # d < point_tolerance returns empty geometry.
    relaxed_grid = max(perturbation / 100, 1e-12)
    for ent in entities_list:
        if not hasattr(ent, "polygons"):
            continue
        if isinstance(ent.polygons, list):
            ent.polygons = [
                shapely.set_precision(p, grid_size=relaxed_grid, mode="pointwise")
                .buffer(perturbation, join_style=2)
                .intersection(global_bbox)
                for p in ent.polygons
            ]
        else:
            buffered = (
                shapely.set_precision(
                    ent.polygons, grid_size=relaxed_grid, mode="pointwise"
                )
                .buffer(perturbation, join_style=2)
                .intersection(global_bbox)
            )
            # PolyPrism guarantees `polygons` is a MultiPolygon, but
            # buffer() dissolves a single-part (or merged) MultiPolygon
            # into a plain Polygon. Re-wrap so downstream `.geoms`
            # consumers keep a MultiPolygon.
            if isinstance(ent.polygons, MultiPolygon) and isinstance(buffered, Polygon):
                buffered = MultiPolygon([buffered])
            ent.polygons = buffered

    # ----- Pass B: resolve each InterfaceTag against the buffered polygons -----
    # Register every polygon-bearing entity under EACH of its physical
    # names, not just the first -- a two-name entity (e.g. a shared
    # region carrying both a material name and an alias) must resolve
    # for an InterfaceTag ``targets=`` referencing either name.
    # InterfaceTag.resolve() deduplicates by identity, so an entity
    # registered under multiple names is still only cut/counted once.
    polygon_ents: dict[str, list[Any]] = {}
    for ent in entities_list:
        if not hasattr(ent, "polygons"):
            continue
        name = ent.physical_name
        names = name if isinstance(name, tuple) else (name,)
        for n in names:
            polygon_ents.setdefault(n, []).append(ent)

    snap = resolve_snap if resolve_snap is not None else perturbation
    for ent in entities_list:
        if isinstance(ent, InterfaceTag):
            ent.resolve(polygon_ents, default_snap=snap)
