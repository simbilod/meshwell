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
from shapely.geometry import box

from meshwell.interface_tag import InterfaceTag


def prepare_entities(
    entities_list: list[Any],
    perturbation: float,
    *,
    skip_buffer: bool = False,
    resolve_snap: float | None = None,
) -> None:
    """In-place pre-pass shared by cad_gmsh and cad_occ.

    Mutates polygon entities and InterfaceTags. Must NOT be called
    twice on the same list -- the second buffer would compound.

    Args:
        entities_list: List of entities to process.
        perturbation: Outward shapely buffer applied to polygon entities.
        skip_buffer: When True, skip the polygon buffering pass entirely.
            Used by the distributed pipeline (workers receive entities
            already buffered by the master).
        resolve_snap: Snap distance passed to InterfaceTag.resolve().
            Defaults to ``perturbation`` when ``None``. cad_gmsh passes
            ``max(perturbation, point_tolerance)`` so the resolved strip
            is wide enough for non-degenerate panels.
    """
    if not entities_list:
        return

    if not skip_buffer:
        # ----- Pass A: buffer all polygon-bearing entities (shapely only) -----
        xmin, ymin, xmax, ymax = (
            float("inf"),
            float("inf"),
            float("-inf"),
            float("-inf"),
        )
        for ent in entities_list:
            if hasattr(ent, "polygons"):
                polys = (
                    ent.polygons if isinstance(ent.polygons, list) else [ent.polygons]
                )
                for p in polys:
                    b = p.bounds
                    xmin = min(xmin, b[0])
                    ymin = min(ymin, b[1])
                    xmax = max(xmax, b[2])
                    ymax = max(ymax, b[3])

        if xmin == float("inf"):
            # No polygon-bearing entities; nothing to buffer or resolve.
            return

        # Slight bbox inflation so the clip doesn't trim the buffer halo
        # at the scene exterior.
        global_bbox = box(
            xmin - perturbation,
            ymin - perturbation,
            xmax + perturbation,
            ymax + perturbation,
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
                ent.polygons = (
                    shapely.set_precision(
                        ent.polygons, grid_size=relaxed_grid, mode="pointwise"
                    )
                    .buffer(perturbation, join_style=2)
                    .intersection(global_bbox)
                )

    # ----- Pass B: resolve each InterfaceTag against the buffered polygons -----
    polygon_ents: dict[str, list[Any]] = {}
    for ent in entities_list:
        if not hasattr(ent, "polygons"):
            continue
        name = ent.physical_name
        if isinstance(name, tuple):
            name = name[0]
        polygon_ents.setdefault(name, []).append(ent)

    snap = resolve_snap if resolve_snap is not None else perturbation
    for ent in entities_list:
        if isinstance(ent, InterfaceTag):
            ent.resolve(polygon_ents, default_snap=snap)
