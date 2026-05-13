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
    resolve_snap: float | None = None,
    structured_slabs_out: list | None = None,
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
        structured_slabs_out: Opt-in hook. When a list is provided, the
            structured-PolyPrism cascade runs after buffering; resolved
            :class:`meshwell.structured_polyprism.Slab` objects are
            appended to this list and each
            :class:`meshwell.structured_polyprism._StructuredPolyPrism`
            in ``entities_list`` is replaced in-place with a batch of
            :class:`meshwell.structured_polyprism._StructuredPhantom`
            entities (one per slab). When ``None`` (default), structured
            prisms flow through unchanged so existing callers are
            unaffected.
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

    # ----- Pass C: structured-PolyPrism cascade (opt-in) -----
    if structured_slabs_out is not None:
        from meshwell.structured_polyprism import (
            _StructuredPhantom,
            _StructuredPolyPrism,
            _validate_slab_face_topology_symmetry,
            resolve_structured_slabs,
        )

        # Run the cascade once on the full list (uses the post-buffer
        # polygons; structured prisms participate in buffering above).
        # ``resolve_structured_slabs`` includes the xy-partition cascade
        # extension, so emitted slabs have single-component bottom/top
        # OCC faces regardless of which non-structured entities touch
        # their z-boundaries. The symmetry validator below is kept as a
        # belt-and-suspenders guard -- it should never trigger.
        slabs = resolve_structured_slabs(entities_list)
        _validate_slab_face_topology_symmetry(slabs, entities_list)
        structured_slabs_out.extend(slabs)

        # Build phantoms in source order so insertion order maps directly
        # back to user intent for fragmentation tie-breaks.
        phantom_entities = [_StructuredPhantom(s) for s in slabs]

        # Replace original structured-mode PolyPrism instances with
        # phantoms in place.
        new_list: list = []
        replaced = False
        for ent in entities_list:
            if isinstance(ent, _StructuredPolyPrism):
                if not replaced:
                    new_list.extend(phantom_entities)
                    replaced = True
                # subsequent structured entries: skip (already represented
                # in the phantom batch).
                continue
            new_list.append(ent)
        if not replaced and phantom_entities:
            # All slabs came from a single structured PolyPrism scanned
            # first; this branch is unreachable in practice because the
            # loop above handles that case, but kept for safety.
            new_list = phantom_entities + new_list

        entities_list[:] = new_list
