"""Backend-agnostic shapely pre-pass for cad_gmsh and cad_occ.

Both backends call ``prepare_entities`` at the start of their own
``process_entities`` to:

1. Compute a global bounding box of polygon-bearing entities (slightly
   inflated so the buffer step doesn't clip beyond the user's intent).
2. Buffer each polygon-bearing entity outward by ``perturbation``,
   relaxing shapely's precision model first so sub-tolerance buffers
   actually take effect. Skipped entirely when ``buffer_polygons=False``.
3. Resolve each :class:`meshwell.interface_tag.InterfaceTag` against
   the polygon entities (buffered or nominal, depending on
   ``buffer_polygons``).

After the call, polygon entities have (buffered, when requested)
``polygons`` and InterfaceTags have populated ``resolved_linestrings``
-- both ready for backend-specific instantiation.

The OCC path (``cad_occ.py``) applies ``perturbation`` analytically at
wire-emission time (canonical circle/line offsets in
``GeometryEntity._make_occ_wire_from_vertices``), so it calls this
function with ``buffer_polygons=False``: entities stay at their nominal
coordinates through this pre-pass and the structured pre-pass, and the
offset is only realized when OCC wires are built. Only the cad_gmsh
mirror still relies on the shapely round-join buffer.
"""
from __future__ import annotations

import logging
from typing import Any

import shapely
from shapely.geometry import box

from meshwell.interface_tag import InterfaceTag

logger = logging.getLogger(__name__)


def prepare_entities(
    entities_list: list[Any],
    perturbation: float,
    resolve_snap: float | None = None,
    buffer_polygons: bool = True,
) -> None:
    """In-place pre-pass shared by cad_gmsh and cad_occ.

    Mutates polygon entities and InterfaceTags. Must NOT be called
    twice on the same list with ``buffer_polygons=True`` -- the second
    buffer would compound (the ``_meshwell_prepared`` guard below only
    ever gets set on that path, so it can only catch a double call
    there). With ``buffer_polygons=False`` (the cad_occ path) Pass A is
    skipped entirely and nothing is mutated in a way that compounds, so
    this guard does not apply.

    Args:
        entities_list: List of entities to process.
        perturbation: Outward shapely buffer applied to polygon entities.
        resolve_snap: Snap distance passed to InterfaceTag.resolve().
            Defaults to ``perturbation`` when ``None``. cad_gmsh passes
            ``max(perturbation, point_tolerance)`` so the resolved strip
            is wide enough for non-degenerate panels.
        buffer_polygons: When True (default), buffer every polygon-bearing
            entity outward by ``perturbation`` (Pass A below) before
            resolving InterfaceTags. cad_gmsh always passes True (it has
            no analytic-offset path). cad_occ passes False: the OCC path
            applies ``perturbation`` analytically at wire-emission time
            (see :func:`meshwell.geometry_entity.GeometryEntity._make_occ_wire_from_vertices`),
            so entities must stay at nominal coordinates through this
            pre-pass. The InterfaceTag resolve pass (Pass B) still runs
            against the (unbuffered, in this case) polygons.
    """
    if not entities_list:
        return

    already = [
        ent
        for ent in entities_list
        if hasattr(ent, "polygons") and getattr(ent, "_meshwell_prepared", False)
    ]
    if already:
        raise RuntimeError(
            f"prepare_entities called twice on already-prepared entities "
            f"(first: {type(already[0]).__name__} "
            f"{getattr(already[0], 'physical_name', '?')}): the perturbation "
            f"buffer would compound. Pass prepared=True to the CAD processor "
            f"when an earlier stage already prepared this list, or rebuild the "
            f"entities from scratch for a fresh run (e.g. a parameter sweep or "
            f"re-mesh on the same inputs)."
        )

    if buffer_polygons:
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
            ent._meshwell_prepared = True

    # ----- Pass B: resolve each InterfaceTag against the polygons -----
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


def apply_arc_params(
    entities: list,
    *,
    identify_arcs: bool,
    min_arc_points: int = 5,
    arc_tolerance: float = 1e-3,
) -> None:
    """Stamp pipeline-level arc-identification parameters onto entities.

    Arc identification determines CROSS-entity interface geometry: two
    entities sharing a circular boundary must agree on arc-vs-chord
    classification or their booleans graze (sliver source). It is
    therefore a scene-level setting stamped uniformly here, not a
    per-entity constructor flag. Entities carrying either ``.polygons``
    (PolySurface/PolyPrism) or ``.linestrings`` (PolyLine) are stamped;
    entities with neither are skipped.
    Non-extrude PolyPrisms (z-varying buffers) do not support arcs: they
    are left at identify_arcs=False with a warning instead of raising.

    ``InterfaceTag`` also carries ``.linestrings`` and so matches this
    guard and gets stamped like a PolyLine, but the stamp is inert for
    it today: InterfaceTag's own wire/face emission never reads
    ``identify_arcs`` / ``min_arc_points`` / ``arc_tolerance``, so arc
    identification has no effect on interface geometry.
    """
    for e in entities:
        has_polygons = getattr(e, "polygons", None) is not None
        has_linestrings = getattr(e, "linestrings", None) is not None
        if not (has_polygons or has_linestrings):
            continue
        wants_arcs = identify_arcs
        if wants_arcs and not getattr(e, "extrude", True):
            logger.warning(
                "identify_arcs: skipping %s (z-varying buffers; arc "
                "identification requires extrude=True)",
                getattr(e, "physical_name", "?"),
            )
            wants_arcs = False
        e.identify_arcs = wants_arcs
        e.min_arc_points = min_arc_points
        e.arc_tolerance = arc_tolerance
