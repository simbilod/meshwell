"""Stage 5 — gmsh meshing hooks for structured cohorts.

pre_2d_hook (freeze_lateral_mesh): emits cohort lateral-face quad
mesh from Python before gmsh's generate(2) runs. Uses
Mesh.MeshOnlyEmpty=1 so the outer 2D mesher leaves cohort laterals
alone. Raises on n_layers mismatch or unsupported lateral topology.

pre_3d_hook (stamp_wedges): per cohort sub-solid, copies bot
triangulation to top and emits wedge elements.
"""
from __future__ import annotations

import contextlib
import itertools
import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import gmsh
import numpy as np
from scipy.spatial import KDTree

from meshwell.structured.exceptions import (
    StructuredError,
    StructuredLateralNLayersMismatchError,
    StructuredTransfiniteRejectedError,
    WedgeBotNodeMismatchError,
    WedgeCountMismatchError,
)
from meshwell.structured.types import ShapeKey, SlabMeta

logger = logging.getLogger(__name__)

# Tight gmsh geometry tolerance used while deduplicating mesh nodes, so
# distinct-but-close points (e.g. on adjacent fine curves) are not merged.
_DEDUP_GEOMETRY_TOLERANCE = 1e-6


def _strip_degenerate_elements() -> None:
    """Remove any 1D/2D/3D mesh elements that have duplicate node tags after node deduplication.

    When ``gmsh.model.mesh.removeDuplicateNodes`` merges two vertices A and B
    that belong to the same sliver element (e.g. a tetrahedron ``(u, v, A, B)``
    or triangle ``(u, A, B)``), Gmsh rewrites ``B -> A`` in-place, leaving a
    collapsed zero-volume/zero-area element ``(u, v, A, A)`` whose remaining
    face ``(u, v, A)`` duplicates the shared face between the two adjacent
    valid elements ``(u, v, w1, A)`` and ``(u, v, w2, A)``.
    """
    for dim in (1, 2, 3):
        for _, ent_tag in gmsh.model.getEntities(dim):
            etypes, etags, enodes = gmsh.model.mesh.getElements(dim, ent_tag)
            if not len(etypes):
                continue
            has_degen = False
            kept_by_type: list[tuple[int, list[int], list[int]]] = []
            for et, tags, nodes in zip(etypes, etags, enodes):
                n_per = gmsh.model.mesh.getElementProperties(int(et))[3]
                arr = nodes.reshape(-1, n_per)
                kept_tags: list[int] = []
                kept_nodes: list[int] = []
                for t, row in zip(tags, arr):
                    r = [int(x) for x in row]
                    if len(set(r)) < n_per:
                        has_degen = True
                    else:
                        kept_tags.append(int(t))
                        kept_nodes.extend(r)
                kept_by_type.append((int(et), kept_tags, kept_nodes))
            if has_degen:
                gmsh.model.mesh.removeElements(dim, ent_tag)
                for et, kept_tags, kept_nodes in kept_by_type:
                    if kept_tags:
                        gmsh.model.mesh.addElementsByType(
                            ent_tag, et, kept_tags, kept_nodes
                        )


def _remove_duplicate_nodes_tight(
    dimtags: list[tuple[int, int]] | None = None,
) -> None:
    """Run ``removeDuplicateNodes`` under a tightened geometry tolerance and strip collapsed elements.

    Pass ``dimtags`` to scope the dedup to specific entities, or ``None``
    for a global pass. The previous ``Geometry.Tolerance`` is restored.
    """
    old_tol = gmsh.option.getNumber("Geometry.Tolerance")
    gmsh.option.setNumber("Geometry.Tolerance", _DEDUP_GEOMETRY_TOLERANCE)
    try:
        if dimtags is None:
            gmsh.model.mesh.removeDuplicateNodes()
        else:
            gmsh.model.mesh.removeDuplicateNodes(dimtags)
        _strip_degenerate_elements()
    finally:
        gmsh.option.setNumber("Geometry.Tolerance", old_tol)


def strip_synthetic_physical_groups() -> None:
    """Remove ``__cohort_*`` and ``__sweep*`` synthetic groups from gmsh.

    Called before writing the .msh so synthetic bookkeeping groups don't
    leak into the output.
    """
    to_remove: list[tuple[int, int]] = []
    names_to_drop: list[str] = []
    for dim, gtag in gmsh.model.getPhysicalGroups():
        gname = gmsh.model.getPhysicalName(dim, gtag)
        if gname.startswith(("__cohort_", "__sweep")):
            to_remove.append((dim, gtag))
            names_to_drop.append(gname)
    if to_remove:
        gmsh.model.removePhysicalGroups(to_remove)
        for gname in names_to_drop:
            with contextlib.suppress(Exception):
                gmsh.model.removePhysicalName(gname)


def has_cohort_groups() -> bool:
    """Return True if the active gmsh model contains any ``__cohort_*`` physical groups."""
    for dim, gtag in gmsh.model.getPhysicalGroups():
        if gmsh.model.getPhysicalName(dim, gtag).startswith("__cohort_"):
            return True
    return False


def discover_cohorts() -> tuple[
    dict[str, SlabMeta], dict[str, int], dict[str, int]
]:
    """Scan the loaded gmsh model's physical groups for ``__cohort_*`` synthetics.

    Reconstructs ``(slab_meta, face_tag_by_key, sub_solid_tag_by_key)`` via
    :meth:`SlabMeta.from_synthetic_groups`. If a legacy XAO omitted dim=2
    synthetic face groups, falls back to inspecting the 3D volume's boundary
    faces in gmsh.
    """
    face_groups: dict[str, int] = {}
    for _, gtag in gmsh.model.getPhysicalGroups(2):
        gname = gmsh.model.getPhysicalName(2, gtag)
        if not gname.startswith("__cohort_"):
            continue
        entities = gmsh.model.getEntitiesForPhysicalGroup(2, gtag)
        if len(entities) >= 1:
            face_groups[gname] = int(entities[0])

    solid_groups: dict[str, int] = {}
    fallback_names_by_vol: dict[int, list[str]] = defaultdict(list)
    for _, gtag in gmsh.model.getPhysicalGroups(3):
        gname = gmsh.model.getPhysicalName(3, gtag)
        entities = gmsh.model.getEntitiesForPhysicalGroup(3, gtag)
        if gname.startswith("__cohort_"):
            if len(entities) >= 1:
                solid_groups[gname] = int(entities[0])
        elif not gname.startswith("__"):
            for tag in entities:
                fallback_names_by_vol[int(tag)].append(gname)

    slab_meta, face_tag_by_key, sub_solid_tag_by_key = (
        SlabMeta.from_synthetic_groups(
            solid_groups=solid_groups,
            face_groups=face_groups,
            fallback_names_by_vol=fallback_names_by_vol,
        )
    )

    # Fallback for legacy XAOs where dim=2 face groups were not present:
    for sub_key, meta in list(slab_meta.items()):
        if (
            meta.bot_face_key in face_tag_by_key
            and meta.top_face_key in face_tag_by_key
        ):
            continue
        vtag = sub_solid_tag_by_key[sub_key]
        bnds = [
            int(t)
            for _, t in gmsh.model.getBoundary([(3, vtag)], oriented=False)
        ]
        horiz = sorted(
            (round(gmsh.model.getBoundingBox(2, t)[2], 6), t)
            for t in bnds
            if abs(
                gmsh.model.getBoundingBox(2, t)[5]
                - gmsh.model.getBoundingBox(2, t)[2]
            )
            < 1e-5
        )
        if len(horiz) >= 2:
            bot_tag, top_tag = horiz[0][1], horiz[-1][1]
            lat_tags = [t for t in bnds if t not in (bot_tag, top_tag)]
            bot_k = f"{sub_key}__bot"
            top_k = f"{sub_key}__top"
            face_tag_by_key[bot_k] = bot_tag
            face_tag_by_key[top_k] = top_tag
            lat_keys: list[str] = []
            for li, lt in enumerate(lat_tags):
                lk = f"{sub_key}__lat_{li}"
                face_tag_by_key[lk] = lt
                lat_keys.append(lk)
            slab_meta[sub_key] = SlabMeta(
                slab_index=meta.slab_index,
                physical_name=meta.physical_name,
                bot_face_key=bot_k,
                top_face_key=top_k,
                lateral_face_keys=tuple(lat_keys),
                keep=meta.keep,
            )

    return slab_meta, face_tag_by_key, sub_solid_tag_by_key


def make_cohort_hooks(
    resolution_specs: dict | None = None,
    point_tolerance: float = 1e-3,
    user_pre_2d: Callable[[], None] | None = None,
    user_pre_3d: Callable[[], None] | None = None,
    user_post_3d: Callable[[], None] | None = None,
) -> tuple[Callable[[], None], Callable[[], None], Callable[[], None]]:
    """Build ``(pre_2d_hook, pre_3d_hook, post_3d_hook)`` for cohort wedge meshing.

    Discovers cohort metadata from the loaded gmsh model inside ``pre_2d_hook``
    (before stripping synthetic groups) so multi-attempt fallback retries that
    reload ``cad_checkpoint.xao`` re-discover fresh gmsh tags cleanly.
    """
    state: dict[str, Any] = {}

    def _pre_2d() -> None:
        slab_meta, face_tag_by_key, sub_solid_tag_by_key = discover_cohorts()
        state["slab_meta"] = slab_meta
        state["face_tag_by_key"] = face_tag_by_key
        state["sub_solid_tag_by_key"] = sub_solid_tag_by_key
        if slab_meta and face_tag_by_key:
            freeze_lateral_mesh(
                slab_meta,
                face_tag_by_key,
                resolution_specs=resolution_specs,
                after_1d_hook=user_pre_2d,
            )
        elif user_pre_2d is not None:
            user_pre_2d()
        strip_synthetic_physical_groups()

    def _pre_3d() -> None:
        slab_meta = state.get("slab_meta")
        face_tag_by_key = state.get("face_tag_by_key")
        sub_solid_tag_by_key = state.get("sub_solid_tag_by_key")
        if slab_meta and face_tag_by_key and sub_solid_tag_by_key:
            stamp_wedges(
                slab_meta,
                face_tag_by_key,
                sub_solid_tag_by_key,
                resolution_specs=resolution_specs,
                point_tolerance=point_tolerance,
            )
            gmsh.option.setNumber("Mesh.MeshOnlyEmpty", 1)
            structured_vol_dimtags = [
                (3, tag) for tag in sub_solid_tag_by_key.values()
            ]
            _remove_duplicate_nodes_tight(structured_vol_dimtags)
        if user_pre_3d is not None:
            user_pre_3d()

    def _post_3d() -> None:
        slab_meta = state.get("slab_meta")
        face_tag_by_key = state.get("face_tag_by_key")
        sub_solid_tag_by_key = state.get("sub_solid_tag_by_key")
        if slab_meta and face_tag_by_key and sub_solid_tag_by_key:
            _remove_duplicate_nodes_tight()
        if user_post_3d is not None:
            user_post_3d()

    return _pre_2d, _pre_3d, _post_3d


def resolve_n_layers(
    physical_name: tuple[str, ...] | str,
    resolution_specs: dict | None,
) -> int:
    """Look up n_layers from resolution_specs for a physical_name tuple or string.

    Inspects all names in ``physical_name``. Returns 1 if no spec matches.
    Raises :class:`StructuredError` if a single name has more than one
    ``StructuredExtrusionResolutionSpec`` or if multiple names on the same
    slab specify conflicting ``n_layers``.
    """
    from meshwell.resolution import StructuredExtrusionResolutionSpec

    if not resolution_specs:
        return 1
    names = (
        (physical_name,)
        if isinstance(physical_name, str)
        else tuple(physical_name)
    )
    matched_specs: list[StructuredExtrusionResolutionSpec] = []
    for key in names:
        specs = [
            s
            for s in resolution_specs.get(key, [])
            if isinstance(s, StructuredExtrusionResolutionSpec)
        ]
        if len(specs) > 1:
            raise StructuredError(
                f"physical_name {key!r} has {len(specs)} "
                "StructuredExtrusionResolutionSpec entries; expected at most 1."
            )
        if specs:
            matched_specs.append(specs[0])
    if not matched_specs:
        return 1
    distinct_n_layers = {s.n_layers for s in matched_specs}
    if len(distinct_n_layers) > 1:
        raise StructuredError(
            f"physical_name {names!r} has conflicting "
            f"StructuredExtrusionResolutionSpec n_layers {sorted(distinct_n_layers)}."
        )
    return matched_specs[0].n_layers


# ---------------------------------------------------------------------------
# Freeze cohort lateral mesh before generate(2)
# ---------------------------------------------------------------------------


def _classify_lateral_face_edges(
    face_tag: int,
    z_bot: float,
    z_top: float,
    z_tol: float = 1e-7,
) -> tuple[int | None, int | None, list[int]]:
    """Return (bot_edge_tag, top_edge_tag, [vertical_edge_tags])."""
    edges = gmsh.model.getBoundary([(2, face_tag)], oriented=False, recursive=False)
    bot_edge = None
    top_edge = None
    vertical: list[int] = []
    for _dim, etag in edges:
        ev = gmsh.model.getBoundary([(1, etag)], oriented=False, recursive=False)
        zs = []
        for _vd, vt in ev:
            pos = gmsh.model.getValue(0, vt, [])
            zs.append(pos[2])
        if len(zs) != 2:
            continue
        if abs(zs[0] - z_bot) < z_tol and abs(zs[1] - z_bot) < z_tol:
            bot_edge = etag
        elif abs(zs[0] - z_top) < z_tol and abs(zs[1] - z_top) < z_tol:
            top_edge = etag
        else:
            vertical.append(etag)
    return bot_edge, top_edge, vertical


def _ordered_curve_nodes(
    curve_tag: int,
) -> list[tuple[int, float, float, float]]:
    """Return curve nodes [(tag, x, y, z)] sorted by parametric coord."""
    tags, coord, param = gmsh.model.mesh.getNodes(
        1, curve_tag, includeBoundary=True, returnParametricCoord=True
    )
    if len(tags) == 0:
        return []
    items = []
    for i, t in enumerate(tags):
        items.append(
            (
                int(t),
                float(param[i]),
                float(coord[3 * i]),
                float(coord[3 * i + 1]),
                float(coord[3 * i + 2]),
            )
        )
    items.sort(key=lambda r: r[1])
    return [(t, x, y, z) for t, _p, x, y, z in items]


def _align_top_to_bot(
    bot_row: list[tuple[int, float, float, float]],
    top_row: list[tuple[int, float, float, float]],
) -> list[tuple[int, float, float, float]]:
    """Reverse top_row if its parametric direction runs opposite to bot."""
    if len(top_row) < 2:
        return top_row
    bot_first = bot_row[0]
    d_forward = (top_row[0][1] - bot_first[1]) ** 2 + (
        top_row[0][2] - bot_first[2]
    ) ** 2
    d_reverse = (top_row[-1][1] - bot_first[1]) ** 2 + (
        top_row[-1][2] - bot_first[2]
    ) ** 2
    return list(reversed(top_row)) if d_reverse < d_forward else top_row


def _vertical_edge_layer_nodes(vertical_edge_tag: int) -> list[int]:
    """Return the vertical edge's nodes ordered z_low -> z_high.

    Sorts by the actual z-coordinate of each node rather than the
    edge's parametric coordinate. Robust to OCC/BOP orientation
    flips that would reverse the parametric direction. Assumes
    ``setTransfiniteCurve(vertical_edge_tag, n_layers + 1)`` was
    called before ``generate(1)`` so there are exactly
    ``n_layers + 1`` uniformly-spaced nodes.
    """
    tags, coord, _param = gmsh.model.mesh.getNodes(
        1,
        vertical_edge_tag,
        includeBoundary=True,
        returnParametricCoord=True,
    )
    # coord is flat [x0,y0,z0, x1,y1,z1, ...]; sort by z (index 2 of each triple)
    items = sorted(
        ((int(t), float(coord[3 * i + 2])) for i, t in enumerate(tags)),
        key=lambda r: r[1],
    )
    return [t for t, _z in items]


def _copy_curve_nodes_to_partner(
    src_row: list[tuple[int, float, float, float]],
    dst_edge_tag: int,
    z_dst: float,
) -> None:
    """Mirror ``src_row``'s (x, y) node layout onto ``dst_edge_tag`` at ``z_dst``."""
    if len(src_row) < 2:
        return
    end_pts = gmsh.model.getBoundary([(1, dst_edge_tag)], oriented=False)
    if len(end_pts) != 2:
        return
    ep_info: list[tuple[float, float, int]] = []
    for _d, ptag in end_pts:
        ptag = abs(int(ptag))
        xyz = gmsh.model.getValue(0, ptag, [])
        ntags, _c, _p = gmsh.model.mesh.getNodes(0, ptag)
        if len(ntags) == 0:
            new_pt_node = gmsh.model.mesh.getMaxNodeTag() + 1
            gmsh.model.mesh.addNodes(
                0, ptag, [new_pt_node], [float(xyz[0]), float(xyz[1]), float(z_dst)]
            )
            ep_info.append((float(xyz[0]), float(xyz[1]), new_pt_node))
        else:
            ep_info.append((float(xyz[0]), float(xyz[1]), int(ntags[0])))

    (pmin,), (pmax,) = gmsh.model.getParametrizationBounds(1, dst_edge_tag)
    base = np.array(gmsh.model.getValue(1, dst_edge_tag, [pmin])[:2])
    far = np.array(gmsh.model.getValue(1, dst_edge_tag, [pmax])[:2])
    span = far - base
    length2 = float(span @ span)

    def _frac(x: float, y: float) -> float:
        if length2 == 0.0:
            return 0.0
        return float(((np.array([x, y]) - base) @ span) / length2)

    ordered_xy = sorted(
        ((_frac(x, y), x, y) for _t, x, y, _z in src_row),
        key=lambda r: r[0],
    )
    _, x0, y0 = ordered_xy[0]
    d0_to_ep0 = (x0 - ep_info[0][0]) ** 2 + (y0 - ep_info[0][1]) ** 2
    d0_to_ep1 = (x0 - ep_info[1][0]) ** 2 + (y0 - ep_info[1][1]) ** 2
    if d0_to_ep0 <= d0_to_ep1:
        start_node, end_node = ep_info[0][2], ep_info[1][2]
    else:
        start_node, end_node = ep_info[1][2], ep_info[0][2]

    gmsh.model.mesh.clear([(1, dst_edge_tag)])
    gmsh.model.mesh.setNode(
        start_node,
        [float(ordered_xy[0][1]), float(ordered_xy[0][2]), float(z_dst)],
        [],
    )
    gmsh.model.mesh.setNode(
        end_node,
        [float(ordered_xy[-1][1]), float(ordered_xy[-1][2]), float(z_dst)],
        [],
    )

    seq = [start_node]
    interior_tags: list[int] = []
    interior_coords: list[float] = []
    interior_params: list[float] = []
    next_tag = gmsh.model.mesh.getMaxNodeTag() + 1
    for frac, x, y in ordered_xy[1:-1]:
        seq.append(next_tag)
        interior_tags.append(next_tag)
        interior_coords.extend([float(x), float(y), float(z_dst)])
        interior_params.append(float(pmin + frac * (pmax - pmin)))
        next_tag += 1
    seq.append(end_node)
    if interior_tags:
        gmsh.model.mesh.addNodes(
            1, dst_edge_tag, interior_tags, interior_coords, interior_params
        )
    lines: list[int] = []
    for a, b in itertools.pairwise(seq):
        lines.extend([a, b])
    gmsh.model.mesh.addElementsByType(dst_edge_tag, 1, [], lines)


def _emit_lateral_face_quads(
    face_tag: int,
    z_bot: float,
    z_top: float,
    n_layers: int,
    owners_per_face: dict[int, list[tuple[int, int]]],
) -> None:
    """Emit the structured quad mesh for one cohort lateral face.

    Builds ``n_layers + 1`` rows of node tags between the bot and top
    edges, reusing the transfinite-placed vertical-edge nodes for the
    left/right endpoints, and connects consecutive rows with quads.
    """
    bot_edge, top_edge, verticals = _classify_lateral_face_edges(face_tag, z_bot, z_top)
    if bot_edge is None or top_edge is None or len(verticals) != 2:
        raise StructuredTransfiniteRejectedError(
            face_tag=face_tag,
            slab_index=owners_per_face[face_tag][0][0],
            reason=(
                f"lateral face must have bot + top + 2 vertical edges; "
                f"got bot={bot_edge}, top={top_edge}, vert={len(verticals)}"
            ),
        )

    bot_row = _ordered_curve_nodes(bot_edge)
    top_row = _ordered_curve_nodes(top_edge)
    if len(bot_row) != len(top_row):
        if len(bot_row) > len(top_row) and len(bot_row) >= 2:
            _copy_curve_nodes_to_partner(bot_row, top_edge, z_top)
            top_row = _ordered_curve_nodes(top_edge)
        elif len(top_row) > len(bot_row) and len(top_row) >= 2:
            _copy_curve_nodes_to_partner(top_row, bot_edge, z_bot)
            bot_row = _ordered_curve_nodes(bot_edge)
    top_row = _align_top_to_bot(bot_row, top_row)
    if len(bot_row) < 2 or len(top_row) != len(bot_row):
        logger.warning(
            "Slab %s: lateral face %s skipped because bot_row len (%s) != "
            "top_row len (%s)",
            owners_per_face[face_tag][0][0],
            face_tag,
            len(bot_row),
            len(top_row),
        )
        return

    # Pick left/right vertical edges by (x, y) proximity to bot row endpoints.
    left_xy = (bot_row[0][1], bot_row[0][2])
    right_xy = (bot_row[-1][1], bot_row[-1][2])
    left_vert = right_vert = None
    for ve in verticals:
        ev = gmsh.model.getBoundary([(1, ve)], oriented=False, recursive=False)
        x_v = y_v = None
        for _vd, vt in ev:
            pos = gmsh.model.getValue(0, vt, [])
            x_v, y_v = pos[0], pos[1]
            break
        d_left = (x_v - left_xy[0]) ** 2 + (y_v - left_xy[1]) ** 2
        d_right = (x_v - right_xy[0]) ** 2 + (y_v - right_xy[1]) ** 2
        if d_left < d_right:
            left_vert = ve
        else:
            right_vert = ve
    if left_vert is None or right_vert is None:
        return

    # Reuse transfinite-placed vertical-edge nodes (no duplicates).
    left_layer_nodes = _vertical_edge_layer_nodes(left_vert)
    right_layer_nodes = _vertical_edge_layer_nodes(right_vert)

    # Build n_layers+1 rows of node tags.
    rows: list[list[int]] = [[t for t, _x, _y, _z in bot_row]]
    for layer in range(1, n_layers):
        z_layer = z_bot + (z_top - z_bot) * layer / n_layers
        row_tags: list[int] = []
        for idx, (_t, x, y, _z) in enumerate(bot_row):
            if idx == 0:
                row_tags.append(left_layer_nodes[layer])
            elif idx == len(bot_row) - 1:
                row_tags.append(right_layer_nodes[layer])
            else:
                new_tag = gmsh.model.mesh.getMaxNodeTag() + 1
                gmsh.model.mesh.addNodes(2, face_tag, [new_tag], [x, y, z_layer])
                row_tags.append(new_tag)
        rows.append(row_tags)
    rows.append([t for t, _x, _y, _z in top_row])

    # Emit quad elements (gmsh type 3 = 4-node quad).
    quad_nodes: list[int] = []
    for r in range(len(rows) - 1):
        for c in range(len(rows[r]) - 1):
            quad_nodes.extend(
                [
                    rows[r][c],
                    rows[r][c + 1],
                    rows[r + 1][c + 1],
                    rows[r + 1][c],
                ]
            )
    if quad_nodes:
        gmsh.model.mesh.addElementsByType(face_tag, 3, [], quad_nodes)


def freeze_lateral_mesh(
    slab_meta: dict[ShapeKey, SlabMeta],
    face_tag_by_key: dict[ShapeKey, int],
    resolution_specs: dict[str, list] | None = None,
    after_1d_hook: Callable[[], None] | None = None,
) -> None:
    """Pre_2d hook: emit cohort lateral-face mesh before generate(2).

    Mechanism:
      1. Validate n_layers consistency on every shared lateral face.
      2. Set transfinite on vertical edges so generate(1) places
         exactly n_layers+1 nodes per vertical edge (uniformly
         spaced in parametric coord).
      3. Call generate(1) explicitly so we have edge nodes available
         to walk in step 4 (and invoke ``after_1d_hook`` if present
         so 2D sweeps stamp their curves/faces after the single 1D pass).
      4. For each lateral face: walk bot/top edge nodes in parametric
         order; reuse the vertical-edge transfinite nodes for the
         left/right endpoints at each layer; create face-interior
         nodes for the rest. Emit quad elements connecting layer rows.
      5. Set Mesh.MeshOnlyEmpty=1 so the outer generate(2) skips
         faces that already have a mesh.

    This never invokes gmsh's 2D mesher or its periodic-surface mesher
    on cohort lateral faces — both sources of past failures.
    """
    # Step 1: per-face n_layers + consistency check.
    logger.debug(
        "freeze_lateral_mesh: checking lateral faces for %d slabs", len(slab_meta)
    )
    owners_per_face: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for meta in slab_meta.values():
        if not meta.keep:
            continue
        n_layers = resolve_n_layers(meta.physical_name, resolution_specs)
        for fk in meta.lateral_face_keys:
            tag = face_tag_by_key.get(fk)
            if tag is None:
                logger.warning(
                    "Slab %s (%s): lateral face key %s not found in face_tag_by_key",
                    meta.slab_index,
                    meta.physical_name,
                    fk,
                )
                continue
            owners_per_face[tag].append((meta.slab_index, n_layers))

    face_n_layers: dict[int, int] = {}
    for face_tag, owners in owners_per_face.items():
        n_layers_set = {n for _, n in owners}
        if len(n_layers_set) > 1:
            (sa, na), (sb, nb) = owners[0], owners[1]
            raise StructuredLateralNLayersMismatchError(
                slab_a=sa,
                slab_b=sb,
                face_tag=face_tag,
                n_layers_a=na,
                n_layers_b=nb,
            )
        face_n_layers[face_tag] = owners[0][1]

    # Per-face z bounds (needed for edge classification).
    face_z_bounds: dict[int, tuple[float, float]] = {}
    for meta in slab_meta.values():
        if not meta.keep:
            continue
        bot_tag = face_tag_by_key.get(meta.bot_face_key)
        top_tag = face_tag_by_key.get(meta.top_face_key)
        if bot_tag is None or top_tag is None:
            continue
        z_bot = _face_centroid_z(bot_tag)
        z_top = _face_centroid_z(top_tag)
        for fk in meta.lateral_face_keys:
            tag = face_tag_by_key.get(fk)
            if tag is None:
                continue
            face_z_bounds[tag] = (z_bot, z_top)

    sweep_curve_tags: set[int] = set()
    for _dim, gtag in gmsh.model.getPhysicalGroups(2):
        if gmsh.model.getPhysicalName(2, gtag).startswith("__sweep|"):
            for ftag in gmsh.model.getEntitiesForPhysicalGroup(2, gtag):
                for _d, ct in gmsh.model.getBoundary(
                    [(2, int(ftag))], oriented=False
                ):
                    sweep_curve_tags.add(abs(int(ct)))

    # Step 2: setTransfiniteCurve on vertical edges and setPeriodic on top/bot curves.
    vertical_edges_done: set[int] = set()
    periodic_edges_done: set[int] = set()
    sweep_partner_pairs: list[tuple[int, int, float, float]] = []
    for face_tag, (z_bot, z_top) in face_z_bounds.items():
        n_layers = face_n_layers.get(face_tag, 1)
        bot_edge, top_edge, verticals = _classify_lateral_face_edges(
            face_tag, z_bot, z_top
        )

        for ve in verticals:
            if ve in vertical_edges_done:
                continue
            vertical_edges_done.add(ve)
            gmsh.model.mesh.setTransfiniteCurve(ve, n_layers + 1)

        if (
            bot_edge is not None
            and top_edge is not None
            and top_edge not in periodic_edges_done
        ):
            periodic_edges_done.add(top_edge)
            if bot_edge in sweep_curve_tags or top_edge in sweep_curve_tags:
                sweep_partner_pairs.append((bot_edge, top_edge, z_bot, z_top))
            else:
                dz = z_top - z_bot
                transform = [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    float(dz),
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]
                try:
                    gmsh.model.mesh.setPeriodic(
                        1, [top_edge], [bot_edge], transform
                    )
                except Exception as per_err:
                    logger.warning(
                        "Failed to set periodic constraint for top_edge %s -> "
                        "bot_edge %s: %s",
                        top_edge,
                        bot_edge,
                        per_err,
                    )

    # Step 3: materialise 1D mesh, then invoke after_1d_hook (e.g. 2D sweep
    # stamping) so sweep curves and 2D sweep faces are stamped AFTER the single
    # generate(1) call without being wiped by a second generate(1).
    gmsh.model.mesh.generate(1)
    if after_1d_hook is not None:
        after_1d_hook()

    for bot_edge, top_edge, z_bot, z_top in sweep_partner_pairs:
        if bot_edge in sweep_curve_tags and top_edge not in sweep_curve_tags:
            _copy_curve_nodes_to_partner(
                _ordered_curve_nodes(bot_edge), top_edge, z_top
            )
        elif top_edge in sweep_curve_tags and bot_edge not in sweep_curve_tags:
            _copy_curve_nodes_to_partner(
                _ordered_curve_nodes(top_edge), bot_edge, z_bot
            )

    # Step 4: emit lateral-face quads.
    for face_tag, (z_bot, z_top) in face_z_bounds.items():
        _emit_lateral_face_quads(
            face_tag, z_bot, z_top, face_n_layers[face_tag], owners_per_face
        )

    # Step 4.5: prevent generate(2) from meshing cohort top faces.
    # We will mesh them in stamp_wedges. If generate(2) meshes them,
    # its interior nodes become orphaned when we overwrite the elements,
    # causing PLC errors in the 3D mesher.
    for meta in slab_meta.values():
        if not meta.keep:
            continue
        top_tag = face_tag_by_key.get(meta.top_face_key)
        if top_tag is None:
            continue
        elem_types, _, _ = gmsh.model.mesh.getElements(2, top_tag)
        if len(elem_types) == 0:
            edges = gmsh.model.getBoundary(
                [(2, top_tag)], oriented=False, recursive=False
            )
            boundary_nodes = []
            for _, etag in edges:
                tags, _, _ = gmsh.model.mesh.getNodes(1, etag)
                boundary_nodes.extend(tags)
            boundary_nodes = list(set(boundary_nodes))
            if len(boundary_nodes) >= 3:
                gmsh.model.mesh.addElementsByType(
                    top_tag,
                    2,
                    [],
                    [
                        int(boundary_nodes[0]),
                        int(boundary_nodes[1]),
                        int(boundary_nodes[2]),
                    ],
                )

    # Step 5: tell outer generate(2) to skip already-meshed faces.
    gmsh.option.setNumber("Mesh.MeshOnlyEmpty", 1)


# ---------------------------------------------------------------------------
# pre_3d_hook: stamp_wedges
# ---------------------------------------------------------------------------


def stamp_wedges(
    slab_meta: dict[ShapeKey, SlabMeta],
    face_tag_by_key: dict[ShapeKey, int],
    sub_solid_tag_by_key: dict[ShapeKey, int],
    resolution_specs: dict | None = None,
    point_tolerance: float = 1e-3,
) -> None:
    """For each cohort sub-solid: read bot tri mesh, stamp on top.

    Emit n_layers wedges per bot triangle into the sub-solid's 3D tag.

    n_layers per sub-solid resolved from resolution_specs via the
    slab's physical_name (defaults to 1 if no spec present).

    Iterates sub-solids in z_lo ascending order so shared bot/top
    faces are stamped from below before being read from above.
    """
    # Order sub-solids by zlo via their bot face z.
    order: list[tuple[float, ShapeKey, SlabMeta]] = []
    for k, meta in slab_meta.items():
        if not meta.keep:
            # Voids: their bodies are excluded from BREP by the XAO writer
            # (keep=False), so they have no gmsh volume tag and no faces to
            # stamp. Skip them outright.
            continue
        bot_tag = face_tag_by_key.get(meta.bot_face_key)
        if bot_tag is None:
            continue
        z = _face_centroid_z(bot_tag)
        order.append((z, k, meta))
    order.sort(key=lambda t: t[0])

    for idx, (_, sub_key, meta) in enumerate(order):
        bot_tag = face_tag_by_key[meta.bot_face_key]
        top_tag = face_tag_by_key[meta.top_face_key]
        vol_tag = sub_solid_tag_by_key[sub_key]
        n_layers = resolve_n_layers(meta.physical_name, resolution_specs)
        logger.debug(
            "stamp_wedges: stamping slab %d/%d (name=%s, n_layers=%s) "
            "bot_tag=%s, top_tag=%s, vol_tag=%s",
            idx + 1,
            len(order),
            meta.physical_name,
            n_layers,
            bot_tag,
            top_tag,
            vol_tag,
        )
        _stamp_one(
            bot_tag,
            top_tag,
            vol_tag,
            meta,
            n_layers,
            point_tolerance,
            face_tag_by_key,
        )


def _face_centroid_z(face_tag: int) -> float:
    """Return the z-coordinate of a face's centroid.

    XY-extruded faces have a flat z.
    """
    bbox = gmsh.model.getBoundingBox(2, face_tag)
    return (bbox[2] + bbox[5]) / 2


def _match_and_create_layer_nodes(
    bot_pts,
    bot_node_tags,
    boundary_node_tags,
    candidate_tags,
    candidate_pts,
    z_target: float,
    add_dim: int,
    add_tag: int,
    snap_tolerance: float,
) -> tuple[dict[int, int], int]:
    """Map every bot-face node index to a node tag at height ``z_target``.

    Boundary bot nodes within ``snap_tolerance`` (in XY) of a candidate
    node reuse that candidate's tag (avoids duplicate positions that a
    later removeDuplicateNodes would merge non-deterministically); every
    remaining bot node is bulk-created at ``(x, y, z_target)`` on the
    ``(add_dim, add_tag)`` entity.

    Returns ``(idx_to_tag, n_unmatched_boundary)`` where ``idx_to_tag`` is
    keyed by position in ``bot_node_tags``.
    """
    matched: dict[int, int] = {}
    unmatched: list[int] = []
    if len(candidate_tags):
        boundary_indices = [
            i for i, t in enumerate(bot_node_tags) if int(t) in boundary_node_tags
        ]
        if boundary_indices:
            boundary_pts = bot_pts[boundary_indices]
            tree = KDTree(candidate_pts[:, :2])
            distances, indices = tree.query(
                boundary_pts[:, :2], distance_upper_bound=snap_tolerance
            )
            for b_idx, dist, idx in zip(boundary_indices, distances, indices):
                if dist < snap_tolerance:
                    matched[b_idx] = int(candidate_tags[idx])
                else:
                    unmatched.append(b_idx)
            interior_indices = [i for i in range(len(bot_pts)) if i not in matched]
        else:
            interior_indices = list(range(len(bot_pts)))
    else:
        interior_indices = list(range(len(bot_pts)))

    idx_to_tag: dict[int, int] = {}
    if interior_indices:
        max_tag = gmsh.model.mesh.getMaxNodeTag()
        new_tags = list(range(max_tag + 1, max_tag + 1 + len(interior_indices)))
        coords: list[float] = []
        for idx in interior_indices:
            bpt = bot_pts[idx]
            coords.extend([float(bpt[0]), float(bpt[1]), float(z_target)])
        gmsh.model.mesh.addNodes(add_dim, add_tag, new_tags, coords)
        idx_to_tag.update(zip(interior_indices, new_tags))
    idx_to_tag.update(matched)
    return idx_to_tag, len(unmatched)


def _stamp_one(
    bot_tag: int,
    top_tag: int,
    vol_tag: int,
    meta: SlabMeta,
    n_layers: int,
    point_tolerance: float,
    face_tag_by_key: dict[ShapeKey, int],
) -> None:
    """Read bot triangulation, stamp on top, emit wedges into volume."""
    snap_tolerance = max(1e-5, point_tolerance * 0.1)
    # 1) Read bot triangulation / quad mesh.
    elem_types, _elem_tags, node_tags = gmsh.model.mesh.getElements(2, bot_tag)
    if 2 not in elem_types and 3 not in elem_types:
        return
    if 2 in elem_types:
        tri_idx = list(elem_types).index(2)
        tris = np.array(node_tags[tri_idx]).reshape(-1, 3)
    else:
        tris = np.zeros((0, 3), dtype=int)
    if 3 in elem_types:
        quad_idx = list(elem_types).index(3)
        quads = np.array(node_tags[quad_idx]).reshape(-1, 4)
    else:
        quads = np.zeros((0, 4), dtype=int)
    bot_node_tags, bot_coord, _ = gmsh.model.mesh.getNodes(
        2, bot_tag, includeBoundary=True
    )
    bot_pts = np.array(bot_coord).reshape(-1, 3)
    bot_z = bot_pts[:, 2].mean()

    # Get boundary nodes of the bottom face. Only boundary nodes
    # can snap to the lateral faces or existing top face boundary nodes.
    edges = gmsh.model.getBoundary([(2, bot_tag)], oriented=False, recursive=False)
    boundary_node_tags = []
    for _, etag in edges:
        tags, _, _ = gmsh.model.mesh.getNodes(1, etag, includeBoundary=True)
        boundary_node_tags.extend(tags)
    boundary_node_tags = {int(t) for t in boundary_node_tags}

    # 2) Determine top z from top face bbox.
    bbox = gmsh.model.getBoundingBox(2, top_tag)
    top_z = bbox[5]
    dz = (top_z - bot_z) / n_layers

    # 3) Snapshot all existing top-face nodes (boundary + interior).
    #
    # We ALWAYS snapshot nodes BEFORE modifying the top face so we can
    # reuse them rather than creating new nodes at the same positions.
    # Using gmsh.model.mesh.clear([(2, top_tag)]) discards elements but
    # leaves orphaned interior nodes floating in the global model: those
    # nodes survive with no element association, get picked up by
    # generate(3) inside adjacent unstructured volumes (e.g. a cap above
    # this slab), and produce duplicate node positions in the final mesh.
    # More critically, a global removeDuplicateNodes() then randomly
    # resolves which tag "wins" the merge, corrupting the boundary mesh
    # of the adjacent volume and causing generate(3) to silently skip it
    # (~40% failure rate when a structured slab is topped by an
    # unstructured neighbour).
    #
    # Instead: remove only the ELEMENTS via removeElements() (which does
    # not touch nodes), then rebuild with the same node tags matched by
    # XY proximity.  Any bot node that has no nearby existing top node
    # gets a fresh tag — but the only case where this happens is the very
    # first stamp of a top face that has never been meshed, e.g. a slab
    # top that is in the interior of the cohort and has no generate(2)
    # mesh yet.  In all other cases (top face is shared with a neighbour
    # that generate(2) already touched) we reuse existing tags end-to-end,
    # producing zero orphaned nodes and zero duplicate positions.
    existing_top_nodes, existing_top_coord, _ = gmsh.model.mesh.getNodes(
        2, top_tag, includeBoundary=True
    )
    existing_top_pts = (
        np.array(existing_top_coord).reshape(-1, 3)
        if len(existing_top_coord)
        else np.zeros((0, 3))
    )

    # bot_node_tag -> top_node_tag map.
    bot_to_top: dict[int, int] = {}

    # 3a) Snap/create top-face nodes for every bot node. Boundary nodes
    # reuse existing top-face nodes (shared with adjacent volumes);
    # the rest are created on the top face at top_z.
    top_idx_to_tag, mismatched = _match_and_create_layer_nodes(
        bot_pts,
        bot_node_tags,
        boundary_node_tags,
        existing_top_nodes,
        existing_top_pts,
        z_target=top_z,
        add_dim=2,
        add_tag=top_tag,
        snap_tolerance=snap_tolerance,
    )
    for idx, tag in top_idx_to_tag.items():
        bot_to_top[int(bot_node_tags[idx])] = tag

    # Remove existing top-face elements WITHOUT removing nodes (so we do
    # not orphan interior nodes that are shared with adjacent volumes).
    # Then re-stamp with the bot-matched triangulation / quad mesh.
    gmsh.model.mesh.removeElements(2, top_tag)
    if len(tris):
        top_tri_nodes: list[int] = []
        for tri in tris:
            top_tri_nodes.extend(bot_to_top[int(t)] for t in tri)
        gmsh.model.mesh.addElementsByType(top_tag, 2, [], top_tri_nodes)
    if len(quads):
        top_quad_nodes: list[int] = []
        for quad in quads:
            top_quad_nodes.extend(bot_to_top[int(t)] for t in quad)
        gmsh.model.mesh.addElementsByType(top_tag, 3, [], top_quad_nodes)

    # 4) Intermediate layer nodes (for n_layers > 1).
    #
    # Boundary nodes of the bot face are shared with the lateral faces of
    # this sub-solid.  After generate(2), those lateral faces already have
    # transfinite-placed nodes at every intermediate z (z_layer).  Creating
    # brand-new intermediate nodes for boundary bot positions would produce
    # duplicate positions that removeDuplicateNodes() must merge — and that
    # merge is non-deterministic, sometimes corrupting adjacent-volume
    # boundary meshes and causing generate(3) to silently skip them.
    #
    # Fix: build a rounded-position → existing_node_tag lookup from the
    # current global model state at the start of each intermediate layer.
    # Reuse an existing node whenever one is within point_tolerance at the
    # target (x, y, z_layer) position; create a new node only when no
    # existing node is close enough (i.e. for interior bot positions that
    # have no pre-existing lateral-face node at z_layer).
    bot_idx_by_tag = {int(t): i for i, t in enumerate(bot_node_tags)}
    layer_maps: list[dict[int, int]] = [
        {i: int(bot_node_tags[i]) for i in range(len(bot_node_tags))}
    ]
    if n_layers > 1:
        # Resolve lateral face tags for this slab.
        lateral_face_tags = [
            face_tag_by_key[fk]
            for fk in meta.lateral_face_keys
            if fk in face_tag_by_key
        ]

        # Snapshot ONLY current slab's lateral nodes ONCE outside the loop.
        all_lateral_tags = []
        all_lateral_coord = []
        for lf_tag in lateral_face_tags:
            tags, coords, _ = gmsh.model.mesh.getNodes(2, lf_tag, includeBoundary=True)
            all_lateral_tags.extend(tags)
            all_lateral_coord.extend(coords)

        if all_lateral_tags:
            # Deduplicate by tag since shared curves return duplicate nodes.
            unique_indices = []
            seen = set()
            for idx, tag in enumerate(all_lateral_tags):
                if tag not in seen:
                    seen.add(tag)
                    unique_indices.append(idx)

            all_lateral_tags = np.array(all_lateral_tags)[unique_indices]
            all_lateral_pts = np.array(all_lateral_coord).reshape(-1, 3)[unique_indices]
        else:
            all_lateral_tags = np.array([])
            all_lateral_pts = np.zeros((0, 3))

        for layer in range(1, n_layers):
            z_layer = bot_z + dz * layer

            # Filter to nodes near z_layer for efficiency.
            if len(all_lateral_tags):
                z_mask = np.abs(all_lateral_pts[:, 2] - z_layer) < point_tolerance
                zlayer_tags = all_lateral_tags[z_mask]
                zlayer_pts = all_lateral_pts[z_mask]
            else:
                zlayer_tags = []
                zlayer_pts = np.zeros((0, 3))

            # Boundary nodes reuse this slab's lateral-face nodes already
            # placed at z_layer; interior nodes are created in the volume.
            this_map, _ = _match_and_create_layer_nodes(
                bot_pts,
                bot_node_tags,
                boundary_node_tags,
                zlayer_tags,
                zlayer_pts,
                z_target=z_layer,
                add_dim=3,
                add_tag=vol_tag,
                snap_tolerance=snap_tolerance,
            )
            layer_maps.append(this_map)
    layer_maps.append(
        {i: bot_to_top[int(bot_node_tags[i])] for i in range(len(bot_node_tags))}
    )

    # 5) Emit wedges (type 6 = 6-node prism) and hexahedra (type 5 = 8-node hex).
    wedge_node_tags: list[int] = []
    hex_node_tags: list[int] = []
    expected = 0
    for layer in range(n_layers):
        bot_map = layer_maps[layer]
        top_map = layer_maps[layer + 1]
        for tri in tris:
            b0, b1, b2 = (bot_idx_by_tag[int(t)] for t in tri)
            # Ensure positive volume by checking triangle orientation relative to extrusion
            p0 = bot_pts[b0]
            p1 = bot_pts[b1]
            p2 = bot_pts[b2]
            v1_x = p1[0] - p0[0]
            v1_y = p1[1] - p0[1]
            v2_x = p2[0] - p0[0]
            v2_y = p2[1] - p0[1]
            cross_z = v1_x * v2_y - v1_y * v2_x
            if cross_z * dz < 0:
                b1_p, b2_p = b2, b1
            else:
                b1_p, b2_p = b1, b2
            wedge_node_tags.extend(
                [
                    bot_map[b0],
                    bot_map[b1_p],
                    bot_map[b2_p],
                    top_map[b0],
                    top_map[b1_p],
                    top_map[b2_p],
                ]
            )
            expected += 1
        for quad in quads:
            b0, b1, b2, b3 = (bot_idx_by_tag[int(t)] for t in quad)
            p0 = bot_pts[b0]
            p1 = bot_pts[b1]
            p2 = bot_pts[b2]
            v1_x = p1[0] - p0[0]
            v1_y = p1[1] - p0[1]
            v2_x = p2[0] - p0[0]
            v2_y = p2[1] - p0[1]
            cross_z = v1_x * v2_y - v1_y * v2_x
            if cross_z * dz < 0:
                b1_p, b2_p, b3_p = b3, b2, b1
            else:
                b1_p, b2_p, b3_p = b1, b2, b3
            hex_node_tags.extend(
                [
                    bot_map[b0],
                    bot_map[b1_p],
                    bot_map[b2_p],
                    bot_map[b3_p],
                    top_map[b0],
                    top_map[b1_p],
                    top_map[b2_p],
                    top_map[b3_p],
                ]
            )
            expected += 1
    if wedge_node_tags:
        gmsh.model.mesh.addElementsByType(vol_tag, 6, [], wedge_node_tags)
    if hex_node_tags:
        gmsh.model.mesh.addElementsByType(vol_tag, 5, [], hex_node_tags)
    emitted = (len(wedge_node_tags) // 6) + (len(hex_node_tags) // 8)
    if emitted != expected:
        raise WedgeCountMismatchError(
            slab_index=meta.slab_index,
            expected=expected,
            got=emitted,
        )
    if mismatched:
        raise WedgeBotNodeMismatchError(
            slab_index=meta.slab_index,
            mismatched_count=mismatched,
        )
