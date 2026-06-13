"""Stage 5 — gmsh meshing hooks for structured cohorts.

pre_2d_hook (freeze_lateral_mesh): emits cohort lateral-face quad
mesh from Python before gmsh's generate(2) runs. Uses
Mesh.MeshOnlyEmpty=1 so the outer 2D mesher leaves cohort laterals
alone. Raises on n_layers mismatch or unsupported lateral topology.

pre_3d_hook (stamp_wedges): per cohort sub-solid, copies bot
triangulation to top and emits wedge elements.
"""
from __future__ import annotations

import logging
from collections import defaultdict

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


def resolve_n_layers(
    physical_name: tuple[str, ...] | str,
    resolution_specs: dict | None,
) -> int:
    """Look up n_layers from resolution_specs for one physical_name.

    Returns 1 if no spec. Raises StructuredError if more than one
    StructuredExtrusionResolutionSpec is present for the name.
    """
    from meshwell.resolution import StructuredExtrusionResolutionSpec

    if not resolution_specs:
        return 1
    key = physical_name[0] if isinstance(physical_name, tuple) else physical_name
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
    return specs[0].n_layers if specs else 1


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
    bot_edge, top_edge, verticals = _classify_lateral_face_edges(
        face_tag, z_bot, z_top
    )
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
    top_row = _align_top_to_bot(bot_row, _ordered_curve_nodes(top_edge))
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
) -> None:
    """Pre_2d hook: emit cohort lateral-face mesh before generate(2).

    Mechanism:
      1. Validate n_layers consistency on every shared lateral face.
      2. Set transfinite on vertical edges so generate(1) places
         exactly n_layers+1 nodes per vertical edge (uniformly
         spaced in parametric coord).
      3. Call generate(1) explicitly so we have edge nodes available
         to walk in step 4.
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
    logger.debug("freeze_lateral_mesh: checking lateral faces for %d slabs", len(slab_meta))
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

    # Step 2: setTransfiniteCurve on vertical edges and setPeriodic on top/bot curves.
    vertical_edges_done: set[int] = set()
    periodic_edges_done: set[int] = set()
    for face_tag, (z_bot, z_top) in face_z_bounds.items():
        n_layers = face_n_layers.get(face_tag, 1)
        bot_edge, top_edge, verticals = _classify_lateral_face_edges(face_tag, z_bot, z_top)
        
        for ve in verticals:
            if ve in vertical_edges_done:
                continue
            vertical_edges_done.add(ve)
            gmsh.model.mesh.setTransfiniteCurve(ve, n_layers + 1)
            
        if bot_edge is not None and top_edge is not None:
            if top_edge not in periodic_edges_done:
                periodic_edges_done.add(top_edge)
                dz = z_top - z_bot
                transform = [
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, float(dz),
                    0.0, 0.0, 0.0, 1.0
                ]
                try:
                    gmsh.model.mesh.setPeriodic(1, [top_edge], [bot_edge], transform)
                except Exception as per_err:
                    logger.warning(
                        "Failed to set periodic constraint for top_edge %s -> "
                        "bot_edge %s: %s",
                        top_edge,
                        bot_edge,
                        per_err,
                    )

    # Step 3: materialise 1D mesh.
    gmsh.model.mesh.generate(1)

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
            edges = gmsh.model.getBoundary([(2, top_tag)], oriented=False, recursive=False)
            boundary_nodes = []
            for _, etag in edges:
                tags, _, _ = gmsh.model.mesh.getNodes(1, etag)
                boundary_nodes.extend(tags)
            boundary_nodes = list(set(boundary_nodes))
            if len(boundary_nodes) >= 3:
                gmsh.model.mesh.addElementsByType(
                    top_tag, 2, [], [int(boundary_nodes[0]), int(boundary_nodes[1]), int(boundary_nodes[2])]
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
        for idx, new_tag in zip(interior_indices, new_tags):
            idx_to_tag[idx] = new_tag
    for b_idx, target_tag in matched.items():
        idx_to_tag[b_idx] = target_tag
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
    snap_tolerance = 1e-6
    # 1) Read bot triangulation.
    elem_types, _elem_tags, node_tags = gmsh.model.mesh.getElements(2, bot_tag)
    if 2 not in elem_types:  # type 2 = 3-node triangle
        return
    tri_idx = list(elem_types).index(2)
    tris = np.array(node_tags[tri_idx]).reshape(-1, 3)
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
    boundary_node_tags = set(int(t) for t in boundary_node_tags)

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
    # Then re-stamp with the bot-matched triangulation.
    gmsh.model.mesh.removeElements(2, top_tag)
    top_tri_nodes: list[int] = []
    for tri in tris:
        top_tri_nodes.extend(bot_to_top[int(t)] for t in tri)
    gmsh.model.mesh.addElementsByType(top_tag, 2, [], top_tri_nodes)

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
            tags, coords, _ = gmsh.model.mesh.getNodes(
                2, lf_tag, includeBoundary=True
            )
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
            all_lateral_pts = np.array(all_lateral_coord).reshape(-1, 3)[
                unique_indices
            ]
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

    # 5) Emit wedges (gmsh element type 6 = 6-node prism).
    wedge_node_tags: list[int] = []
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
    gmsh.model.mesh.addElementsByType(vol_tag, 6, [], wedge_node_tags)
    emitted = len(wedge_node_tags) // 6
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
