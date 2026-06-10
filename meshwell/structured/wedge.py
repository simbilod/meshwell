"""Stage 5 — gmsh meshing hooks for structured cohorts.

pre_2d_hook (freeze_lateral_mesh): emits cohort lateral-face quad
mesh from Python before gmsh's generate(2) runs. Uses
Mesh.MeshOnlyEmpty=1 so the outer 2D mesher leaves cohort laterals
alone. Raises on n_layers mismatch or unsupported lateral topology.

pre_3d_hook (stamp_wedges): per cohort sub-solid, copies bot
triangulation to top and emits wedge elements.
"""
from __future__ import annotations

from collections import defaultdict

import gmsh
import numpy as np

from meshwell.structured.exceptions import (
    StructuredError,
    StructuredLateralNLayersMismatchError,
    StructuredTransfiniteRejectedError,
    WedgeBotNodeMismatchError,
    WedgeCountMismatchError,
)
from meshwell.structured.types import ShapeKey, SlabMeta


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
# Alt B: freeze cohort lateral mesh before generate(2)
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


def freeze_lateral_mesh(
    slab_meta: dict[ShapeKey, SlabMeta],
    face_tag_by_key: dict[ShapeKey, int],
    resolution_specs: dict | None = None,
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

    The result is identical in physical-group output to the old
    freeze_lateral_mesh path but never invokes gmsh's
    2D mesher or its periodic-surface mesher on cohort lateral
    faces — both sources of past failures.
    """
    # Step 1: per-face n_layers + consistency check.
    owners_per_face: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for meta in slab_meta.values():
        if not meta.keep:
            continue
        n_layers = resolve_n_layers(meta.physical_name, resolution_specs)
        for fk in meta.lateral_face_keys:
            tag = face_tag_by_key.get(fk)
            if tag is None:
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

    # Step 2: setTransfiniteCurve on vertical edges.
    vertical_edges_done: set[int] = set()
    for face_tag, (z_bot, z_top) in face_z_bounds.items():
        n_layers = face_n_layers.get(face_tag, 1)
        _, _, verticals = _classify_lateral_face_edges(face_tag, z_bot, z_top)
        for ve in verticals:
            if ve in vertical_edges_done:
                continue
            vertical_edges_done.add(ve)
            gmsh.model.mesh.setTransfiniteCurve(ve, n_layers + 1)

    # Step 3: materialise 1D mesh.
    gmsh.model.mesh.generate(1)

    # Step 4: emit lateral-face quads.
    for face_tag, (z_bot, z_top) in face_z_bounds.items():
        n_layers = face_n_layers[face_tag]
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
            continue

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
            continue

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

    for _, sub_key, meta in order:
        bot_tag = face_tag_by_key[meta.bot_face_key]
        top_tag = face_tag_by_key[meta.top_face_key]
        vol_tag = sub_solid_tag_by_key[sub_key]
        n_layers = resolve_n_layers(meta.physical_name, resolution_specs)
        _stamp_one(bot_tag, top_tag, vol_tag, meta, n_layers, point_tolerance)


def _face_centroid_z(face_tag: int) -> float:
    """Return the z-coordinate of a face's centroid.

    XY-extruded faces have a flat z.
    """
    bbox = gmsh.model.getBoundingBox(2, face_tag)
    return (bbox[2] + bbox[5]) / 2


def _stamp_one(
    bot_tag: int,
    top_tag: int,
    vol_tag: int,
    meta: SlabMeta,
    n_layers: int,
    point_tolerance: float,
) -> None:
    """Read bot triangulation, stamp on top, emit wedges into volume."""
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
    mismatched = 0
    for bnt, bpt in zip(bot_node_tags, bot_pts):
        if len(existing_top_pts):
            d = np.linalg.norm(existing_top_pts[:, :2] - bpt[:2], axis=1)
            if d.min() < point_tolerance:
                bot_to_top[int(bnt)] = int(existing_top_nodes[int(np.argmin(d))])
                continue
        new_tag = gmsh.model.mesh.getMaxNodeTag() + 1
        gmsh.model.mesh.addNodes(
            2,
            top_tag,
            [new_tag],
            [float(bpt[0]), float(bpt[1]), float(top_z)],
        )
        bot_to_top[int(bnt)] = int(new_tag)

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
        for layer in range(1, n_layers):
            z_layer = bot_z + dz * layer
            # Snapshot all current model nodes to look up existing z_layer nodes.
            all_model_tags, all_model_coord, _ = gmsh.model.mesh.getNodes()
            all_model_pts = np.array(all_model_coord).reshape(-1, 3)
            # Filter to nodes near z_layer for efficiency.
            z_mask = np.abs(all_model_pts[:, 2] - z_layer) < point_tolerance
            zlayer_tags = all_model_tags[z_mask]
            zlayer_pts = all_model_pts[z_mask]

            this_map: dict[int, int] = {}
            for i, bpt in enumerate(bot_pts):
                if len(zlayer_tags):
                    d = np.linalg.norm(zlayer_pts[:, :2] - bpt[:2], axis=1)
                    if d.min() < point_tolerance:
                        # Reuse the existing node at this z_layer position.
                        this_map[i] = int(zlayer_tags[int(np.argmin(d))])
                        continue
                # No existing node close enough — create a fresh one.
                new_tag = gmsh.model.mesh.getMaxNodeTag() + 1
                gmsh.model.mesh.addNodes(
                    3,
                    vol_tag,
                    [new_tag],
                    [float(bpt[0]), float(bpt[1]), float(z_layer)],
                )
                this_map[i] = new_tag
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
            wedge_node_tags.extend(
                [
                    bot_map[b0],
                    bot_map[b1],
                    bot_map[b2],
                    top_map[b0],
                    top_map[b1],
                    top_map[b2],
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
