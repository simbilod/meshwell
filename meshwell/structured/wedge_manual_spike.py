"""SPIKE: Python-constructed lateral mesh, no transfinite hints.

Investigation into eliminating per-lateral-face transfinite hints in
favour of directly emitting lateral-face quads from Python after the
horizontal-face 2D mesh is built.

The current pipeline (wedge.py) leans on gmsh's transfinite mesher to
place intermediate-layer nodes on lateral faces, then ``stamp_wedges``
mirrors them into wedge elements. This spike asks: if we know the
cohort topology, can we emit the lateral quad strips ourselves and
sidestep transfinite entirely?

Flow:
  pre_2d_hook (manual_pre_2d_validate):
    - Skip ``apply_lateral_transfinite_hints``.
    - Validate n_layers consistency only.
  generate(2):
    - Horizontal faces get free triangulation as before.
    - Lateral faces get free triangulation too (we'll discard it).
  pre_3d_hook (construct_lateral_quads + stamp_wedges):
    - For each lateral face: clear free triangulation, walk its bot
      edge nodes in parametric order, generate intermediate-row +
      top-row nodes by vertical translation, emit quad elements.
    - Then ``stamp_wedges`` runs as before; intermediate-layer nodes
      now exist on lateral faces, so stamp_wedges' node-lookup path
      finds them rather than creating new ones.

This is a spike — kept in a separate module so the production hooks
(wedge.apply_lateral_transfinite_hints) remain the default. To use it,
monkey-patch the orchestrator's hooks.
"""
from __future__ import annotations

from collections import defaultdict

import gmsh

from meshwell.structured.exceptions import (
    StructuredLateralNLayersMismatchError,
)
from meshwell.structured.types import ShapeKey, SlabMeta
from meshwell.structured.wedge import (
    _face_centroid_z,
    resolve_n_layers,
    stamp_wedges,
)


def manual_pre_2d_validate(
    slab_meta: dict[ShapeKey, SlabMeta],
    face_tag_by_key: dict[ShapeKey, int],
    resolution_specs: dict | None = None,
) -> None:
    """Validate n_layers consistency without applying transfinite hints.

    Mirrors the check in apply_lateral_transfinite_hints but skips the
    gmsh transfinite/recombine calls. Lateral faces will get free
    triangulation from generate(2); the pre_3d hook will rebuild them.
    """
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


def _ordered_nodes_on_curve(curve_tag: int) -> list[tuple[int, float, float, float]]:
    """Return [(node_tag, x, y, z)] in parametric order along the curve.

    Uses parametric coords to sort.
    """
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


def _classify_lateral_face_edges(
    face_tag: int,
    z_bot: float,
    z_top: float,
    z_tol: float = 1e-7,
) -> tuple[int | None, int | None, list[int]]:
    """Return (bot_edge, top_edge, vertical_edges) of a lateral face.

    A lateral face has 4 boundary edges: bot (at z=z_bot), top
    (at z=z_top), and 2 verticals.
    """
    edges = gmsh.model.getBoundary([(2, face_tag)], oriented=False, recursive=False)
    bot_edge = None
    top_edge = None
    vertical = []
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


def construct_lateral_quads(
    slab_meta: dict[ShapeKey, SlabMeta],
    face_tag_by_key: dict[ShapeKey, int],
    resolution_specs: dict | None = None,
) -> None:
    """Construct quad mesh on every lateral face of every cohort sub-solid.

    Steps per face:
      1. Clear gmsh's free-triangulation.
      2. Walk bot edge nodes in parametric order.
      3. Generate intermediate-row + top-row nodes at (x, y, z_layer).
         (Vertical extrusion preserves (x, y); even for cylindrical
         laterals, same theta means same (x, y) at different z.)
      4. Emit quads stitching consecutive bot-edge segments across layers.

    After this, intermediate-layer nodes exist on lateral faces and the
    face has a proper 2D quad mesh. ``stamp_wedges`` then proceeds as
    normal — its existing-node lookup will find these and reuse them.

    Per-row-z node cache key is keyed by (entity_dim, entity_tag, x_key, y_key)
    so two lateral faces sharing a vertical seam edge reuse seam nodes.
    """
    # Per-face info: pick owner's n_layers, z_bot, z_top.
    lateral_info: dict[int, tuple[int, float, float]] = {}
    for meta in slab_meta.values():
        if not meta.keep:
            continue
        bot_tag = face_tag_by_key.get(meta.bot_face_key)
        top_tag = face_tag_by_key.get(meta.top_face_key)
        if bot_tag is None or top_tag is None:
            continue
        n_layers = resolve_n_layers(meta.physical_name, resolution_specs)
        z_bot = _face_centroid_z(bot_tag)
        z_top = _face_centroid_z(top_tag)
        for fk in meta.lateral_face_keys:
            face_tag = face_tag_by_key.get(fk)
            if face_tag is None:
                continue
            lateral_info[face_tag] = (n_layers, z_bot, z_top)

    # Vertical-edge node caches: shared between adjacent lateral faces
    # so seam nodes are unified.
    # Keyed by (vertical_edge_tag, layer) -> node_tag
    vert_edge_layer_node: dict[tuple[int, int], int] = {}

    for face_tag, (n_layers, z_bot, z_top) in lateral_info.items():
        # 1. Clear free triangulation
        gmsh.model.mesh.clear([(2, face_tag)])

        # 2. Identify edges of this face
        bot_edge, top_edge, vertical_edges = _classify_lateral_face_edges(
            face_tag, z_bot, z_top
        )
        if bot_edge is None or top_edge is None or len(vertical_edges) != 2:
            # Unsupported topology; skip silently for the spike
            continue

        # 3. Get ordered bot edge nodes
        bot_row = _ordered_nodes_on_curve(bot_edge)
        if len(bot_row) < 2:
            continue
        # Top edge ordered nodes (from generate(1) — top edge curves
        # were 1D-meshed too).
        top_row = _ordered_nodes_on_curve(top_edge)
        if len(top_row) != len(bot_row):
            # Mismatch: bot and top edges should have same number of
            # nodes if generate(1) used the same characteristic length.
            # For the spike, fall back to skipping this face.
            continue

        # Align top_row to bot_row by x,y proximity (parametric direction
        # may run opposite to bot)
        bot_xy = [(x, y) for _t, x, y, _z in bot_row]
        top_xy = [(x, y) for _t, x, y, _z in top_row]
        if len(top_xy) >= 2 and (
            (top_xy[0][0] - bot_xy[0][0]) ** 2 + (top_xy[0][1] - bot_xy[0][1]) ** 2
        ) > ((top_xy[-1][0] - bot_xy[0][0]) ** 2 + (top_xy[-1][1] - bot_xy[0][1]) ** 2):
            top_row = list(reversed(top_row))

        # Identify which vertical edge sits at each end of the bot row.
        # bot_row[0] is the "left" end, bot_row[-1] is the "right" end.
        left_pos = (bot_row[0][1], bot_row[0][2])
        right_pos = (bot_row[-1][1], bot_row[-1][2])
        left_vert = None
        right_vert = None
        for ve in vertical_edges:
            ev = gmsh.model.getBoundary([(1, ve)], oriented=False, recursive=False)
            x_, y_ = None, None
            for _vd, vt in ev:
                pos = gmsh.model.getValue(0, vt, [])
                x_, y_ = pos[0], pos[1]
                break  # both vertices have same (x,y)
            if x_ is None:
                continue
            d_left = (x_ - left_pos[0]) ** 2 + (y_ - left_pos[1]) ** 2
            d_right = (x_ - right_pos[0]) ** 2 + (y_ - right_pos[1]) ** 2
            if d_left < d_right:
                left_vert = ve
            else:
                right_vert = ve
        if left_vert is None or right_vert is None:
            continue

        # 4. Per intermediate layer, build a row of nodes at (x, y, z_layer).
        #    For interior nodes (not at left/right end): new node on face.
        #    For end nodes (left/right vertical edge): cached on the vertical edge.
        rows: list[list[int]] = [[t for t, _x, _y, _z in bot_row]]

        # Per-face cache for interior nodes per layer
        for layer in range(1, n_layers):
            z_layer = z_bot + (z_top - z_bot) * layer / n_layers
            row_tags: list[int] = []
            for idx, (_t, x, y, _z) in enumerate(bot_row):
                if idx == 0:
                    # Left vertical edge node at this layer
                    key = (left_vert, layer)
                    if key in vert_edge_layer_node:
                        row_tags.append(vert_edge_layer_node[key])
                    else:
                        new_tag = gmsh.model.mesh.getMaxNodeTag() + 1
                        gmsh.model.mesh.addNodes(
                            1, left_vert, [new_tag], [x, y, z_layer]
                        )
                        vert_edge_layer_node[key] = new_tag
                        row_tags.append(new_tag)
                elif idx == len(bot_row) - 1:
                    key = (right_vert, layer)
                    if key in vert_edge_layer_node:
                        row_tags.append(vert_edge_layer_node[key])
                    else:
                        new_tag = gmsh.model.mesh.getMaxNodeTag() + 1
                        gmsh.model.mesh.addNodes(
                            1, right_vert, [new_tag], [x, y, z_layer]
                        )
                        vert_edge_layer_node[key] = new_tag
                        row_tags.append(new_tag)
                else:
                    new_tag = gmsh.model.mesh.getMaxNodeTag() + 1
                    gmsh.model.mesh.addNodes(2, face_tag, [new_tag], [x, y, z_layer])
                    row_tags.append(new_tag)
            rows.append(row_tags)

        # Top row from top_row
        rows.append([t for t, _x, _y, _z in top_row])

        # 5. Emit quad elements connecting consecutive rows.
        # Element type 3 = 4-node quad.
        quad_nodes: list[int] = []
        for r in range(len(rows) - 1):
            for c in range(len(rows[r]) - 1):
                # CCW: bot_left, bot_right, top_right, top_left
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


def manual_pre_3d_hook(
    slab_meta: dict[ShapeKey, SlabMeta],
    face_tag_by_key: dict[ShapeKey, int],
    sub_solid_tag_by_key: dict[ShapeKey, int],
    resolution_specs: dict | None = None,
    point_tolerance: float = 1e-3,
) -> None:
    """Combined pre_3d_hook: construct lateral quads, then stamp wedges.

    Replaces ``stamp_wedges`` alone as the pre_3d action when running the
    spike (where pre_2d_hook didn't apply transfinite hints).
    """
    construct_lateral_quads(
        slab_meta=slab_meta,
        face_tag_by_key=face_tag_by_key,
        resolution_specs=resolution_specs,
    )
    stamp_wedges(
        slab_meta=slab_meta,
        face_tag_by_key=face_tag_by_key,
        sub_solid_tag_by_key=sub_solid_tag_by_key,
        resolution_specs=resolution_specs,
        point_tolerance=point_tolerance,
    )
