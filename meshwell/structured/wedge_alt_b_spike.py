"""SPIKE Alt B: freeze the cohort mesh before gmsh meshes anything in 2D.

The current manual spike (wedge_manual_spike.py) lets gmsh free-mesh
lateral faces during generate(2), then clears them and rebuilds. Alt B
avoids that waste by emitting the lateral mesh *before* generate(2)
runs and using ``Mesh.MeshOnlyEmpty=1`` to make gmsh leave those faces
alone.

Flow:
  pre_2d_hook (alt_b_pre_2d):
    - Set transfinite on vertical edges (n_layers+1 nodes) so they
      have predictable node counts.
    - Call gmsh.model.mesh.generate(1) explicitly to materialise edge
      nodes (this is normally done implicitly by generate(2), but we
      need the nodes before we emit the lateral mesh).
    - For each lateral face: walk bot edge nodes in parametric order,
      generate intermediate-row + top-row nodes at corresponding
      (x, y, z_layer) positions, emit quad elements.
    - Set Mesh.MeshOnlyEmpty=1 so generate(2) skips faces that already
      have a mesh.
  generate(2):
    - Horizontal faces: triangulated by gmsh.
    - Lateral faces: skipped (already have quads).
    - Neighbour faces: free-meshed conforming to the cohort boundary.
  pre_3d_hook (stamp_wedges):
    - Bot triangulation is available; stamp wedges as in production.

Compared to the manual spike, Alt B:
  - Never invokes gmsh's 2D mesher on cohort lateral faces (no free-
    mesh then clear).
  - Never invokes gmsh's periodic-surface mesher on cohort cylindrical
    laterals (we mesh them ourselves before generate(2)).
  - Keeps a single global ``Mesh.MeshOnlyEmpty=1`` toggle that
    survives into pre_3d_hook (already needed there for volumes).
"""
from __future__ import annotations

from collections import defaultdict

import gmsh

from meshwell.structured.exceptions import (
    StructuredLateralNLayersMismatchError,
    StructuredTransfiniteRejectedError,
)
from meshwell.structured.types import ShapeKey, SlabMeta
from meshwell.structured.wedge import (
    _face_centroid_z,
    resolve_n_layers,
)


def _classify_lateral_face_edges(
    face_tag: int,
    z_bot: float,
    z_top: float,
    z_tol: float = 1e-7,
) -> tuple[int | None, int | None, list[int]]:
    """Return (bot_edge_tag, top_edge_tag, [vertical_edge_tags]) of a lateral face."""
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


def _ordered_curve_nodes(curve_tag: int) -> list[tuple[int, float, float, float]]:
    """Return curve nodes [(tag, x, y, z)] sorted by parametric coordinate."""
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


def alt_b_pre_2d(
    slab_meta: dict[ShapeKey, SlabMeta],
    face_tag_by_key: dict[ShapeKey, int],
    resolution_specs: dict | None = None,
) -> None:
    """Pre_2d hook: freeze cohort lateral mesh before generate(2)."""
    # --- Step 1: validate n_layers consistency on shared lateral faces ---
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

    # --- Step 2: set transfinite on vertical edges so they get the right
    # node count when generate(1) runs ---
    # Per-face z bounds, needed both here and in step 4.
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

    # Collect unique vertical edges, set transfinite once per edge.
    vertical_edges_done: set[int] = set()
    for face_tag, (z_bot, z_top) in face_z_bounds.items():
        n_layers = face_n_layers.get(face_tag, 1)
        _, _, verticals = _classify_lateral_face_edges(face_tag, z_bot, z_top)
        for ve in verticals:
            if ve in vertical_edges_done:
                continue
            vertical_edges_done.add(ve)
            gmsh.model.mesh.setTransfiniteCurve(ve, n_layers + 1)

    # --- Step 3: explicitly generate 1D so we have edge nodes ---
    gmsh.model.mesh.generate(1)

    # --- Step 4: emit lateral face quads ---
    # Per-(vertical_edge, layer) cache so adjacent lateral faces share
    # seam-edge intermediate nodes.
    vert_edge_layer_node: dict[tuple[int, int], int] = {}

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
            # Mismatch in bot/top discretization — skip (would normally
            # be caught by validators upstream).
            continue

        # Pick left/right vertical by (x, y) proximity to bot row endpoints.
        left_xy = (bot_row[0][1], bot_row[0][2])
        right_xy = (bot_row[-1][1], bot_row[-1][2])
        left_vert = right_vert = None
        for ve in verticals:
            ev = gmsh.model.getBoundary([(1, ve)], oriented=False, recursive=False)
            x_v, y_v = None, None
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

        # Build n_layers+1 rows of node tags.
        rows: list[list[int]] = [[t for t, _x, _y, _z in bot_row]]
        for layer in range(1, n_layers):
            z_layer = z_bot + (z_top - z_bot) * layer / n_layers
            row_tags: list[int] = []
            for idx, (_t, x, y, _z) in enumerate(bot_row):
                if idx == 0:
                    key = (left_vert, layer)
                    if key in vert_edge_layer_node:
                        row_tags.append(vert_edge_layer_node[key])
                    else:
                        # Vertical edges have transfinite-placed nodes at
                        # this layer; find by parametric position.
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
        rows.append([t for t, _x, _y, _z in top_row])

        # Emit quad elements (gmsh type 3 = 4-node quad).
        quad_nodes: list[int] = []
        for r in range(len(rows) - 1):
            for c in range(len(rows[r]) - 1):
                quad_nodes.extend(
                    [rows[r][c], rows[r][c + 1], rows[r + 1][c + 1], rows[r + 1][c]]
                )
        if quad_nodes:
            gmsh.model.mesh.addElementsByType(face_tag, 3, [], quad_nodes)

    # --- Step 5: tell gmsh's outer generate(2) to skip already-meshed
    # faces (i.e. our cohort lateral faces). ---
    gmsh.option.setNumber("Mesh.MeshOnlyEmpty", 1)
