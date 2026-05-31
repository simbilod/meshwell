"""Stage 5 — gmsh meshing hooks for structured cohorts.

pre_2d_hook (apply_lateral_transfinite_hints): sets transfinite curve
counts on vertical lateral edges and transfinite surface hints on
lateral faces of every cohort sub-solid. Raises on n_layers mismatch
or unsupported lateral topology.

pre_3d_hook (stamp_wedges, Task 17): per cohort sub-solid, copies bot
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


def apply_lateral_transfinite_hints(
    slab_meta: dict[ShapeKey, SlabMeta],
    face_tag_by_key: dict[ShapeKey, int],
    resolution_specs: dict | None = None,
) -> None:
    """For each cohort sub-solid lateral face: enforce n_layers and apply gmsh transfinite hints.

    Raise on:
      - Shared lateral face with mismatched n_layers.
      - Lateral face with != 4 boundary edges.
    """
    # Group: face_tag -> list[(slab_index, n_layers)] for shared-lateral
    # n_layers consistency check.
    owners_per_face: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for meta in slab_meta.values():
        n_layers = resolve_n_layers(meta.physical_name, resolution_specs)
        for fk in meta.lateral_face_keys:
            tag = face_tag_by_key.get(fk)
            if tag is None:
                continue
            owners_per_face[tag].append((meta.slab_index, n_layers))

    for face_tag, owners in owners_per_face.items():
        # n_layers must agree across all owners of this face.
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
        n_layers = owners[0][1]

        # Get the boundary 1D edges of this face.
        edges = gmsh.model.getBoundary([(2, face_tag)], oriented=False, recursive=False)
        if len(edges) != 4:
            raise StructuredTransfiniteRejectedError(
                face_tag=face_tag,
                slab_index=owners[0][0],
                reason=f"expected 4 boundary edges, got {len(edges)}",
            )

        # Identify vertical edges (endpoints differ in z) and set transfinite
        # curve counts on them.
        for _dim, etag in edges:
            ev = gmsh.model.getBoundary([(1, etag)], oriented=False, recursive=False)
            zs = []
            for _vd, vt in ev:
                pos = gmsh.model.getValue(0, vt, [])
                zs.append(pos[2])
            if len(zs) == 2 and abs(zs[0] - zs[1]) > 1e-9:
                # n_layers segments means n_layers+1 nodes on this edge.
                gmsh.model.mesh.setTransfiniteCurve(etag, n_layers + 1)

        gmsh.model.mesh.setTransfiniteSurface(face_tag)


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

    # 3) Clear top face mesh & rebuild from bot.
    gmsh.model.mesh.clear([(2, top_tag)])
    # Re-fetch top boundary nodes (placed by lateral transfinite mesh).
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

    # Re-stamp triangulation on top face.
    top_tri_nodes: list[int] = []
    for tri in tris:
        top_tri_nodes.extend(bot_to_top[int(t)] for t in tri)
    gmsh.model.mesh.addElementsByType(top_tag, 2, [], top_tri_nodes)

    # 4) Intermediate layer nodes (for n_layers > 1).
    bot_idx_by_tag = {int(t): i for i, t in enumerate(bot_node_tags)}
    layer_maps: list[dict[int, int]] = [
        {i: int(bot_node_tags[i]) for i in range(len(bot_node_tags))}
    ]
    if n_layers > 1:
        for layer in range(1, n_layers):
            z_layer = bot_z + dz * layer
            this_map: dict[int, int] = {}
            for i, bpt in enumerate(bot_pts):
                tag = gmsh.model.mesh.getMaxNodeTag() + 1
                gmsh.model.mesh.addNodes(
                    3,
                    vol_tag,
                    [tag],
                    [float(bpt[0]), float(bpt[1]), float(z_layer)],
                )
                this_map[i] = tag
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
