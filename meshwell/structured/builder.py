"""Phase-3: mesh-stage builder (Layer C).

Public entry points (added incrementally):

- :func:`resolve_mesh_plan` — second-pass over the spec list to attach
  ``n_layers`` and ``recombine`` to each slab; also cross-checks
  Phase-1 OverlapPairs.
- :func:`apply_structured_mesh` — full mesh-stage execution: stamp
  derived top meshes, build discrete 3D entities per slab, run global
  removeDuplicateNodes.
"""
from __future__ import annotations

import contextlib
from typing import Any

import numpy as np

from meshwell.structured.spec import (
    StructuredExtrusionResolutionSpec,
    StructuredMeshOverlapError,
    StructuredMeshPlan,
    StructuredPlan,
)


def _spec_of(entity: Any) -> StructuredExtrusionResolutionSpec | None:
    for r in getattr(entity, "resolutions", None) or []:
        if isinstance(r, StructuredExtrusionResolutionSpec):
            return r
    return None


def resolve_mesh_plan(plan: StructuredPlan, entities: list[Any]) -> StructuredMeshPlan:
    """Look up (n_layers, recombine) for each slab via its owning spec.

    Cross-checks every ``OverlapPair`` in the plan: if the loser slab's
    spec n_layers != the winner's, raises
    ``StructuredMeshOverlapError``. This is a paranoid double-check;
    Phase-1's Policy B already catches direct mismatches at plan time.
    """
    n_layers_list: list[int] = []
    recombine_list: list[bool] = []
    for slab in plan.slabs:
        owner = entities[slab.source_index]
        spec = _spec_of(owner)
        if spec is None:
            raise StructuredMeshOverlapError(
                f"Slab {slab.physical_name} source entity has no "
                f"StructuredExtrusionResolutionSpec attached."
            )
        n_layers_list.append(int(spec.n_layers[slab.z_interval_index]))
        recombine_list.append(bool(spec.recombine))

    for op in plan.overlaps:
        winner = plan.slabs[op.winner_slab_index]
        winner_spec = _spec_of(entities[winner.source_index])
        loser_owner = entities[op.loser_source_index]
        loser_spec = _spec_of(loser_owner)
        if winner_spec is None or loser_spec is None:
            continue
        winner_n = winner_spec.n_layers[winner.z_interval_index]
        loser_n = loser_spec.n_layers[op.loser_z_interval_index]
        if winner_n != loser_n:
            raise StructuredMeshOverlapError(
                f"OverlapPair winner {winner.physical_name} "
                f"(n_layers={winner_n}) and loser source_index="
                f"{op.loser_source_index} (n_layers={loser_n}) "
                f"at z={op.z_extent}: n_layers must match for the "
                f"overlap to be valid."
            )

    return StructuredMeshPlan(
        slabs=plan.slabs,
        n_layers=tuple(n_layers_list),
        recombine=tuple(recombine_list),
    )


def _stamp_top_face_mesh(
    bottom_face_tag: int,
    top_face_tag: int,
    zlo: float,
    zhi: float,
) -> dict[int, int]:
    """Replace the top OCC face's 2D mesh with a translated copy of the bottom's.

    Phase 3 minimum: pure translation. Boundary node positions for the
    top are computed as ``bottom_xy + (0, 0, zhi - zlo)``. The full Layer
    B vertex-map lookup (for BOP-displaced boundary nodes) lands in
    Phase 4.

    Returns a dict mapping bottom node tag -> newly-allocated top node
    tag (so the volume builder can use it for prism construction).
    """
    import gmsh

    height = zhi - zlo

    # Collect bottom mesh data before clearing anything.
    bot_all_tags_arr, bot_all_coords_flat, _ = gmsh.model.mesh.getNodes(
        2, bottom_face_tag, includeBoundary=True
    )
    bot_all_tags = np.asarray(bot_all_tags_arr, dtype=np.int64)
    bot_all_coords = np.asarray(bot_all_coords_flat, dtype=float).reshape(-1, 3)

    bot_int_tags_arr, _, _ = gmsh.model.mesh.getNodes(
        2, bottom_face_tag, includeBoundary=False
    )
    bot_int_tags = set(bot_int_tags_arr.tolist())

    elem_types, _, elem_nodes_per_type = gmsh.model.mesh.getElements(2, bottom_face_tag)
    bot_tri_nodes: list[np.ndarray] = []
    for et, en in zip(elem_types, elem_nodes_per_type):
        if et == 2:
            bot_tri_nodes.append(np.asarray(en, dtype=np.int64).reshape(-1, 3))
    if not bot_tri_nodes:
        return {}
    bot_triangles = np.concatenate(bot_tri_nodes, axis=0)

    # Clear the top face's 2D mesh. Boundary nodes (on curves/vertices) survive.
    with contextlib.suppress(Exception):
        gmsh.model.mesh.clear([(2, top_face_tag)])

    # After clearing, the top face retains its boundary (curve) nodes.
    # Match them to bottom boundary nodes by XY position.
    top_bnd_tags_arr, top_bnd_coords_flat, _ = gmsh.model.mesh.getNodes(
        2, top_face_tag, includeBoundary=True
    )
    top_bnd_tags = np.asarray(top_bnd_tags_arr, dtype=np.int64)
    top_bnd_coords = np.asarray(top_bnd_coords_flat, dtype=float).reshape(-1, 3)

    bot_to_top_tag: dict[int, int] = {}

    # Build XY -> top-boundary-tag lookup for fast matching.
    top_bnd_xy_to_tag: dict[tuple[float, float], int] = {}
    for i, tt in enumerate(top_bnd_tags):
        key = (round(top_bnd_coords[i, 0], 9), round(top_bnd_coords[i, 1], 9))
        top_bnd_xy_to_tag[key] = int(tt)

    # Map bottom boundary nodes -> existing top boundary nodes.
    new_interior_tags: list[int] = []
    new_interior_coords_flat: list[float] = []
    next_tag = int(gmsh.model.mesh.getMaxNodeTag()) + 1
    new_node_counter = 0

    for i, bt in enumerate(bot_all_tags):
        bt_int = int(bt)
        if bt_int not in bot_int_tags:
            # Boundary node: match to existing top boundary node by XY.
            key = (round(bot_all_coords[i, 0], 9), round(bot_all_coords[i, 1], 9))
            top_tt = top_bnd_xy_to_tag.get(key)
            if top_tt is not None:
                bot_to_top_tag[bt_int] = top_tt
            # If not found (shouldn't happen in Phase 3), skip mapping.
        else:
            # Interior node: allocate a fresh tag on the top face.
            new_tag = next_tag + new_node_counter
            new_node_counter += 1
            bot_to_top_tag[bt_int] = new_tag
            new_interior_tags.append(new_tag)
            new_interior_coords_flat.extend(
                [bot_all_coords[i, 0], bot_all_coords[i, 1], zlo + height]
            )

    if new_interior_tags:
        gmsh.model.mesh.addNodes(
            2, top_face_tag, new_interior_tags, new_interior_coords_flat
        )

    top_tri_nodes_flat: list[int] = []
    for tri in bot_triangles:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        top_tri_nodes_flat.extend(
            [bot_to_top_tag[a], bot_to_top_tag[b], bot_to_top_tag[c]]
        )

    next_elem_tag = int(gmsh.model.mesh.getMaxElementTag()) + 1
    elem_tags_list = list(range(next_elem_tag, next_elem_tag + bot_triangles.shape[0]))
    gmsh.model.mesh.addElements(
        2, top_face_tag, [2], [elem_tags_list], [top_tri_nodes_flat]
    )

    return bot_to_top_tag


def _build_slab_volume(
    bottom_face_tag: int,
    bot_to_top_layer_tags: list[dict[int, int]],
    n_layers: int,
    recombine: bool,
    pre_allocated_vol_tag: int | None = None,
) -> int:
    """Create one discrete 3D entity with wedge or hex elements.

    Args:
        bottom_face_tag: OCC face tag whose 2D mesh is the source
            triangulation (or quad mesh, if recombine).
        bot_to_top_layer_tags: list of length n_layers, each a dict
            mapping bottom node tag -> the node tag in that layer.
            Layer 0 is implicit (= bottom node tags themselves);
            bot_to_top_layer_tags[i] maps to layer i+1.
        n_layers: number of element layers in z. Must equal
            len(bot_to_top_layer_tags).
        recombine: if True, build hex elements (type 5) instead of
            wedges (type 6). Bottom must have quads in that case.
        pre_allocated_vol_tag: if not None, reuse this pre-allocated
            discrete entity tag instead of creating a new one.  The
            caller must have already added any interior nodes to it
            before calling this function so that addElements finds all
            referenced node tags in the model.

    Returns:
        The discrete 3D entity's tag.
    """
    import gmsh

    if len(bot_to_top_layer_tags) != n_layers:
        raise ValueError(
            f"Expected {n_layers} layer maps, got {len(bot_to_top_layer_tags)}"
        )

    elem_types, _, elem_nodes_per_type = gmsh.model.mesh.getElements(2, bottom_face_tag)
    target_type = 3 if recombine else 2
    elem_3d_type = 5 if recombine else 6
    cells_per_face = 4 if recombine else 3
    bot_cells: list[np.ndarray] = []
    for et, en in zip(elem_types, elem_nodes_per_type):
        if et == target_type:
            bot_cells.append(np.asarray(en, dtype=np.int64).reshape(-1, cells_per_face))
    if not bot_cells:
        raise RuntimeError(
            f"Bottom OCC face {bottom_face_tag} has no element type "
            f"{target_type} (need {'quads' if recombine else 'triangles'})"
        )
    bot_cells_flat = np.concatenate(bot_cells, axis=0)
    n_cells = bot_cells_flat.shape[0]

    if pre_allocated_vol_tag is not None:
        vol_tag = pre_allocated_vol_tag
    else:
        vol_tag = gmsh.model.addDiscreteEntity(3, -1, [])

    layer_maps_with_zero: list[dict[int, int] | None] = [
        None,
        *list(bot_to_top_layer_tags),
    ]

    def _layer_tag(layer_idx: int, bot_node_tag: int) -> int:
        if layer_idx == 0:
            return bot_node_tag
        return layer_maps_with_zero[layer_idx][bot_node_tag]

    all_volume_nodes: list[int] = []
    for cell in bot_cells_flat:
        for layer_i in range(n_layers):
            all_volume_nodes.extend(_layer_tag(layer_i, int(c)) for c in cell)
            all_volume_nodes.extend(_layer_tag(layer_i + 1, int(c)) for c in cell)

    next_elem_tag = int(gmsh.model.mesh.getMaxElementTag()) + 1
    n_3d = n_cells * n_layers
    elem_tags = list(range(next_elem_tag, next_elem_tag + n_3d))
    gmsh.model.mesh.addElements(
        3, vol_tag, [elem_3d_type], [elem_tags], [all_volume_nodes]
    )
    return vol_tag


def _find_horizontal_face_at_z(z: float, tol: float = 1e-6) -> int | None:
    """Return the gmsh face tag whose z-bbox is at z (within tol), else None.

    Phase 3 minimum assumption: single piece, no neighbour cuts -> exactly
    one bottom face at z=zlo and one top face at z=zhi for the slab.
    """
    import gmsh

    for dim, tag in gmsh.model.getEntities(2):
        bb = gmsh.model.getBoundingBox(dim, tag)
        if abs(bb[2] - z) < tol and abs(bb[5] - z) < tol:
            return tag
    return None


def _find_volume_containing_faces(
    bottom_face_tag: int, top_face_tag: int
) -> int | None:
    """Return the gmsh 3D entity tag whose boundary includes both faces, or None.

    Used to identify the OCC volume for a slab so we can add structured
    elements directly to it (rather than creating a separate discrete entity).
    """
    import gmsh

    for dim, vol_tag in gmsh.model.getEntities(3):
        boundaries = gmsh.model.getBoundary([(dim, vol_tag)], oriented=False)
        bnd_tags = {t for _, t in boundaries}
        if bottom_face_tag in bnd_tags and top_face_tag in bnd_tags:
            return vol_tag
    return None


def apply_structured_mesh(
    plan: StructuredPlan,
    mesh_plan: StructuredMeshPlan,
    phantom_map: Any,  # PhantomMap — Any to avoid circular type imports; reserved for Phase 4  # noqa: ARG001
    fuzzy_tol: float = 1e-6,
) -> list[int]:
    """Run the mesh-stage Layer C: derive top meshes + build discrete 3D volumes.

    Returns a list of (slab-index-parallel) discrete-3D entity tags so the
    caller can assert/inspect.

    Phase 3 minimum: assumes single-piece partition per slab and no
    neighbour cuts of horizontal faces. Multi-piece + neighbour-cut
    handling lands in Phase 4.
    """
    import gmsh

    vol_tags: list[int] = []
    for slab_idx, slab in enumerate(plan.slabs):
        n_layers = mesh_plan.n_layers[slab_idx]
        recombine = mesh_plan.recombine[slab_idx]

        bot_tag = _find_horizontal_face_at_z(slab.zlo, tol=fuzzy_tol)
        top_tag = _find_horizontal_face_at_z(slab.zhi, tol=fuzzy_tol)
        if bot_tag is None or top_tag is None:
            raise RuntimeError(
                f"Slab {slab.physical_name}: could not find bottom face "
                f"at z={slab.zlo} or top face at z={slab.zhi} in gmsh model"
            )

        height = slab.zhi - slab.zlo

        if n_layers == 1:
            # Single layer: bottom -> top is the only layer map.
            layer_maps = [_stamp_top_face_mesh(bot_tag, top_tag, slab.zlo, slab.zhi)]
            interior_layer_maps: list[dict[int, int]] = []
            interior_layer_coords: list[list[float]] = []
        else:
            # Multi-layer: build (n_layers - 1) intermediate node maps,
            # then stamp the top mesh as the n-th layer.
            bot_node_tags_arr, bot_coords_flat, _ = gmsh.model.mesh.getNodes(
                2, bot_tag, includeBoundary=True
            )
            bot_node_tags = np.asarray(bot_node_tags_arr, dtype=np.int64)
            bot_coords = np.asarray(bot_coords_flat, dtype=float).reshape(-1, 3)
            next_tag = int(gmsh.model.mesh.getMaxNodeTag()) + 1
            interior_layer_maps = []
            interior_layer_coords = []
            for i_layer in range(1, n_layers):
                m: dict[int, int] = {}
                coords: list[float] = []
                z_i = slab.zlo + height * (i_layer / n_layers)
                for j, bt in enumerate(bot_node_tags):
                    new_tag = next_tag
                    next_tag += 1
                    m[int(bt)] = new_tag
                    coords.extend([bot_coords[j, 0], bot_coords[j, 1], z_i])
                interior_layer_maps.append(m)
                interior_layer_coords.append(coords)
            top_map = _stamp_top_face_mesh(bot_tag, top_tag, slab.zlo, slab.zhi)
            layer_maps = [*interior_layer_maps, top_map]

        # Prefer adding elements directly to the existing OCC volume entity
        # (identified by containment of both horizontal faces in its boundary).
        # This means the OCC entity already has a mesh after apply_structured_mesh
        # and Mesh.MeshOnlyEmpty=1 will skip it during generate(3).
        # If no OCC volume is found (shouldn't happen in Phase 3), fall back to
        # creating a new discrete entity.
        occ_vol_tag = _find_volume_containing_faces(bot_tag, top_tag)

        # Register interior layer nodes BEFORE addElements so all referenced
        # node tags exist in gmsh at the time addElements validates them.
        # We need a valid 3D entity tag for addNodes: use occ_vol_tag if available,
        # else create a new discrete entity.
        pre_vol_tag: int | None = occ_vol_tag
        if n_layers > 1:
            if pre_vol_tag is None:
                pre_vol_tag = gmsh.model.addDiscreteEntity(3, -1, [])
            all_interior_tags: list[int] = []
            all_interior_coords: list[float] = []
            for m, coords in zip(interior_layer_maps, interior_layer_coords):
                all_interior_tags.extend(m.values())
                all_interior_coords.extend(coords)
            if all_interior_tags:
                gmsh.model.mesh.addNodes(
                    3, pre_vol_tag, all_interior_tags, all_interior_coords
                )

        vol_tag = _build_slab_volume(
            bottom_face_tag=bot_tag,
            bot_to_top_layer_tags=layer_maps,
            n_layers=n_layers,
            recombine=recombine,
            pre_allocated_vol_tag=pre_vol_tag,
        )

        vol_tags.append(vol_tag)

    # Global cleanup: merge ~coincident nodes.
    fuzzy_values = [
        slab.fragment_fuzzy_value
        for slab in plan.slabs
        if slab.fragment_fuzzy_value is not None
    ]
    fuzzy = max(fuzzy_values) if fuzzy_values else fuzzy_tol
    try:
        gmsh.model.mesh.removeDuplicateNodes(tag=[], tol=2 * fuzzy)
    except TypeError:
        gmsh.model.mesh.removeDuplicateNodes()

    return vol_tags


def _occ_face_bbox(face: Any) -> tuple[float, float, float, float, float, float]:
    """Return AABB (xmin, ymin, zmin, xmax, ymax, zmax) of an OCP TopoDS_Face."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    b = Bnd_Box()
    BRepBndLib.Add_s(face, b)
    xmin, ymin, zmin, xmax, ymax, zmax = b.Get()
    return (xmin, ymin, zmin, xmax, ymax, zmax)


def _map_phantom_faces_to_gmsh(
    phantom_map: Any,  # PhantomMap
    tol: float = 1e-6,
) -> dict[Any, list[int]]:  # dict[FaceKey, list[int]]
    """Match each PhantomMap.output_faces entry (OCP TopoDS_Face) to a gmsh face tag.

    Returns a dict[FaceKey, list[int]] — the same shape as
    phantom_map.output_faces but with gmsh tags as values. Faces with no
    match raise RuntimeError.
    """
    import gmsh

    def _round_bbox(bb: tuple[float, ...], tol: float) -> tuple[int, ...]:
        scale = 1.0 / tol
        return tuple(round(v * scale) for v in bb)

    gmsh_by_bbox: dict[tuple[int, ...], list[int]] = {}
    for dim, tag in gmsh.model.getEntities(2):
        bb = gmsh.model.getBoundingBox(dim, tag)
        gmsh_by_bbox.setdefault(_round_bbox(bb, tol), []).append(tag)

    out: dict[Any, list[int]] = {}
    for face_key, occ_faces in phantom_map.output_faces.items():
        gmsh_tags: list[int] = []
        for occ_face in occ_faces:
            bb = _occ_face_bbox(occ_face)
            key = _round_bbox(bb, tol)
            matches = gmsh_by_bbox.get(key, [])
            if not matches:
                # Fallback: linear scan with loose tolerance (10x tol).
                for dim, tag in gmsh.model.getEntities(2):
                    gbb = gmsh.model.getBoundingBox(dim, tag)
                    if all(abs(bb[i] - gbb[i]) < 10 * tol for i in range(6)):
                        matches = [tag]
                        break
            gmsh_tags.extend(matches)
        if not gmsh_tags:
            raise RuntimeError(
                f"PhantomMap face {face_key} has no matching gmsh face "
                f"(OCP bbox {_occ_face_bbox(occ_faces[0])} not found in gmsh model)."
            )
        out[face_key] = gmsh_tags
    return out
