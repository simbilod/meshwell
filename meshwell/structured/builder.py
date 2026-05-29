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

from meshwell.structured.logging import phase_timed, phase_timer
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


@phase_timed("resolve_mesh_plan")
def resolve_mesh_plan(plan: StructuredPlan, entities: list[Any]) -> StructuredMeshPlan:
    """Look up (n_layers, recombine) for each slab via its owning spec.

    Cross-checks every ``OverlapPair`` in the plan: if the loser slab's
    spec n_layers != the winner's, raises
    ``StructuredMeshOverlapError``. This is a paranoid double-check;
    Phase-1's Policy B already catches direct mismatches at plan time.
    """
    n_layers_list: list[int] = []
    recombine_list: list[bool] = []
    recombine_lat_list: list[bool] = []
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
        recombine_lat_list.append(bool(spec.recombine_lateral_faces))

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
        recombine_lateral_faces=tuple(recombine_lat_list),
    )


def _stamp_top_face_mesh(
    bottom_face_tag: int,
    top_face_tag: int,
    zlo: float,
    zhi: float,
    edge_correspondence: dict[int, int] | None = None,
) -> dict[int, int]:
    """Replace the top OCC face's 2D mesh with a translated copy of the bottom's.

    When ``edge_correspondence`` is provided (Phase 5(a)+), boundary node
    correspondence is established by edge identity + parametric position
    instead of XY matching.  ``edge_correspondence`` maps
    bot_edge_gmsh_tag -> top_edge_gmsh_tag for each bounding edge of the
    bottom face.

    When ``edge_correspondence`` is None (legacy path), falls back to the
    old XY-matching approach (kept for single-piece no-BOP scenes).

    Returns a dict mapping bottom node tag -> newly-allocated top node
    tag (so the volume builder can use it for prism construction).
    """
    import gmsh

    height = zhi - zlo

    # Collect bottom face's triangulation.
    elem_types, _, elem_nodes_per_type = gmsh.model.mesh.getElements(2, bottom_face_tag)
    bot_tri_nodes: list[np.ndarray] = []
    for et, en in zip(elem_types, elem_nodes_per_type):
        if et == 2:
            bot_tri_nodes.append(np.asarray(en, dtype=np.int64).reshape(-1, 3))
    if not bot_tri_nodes:
        return {}
    bot_triangles = np.concatenate(bot_tri_nodes, axis=0)

    bot_to_top_tag: dict[int, int] = {}

    if edge_correspondence:
        # --- Phase 5(a): edge-identity + parametric-position matching ---

        # Step 1: match boundary nodes per edge pair using XY position.
        # Parametric matching is unreliable because BOP can reverse edge
        # orientation (making top parametrize in the opposite direction).
        # Since bot and top edges are the same curve at different Z,
        # XY-distance matching is robust regardless of parametric orientation.
        bot_face_boundary = gmsh.model.getBoundary(
            [(2, bottom_face_tag)], oriented=False, recursive=False
        )
        for dim_b, bot_edge_tag in bot_face_boundary:
            if dim_b != 1:
                continue
            top_edge_tag = edge_correspondence.get(bot_edge_tag)
            if top_edge_tag is None:
                continue

            bot_ntags, bot_coords_flat_e, _ = gmsh.model.mesh.getNodes(
                1, bot_edge_tag, includeBoundary=True
            )
            top_ntags, top_coords_flat_e, _ = gmsh.model.mesh.getNodes(
                1, top_edge_tag, includeBoundary=True
            )

            bot_ntags = np.asarray(bot_ntags, dtype=np.int64)
            top_ntags = np.asarray(top_ntags, dtype=np.int64)
            bot_coords_e = np.asarray(bot_coords_flat_e, dtype=float).reshape(-1, 3)
            top_coords_e = np.asarray(top_coords_flat_e, dtype=float).reshape(-1, 3)

            if len(bot_ntags) != len(top_ntags):
                raise RuntimeError(
                    "Phase 5(a): bottom/top edge node count mismatch — "
                    "gmsh's 1D mesher produced different node counts on "
                    f"bottom edge {bot_edge_tag} ({len(bot_ntags)} nodes) "
                    f"vs top edge {top_edge_tag} ({len(top_ntags)} nodes); "
                    "need setPeriodic or remesh fix (future Phase work)."
                )

            # For each bot node, find the top node with matching XY.
            for bi in range(len(bot_ntags)):
                bx, by = bot_coords_e[bi, 0], bot_coords_e[bi, 1]
                dists = np.hypot(top_coords_e[:, 0] - bx, top_coords_e[:, 1] - by)
                ti = int(np.argmin(dists))
                bot_to_top_tag[int(bot_ntags[bi])] = int(top_ntags[ti])

        # Phase 6 cleanup: defensive check. Greedy argmin pairing can't produce
        # duplicates for valid vertical extrusions (corresponding nodes are at
        # XY distance ~0). If it does, top mesh stamping is broken.
        _n_vals = len(bot_to_top_tag.values())
        _n_unique = len(set(bot_to_top_tag.values()))
        if _n_unique != _n_vals:
            raise RuntimeError(
                f"_stamp_top_face_mesh: greedy argmin produced duplicate top tags "
                f"for distinct bot nodes ({_n_vals} mappings, "
                f"{_n_unique} unique). Inputs likely have "
                f"a degenerate edge (zero XY extent)."
            )

        # Step 2: collect interior bottom nodes (boundary=False).
        bot_int_tags_arr, bot_int_coords_flat, _ = gmsh.model.mesh.getNodes(
            2, bottom_face_tag, includeBoundary=False
        )
        bot_int_tags_arr = np.asarray(bot_int_tags_arr, dtype=np.int64)
        bot_int_coords = np.asarray(bot_int_coords_flat, dtype=float).reshape(-1, 3)

        # Clear the top face's 2D mesh. Boundary nodes (on curves/vertices) survive.
        with contextlib.suppress(Exception):
            gmsh.model.mesh.clear([(2, top_face_tag)])

        # Step 3: allocate fresh tags for interior nodes only.
        next_tag = int(gmsh.model.mesh.getMaxNodeTag()) + 1
        new_interior_tags: list[int] = []
        new_interior_coords_flat: list[float] = []
        for i, bt in enumerate(bot_int_tags_arr):
            new_tag = next_tag + i
            bot_to_top_tag[int(bt)] = new_tag
            new_interior_tags.append(new_tag)
            new_interior_coords_flat.extend(
                [bot_int_coords[i, 0], bot_int_coords[i, 1], zlo + height]
            )

        if new_interior_tags:
            gmsh.model.mesh.addNodes(
                2, top_face_tag, new_interior_tags, new_interior_coords_flat
            )

    else:
        # --- Legacy path: XY-matching (no edge correspondence available) ---

        # Collect all bottom mesh data before clearing anything.
        bot_all_tags_arr, bot_all_coords_flat, _ = gmsh.model.mesh.getNodes(
            2, bottom_face_tag, includeBoundary=True
        )
        bot_all_tags = np.asarray(bot_all_tags_arr, dtype=np.int64)
        bot_all_coords = np.asarray(bot_all_coords_flat, dtype=float).reshape(-1, 3)

        bot_int_tags_arr2, _, _ = gmsh.model.mesh.getNodes(
            2, bottom_face_tag, includeBoundary=False
        )
        bot_int_tags = set(bot_int_tags_arr2.tolist())

        # Clear the top face's 2D mesh. Boundary nodes (on curves/vertices) survive.
        with contextlib.suppress(Exception):
            gmsh.model.mesh.clear([(2, top_face_tag)])

        top_bnd_tags_arr, top_bnd_coords_flat, _ = gmsh.model.mesh.getNodes(
            2, top_face_tag, includeBoundary=True
        )
        top_bnd_tags = np.asarray(top_bnd_tags_arr, dtype=np.int64)
        top_bnd_coords = np.asarray(top_bnd_coords_flat, dtype=float).reshape(-1, 3)

        # Build XY -> top-boundary-tag lookup for fast matching.
        top_bnd_xy_to_tag: dict[tuple[float, float], int] = {}
        for i, tt in enumerate(top_bnd_tags):
            key = (round(top_bnd_coords[i, 0], 9), round(top_bnd_coords[i, 1], 9))
            top_bnd_xy_to_tag[key] = int(tt)

        new_interior_tags2: list[int] = []
        new_interior_coords_flat2: list[float] = []
        next_tag = int(gmsh.model.mesh.getMaxNodeTag()) + 1
        new_node_counter = 0

        for i, bt in enumerate(bot_all_tags):
            bt_int = int(bt)
            if bt_int not in bot_int_tags:
                # Boundary node: match by XY.
                key = (round(bot_all_coords[i, 0], 9), round(bot_all_coords[i, 1], 9))
                top_tt = top_bnd_xy_to_tag.get(key)
                if top_tt is not None:
                    bot_to_top_tag[bt_int] = top_tt
            else:
                new_tag = next_tag + new_node_counter
                new_node_counter += 1
                bot_to_top_tag[bt_int] = new_tag
                new_interior_tags2.append(new_tag)
                new_interior_coords_flat2.extend(
                    [bot_all_coords[i, 0], bot_all_coords[i, 1], zlo + height]
                )

        if new_interior_tags2:
            gmsh.model.mesh.addNodes(
                2, top_face_tag, new_interior_tags2, new_interior_coords_flat2
            )

    # Stamp triangles using the (boundary + interior) mapping.
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


def _aggregate_slab_fuzzy(
    slab_fuzzy_values: list[float | None],
    default: float,
) -> float:
    """Aggregate per-slab fragment_fuzzy_value into a single dedup tolerance.

    Returns ``max(non-None values)`` or ``default`` if all are None.
    Logs a WARNING when the non-None values are heterogeneous, because
    ``removeDuplicateNodes`` uses a single tolerance for the entire
    mesh; the loosest slab silently couples to the tightest.
    """
    import logging

    logger = logging.getLogger(__name__)
    non_none = [v for v in slab_fuzzy_values if v is not None]
    if not non_none:
        return default
    if len(set(non_none)) > 1:
        logger.warning(
            "heterogeneous fragment_fuzzy_value across slabs: %s; "
            "removeDuplicateNodes will use max=%s, which may silently "
            "merge nodes in tighter slabs.",
            sorted(set(non_none)),
            max(non_none),
        )
    return max(non_none)


def _stamp_top_from_discrete_bot(
    bot_disc_tag: int,
    top_occ_tag: int,
    zhi: float,
) -> dict[int, int]:
    """Stamp the top OCC face's mesh using an interior discrete bot entity.

    Used in Phase 3 when the bot face is a discrete interior entity (upper
    slab). The discrete entity has no bounding curves, so we can't use
    edge-correspondence or the standard interior/boundary split. Instead:
    1. XY-match discrete entity nodes against top OCC face boundary nodes.
    2. Create new interior nodes at z=zhi for unmatched discrete nodes.
    3. Stamp triangles onto the top OCC face.

    Returns bot_to_top_map (discrete_node_tag → top_face_node_tag).
    """
    import gmsh

    # Read the discrete bot entity's triangulation and all nodes.
    bot_all_tags_arr, bot_all_coords_flat, _ = gmsh.model.mesh.getNodes(
        2, bot_disc_tag, includeBoundary=True
    )
    bot_all_tags = np.asarray(bot_all_tags_arr, dtype=np.int64)
    bot_all_coords = np.asarray(bot_all_coords_flat, dtype=float).reshape(-1, 3)

    elem_types, _, elem_nodes_per_type = gmsh.model.mesh.getElements(2, bot_disc_tag)
    bot_tri_nodes: list[np.ndarray] = []
    for et, en in zip(elem_types, elem_nodes_per_type):
        if et == 2:
            bot_tri_nodes.append(np.asarray(en, dtype=np.int64).reshape(-1, 3))
    if not bot_tri_nodes:
        return {}
    bot_triangles = np.concatenate(bot_tri_nodes, axis=0)

    # Clear the top face's 2D mesh (boundary nodes on curves survive).
    import contextlib

    with contextlib.suppress(Exception):
        gmsh.model.mesh.clear([(2, top_occ_tag)])

    # Re-read just boundary nodes (survived the clear).
    top_bnd_only_tags_arr, top_bnd_only_coords_flat, _ = gmsh.model.mesh.getNodes(
        2, top_occ_tag, includeBoundary=True
    )
    top_bnd_only_tags = np.asarray(top_bnd_only_tags_arr, dtype=np.int64)
    top_bnd_only_coords = np.asarray(top_bnd_only_coords_flat, dtype=float).reshape(
        -1, 3
    )

    # Build XY → top boundary node tag lookup.
    top_xy_to_tag: dict[tuple[float, float], int] = {}
    for i, tt in enumerate(top_bnd_only_tags):
        key = (round(top_bnd_only_coords[i, 0], 9), round(top_bnd_only_coords[i, 1], 9))
        top_xy_to_tag[key] = int(tt)

    # Match each discrete bot node to a top boundary node (by XY) or create new.
    bot_to_top: dict[int, int] = {}
    new_tags: list[int] = []
    new_coords: list[float] = []
    next_tag = int(gmsh.model.mesh.getMaxNodeTag()) + 1

    for i, bt in enumerate(bot_all_tags):
        xy_key = (round(bot_all_coords[i, 0], 9), round(bot_all_coords[i, 1], 9))
        top_tt = top_xy_to_tag.get(xy_key)
        if top_tt is not None:
            bot_to_top[int(bt)] = top_tt
        else:
            new_tag = next_tag
            next_tag += 1
            bot_to_top[int(bt)] = new_tag
            new_tags.append(new_tag)
            new_coords.extend([bot_all_coords[i, 0], bot_all_coords[i, 1], zhi])

    if new_tags:
        gmsh.model.mesh.addNodes(2, top_occ_tag, new_tags, new_coords)

    # Stamp triangles.
    top_tri_flat: list[int] = []
    for tri in bot_triangles:
        top_tri_flat.extend(
            [bot_to_top[int(tri[0])], bot_to_top[int(tri[1])], bot_to_top[int(tri[2])]]
        )

    next_elem = int(gmsh.model.mesh.getMaxElementTag()) + 1
    elem_tags = list(range(next_elem, next_elem + bot_triangles.shape[0]))
    gmsh.model.mesh.addElements(2, top_occ_tag, [2], [elem_tags], [top_tri_flat])

    return bot_to_top


def _create_discrete_interior_face(
    bot_face_tag: int,
    z_top: float,
) -> tuple[int, dict[int, int]]:
    """Create a discrete 2D entity for an interior Phase 3 cohort interface.

    Used when the interface has no OCC backing (it was intentionally excluded
    from the cohort envelope solid). Creates a new discrete 2D entity at z_top
    by copying the bot face's triangulation and translating nodes to z_top.

    Returns:
        (disc_tag, bot_to_top_map) where disc_tag is the new gmsh 2D entity tag
        and bot_to_top_map maps each bot node tag to its new top node tag.
    """
    import gmsh

    # Read bot face's full mesh (nodes + triangles).
    bot_all_tags_arr, bot_all_coords_flat, _ = gmsh.model.mesh.getNodes(
        2, bot_face_tag, includeBoundary=True
    )
    bot_all_tags = np.asarray(bot_all_tags_arr, dtype=np.int64)
    bot_all_coords = np.asarray(bot_all_coords_flat, dtype=float).reshape(-1, 3)

    elem_types, _, elem_nodes_per_type = gmsh.model.mesh.getElements(2, bot_face_tag)
    bot_tri_nodes: list[np.ndarray] = []
    for et, en in zip(elem_types, elem_nodes_per_type):
        if et == 2:  # triangle
            bot_tri_nodes.append(np.asarray(en, dtype=np.int64).reshape(-1, 3))
    if not bot_tri_nodes:
        raise RuntimeError(
            f"Interior Phase 3 face: bot face {bot_face_tag} has no triangle elements."
        )
    bot_triangles = np.concatenate(bot_tri_nodes, axis=0)

    # Create a new discrete 2D entity (no bounding curves — standalone discrete face).
    disc_tag = gmsh.model.addDiscreteEntity(2, -1, [])

    # Allocate new node tags for all bot nodes (translated to z_top).
    next_tag = int(gmsh.model.mesh.getMaxNodeTag()) + 1
    bot_to_top: dict[int, int] = {}
    new_tags: list[int] = []
    new_coords: list[float] = []
    for i, bt in enumerate(bot_all_tags):
        new_tag = next_tag + i
        bot_to_top[int(bt)] = new_tag
        new_tags.append(new_tag)
        new_coords.extend([bot_all_coords[i, 0], bot_all_coords[i, 1], z_top])

    gmsh.model.mesh.addNodes(2, disc_tag, new_tags, new_coords)

    # Stamp triangles using the mapping.
    top_tri_nodes_flat: list[int] = []
    for tri in bot_triangles:
        top_tri_nodes_flat.extend(
            [bot_to_top[int(tri[0])], bot_to_top[int(tri[1])], bot_to_top[int(tri[2])]]
        )

    next_elem = int(gmsh.model.mesh.getMaxElementTag()) + 1
    n_tri = bot_triangles.shape[0]
    elem_tags = list(range(next_elem, next_elem + n_tri))
    gmsh.model.mesh.addElements(2, disc_tag, [2], [elem_tags], [top_tri_nodes_flat])

    return disc_tag, bot_to_top


def _stamp_phase3_interior_interfaces(
    plan: StructuredPlan,
    phantom_map: Any,
    bot_node_tags_per_piece: dict[tuple[int, int], np.ndarray],  # noqa: ARG001
    top_node_tags_per_piece: dict[tuple[int, int], np.ndarray],  # noqa: ARG001
    bot_cells_per_piece: dict[tuple[int, int], np.ndarray],
    interface_delimiter: str = "___",
) -> None:
    """Materialize discrete 2D entities for horizontal interior cohort interfaces.

    Walks each cohort's stacked slab pairs. Where the upper slab's bottom
    XY footprint overlaps the lower slab's top, the shared mesh becomes a
    discrete 2D entity tagged "{lower}___{upper}".

    Records each interface's FaceKey in phantom_map.face_keys_to_discrete.

    Task 16 will extend this with vertical interior interfaces.
    """
    import gmsh

    from meshwell.structured.spec import FaceKey

    cohort_slabs_by_idx: dict[int, list[tuple[int, Any]]] = {}
    for slab_idx, slab in enumerate(plan.slabs):
        cohort_slabs_by_idx.setdefault(slab.component_index, []).append(
            (slab_idx, slab)
        )

    for slab_list in cohort_slabs_by_idx.values():
        slab_list.sort(key=lambda pair: pair[1].zlo)
        for i in range(len(slab_list) - 1):
            _lower_idx, lower = slab_list[i]
            upper_idx, upper = slab_list[i + 1]
            if abs(upper.zlo - lower.zhi) > 1e-9:
                continue  # not touching in z — not an interface

            # For each piece in the upper slab, if its bot mesh exists in
            # the per-piece bookkeeping, stamp a discrete 2D entity with
            # those cells.
            for u_pidx, _piece in enumerate(upper.face_partition):
                key = (upper_idx, u_pidx)
                if key not in bot_cells_per_piece:
                    continue
                cells = bot_cells_per_piece[key]
                if cells.size == 0:
                    continue
                disc_tag = gmsh.model.addDiscreteEntity(2, -1, [])
                next_elem = int(gmsh.model.mesh.getMaxElementTag()) + 1
                n_elem = cells.shape[0]
                elem_tags = list(range(next_elem, next_elem + n_elem))
                elem_type = 3 if cells.shape[1] == 4 else 2  # quad / tri
                gmsh.model.mesh.addElements(
                    2, disc_tag, [elem_type], [elem_tags], [cells.flatten().tolist()]
                )
                lo_name = "/".join(lower.physical_name)
                up_name = "/".join(upper.physical_name)
                pg_name = f"{lo_name}{interface_delimiter}{up_name}"
                pg_tag = gmsh.model.addPhysicalGroup(2, [disc_tag])
                gmsh.model.setPhysicalName(2, pg_tag, pg_name)
                phantom_map.face_keys_to_discrete[
                    FaceKey(upper_idx, "bot", u_pidx)
                ] = disc_tag


def _stamp_vertical_interior_interfaces(
    plan: StructuredPlan,
    phantom_map: Any,  # noqa: ARG001
    bot_node_tags_per_piece: dict[tuple[int, int], "np.ndarray"],
    slab_piece_top_map: dict[tuple[int, int], dict[int, int]],
    slab_piece_layer_maps: dict[tuple[int, int], list[dict[int, int]]] | None = None,
    interface_delimiter: str = "___",
) -> None:
    """Stamp discrete 2D entities for vertical interior cohort interfaces.

    For each cohort, groups slabs by (zlo, zhi).  In each z-interval that
    has ≥2 pieces across cohort slabs, finds pairs of pieces whose XY
    footprint polygons share a boundary segment (the shared arrangement
    edge).  Stamps a vertical rectangular strip of quads over the slab's
    z-extent and assigns a physical group "{slab_a}___{slab_b}".

    Supports n_layers >= 1 (Task 17).  When slab_piece_layer_maps is
    provided, the strip spans n_layers + 1 z-rows producing n_layers rows
    of quads.  Falls back to the n_layers=1 behaviour when the map is
    absent (Task 16 compatibility).
    """
    import gmsh
    from shapely.ops import linemerge

    cohort_slabs_by_idx: dict[int, list[tuple[int, Any]]] = {}
    for slab_idx, slab in enumerate(plan.slabs):
        cohort_slabs_by_idx.setdefault(slab.component_index, []).append(
            (slab_idx, slab)
        )

    for slab_list in cohort_slabs_by_idx.values():
        # Group by z-interval.
        z_interval_groups: dict[tuple[float, float], list[tuple[int, Any]]] = {}
        for slab_idx, slab in slab_list:
            key = (slab.zlo, slab.zhi)
            z_interval_groups.setdefault(key, []).append((slab_idx, slab))

        for interval_slabs in z_interval_groups.values():
            if len(interval_slabs) < 2:
                continue  # Only one slab at this z-interval — no vertical interface.

            # For each pair of slabs, check if their XY footprints share a
            # boundary edge (LineString intersection).
            for i_a in range(len(interval_slabs)):
                slab_a_idx, slab_a = interval_slabs[i_a]
                for i_b in range(i_a + 1, len(interval_slabs)):
                    _slab_b_idx, slab_b = interval_slabs[i_b]

                    # Each slab in the same z-interval has exactly one piece (piece 0)
                    # for the simple case.  For multi-piece per slab, we'd need to
                    # iterate pieces — but for Task 16 n_layers=1 the common case is
                    # one piece per slab.
                    for piece_a_idx in range(len(slab_a.face_partition)):
                        for piece_b_idx in range(len(slab_b.face_partition)):
                            poly_a = slab_a.face_partition[piece_a_idx]
                            poly_b = slab_b.face_partition[piece_b_idx]
                            shared = poly_a.boundary.intersection(poly_b.boundary)

                            # Only proceed if the shared geometry is a line.
                            if shared.is_empty:
                                continue
                            if shared.geom_type == "MultiLineString":
                                shared = linemerge(shared)
                            if shared.geom_type not in (
                                "LineString",
                                "MultiLineString",
                            ):
                                continue
                            if shared.is_empty or shared.length < 1e-9:
                                continue

                            # Get bot nodes for piece_a from the already-collected map.
                            key_a = (slab_a_idx, piece_a_idx)
                            if key_a not in bot_node_tags_per_piece:
                                continue
                            bot_tags_a = bot_node_tags_per_piece[key_a]
                            if len(bot_tags_a) == 0:
                                continue

                            top_map_a = slab_piece_top_map.get(key_a)
                            if top_map_a is None:
                                continue

                            # Build tag → (x, y, z) for bot nodes of piece_a.
                            # gmsh.model.mesh.getNode(t) is acceptable for the
                            # typically small number of boundary nodes.
                            bot_xyzs: dict[int, tuple[float, float, float]] = {}
                            for bt in bot_tags_a:
                                (
                                    coords_flat,
                                    _pcoords,
                                    _edim,
                                    _etag,
                                ) = gmsh.model.mesh.getNode(int(bt))
                                bot_xyzs[int(bt)] = (
                                    float(coords_flat[0]),
                                    float(coords_flat[1]),
                                    float(coords_flat[2]),
                                )

                            # Filter nodes within 1e-7 of the shared LineString.
                            import shapely.geometry as sg

                            on_shared: list[int] = []
                            for bt, (x, y, _z) in bot_xyzs.items():
                                pt = sg.Point(x, y)
                                if shared.distance(pt) < 1e-7:
                                    on_shared.append(bt)

                            if len(on_shared) < 2:
                                continue

                            # Order nodes along the shared LineString by projection.
                            _shared_ref = shared
                            _bot_xyzs_ref = bot_xyzs
                            on_shared.sort(
                                key=lambda tag: _shared_ref.project(
                                    sg.Point(
                                        _bot_xyzs_ref[tag][0], _bot_xyzs_ref[tag][1]
                                    )
                                )
                            )

                            # Build vertical quad strip spanning n_layers z-rows.
                            # Row 0 = bot_row (on_shared), row i = layer_maps[i-1].
                            # Each consecutive row pair produces (n_nodes-1) quads.
                            n_nodes = len(on_shared)
                            n_edge_quads = n_nodes - 1
                            if n_edge_quads < 1:
                                continue

                            # Resolve the full ordered list of layer maps for piece_a.
                            _lm_key_a = (slab_a_idx, piece_a_idx)
                            if (
                                slab_piece_layer_maps is not None
                                and _lm_key_a in slab_piece_layer_maps
                            ):
                                _all_layer_maps = slab_piece_layer_maps[_lm_key_a]
                            else:
                                # Fallback: single layer from slab_piece_top_map (Task 16).
                                _all_layer_maps = [top_map_a]

                            # Build rows: row 0 is on_shared; subsequent rows look up
                            # each bot-tag in the successive layer maps.
                            rows: list[list[int]] = [list(on_shared)]
                            for lm in _all_layer_maps:
                                row = [lm.get(bt) for bt in on_shared]  # type: ignore[assignment]
                                rows.append(row)  # type: ignore[arg-type]

                            disc_tag = gmsh.model.addDiscreteEntity(2, -1, [])
                            quad_nodes: list[int] = []
                            for z_slice in range(len(_all_layer_maps)):
                                row_lo = rows[z_slice]
                                row_hi = rows[z_slice + 1]
                                for k in range(n_edge_quads):
                                    b0 = row_lo[k]
                                    b1 = row_lo[k + 1]
                                    t0 = row_hi[k]
                                    t1 = row_hi[k + 1]
                                    if (
                                        b0 is None
                                        or b1 is None
                                        or t0 is None
                                        or t1 is None
                                    ):
                                        continue
                                    quad_nodes.extend([b0, b1, t1, t0])

                            if not quad_nodes:
                                continue

                            actual_quads = len(quad_nodes) // 4
                            next_elem_tag = int(gmsh.model.mesh.getMaxElementTag()) + 1
                            elem_tags = list(
                                range(next_elem_tag, next_elem_tag + actual_quads)
                            )
                            gmsh.model.mesh.addElements(
                                2, disc_tag, [3], [elem_tags], [quad_nodes]
                            )

                            # Assign physical group.
                            name_a = "/".join(slab_a.physical_name)
                            name_b = "/".join(slab_b.physical_name)
                            pg_name = f"{name_a}{interface_delimiter}{name_b}"
                            pg_tag = gmsh.model.addPhysicalGroup(2, [disc_tag])
                            gmsh.model.setPhysicalName(2, pg_tag, pg_name)


def _remove_empty_cohort_envelope_volumes() -> None:
    """Remove cohort envelope OCC 3D entities that have no mesh elements.

    Called at the end of apply_structured_mesh when _USE_DISCRETE_COHORT_MESH
    is True.  At that point, every per-piece discrete 3D entity has already
    been stamped with structured wedge/hex elements.  The only remaining
    un-meshed 3D entities are the cohort envelope OCC volumes — their 2D
    faces carry sub-face mesh stamps, but the volumes themselves are empty.
    Removing them (non-recursively) prevents the 3D mesh pass from
    tetrahedralizing them when MeshOnlyEmpty=1 is active.
    """
    import logging

    import gmsh

    logger = logging.getLogger(__name__)

    to_remove: list[tuple[int, int]] = []
    for _dim, vol_tag in gmsh.model.getEntities(3):
        elem_types, _, _ = gmsh.model.mesh.getElements(3, vol_tag)
        if len(elem_types) == 0:
            to_remove.append((3, vol_tag))

    if to_remove:
        logger.debug(
            "Task 18: removing %d empty cohort envelope volume(s): %s",
            len(to_remove),
            [t for _, t in to_remove],
        )
        gmsh.model.removeEntities(to_remove, recursive=False)


@phase_timed("mesh_apply")
def apply_structured_mesh(
    plan: StructuredPlan,
    mesh_plan: StructuredMeshPlan,
    phantom_map: Any,
    occ_entities: list[Any],
    fuzzy_tol: float = 1e-6,
) -> list[int]:
    """Run the mesh-stage Layer C: per-piece top mesh stamp + slab volume build.

    Uses PhantomMap to route per (slab, piece). Phase 5(d): each piece has
    exactly 1 bot + 1 top OCC face by construction (common-refinement partition).

    Returns a flat list of all per-piece volume entity tags created or
    populated (one per piece, not per slab).
    """
    import logging

    import gmsh

    from meshwell.structured.spec import EdgeKey, FaceKey

    logger = logging.getLogger(__name__)

    face_map = _map_phantom_faces_to_gmsh(phantom_map, occ_entities)
    edge_map = _map_phantom_edges_to_gmsh(phantom_map, occ_entities)

    # Suppress interior-seam 2D meshes: each output lateral face shared between
    # two pieces of the same slab is an internal seam that doesn't need
    # a 2D mesh (the wedge volume covers it implicitly). Clear those faces.
    if phantom_map.output_laterals:
        lateral_gmsh_map = _map_phantom_laterals_to_gmsh(phantom_map, occ_entities)
        face_tag_to_keys: dict[int, list[Any]] = {}
        for lk, gtags in lateral_gmsh_map.items():
            for gtag in gtags:
                face_tag_to_keys.setdefault(gtag, []).append(lk)
        for face_tag, keys in face_tag_to_keys.items():
            if len(keys) >= 2:
                slabs = {k.slab_index for k in keys}
                pieces = {k.piece_index for k in keys}
                if len(slabs) == 1 and len(pieces) >= 2:
                    try:
                        gmsh.model.mesh.clear([(2, face_tag)])
                        logger.debug(
                            "Cleared interior seam face %d (keys: %s)", face_tag, keys
                        )
                    except Exception as e:
                        logger.warning(
                            "clear failed on interior seam face %d: %s", face_tag, e
                        )

    bot_node_tags_per_piece: dict[tuple[int, int], np.ndarray] = {}
    top_node_tags_per_piece: dict[tuple[int, int], np.ndarray] = {}
    bot_cells_per_piece: dict[tuple[int, int], np.ndarray] = {}

    from meshwell.structured.phantom import _USE_DISCRETE_COHORT_MESH

    # Phase 3 pre-pass: create discrete 2D entities for interior cohort interfaces.
    # An interior interface is where the top face of the lower slab (or bot face of
    # the upper slab) has no OCC backing. We create a discrete entity by extruding
    # the lower slab's OCC bot face to z_interface, and record the mapping so that
    # both FaceKey(lower_idx, 'top', piece) and FaceKey(upper_idx, 'bot', piece)
    # resolve to the same discrete entity tag in the per-slab loop below.
    interior_discrete: dict[FaceKey, int] = {}  # FaceKey -> discrete gmsh tag
    interior_bot_to_top: dict[FaceKey, dict[int, int]] = {}  # FaceKey -> bot_to_top map
    if _USE_DISCRETE_COHORT_MESH:
        cohort_slabs_by_idx_pre: dict[int, list[tuple[int, Any]]] = {}
        for slab_idx_pre, slab_pre in enumerate(plan.slabs):
            cohort_slabs_by_idx_pre.setdefault(slab_pre.component_index, []).append(
                (slab_idx_pre, slab_pre)
            )
        for slab_list_pre in cohort_slabs_by_idx_pre.values():
            slab_list_pre.sort(key=lambda pair: pair[1].zlo)
            for i_pre in range(len(slab_list_pre) - 1):
                lower_idx_pre, lower_pre = slab_list_pre[i_pre]
                upper_idx_pre, upper_pre = slab_list_pre[i_pre + 1]
                if abs(upper_pre.zlo - lower_pre.zhi) > 1e-9:
                    continue
                # Check that the top of the lower slab is indeed interior.
                for p_pre in range(len(lower_pre.face_partition)):
                    top_key_pre = FaceKey(lower_idx_pre, "top", p_pre)
                    if face_map.get(top_key_pre):
                        continue  # has OCC backing — not interior
                    bot_key_pre = FaceKey(lower_idx_pre, "bot", p_pre)
                    bot_tags_pre = face_map.get(bot_key_pre, [])
                    if len(bot_tags_pre) != 1:
                        continue  # can't create interior face without bot
                    disc_tag_pre, b2t_pre = _create_discrete_interior_face(
                        bot_tags_pre[0], lower_pre.zhi
                    )
                    # Add physical group: "lower___upper" interface name.
                    lo_name = "/".join(lower_pre.physical_name)
                    up_name = "/".join(upper_pre.physical_name)
                    pg_name = f"{lo_name}___{up_name}"
                    pg_tag = gmsh.model.addPhysicalGroup(2, [disc_tag_pre])
                    gmsh.model.setPhysicalName(2, pg_tag, pg_name)
                    # Record in phantom_map.
                    upper_bot_key = FaceKey(upper_idx_pre, "bot", p_pre)
                    phantom_map.face_keys_to_discrete[upper_bot_key] = disc_tag_pre
                    # Record: lower's top AND upper's bot both map to this discrete entity.
                    interior_discrete[top_key_pre] = disc_tag_pre
                    interior_bot_to_top[top_key_pre] = b2t_pre
                    interior_discrete[upper_bot_key] = disc_tag_pre
                    interior_bot_to_top[upper_bot_key] = b2t_pre

    # Cache: (bot_face_tag, top_face_tag) -> (top_map, layer_maps, vol_tag)
    # for shared-face pairs.
    # When two same-z-interval cohort slabs share the same union bot/top face (Task 16),
    # the second slab must reuse the first slab's already-built mesh and volume.
    _shared_face_cache: dict[
        tuple[int, int], tuple[dict[int, int], list[dict[int, int]], int]
    ] = {}
    # Track which slabs were given a "shared" volume so we know their top_map too.
    _slab_piece_top_map: dict[tuple[int, int], dict[int, int]] = {}
    # Task 17: full layer_maps per piece (list of n_layers dicts bot->layer_i node tag).
    _slab_piece_layer_maps: dict[tuple[int, int], list[dict[int, int]]] = {}

    vol_tags: list[int] = []
    for slab_idx, slab in enumerate(plan.slabs):
        n_layers = mesh_plan.n_layers[slab_idx]
        recombine = mesh_plan.recombine[slab_idx]
        height = slab.zhi - slab.zlo

        for piece_idx in range(len(slab.face_partition)):
            with phase_timer("mesh_apply_per_slab"):
                bot_key = FaceKey(slab_idx, "bot", piece_idx)
                top_key = FaceKey(slab_idx, "top", piece_idx)
                bot_tags = face_map.get(bot_key, [])
                top_tags = face_map.get(top_key, [])

                # Phase 3: interior faces may not be in face_map — use pre-created
                # discrete entities from the interior_discrete lookup instead.
                if len(bot_tags) == 0 and bot_key in interior_discrete:
                    bot_tags = [interior_discrete[bot_key]]
                if len(top_tags) == 0 and top_key in interior_discrete:
                    top_tags = [interior_discrete[top_key]]

                if len(bot_tags) != 1 or len(top_tags) != 1:
                    raise RuntimeError(
                        f"Slab {slab.physical_name} piece {piece_idx}: Phase 5(d) "
                        f"compute_face_partition should produce a common-refinement "
                        f"partition where each piece has exactly 1 bot + 1 top OCC face "
                        f"after BOP. Got bot={bot_tags}, top={top_tags}. This indicates a "
                        f"BOP outcome the planner didn't anticipate (fuzzy tolerance edge "
                        f"case, non-axis-aligned neighbour producing extra micro-faces, "
                        f"non-polygonal cut, etc.)."
                    )
                bot_tag = bot_tags[0]
                top_tag = top_tags[0]

                # Task 16: detect same-z-interval cohort slabs sharing the same
                # union bot/top OCC face pair.  The first slab to process a given
                # (bot_tag, top_tag) pair does the normal stamp+volume work and
                # caches the result.  Subsequent slabs with the same pair reuse
                # the cached top_map and vol_tag, and only register a new physical
                # group that points to the existing volume entity.
                _face_pair = (bot_tag, top_tag)
                if _USE_DISCRETE_COHORT_MESH and _face_pair in _shared_face_cache:
                    (
                        cached_top_map,
                        cached_layer_maps,
                        cached_vol_tag,
                    ) = _shared_face_cache[_face_pair]
                    _slab_piece_top_map[(slab_idx, piece_idx)] = cached_top_map
                    _slab_piece_layer_maps[(slab_idx, piece_idx)] = cached_layer_maps
                    # Read bot nodes for vertical-interface stamping bookkeeping.
                    bot_node_tags_arr_c, _, _ = gmsh.model.mesh.getNodes(
                        2, bot_tag, includeBoundary=True
                    )
                    bot_node_tags_per_piece[(slab_idx, piece_idx)] = np.asarray(
                        bot_node_tags_arr_c, dtype=np.int64
                    )
                    top_node_tags_per_piece[(slab_idx, piece_idx)] = np.asarray(
                        list(cached_top_map.values()), dtype=np.int64
                    )
                    elem_types_c, _, elem_nodes_c = gmsh.model.mesh.getElements(
                        2, bot_tag
                    )
                    bot_cells_c: list[np.ndarray] = []
                    for et, en in zip(elem_types_c, elem_nodes_c):
                        if et == 2:
                            bot_cells_c.append(
                                np.asarray(en, dtype=np.int64).reshape(-1, 3)
                            )
                        elif et == 3:
                            bot_cells_c.append(
                                np.asarray(en, dtype=np.int64).reshape(-1, 4)
                            )
                    if bot_cells_c:
                        bot_cells_per_piece[(slab_idx, piece_idx)] = np.concatenate(
                            bot_cells_c, axis=0
                        )
                    # Register a new physical group for this slab pointing to the
                    # shared cohort volume.
                    pg_name = "/".join(slab.physical_name)
                    pg_tag = gmsh.model.addPhysicalGroup(3, [cached_vol_tag])
                    gmsh.model.setPhysicalName(3, pg_tag, pg_name)
                    vol_tags.append(cached_vol_tag)
                    continue

                # Build per-piece edge_correspondence: dict[bot_edge_gmsh_tag, top_edge_gmsh_tag].
                edge_correspondence: dict[int, int] = {}
                bot_face_boundary = gmsh.model.getBoundary(
                    [(2, bot_tag)], oriented=False, recursive=False
                )
                for dim_b, bot_edge_tag in bot_face_boundary:
                    if dim_b != 1:
                        continue
                    for ek, tags in edge_map.items():
                        if (
                            ek.slab_index == slab_idx
                            and ek.side == "bot"
                            and ek.piece_index == piece_idx
                            and bot_edge_tag in tags
                        ):
                            top_ek = EdgeKey(slab_idx, "top", piece_idx, ek.edge_index)
                            if edge_map.get(top_ek):
                                edge_correspondence[bot_edge_tag] = edge_map[top_ek][0]
                            break

                # Phase 3: if the top face is a discrete interior entity, its mesh
                # was already created by _create_discrete_interior_face. Use the
                # pre-computed bot_to_top map directly instead of calling
                # _stamp_top_face_mesh (which would clear and re-stamp the entity).
                top_is_interior = (
                    _USE_DISCRETE_COHORT_MESH and top_key in interior_discrete
                )
                # Phase 3: if the bot face is a discrete interior entity (upper slab),
                # we need to use the mapping that was established when creating it.
                bot_is_interior = (
                    _USE_DISCRETE_COHORT_MESH and bot_key in interior_discrete
                )

                if top_is_interior:
                    # Lower slab with interior top: use pre-computed map.
                    top_map = interior_bot_to_top[top_key]
                    layer_maps = [top_map]
                    interior_layer_maps = []
                    interior_layer_coords = []
                elif bot_is_interior:
                    # Upper slab with interior bot: the bot face is the discrete
                    # interior entity created during pre-processing. Use the
                    # dedicated helper that handles discrete entities without
                    # bounding curves (XY-matching instead of edge correspondence).
                    if n_layers == 1:
                        top_map = _stamp_top_from_discrete_bot(
                            bot_tag, top_tag, slab.zhi
                        )
                        layer_maps = [top_map]
                        interior_layer_maps = []
                        interior_layer_coords = []
                    else:
                        top_map = _stamp_top_from_discrete_bot(
                            bot_tag, top_tag, slab.zhi
                        )
                        (
                            bot_node_tags_arr,
                            bot_coords_flat,
                            _,
                        ) = gmsh.model.mesh.getNodes(2, bot_tag, includeBoundary=True)
                        bot_node_tags = np.asarray(bot_node_tags_arr, dtype=np.int64)
                        bot_coords = np.asarray(bot_coords_flat, dtype=float).reshape(
                            -1, 3
                        )
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
                        layer_maps = [*interior_layer_maps, top_map]
                elif n_layers == 1:
                    # Single layer: bottom -> top is the only layer map.
                    top_map = _stamp_top_face_mesh(
                        bot_tag, top_tag, slab.zlo, slab.zhi, edge_correspondence
                    )
                    layer_maps = [top_map]
                    interior_layer_maps: list[dict[int, int]] = []
                    interior_layer_coords: list[list[float]] = []
                else:
                    # Multi-layer: stamp the top face first (so new interior node
                    # tags are registered before allocating intermediate-layer tags,
                    # avoiding tag collisions), then build (n_layers - 1) intermediate
                    # node maps between bottom and top.
                    top_map = _stamp_top_face_mesh(
                        bot_tag, top_tag, slab.zlo, slab.zhi, edge_correspondence
                    )
                    bot_node_tags_arr, bot_coords_flat, _ = gmsh.model.mesh.getNodes(
                        2, bot_tag, includeBoundary=True
                    )
                    bot_node_tags = np.asarray(bot_node_tags_arr, dtype=np.int64)
                    bot_coords = np.asarray(bot_coords_flat, dtype=float).reshape(-1, 3)
                    # Use max tag AFTER stamp so interior tags don't collide with
                    # the new top-interior nodes allocated by _stamp_top_face_mesh.
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
                    layer_maps = [*interior_layer_maps, top_map]

                # Capture per-piece bot face data for horizontal interior interface
                # stamping (_stamp_phase3_interior_interfaces, Task 15).
                bot_node_tags_arr_b, _, _ = gmsh.model.mesh.getNodes(
                    2, bot_tag, includeBoundary=True
                )
                bot_node_tags_per_piece[(slab_idx, piece_idx)] = np.asarray(
                    bot_node_tags_arr_b, dtype=np.int64
                )
                top_node_tags_per_piece[(slab_idx, piece_idx)] = np.asarray(
                    list(top_map.values()), dtype=np.int64
                )
                # Collect bot triangulation for the horizontal interior interface
                # stamp (Task 15). The bot face's 2D mesh at this point is always
                # triangles (type 2): gmsh meshes it with triangles and _stamp_top_face_mesh
                # reads and copies them; recombine happens in a later pass.
                elem_types_b, _, elem_nodes_b = gmsh.model.mesh.getElements(2, bot_tag)
                bot_cells_list: list[np.ndarray] = []
                for et, en in zip(elem_types_b, elem_nodes_b):
                    if et == 2:  # triangle
                        bot_cells_list.append(
                            np.asarray(en, dtype=np.int64).reshape(-1, 3)
                        )
                    elif et == 3:  # quad (if recombine was already applied)
                        bot_cells_list.append(
                            np.asarray(en, dtype=np.int64).reshape(-1, 4)
                        )
                if bot_cells_list:
                    bot_cells_per_piece[(slab_idx, piece_idx)] = np.concatenate(
                        bot_cells_list, axis=0
                    )

                if _USE_DISCRETE_COHORT_MESH:
                    # Phase 3: never share a cohort envelope's OCC volume across
                    # pieces — each piece needs its own discrete 3D entity so
                    # per-piece physical groups stay distinct.
                    occ_vol_tag = None
                else:
                    # Find the OCC volume entity containing both the bottom face and the
                    # top face. Add wedge elements directly to that OCC volume so it
                    # retains its physical-group membership and MeshOnlyEmpty=1 skips
                    # re-meshing it in the 3D pass.
                    occ_vol_tag: int | None = _find_volume_containing_faces(
                        bot_tag, top_tag
                    )

                # Register all interior nodes BEFORE addElements so all referenced
                # node tags exist in gmsh at the time addElements validates them.
                pre_vol_tag: int | None = occ_vol_tag
                if pre_vol_tag is None:
                    pre_vol_tag = gmsh.model.addDiscreteEntity(3, -1, [])
                # Phase 3: when we allocated a fresh discrete 3D entity (no
                # OCC volume owns the bot/top faces), the orchestrator's
                # load_occ_entities didn't attach a physical group to it.
                # Attach the slab/piece physical group directly so per-piece
                # tagging survives.
                if _USE_DISCRETE_COHORT_MESH and occ_vol_tag is None:
                    pg_name = "/".join(slab.physical_name)
                    # Find an existing physical group for this name (set by
                    # load_occ_entities for the cohort envelope OCC volume) and
                    # add the discrete volume to it.  This avoids the gmsh
                    # behaviour where setPhysicalName(dim, new_tag, name)
                    # reassigns the name to the group that already holds it,
                    # leaving the new group unnamed.
                    existing_pg_tag: int | None = None
                    for _pg_d, _pg_t in gmsh.model.getPhysicalGroups(dim=3):
                        if gmsh.model.getPhysicalName(3, _pg_t) == pg_name:
                            existing_pg_tag = _pg_t
                            break
                    if existing_pg_tag is not None:
                        # Replace the cohort envelope entity with the discrete
                        # volume in the existing physical group so the name is
                        # correctly associated with the structured mesh.
                        gmsh.model.setPhysicalName(3, existing_pg_tag, pg_name)
                        _old_ents = list(
                            gmsh.model.getEntitiesForPhysicalGroup(3, existing_pg_tag)
                        )
                        gmsh.model.removePhysicalGroups([(3, existing_pg_tag)])
                        _new_ents = [e for e in _old_ents if e != pre_vol_tag] + [
                            pre_vol_tag
                        ]
                        new_pg = gmsh.model.addPhysicalGroup(
                            3, _new_ents, tag=existing_pg_tag
                        )
                        gmsh.model.setPhysicalName(3, new_pg, pg_name)
                    else:
                        pg_tag = gmsh.model.addPhysicalGroup(3, [pre_vol_tag])
                        gmsh.model.setPhysicalName(3, pg_tag, pg_name)

                all_pre_tags: list[int] = []
                all_pre_coords: list[float] = []

                if n_layers > 1 and pre_vol_tag is not None:
                    for m, coords in zip(interior_layer_maps, interior_layer_coords):
                        all_pre_tags.extend(m.values())
                        all_pre_coords.extend(coords)

                if all_pre_tags and pre_vol_tag is not None:
                    gmsh.model.mesh.addNodes(
                        3, pre_vol_tag, all_pre_tags, all_pre_coords
                    )

                vol_tag = _build_slab_volume(
                    bottom_face_tag=bot_tag,
                    bot_to_top_layer_tags=layer_maps,
                    n_layers=n_layers,
                    recombine=recombine,
                    pre_allocated_vol_tag=pre_vol_tag,
                )

                # Cache the (bot_tag, top_tag) → (top_map, layer_maps, vol_tag) so that
                # same-z-interval cohort slabs sharing the union faces can
                # reuse the already-built volume (Task 16/17).
                if _USE_DISCRETE_COHORT_MESH:
                    _shared_face_cache[_face_pair] = (top_map, layer_maps, vol_tag)
                    _slab_piece_top_map[(slab_idx, piece_idx)] = top_map
                    _slab_piece_layer_maps[(slab_idx, piece_idx)] = layer_maps

                vol_tags.append(vol_tag)

    # Task 16/17: stamp vertical interior interfaces between laterally-adjacent
    # same-z-interval cohort slab pairs.  Task 17 extends to n_layers > 1.
    if _USE_DISCRETE_COHORT_MESH:
        _stamp_vertical_interior_interfaces(
            plan=plan,
            phantom_map=phantom_map,
            bot_node_tags_per_piece=bot_node_tags_per_piece,
            slab_piece_top_map=_slab_piece_top_map,
            slab_piece_layer_maps=_slab_piece_layer_maps,
        )

    # Task 18: Remove cohort envelope OCC 3D entities so the 3D mesh pass
    # doesn't tetrahedralize them.  Per-piece discrete 3D entities now own
    # all structured elements; the cohort envelope volumes have no elements
    # and must be dropped BEFORE generate(3) runs with MeshOnlyEmpty=1.
    # 2D OCC faces (top / bot / lateral) are intentionally kept — they carry
    # the per-piece mesh stamps used by all sub-face and interior-interface
    # entities. recursive=False ensures they are not removed.
    if _USE_DISCRETE_COHORT_MESH:
        _remove_empty_cohort_envelope_volumes()

    # Global cleanup: merge ~coincident nodes.
    fuzzy = _aggregate_slab_fuzzy(
        [slab.fragment_fuzzy_value for slab in plan.slabs],
        default=fuzzy_tol,
    )
    try:
        with phase_timer("removeDuplicateNodes"):
            gmsh.model.mesh.removeDuplicateNodes(tag=[], tol=2 * fuzzy)
    except TypeError:
        with phase_timer("removeDuplicateNodes"):
            gmsh.model.mesh.removeDuplicateNodes()

    return vol_tags


def _build_xao_compound(occ_entities: list[Any]) -> Any:
    """Reproduce the BREP compound that meshwell.occ_xao_writer.write_xao builds.

    The compound is built from OCCLabeledEntity.shapes, filtering out
    top-dim keep=False entities (mirrors occ_xao_writer.py:300-307).
    gmsh loads exactly this compound from the XAO, assigning face tags
    in TopExp::MapShapes(compound, TopAbs_FACE) order.
    """
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound

    max_dim = max((e.dim for e in occ_entities if e.shapes), default=0)
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    for ent in occ_entities:
        if ent.dim == max_dim and not ent.keep:
            continue
        for s in ent.shapes:
            builder.Add(compound, s)
    return compound


def _map_phantom_edges_to_gmsh(
    phantom_map: Any,  # PhantomMap
    occ_entities: list[Any],  # list[OCCLabeledEntity]
) -> dict[Any, list[int]]:  # dict[EdgeKey, list[int]]
    """Map each PhantomMap.output_edges entry (OCP TopoDS_Edge) to gmsh edge tags.

    Mirrors _map_phantom_faces_to_gmsh but uses TopAbs_EDGE. Per the spike
    (docs/superpowers/spikes/xao_load_tag_determinism.py), gmsh assigns
    tags in TopExp::MapShapes order per-dim, so the same lookup pattern
    works for edges.

    Returns dict[EdgeKey, list[int]] — gmsh edge tags per PhantomMap entry.
    Raises RuntimeError if any input edge has no IsSame() match.
    """
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape

    compound = _build_xao_compound(occ_entities)
    emap = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(compound, TopAbs_EDGE, emap)

    out: dict[Any, list[int]] = {}
    for edge_key, occ_edges in phantom_map.output_edges.items():
        tags: list[int] = []
        for e in occ_edges:
            idx = emap.FindIndex(e)
            if idx == 0:
                raise RuntimeError(
                    f"PhantomMap edge {edge_key} has no IsSame() match in "
                    f"the XAO compound."
                )
            tags.append(idx)
        out[edge_key] = tags
    return out


def _map_phantom_laterals_to_gmsh(
    phantom_map: Any,  # PhantomMap
    occ_entities: list[Any],  # list[OCCLabeledEntity]
) -> dict[Any, list[int]]:  # dict[LateralKey, list[int]]
    """Map each PhantomMap.output_laterals entry (OCP TopoDS_Face) to gmsh face tags.

    Mirrors _map_phantom_faces_to_gmsh but walks ``phantom_map.output_laterals``
    using TopAbs_FACE (lateral faces are faces in the compound too).

    Returns dict[LateralKey, list[int]] — gmsh face tags per LateralKey.
    Raises RuntimeError if any lateral face has no IsSame() match.
    """
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape

    compound = _build_xao_compound(occ_entities)
    fmap = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(compound, TopAbs_FACE, fmap)

    out: dict[Any, list[int]] = {}
    for lat_key, occ_faces in phantom_map.output_laterals.items():
        tags: list[int] = []
        for f in occ_faces:
            idx = fmap.FindIndex(f)
            if idx == 0:
                raise RuntimeError(
                    f"PhantomMap lateral {lat_key} has no IsSame() match in "
                    f"the XAO compound."
                )
            tags.append(idx)
        out[lat_key] = tags
    return out


@phase_timed("transfinite_hints")
def apply_structured_transfinite_hints(
    mesh_plan: Any,  # StructuredMeshPlan
    phantom_map: Any,  # PhantomMap
    occ_entities: list[Any],  # list[OCCLabeledEntity]
) -> None:
    """Apply setTransfiniteSurface to lateral OCC faces before 2D meshing.

    For each LateralKey in phantom_map.output_laterals where
    lateral_has_midheight_cut is False, set setTransfiniteCurve(n_layers+1)
    on each VERTICAL bounding edge and setTransfiniteSurface on the face.
    This forces gmsh's 2D mesher to produce a structured grid that matches
    the wedge volume's implicit lateral structure exactly.

    A VERTICAL edge is one whose two endpoint vertices have different z values.
    Only vertical edges are constrained; horizontal edges may be shared with
    non-structured neighbours and should not be forced to a conflicting count.

    Each call is wrapped in try/except so that if gmsh complains (e.g. the
    face has >4 corners due to a mid-height cut we didn't detect, or a curve
    is already transfinite with a different count), that face is silently
    skipped with a warning.
    """
    import logging

    import gmsh

    logger = logging.getLogger(__name__)

    if not phantom_map.output_laterals:
        return

    lateral_map = _map_phantom_laterals_to_gmsh(phantom_map, occ_entities)

    for lat_key, face_tags in lateral_map.items():
        if phantom_map.lateral_has_midheight_cut.get(lat_key, False):
            continue

        slab_idx = lat_key.slab_index
        n_layers = mesh_plan.n_layers[slab_idx]
        recombine_lat = mesh_plan.recombine_lateral_faces[slab_idx]

        for face_tag in face_tags:
            try:
                bnd = gmsh.model.getBoundary(
                    [(2, face_tag)], oriented=False, recursive=False
                )
                # Skip faces that gmsh's transfinite mesher will reject:
                #   - degenerate (< 3 boundary edges): BOP can leave
                #     0-area "seam" faces when phantom solids of multiple
                #     pieces share a near-coincident interface.
                #   - multi-wire (face with hole): BOP can produce a
                #     shared lateral face that absorbs an inner ring
                #     from an adjacent piece's footprint, turning a clean
                #     rectangular lateral into a 5+ corner face with hole.
                # In both cases gmsh would reject the transfinite hint
                # later with "Surface N is transfinite but has K corners";
                # skipping the hint here lets gmsh fall back to its
                # default mesher without crashing the 2D pass.
                edge_tags_set = {tag for d, tag in bnd if d == 1}
                if len(edge_tags_set) < 3:
                    continue
                # Multi-wire detection: walk the face wires explicitly
                # via the gmsh face's underlying topology. A single-wire
                # face's edges form ONE traversal loop; multi-wire faces
                # produce N disjoint loops. Easier proxy: check whether
                # the boundary call with recursive=True returns more
                # vertices than the edge set suggests for a simple loop
                # (simple loop has #vertices == #edges).
                v_recursive = gmsh.model.getBoundary(
                    [(2, face_tag)], oriented=False, recursive=True
                )
                n_vertices = len({tag for d, tag in v_recursive if d == 0})
                if n_vertices != len(edge_tags_set):
                    # Wire count mismatch: face is non-simply-connected.
                    continue
                for dim_e, edge_tag in bnd:
                    if dim_e != 1:
                        continue
                    # Determine if this edge is VERTICAL: endpoints differ in z.
                    verts = gmsh.model.getBoundary(
                        [(1, edge_tag)], oriented=False, recursive=False
                    )
                    if len(verts) != 2:
                        continue
                    z_vals = []
                    for dim_v, vtag in verts:
                        if dim_v != 0:
                            continue
                        coords = gmsh.model.getValue(0, vtag, [])
                        z_vals.append(coords[2])
                    if len(z_vals) == 2 and abs(z_vals[0] - z_vals[1]) > 1e-10:
                        # Vertical edge: constrain node count to n_layers+1.
                        try:
                            gmsh.model.mesh.setTransfiniteCurve(edge_tag, n_layers + 1)
                        except Exception as exc:
                            logger.warning(
                                "setTransfiniteCurve failed on edge %d "
                                "(lateral %s): %s",
                                edge_tag,
                                lat_key,
                                exc,
                            )
                gmsh.model.mesh.setTransfiniteSurface(face_tag)
                if recombine_lat:
                    gmsh.model.mesh.setRecombine(2, face_tag)
            except Exception as exc:
                logger.warning(
                    "setTransfiniteSurface failed on face %d (lateral %s): %s",
                    face_tag,
                    lat_key,
                    exc,
                )


def _map_phantom_faces_to_gmsh(
    phantom_map: Any,  # PhantomMap
    occ_entities: list[Any],  # list[OCCLabeledEntity]
) -> dict[Any, list[int]]:  # dict[FaceKey, list[int]]
    """Map each PhantomMap.output_faces entry (OCP TopoDS_Face) to gmsh face tags.

    Implementation: gmsh assigns face tags in TopExp::MapShapes order
    on the loaded BREP compound (empirically verified — see spike
    docs/superpowers/spikes/xao_load_tag_determinism.py). We reproduce
    the XAO compound, build a face-only IndexedMapOfShape on it, then
    FindIndex(face) returns the 1-based index which equals the gmsh tag.

    IsSame() comparison handles the case where phantom faces are
    sub-shapes of user-entity solids (BOP TShape sharing).

    Raises RuntimeError if any phantom face has no IsSame() match in
    the compound (indicates a real architectural error, not a tolerance
    issue).
    """
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape

    compound = _build_xao_compound(occ_entities)
    fmap = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(compound, TopAbs_FACE, fmap)

    from meshwell.structured.phantom import _USE_DISCRETE_COHORT_MESH

    out: dict[Any, list[int]] = {}
    for face_key, occ_faces in phantom_map.output_faces.items():
        tags: list[int] = []
        for f in occ_faces:
            idx = fmap.FindIndex(f)  # 0 if not present
            if idx == 0:
                if _USE_DISCRETE_COHORT_MESH:
                    # Phase 3: interior cohort FaceKeys have no OCC backing — they're
                    # intentionally omitted from the cohort envelope solid (to avoid
                    # non-manifold geometry) and will be materialized as gmsh discrete
                    # 2D entities by _stamp_phase3_interior_interfaces.
                    continue
                raise RuntimeError(
                    f"PhantomMap face {face_key} has no IsSame() match in "
                    f"the XAO compound. This indicates the phantom shape "
                    f"was not properly included in cad_occ's BOP, OR "
                    f"BOP eliminated it."
                )
            tags.append(idx)
        out[face_key] = tags
    return out
