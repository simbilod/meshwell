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
