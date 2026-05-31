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

from meshwell.structured.exceptions import (
    StructuredError,
    StructuredLateralNLayersMismatchError,
    StructuredTransfiniteRejectedError,
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
