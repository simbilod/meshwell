"""Custom shell builder for CohortNeighbourUnstructured.

Constructs an OCC solid that:
  - For each tile, uses the cached cohort TopoDS_Face at the touched
    z-plane as the prism base. ``BRepPrimAPI_MakePrism`` preserves the
    input face's TShape in the resulting solid, so the cohort sub-piece's
    bot/top face survives intact in the neighbour's prism.
  - When there are multiple tiles, fuses the per-tile prisms into a
    single TopoDS_Solid via ``BRepAlgoAPI_Fuse`` (mirrors Task 7's
    multi-tile combiner). Fuse rebuilds inter-tile interfaces, but the
    outer-tile cached top faces remain TShape-identical to the cohort
    sub-piece faces because Fuse only modifies the shared (inter-tile)
    boundaries — not the per-tile top faces that lie within a single
    tile's footprint.

This file is the public entry point for ``CohortNeighbourUnstructured.
instanciate_occ``; centralising the shell-construction logic here keeps
the neighbour class small and isolates the OCC plumbing in one place.
"""
from __future__ import annotations

from shapely.geometry import Polygon


def build_neighbour_shell(
    tiles: tuple[Polygon, ...],
    z_touched: float,
    z_far: float,
    face_registry,
    edge_registry,  # noqa: ARG001 - reserved for future shell-construction
    identify_arcs: bool,
    min_arc_points: int,
    arc_tolerance: float,
):
    """Build a CohortNeighbourUnstructured's TopoDS_Solid (or compound).

    Returns a single ``TopoDS_Solid`` (single tile case or Fuse'd
    multi-tile case).

    The ``edge_registry`` is accepted but unused in the current
    implementation; it is retained on the public signature so a future
    refactor to true shell construction (per-tile bot/top faces plus
    perimeter laterals built directly via ``BRep_Builder``) can be
    swapped in without changing call sites.
    """
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.gp import gp_Vec

    height = z_far - z_touched
    vec = gp_Vec(0, 0, height)

    prisms = []
    for tile in tiles:
        top_face = face_registry.face_xy(
            tile,
            z_touched,
            identify_arcs,
            min_arc_points,
            arc_tolerance,
        )
        # MakePrism preserves the input face's TShape in the resulting
        # solid, so the cohort-cached top face survives by reference.
        prisms.append(BRepPrimAPI_MakePrism(top_face, vec).Shape())

    if len(prisms) == 1:
        return prisms[0]

    # Multi-tile: Fuse together (mirrors Task 7's combiner). The cached
    # top faces of EACH tile are preserved by Fuse where they don't touch
    # other tiles' faces; inter-tile interfaces get cleaned up. Without
    # this Fuse, the downstream ``BOPAlgo_Builder`` would see each prism
    # as a separate solid and fragment the inter-tile lateral interfaces,
    # which in turn would fragment cohort sub-piece bot/top faces that
    # span the inter-tile boundary.
    result = prisms[0]
    for v in prisms[1:]:
        fuse_api = BRepAlgoAPI_Fuse(result, v)
        fuse_api.Build()
        result = fuse_api.Shape()
    return result
