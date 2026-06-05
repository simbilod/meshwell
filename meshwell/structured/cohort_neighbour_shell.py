"""Custom shell builder for CohortNeighbourUnstructured.

Two construction paths:

1. **Direct shell assembly** (preferred). When the union of tile polygons
   has a STRAIGHT-EDGE outer boundary (no arcs on the perimeter), assemble
   a ``TopoDS_Solid`` directly via ``BRep_Builder``:
     - Top faces: cached cohort ``TopoDS_Face`` per tile (TShape-shared
       with the cohort sub-piece).
     - Bot face: single planar face built fresh at ``z_far`` over the
       union of tile footprints.
     - Lateral faces: one per outer-edge segment, threaded through
       ``EdgeRegistry`` so vertical edges share TShapes with adjacent
       neighbours.
   No ``BRepPrimAPI_MakePrism``, no ``BRepAlgoAPI_Fuse`` — the shared
   ``TopoDS_Face`` TShapes survive intact through BOP. Eliminates
   cohort↔neighbour AABB rescues for scenes with rectangular outer
   boundaries (e.g., the meander stress scene: 6 → 0 rescues).

2. **MakePrism + Fuse fallback**. When the outer boundary contains arc
   edges (e.g., disc-cohort scenes), direct shell assembly produces
   cylindrical lateral faces whose surface parameterisations don't align
   precisely with the planar bot face's, causing gmsh PLC errors
   ("a segment and a facet intersect at point") at fine mesh resolutions
   (cl ≤ ~0.4 on a unit disc). Until that interaction can be resolved
   (likely needs gmsh-side discretisation control or sharing the
   cylindrical surface TShape across cohort/neighbour laterals), fall
   back to the per-tile ``BRepPrimAPI_MakePrism`` + multi-tile
   ``BRepAlgoAPI_Fuse`` path. This preserves the cohort top-face TShape
   in the single-tile case (no Fuse runs) and matches Task 7's behaviour
   in the multi-tile case.

The branch is selected automatically by inspecting the union's outer
boundary for arc edges before construction.
"""
from __future__ import annotations

from shapely.geometry import Polygon
from shapely.ops import unary_union


def build_neighbour_shell(
    tiles: tuple[Polygon, ...],
    z_touched: float,
    z_far: float,
    face_registry,
    edge_registry,
    identify_arcs: bool,
    min_arc_points: int,
    arc_tolerance: float,
):
    """Build a CohortNeighbourUnstructured's TopoDS_Solid.

    Selects between direct shell assembly (TShape-preserving) and a
    per-tile MakePrism + Fuse fallback based on whether the union of
    tile polygons has arc edges on its outer boundary.
    """
    union_poly = unary_union(list(tiles))
    # Direct shell assembly only succeeds when the cohort top faces have
    # no arc edges on ANY of their boundaries (exterior OR interior).
    # Arc edges shared between cohort cylindrical laterals and planar
    # top/bot faces of the neighbour cause gmsh PLC errors at fine mesh
    # resolutions ("a segment and a facet intersect at point"). The
    # planar bot face's mesh discretisation doesn't align with the
    # cohort's cylindrical surface at the shared arc edge.
    if identify_arcs and (
        _outer_has_arcs(
            union_poly, edge_registry, z_touched, min_arc_points, arc_tolerance
        )
        or _any_tile_has_arcs(
            tiles, edge_registry, z_touched, min_arc_points, arc_tolerance
        )
    ):
        return _build_via_prism_fuse(
            tiles,
            z_touched,
            z_far,
            face_registry,
            identify_arcs,
            min_arc_points,
            arc_tolerance,
        )
    return _build_via_direct_shell(
        tiles,
        union_poly,
        z_touched,
        z_far,
        face_registry,
        edge_registry,
        identify_arcs,
        min_arc_points,
        arc_tolerance,
    )


def _outer_has_arcs(
    union_poly: Polygon,
    edge_registry,
    z: float,
    min_arc_points: int,
    arc_tolerance: float,
) -> bool:
    """Return True iff polyline_xy on the outer boundary returns any arc edge.

    Uses ``_is_arc_edge`` from build.py to check the geometry of each
    edge produced by ``EdgeRegistry.polyline_xy``.
    """
    from meshwell.structured.build import _is_arc_edge, _ring_coords

    outer_coords = _ring_coords(union_poly.exterior)
    edges = edge_registry.polyline_xy(
        outer_coords,
        z,
        True,
        min_arc_points,
        arc_tolerance,
    )
    return any(_is_arc_edge(e) for e in edges)


def _any_tile_has_arcs(
    tiles: tuple[Polygon, ...],
    edge_registry,
    z: float,
    min_arc_points: int,
    arc_tolerance: float,
) -> bool:
    """Return True iff any tile polygon's boundary produces an arc edge.

    Walks each tile's exterior and interior rings, runs them through
    ``EdgeRegistry.polyline_xy``, and uses ``_is_arc_edge`` to detect
    any arc edges.
    """
    from meshwell.structured.build import _is_arc_edge, _ring_coords

    for tile in tiles:
        rings = [tile.exterior, *tile.interiors]
        for ring in rings:
            coords = _ring_coords(ring)
            edges = edge_registry.polyline_xy(
                coords,
                z,
                True,
                min_arc_points,
                arc_tolerance,
            )
            if any(_is_arc_edge(e) for e in edges):
                return True
    return False


def _build_via_direct_shell(
    tiles: tuple[Polygon, ...],
    union_poly: Polygon,
    z_touched: float,
    z_far: float,
    face_registry,
    edge_registry,
    identify_arcs: bool,
    min_arc_points: int,
    arc_tolerance: float,
):
    """Direct shell assembly. Used when the outer boundary is straight-only."""
    from OCP.BRep import BRep_Builder, BRep_Tool
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS_Shell

    from meshwell.structured.build import (
        _build_horizontal_face,
        _build_lateral_face,
        _ring_coords,
    )

    builder = BRep_Builder()
    shell = TopoDS_Shell()
    builder.MakeShell(shell)

    # 1. Top faces: cached cohort TopoDS_Face per tile. Reverse the
    #    orientation when the neighbour extends BELOW z_touched so the
    #    face's outward normal points away from the neighbour's interior.
    for tile in tiles:
        top_face = face_registry.face_xy(
            tile,
            z_touched,
            identify_arcs,
            min_arc_points,
            arc_tolerance,
        )
        if z_far < z_touched:
            builder.Add(shell, top_face.Reversed())
        else:
            builder.Add(shell, top_face)

    # 2. Bot face: planar face at z_far over the union of tile footprints.
    bot_face = _build_horizontal_face(
        union_poly,
        z_far,
        edge_registry,
        identify_arcs,
        min_arc_points,
        arc_tolerance,
        face_registry=None,
    )
    if z_far < z_touched:
        builder.Add(shell, bot_face)
    else:
        builder.Add(shell, bot_face.Reversed())

    # 3. Outer lateral faces: one per outer-edge segment. Derive each
    #    lateral's left/right verticals from the actual edge endpoints
    #    (not from the polygon vertex array — polyline_xy can collapse
    #    runs of polygon vertices into single arc edges).
    outer_coords = _ring_coords(union_poly.exterior)
    top_edges = edge_registry.polyline_xy(
        outer_coords,
        z_touched,
        identify_arcs,
        min_arc_points,
        arc_tolerance,
    )
    bot_edges = edge_registry.polyline_xy(
        outer_coords,
        z_far,
        identify_arcs,
        min_arc_points,
        arc_tolerance,
    )
    for top_edge, bot_edge in zip(top_edges, bot_edges):
        p_left = BRep_Tool.Pnt_s(TopExp.FirstVertex_s(top_edge))
        p_right = BRep_Tool.Pnt_s(TopExp.LastVertex_s(top_edge))
        v_left = edge_registry.vertical(p_left.X(), p_left.Y(), z_touched, z_far)
        v_right = edge_registry.vertical(p_right.X(), p_right.Y(), z_touched, z_far)
        lateral = _build_lateral_face(bot_edge, top_edge, v_left, v_right)
        builder.Add(shell, lateral)

    # 4. Wrap shell into a solid.
    ms = BRepBuilderAPI_MakeSolid(shell)
    if not ms.IsDone():
        raise RuntimeError(
            "BRepBuilderAPI_MakeSolid failed to assemble neighbour shell"
        )
    return ms.Solid()


def _build_via_prism_fuse(
    tiles: tuple[Polygon, ...],
    z_touched: float,
    z_far: float,
    face_registry,
    identify_arcs: bool,
    min_arc_points: int,
    arc_tolerance: float,
):
    """Per-tile MakePrism + multi-tile Fuse. Used when the outer has arcs."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.gp import gp_Vec

    vec = gp_Vec(0, 0, z_far - z_touched)
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

    result = prisms[0]
    for v in prisms[1:]:
        fuse_api = BRepAlgoAPI_Fuse(result, v)
        fuse_api.Build()
        result = fuse_api.Shape()
    return result
