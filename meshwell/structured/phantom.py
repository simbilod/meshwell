"""Phase-2: CAD-stage phantom shape construction + BOP-history-based PhantomMap.

The two public entry points (added in later tasks) are:

- :func:`build_phantom_shapes` — turn a ``StructuredPlan`` into
  per-piece OCP sub-prisms, recording input OCC tags into
  ``PhantomBuildResult``.
- :func:`extract_phantom_map` — given a post-Perform
  ``BOPAlgo_Builder`` (or any builder exposing the Modified() /
  Generated() / IsDeleted() history API), walk the recorded input
  tags to produce the ``PhantomMap``.

Phase 2 does not integrate with ``cad_occ`` (that's Phase 3). All
tests here use OCP directly with handcrafted scenes.
"""
from __future__ import annotations

from typing import Any

from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from meshwell.structured.logging import phase_timed
from meshwell.structured.spec import (
    EdgeKey,
    FaceKey,
    LateralKey,
    PhantomBuildResult,
    PhantomMap,
    PhantomShape,
    StructuredPlan,
    VertexKey,
)


def _make_face_from_polygon(polygon: Polygon, z: float) -> Any:
    """Build a planar TopoDS_Face at the given z from a shapely Polygon.

    Handles interior holes (rings) by adding each as a reversed wire to
    the face builder. Forces CCW exterior + CW interior orientation to
    match OCC convention.
    """
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakeWire,
    )
    from OCP.gp import gp_Pnt

    poly = orient(polygon, sign=1.0)

    def _wire_from_coords(coords: list[tuple[float, float]]) -> Any:
        if coords[0] == coords[-1]:
            coords = coords[:-1]
        wire_builder = BRepBuilderAPI_MakeWire()
        for i in range(len(coords)):
            p1 = gp_Pnt(coords[i][0], coords[i][1], z)
            p2 = gp_Pnt(
                coords[(i + 1) % len(coords)][0], coords[(i + 1) % len(coords)][1], z
            )
            edge = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
            wire_builder.Add(edge)
        return wire_builder.Wire()

    outer_wire = _wire_from_coords(list(poly.exterior.coords))
    face_builder = BRepBuilderAPI_MakeFace(outer_wire)
    for ring in poly.interiors:
        inner_wire = _wire_from_coords(list(ring.coords))
        face_builder.Add(inner_wire)
    return face_builder.Face()


def _build_sub_prism(
    piece: Polygon,
    zlo: float,
    zhi: float,
    slab_index: int = 0,
    piece_index: int = 0,
) -> PhantomShape:
    """Build one OCP sub-prism for a single partition piece.

    Returns a :class:`PhantomShape` carrying:

    - The TopoDS_Solid produced by extruding the piece face from zlo to zhi.
    - The input OCC tags for bottom face, top face, outer-edge edges,
      outer-edge vertices, and lateral faces — keyed by our Phase-2 key
      types so the post-BOP map can index them.

    Inner-ring edges/vertices are NOT keyed (Layer A's outer-only
    contract: lateral OCC faces are 4-corner on the outer boundary; hole
    boundaries are not in the structured pipeline's correspondence map).
    """
    from OCP.BRep import BRep_Tool
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.gp import gp_Vec
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer

    height = zhi - zlo
    poly = orient(piece, sign=1.0)
    bottom_face = _make_face_from_polygon(poly, z=zlo)
    prism_builder = BRepPrimAPI_MakePrism(bottom_face, gp_Vec(0.0, 0.0, height))
    solid = prism_builder.Shape()
    top_face = prism_builder.LastShape()

    input_faces: dict[FaceKey, Any] = {
        FaceKey(slab_index, "bot", piece_index): bottom_face,
        FaceKey(slab_index, "top", piece_index): top_face,
    }

    outer_coords = list(poly.exterior.coords)
    if outer_coords[0] == outer_coords[-1]:
        outer_coords = outer_coords[:-1]
    n_outer = len(outer_coords)

    # Collect all bottom-face edges, then match each outer segment by
    # vertex coordinates (robust against any traversal-order variance).
    bot_edges_all = []
    edge_explorer = TopExp_Explorer(bottom_face, TopAbs_EDGE)
    while edge_explorer.More():
        bot_edges_all.append(edge_explorer.Current())
        edge_explorer.Next()

    def _edge_endpoints(shape: Any) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return (x, y) of the two endpoints of a TopoDS_Edge (or Shape) at zlo."""
        from OCP.BRep import BRep_Tool
        from OCP.TopExp import TopExp
        from OCP.TopoDS import TopoDS, TopoDS_Vertex

        edge = TopoDS.Edge_s(shape)
        v1 = TopoDS_Vertex()
        v2 = TopoDS_Vertex()
        TopExp.Vertices_s(edge, v1, v2)
        p1 = BRep_Tool.Pnt_s(v1)
        p2 = BRep_Tool.Pnt_s(v2)
        return (p1.X(), p1.Y()), (p2.X(), p2.Y())

    def _coords_close(
        a: tuple[float, float], b: tuple[float, float], tol: float = 1e-9
    ) -> bool:
        return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol

    def _find_outer_edge(
        edges: list[Any],
        c1: tuple[float, float],
        c2: tuple[float, float],
    ) -> Any | None:
        for e in edges:
            ep1, ep2 = _edge_endpoints(e)
            if (_coords_close(ep1, c1) and _coords_close(ep2, c2)) or (
                _coords_close(ep1, c2) and _coords_close(ep2, c1)
            ):
                return e
        return None

    input_edges: dict[EdgeKey, Any] = {}
    input_laterals: dict[int, Any] = {}

    # Collect top-face edges for top key recording.
    top_edges_all = []
    edge_explorer = TopExp_Explorer(top_face, TopAbs_EDGE)
    while edge_explorer.More():
        top_edges_all.append(edge_explorer.Current())
        edge_explorer.Next()

    for edge_i in range(n_outer):
        c1 = outer_coords[edge_i]
        c2 = outer_coords[(edge_i + 1) % n_outer]

        bot_edge = _find_outer_edge(bot_edges_all, c1, c2)
        if bot_edge is None:
            raise RuntimeError(
                f"_find_outer_edge: no outer edge match for polygon edge {edge_i} "
                f"of piece. This indicates an oriented-polygon mismatch between "
                f"shapely and OCC (run shapely.geometry.polygon.orient at the "
                f"piece level before building OCC faces)."
            )
        input_edges[EdgeKey(slab_index, "bot", piece_index, edge_i)] = bot_edge
        lateral_face = prism_builder.Generated(bot_edge).First()
        input_laterals[edge_i] = lateral_face

        # Top edges have z=zhi; match by same (x,y) coordinate pair.
        top_edge = _find_outer_edge(top_edges_all, c1, c2)
        if top_edge is None:
            raise RuntimeError(
                f"_find_outer_edge: no outer edge match for polygon edge {edge_i} "
                f"of piece. This indicates an oriented-polygon mismatch between "
                f"shapely and OCC (run shapely.geometry.polygon.orient at the "
                f"piece level before building OCC faces)."
            )
        input_edges[EdgeKey(slab_index, "top", piece_index, edge_i)] = top_edge

    from OCP.TopoDS import TopoDS

    # Vertices: walk bottom face vertices, dedupe via IsSame, take outer ones
    # by coordinate matching against outer_coords.
    bot_verts_all = []
    vert_explorer = TopExp_Explorer(bottom_face, TopAbs_VERTEX)
    while vert_explorer.More():
        bot_verts_all.append(TopoDS.Vertex_s(vert_explorer.Current()))
        vert_explorer.Next()

    seen: list[Any] = []
    for v in bot_verts_all:
        if not any(v.IsSame(s) for s in seen):
            seen.append(v)

    input_vertices: dict[VertexKey, Any] = {}
    for corner_i, (cx, cy) in enumerate(outer_coords):
        for v in seen:
            p = BRep_Tool.Pnt_s(v)
            if abs(p.X() - cx) < 1e-9 and abs(p.Y() - cy) < 1e-9:
                input_vertices[VertexKey(slab_index, "bot", piece_index, corner_i)] = v
                break

    top_verts_all = []
    vert_explorer = TopExp_Explorer(top_face, TopAbs_VERTEX)
    while vert_explorer.More():
        top_verts_all.append(TopoDS.Vertex_s(vert_explorer.Current()))
        vert_explorer.Next()

    seen = []
    for v in top_verts_all:
        if not any(v.IsSame(s) for s in seen):
            seen.append(v)

    for corner_i, (cx, cy) in enumerate(outer_coords):
        for v in seen:
            p = BRep_Tool.Pnt_s(v)
            if abs(p.X() - cx) < 1e-9 and abs(p.Y() - cy) < 1e-9:
                input_vertices[VertexKey(slab_index, "top", piece_index, corner_i)] = v
                break

    return PhantomShape(
        slab_index=slab_index,
        piece_index=piece_index,
        solid=solid,
        input_faces_by_key=input_faces,
        input_edges_by_key=input_edges,
        input_vertices_by_key=input_vertices,
        input_laterals_by_outer_edge=input_laterals,
    )


@phase_timed("phantom_build")
def build_phantom_shapes(plan: StructuredPlan) -> PhantomBuildResult:
    """For each slab, build one OCP sub-prism per partition piece.

    Returns a :class:`PhantomBuildResult` with shapes in
    (slab_index, piece_index) ascending order for deterministic
    downstream processing.
    """
    shapes: list[PhantomShape] = []
    for slab_index, slab in enumerate(plan.slabs):
        if not slab.face_partition:
            continue
        for piece_index, piece in enumerate(slab.face_partition):
            shapes.append(
                _build_sub_prism(
                    piece=piece,
                    zlo=slab.zlo,
                    zhi=slab.zhi,
                    slab_index=slab_index,
                    piece_index=piece_index,
                )
            )
    return PhantomBuildResult(shapes=tuple(shapes))


_MIDHEIGHT_TOL = 1e-7


def _slab_z_range_for_shape(shape: PhantomShape) -> tuple[float, float]:
    """Recover (zlo, zhi) for a shape from its input bottom/top faces.

    PhantomShape doesn't store zlo/zhi directly; read it from any
    vertex on the bottom face (= zlo) and any vertex on the top face
    (= zhi).
    """
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    bot_key = FaceKey(
        slab_index=shape.slab_index, side="bot", piece_index=shape.piece_index
    )
    top_key = FaceKey(
        slab_index=shape.slab_index, side="top", piece_index=shape.piece_index
    )
    bot_face = shape.input_faces_by_key[bot_key]
    top_face = shape.input_faces_by_key[top_key]

    def _any_z(face: Any) -> float:
        exp = TopExp_Explorer(face, TopAbs_VERTEX)
        v = TopoDS.Vertex_s(exp.Current())
        return BRep_Tool.Pnt_s(v).Z()

    return _any_z(bot_face), _any_z(top_face)


def _has_midheight_vertex(
    builder: Any, lateral_face: Any, zlo: float, zhi: float
) -> bool:
    """True if BOP.Generated(lateral_face) produced any vertex with zlo < z < zhi."""
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_VERTEX
    from OCP.TopoDS import TopoDS

    generated = builder.Generated(lateral_face)
    if generated.IsEmpty():
        return False
    for sub in list(generated):
        if sub.ShapeType() == TopAbs_VERTEX:
            v = TopoDS.Vertex_s(sub)
            z = BRep_Tool.Pnt_s(v).Z()
            if zlo + _MIDHEIGHT_TOL < z < zhi - _MIDHEIGHT_TOL:
                return True
    return False


def _modified_or_unchanged(builder: Any, input_shape: Any) -> list[Any]:
    """Return list of output shapes for input_shape.

    Mirrors the cad_occ.py pattern: if Modified() is empty AND the shape
    is not deleted, the shape passed through unchanged (input == output,
    one element). Otherwise Modified() gives the actual successor list.
    """
    modified = builder.Modified(input_shape)
    if modified.IsEmpty():
        if builder.IsDeleted(input_shape):
            return []
        return [input_shape]
    return list(modified)


@phase_timed("phantom_map")
def extract_phantom_map(
    build_result: PhantomBuildResult,
    builder: Any,
) -> PhantomMap:
    """Walk the post-Perform BOP history to build the PhantomMap.

    For every input OCC tag recorded in ``build_result``, ask the
    ``builder`` (a ``BOPAlgo_Builder`` or any object exposing
    ``Modified(shape)`` / ``IsDeleted(shape)``) what the input became
    in the output.

    Args:
        build_result: From :func:`build_phantom_shapes`.
        builder: Post-Perform BOP builder.

    Returns:
        :class:`PhantomMap` with all four output_*_by_key dicts
        populated. Each value is a list because a single input can
        split into many outputs.
    """
    pmap = PhantomMap()
    for shape in build_result.shapes:
        for face_key, in_face in shape.input_faces_by_key.items():
            pmap.output_faces[face_key] = _modified_or_unchanged(builder, in_face)
        for edge_key, in_edge in shape.input_edges_by_key.items():
            pmap.output_edges[edge_key] = _modified_or_unchanged(builder, in_edge)
        for vert_key, in_vert in shape.input_vertices_by_key.items():
            pmap.output_vertices[vert_key] = _modified_or_unchanged(builder, in_vert)
        for outer_edge_idx, in_lateral in shape.input_laterals_by_outer_edge.items():
            lateral_key = LateralKey(
                slab_index=shape.slab_index,
                piece_index=shape.piece_index,
                outer_edge_index=outer_edge_idx,
            )
            pmap.output_laterals[lateral_key] = _modified_or_unchanged(
                builder, in_lateral
            )
        slab_zlo, slab_zhi = _slab_z_range_for_shape(shape)
        for outer_edge_idx, in_lateral in shape.input_laterals_by_outer_edge.items():
            lateral_key = LateralKey(
                slab_index=shape.slab_index,
                piece_index=shape.piece_index,
                outer_edge_index=outer_edge_idx,
            )
            pmap.lateral_has_midheight_cut[lateral_key] = _has_midheight_vertex(
                builder, in_lateral, slab_zlo, slab_zhi
            )
    return pmap
