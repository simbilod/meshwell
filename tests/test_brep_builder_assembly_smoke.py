"""Validation gate for Phase 2 cohort topology builder design.

Validates the two OCC behaviors the design depends on:
1. BRep_Builder.MakeSolid can assemble a closed, topologically VALID solid
   from explicit faces — provided the faces share vertex and edge TShapes
   across their boundaries (which is exactly what the cohort topology
   registries provide).
2. BOPAlgo_Builder.Modified() returns per-argument history when those
   manually-assembled solids are passed as individual arguments and one
   is overlapped by a third solid.

The helper deliberately mirrors the cohort topology builder's approach:
build vertices once, edges once, faces from shared edges — so a passing
test here directly evidences that the design will produce valid solids.
"""

from __future__ import annotations

from OCP.BOPAlgo import BOPAlgo_Builder
from OCP.BRep import BRep_Builder
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeVertex,
    BRepBuilderAPI_MakeWire,
)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Shell, TopoDS_Solid


def _faces(shape):
    out = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        out.append(exp.Current())
        exp.Next()
    return out


def _rev_edge(edge):
    """edge.Reversed() returns TopoDS_Shape; cast back to TopoDS_Edge."""
    return TopoDS.Edge_s(edge.Reversed())


def _wire(*edges):
    mw = BRepBuilderAPI_MakeWire()
    for e in edges:
        mw.Add(e)
    return mw.Wire()


def _face(wire):
    return BRepBuilderAPI_MakeFace(wire).Face()


def _make_box_solid_shared_topology(
    x0,
    y0,
    z0,
    x1,
    y1,
    z1,
    shared_bottom_vertices=None,
    shared_bottom_edges=None,
    shared_bottom_face=None,
):
    """Assemble a box solid with internally-shared vertex/edge TShapes.

    When shared_bottom_* are provided (the vertical-stacking pattern),
    this box's BOTTOM reuses the supplied vertices/edges/face — so the
    box sits on top of a previously-built box that gave us these.

    Returns (solid, top_vertices, top_edges, top_face). The returned
    top_* can be threaded into another call's `shared_bottom_*` arguments
    to make the next solid sit on top of this one.
    """
    b = BRep_Builder()

    # 4 bottom vertices (shared or fresh).
    if shared_bottom_vertices is not None:
        v_bb00, v_bb10, v_bb11, v_bb01 = shared_bottom_vertices
    else:
        v_bb00 = BRepBuilderAPI_MakeVertex(gp_Pnt(x0, y0, z0)).Vertex()
        v_bb10 = BRepBuilderAPI_MakeVertex(gp_Pnt(x1, y0, z0)).Vertex()
        v_bb11 = BRepBuilderAPI_MakeVertex(gp_Pnt(x1, y1, z0)).Vertex()
        v_bb01 = BRepBuilderAPI_MakeVertex(gp_Pnt(x0, y1, z0)).Vertex()
    v_bb = {
        (x0, y0): v_bb00,
        (x1, y0): v_bb10,
        (x1, y1): v_bb11,
        (x0, y1): v_bb01,
    }

    # 4 top vertices (always fresh; this box's top isn't shared with
    # anything yet).
    v_tt00 = BRepBuilderAPI_MakeVertex(gp_Pnt(x0, y0, z1)).Vertex()
    v_tt10 = BRepBuilderAPI_MakeVertex(gp_Pnt(x1, y0, z1)).Vertex()
    v_tt11 = BRepBuilderAPI_MakeVertex(gp_Pnt(x1, y1, z1)).Vertex()
    v_tt01 = BRepBuilderAPI_MakeVertex(gp_Pnt(x0, y1, z1)).Vertex()
    v_tt = {
        (x0, y0): v_tt00,
        (x1, y0): v_tt10,
        (x1, y1): v_tt11,
        (x0, y1): v_tt01,
    }

    # 4 bottom edges (shared or fresh).
    if shared_bottom_edges is not None:
        be0, be1, be2, be3 = shared_bottom_edges
    else:
        be0 = BRepBuilderAPI_MakeEdge(v_bb[(x0, y0)], v_bb[(x1, y0)]).Edge()
        be1 = BRepBuilderAPI_MakeEdge(v_bb[(x1, y0)], v_bb[(x1, y1)]).Edge()
        be2 = BRepBuilderAPI_MakeEdge(v_bb[(x1, y1)], v_bb[(x0, y1)]).Edge()
        be3 = BRepBuilderAPI_MakeEdge(v_bb[(x0, y1)], v_bb[(x0, y0)]).Edge()

    # 4 top edges (always fresh).
    te0 = BRepBuilderAPI_MakeEdge(v_tt[(x0, y0)], v_tt[(x1, y0)]).Edge()
    te1 = BRepBuilderAPI_MakeEdge(v_tt[(x1, y0)], v_tt[(x1, y1)]).Edge()
    te2 = BRepBuilderAPI_MakeEdge(v_tt[(x1, y1)], v_tt[(x0, y1)]).Edge()
    te3 = BRepBuilderAPI_MakeEdge(v_tt[(x0, y1)], v_tt[(x0, y0)]).Edge()
    top_edges = (te0, te1, te2, te3)

    # 4 vertical edges, each connecting a bottom corner to its top counterpart.
    ve_00 = BRepBuilderAPI_MakeEdge(v_bb[(x0, y0)], v_tt[(x0, y0)]).Edge()
    ve_10 = BRepBuilderAPI_MakeEdge(v_bb[(x1, y0)], v_tt[(x1, y0)]).Edge()
    ve_11 = BRepBuilderAPI_MakeEdge(v_bb[(x1, y1)], v_tt[(x1, y1)]).Edge()
    ve_01 = BRepBuilderAPI_MakeEdge(v_bb[(x0, y1)], v_tt[(x0, y1)]).Edge()

    # Faces — bottom and top use only horizontal edges.
    if shared_bottom_face is not None:
        bot_face = shared_bottom_face
    else:
        bot_face = _face(_wire(be0, be1, be2, be3))
    top_face = _face(_wire(te0, te1, te2, te3))

    # Lateral faces — each uses 2 horizontal edges (one bottom, one top
    # reversed) + 2 vertical edges. Edges are shared across adjacent
    # laterals via ve_*.
    lat0 = _face(_wire(be0, ve_10, _rev_edge(te0), _rev_edge(ve_00)))
    lat1 = _face(_wire(be1, ve_11, _rev_edge(te1), _rev_edge(ve_10)))
    lat2 = _face(_wire(be2, ve_01, _rev_edge(te2), _rev_edge(ve_11)))
    lat3 = _face(_wire(be3, ve_00, _rev_edge(te3), _rev_edge(ve_01)))

    # Assemble shell with all 6 faces; bottom face is reversed so its
    # normal points outward (down).
    shell = TopoDS_Shell()
    b.MakeShell(shell)
    b.Add(shell, bot_face.Reversed())
    b.Add(shell, top_face)
    for lat in (lat0, lat1, lat2, lat3):
        b.Add(shell, lat)

    solid = TopoDS_Solid()
    b.MakeSolid(solid)
    b.Add(solid, shell)

    return solid, (v_tt00, v_tt10, v_tt11, v_tt01), top_edges, top_face


def test_manually_assembled_solid_is_valid():
    """BRep_Builder solid with shared topology passes BRepCheck."""
    solid, _, _, _ = _make_box_solid_shared_topology(0, 0, 0, 1, 1, 1)
    analyzer = BRepCheck_Analyzer(solid)
    assert analyzer.IsValid(), "Manually-assembled solid is not valid per BRepCheck"


def test_two_manually_assembled_solids_share_face_tshape():
    """B sits on top of A: B's BOTTOM is A's TOP. Shared TShape identity."""
    A, A_top_vertices, A_top_edges, A_top_face = _make_box_solid_shared_topology(
        0, 0, 0, 1, 1, 1
    )
    B, _, _, _ = _make_box_solid_shared_topology(
        0,
        0,
        1,
        1,
        1,
        2,
        shared_bottom_vertices=A_top_vertices,
        shared_bottom_edges=A_top_edges,
        shared_bottom_face=A_top_face,
    )

    a_face_hashes = {hash(f) for f in _faces(A)}
    b_face_hashes = {hash(f) for f in _faces(B)}
    shared = a_face_hashes & b_face_hashes
    assert shared, (
        "Vertically-stacked manually-assembled solids did NOT share the "
        "interface TopoDS_Face TShape."
    )


def test_bopalgo_modified_works_for_manually_assembled_shared_face_solids():
    """The critical design-validation test.

    Two manually-assembled solids (A and B). A third solid C overlaps A.
    Add A, B, C as separate BOP arguments. Modified(A) must produce
    non-empty per-argument history; B must not be deleted by BOP.
    """
    A, _, _, _ = _make_box_solid_shared_topology(0, 0, 0, 1, 1, 1)
    B, _, _, _ = _make_box_solid_shared_topology(0, 0, 1, 1, 1, 2)
    C = BRepPrimAPI_MakeBox(gp_Pnt(0.5, 0.5, 0.5), gp_Pnt(1.5, 1.5, 1.5)).Solid()

    builder = BOPAlgo_Builder()
    builder.AddArgument(A)
    builder.AddArgument(B)
    builder.AddArgument(C)
    builder.Perform()

    a_mod = builder.Modified(A)
    assert not a_mod.IsEmpty(), (
        "Modified(A) empty — BRep_Builder-assembled solid is not tracked "
        "by BOPAlgo per-argument. Design must change."
    )
    assert not builder.IsDeleted(
        B
    ), "BOPAlgo deleted B (a manually-assembled solid not overlapping C)."
