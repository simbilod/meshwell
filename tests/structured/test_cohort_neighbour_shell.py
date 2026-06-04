"""Tests for the custom shell builder used by CohortNeighbourUnstructured."""
from __future__ import annotations

from shapely.geometry import Polygon

from meshwell.structured.build import EdgeRegistry, FaceRegistry, VertexRegistry
from meshwell.structured.cohort_neighbour_shell import build_neighbour_shell


def _rect(x1, y1, x2, y2):
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def test_build_neighbour_shell_single_tile_produces_solid():
    vreg = VertexRegistry(point_tolerance=1e-3)
    ereg = EdgeRegistry(vertices=vreg, point_tolerance=1e-3)
    freg = FaceRegistry(edges=ereg, point_tolerance=1e-3)
    poly = _rect(0, 0, 10, 10)
    # Pre-cache the touched-plane face at z=0 (cohort-side).
    top_face = freg.face_xy(
        poly,
        z=0.0,
        identify_arcs=False,
        min_arc_points=4,
        arc_tolerance=1e-3,
    )
    shape = build_neighbour_shell(
        tiles=(poly,),
        z_touched=0.0,
        z_far=-1.0,
        face_registry=freg,
        edge_registry=ereg,
        identify_arcs=False,
        min_arc_points=4,
        arc_tolerance=1e-3,
    )
    # The result must be a single solid; its top face must IsSame as the
    # cached top_face.
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    solids = []
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        solids.append(exp.Current())
        exp.Next()
    assert len(solids) == 1
    found_match = False
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        if exp.Current().IsSame(top_face):
            found_match = True
            break
        exp.Next()
    assert found_match, "the shared top face TShape is NOT in the resulting shell"


def test_build_neighbour_shell_multi_tile_preserves_all_tops():
    vreg = VertexRegistry(point_tolerance=1e-3)
    ereg = EdgeRegistry(vertices=vreg, point_tolerance=1e-3)
    freg = FaceRegistry(edges=ereg, point_tolerance=1e-3)
    tile_a = _rect(0, 0, 5, 10)
    tile_b = _rect(5, 0, 10, 10)
    top_a = freg.face_xy(
        tile_a,
        z=0.0,
        identify_arcs=False,
        min_arc_points=4,
        arc_tolerance=1e-3,
    )
    top_b = freg.face_xy(
        tile_b,
        z=0.0,
        identify_arcs=False,
        min_arc_points=4,
        arc_tolerance=1e-3,
    )
    shape = build_neighbour_shell(
        tiles=(tile_a, tile_b),
        z_touched=0.0,
        z_far=-1.0,
        face_registry=freg,
        edge_registry=ereg,
        identify_arcs=False,
        min_arc_points=4,
        arc_tolerance=1e-3,
    )
    # Both top faces must be present in the shape by TShape identity.
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer

    seen_a = seen_b = False
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        f = exp.Current()
        if f.IsSame(top_a):
            seen_a = True
        if f.IsSame(top_b):
            seen_b = True
        exp.Next()
    assert seen_a, "top_a TShape missing from shell"
    assert seen_b, "top_b TShape missing from shell"
