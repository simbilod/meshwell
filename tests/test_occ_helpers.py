"""Shared OCP shape-builder callables used by tests migrated off GMSH_entity.

Each helper returns a zero-arg callable that produces a ``TopoDS_Shape``
when called - the exact contract ``OCC_entity(occ_function=...)`` expects.
"""

from __future__ import annotations

from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeVertex,
)
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere
from OCP.gp import gp_Pnt


def _occ_box(x: float, y: float, z: float, dx: float, dy: float, dz: float):
    def build():
        return BRepPrimAPI_MakeBox(gp_Pnt(x, y, z), dx, dy, dz).Shape()

    return build


def _occ_sphere(xc: float, yc: float, zc: float, radius: float):
    def build():
        return BRepPrimAPI_MakeSphere(gp_Pnt(xc, yc, zc), radius).Shape()

    return build


def _occ_rectangle(x: float, y: float, z: float, dx: float, dy: float):
    """Axis-aligned planar rectangle in the z=const plane."""

    def build():
        poly = BRepBuilderAPI_MakePolygon()
        poly.Add(gp_Pnt(x, y, z))
        poly.Add(gp_Pnt(x + dx, y, z))
        poly.Add(gp_Pnt(x + dx, y + dy, z))
        poly.Add(gp_Pnt(x, y + dy, z))
        poly.Close()
        return BRepBuilderAPI_MakeFace(poly.Wire()).Shape()

    return build


def _occ_point(x: float, y: float, z: float):
    def build():
        return BRepBuilderAPI_MakeVertex(gp_Pnt(x, y, z)).Shape()

    return build


def _occ_line(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float):
    """Straight edge between two points."""

    def build():
        return BRepBuilderAPI_MakeEdge(gp_Pnt(x1, y1, z1), gp_Pnt(x2, y2, z2)).Edge()

    return build


def _occ_circle(x: float, y: float, z: float, r: float):
    """Circular edge centered at (x,y,z) with axis +Z."""
    from OCP.GC import GC_MakeCircle
    from OCP.gp import gp_Ax2, gp_Dir

    def build():
        ax = gp_Ax2(gp_Pnt(x, y, z), gp_Dir(0, 0, 1))
        curve = GC_MakeCircle(ax, r).Value()
        return BRepBuilderAPI_MakeEdge(curve).Edge()

    return build


def test_occ_helpers_produce_valid_shapes():
    """Sanity check that each helper returns a non-null TopoDS_Shape."""
    for builder in (
        _occ_box(0, 0, 0, 1, 1, 1),
        _occ_sphere(0, 0, 0, 1),
        _occ_rectangle(0, 0, 0, 1, 1),
        _occ_point(0, 0, 0),
        _occ_line(0, 0, 0, 1, 0, 0),
        _occ_circle(0, 0, 0, 0.5),
    ):
        s = builder()
        assert not s.IsNull()
