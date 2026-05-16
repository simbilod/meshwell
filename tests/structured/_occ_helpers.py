"""Shared OCP test fixtures for the structured-polyprism test suite.

Underscore prefix marks this as private to tests/structured/.
"""
from __future__ import annotations

from typing import Any


def make_box(x0: float, y0: float, z0: float, dx: float, dy: float, dz: float) -> Any:
    """Return a TopoDS_Solid for an axis-aligned box."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    return BRepPrimAPI_MakeBox(
        gp_Pnt(x0, y0, z0), gp_Pnt(x0 + dx, y0 + dy, z0 + dz)
    ).Solid()


def make_stick(
    x0: float, y0: float, z_lo: float, z_hi: float, dx: float, dy: float
) -> Any:
    """Convenience: a tall thin box for through-cut tests."""
    return make_box(x0, y0, z_lo, dx, dy, z_hi - z_lo)


def count_subshapes(shape: Any, sub_type: Any) -> int:
    """Count distinct sub-shapes of ``sub_type`` (TopAbs_FACE etc.) in shape."""
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape

    m = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, sub_type, m)
    return m.Size()


def _list_of_shape_to_list(lst: Any) -> list[Any]:
    """Convert an OCP TopTools_ListOfShape to a plain Python list of TopoDS_*."""
    return list(lst)
