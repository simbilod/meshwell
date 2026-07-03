"""OCP-dependent shape helpers shared by cad_occ and occ_xao_writer.

Split out from :mod:`meshwell.cad_common` so that module stays importable
without the OCP bindings; anything here is free to depend on ``OCP``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib

if TYPE_CHECKING:
    from OCP.TopoDS import TopoDS_Shape


def shape_bbox(
    shape: TopoDS_Shape,
) -> tuple[float, float, float, float, float, float] | None:
    """Return (xmin, ymin, zmin, xmax, ymax, zmax) bounding box of shape.

    Returns ``None`` for void / empty shapes. Shared verbatim by
    ``cad_occ`` (pre-fragment AABB-overlap pruning) and
    ``occ_xao_writer`` (interface AABB pre-filtering) -- the two
    original implementations (``cad_occ._shape_bbox``,
    ``occ_xao_writer._shape_aabb``) were byte-identical.
    """
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    if box.IsVoid():
        return None
    return box.Get()
