"""OCP-dependent shape helpers shared by cad_occ and occ_xao_writer.

Split out from :mod:`meshwell.cad_common` so that module stays importable
without the OCP bindings; anything here is free to depend on ``OCP``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.TopTools import TopTools_IndexedMapOfShape

if TYPE_CHECKING:
    from OCP.TopoDS import TopoDS_Shape


class IndexedShapeRegistry:
    """Per-run, collision-free integer identity for ``TopoDS_Shape`` objects.

    Thin wrapper over ``TopTools_IndexedMapOfShape``. ``index_of`` returns a
    stable positive integer for each distinct shape (compared by the map's
    real ``IsSame`` test, which keys on the underlying ``TShape`` pointer +
    location and *ignores orientation*), assigning a new index the first time
    a shape is seen. Python dicts/sets then key on the returned ``int``.

    This replaces the previous use of ``TopTools_ShapeMapHasher`` hash values
    directly as dict/set keys. Those hash values can *collide*, silently
    merging two distinct shapes into one identity; ``IsSame`` cannot.

    Because the underlying map accumulates state, a registry MUST NOT be a
    module global -- create one per pipeline run (per ``CAD_OCC`` fragment
    pass / per ``write_xao`` call) so identities never leak across runs.
    """

    __slots__ = ("_map",)

    def __init__(self) -> None:
        self._map = TopTools_IndexedMapOfShape()

    def index_of(self, shape: TopoDS_Shape) -> int:
        """Return this shape's collision-free identity (>= 1), adding on first sight.

        Orientation-insensitive: a shape and its ``Reversed()`` share an index.
        """
        return self._map.Add(shape)

    def oriented_key(self, shape: TopoDS_Shape) -> tuple[int, int]:
        """Return ``(index, orientation_int)`` -- an orientation-sensitive identity.

        Use where reversed shapes must compare distinct (the fragment
        ownership pass in :mod:`meshwell.cad_occ`). Two handles to the same
        oriented shape share the tuple; a shape and its ``Reversed()`` share
        the index component but differ in the orientation component.
        """
        return (self._map.Add(shape), int(shape.Orientation()))


def validated_find_index(
    shape_reference_map: TopTools_IndexedMapOfShape,
    shape: TopoDS_Shape,
    group_name: str,
) -> int:
    """Return ``shape_reference_map.FindIndex(shape)``, guaranteed ``> 0``.

    ``TopTools_IndexedMapOfShape.FindIndex`` returns ``0`` for a shape that is
    not in the map. Serializing that ``0`` into an XAO topology ``reference``
    silently produces a dangling group. Raise a ``ValueError`` naming the
    offending group instead, so the failure is loud and diagnosable.
    """
    reference = shape_reference_map.FindIndex(shape)
    if reference <= 0:
        raise ValueError(
            f"Shape referenced by group {group_name!r} was not found in the "
            f"BREP shape reference map (FindIndex returned {reference}); it "
            "would serialize as a dangling reference=0."
        )
    return reference


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
