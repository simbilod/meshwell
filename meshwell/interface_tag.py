"""InterfaceTag: name an existing interface between buffered polygon entities.

Unlike :class:`meshwell.polysurface.PolySurface` or a `gmsh_entity` plane,
:class:`InterfaceTag` does not introduce new geometry into the model. It
declares a *nominal* trace where the user expects an interface to lie, and
at fragment time it resolves itself onto the boundary of the polygon entity
that wins the cad_gmsh cut cascade in that region. This avoids the sliver
slabs that result from co-positioning a `gmsh_entity` plane with an
asymmetrically-buffered ``PolyPrism`` edge.

Use ``gmsh_entity`` (or ``PolySurface``) when you need a NEW internal cut.
Use ``InterfaceTag`` when you only want to name an interface that already
exists between two polygon entities.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import shapely
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
)

from meshwell.geometry_entity import GeometryEntity

if TYPE_CHECKING:
    from OCP.TopoDS import TopoDS_Shape


def _flatten_to_linestrings(geoms) -> list[LineString]:
    """Collect plain LineStrings from any nesting of LineString / MultiLineString / GeometryCollection.

    Empty geometries and Points (degenerate intersections) are dropped.
    """
    out: list[LineString] = []
    for g in geoms:
        if g.is_empty:
            continue
        if isinstance(g, LineString):
            out.append(g)
        elif isinstance(g, MultiLineString):
            out.extend(list(g.geoms))
        elif isinstance(g, GeometryCollection):
            out.extend(_flatten_to_linestrings(list(g.geoms)))
        # Points and Polygons are silently dropped (degenerate or N/A).
    return out


class InterfaceTag(GeometryEntity):
    """Snap-to-boundary interface tag for ``cad_gmsh``.

    Attributes:
        linestrings: list of shapely ``LineString`` giving the nominal XY
            trace of the interface.
        zmin: lower z-extent of the resulting vertical 2D surface.
        zmax: upper z-extent of the resulting vertical 2D surface.
        physical_name: name of the physical group this entity will own.
        targets: explicit list of polygon-entity physical names to snap
            to. ``None`` means "any polygon-bearing entity in the scene".
        snap_distance: how far the nominal trace is allowed to be from a
            target boundary. ``None`` means "inherit the cad processor's
            ``perturbation``".
        mesh_order: ownership priority on overlapping pieces.
        mesh_bool: whether to keep the resulting surfaces in the mesh.
    """

    def __init__(
        self,
        linestrings: LineString | list[LineString] | MultiLineString,
        zmin: float,
        zmax: float,
        physical_name: str | tuple[str, ...] | None = None,
        targets: list[str] | None = None,
        snap_distance: float | None = None,
        mesh_order: float | None = None,
        mesh_bool: bool = True,
        point_tolerance: float = 1e-3,
    ):
        super().__init__(point_tolerance=point_tolerance)

        # Normalize linestrings to a flat list[LineString].
        if isinstance(linestrings, list):
            normalized: list[LineString] = []
            for item in linestrings:
                if isinstance(item, MultiLineString):
                    normalized.extend(list(item.geoms))
                else:
                    normalized.append(item)
            self.linestrings = normalized
        elif isinstance(linestrings, MultiLineString):
            self.linestrings = list(linestrings.geoms)
        else:
            self.linestrings = [linestrings]

        # Snap input to the tolerance grid for determinism.
        if point_tolerance > 0:
            self.linestrings = [
                shapely.set_precision(ls, grid_size=point_tolerance, mode="pointwise")
                for ls in self.linestrings
            ]

        self.zmin = float(zmin)
        self.zmax = float(zmax)
        if isinstance(physical_name, str):
            self.physical_name = (physical_name,)
        else:
            self.physical_name = physical_name
        self.targets = list(targets) if targets is not None else None
        self.snap_distance = snap_distance
        self.mesh_order = mesh_order
        self.mesh_bool = mesh_bool
        self.dimension = 2

        # Populated by :meth:`resolve` before :meth:`instanciate` runs.
        self.resolved_linestrings: list[LineString] = []

    def resolve(self, polygon_ents: dict[str, Any], default_snap: float) -> None:
        """Compute the snapped trace from buffered polygon entities.

        Must be called before :meth:`instanciate`. See spec for algorithm.
        """
        raise NotImplementedError("InterfaceTag.resolve not yet implemented")

    def instanciate(
        self,
        cad_model: Any | None = None,
    ) -> list[tuple[int, int]]:
        """Build vertical 2D surfaces from ``self.resolved_linestrings``."""
        raise NotImplementedError("InterfaceTag.instanciate not yet implemented")

    def instanciate_occ(self) -> "TopoDS_Shape":
        """OCC instantiation not supported for v1.

        InterfaceTag is gmsh-only for v1; use the cad_gmsh backend.
        """
        raise NotImplementedError(
            "InterfaceTag is gmsh-only for v1; use the cad_gmsh backend."
        )
