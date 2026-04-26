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

import itertools
import warnings
from typing import TYPE_CHECKING, Any

from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
)
from shapely.ops import unary_union

import gmsh
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
        physical_name: str | tuple[str, ...] | None,
        targets: list[str] | None = None,
        snap_distance: float | None = None,
        mesh_order: float | None = None,
        mesh_bool: bool = True,
        point_tolerance: float = 1e-3,
    ):
        super().__init__(point_tolerance=point_tolerance)

        if physical_name is None or physical_name == "" or physical_name == ():
            raise ValueError("InterfaceTag requires a non-empty physical_name")

        if snap_distance is not None and snap_distance <= 0:
            raise ValueError(f"snap_distance must be positive, got {snap_distance}")

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

    def resolve(
        self,
        polygon_ents: dict[str, list[Any]],
        default_snap: float,
    ) -> None:
        """Compute the snapped trace by replicating the cad_gmsh cut cascade.

        Args:
            polygon_ents: Mapping from physical name to list of polygon-bearing
                entities sharing that name. Multiple entities may share a name
                (e.g. same-material disjoint regions).
            default_snap: Fall-back snap distance when ``self.snap_distance``
                is ``None``.

        Buffers are intersected with the user's nominal strip after the
        cascade is applied in shapely.

        Targets are processed lowest ``mesh_order`` first. Each target's
        polygon has all higher-priority targets' polygons subtracted
        from it (mirroring the cad_gmsh cut step), so the resulting
        boundaries are exactly what would survive the cuts. Intersecting
        those boundaries with the nominal strip - built with flat caps,
        so it does not extend past the user's linestring endpoints -
        gives the snapped interface trace. Coincident contributions
        from neighbouring targets collapse via a final ``unary_union``.
        """
        snap = self.snap_distance if self.snap_distance is not None else default_snap

        if self.targets is not None:
            # Flatten lists for explicitly-named targets; skip names not
            # present in the scene rather than KeyError-ing.
            targets = [
                ent
                for n in self.targets
                if n in polygon_ents
                for ent in polygon_ents[n]
            ]
        else:
            targets = [ent for ents in polygon_ents.values() for ent in ents]

        # Stable sort by mesh_order; ties resolve to the entity that
        # appears earliest in `polygon_ents` iteration order (insertion
        # order in CPython). The explicit second key makes this contract
        # part of the algorithm rather than an implicit Python detail.
        targets_with_pos = list(enumerate(targets))
        targets_with_pos.sort(
            key=lambda pair: (
                pair[1].mesh_order if pair[1].mesh_order is not None else float("inf"),
                pair[0],
            )
        )
        targets = [t for _, t in targets_with_pos]

        # Flat caps: do not extend past the user's linestring endpoints,
        # so we don't pick up corner artifacts where the strip crosses
        # the prism's lateral faces near the linestring's endpoints.
        nominal_strip = unary_union(
            [ls.buffer(snap, join_style=2, cap_style="flat") for ls in self.linestrings]
        )

        # Replicate the cad_gmsh sequential cut cascade in shapely.
        cut_polys: list = []
        for tgt in targets:
            polys = tgt.polygons if isinstance(tgt.polygons, list) else [tgt.polygons]
            tgt_geom = unary_union(polys)
            for prev in cut_polys:
                tgt_geom = tgt_geom.difference(prev)
            cut_polys.append(tgt_geom)

        snapped: list = []
        for cp in cut_polys:
            if cp.is_empty:
                continue
            hit = cp.boundary.intersection(nominal_strip)
            if not hit.is_empty:
                snapped.append(hit)

        # Coincident contributions from neighbouring targets collapse here.
        if snapped:
            merged = unary_union(snapped)
            self.resolved_linestrings = _flatten_to_linestrings([merged])
        else:
            self.resolved_linestrings = []

        if not self.resolved_linestrings:
            warnings.warn(
                f"InterfaceTag {self.physical_name} resolved to no segments",
                stacklevel=2,
            )

    def instanciate(
        self,
        cad_model: Any | None = None,  # noqa: ARG002
    ) -> list[tuple[int, int]]:
        """Build vertical 2D surfaces from ``self.resolved_linestrings``.

        Each linestring segment is turned into a vertical rectangular
        panel by placing points at ``zmin`` and ``zmax``, forming a
        closed curve loop, and adding a plane surface. All panel
        dimtags are collected and returned.
        """
        dimtags: list[tuple[int, int]] = []
        dz = self.zmax - self.zmin
        if dz == 0.0:
            return dimtags

        # Per-segment plane surfaces (instead of extrude(wire) per spec):
        # wire-extrude hit a gmsh sync issue producing double-height surfaces
        # when multiple unsynchronized wires were in the model.
        for ls in self.resolved_linestrings:
            coords = list(ls.coords)
            if len(coords) < 2:
                continue

            # Build one vertical panel per consecutive pair of XY coords.
            for (x1, y1), (x2, y2) in itertools.pairwise(coords):
                p_bot1 = self._add_point_with_tolerance(x1, y1, self.zmin)
                p_bot2 = self._add_point_with_tolerance(x2, y2, self.zmin)
                p_top2 = self._add_point_with_tolerance(x2, y2, self.zmax)
                p_top1 = self._add_point_with_tolerance(x1, y1, self.zmax)

                l_bot = self._add_line_with_cache(p_bot1, p_bot2)
                l_right = self._add_line_with_cache(p_bot2, p_top2)
                l_top = self._add_line_with_cache(p_top2, p_top1)
                l_left = self._add_line_with_cache(p_top1, p_bot1)

                active_lines = [lt for lt in (l_bot, l_right, l_top, l_left) if lt != 0]
                if len(active_lines) < 3:
                    continue

                cl = gmsh.model.occ.addCurveLoop(active_lines)
                surf = gmsh.model.occ.addPlaneSurface([cl])
                dimtags.append((2, surf))

        gmsh.model.occ.synchronize()
        return dimtags

    def instanciate_occ(self) -> "TopoDS_Shape":
        """OCC instantiation not supported for v1.

        InterfaceTag is gmsh-only for v1; use the cad_gmsh backend.
        """
        raise NotImplementedError(
            "InterfaceTag is gmsh-only for v1; use the cad_gmsh backend."
        )
