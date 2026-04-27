"""Structured (gmsh-tutorial-t3-style) layered extrusion for ``PolyPrism``.

This module hosts everything that is structured-specific:

* :class:`_StructuredPolyPrism` -- private subclass of ``PolyPrism`` that
  the ``PolyPrism.__new__`` dispatcher returns when the user passes
  ``n_layers=``. Same user-visible class identity (``isinstance(p, PolyPrism)``
  remains true), but with the structured-mode validation rules.
* :class:`Slab` -- one pairwise-disjoint piece of a structured prism after
  the cascade.

Future tasks will add ``expand_slabs_for_entity``, ``resolve_structured_slabs``,
``_StructuredPhantom``, and ``apply_structured_slabs``. They all live in this
module to keep the structured pipeline co-located.

Why a private subclass rather than a flag on ``PolyPrism``: structured-mode
behavior diverges in three places (validation, ``mesh_bool`` semantics,
``instanciate``-time replacement by ``_StructuredPhantom``); a subclass
keeps each branch in one place and lets the rest of meshwell dispatch via
``isinstance``.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

from shapely.geometry import MultiPolygon, Polygon

from meshwell.polyprism import PolyPrism


@dataclass
class Slab:
    """One pairwise-disjoint piece of a structured prism after resolution.

    Attributes:
        footprint: 2D shapely Polygon or MultiPolygon at z = zlo.
        zlo: Bottom z of this slab.
        zhi: Top z of this slab. Strictly greater than ``zlo``.
        n_layers: Number of element layers in [zlo, zhi].
        recombine: Whether to recombine the swept mesh into hex elements.
        physical_name: Tuple of physical-group names carried from the
            originating ``PolyPrism(..., n_layers=...)``.
        source_index: Index of the originating entity in the user-supplied
            list. Deterministic tie-break for the cascade.
        mesh_order: Ownership priority (lower = higher priority).
    """

    footprint: Polygon | MultiPolygon
    zlo: float
    zhi: float
    n_layers: int
    recombine: bool
    physical_name: tuple[str, ...]
    source_index: int = 0
    mesh_order: float = float("inf")


class _StructuredPolyPrism(PolyPrism):
    """Private subclass of ``PolyPrism`` for structured (layered) mode.

    Constructed indirectly by ``PolyPrism(..., n_layers=...)``. Users
    never reference this class by name; meshwell internals do, via
    ``isinstance``.
    """

    def __init__(
        self,
        polygons,
        buffers,
        n_layers: list[int],
        recombine: bool = False,
        **kwargs,
    ):
        # Structured-mode validation (run before anything else so error
        # messages are immediate and don't depend on parent state).
        if not all(b == 0.0 for b in buffers.values()):
            raise ValueError(
                "PolyPrism with n_layers requires all buffer values to be zero "
                "(taper is not supported in structured mode)."
            )
        z_keys = list(buffers.keys())
        if len(z_keys) < 2:
            raise ValueError(
                "PolyPrism with n_layers needs at least 2 z-breakpoints in `buffers`."
            )
        for a, b in pairwise(z_keys):
            if not (b > a):
                raise ValueError(
                    f"PolyPrism with n_layers requires `buffers` z keys to be "
                    f"strictly increasing; got {a} then {b}."
                )
        if len(n_layers) != len(z_keys) - 1:
            raise ValueError(
                f"PolyPrism `n_layers` must have length {len(z_keys) - 1} "
                f"(one per z-interval); got {len(n_layers)}."
            )
        if any(n < 1 for n in n_layers):
            raise ValueError(
                f"PolyPrism `n_layers` entries must all be >= 1; got {n_layers}."
            )
        if kwargs.get("additive") or kwargs.get("subdivision") is not None:
            raise ValueError(
                "`additive=True` and `subdivision=` are not supported in "
                "structured mode (n_layers given). Drop n_layers or remove "
                "those kwargs."
            )

        # Defer the heavy polygon snap + state set to the parent. The
        # parent's extrude=True path is selected because all buffers are
        # zero (validated above).
        super().__init__(polygons=polygons, buffers=buffers, **kwargs)

        # Structured-mode-only state.
        self.n_layers = list(n_layers)
        self.recombine = recombine

    @property
    def z_breakpoints(self) -> list[float]:
        return list(self.buffers.keys())


def _polygon_to_multipolygon(geom: Polygon | MultiPolygon | list) -> MultiPolygon:
    """Coerce shapely Polygon / list / MultiPolygon to a MultiPolygon."""
    if isinstance(geom, MultiPolygon):
        return geom
    if isinstance(geom, list):
        flat: list[Polygon] = []
        for g in geom:
            if isinstance(g, MultiPolygon):
                flat.extend(g.geoms)
            else:
                flat.append(g)
        return MultiPolygon(flat)
    return MultiPolygon([geom])


def expand_slabs_for_entity(
    entity: "_StructuredPolyPrism", source_index: int
) -> list[Slab]:
    """Expand a structured ``PolyPrism`` into per-z-interval slabs.

    Each adjacent pair of z-keys becomes one slab with the corresponding
    ``n_layers``. The 2D footprint is the same for every slab (no taper
    in structured mode).
    """
    z_keys = entity.z_breakpoints
    mp = _polygon_to_multipolygon(entity.polygons)
    mesh_order = entity.mesh_order if entity.mesh_order is not None else float("inf")
    out: list[Slab] = []
    for (zlo, zhi), n in zip(pairwise(z_keys), entity.n_layers):
        out.append(
            Slab(
                footprint=mp,
                zlo=zlo,
                zhi=zhi,
                n_layers=int(n),
                recombine=entity.recombine,
                physical_name=entity.physical_name,
                source_index=source_index,
                mesh_order=mesh_order,
            )
        )
    return out


def _z_overlap(
    a_lo: float, a_hi: float, b_lo: float, b_hi: float
) -> tuple[float, float] | None:
    """Return overlapping z-interval, or None if disjoint or zero-volume touching."""
    lo = max(a_lo, b_lo)
    hi = min(a_hi, b_hi)
    if hi <= lo:
        return None
    return (lo, hi)


def _difference_footprint(
    low: Polygon | MultiPolygon, high: Polygon | MultiPolygon
) -> Polygon | MultiPolygon | None:
    """Shapely difference returning ``None`` when the result is empty.

    Filters non-polygonal residue (lines/points produced by degenerate
    subtractions) so callers always see a polygon-typed result.
    """
    diff = low.difference(high)
    if diff.is_empty:
        return None
    if isinstance(diff, Polygon):
        return diff
    if isinstance(diff, MultiPolygon):
        return diff
    polys = [g for g in getattr(diff, "geoms", []) if isinstance(g, Polygon)]
    if not polys:
        return None
    return polys[0] if len(polys) == 1 else MultiPolygon(polys)


def resolve_structured_slabs(entities_list: list) -> list[Slab]:
    """Decompose structured ``PolyPrism`` (``_StructuredPolyPrism``) entities into 3D-disjoint slabs.

    For overlapping prisms, lower ``mesh_order`` wins; ties resolved by
    insertion order. After this call, every returned :class:`Slab` is
    pairwise 3D-disjoint with every other returned slab (touching faces
    are allowed; volumetric intersection is not).

    Non-structured entities (``isinstance(ent, _StructuredPolyPrism)`` is
    False) are skipped.
    """
    raw_slabs: list[Slab] = []
    for idx, ent in enumerate(entities_list):
        if not isinstance(ent, _StructuredPolyPrism):
            continue
        raw_slabs.extend(expand_slabs_for_entity(ent, source_index=idx))

    if len(raw_slabs) <= 1:
        return raw_slabs

    raw_slabs.sort(key=lambda s: (s.mesh_order, s.source_index, s.zlo))

    resolved: list[Slab] = []
    for slab in raw_slabs:
        # Working list: list of (zlo, zhi, footprint) sub-pieces still to
        # resolve against subsequent higher-priority slabs.
        pieces: list[tuple[float, float, Polygon | MultiPolygon]] = [
            (slab.zlo, slab.zhi, slab.footprint)
        ]
        for hi in resolved:
            new_pieces: list[tuple[float, float, Polygon | MultiPolygon]] = []
            for p_lo, p_hi, p_fp in pieces:
                overlap = _z_overlap(p_lo, p_hi, hi.zlo, hi.zhi)
                if overlap is None:
                    new_pieces.append((p_lo, p_hi, p_fp))
                    continue
                ov_lo, ov_hi = overlap
                if p_lo < ov_lo:
                    new_pieces.append((p_lo, ov_lo, p_fp))
                ov_fp = _difference_footprint(p_fp, hi.footprint)
                if ov_fp is not None:
                    new_pieces.append((ov_lo, ov_hi, ov_fp))
                if p_hi > ov_hi:
                    new_pieces.append((ov_hi, p_hi, p_fp))
            pieces = new_pieces
        for p_lo, p_hi, p_fp in pieces:
            resolved.append(
                Slab(
                    footprint=p_fp,
                    zlo=p_lo,
                    zhi=p_hi,
                    n_layers=slab.n_layers,
                    recombine=slab.recombine,
                    physical_name=slab.physical_name,
                    source_index=slab.source_index,
                    mesh_order=slab.mesh_order,
                )
            )
    return resolved


class _StructuredPhantom:
    """Internal CAD-stage phantom for one resolved structured ``Slab``.

    Quacks like a meshwell entity for the purposes of ``cad_gmsh`` /
    ``cad_occ``: exposes ``polygons``, ``physical_name``, ``mesh_order``,
    ``mesh_bool``, ``dimension``, and the ``instanciate`` /
    ``instanciate_occ`` hooks. Built from a fully-resolved ``Slab`` (so
    the cascade has already pinned its footprint) and always sets
    ``mesh_bool = False`` so the existing keep=False top-dim machinery
    removes the body after fragmentation, leaving a void.
    """

    def __init__(self, slab: Slab):
        self.slab = slab
        self.polygons = slab.footprint  # MultiPolygon, used by prepare_entities
        self.physical_name = slab.physical_name
        self.mesh_order = slab.mesh_order if slab.mesh_order != float("inf") else None
        self.mesh_bool = False  # phantom -> removed after fragmentation
        self.dimension = 3

    # gmsh path: build via gmsh.model.occ.extrude
    def instanciate(self, cad_model) -> list[tuple[int, int]]:  # noqa: ARG002
        import gmsh

        height = self.slab.zhi - self.slab.zlo
        polys = (
            self.slab.footprint.geoms
            if hasattr(self.slab.footprint, "geoms")
            else [self.slab.footprint]
        )
        out_dimtags: list[tuple[int, int]] = []
        for poly in polys:
            ext_tags = self._add_polygon_face_gmsh(poly, z=self.slab.zlo)
            ext_dimtags = [(2, t) for t in ext_tags]
            extruded = gmsh.model.occ.extrude(ext_dimtags, 0, 0, height)
            out_dimtags.extend([dt for dt in extruded if dt[0] == 3])
        gmsh.model.occ.synchronize()
        return out_dimtags

    def _add_polygon_face_gmsh(self, polygon: Polygon, z: float) -> list[int]:
        """Construct a planar face for ``polygon`` at ``z`` in the OCC kernel.

        Holes are honored: the exterior is a curve loop; interiors become
        additional curve loops passed to ``addPlaneSurface``.
        """
        import gmsh

        def _curve_loop(coords) -> int:
            pts: list[int] = []
            for x, y in coords:
                pts.append(gmsh.model.occ.addPoint(x, y, z))
            lines: list[int] = []
            for a, b in pairwise(pts):
                lines.append(gmsh.model.occ.addLine(a, b))
            # Close the loop
            lines.append(gmsh.model.occ.addLine(pts[-1], pts[0]))
            return gmsh.model.occ.addCurveLoop(lines)

        # Drop the duplicated closing vertex shapely returns
        ext_coords = list(polygon.exterior.coords)[:-1]
        loops = [_curve_loop(ext_coords)]
        for interior in polygon.interiors:
            int_coords = list(interior.coords)[:-1]
            loops.append(_curve_loop(int_coords))
        face = gmsh.model.occ.addPlaneSurface(loops)
        return [face]

    # OCC path: build via OCP BRepPrimAPI_MakePrism
    def instanciate_occ(self):
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
        from OCP.gp import gp_Vec

        height = self.slab.zhi - self.slab.zlo
        vec = gp_Vec(0, 0, height)
        polys = (
            self.slab.footprint.geoms
            if hasattr(self.slab.footprint, "geoms")
            else [self.slab.footprint]
        )
        # Delegate wire construction to GeometryEntity helpers via a
        # tiny adapter so we don't duplicate arc handling. The phantom
        # itself never carries arcs (slabs are linear-segment polygons
        # produced by shapely), so we use the simple polyline path.
        from meshwell.geometry_entity import GeometryEntity

        adapter = GeometryEntity(point_tolerance=0.0)
        result_solids = []
        for poly in polys:
            ext_vertices = [(x, y, self.slab.zlo) for x, y in poly.exterior.coords]
            outer = adapter._make_occ_wire_from_vertices(
                ext_vertices, identify_arcs=False, min_arc_points=4, arc_tolerance=0.0
            )
            mf = BRepBuilderAPI_MakeFace(outer)
            for interior in poly.interiors:
                int_vertices = [(x, y, self.slab.zlo) for x, y in interior.coords]
                hole = adapter._make_occ_wire_from_vertices(
                    int_vertices,
                    identify_arcs=False,
                    min_arc_points=4,
                    arc_tolerance=0.0,
                )
                hole.Reverse()
                mf.Add(hole)
            face = mf.Face()
            result_solids.append(BRepPrimAPI_MakePrism(face, vec).Shape())

        if len(result_solids) == 1:
            return result_solids[0]

        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse

        result = result_solids[0]
        for s in result_solids[1:]:
            fuser = BRepAlgoAPI_Fuse(result, s)
            fuser.Build()
            result = fuser.Shape()
        return result
