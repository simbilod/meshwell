"""Structured (gmsh-tutorial-t3-style) layered extrusion for ``PolyPrism``.

This module hosts everything that is structured-specific:

* :class:`_StructuredPolyPrism` -- private subclass of ``PolyPrism`` that
  the ``PolyPrism.__new__`` dispatcher returns when the user passes
  ``n_layers=``. Same user-visible class identity (``isinstance(p, PolyPrism)``
  remains true), but with the structured-mode validation rules.
* :class:`Slab` -- one pairwise-disjoint piece of a structured prism after
  the cascade.
* :class:`_StructuredPhantom` -- CAD-stage proxy that punches a void in
  the OCC fragmentation. Marked ``is_structured_phantom=True`` so the
  CAD backends remove only its 3D solid (non-recursive), leaving its
  bottom, top, and lateral OCC sub-faces alive for the mesh stage to
  consume and tag.
* :func:`apply_structured_slabs` -- the conformal mesh-stage refill.
  Runs ``mesh.generate(2)`` first to give every OCC face a 2D mesh, then
  for each slab builds a layered wedge (or hex, when
  ``slab.recombine=True``) mesh into a discrete 3D entity. Layer 0
  reuses the bottom OCC face's existing node tags so the slab is
  conformal at z=zlo by construction; the top OCC face's mesh is then
  overridden with the translated triangulation so the slab is conformal
  at z=zhi too. Lateral OCC faces are flagged transfinite (and
  recombined when the slab is recombined) so they mesh as structured
  grids whose nodes align with the wedge/hex lateral edges.
"""
from __future__ import annotations

from collections import defaultdict
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
        fragment_fuzzy_value: BOP fuzzy used by ``cad_occ`` when this
            slab's geometry was fragmented. Drives the bbox-tolerance
            floor at mesh stage (``_find_occ_face_for_slab``,
            ``_apply_lateral_transfinite_hints``): after BOP, a planar
            face's bbox z-extent is bounded by this value plus gmsh's
            ``Precision::Confusion`` (~1e-7) bbox pad. ``None`` when the
            slab predates this plumbing (legacy sidecars) -- callers fall
            back to ``model_manager.point_tolerance``.
    """

    footprint: Polygon | MultiPolygon
    zlo: float
    zhi: float
    n_layers: int
    recombine: bool
    physical_name: tuple[str, ...]
    source_index: int = 0
    mesh_order: float = float("inf")
    fragment_fuzzy_value: float | None = None


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
        # Working list: list of (zlo, zhi, footprint, n_layers_for_this_piece)
        # sub-pieces still to resolve against subsequent higher-priority slabs.
        # n_layers is distributed proportionally on each z-split so total layer
        # count is preserved and donut-loser pieces sharing a z-range with the
        # winner end up with matching layer counts.
        pieces: list[tuple[float, float, Polygon | MultiPolygon, int]] = [
            (slab.zlo, slab.zhi, slab.footprint, slab.n_layers)
        ]
        for hi in resolved:
            new_pieces: list[tuple[float, float, Polygon | MultiPolygon, int]] = []
            for p_lo, p_hi, p_fp, p_n in pieces:
                overlap = _z_overlap(p_lo, p_hi, hi.zlo, hi.zhi)
                if overlap is None:
                    new_pieces.append((p_lo, p_hi, p_fp, p_n))
                    continue
                ov_lo, ov_hi = overlap
                # Proportional distribution of p_n across the (up to) three
                # z-pieces. The remainder goes into the overlap piece so the
                # total stays exactly equal to p_n.
                total_h = p_hi - p_lo
                n_pre = round(p_n * (ov_lo - p_lo) / total_h) if p_lo < ov_lo else 0
                n_post = round(p_n * (p_hi - ov_hi) / total_h) if p_hi > ov_hi else 0
                n_ov = p_n - n_pre - n_post
                # A pre/post piece with positive z-extent but 0 (or negative)
                # layers is degenerate -- gmsh can't extrude a 0-layer slab.
                # Treat that as the same alignment failure as n_ov <= 0.
                pre_bad = (p_lo < ov_lo) and n_pre <= 0
                post_bad = (p_hi > ov_hi) and n_post <= 0
                if n_ov <= 0 or pre_bad or post_bad or n_pre < 0 or n_post < 0:
                    raise StructuredLayerMismatchError(
                        f"Cannot split structured slab {slab.physical_name} "
                        f"(z=[{p_lo}, {p_hi}], n_layers={p_n}) at "
                        f"z=[{ov_lo}, {ov_hi}]: layer count cannot be "
                        f"distributed evenly. Choose z-breakpoints that align "
                        f"with the layer grid."
                    )
                if p_lo < ov_lo:
                    new_pieces.append((p_lo, ov_lo, p_fp, n_pre))
                ov_fp = _difference_footprint(p_fp, hi.footprint)
                if ov_fp is not None:
                    new_pieces.append((ov_lo, ov_hi, ov_fp, n_ov))
                if p_hi > ov_hi:
                    new_pieces.append((ov_hi, p_hi, p_fp, n_post))
            pieces = new_pieces
        for p_lo, p_hi, p_fp, p_n in pieces:
            resolved.append(
                Slab(
                    footprint=p_fp,
                    zlo=p_lo,
                    zhi=p_hi,
                    n_layers=p_n,
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

    is_structured_phantom = True

    def __init__(self, slab: Slab):
        self.slab = slab
        self.polygons = slab.footprint  # MultiPolygon, used by prepare_entities
        self.physical_name = slab.physical_name
        self.mesh_order = slab.mesh_order if slab.mesh_order != float("inf") else None
        self.mesh_bool = False  # phantom -> volume removed after fragmentation
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


class StructuredLayerMismatchError(ValueError):
    """Raised on conflicting ``n_layers`` across a shared horizontal face.

    Two structured slabs that stack vertically (one's ``zhi`` equals the
    other's ``zlo``) and overlap in xy must agree on ``n_layers`` so the
    shared face has a single, consistent mesh.
    """


class StructuredFaceTopologySplitError(ValueError):
    """Raised when an entity asymmetrically splits a slab's z=zlo or z=zhi face.

    The mesh-stage builder (``_build_one_slab_conformal``) reads the slab's
    bottom face mesh and translates it upward to populate the top face. For
    this to work, gmsh's ``setPeriodic(2, top_face, bottom_face, T)`` must
    succeed -- which requires the two faces to have **matching OCC topology**.
    When some entity touches z=zlo with footprint F1 but nothing matches it
    at z=zhi (or vice versa), the slab's bottom and top OCC faces partition
    differently and periodic fails. The "Unknown node" crash in the slab
    build follows.

    Until the cascade-extension fix (splitting slab footprints at xy-boundaries
    of entities that touch z=zlo or z=zhi) is implemented, the only safe
    behaviour is to fail at CAD stage with a clear pointer to the offending
    entity.
    """


def _entity_z_face_set(entity) -> list[float]:
    """Return z-values of horizontal faces that the entity's BODY exposes.

    A PolyPrism with ``buffers={a: 0, b: 0}`` exposes faces at z=a and z=b
    (extruded mode -- single body). With non-zero or varying buffers (rib
    surfaces at intermediate z keys), each z key contributes a face. A
    PolySurface / PolyLine at ``translation=(0,0,z)`` exposes one face/curve
    at z. Returns ``[]`` for entities without an inherent z-localization
    (e.g. InterfaceTag).
    """
    if hasattr(entity, "buffers") and entity.buffers:
        return list(entity.buffers.keys())
    if hasattr(entity, "translation") and entity.translation is not None:
        return [float(entity.translation[2])]
    return []


def _entity_z_range(entity) -> tuple[float, float] | None:
    """Return (z_min, z_max) of the entity's vertical extent, or None."""
    zs = _entity_z_face_set(entity)
    if not zs:
        return None
    return (min(zs), max(zs))


def _entity_footprint_multi(entity):
    """Return the entity's xy footprint as a MultiPolygon, or ``None``."""
    polys = getattr(entity, "polygons", None)
    if polys is None:
        return None
    if isinstance(polys, MultiPolygon):
        return polys
    if isinstance(polys, Polygon):
        return MultiPolygon([polys])
    if isinstance(polys, list):
        flat: list[Polygon] = []
        for p in polys:
            if isinstance(p, MultiPolygon):
                flat.extend(p.geoms)
            elif isinstance(p, Polygon):
                flat.append(p)
        return MultiPolygon(flat) if flat else None
    return None


def _validate_slab_face_topology_symmetry(
    slabs: list["Slab"], entities_list: list, tol: float = 1e-3
) -> None:
    """Raise when a non-structured entity has a z-face within a slab's z-range.

    The structured-slab mesh-stage builder is built around the invariant that
    every slab is a simple z-extrusion of a 2D footprint: its OCC bottom and
    top faces are single-component and topologically identical, and its only
    lateral faces span the full ``[zlo, zhi]`` z-range. Any non-structured
    entity that has a horizontal face strictly inside the slab z-range
    (``zlo < z < zhi``), or one that partially covers the slab footprint at
    z=zlo or z=zhi, breaks the invariant:

    * **Interior z-face**: introduces a lateral perimeter face whose
      z-extent is a sub-interval of the slab z-range.
      ``_apply_lateral_transfinite_hints`` misses it (its lateral filter
      only accepts faces spanning the full slab z-range) and the slab
      build silently produces overlapping wedges + neighbour tets.
    * **Partial coverage at z=zlo or z=zhi**: splits the slab's bottom
      or top OCC face asymmetrically with the opposite z-face, so
      ``setPeriodic`` fails and ``_build_one_slab_conformal`` references
      un-meshed nodes.

    Structured PolyPrism entities are skipped: ``resolve_structured_slabs``
    has its own structured-vs-structured z-cascade for them.
    """
    import logging

    logger = logging.getLogger(__name__)

    for slab in slabs:
        slab_area = slab.footprint.area
        if slab_area <= 0:
            continue
        rel_tol = max(tol, 1e-9) * max(slab_area, 1.0)
        for idx, ent in enumerate(entities_list):
            if idx == slab.source_index:
                continue
            if isinstance(ent, _StructuredPolyPrism):
                continue
            z_range = _entity_z_range(ent)
            if z_range is None:
                continue
            ent_fp = _entity_footprint_multi(ent)
            if ent_fp is None or ent_fp.is_empty:
                continue
            try:
                inter = slab.footprint.intersection(ent_fp)
            except Exception as exc:
                logger.debug(
                    "shapely intersection failed for slab %s vs entity %d: %s",
                    slab.physical_name,
                    idx,
                    exc,
                )
                continue
            if inter.is_empty or inter.area <= rel_tol:
                continue
            full_coverage = abs(inter.area - slab_area) <= rel_tol
            emin, emax = z_range
            violations: list[str] = []
            for z_label, z in (("z-start", emin), ("z-end", emax)):
                # Strictly interior z-face: always rejected.
                if slab.zlo + tol < z < slab.zhi - tol:
                    violations.append(f"{z_label}={z} is interior to slab z-range")
                # At slab z-boundary AND partial coverage: rejected
                # (asymmetric split with the opposite z-face).
                elif (
                    abs(z - slab.zlo) <= tol or abs(z - slab.zhi) <= tol
                ) and not full_coverage:
                    boundary = "z=zlo" if abs(z - slab.zlo) <= tol else "z=zhi"
                    violations.append(
                        f"{z_label}={z} sits on slab {boundary} with partial "
                        f"xy coverage ({inter.area:.3g} of {slab_area:.3g})"
                    )
            if not violations:
                continue
            name = getattr(ent, "physical_name", None) or f"entity[{idx}]"
            if isinstance(name, tuple):
                name = "/".join(name) or f"entity[{idx}]"
            raise StructuredFaceTopologySplitError(
                f"Structured slab {slab.physical_name} at "
                f"z=[{slab.zlo}, {slab.zhi}] is incompatible with entity "
                f"{name} (index {idx}, z-range=[{emin}, {emax}], clipped "
                f"bbox={inter.bounds}):\n"
                + "\n".join(f"  - {v}" for v in violations)
                + "\n\nThe structured-slab builder requires every "
                "non-structured entity intersecting the slab to either "
                "(a) lie strictly outside the slab z-range, (b) fully "
                "cross the slab (z-range strictly contains [zlo, zhi]), "
                "or (c) sit at z=zlo or z=zhi with footprint fully "
                "covering the slab footprint (mirror)."
            )


# gmsh's OCC ``getBoundingBox`` adds ``Precision::Confusion()`` (~1e-7) of
# padding to every face bounding box, so a genuinely planar face reports a
# z-extent of ~2e-7 even before any BOP fuzzy snapping. Floor for the
# bbox-based face-locator helpers.
_OCC_BBOX_PAD = 2e-7


def _bbox_tol_for_slab(slab: Slab, fallback_tol: float) -> float:
    """Tolerance for OCC-face bbox comparisons against a slab's z-planes.

    After ``cad_occ``/``cad_gmsh`` BOP fragmentation with fuzzy ``F``, a
    nominally-planar face's bbox z-extent is bounded by ``F`` plus the
    constant ``_OCC_BBOX_PAD`` gmsh adds. To accept such faces without
    rejecting them or matching unrelated planes, use
    ``max(F, _OCC_BBOX_PAD)``.

    ``slab.fragment_fuzzy_value`` carries ``F`` from the CAD stage; when
    it is ``None`` (legacy sidecars, pre-plumbing slabs), fall back to
    ``fallback_tol`` -- typically ``model_manager.point_tolerance``,
    which conventionally matches the cad-stage fuzzy.
    """
    fuzzy = slab.fragment_fuzzy_value
    if fuzzy is None:
        fuzzy = fallback_tol
    return max(fuzzy, _OCC_BBOX_PAD)


def _validate_slab_layer_mating(slabs: list[Slab], tol: float) -> None:
    """Raise if any pair of slabs sharing a horizontal face disagrees on n_layers.

    Two slabs share a horizontal face iff ``a.zhi == b.zlo`` (within
    ``tol``) and their xy footprints overlap with non-zero area.
    """
    for i, a in enumerate(slabs):
        for b in slabs[i + 1 :]:
            # Slabs from the same originating entity are consistent by
            # construction (the user supplied one n_layers per z-interval);
            # intra-entity stacking with differing counts is allowed.
            if a.source_index == b.source_index:
                continue
            # Order so lo is below hi (lo.zhi == hi.zlo).
            if abs(a.zhi - b.zlo) <= tol:
                lo, hi = a, b
            elif abs(b.zhi - a.zlo) <= tol:
                lo, hi = b, a
            else:
                continue
            shared = lo.footprint.intersection(hi.footprint)
            if shared.is_empty or shared.area < tol * tol:
                continue
            if lo.n_layers != hi.n_layers:
                raise StructuredLayerMismatchError(
                    f"Stacked structured slabs {lo.physical_name} (n_layers="
                    f"{lo.n_layers}) and {hi.physical_name} (n_layers="
                    f"{hi.n_layers}) share a horizontal face at z="
                    f"{lo.zhi} but disagree on n_layers. v1 requires "
                    f"matching layer counts on shared horizontal faces."
                )


def apply_structured_slabs(model_manager, slabs: list[Slab]) -> None:
    """Build conformal layered meshes for each ``Slab`` in the gmsh model.

    Conformal pipeline:

    1. Apply transfinite hints to each slab's lateral OCC faces (and
       ``setRecombine`` on bottom/top/lateral when ``slab.recombine``).
    2. ``mesh.generate(2)`` -- gives every OCC face (slab's own bottom,
       top, and lateral, plus any neighbor faces) a 2D mesh.
    3. For each slab, ``_build_one_slab_conformal`` reads the bottom
       face's 2D mesh and constructs a layered wedge (or hex, when the
       bottom was recombined to quads) mesh in a fresh discrete 3D
       entity. Layer 0 reuses the bottom face's existing node tags so
       the slab is conformal at ``zlo`` by construction; the top OCC
       face's mesh is then overridden with the translated cells so the
       slab is conformal at ``zhi`` too. Lateral nodes coincide with the
       transfinite lateral OCC face nodes; ``removeDuplicateNodes`` (run
       after ``mesh.generate(3)``) fuses them into shared tags.

    Boundary tagging (``slab___None``, ``slabA___slabB``) is handled by
    the regular ``_tag_physicals`` / ``_compute_physical_groups``
    machinery, which treats structured phantoms as kept entities for
    tagging purposes. Phantom OCC sub-faces survive thanks to the
    non-recursive removal in ``_remove_keep_false_top_dim``.
    """
    import gmsh

    if not slabs:
        return

    tol = model_manager.point_tolerance or 1e-9
    _validate_slab_layer_mating(slabs, tol)

    # Process slabs bottom-up so when slab B sits on top of slab A and
    # they share a horizontal face, A has already overridden the shared
    # face's mesh; B reads that overridden mesh as its own bottom.
    sorted_slabs = sorted(slabs, key=lambda s: (s.zlo, s.zhi))

    # Apply lateral transfinite hints BEFORE 2D meshing so the void's
    # lateral OCC faces are meshed as structured grids whose nodes align
    # with the structured wedges' lateral edges. After mesh.generate(3),
    # the (now-coincident) lateral nodes on either side of each lateral
    # interface are merged into shared tags via removeDuplicateNodes
    # (called from process_mesh).
    #
    # When ``slab.recombine`` is True, also flag the slab's bottom, top,
    # and lateral OCC faces for recombination so their 2D meshes come
    # out as quads -- the conformal builder then produces hex elements.
    for slab in sorted_slabs:
        _apply_lateral_transfinite_hints(slab, tol)
        if slab.recombine:
            _apply_horizontal_recombine_hints(slab, tol)
        # Make the top horizontal OCC face a periodic translation of the
        # bottom. Without this, gmsh's size-field-driven 1D mesher can give
        # the bottom and top boundary curves DIFFERENT node counts, which
        # makes the lateral ``setTransfiniteSurface`` fail at mesh.generate(2)
        # (opposite sides of a transfinite rectangle must match). The
        # structured slab takes precedence over any non-structured neighbour
        # on the shared horizontal plane.
        _apply_slab_horizontal_periodicity(slab, tol)
        # Likewise pin every vertical edge as a horizontal translation of
        # one master vertical edge of the same slab. Transfinite alone pins
        # node *count* (n_layers+1) but not positions; this guarantees every
        # vertical edge of the slab meshes at identical z values even if a
        # non-structured neighbour or size-field interaction would have
        # perturbed an individual edge.
        _apply_slab_vertical_periodicity(slab, tol)

    # Step 1: mesh all 2D OCC faces so we can read their triangulations.
    gmsh.model.mesh.generate(2)

    # Collect (physical_name -> list of 3D volume tags) across all slabs
    # so we can emit one physical group per name at the end. Slabs that
    # share a physical_name (e.g. a single structured PolyPrism split into
    # multiple sub-slabs by the cascade) must collapse to one group.
    volumes_by_name: dict[str, list[int]] = defaultdict(list)
    for slab in sorted_slabs:
        vol_tag, _info = _build_one_slab_conformal(slab, tol)
        for name in slab.physical_name:
            volumes_by_name[name].append(vol_tag)

    # Emit one 3D physical group per name. Boundary tagging is handled by
    # the regular _tag_physicals (cad_gmsh) / _compute_physical_groups
    # (cad_occ) machinery, which treats structured phantoms as kept for
    # tagging purposes; their OCC sub-faces survive non-recursive removal,
    # get meshed by mesh.generate(2), and pick up physical groups there.
    for name, vols in volumes_by_name.items():
        if not vols:
            continue
        pg = gmsh.model.addPhysicalGroup(3, vols)
        gmsh.model.setPhysicalName(3, pg, name)


def _apply_lateral_transfinite_hints(slab: Slab, tol: float) -> None:
    """Force lateral OCC faces of the slab void to mesh as structured grids.

    For each OCC face whose z-extent spans ``[slab.zlo, slab.zhi]`` (within
    ``tol``) and whose xy-bounding-box intersects the slab footprint
    boundary (i.e. it is a lateral face of the void), set every fully
    vertical bounding edge to ``n_layers + 1`` nodes via
    ``setTransfiniteCurve`` and mark the face itself as
    ``setTransfiniteSurface`` so gmsh meshes it as a structured grid.

    The grid's rows are layer z-positions (matching the structured wedges'
    layer 0..n_layers nodes); its columns inherit the polygon edge's 1D
    mesh from the bottom face (shared OCC edge topology), so the resulting
    lateral nodes coincide positionally with the wedges' lateral edge
    nodes. ``removeDuplicateNodes`` (called once after ``mesh.generate(3)``)
    fuses the coincident pairs into shared tags.

    Silently no-ops when no matching OCC faces exist (e.g. isolated slab
    handled by the geo-fallback path), or when an individual transfinite
    call fails (logs at ``warning`` and continues).
    """
    import logging

    import gmsh
    from shapely.geometry import box as sh_box

    logger = logging.getLogger(__name__)

    n_layers = slab.n_layers
    occ_faces = gmsh.model.occ.getEntities(2)
    boundary = slab.footprint.boundary

    for dim, ftag in occ_faces:
        if dim != 2:
            continue
        try:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(2, ftag)
        except Exception as exc:
            logger.debug("getBoundingBox failed for face %d: %s", ftag, exc)
            continue
        # Lateral face: spans full slab z-range. The bbox tolerance must
        # accept faces whose vertices were snapped by the CAD-stage BOP
        # fuzzy AND the constant ~2e-7 pad gmsh's OCC adds to every
        # bounding box. See ``_bbox_tol_for_slab``.
        bbox_tol = _bbox_tol_for_slab(slab, tol)
        if abs(zmin - slab.zlo) > bbox_tol or abs(zmax - slab.zhi) > bbox_tol:
            continue
        if abs(zmax - zmin) < bbox_tol:
            continue  # horizontal face (bottom/top)
        # xy-bbox must touch the slab footprint boundary.
        if xmax - xmin < tol and ymax - ymin < tol:
            continue
        face_xy_box = sh_box(xmin, ymin, xmax, ymax)
        # Use ``dwithin`` instead of ``intersects`` because gmsh's bbox
        # is padded by ~1e-5; a face whose xy strip lies exactly on the
        # footprint perimeter ends up offset and fails strict intersect.
        try:
            if not boundary.dwithin(face_xy_box, max(tol, 1e-4)):
                continue
        except Exception as exc:
            logger.debug("shapely boundary.dwithin failed for face %d: %s", ftag, exc)
            continue

        # Find vertical edges of this face.
        try:
            boundary_curves = gmsh.model.getBoundary(
                [(2, ftag)],
                oriented=False,
                recursive=False,
            )
        except Exception as exc:
            logger.debug(
                "getBoundary failed for face %d (slab %s): %s",
                ftag,
                slab.physical_name,
                exc,
            )
            continue
        for cdim, ctag in boundary_curves:
            if cdim != 1:
                continue
            try:
                (
                    cxmin,
                    cymin,
                    czmin,
                    cxmax,
                    cymax,
                    czmax,
                ) = gmsh.model.occ.getBoundingBox(1, ctag)
            except Exception as exc:
                logger.debug("getBoundingBox failed for curve %d: %s", ctag, exc)
                continue
            # ``getBoundingBox`` for OCC curves can add a non-trivial
            # tolerance pad (~1e-3) after a BREP round-trip. Use a more
            # forgiving xy-extent threshold relative to the slab z-span
            # so we still classify vertical edges correctly while not
            # mistaking near-vertical short curves for vertical ones.
            xy_extent_tol = max(tol * 10, 0.1 * (slab.zhi - slab.zlo))
            if (
                abs(czmin - slab.zlo) < xy_extent_tol
                and abs(czmax - slab.zhi) < xy_extent_tol
                and abs(cxmax - cxmin) < xy_extent_tol
                and abs(cymax - cymin) < xy_extent_tol
                and (czmax - czmin) > 0.5 * (slab.zhi - slab.zlo)
            ):
                try:
                    gmsh.model.mesh.setTransfiniteCurve(ctag, n_layers + 1)
                except Exception as exc:
                    logger.warning(
                        "Failed to set transfinite on vertical edge %d for "
                        "slab %s: %s",
                        ctag,
                        slab.physical_name,
                        exc,
                    )
        # Mark the lateral face as transfinite.
        try:
            gmsh.model.mesh.setTransfiniteSurface(ftag)
        except Exception as exc:
            logger.warning(
                "Failed to set transfinite on lateral face %d for slab %s: %s",
                ftag,
                slab.physical_name,
                exc,
            )
        # If the slab is recombined, the lateral grid must be quads so
        # it matches the hex/wedge lateral facets one-for-one. Without
        # setRecombine the transfinite grid is triangulated by default.
        if slab.recombine:
            try:
                gmsh.model.mesh.setRecombine(2, ftag)
            except Exception as exc:
                logger.warning(
                    "Failed to setRecombine on lateral face %d for slab %s: %s",
                    ftag,
                    slab.physical_name,
                    exc,
                )


def _apply_slab_horizontal_periodicity(slab: Slab, tol: float) -> None:
    """Pin the top horizontal OCC face's mesh as a translation of the bottom.

    Sets ``gmsh.model.mesh.setPeriodic(2, [top_face], [bottom_face], T)``
    with ``T`` a pure translation by ``(0, 0, slab.zhi - slab.zlo)``. After
    ``mesh.generate(2)``, gmsh meshes the master (bottom) under the size
    field, then constructs the slave (top) by applying ``T`` — including
    every bounding 1D curve and 0D vertex. This guarantees:

    * Bottom/top boundary 1D curves have identical node counts at
      translation-paired positions, so the lateral
      ``setTransfiniteSurface`` constraint always finds matching opposing
      sides (the production failure this fixes).
    * The top face's interior 2D mesh is a translated copy of the bottom's,
      eliminating any chance of mismatch.

    Stacked slabs (``A.zhi == B.zlo``) chain naturally: ``A_top`` is slave
    of ``A_bottom`` and master of ``B_top`` — gmsh composes the translations.
    The structured slab's periodic constraint dominates any non-structured
    neighbour sharing the horizontal plane.

    Silently no-ops when either face can't be located (e.g. embedded
    surface splits one face but not the other — lateral transfinite would
    already fail for that geometry and this helper makes things no worse).
    """
    import logging

    import gmsh

    logger = logging.getLogger(__name__)

    occ_faces = gmsh.model.occ.getEntities(2)
    bottom_face = _find_occ_face_for_slab(occ_faces, slab, slab.zlo, tol)
    top_face = _find_occ_face_for_slab(occ_faces, slab, slab.zhi, tol)
    if bottom_face is None or top_face is None:
        logger.debug(
            "Skipping periodic for slab %s at z=[%s, %s]: bottom=%s top=%s",
            slab.physical_name,
            slab.zlo,
            slab.zhi,
            bottom_face,
            top_face,
        )
        return

    dz = slab.zhi - slab.zlo
    affine = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        dz,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    try:
        gmsh.model.mesh.setPeriodic(2, [top_face], [bottom_face], affine)
    except Exception as exc:
        logger.warning(
            "Failed to set periodic top->bottom face for slab %s: %s",
            slab.physical_name,
            exc,
        )


def _collect_slab_vertical_edges(
    slab: Slab, tol: float
) -> list[tuple[int, tuple[float, float]]]:
    """Return ``(edge_tag, (x_mid, y_mid))`` for every vertical OCC edge of ``slab``.

    A vertical edge is one whose bbox spans ``[slab.zlo, slab.zhi]`` in z
    with negligible xy extent, AND that bounds at least one of the slab's
    lateral OCC faces. Dedup'd across faces (a shared vertical edge between
    two lateral faces appears once).
    """
    import logging

    import gmsh
    from shapely.geometry import box as sh_box

    logger = logging.getLogger(__name__)

    occ_faces = gmsh.model.occ.getEntities(2)
    boundary = slab.footprint.boundary
    bbox_tol = _bbox_tol_for_slab(slab, tol)
    xy_extent_tol = max(tol * 10, 0.1 * (slab.zhi - slab.zlo))

    edges: list[tuple[int, tuple[float, float]]] = []
    seen: set[int] = set()
    for dim, ftag in occ_faces:
        if dim != 2:
            continue
        try:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(2, ftag)
        except Exception as exc:
            logger.debug("getBoundingBox failed for face %d: %s", ftag, exc)
            continue
        # Lateral face filter -- same predicate as _apply_lateral_transfinite_hints.
        if abs(zmin - slab.zlo) > bbox_tol or abs(zmax - slab.zhi) > bbox_tol:
            continue
        if abs(zmax - zmin) < bbox_tol:
            continue
        if xmax - xmin < tol and ymax - ymin < tol:
            continue
        face_xy_box = sh_box(xmin, ymin, xmax, ymax)
        try:
            if not boundary.dwithin(face_xy_box, max(tol, 1e-4)):
                continue
        except Exception as exc:
            logger.debug("shapely boundary.dwithin failed for face %d: %s", ftag, exc)
            continue
        try:
            curves = gmsh.model.getBoundary(
                [(2, ftag)], oriented=False, recursive=False
            )
        except Exception as exc:
            logger.debug("getBoundary failed for face %d: %s", ftag, exc)
            continue
        for cdim, ctag in curves:
            if cdim != 1 or ctag in seen:
                continue
            try:
                (
                    cxmin,
                    cymin,
                    czmin,
                    cxmax,
                    cymax,
                    czmax,
                ) = gmsh.model.occ.getBoundingBox(1, ctag)
            except Exception as exc:
                logger.debug("getBoundingBox failed for curve %d: %s", ctag, exc)
                continue
            if (
                abs(czmin - slab.zlo) < xy_extent_tol
                and abs(czmax - slab.zhi) < xy_extent_tol
                and abs(cxmax - cxmin) < xy_extent_tol
                and abs(cymax - cymin) < xy_extent_tol
                and (czmax - czmin) > 0.5 * (slab.zhi - slab.zlo)
            ):
                seen.add(ctag)
                edges.append((ctag, (0.5 * (cxmin + cxmax), 0.5 * (cymin + cymax))))
    return edges


def _apply_slab_vertical_periodicity(slab: Slab, tol: float) -> None:
    """Pin every vertical edge of the slab as a horizontal translation of one master.

    Counterpart to :func:`_apply_slab_horizontal_periodicity` for the slab's
    vertical 1D edges. ``setTransfiniteCurve`` pins node *count*
    (``n_layers + 1``); ``setPeriodic`` pins node *positions* (slave nodes =
    master nodes plus a pure horizontal translation). The two together
    guarantee that every vertical edge of a slab meshes at identical z
    values regardless of how the size field or any neighbour might
    otherwise have perturbed an individual edge.

    Master selection is deterministic (lexicographically smallest
    ``(x_mid, y_mid)``), so re-running the same model produces identical
    periodic relationships.

    No-op when fewer than two vertical edges are found (nothing to make
    periodic).
    """
    import logging

    import gmsh

    logger = logging.getLogger(__name__)

    edges = _collect_slab_vertical_edges(slab, tol)
    if len(edges) < 2:
        return

    edges.sort(key=lambda item: item[1])
    master_tag, (mx, my) = edges[0]
    for slave_tag, (sx, sy) in edges[1:]:
        dx = sx - mx
        dy = sy - my
        affine = [
            1.0,
            0.0,
            0.0,
            dx,
            0.0,
            1.0,
            0.0,
            dy,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
        try:
            gmsh.model.mesh.setPeriodic(1, [slave_tag], [master_tag], affine)
        except Exception as exc:
            logger.warning(
                "Failed to set vertical-edge periodic for slab %s "
                "(master=%d, slave=%d): %s",
                slab.physical_name,
                master_tag,
                slave_tag,
                exc,
            )


def _apply_horizontal_recombine_hints(slab: Slab, tol: float) -> None:
    """Mark the slab's bottom and top OCC faces for recombination.

    Required when ``slab.recombine`` is True so the 2D mesh of those
    faces comes out as quads -- the conformal builder reads the bottom
    quad mesh and builds hex elements (gmsh type 5) instead of wedges.
    """
    import logging

    import gmsh

    logger = logging.getLogger(__name__)
    occ_faces = gmsh.model.occ.getEntities(2)
    bot = _find_occ_face_for_slab(occ_faces, slab, slab.zlo, tol)
    top = _find_occ_face_for_slab(occ_faces, slab, slab.zhi, tol)
    for face_tag in (bot, top):
        if face_tag is None:
            continue
        try:
            gmsh.model.mesh.setRecombine(2, face_tag)
        except Exception as exc:
            logger.warning(
                "Failed to setRecombine on horizontal face %d for slab %s: %s",
                face_tag,
                slab.physical_name,
                exc,
            )


def _build_one_slab_conformal(slab: Slab, tol: float) -> tuple[int, dict]:
    """Build a discrete 3D entity for ``slab`` with conformal bottom/top.

    Returns ``(vol_tag, lateral_info)`` where ``lateral_info`` is a dict
    with keys ``layers`` (list of np.ndarray of node tags per layer 0..n),
    ``triangles`` (np.ndarray of bottom-face cell connectivity in tag
    space; triangles for wedge slabs and quads for hex slabs),
    ``tag_to_idx`` (bottom-node-tag -> index), and ``coord`` (the
    bottom-face node coordinates at z=zlo).

    Phantom OCC sub-faces survive ``_remove_keep_false_top_dim`` (which
    removes the volume non-recursively for ``is_structured_phantom``
    entities), so ``_find_occ_face_for_slab`` always returns a valid
    face for both ``slab.zlo`` and ``slab.zhi``.
    """
    import logging

    import gmsh
    import numpy as np

    logger = logging.getLogger(__name__)

    occ_faces = gmsh.model.occ.getEntities(2)
    bottom_face = _find_occ_face_for_slab(occ_faces, slab, slab.zlo, tol)
    top_face = _find_occ_face_for_slab(occ_faces, slab, slab.zhi, tol)

    if bottom_face is None or top_face is None:
        raise RuntimeError(
            f"Conformal slab build for {slab.physical_name} at "
            f"z=[{slab.zlo}, {slab.zhi}] could not locate "
            f"{'bottom' if bottom_face is None else 'top'} OCC face. "
            f"This indicates a bug in phantom sub-face preservation; "
            f"check that _StructuredPhantom volumes are removed "
            f"non-recursively in _remove_keep_false_top_dim."
        )

    # Read bottom face's 2D mesh (interior + boundary nodes).
    node_tags, coord, _ = gmsh.model.mesh.getNodes(2, bottom_face, includeBoundary=True)
    if len(node_tags) == 0:
        raise RuntimeError(
            f"Bottom OCC face {bottom_face} for slab {slab.physical_name} "
            f"at z={slab.zlo} has no 2D mesh nodes after mesh.generate(2)."
        )
    coord = np.asarray(coord, dtype=float).reshape(-1, 3)
    node_tags = np.asarray(node_tags, dtype=np.uint64)

    # Identify which bottom-face nodes lie on the bounding curves (1D
    # entities). For those, layer n must reuse the corresponding top-face
    # boundary curve's existing node tags rather than creating new ones,
    # otherwise we duplicate the 1D-curve nodes that the surrounding OCC
    # face also owns.
    # Boundary-only nodes: exclude (dim=2) interior nodes.
    bottom_iface_nodes, _, _ = gmsh.model.mesh.getNodes(
        2, bottom_face, includeBoundary=False
    )
    interior_set = {int(t) for t in bottom_iface_nodes}
    is_boundary = np.array(
        [int(t) not in interior_set for t in node_tags.tolist()], dtype=bool
    )

    # Build a mapping from bottom-boundary xy -> top-face boundary node tag,
    # by collecting all nodes on top_face's boundary curves at z=zhi.
    top_boundary_lookup: dict[tuple[float, float], int] = {}
    # Get curves bounding top_face.
    top_curves = gmsh.model.getBoundary(
        [(2, top_face)], oriented=False, recursive=False
    )
    # Collect 1D nodes on each bounding curve of top_face. includeBoundary=True
    # also picks up the 0D vertex nodes shared between adjacent curves.
    snap = max(tol, 1e-9)
    for cdim, ctag in top_curves:
        if cdim != 1:
            continue
        try:
            tnt, tcd, _ = gmsh.model.mesh.getNodes(1, ctag, includeBoundary=True)
        except Exception as exc:
            logger.debug("Skipping top-face curve %d (no 1D mesh nodes): %s", ctag, exc)
            continue
        tcd = np.asarray(tcd, dtype=float).reshape(-1, 3)
        for i, tag in enumerate(tnt):
            x, y = tcd[i, 0], tcd[i, 1]
            key = (round(x / snap), round(y / snap))
            top_boundary_lookup[key] = int(tag)

    # Read bottom face's element mesh. The conformal builder handles
    # either triangles (-> wedge volume, type 6) or quads (-> hex volume,
    # type 5). The latter only happens when ``slab.recombine`` is True
    # and ``_apply_horizontal_recombine_hints`` flagged the face.
    elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(2, bottom_face)
    cells_nodes_flat: np.ndarray | None = None
    cells_per_face = 3  # triangle by default
    gmsh_3d_type = 6  # wedge
    top_elem_gmsh_type = 2  # triangle
    for et, en in zip(elem_types, elem_node_tags):
        if int(et) == 2:  # triangle -> wedge
            cells_nodes_flat = np.asarray(en, dtype=np.uint64)
            cells_per_face = 3
            gmsh_3d_type = 6
            top_elem_gmsh_type = 2
            break
        if int(et) == 3:  # quad -> hex
            cells_nodes_flat = np.asarray(en, dtype=np.uint64)
            cells_per_face = 4
            gmsh_3d_type = 5
            top_elem_gmsh_type = 3
            break
    if cells_nodes_flat is None or cells_nodes_flat.size == 0:
        raise RuntimeError(
            f"Bottom OCC face {bottom_face} for slab {slab.physical_name} "
            f"at z={slab.zlo} has no triangle/quad elements after "
            f"mesh.generate(2)."
        )
    triangles = cells_nodes_flat.reshape(-1, cells_per_face)

    # tag -> index in coord array
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags.tolist())}
    n_nodes = len(node_tags)
    n_layers = slab.n_layers

    # Build per-layer node tags: layers[i] is a numpy array of length
    # n_nodes giving the gmsh node tag for column j at layer i.
    # Layer 0 reuses bottom's existing tags.
    layers: list[np.ndarray] = [node_tags.copy()]

    next_node_tag = int(gmsh.model.mesh.getMaxNodeTag()) + 1
    new_layer_tags: list[np.ndarray] = []  # tags for layers 1..n_layers
    new_layer_coords: list[np.ndarray] = []  # coords for layers 1..n_layers
    height = slab.zhi - slab.zlo
    for i in range(1, n_layers + 1):
        # Linear interpolation from zlo to zhi.
        z_i = slab.zlo + height * (i / n_layers)
        coords_i = coord.copy()
        coords_i[:, 2] = z_i
        is_top_layer = i == n_layers
        tags_i = np.empty(n_nodes, dtype=np.uint64)
        for j in range(n_nodes):
            if is_top_layer and is_boundary[j]:
                # Reuse the existing top-face 1D-curve node at this xy.
                key = (
                    round(coord[j, 0] / snap),
                    round(coord[j, 1] / snap),
                )
                tag = top_boundary_lookup.get(key)
                if tag is None:
                    # No matching top boundary node; allocate a new tag
                    # (this happens when bottom and top boundaries don't
                    # align, e.g. tapered/snap-perturbed geometries).
                    tag = next_node_tag
                    next_node_tag += 1
                tags_i[j] = tag
            else:
                tags_i[j] = next_node_tag
                next_node_tag += 1
        layers.append(tags_i)
        new_layer_tags.append(tags_i)
        new_layer_coords.append(coords_i)

    # Create the discrete volume. We deliberately do NOT pass a boundary
    # list -- declaring [bottom_face, top_face] as the boundary makes
    # gmsh's 3D mesher think the discrete volume is bounded by ONLY
    # those faces (so it tries to "fill" the volume between them with
    # tets generated from the neighbor side, which conflicts with our
    # wedges and triggers a PLC error). Leaving the discrete entity
    # topologically free lets us deposit a closed wedge mesh into it
    # without gmsh attempting to remesh it later (we also set
    # Mesh.MeshOnlyEmpty=1 on the way out).
    vol_tag = gmsh.model.addDiscreteEntity(3, -1, [])

    # Add interior layer nodes (layers 1..n_layers-1) to the discrete
    # volume. We must skip nodes whose tag already exists (e.g. matched
    # top-boundary tags reused by layer n -- but layer n is added to
    # top_face below, not here, so for layers 1..n-1 every tag is fresh).
    if n_layers >= 2:
        interior_tags = np.concatenate(new_layer_tags[:-1])
        interior_coords = np.concatenate(new_layer_coords[:-1])
        gmsh.model.mesh.addNodes(
            3,
            vol_tag,
            interior_tags.tolist(),
            interior_coords.reshape(-1).tolist(),
        )

    # Override top face: clear its 2D mesh (NOT its 1D bounding curves --
    # those are shared with surrounding OCC faces) and add only the new
    # interior nodes plus triangles with the bottom's connectivity. The
    # boundary nodes already live on the 1D curves and are reused via
    # top_boundary_lookup above.
    top_layer_tags = new_layer_tags[-1]
    top_layer_coords = new_layer_coords[-1]

    try:
        gmsh.model.mesh.clear([(2, top_face)])
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "mesh.clear failed for top face %d (slab %s): %s",
            top_face,
            slab.physical_name,
            exc,
        )

    interior_mask = ~is_boundary
    if interior_mask.any():
        gmsh.model.mesh.addNodes(
            2,
            top_face,
            top_layer_tags[interior_mask].tolist(),
            top_layer_coords[interior_mask].reshape(-1).tolist(),
        )

    # Build top-face elements by remapping each bottom-cell node tag
    # via tag_to_idx -> top_layer_tags[idx]. Same element type as the
    # bottom (triangle for wedge slabs, quad for hex slabs).
    top_cell_nodes = np.empty_like(triangles, dtype=np.uint64)
    for r in range(triangles.shape[0]):
        for c in range(cells_per_face):
            top_cell_nodes[r, c] = top_layer_tags[tag_to_idx[int(triangles[r, c])]]

    next_elem_tag = int(gmsh.model.mesh.getMaxElementTag()) + 1
    n_cells = triangles.shape[0]
    top_elem_tags = np.arange(next_elem_tag, next_elem_tag + n_cells, dtype=np.uint64)
    next_elem_tag += n_cells
    gmsh.model.mesh.addElements(
        2,
        top_face,
        [top_elem_gmsh_type],
        [top_elem_tags.tolist()],
        [top_cell_nodes.reshape(-1).tolist()],
    )

    # Build 3D elements: for each bottom cell (a, b, c[, d]) and each
    # layer i in 0..n_layers-1, the volume element nodes are
    # [layers[i][a_idx], ..., layers[i+1][a_idx], ...]. Wedges are
    # 6-node ``[lo3 | hi3]``; hexes are 8-node ``[lo4 | hi4]``.
    cell_indices = np.empty_like(triangles, dtype=np.int64)
    for r in range(triangles.shape[0]):
        for c in range(cells_per_face):
            cell_indices[r, c] = tag_to_idx[int(triangles[r, c])]

    volume_node_lists: list[np.ndarray] = []
    for i in range(n_layers):
        lo_tags = layers[i][cell_indices]  # shape (n_cells, cells_per_face)
        hi_tags = layers[i + 1][cell_indices]
        block = np.concatenate([lo_tags, hi_tags], axis=1)
        volume_node_lists.append(block.reshape(-1))
    volume_nodes_flat = np.concatenate(volume_node_lists)
    n_3d = n_cells * n_layers
    volume_tags_arr = np.arange(next_elem_tag, next_elem_tag + n_3d, dtype=np.uint64)

    gmsh.model.mesh.addElements(
        3,
        vol_tag,
        [gmsh_3d_type],
        [volume_tags_arr.tolist()],
        [volume_nodes_flat.tolist()],
    )

    return vol_tag, {
        "layers": layers,
        "triangles": triangles,
        "tag_to_idx": tag_to_idx,
        "coord": coord,
    }


def _find_occ_face_for_slab(
    candidates, slab: Slab, target_z: float, tol: float
) -> int | None:
    """Return tag of an OCC face bounding the slab at ``target_z``.

    The matched face must lie at z = ``target_z`` (within ``tol``) AND its
    xy bounding box must sit substantially inside the slab footprint.
    Stricter than :func:`_find_occ_face_at_z`: verifies the face actually
    bounds *this* slab's void rather than some unrelated z-coplanar OCC
    face elsewhere in the scene.
    """
    import gmsh
    from shapely.geometry import box

    fp = slab.footprint
    fp_area = fp.area
    if fp_area <= 0:
        return None
    best_tag: int | None = None
    best_score = -float("inf")
    bbox_tol = _bbox_tol_for_slab(slab, tol)
    # Require >= 50% of the face's bbox area to lie inside the slab footprint.
    # Stricter than "any overlap" -- prevents picking a neighbor's coplanar
    # face whose edge merely touches the slab footprint along one boundary.
    for dim, tag in candidates:
        if dim != 2:
            continue
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(2, tag)
        if abs(zmin - zmax) > bbox_tol:
            continue
        z_face = 0.5 * (zmin + zmax)
        if abs(z_face - target_z) > bbox_tol:
            continue
        face_box = box(xmin, ymin, xmax, ymax)
        face_box_area = face_box.area
        if face_box_area <= 0:
            continue
        inter = fp.intersection(face_box)
        if inter.is_empty:
            continue
        # Require the face bbox to be substantially inside the slab footprint.
        # A neighbor face that merely shares an edge yields inter.area ~ 0
        # relative to its own bbox area, so the ratio filters it out.
        coverage = inter.area / face_box_area
        if coverage < 0.5:
            continue
        # Prefer the face whose footprint best fills the slab footprint.
        score = inter.area / fp_area
        if score > best_score:
            best_score = score
            best_tag = tag
    return best_tag


def slabs_to_json(slabs: list[Slab]) -> list[dict]:
    """Serialize a slab list to a JSON-safe list of dicts."""
    import shapely.wkt

    return [
        {
            "footprint_wkt": shapely.wkt.dumps(s.footprint, rounding_precision=12),
            "zlo": s.zlo,
            "zhi": s.zhi,
            "n_layers": s.n_layers,
            "recombine": s.recombine,
            "physical_name": list(s.physical_name),
            "source_index": s.source_index,
            "mesh_order": s.mesh_order if s.mesh_order != float("inf") else None,
            "fragment_fuzzy_value": s.fragment_fuzzy_value,
        }
        for s in slabs
    ]


def slabs_from_json(data: list[dict]) -> list[Slab]:
    """Inverse of ``slabs_to_json``."""
    import shapely.wkt

    return [
        Slab(
            footprint=shapely.wkt.loads(d["footprint_wkt"]),
            zlo=d["zlo"],
            zhi=d["zhi"],
            n_layers=d["n_layers"],
            recombine=d["recombine"],
            physical_name=tuple(d["physical_name"]),
            source_index=d["source_index"],
            mesh_order=d["mesh_order"] if d["mesh_order"] is not None else float("inf"),
            fragment_fuzzy_value=d.get("fragment_fuzzy_value"),
        )
        for d in data
    ]
