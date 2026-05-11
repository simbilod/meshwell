"""Structured (gmsh-tutorial-t3-style) layered extrusion for ``PolyPrism``.

This module hosts everything that is structured-specific:

* :class:`_StructuredPolyPrism` -- private subclass of ``PolyPrism`` that
  the ``PolyPrism.__new__`` dispatcher returns when the user passes
  ``n_layers=``. Same user-visible class identity (``isinstance(p, PolyPrism)``
  remains true), but with the structured-mode validation rules.
* :class:`Slab` -- one pairwise-disjoint piece of a structured prism after
  the cascade.
* :class:`_StructuredPhantom` -- CAD-stage proxy that punches a void in
  the OCC fragmentation and is removed so neighbors own the void's faces.
* :func:`apply_structured_slabs` -- the conformal v2 mesh-stage refill.
  Runs ``mesh.generate(2)`` first to give every OCC face a 2D mesh, then
  for each slab builds a layered wedge mesh into a discrete 3D entity.
  Layer 0 reuses the bottom OCC face's existing node tags so the slab is
  conformal at z=zlo by construction; the top OCC face's mesh is then
  overridden with the translated triangulation so the slab is conformal
  at z=zhi too. When a slab has no OCC neighbor on either face (e.g. an
  isolated single-prism scene) we fall back to a geo-kernel extrude that
  ``mesh.generate(3)`` will pick up afterwards.
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


class StructuredLayerMismatchError(ValueError):
    """Raised on conflicting ``n_layers`` across a shared horizontal face.

    Two structured slabs that stack vertically (one's ``zhi`` equals the
    other's ``zlo``) and overlap in xy must agree on ``n_layers`` so the
    shared face has a single, consistent mesh.
    """


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

    Conformal v2 pipeline:

    1. ``mesh.generate(2)`` -- gives every OCC face (including the void's
       bottom/top, owned by neighbors) a triangulated 2D mesh.
    2. For each slab, ``_build_one_slab_conformal`` finds the bottom
       (``zlo``) and top (``zhi``) OCC faces, reads the bottom face's
       2D mesh, and constructs a wedge-element layered mesh in a fresh
       discrete 3D entity. Layer 0 reuses the bottom face's existing
       node tags so the slab is conformal at ``zlo`` by construction;
       the top OCC face's mesh is then overridden with the translated
       triangulation so the slab is conformal at ``zhi`` too.
    3. When a slab has no OCC neighbor on either face (isolated single-
       prism scene), fall back to ``_build_one_slab_geo_fallback`` which
       reuses the v1 geo-kernel extrude. ``mesh.generate(3)`` will pick
       it up afterwards.
    4. Set ``Mesh.MeshOnlyEmpty = 1`` so the subsequent ``mesh.generate(3)``
       does not retouch entities we've already populated.

    Side-face conformality (the slab's vertical faces vs OCC neighbor
    side faces) is not addressed in this iteration; gmsh will mesh those
    lateral neighbor faces independently.
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
    for slab in sorted_slabs:
        _apply_lateral_transfinite_hints(slab, tol)

    # Step 1: mesh all 2D OCC faces so we can read their triangulations.
    gmsh.model.mesh.generate(2)

    # Collect (physical_name -> list of 3D volume tags) across all slabs
    # so we can emit one physical group per name at the end. Slabs that
    # share a physical_name (e.g. a single structured PolyPrism split into
    # multiple sub-slabs by the cascade) must collapse to one group.
    volumes_by_name: dict[str, list[int]] = defaultdict(list)
    geo_fallback_used = False
    # Records used for lateral surface tagging post-pass.
    conformal_records: list[tuple[Slab, dict]] = []
    geo_records: list[tuple[Slab, list[int]]] = []
    for slab in sorted_slabs:
        result = _build_one_slab_conformal(slab, tol)
        if result is None:
            # No OCC neighbor on bottom or top -- fall back to geo extrude.
            vol_tags, used_geo, exposed_faces = _build_one_slab_geo_fallback(slab, tol)
            geo_fallback_used = geo_fallback_used or used_geo
            geo_records.append((slab, exposed_faces))
        else:
            vol_tag, lateral_info = result
            vol_tags = [vol_tag]
            conformal_records.append((slab, lateral_info))
        for name in slab.physical_name:
            volumes_by_name[name].extend(vol_tags)

    if geo_fallback_used:
        gmsh.model.geo.synchronize()

    # Emit one physical group per name.
    for name, vols in volumes_by_name.items():
        if not vols:
            continue
        pg = gmsh.model.addPhysicalGroup(3, vols)
        gmsh.model.setPhysicalName(3, pg, name)

    # Lateral surface tagging: emit ``slab___None`` for facets exposed to
    # the outer domain and ``slabA___slabB`` for structured/structured
    # interfaces. Conformal-path slabs need wedge-facet extraction; geo-
    # fallback slabs already have the side faces as 2D geo entities.
    _emit_structured_lateral_tags(conformal_records, geo_records, tol)


def _emit_structured_lateral_tags(
    conformal_records: list[tuple[Slab, dict]],
    geo_records: list[tuple[Slab, list[int]]],
    tol: float,
) -> None:
    """Tag the lateral surfaces of structured slabs.

    For conformal-path slabs we extract wedge lateral quads from the
    bottom-face triangulation's boundary edges and the per-layer node
    tag arrays. Each quad is keyed by its 3D midpoint so we can detect
    facets shared between two slabs (structured/structured interface)
    vs. exposed to None. Facets covered by a surviving OCC lateral face
    are skipped (the regular ``_tag_physicals`` machinery has already
    tagged them).

    For geo-fallback slabs the lateral side faces are already 2D geo
    entities returned by ``geo.extrude``; we add them straight into a
    discrete physical group named ``slab___None`` (geo-fallback only
    fires for isolated slabs, so we don't expect structured/structured
    interfaces in this path).
    """
    from collections import Counter, defaultdict

    import gmsh

    # Build a single fused dict mapping spatial midpoint key -> claimants
    # (list of (slab_id, quad_node_tags)). Then resolve tags.
    snap = max(tol, 1e-9)

    def _key(x: float, y: float, z: float) -> tuple[int, int, int]:
        return (round(x / snap), round(y / snap), round(z / snap))

    quad_index: dict[
        tuple[int, int, int], list[tuple[int, tuple[int, int, int, int]]]
    ] = defaultdict(list)
    quad_midpoints: dict[
        tuple[int, tuple[int, int, int, int]], tuple[float, float, float]
    ] = {}

    for slab_id, (slab, info) in enumerate(conformal_records):
        triangles = info["triangles"]
        layers = info["layers"]
        tag_to_idx = info["tag_to_idx"]
        coord = info["coord"]  # (n_nodes, 3) with z=zlo for layer 0
        n_layers = slab.n_layers
        height = slab.zhi - slab.zlo

        # Boundary edges of the bottom triangulation: undirected edges
        # appearing in exactly one triangle. Keys are sorted tag pairs.
        edge_counter: Counter[tuple[int, int]] = Counter()
        for tri in triangles.tolist():
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            for x, y in ((a, b), (b, c), (c, a)):
                edge_counter[tuple(sorted((x, y)))] += 1
        boundary_edges = [e for e, c in edge_counter.items() if c == 1]

        for a_tag, b_tag in boundary_edges:
            ia = tag_to_idx[a_tag]
            ib = tag_to_idx[b_tag]
            xa, ya = float(coord[ia, 0]), float(coord[ia, 1])
            xb, yb = float(coord[ib, 0]), float(coord[ib, 1])
            xy_mid_x = 0.5 * (xa + xb)
            xy_mid_y = 0.5 * (ya + yb)
            for i in range(n_layers):
                z_lo = slab.zlo + height * (i / n_layers)
                z_hi = slab.zlo + height * ((i + 1) / n_layers)
                z_mid = 0.5 * (z_lo + z_hi)
                lo = layers[i]
                hi = layers[i + 1]
                quad = (
                    int(lo[ia]),
                    int(lo[ib]),
                    int(hi[ib]),
                    int(hi[ia]),
                )
                key = _key(xy_mid_x, xy_mid_y, z_mid)
                quad_index[key].append((slab_id, quad))
                quad_midpoints[(slab_id, quad)] = (xy_mid_x, xy_mid_y, z_mid)

    # Collect surviving OCC lateral faces (those with z-extent > tol).
    import logging

    logger = logging.getLogger(__name__)
    occ_lateral_bboxes: list[tuple[float, float, float, float, float, float]] = []
    for dim, ftag in gmsh.model.occ.getEntities(2):
        if dim != 2:
            continue
        try:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(2, ftag)
        except Exception as exc:
            logger.debug("getBoundingBox failed for OCC face %d: %s", ftag, exc)
            continue
        if zmax - zmin > snap:  # lateral, not horizontal
            occ_lateral_bboxes.append((xmin, ymin, zmin, xmax, ymax, zmax))

    def _occ_lateral_covers(midpoint: tuple[float, float, float]) -> bool:
        x, y, z = midpoint
        for xmin, ymin, zmin, xmax, ymax, zmax in occ_lateral_bboxes:
            if (
                xmin - snap <= x <= xmax + snap
                and ymin - snap <= y <= ymax + snap
                and zmin - snap <= z <= zmax + snap
            ):
                return True
        return False

    # Group quads by physical-group tag name.
    quads_by_tag: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    for claimants in quad_index.values():
        if len(claimants) >= 2:
            # Structured/structured interface: pick one representative
            # quad (post-removeDuplicateNodes the node tags merge anyway).
            (sid_a, quad_a), (sid_b, _) = claimants[0], claimants[1]
            name_a = conformal_records[sid_a][0].physical_name[0]
            name_b = conformal_records[sid_b][0].physical_name[0]
            if name_a == name_b:
                tag = f"{name_a}___{name_a}"
            else:
                lo_name, hi_name = sorted([name_a, name_b])
                tag = f"{lo_name}___{hi_name}"
            quads_by_tag[tag].append(quad_a)
        else:
            sid, quad = claimants[0]
            midpoint = quad_midpoints[(sid, quad)]
            if _occ_lateral_covers(midpoint):
                # Already tagged by regular _tag_physicals via the OCC face.
                continue
            name = conformal_records[sid][0].physical_name[0]
            quads_by_tag[f"{name}___None"].append(quad)

    # Materialize: one discrete 2D entity per tag, then a physical group.
    for tag, quads in quads_by_tag.items():
        if not quads:
            continue
        ent_tag = gmsh.model.addDiscreteEntity(2)
        next_elem = int(gmsh.model.mesh.getMaxElementTag()) + 1
        elem_tags = list(range(next_elem, next_elem + len(quads)))
        flat_nodes: list[int] = []
        for q in quads:
            flat_nodes.extend(q)
        gmsh.model.mesh.addElements(
            2, ent_tag, [3], [elem_tags], [flat_nodes]
        )  # gmsh element type 3 = 4-node quad
        pg = gmsh.model.addPhysicalGroup(2, [ent_tag])
        gmsh.model.setPhysicalName(2, pg, tag)

    # Geo-fallback slabs: tag exposed 2D geo faces (lateral + bottom/top
    # when not embedded in an OCC neighbor) directly.
    for slab, exposed_faces in geo_records:
        if not exposed_faces:
            continue
        name = slab.physical_name[0]
        pg = gmsh.model.addPhysicalGroup(2, exposed_faces)
        gmsh.model.setPhysicalName(2, pg, f"{name}___None")


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
        # Lateral face: spans full slab z-range.
        if abs(zmin - slab.zlo) > tol or abs(zmax - slab.zhi) > tol:
            continue
        if abs(zmax - zmin) < tol:
            continue  # horizontal face (bottom/top)
        # xy-bbox must touch the slab footprint boundary.
        if xmax - xmin < tol and ymax - ymin < tol:
            continue
        face_xy_box = sh_box(xmin, ymin, xmax, ymax)
        try:
            if not boundary.intersects(face_xy_box):
                continue
        except Exception as exc:
            logger.debug(
                "shapely boundary.intersects failed for face %d: %s", ftag, exc
            )
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
            if (
                abs(czmin - slab.zlo) < tol
                and abs(czmax - slab.zhi) < tol
                and abs(cxmax - cxmin) < tol
                and abs(cymax - cymin) < tol
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


def _build_one_slab_conformal(slab: Slab, tol: float) -> tuple[int, dict] | None:
    """Build a discrete 3D entity for ``slab`` with conformal bottom/top.

    Returns ``(vol_tag, lateral_info)`` where ``lateral_info`` is a dict
    with keys ``layers`` (list of np.ndarray of node tags per layer 0..n),
    ``triangles`` (np.ndarray of bottom-face triangle connectivity in tag
    space), ``tag_to_idx`` (bottom-node-tag -> index), and ``coord`` (the
    bottom-face node coordinates at z=zlo). Returns ``None`` when neither
    the bottom nor the top OCC face was found (caller should use the
    geo-fallback).
    """
    import logging

    import gmsh
    import numpy as np

    logger = logging.getLogger(__name__)

    occ_faces = gmsh.model.occ.getEntities(2)
    bottom_face = _find_occ_face_for_slab(occ_faces, slab, slab.zlo, tol)
    top_face = _find_occ_face_for_slab(occ_faces, slab, slab.zhi, tol)

    if bottom_face is None or top_face is None:
        return None

    # Read bottom face's 2D mesh (interior + boundary nodes).
    node_tags, coord, _ = gmsh.model.mesh.getNodes(2, bottom_face, includeBoundary=True)
    if len(node_tags) == 0:
        logger.warning(
            "Bottom OCC face %d for slab %s at z=%g has no 2D mesh nodes; "
            "skipping conformal build for this slab.",
            bottom_face,
            slab.physical_name,
            slab.zlo,
        )
        return None
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

    elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(2, bottom_face)
    tri_nodes_flat: np.ndarray | None = None
    for et, en in zip(elem_types, elem_node_tags):
        if int(et) == 2:  # triangle
            tri_nodes_flat = np.asarray(en, dtype=np.uint64)
            break
    if tri_nodes_flat is None or tri_nodes_flat.size == 0:
        logger.warning(
            "Bottom OCC face %d for slab %s at z=%g has no triangles; "
            "skipping conformal build for this slab.",
            bottom_face,
            slab.physical_name,
            slab.zlo,
        )
        return None
    triangles = tri_nodes_flat.reshape(-1, 3)

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

    # Build top-face triangles by remapping each bottom-triangle node tag
    # via tag_to_idx -> top_layer_tags[idx].
    top_tri_nodes = np.empty_like(triangles, dtype=np.uint64)
    for r in range(triangles.shape[0]):
        for c in range(3):
            top_tri_nodes[r, c] = top_layer_tags[tag_to_idx[int(triangles[r, c])]]

    next_elem_tag = int(gmsh.model.mesh.getMaxElementTag()) + 1
    n_tri = triangles.shape[0]
    top_tri_tags = np.arange(next_elem_tag, next_elem_tag + n_tri, dtype=np.uint64)
    next_elem_tag += n_tri
    gmsh.model.mesh.addElements(
        2,
        top_face,
        [2],  # triangle type
        [top_tri_tags.tolist()],
        [top_tri_nodes.reshape(-1).tolist()],
    )

    # Build wedges: for each bottom triangle (a, b, c) and each layer
    # i in 0..n_layers-1, wedge nodes are
    # [layers[i][a_idx], layers[i][b_idx], layers[i][c_idx],
    #  layers[i+1][a_idx], layers[i+1][b_idx], layers[i+1][c_idx]].
    # We build idx triplets once and index into each layer's tag array.
    tri_indices = np.empty_like(triangles, dtype=np.int64)
    for r in range(triangles.shape[0]):
        for c in range(3):
            tri_indices[r, c] = tag_to_idx[int(triangles[r, c])]

    wedge_node_lists: list[np.ndarray] = []
    for i in range(n_layers):
        lo_tags = layers[i][tri_indices]  # shape (n_tri, 3)
        hi_tags = layers[i + 1][tri_indices]
        # interleave: 6 nodes per wedge in [lo3 | hi3] order
        block = np.concatenate([lo_tags, hi_tags], axis=1)  # shape (n_tri, 6)
        wedge_node_lists.append(block.reshape(-1))
    wedge_nodes_flat = np.concatenate(wedge_node_lists)
    n_wedges = n_tri * n_layers
    wedge_tags = np.arange(next_elem_tag, next_elem_tag + n_wedges, dtype=np.uint64)

    gmsh.model.mesh.addElements(
        3,
        vol_tag,
        [6],  # wedge type
        [wedge_tags.tolist()],
        [wedge_nodes_flat.tolist()],
    )

    return vol_tag, {
        "layers": layers,
        "triangles": triangles,
        "tag_to_idx": tag_to_idx,
        "coord": coord,
    }


def _build_one_slab_geo_fallback(
    slab: Slab, tol: float
) -> tuple[list[int], bool, list[int]]:
    """Build geo-kernel replicas (the v1 path) for slabs with no OCC neighbors.

    Returns (volume_tags, used_flag, exposed_face_tags) where used_flag
    is True if any geo work was actually done (caller uses it to decide
    whether to call ``geo.synchronize`` afterwards) and exposed_face_tags
    is the list of 2D geo entity tags for faces NOT mated to a surviving
    OCC neighbor (caller emits ``slab___None`` for them). The bottom and
    top are included only when no OCC face at the corresponding z covers
    the slab footprint; lateral sides are always exposed in this path
    (geo-fallback only fires when at least one of bottom/top is absent,
    and in those scenes the lateral OCC faces are also orphan-removed).
    """
    import gmsh

    polys = (
        slab.footprint.geoms if hasattr(slab.footprint, "geoms") else [slab.footprint]
    )
    height = slab.zhi - slab.zlo
    volume_tags: list[int] = []
    exposed_face_tags: list[int] = []

    # Determine OCC neighbor presence ONCE before adding geo bodies, so
    # the surviving OCC faces are the only ones in the candidate list.
    occ_face_candidates = gmsh.model.occ.getEntities(2)
    has_bottom_neighbor = (
        _find_occ_face_for_slab(occ_face_candidates, slab, slab.zlo, tol) is not None
    )
    has_top_neighbor = (
        _find_occ_face_for_slab(occ_face_candidates, slab, slab.zhi, tol) is not None
    )

    for poly in polys:
        loops = [_geo_curve_loop(list(poly.exterior.coords)[:-1], slab.zlo)]
        loops.extend(
            _geo_curve_loop(list(interior.coords)[:-1], slab.zlo)
            for interior in poly.interiors
        )
        bottom_surface = gmsh.model.geo.addPlaneSurface(loops)

        extruded = gmsh.model.geo.extrude(
            [(2, bottom_surface)],
            0,
            0,
            height,
            numElements=[slab.n_layers],
            heights=[1.0],
            recombine=slab.recombine,
        )
        # extruded layout: [(2, top), (3, vol), (2, side_0), (2, side_1), ...]
        volume_dt = next(dt for dt in extruded if dt[0] == 3)
        volume_tags.append(volume_dt[1])

        face_dts = [dt for dt in extruded if dt[0] == 2]
        top_face = face_dts[0][1]
        side_face_tags = [t for _, t in face_dts[1:]]

        # Always tag lateral sides as exposed (see docstring).
        exposed_face_tags.extend(side_face_tags)
        # Bottom and top only when no covering OCC face exists.
        if not has_bottom_neighbor:
            exposed_face_tags.append(bottom_surface)
        if not has_top_neighbor:
            exposed_face_tags.append(top_face)

        gmsh.model.geo.synchronize()

        # If any OCC neighbors do exist (e.g. only one of bottom/top is
        # missing), still try to embed the geo face into them so the
        # available shared face is at least mated kernel-to-kernel.
        _embed_geo_into_occ_neighbors(
            geo_bottom=bottom_surface,
            geo_top=top_face,
            slab=slab,
            tol=tol,
        )
    return volume_tags, True, exposed_face_tags


def _geo_curve_loop(coords, z: float) -> int:
    """Build a closed curve loop in the geo kernel at given z."""
    import gmsh

    pts = [gmsh.model.geo.addPoint(x, y, z) for x, y in coords]
    lines = [gmsh.model.geo.addLine(a, b) for a, b in pairwise(pts)]
    lines.append(gmsh.model.geo.addLine(pts[-1], pts[0]))
    return gmsh.model.geo.addCurveLoop(lines)


def _embed_geo_into_occ_neighbors(
    geo_bottom: int,
    geo_top: int,
    slab: Slab,
    tol: float,
) -> None:
    """Best-effort embed for geo-fallback path when one OCC face exists.

    Lateral side embedding is intricate and out of scope; we only mate
    the bottom/top faces.
    """
    import logging

    import gmsh

    logger = logging.getLogger(__name__)
    occ_faces = gmsh.model.occ.getEntities(2)
    for geo_face_tag, expected_z in [
        (geo_bottom, slab.zlo),
        (geo_top, slab.zhi),
    ]:
        match = _find_occ_face_at_z(occ_faces, expected_z, tol)
        if match is None:
            continue
        try:
            gmsh.model.mesh.embed(2, [geo_face_tag], 2, match)
        except Exception as exc:
            logger.warning(
                "Failed to embed structured-slab geo face %d into OCC face %d "
                "for slab %s at z=%g: %s. Mesh may be non-conformal at this "
                "interface.",
                geo_face_tag,
                match,
                slab.physical_name,
                expected_z,
                exc,
            )


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
    # Require >= 50% of the face's bbox area to lie inside the slab footprint.
    # Stricter than "any overlap" -- prevents picking a neighbor's coplanar
    # face whose edge merely touches the slab footprint along one boundary.
    for dim, tag in candidates:
        if dim != 2:
            continue
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(2, tag)
        if abs(zmin - zmax) > tol:
            continue
        z_face = 0.5 * (zmin + zmax)
        if abs(z_face - target_z) > tol:
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


def _find_occ_face_at_z(candidates, target_z: float, tol: float) -> int | None:
    """Return tag of an OCC face whose centroid z is within tol of target_z.

    Uses ``occ.getBoundingBox`` (only works on OCC entities). The
    candidate list is from ``occ.getEntities(2)`` so all candidates are
    OCC-side; geo faces are excluded by construction.
    """
    import gmsh

    best_tag: int | None = None
    best_delta = float("inf")
    for dim, tag in candidates:
        if dim != 2:
            continue
        _xmin, _ymin, zmin, _xmax, _ymax, zmax = gmsh.model.occ.getBoundingBox(2, tag)
        if abs(zmin - zmax) > tol:
            continue  # not an axis-aligned constant-z face
        z_face = 0.5 * (zmin + zmax)
        delta = abs(z_face - target_z)
        if delta < tol and delta < best_delta:
            best_tag = tag
            best_delta = delta
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
        )
        for d in data
    ]
