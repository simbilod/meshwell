"""End-to-end driver for the structured pre-pass and post-pass.

Pre-pass: collect → cohort → validate z-stacks → decompose → build →
swap entities. Returns a StructuredState that the orchestrator
threads forward to the cad_occ and meshing stages.

Post-pass: expand cohort OCCLabeledEntity into per-sub-solid
entities + record post-BOP face ShapeKeys.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import shapely

from meshwell.structured.build import (
    EdgeRegistry,
    FaceRegistry,
    VertexRegistry,
    build_cohort_compound,
)
from meshwell.structured.cohort import build_cohorts
from meshwell.structured.cohort_entity import _CohortEntity
from meshwell.structured.collect import collect_structured_slabs
from meshwell.structured.decompose import decompose_cohorts
from meshwell.structured.types import ShapeKey, SlabMeta
from meshwell.structured.validators import (
    validate_no_volumetric_cohort_overlap,
    validate_z_stacks,
)


@dataclass
class StructuredState:
    """Threaded between pre-pass, cad_occ, and post-pass.

    ``face_name_by_key`` and ``sub_solid_name_by_key`` map pre-BOP
    ShapeKeys to synthetic physical-group names assigned during the
    pre-pass. These names are written into the XAO at post-pass time
    via synthetic dim=2 / dim=3 OCCLabeledEntities so the orchestrator
    can recover ShapeKey -> gmsh-tag bindings by name lookup after
    XAO load (replaces the centroid+bbox fingerprint matcher).
    """

    entities_out: list[Any]
    slab_meta: dict[ShapeKey, SlabMeta] = field(default_factory=dict)
    cohort_entities: list[_CohortEntity] = field(default_factory=list)
    face_name_by_key: dict[ShapeKey, str] = field(default_factory=dict)
    sub_solid_name_by_key: dict[ShapeKey, str] = field(default_factory=dict)

    # Per-cohort (VertexRegistry, EdgeRegistry, FaceRegistry) triples,
    # indexed by cohort_index. Constructed in structured_pre_pass and
    # threaded into build_cohort_compound so the cohort's pre-baked TShapes
    # (sub-piece bot/top faces, lateral edges) are shared by identity within
    # each cohort. Aggregated here for diagnostic access; no downstream
    # stage reads this field directly.
    cohort_registries: list[tuple[VertexRegistry, EdgeRegistry, FaceRegistry]] = field(
        default_factory=list
    )


def structured_pre_pass(
    entities: list[Any],
    point_tolerance: float,
) -> StructuredState:
    """Run Stages 1-4 and return entities_out for cad_occ.

    If no structured entities are present, returns the input list
    unchanged with an empty slab_meta.
    """
    structured_slabs, unstructured = collect_structured_slabs(entities)
    if not structured_slabs:
        return StructuredState(entities_out=entities)
    # Snap every structured slab footprint to the point_tolerance grid.
    # Without this, the cad_common.prepare_entities intake perturbation
    # (default 1e-5 outward buffer applied BEFORE this pre-pass) turns
    # laterally-touching polygons into polygons that overlap by ~2e-5.
    # The cohort decomposition then produces sliver subpieces whose
    # endpoints all dedupe to the same vertex (point_tolerance ~ 1e-3),
    # which makes BRepBuilderAPI_MakeEdge raise StdFail_NotDone. Snap
    # restores the original touch geometry while keeping the buffered
    # XY available to cad_occ for the BOP step.
    structured_slabs = [
        dataclasses.replace(
            s,
            footprint=shapely.set_precision(
                s.footprint, grid_size=point_tolerance, mode="valid_output"
            ),
        )
        for s in structured_slabs
    ]
    cohorts = build_cohorts(structured_slabs)
    validate_z_stacks(cohorts, entities)
    validate_no_volumetric_cohort_overlap(cohorts, entities)
    # decompose_cohorts returns the unstructured list unchanged (third slot).
    subpieces_per_cohort, unstructured_out, arrangements = decompose_cohorts(
        cohorts, unstructured, point_tolerance=point_tolerance
    )

    cohort_entities: list[_CohortEntity] = []
    all_slab_meta: dict[ShapeKey, SlabMeta] = {}
    face_name_by_key: dict[ShapeKey, str] = {}
    sub_solid_name_by_key: dict[ShapeKey, str] = {}
    cohort_registries: list[tuple[VertexRegistry, EdgeRegistry, FaceRegistry]] = []
    for ci, (cohort, subs) in enumerate(zip(cohorts, subpieces_per_cohort)):
        vreg = VertexRegistry(point_tolerance=point_tolerance)
        ereg = EdgeRegistry(vertices=vreg, point_tolerance=point_tolerance)
        freg = FaceRegistry(edges=ereg, point_tolerance=point_tolerance)
        cohort_registries.append((vreg, ereg, freg))
        compound, slab_meta = build_cohort_compound(
            cohort,
            subs,
            point_tolerance,
            vertex_registry=vreg,
            edge_registry=ereg,
            face_registry=freg,
            arrangement=arrangements[ci],
        )
        ce = _CohortEntity(
            compound=compound,
            slab_meta=slab_meta,
            cohort=cohort,
            cohort_index=ci,
        )
        cohort_entities.append(ce)
        all_slab_meta.update(slab_meta)
        # Assign synthetic per-face / per-sub-solid names. The orchestrator
        # writes these into the XAO via synthetic 2D / 3D entities so it
        # can recover ShapeKey -> gmsh-tag mappings by name lookup after
        # XAO load.
        # MANUAL_NOTE: investigate alternative, e.g. brep sidecar w/
        # deterministic import
        for si, (sub_key, meta) in enumerate(slab_meta.items()):
            sub_solid_name_by_key[sub_key] = f"__cohort_{ci}__slab_{si}"
            face_name_by_key[meta.bot_face_key] = f"__cohort_{ci}__slab_{si}__bot"
            face_name_by_key[meta.top_face_key] = f"__cohort_{ci}__slab_{si}__top"
            for li, lk in enumerate(meta.lateral_face_keys):
                face_name_by_key[lk] = f"__cohort_{ci}__slab_{si}__lat_{li}"

    entities_out = cohort_entities + unstructured_out
    return StructuredState(
        entities_out=entities_out,
        slab_meta=all_slab_meta,
        cohort_entities=cohort_entities,
        face_name_by_key=face_name_by_key,
        sub_solid_name_by_key=sub_solid_name_by_key,
        cohort_registries=cohort_registries,
    )


def structured_post_pass(
    occ_entities: list,
    state: StructuredState,
) -> list:
    """Expand every cohort OCCLabeledEntity into per-sub-solid entities.

    Matches each surviving post-BOP shape to its slab_meta entry by
    ShapeKey (fast path) or by bounding-box fingerprint (fallback for
    the multi-cohort case where BOPAlgo_Builder regenerates TShape IDs
    even for geometrically unchanged shapes).

    One OCCLabeledEntity per sub-solid, carrying the source slab's
    physical_name and a synthetic index.

    Additionally emits synthetic dim=2 OCCLabeledEntities for every
    tracked cohort face (bot / top / lateral) carrying the synthetic
    ``__cohort_<ci>__slab_<si>__<role>`` name from
    ``state.face_name_by_key``. Each sub-solid entity gets its synthetic
    ``__cohort_<ci>__slab_<si>`` name appended to its ``physical_name``
    tuple. These synthetic names give the orchestrator a stable lookup
    from pre-BOP ShapeKey to post-XAO-load gmsh entity tag.
    """
    from OCP.TopAbs import TopAbs_FACE, TopAbs_ShapeEnum, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    from meshwell.cad_occ import OCCLabeledEntity
    from meshwell.structured.build import _shape_key

    # Build a bbox-keyed lookup from pre-BOP slab ShapeKeys.
    # Used as fallback when the ShapeKey doesn't survive BOP.
    slab_fp_by_key = _build_slab_fingerprints(state)
    # Pre-BOP face bbox -> ShapeKey for the fallback face match.
    face_fp_by_key = _build_face_fingerprints(state)

    def _iter_solids(shape):
        """Yield individual solids from a shape (unwrap compound if needed)."""
        if shape.ShapeType() == TopAbs_ShapeEnum.TopAbs_COMPOUND:
            exp = TopExp_Explorer(shape, TopAbs_SOLID)
            while exp.More():
                yield exp.Current()
                exp.Next()
        else:
            yield shape

    expanded: list = []
    next_index = max((e.index for e in occ_entities), default=-1) + 1
    cohort_pnames = {ce.physical_name for ce in state.cohort_entities}

    # Walk every post-BOP cohort sub-solid and emit:
    #   - one dim=3 OCCLabeledEntity per sub-solid, with the source
    #     slab's physical name PLUS the synthetic
    #     __cohort_<ci>__slab_<si> name.
    #   - one dim=2 OCCLabeledEntity per (slab, face_role) tracked in
    #     SlabMeta — driven by the meta's per-role face keys (bot, top,
    #     lateral_<i>), so two slabs that share a lateral face TShape
    #     each emit their OWN synthetic group; post-XAO-load both names
    #     resolve to the same gmsh entity tag (the merged face), which
    #     is what the lateral-n_layers-mismatch check needs.
    for ent in occ_entities:
        if ent.physical_name not in cohort_pnames:
            expanded.append(ent)
            continue
        for raw_shape in ent.shapes:
            for shape in _iter_solids(raw_shape):
                key = _shape_key(shape)
                meta = state.slab_meta.get(key)
                if meta is None:
                    # Fast-path miss: BOPAlgo_Builder regenerated the TShape.
                    # Fall back to spatial fingerprint matching.
                    meta = _match_by_bbox(shape, slab_fp_by_key, state.slab_meta)
                if meta is None:
                    # Leftover fragment from a BOP boundary split with no
                    # matching slab. Represent it as-is under the cohort name
                    # so it still gets meshed. (Voids ARE matched here — they
                    # are first-class slab_meta entries with keep=False.)
                    expanded.append(_copy_with(ent, [shape], next_index))
                    next_index += 1
                    continue
                # Recover the SlabMeta's pre-BOP sub-solid ShapeKey so we
                # can look up the synthetic name. If we hit the fast path,
                # ``key`` is the meta's own key. If we fell back to bbox,
                # we need the original slab_meta key.
                sub_key = (
                    key
                    if key in state.slab_meta
                    else _find_slab_key(state.slab_meta, meta)
                )
                sub_solid_name = state.sub_solid_name_by_key.get(sub_key)
                names: tuple[str, ...] = meta.physical_name
                if sub_solid_name is not None:
                    names = (*meta.physical_name, sub_solid_name)
                sub_ent = OCCLabeledEntity(
                    shapes=[shape],
                    physical_name=names,
                    index=next_index,
                    keep=meta.keep,
                    dim=3,
                    mesh_order=ent.mesh_order,
                )
                expanded.append(sub_ent)
                next_index += 1

                # Pre-compute (z_centroid, z_extent, face) tuples for
                # every face in this sub-solid so the per-role match
                # below is O(role_count) without re-walking the solid.
                solid_faces: list = []
                exp_f = TopExp_Explorer(shape, TopAbs_FACE)
                seen_face_keys: set[ShapeKey] = set()
                while exp_f.More():
                    f = exp_f.Current()
                    fk_ = _shape_key(f)
                    if fk_ not in seen_face_keys:
                        seen_face_keys.add(fk_)
                        solid_faces.append((fk_, f))
                    exp_f.Next()

                # Emit one synthetic 2D entity per (slab, role) in the
                # SlabMeta. Match by ShapeKey first (fast path when the
                # face TShape survived BOP) then by bbox fallback.
                roles: list[tuple[str, ShapeKey]] = [
                    ("bot", meta.bot_face_key),
                    ("top", meta.top_face_key),
                ] + [(f"lat_{i}", lk) for i, lk in enumerate(meta.lateral_face_keys)]
                pre_fp_for_meta = {
                    rk: face_fp_by_key[rk] for _r, rk in roles if rk in face_fp_by_key
                }
                for _role, fk in roles:
                    face_name = state.face_name_by_key.get(fk)
                    if face_name is None:
                        continue
                    matched = _match_role_face(fk, pre_fp_for_meta, solid_faces)
                    if matched is None:
                        continue
                    face_ent = OCCLabeledEntity(
                        shapes=[matched],
                        physical_name=(face_name,),
                        index=next_index,
                        keep=True,
                        dim=2,
                        mesh_order=None,
                    )
                    expanded.append(face_ent)
                    next_index += 1
    return expanded


def _build_slab_fingerprints(
    state: StructuredState,
) -> dict[ShapeKey, tuple[float, ...]]:
    """Return bbox fingerprint (xmin,ymin,zmin,xmax,ymax,zmax) per slab ShapeKey.

    Walks the pre-BOP cohort compounds to compute bounding boxes for
    every solid whose ShapeKey is in slab_meta.
    """
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    from meshwell.structured.build import _shape_key

    result: dict[ShapeKey, tuple[float, ...]] = {}
    for ce in state.cohort_entities:
        exp = TopExp_Explorer(ce.compound, TopAbs_SOLID)
        while exp.More():
            solid = exp.Current()
            sk = _shape_key(solid)
            if sk in state.slab_meta:
                box = Bnd_Box()
                BRepBndLib.Add_s(solid, box)
                if not box.IsVoid():
                    result[sk] = box.Get()
            exp.Next()
    return result


def _build_face_fingerprints(
    state: StructuredState,
) -> dict[ShapeKey, tuple[float, ...]]:
    """Return bbox fingerprint per pre-BOP face ShapeKey (tracked faces only)."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer

    from meshwell.structured.build import _shape_key

    result: dict[ShapeKey, tuple[float, ...]] = {}
    for ce in state.cohort_entities:
        exp = TopExp_Explorer(ce.compound, TopAbs_FACE)
        while exp.More():
            face = exp.Current()
            fk = _shape_key(face)
            if fk in state.face_name_by_key and fk not in result:
                box = Bnd_Box()
                BRepBndLib.Add_s(face, box)
                if not box.IsVoid():
                    result[fk] = box.Get()
            exp.Next()
    return result


def _match_role_face(
    role_key: "ShapeKey",
    pre_fp_for_meta: "dict[ShapeKey, tuple[float, ...]]",
    solid_faces: "list[tuple[ShapeKey, Any]]",
    tol: float = 1e-3,
):
    """Find the post-BOP TopoDS_Face whose bbox best matches a SlabMeta role key.

    Args:
        role_key: Pre-BOP ShapeKey for one role (bot/top/lateral_i).
        pre_fp_for_meta: ``{role_key: bbox}`` for the SlabMeta's own roles.
        solid_faces: ``[(post_bop_key, post_bop_face)]`` pairs from the
            sub-solid's exploration.
        tol: Per-axis tolerance.

    Returns the post-BOP face whose bbox matches ``role_key``'s pre-BOP
    bbox within ``tol`` on every axis (closest L1 distance breaks ties).
    Falls back to fast-path key match when the post-BOP key still equals
    the pre-BOP key.
    """
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    # Fast path: same TShape pointer survived BOP.
    for fk, face in solid_faces:
        if fk == role_key:
            return face

    target = pre_fp_for_meta.get(role_key)
    if target is None:
        return None

    best_face = None
    best_d = float("inf")
    for _fk, face in solid_faces:
        box = Bnd_Box()
        BRepBndLib.Add_s(face, box)
        if box.IsVoid():
            continue
        g = box.Get()
        if not all(abs(g[i] - target[i]) <= tol for i in range(6)):
            continue
        d = sum(abs(g[i] - target[i]) for i in range(6))
        if d < best_d:
            best_d = d
            best_face = face
    return best_face


def _find_slab_key(
    slab_meta: "dict[ShapeKey, SlabMeta]",
    meta: "SlabMeta",
) -> "ShapeKey | None":
    """Reverse-lookup the ShapeKey whose value is ``meta`` in slab_meta."""
    for sk, m in slab_meta.items():
        if m is meta:
            return sk
    return None


def _match_by_bbox(
    shape,
    slab_fp_by_key: dict[ShapeKey, tuple[float, ...]],
    slab_meta: "dict[ShapeKey, SlabMeta]",
    tol: float = 1e-3,
) -> "SlabMeta | None":
    """Match a post-BOP solid to a slab_meta entry by bounding box.

    Returns the SlabMeta whose pre-BOP bbox agrees with ``shape``'s
    bbox within ``tol`` on every axis, or None if no match.
    """
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    if box.IsVoid():
        return None
    g = box.Get()  # (xmin, ymin, zmin, xmax, ymax, zmax)

    best_key: "ShapeKey | None" = None
    best_volume_overlap = -1.0
    for sk, fp in slab_fp_by_key.items():
        # Check bbox corner agreement within tolerance.
        if all(abs(g[i] - fp[i]) <= tol for i in range(6)):
            # Volume of bbox overlap (approximation of "most similar").
            overlap_vol = (
                (min(g[3], fp[3]) - max(g[0], fp[0]))
                * (min(g[4], fp[4]) - max(g[1], fp[1]))
                * (min(g[5], fp[5]) - max(g[2], fp[2]))
            )
            if overlap_vol > best_volume_overlap:
                best_volume_overlap = overlap_vol
                best_key = sk

    if best_key is not None:
        return slab_meta.get(best_key)
    return None


def _copy_with(ent, shapes, idx: int):
    from meshwell.cad_occ import OCCLabeledEntity

    return OCCLabeledEntity(
        shapes=list(shapes),
        physical_name=ent.physical_name,
        index=idx,
        keep=ent.keep,
        dim=ent.dim,
        mesh_order=ent.mesh_order,
    )
