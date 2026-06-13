"""Unified CAD -> XAO -> mesh pipeline."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gmsh

from meshwell.cad_common import prepare_entities
from meshwell.cad_occ import cad_occ
from meshwell.mesh import mesh
from meshwell.model import ModelManager
from meshwell.occ_xao_writer import default_interface_aabb_tolerance
from meshwell.structured.pipeline import (
    StructuredState,
    structured_post_pass,
    structured_pre_pass,
)
from meshwell.structured.types import ShapeKey
from meshwell.structured.validators import validate_cohort_shells
from meshwell.structured.wedge import (
    freeze_lateral_mesh,
    stamp_wedges,
)
from meshwell.utils import deserialize

# Tight gmsh geometry tolerance used while deduplicating mesh nodes, so
# distinct-but-close points (e.g. on adjacent fine curves) are not merged.
_DEDUP_GEOMETRY_TOLERANCE = 1e-6


def _remove_duplicate_nodes_tight(dimtags: list[tuple[int, int]] | None = None) -> None:
    """Run ``removeDuplicateNodes`` under a tightened geometry tolerance.

    Pass ``dimtags`` to scope the dedup to specific entities, or ``None``
    for a global pass. The previous ``Geometry.Tolerance`` is restored.
    """
    old_tol = gmsh.option.getNumber("Geometry.Tolerance")
    gmsh.option.setNumber("Geometry.Tolerance", _DEDUP_GEOMETRY_TOLERANCE)
    try:
        if dimtags is None:
            gmsh.model.mesh.removeDuplicateNodes()
        else:
            gmsh.model.mesh.removeDuplicateNodes(dimtags)
    finally:
        gmsh.option.setNumber("Geometry.Tolerance", old_tol)


def generate_mesh(
    entities: list[Any],
    dim: int,
    output_mesh: Path | str | None = None,
    checkpoint_cad: Path | str | None = None,
    registry: dict[str, Callable[..., Any]] | None = None,
    backend: str | None = None,  # deprecated
    **mesh_kwargs,
) -> Any:
    """Generate a mesh from a list of entities.

    Pipeline: structured pre-pass -> ``cad_occ`` fragments -> structured
    post-pass -> :func:`write_xao` serializes to a tagged XAO -> gmsh load
    -> structured pre-2D/pre-3D wedge hooks -> :func:`mesh`.

    Args:
        entities: List of meshwell entities or their dictionary representations.
        dim: Dimension of the mesh to generate.
        output_mesh: Optional path to save the generated mesh (.msh).
        checkpoint_cad: Optional path to save the CAD state (.xao).
        registry: Optional registry for ``OCC_entity`` function resolution.
        backend: Deprecated; only ``"occ"`` or ``None`` is accepted.
        **mesh_kwargs: Additional arguments forwarded to :func:`mesh`,
            plus a few CAD-side kwargs consumed here:

            - ``progress_bars`` (bool): status output during the bridge.
            - ``cut_fuzzy_value`` (float): ``BRepAlgoAPI_Cut`` fuzzy
              for the sequential per-entity cut cascade.
            - ``fragment_fuzzy_value`` (float): ``BOPAlgo_Builder``
              fuzzy for the all-fragment pass.
            - ``canonicalize_topology`` (bool): run the OCP post-fragment
              TShape canonicalization pass.
            - ``remove_all_duplicates`` (bool, default ``False``):
              gmsh-level fragment safety net after XAO load. Opt-in;
              only catches OCC-identical coincident TShapes.
            - ``interface_delimiter``, ``boundary_delimiter``: XAO group
              name delimiters.
            - ``pre_2d_hook`` / ``pre_3d_hook`` (callables): composed with
              the structured wedge hooks (run after the structured pass),
              not replacing them.

    Returns:
        meshio.Mesh: The generated mesh object (or ``None`` if
        ``output_mesh`` is provided and the mesh pipeline does not
        return in-memory).
    """
    if backend is not None and backend != "occ":
        raise ValueError(
            f"backend={backend!r} is no longer supported. "
            "Meshwell now uses OCC exclusively for CAD."
        )

    entities = deserialize(entities, registry=registry)

    # --- Stage 1: OCC fragmentation (cad_occ kwargs). -------------------
    cad_kwargs: dict[str, Any] = {}
    if "n_threads" in mesh_kwargs:
        cad_kwargs["n_threads"] = mesh_kwargs["n_threads"]
    if "point_tolerance" in mesh_kwargs:
        cad_kwargs["point_tolerance"] = mesh_kwargs["point_tolerance"]
    if "cut_fuzzy_value" in mesh_kwargs:
        cad_kwargs["cut_fuzzy_value"] = mesh_kwargs.pop("cut_fuzzy_value")
    if "fragment_fuzzy_value" in mesh_kwargs:
        cad_kwargs["fragment_fuzzy_value"] = mesh_kwargs.pop("fragment_fuzzy_value")
    if "canonicalize_topology" in mesh_kwargs:
        cad_kwargs["canonicalize_topology"] = mesh_kwargs.pop("canonicalize_topology")
    if "perturbation" in mesh_kwargs:
        cad_kwargs["perturbation"] = mesh_kwargs.pop("perturbation")
    progress_bars = mesh_kwargs.pop("progress_bars", False)
    cad_kwargs["progress_bars"] = progress_bars

    point_tolerance = cad_kwargs.get(
        "point_tolerance", mesh_kwargs.get("point_tolerance", 1e-3)
    )
    # Default mirrors ``CAD_OCC.__init__`` (perturbation=1e-5 when None);
    # use ``is None`` so an explicit ``perturbation=0.0`` is respected.
    perturbation = cad_kwargs.get("perturbation")
    if perturbation is None:
        perturbation = 1e-5

    # --- Stage 1a: shapely intake pre-pass. -----------------------------
    # Apply the polygon-buffer + InterfaceTag resolve BEFORE the
    # structured pre-pass so the cohort compound is built from the same
    # perturbed XY that unstructured neighbours see at BOP time.
    # ``prepare_entities`` is NOT idempotent; cad_occ is invoked with
    # ``prepared=True`` below to skip the duplicate buffer.
    prepare_entities(
        entities,
        perturbation=perturbation,
        resolve_snap=max(perturbation, point_tolerance),
    )

    # --- Stage 1b: structured pre-pass. ---------------------------------
    state = structured_pre_pass(entities, point_tolerance=point_tolerance)

    occ_entities_raw, _cad_processor = cad_occ(
        state.entities_out, return_processor=True, prepared=True, **cad_kwargs
    )

    # Diagnostic: confirm BOP didn't subdivide any pre-baked cohort
    # shell face. Walks every cohort compound, collects the original
    # TopoDS_Face by ShapeKey, then calls validate_cohort_shells, which
    # raises CohortShellModifiedError on a >1 fragment count.
    if state.slab_meta and _cad_processor.last_fragment_builder is not None:
        faces_by_key = _collect_faces_by_key(state)
        validate_cohort_shells(
            state.slab_meta,
            faces_by_key,
            builder=_cad_processor.last_fragment_builder,
        )

    occ_entities = structured_post_pass(occ_entities_raw, state)

    # --- Stage 2: XAO emit (+ optional checkpoint) + gmsh load. ---------
    interface_delimiter = mesh_kwargs.pop("interface_delimiter", "___")
    boundary_delimiter = mesh_kwargs.pop("boundary_delimiter", "None")
    remove_all_duplicates = mesh_kwargs.pop("remove_all_duplicates", False)

    mm = ModelManager()
    mm.ensure_initialized(str(mm.filename))
    gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)

    mm.load_occ_entities(
        occ_entities,
        remove_all_duplicates=remove_all_duplicates,
        interface_delimiter=interface_delimiter,
        boundary_delimiter=boundary_delimiter,
        interface_aabb_tolerance=default_interface_aabb_tolerance(point_tolerance),
    )

    if checkpoint_cad:
        mm.save_to_xao(Path(checkpoint_cad))

    # --- Stage 2a: build ShapeKey -> gmsh-tag maps for the wedge hooks. -
    face_tag_by_key: dict[ShapeKey, int] = {}
    sub_solid_tag_by_key: dict[ShapeKey, int] = {}
    if state.slab_meta:
        face_tag_by_key, sub_solid_tag_by_key = _build_tag_maps_from_names(state)

    # --- Stage 2b: hook wiring. -----------------------------------------
    user_pre_2d = mesh_kwargs.pop("pre_2d_hook", None)
    user_pre_3d = mesh_kwargs.pop("pre_3d_hook", None)
    resolution_specs_for_wedge = mesh_kwargs.get("resolution_specs")

    def _structured_pre_2d() -> None:
        if state.slab_meta and face_tag_by_key:
            freeze_lateral_mesh(
                state.slab_meta,
                face_tag_by_key,
                resolution_specs=resolution_specs_for_wedge,
            )
        if user_pre_2d is not None:
            user_pre_2d()
        # Strip synthetic ``__cohort_*`` bookkeeping groups so they
        # don't leak into the .msh output. Done after the user's hook
        # so user code can still inspect them if needed. (Face and
        # solid tag maps were resolved before this hook runs, so the
        # synthetic groups are no longer needed downstream.)
        if state.slab_meta:
            _strip_synthetic_physical_groups()

    def _structured_pre_3d() -> None:
        if state.slab_meta and face_tag_by_key and sub_solid_tag_by_key:
            stamp_wedges(
                state.slab_meta,
                face_tag_by_key,
                sub_solid_tag_by_key,
                resolution_specs=resolution_specs_for_wedge,
                point_tolerance=point_tolerance,
            )
            # The cohort sub-solids are now fully meshed with wedges;
            # tell gmsh's tet algorithm to only fill the remaining
            # unstructured volumes.
            gmsh.option.setNumber("Mesh.MeshOnlyEmpty", 1)
            # Deduplicate nodes only within the structured sub-solid volumes.
            # A global removeDuplicateNodes() can corrupt boundary meshes of
            # adjacent unstructured volumes: BOP sometimes produces duplicate
            # topological curves/faces at the interface between structured and
            # unstructured regions.  The duplicate curves have their own node
            # sets; after global dedup one curve's nodes survive and the other
            # curve's elements reference those surviving nodes, which may be
            # classified on the wrong topological entity, causing generate(3)
            # to fail for the unstructured volume.
            #
            # Scoping dedup to the structured volumes is safe because the
            # intermediate-layer duplicate nodes (z_layer from stamp_wedges vs
            # lateral transfinite edge nodes) are eliminated upstream in
            # _stamp_one (step 4 reuses existing nodes instead of creating new
            # ones).  No genuine duplicates remain within the structured volumes
            # after stamp_wedges; the scoped call is a safe no-op for them
            # while leaving the unstructured boundary mesh untouched.
            structured_vol_dimtags = [(3, tag) for tag in sub_solid_tag_by_key.values()]
            _remove_duplicate_nodes_tight(structured_vol_dimtags)
        if user_pre_3d is not None:
            user_pre_3d()

    def _structured_post_3d() -> None:
        if state.slab_meta and face_tag_by_key and sub_solid_tag_by_key:
            # After generate(3) has filled all unstructured volumes, perform a
            # global removeDuplicateNodes() to clean up the z-interface dups
            # that were intentionally left by the scoped pre-3D dedup.  At
            # this point all volumes are fully meshed, so the dedup cannot
            # corrupt any pending generate() pass.
            _remove_duplicate_nodes_tight()

    has_structured = bool(state.slab_meta)
    pre_2d_hook = _structured_pre_2d if (has_structured or user_pre_2d) else None
    pre_3d_hook = _structured_pre_3d if (has_structured or user_pre_3d) else None
    post_3d_hook = _structured_post_3d if has_structured else None

    # --- Stage 3: mesh. -------------------------------------------------
    return mesh(
        dim=dim,
        model=mm,
        output_file=Path(output_mesh) if output_mesh else None,
        pre_2d_hook=pre_2d_hook,
        pre_3d_hook=pre_3d_hook,
        post_3d_hook=post_3d_hook,
        **mesh_kwargs,
    )


# ---------------------------------------------------------------------------
# Walk pre-BOP cohort compounds and index every face by ShapeKey so the
# shell-invariance validator can hand TopoDS_Face objects to
# BOPAlgo_Builder.Modified(). The faces collected here are the exact
# same TShapes referenced by slab_meta (bot/top/lateral) because the
# pre-pass build constructed each face once.
# ---------------------------------------------------------------------------


def _collect_faces_by_key(state: StructuredState):
    """Return {ShapeKey: TopoDS_Face} for every face in every cohort compound."""
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer

    from meshwell.structured.build import _shape_key

    out: dict[ShapeKey, Any] = {}
    for ce in state.cohort_entities:
        exp = TopExp_Explorer(ce.compound, TopAbs_FACE)
        while exp.More():
            face = exp.Current()
            fk = _shape_key(face)
            out.setdefault(fk, face)
            exp.Next()
    return out


# ---------------------------------------------------------------------------
# Synthetic-physical-name lookup. The pre-pass assigns each tracked
# cohort face/solid a unique synthetic name (e.g.
# ``__cohort_0__slab_3__bot``); the post-pass writes them into the XAO
# via synthetic dim=2/dim=3 OCCLabeledEntities. After gmsh loads the
# XAO, we recover each pre-BOP ShapeKey -> gmsh entity tag binding by
# looking up the synthetic name in the gmsh physical-group table.
# ---------------------------------------------------------------------------


def _build_tag_maps_from_names(
    state: StructuredState,
) -> tuple[dict[ShapeKey, int], dict[ShapeKey, int]]:
    """Resolve synthetic physical-group names into ``{ShapeKey: gmsh_tag}``.

    Returns ``(face_tag_by_key, sub_solid_tag_by_key)``. Missing names
    (e.g. a face that BOP merged into a neighbour and no longer exists
    under its synthetic name) are silently skipped.
    """
    # Pre-compute name -> (dim, gmsh_tag) once.
    name_to_entity: dict[str, tuple[int, int]] = {}
    for dim, gtag in gmsh.model.getPhysicalGroups():
        gname = gmsh.model.getPhysicalName(dim, gtag)
        if not gname.startswith("__cohort_"):
            continue
        entities = gmsh.model.getEntitiesForPhysicalGroup(dim, gtag)
        if len(entities) >= 1:
            name_to_entity[gname] = (dim, int(entities[0]))

    face_tag_by_key: dict[ShapeKey, int] = {}
    for fk, name in state.face_name_by_key.items():
        hit = name_to_entity.get(name)
        if hit is not None and hit[0] == 2:
            face_tag_by_key[fk] = hit[1]

    sub_solid_tag_by_key: dict[ShapeKey, int] = {}
    for sk, name in state.sub_solid_name_by_key.items():
        hit = name_to_entity.get(name)
        if hit is not None and hit[0] == 3:
            sub_solid_tag_by_key[sk] = hit[1]

    return face_tag_by_key, sub_solid_tag_by_key


def _strip_synthetic_physical_groups() -> None:
    """Remove ``__cohort_`` synthetic groups (and their now-stale names) from gmsh.

    Called right before :func:`mesh` writes the .msh so synthetic
    bookkeeping groups don't leak into the output.
    """
    to_remove: list[tuple[int, int]] = []
    names_to_drop: list[str] = []
    for dim, gtag in gmsh.model.getPhysicalGroups():
        gname = gmsh.model.getPhysicalName(dim, gtag)
        if gname.startswith("__cohort_"):
            to_remove.append((dim, gtag))
            names_to_drop.append(gname)
    if to_remove:
        gmsh.model.removePhysicalGroups(to_remove)
        for gname in names_to_drop:
            with contextlib.suppress(Exception):
                gmsh.model.removePhysicalName(gname)
