"""Unified CAD -> XAO -> mesh pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gmsh

from meshwell.cad_occ import cad_occ
from meshwell.mesh import mesh
from meshwell.model import ModelManager
from meshwell.structured.pipeline import (
    StructuredState,
    structured_post_pass,
    structured_pre_pass,
)
from meshwell.structured.types import ShapeKey
from meshwell.structured.wedge import (
    apply_lateral_transfinite_hints,
    stamp_wedges,
)
from meshwell.utils import deserialize


def generate_mesh(
    entities: list[Any],
    dim: int,
    output_mesh: Path | str | None = None,
    checkpoint_cad: Path | str | None = None,
    registry: dict[str, callable] | None = None,
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
    progress_bars = mesh_kwargs.pop("progress_bars", False)
    cad_kwargs["progress_bars"] = progress_bars

    point_tolerance = cad_kwargs.get(
        "point_tolerance", mesh_kwargs.get("point_tolerance", 1e-3)
    )

    # --- Stage 1a: structured pre-pass. ---------------------------------
    state = structured_pre_pass(entities, point_tolerance=point_tolerance)

    # Pre-compute spatial fingerprints (centroid + bbox) keyed by the
    # pre-BOP ShapeKey for every solid and face the wedge hooks need to
    # locate after the XAO round-trip. The XAO serialize -> gmsh.open
    # cycle regenerates all OCC TShapes, so we cannot use TShape-pointer
    # identity to bridge OCP-space to gmsh-tag-space; spatial matching
    # is the robust path.
    shape_fingerprints = _collect_shape_fingerprints(state)

    occ_entities_raw = cad_occ(state.entities_out, **cad_kwargs)
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
    )

    if checkpoint_cad:
        mm.save_to_xao(Path(checkpoint_cad))

    # --- Stage 2a: build ShapeKey -> gmsh-tag maps for the wedge hooks. -
    face_tag_by_key: dict[ShapeKey, int] = {}
    sub_solid_tag_by_key: dict[ShapeKey, int] = {}
    if state.slab_meta and shape_fingerprints:
        face_tag_by_key, sub_solid_tag_by_key = _match_gmsh_tags(
            shape_fingerprints, point_tolerance
        )

    # --- Stage 2b: hook wiring. -----------------------------------------
    user_pre_2d = mesh_kwargs.pop("pre_2d_hook", None)
    user_pre_3d = mesh_kwargs.pop("pre_3d_hook", None)
    resolution_specs_for_wedge = mesh_kwargs.get("resolution_specs")

    def _structured_pre_2d() -> None:
        if state.slab_meta and face_tag_by_key:
            apply_lateral_transfinite_hints(
                state.slab_meta,
                face_tag_by_key,
                resolution_specs=resolution_specs_for_wedge,
            )
        if user_pre_2d is not None:
            user_pre_2d()

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
            gmsh.model.mesh.removeDuplicateNodes()
        if user_pre_3d is not None:
            user_pre_3d()

    has_structured = bool(state.slab_meta)
    pre_2d_hook = _structured_pre_2d if (has_structured or user_pre_2d) else None
    pre_3d_hook = _structured_pre_3d if (has_structured or user_pre_3d) else None

    # --- Stage 3: mesh. -------------------------------------------------
    return mesh(
        dim=dim,
        model=mm,
        output_file=Path(output_mesh) if output_mesh else None,
        pre_2d_hook=pre_2d_hook,
        pre_3d_hook=pre_3d_hook,
        **mesh_kwargs,
    )


# ---------------------------------------------------------------------------
# Spatial-fingerprint matching to bridge pre-BOP OCP shapes to post-load
# gmsh tags. The XAO serialize -> gmsh.open cycle regenerates all OCC
# TShapes, so we identify by centroid + bbox.
# ---------------------------------------------------------------------------


def _collect_shape_fingerprints(
    state: StructuredState,
) -> dict[str, dict[ShapeKey, tuple[float, ...]]]:
    """Walk cohort compounds and record (centroid, bbox, mass) per ShapeKey.

    Returns ``{"solid": {ShapeKey: fingerprint}, "face": {ShapeKey: fingerprint}}``
    where fingerprint == ``(cx, cy, cz, xmin, ymin, zmin, xmax, ymax, zmax)``.
    Mass is implicit via bbox volume — adequate to disambiguate the
    flat-bottom / flat-top / lateral-quad structured faces we track.
    """
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    from meshwell.structured.build import _shape_key

    solids: dict[ShapeKey, tuple[float, ...]] = {}
    faces: dict[ShapeKey, tuple[float, ...]] = {}

    def _fingerprint(shape) -> tuple[float, ...] | None:
        box = Bnd_Box()
        BRepBndLib.Add_s(shape, box)
        if box.IsVoid():
            return None
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        cz = (zmin + zmax) / 2.0
        return (cx, cy, cz, xmin, ymin, zmin, xmax, ymax, zmax)

    for ce in state.cohort_entities:
        exp_solids = TopExp_Explorer(ce.compound, TopAbs_SOLID)
        while exp_solids.More():
            solid = exp_solids.Current()
            sk = _shape_key(solid)
            if sk in state.slab_meta:
                fp = _fingerprint(solid)
                if fp is not None:
                    solids[sk] = fp
            exp_solids.Next()

        # All tracked face ShapeKeys live on solid boundaries in the compound.
        exp_faces = TopExp_Explorer(ce.compound, TopAbs_FACE)
        seen: set[ShapeKey] = set()
        while exp_faces.More():
            face = exp_faces.Current()
            fk = _shape_key(face)
            if fk not in seen:
                seen.add(fk)
                fp = _fingerprint(face)
                if fp is not None:
                    faces[fk] = fp
            exp_faces.Next()

    return {"solid": solids, "face": faces}


def _match_gmsh_tags(
    shape_fingerprints: dict[str, dict[ShapeKey, tuple[float, ...]]],
    point_tolerance: float,
) -> tuple[dict[ShapeKey, int], dict[ShapeKey, int]]:
    """Match gmsh entities to pre-BOP ShapeKeys by centroid + bbox.

    A gmsh face/solid matches a pre-BOP ShapeKey iff their bbox
    corners agree within a relaxed spatial tolerance. Centroid is the
    primary discriminator; bbox the secondary.
    """
    # gmsh imported BREP via XAO may have small numerical drift compared
    # to the OCP build-time shapes. Relax the per-axis match tolerance to
    # max(point_tolerance, 10x machine eps) so the lateral edges of a
    # buffer=0 prism (zero thickness in one axis) still match cleanly.
    tol = max(point_tolerance, 1e-6) * 10.0

    def _index_gmsh(dim: int) -> dict[tuple, list[tuple[int, tuple[float, ...]]]]:
        """Return centroid-bucketed gmsh fingerprints for entities of dim."""
        out: dict[tuple, list[tuple[int, tuple[float, ...]]]] = {}
        for _d, tag in gmsh.model.getEntities(dim):
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            cz = (zmin + zmax) / 2.0
            fp = (cx, cy, cz, xmin, ymin, zmin, xmax, ymax, zmax)
            bucket = (
                round(cx / tol),
                round(cy / tol),
                round(cz / tol),
            )
            out.setdefault(bucket, []).append((int(tag), fp))
        return out

    def _match(
        targets: dict[ShapeKey, tuple[float, ...]],
        gmsh_index: dict[tuple, list[tuple[int, tuple[float, ...]]]],
    ) -> dict[ShapeKey, int]:
        result: dict[ShapeKey, int] = {}
        for sk, fp in targets.items():
            cx, cy, cz = fp[0], fp[1], fp[2]
            best_tag: int | None = None
            best_d = float("inf")
            # Probe the centroid bucket and immediate neighbours to absorb
            # the tol-bin boundary case.
            base = (
                round(cx / tol),
                round(cy / tol),
                round(cz / tol),
            )
            candidates: list[tuple[int, tuple[float, ...]]] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        key = (base[0] + dx, base[1] + dy, base[2] + dz)
                        candidates.extend(gmsh_index.get(key, ()))
            for tag, g_fp in candidates:
                # Compare bbox corners (and implicitly centroid).
                if all(abs(fp[3 + i] - g_fp[3 + i]) < tol for i in range(6)):
                    d = (
                        (fp[0] - g_fp[0]) ** 2
                        + (fp[1] - g_fp[1]) ** 2
                        + (fp[2] - g_fp[2]) ** 2
                    )
                    if d < best_d:
                        best_d = d
                        best_tag = tag
            if best_tag is not None:
                result[sk] = best_tag
        return result

    face_index = _index_gmsh(2)
    solid_index = _index_gmsh(3)
    face_tag_by_key = _match(shape_fingerprints["face"], face_index)
    sub_solid_tag_by_key = _match(shape_fingerprints["solid"], solid_index)
    return face_tag_by_key, sub_solid_tag_by_key
