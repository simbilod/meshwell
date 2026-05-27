"""Unified CAD -> XAO -> mesh pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gmsh

from meshwell.cad_occ import cad_occ
from meshwell.mesh import mesh
from meshwell.model import ModelManager
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

    Three-stage pipeline: ``cad_occ`` fragments, :func:`write_xao`
    serializes to a tagged XAO, and :func:`mesh` runs gmsh.

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
    if "perturbation" in mesh_kwargs:
        cad_kwargs["perturbation"] = mesh_kwargs["perturbation"]
    if "cut_fuzzy_value" in mesh_kwargs:
        cad_kwargs["cut_fuzzy_value"] = mesh_kwargs.pop("cut_fuzzy_value")
    if "fragment_fuzzy_value" in mesh_kwargs:
        cad_kwargs["fragment_fuzzy_value"] = mesh_kwargs.pop("fragment_fuzzy_value")
    if "canonicalize_topology" in mesh_kwargs:
        cad_kwargs["canonicalize_topology"] = mesh_kwargs.pop("canonicalize_topology")
    progress_bars = mesh_kwargs.pop("progress_bars", False)
    cad_kwargs["progress_bars"] = progress_bars

    # --- Detect structured entities and set up the structured pipeline. ---
    has_structured = any(getattr(e, "structured", False) for e in entities)
    structured_state: tuple | None = None

    if has_structured:
        from meshwell.cad_common import prepare_entities
        from meshwell.structured import (
            build_phantom_shapes,
            build_plan,
            extract_phantom_map,
        )
        from meshwell.structured.builder import (
            apply_structured_mesh,
            apply_structured_transfinite_hints,
            resolve_mesh_plan,
        )
        from meshwell.structured.phantom import _group_phantom_solids_by_entity

        # Apply the shapely perturbation buffer BEFORE structured planning
        # WHEN it actually matters — i.e. when non-structured entities are
        # mixed in. cad_occ's per-entity cut for non-structured entities
        # uses their perturbed `instanciate_occ()` shape; if structured
        # phantom solids were planned on UNPERTURBED coords, their shared
        # boundary with a perturbed neighbour drifts by ~perturbation.
        # Running prepare_entities here aligns the planner + phantoms to
        # the perturbed grid; the cad_occ-internal call becomes a no-op
        # via prepare_entities' `_meshwell_prepared` idempotency flag.
        #
        # For PURE-structured scenes, every entity is overridden so
        # `instanciate_occ()` is never called in the BOP substrate path;
        # the perturbation has no purpose and would only churn polygon
        # seam choice (shapely's buffer + intersection re-roots the
        # polygon's start vertex), causing the arc-decomposition pass to
        # produce a different provenance and break downstream OCC topology
        # on arc-heavy scenes (concentric-disc transfinite mesh, etc.).
        _perturbation = cad_kwargs.get("perturbation", 1e-5)
        _point_tolerance = cad_kwargs.get("point_tolerance", 1e-3)
        prepare_entities(
            entities,
            perturbation=_perturbation,
            resolve_snap=max(_perturbation, _point_tolerance),
        )
        # prepare_entities' shapely buffer+intersection re-roots the
        # polygon's vertex order (start vertex moves). The planner's
        # unary_union+polygonize propagates that into arrangement-edge
        # break points → different arc provenance ordering and downstream
        # OCC topology changes. Re-snap to point_tolerance and re-orient
        # CCW after buffer so the downstream planner sees a canonical
        # vertex ordering. The snap also undoes any sub-tolerance vertex
        # drift; the perturbation halo remains useful for cad_occ later
        # (it set the `_meshwell_prepared` flag, so cad_occ's internal
        # prepare_entities call is a no-op).
        import shapely
        from shapely.geometry import Polygon as _Poly
        from shapely.geometry.polygon import orient as _orient

        def _rotate_ring_to_lex_min(ring_coords):
            """Rotate a closed-ring coord list to start at its lex-min vertex."""
            coords = list(ring_coords)
            if len(coords) < 2:
                return coords
            # Drop closing duplicate if present.
            closed = coords[0] == coords[-1]
            verts = coords[:-1] if closed else coords[:]
            min_i = min(range(len(verts)), key=lambda i: verts[i])
            rotated = verts[min_i:] + verts[:min_i]
            if closed:
                rotated.append(rotated[0])
            return rotated

        def _normalize_polygon(p):
            snapped = shapely.set_precision(
                p, grid_size=_point_tolerance, mode="pointwise"
            )
            oriented = _orient(snapped, sign=1.0)
            ext = _rotate_ring_to_lex_min(list(oriented.exterior.coords))
            interiors = [
                _rotate_ring_to_lex_min(list(r.coords)) for r in oriented.interiors
            ]
            return _Poly(ext, interiors)

        for ent in entities:
            if not hasattr(ent, "polygons"):
                continue
            polys = ent.polygons if isinstance(ent.polygons, list) else [ent.polygons]
            normalized = [_normalize_polygon(p) for p in polys]
            ent.polygons = (
                normalized if isinstance(ent.polygons, list) else normalized[0]
            )

        plan = build_plan(entities)
        phantom_result = build_phantom_shapes(plan)
        overrides = _group_phantom_solids_by_entity(plan, phantom_result)
        captured_builder: list = []
        occ_entities = cad_occ(
            entities,
            entity_shape_overrides=overrides,
            cad_occ_callback=lambda b: captured_builder.append(b),
            **cad_kwargs,
        )
        if len(captured_builder) != 1:
            raise RuntimeError(
                f"cad_occ_callback should fire exactly once after BOPAlgo_Builder.Perform(); "
                f"got {len(captured_builder)} invocations."
            )
        phantom_map = extract_phantom_map(phantom_result, captured_builder[0])
        mesh_plan_obj = resolve_mesh_plan(plan, entities)
        structured_state = (plan, mesh_plan_obj, phantom_map, occ_entities)
    else:
        occ_entities = cad_occ(entities, **cad_kwargs)

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

    # --- Stage 3: mesh. -------------------------------------------------
    if has_structured:
        # Two-pass meshing: 2D first, apply_structured_mesh hook, then 3D
        # with Mesh.MeshOnlyEmpty=1 so the discrete entities built by
        # apply_structured_mesh are not re-meshed.
        _plan, _mesh_plan, _phantom_map, _occ_entities = structured_state

        def _structured_hook() -> None:
            apply_structured_mesh(_plan, _mesh_plan, _phantom_map, _occ_entities)

        def _pre_2d_hook() -> None:
            apply_structured_transfinite_hints(_mesh_plan, _phantom_map, _occ_entities)

        return mesh(
            dim=dim,
            model=mm,
            output_file=Path(output_mesh) if output_mesh else None,
            pre_3d_hook=_structured_hook,
            pre_2d_hook=_pre_2d_hook,
            **mesh_kwargs,
        )
    return mesh(
        dim=dim,
        model=mm,
        output_file=Path(output_mesh) if output_mesh else None,
        **mesh_kwargs,
    )
