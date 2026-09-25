"""Unified CAD -> XAO -> mesh pipeline."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import gmsh

from meshwell.cad_common import apply_arc_params, prepare_entities
from meshwell.cad_occ import cad_occ
from meshwell.cad_settings import CADSettings
from meshwell.mesh import mesh
from meshwell.model import ModelManager
from meshwell.occ_xao_writer import default_interface_aabb_tolerance, write_xao
from meshwell.structured.pipeline import (
    StructuredState,
    structured_post_pass,
    structured_pre_pass,
)
from meshwell.structured.types import ShapeKey
from meshwell.structured.validators import validate_cohort_shells
from meshwell.utils import deserialize


def cad(
    entities: list[Any],
    output_file: Path | str | None = None,
    registry: dict[str, Callable[..., Any]] | None = None,
    sweeps: list[Any] | None = None,
    point_tolerance: float | None = None,
    perturbation: float | None = None,
    identify_arcs: bool | None = None,
    min_arc_points: int | None = None,
    arc_tolerance: float | None = None,
    cut_fuzzy_value: float | None = None,
    fragment_fuzzy_value: float | None = None,
    canonicalize_topology: bool | None = None,
    n_threads: int | None = None,
    progress_bars: bool = False,
    interface_delimiter: str = "___",
    boundary_delimiter: str = "None",
    model_name: str = "meshwell",
    cad_settings: CADSettings | None = None,
) -> list[Any]:
    """Run the OpenCASCADE + structured CAD pipeline and optionally write a self-describing ``.xao``.

    Does not initialize or require Gmsh. Both 3D structured cohorts
    (``PolyPrism(structured=True)``) and 2D structured sweeps (``sweeps``)
    encode their metadata into synthetic physical groups in the output
    entities / ``.xao`` so a subsequent :func:`meshwell.mesh.mesh` call
    with ``input_file=output_file`` automatically discovers and executes
    the appropriate structured meshing hooks. The resolved
    :class:`~meshwell.cad_settings.CADSettings` are embedded in the
    ``.xao`` as well, so the mesh stage uses the same ``point_tolerance``.

    Numerical settings are given EITHER as ``cad_settings`` OR as the
    individual kwargs below (``None`` = package default from
    :mod:`meshwell.cad_settings`); mixing both raises ``TypeError``.

    Args:
        entities: List of meshwell entities or their ``to_dict()`` dicts.
        output_file: Optional destination ``.xao`` path.
        registry: Optional callable registry for ``OCC_entity`` deserialization.
        sweeps: Optional list of ``StructuredSweep`` instances or dicts.
        point_tolerance: Coordinate quantization and grid-snap tolerance.
        perturbation: Analytic outward offset for same-``mesh_order`` boundaries.
        identify_arcs: Scene-wide arc identification flag.
        min_arc_points: Minimum run length for arc fitting when ``identify_arcs`` is set.
        arc_tolerance: Circle-fit tolerance when ``identify_arcs`` is set.
        cut_fuzzy_value: Optional ``BRepAlgoAPI_Cut`` fuzzy override.
        fragment_fuzzy_value: Optional ``BOPAlgo_Builder`` fragment fuzzy override.
        canonicalize_topology: Optional post-fragment TShape canonicalization flag.
        n_threads: Optional thread count for OCC boolean fragmentation.
        progress_bars: Whether to display progress bars during CAD booleans.
        interface_delimiter: Delimiter for ``A___B`` interface physical groups.
        boundary_delimiter: Delimiter for ``A___None`` exterior boundary groups.
        model_name: XAO ``<geometry name=...>`` attribute.
        cad_settings: Complete :class:`~meshwell.cad_settings.CADSettings`.

    Returns:
        list[OCCLabeledEntity]: Post-BOP labeled OCC entities ready for XAO serialization.
    """
    settings = CADSettings.from_kwargs(
        cad_settings,
        point_tolerance=point_tolerance,
        perturbation=perturbation,
        identify_arcs=identify_arcs,
        min_arc_points=min_arc_points,
        arc_tolerance=arc_tolerance,
        cut_fuzzy_value=cut_fuzzy_value,
        fragment_fuzzy_value=fragment_fuzzy_value,
    )
    point_tolerance = settings.point_tolerance

    entities = deserialize(entities, registry=registry)

    # Arc identification is a scene-level setting (see apply_arc_params):
    # resolve it onto every entity once, here, so downstream code reads
    # the entity attributes directly instead of guessing defaults.
    apply_arc_params(
        entities,
        identify_arcs=settings.identify_arcs,
        min_arc_points=settings.min_arc_points,
        arc_tolerance=settings.arc_tolerance,
    )

    prepare_entities(
        entities,
        perturbation=settings.perturbation,
        resolve_snap=settings.resolve_snap,
        buffer_polygons=False,
    )

    parsed_sweeps = None
    if sweeps:
        from meshwell.structured.sweep import StructuredSweep

        parsed_sweeps = [
            StructuredSweep.from_dict(s) if isinstance(s, dict) else s for s in sweeps
        ]

    state = structured_pre_pass(
        entities, point_tolerance=point_tolerance, sweeps=parsed_sweeps
    )

    cad_kwargs: dict[str, Any] = {
        "point_tolerance": point_tolerance,
        "perturbation": settings.perturbation,
        "cut_fuzzy_value": settings.cut_fuzzy_value,
        "fragment_fuzzy_value": settings.fragment_fuzzy_value,
        "progress_bars": progress_bars,
    }
    if canonicalize_topology is not None:
        cad_kwargs["canonicalize_topology"] = canonicalize_topology
    if n_threads is not None:
        cad_kwargs["n_threads"] = n_threads

    occ_entities_raw, _cad_processor = cad_occ(
        state.entities_out, return_processor=True, prepared=True, **cad_kwargs
    )

    if state.slab_meta and _cad_processor.last_fragment_builder is not None:
        faces_by_key = _collect_faces_by_key(state)
        validate_cohort_shells(
            state.slab_meta,
            faces_by_key,
            builder=_cad_processor.last_fragment_builder,
        )

    occ_entities = structured_post_pass(occ_entities_raw, state)

    if parsed_sweeps:
        from meshwell.structured.sweep_cad import sweep_imprint_pass

        occ_entities = sweep_imprint_pass(
            occ_entities, parsed_sweeps, entities, point_tolerance
        )

    # Provenance travels with the entities (write_xao / load_occ_entities
    # pick it up); the pipeline-level settings are authoritative.
    for ent in occ_entities:
        ent.cad_settings = settings

    if output_file is not None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_xao(
            occ_entities,
            output_path,
            model_name=model_name,
            interface_delimiter=interface_delimiter,
            boundary_delimiter=boundary_delimiter,
            interface_aabb_tolerance=default_interface_aabb_tolerance(point_tolerance),
            cad_settings=settings,
        )

    return occ_entities


generate_cad = cad


def generate_mesh(
    entities: list[Any],
    dim: int,
    output_mesh: Path | str | None = None,
    checkpoint_cad: Path | str | None = None,
    registry: dict[str, Callable[..., Any]] | None = None,
    backend: str | None = None,  # deprecated
    sweeps: list[Any] | None = None,
    **mesh_kwargs,
) -> Any:
    """Generate a mesh from a list of entities.

    Pipeline: :func:`cad` (structured pre-pass -> ``cad_occ`` -> structured
    post-pass -> sweep imprint pass) -> XAO load into :class:`ModelManager`
    -> :func:`mesh` (which auto-discovers structured cohort and sweep
    physical groups from the loaded model).
    """
    if backend is not None and backend != "occ":
        raise ValueError(
            f"backend={backend!r} is no longer supported. "
            "Meshwell now uses OCC exclusively for CAD."
        )

    identify_arcs = mesh_kwargs.pop("identify_arcs", None)
    arc_min_points = mesh_kwargs.pop("min_arc_points", None)
    arc_fit_tolerance = mesh_kwargs.pop("arc_tolerance", None)
    cut_fuzzy_value = mesh_kwargs.pop("cut_fuzzy_value", None)
    fragment_fuzzy_value = mesh_kwargs.pop("fragment_fuzzy_value", None)
    canonicalize_topology = mesh_kwargs.pop("canonicalize_topology", None)
    perturbation = mesh_kwargs.pop("perturbation", None)
    progress_bars = mesh_kwargs.pop("progress_bars", False)
    remove_all_duplicates = mesh_kwargs.pop("remove_all_duplicates", False)
    cad_settings = mesh_kwargs.pop("cad_settings", None)

    settings = CADSettings.from_kwargs(
        cad_settings,
        point_tolerance=mesh_kwargs.pop("point_tolerance", None),
        perturbation=perturbation,
        identify_arcs=identify_arcs,
        min_arc_points=arc_min_points,
        arc_tolerance=arc_fit_tolerance,
        cut_fuzzy_value=cut_fuzzy_value,
        fragment_fuzzy_value=fragment_fuzzy_value,
    )
    point_tolerance = settings.point_tolerance
    n_threads = mesh_kwargs.get("n_threads")
    interface_delimiter = mesh_kwargs.get("interface_delimiter", "___")
    boundary_delimiter = mesh_kwargs.get("boundary_delimiter", "None")

    occ_entities = cad(
        entities=entities,
        registry=registry,
        sweeps=sweeps,
        cad_settings=settings,
        canonicalize_topology=canonicalize_topology,
        n_threads=n_threads,
        progress_bars=progress_bars,
        interface_delimiter=interface_delimiter,
        boundary_delimiter=boundary_delimiter,
    )

    mm = ModelManager(point_tolerance=point_tolerance)
    mm.cad_settings = settings
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

    return mesh(
        dim=dim,
        model=mm,
        output_file=Path(output_mesh) if output_mesh else None,
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
