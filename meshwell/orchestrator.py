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
            - ``fuzzy_value`` (float): BOPAlgo fuzzy value for the
              all-fragment pass.
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
    if "fuzzy_value" in mesh_kwargs:
        cad_kwargs["fuzzy_value"] = mesh_kwargs.pop("fuzzy_value")
    if "canonicalize_topology" in mesh_kwargs:
        cad_kwargs["canonicalize_topology"] = mesh_kwargs.pop("canonicalize_topology")
    progress_bars = mesh_kwargs.pop("progress_bars", False)
    cad_kwargs["progress_bars"] = progress_bars

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
    return mesh(
        dim=dim,
        model=mm,
        output_file=Path(output_mesh) if output_mesh else None,
        **mesh_kwargs,
    )
