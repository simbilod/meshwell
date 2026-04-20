"""Unified CAD/mesh pipeline orchestrator (OCC backend only)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

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
    """Generate a mesh from a list of entities via the OCC CAD pipeline.

    Args:
        entities: List of meshwell entities or their dictionary representations.
        dim: Dimension of the mesh to generate.
        output_mesh: Optional path to save the generated mesh (.msh).
        checkpoint_cad: Optional path to save the CAD state (.xao).
        registry: Optional registry for ``OCC_entity`` function resolution.
        backend: Deprecated. Only ``"occ"`` (or ``None``) is accepted; any
            other value raises ``ValueError``.
        **mesh_kwargs: Additional arguments forwarded to the ``mesh()``
            function. Also consumes a few OCC-backend kwargs:

            - ``progress_bars`` (bool): tqdm progress bars for CAD and
              OCC→gmsh steps.
            - ``fuzzy_value`` (float): BOPAlgo fuzzy value for the
              all-fragment pass; independent of ``point_tolerance`` (which
              drives the geometry cache quantizer) so you can raise
              boolean slop without collapsing small features.
            - ``canonicalize_topology`` (bool): post-fragmentation TShape
              canonicalization across entities (see
              :func:`meshwell.occ_canonicalize.canonicalize_topology`).
            - ``remove_all_duplicates`` (bool): gmsh-level fragment safety
              net across imported dimtags.
            - ``use_xao`` (bool): import OCC shapes via an inline XAO file
              with per-entity marker physical groups (TShape-exact
              identity, no mass/centroid heuristic).

    Returns:
        meshio.Mesh: The generated mesh object.
    """
    if backend is not None and backend != "occ":
        raise ValueError(
            f"backend={backend!r} is no longer supported. "
            "Meshwell now uses OCC exclusively for CAD."
        )

    entities = deserialize(entities, registry=registry)

    # Extract OCC-backend kwargs from mesh_kwargs.
    backend_kwargs: dict[str, Any] = {}
    if "n_threads" in mesh_kwargs:
        backend_kwargs["n_threads"] = mesh_kwargs["n_threads"]
    if "point_tolerance" in mesh_kwargs:
        backend_kwargs["point_tolerance"] = mesh_kwargs["point_tolerance"]
    if "fuzzy_value" in mesh_kwargs:
        backend_kwargs["fuzzy_value"] = mesh_kwargs.pop("fuzzy_value")
    if "canonicalize_topology" in mesh_kwargs:
        backend_kwargs["canonicalize_topology"] = mesh_kwargs.pop(
            "canonicalize_topology"
        )
    backend_kwargs["progress_bars"] = mesh_kwargs.pop("progress_bars", False)
    backend_kwargs["remove_all_duplicates"] = mesh_kwargs.pop(
        "remove_all_duplicates", False
    )
    backend_kwargs["use_xao"] = mesh_kwargs.pop("use_xao", False)

    from meshwell.backend_occ import OccBackend

    backend_obj = OccBackend(**backend_kwargs)
    backend_obj.process_entities(entities)

    if checkpoint_cad:
        backend_obj.save_checkpoint(Path(checkpoint_cad))

    mm = ModelManager()
    backend_obj.to_gmsh_model(mm)

    return mesh(
        dim=dim,
        model=mm,
        output_file=Path(output_mesh) if output_mesh else None,
        **mesh_kwargs,
    )
