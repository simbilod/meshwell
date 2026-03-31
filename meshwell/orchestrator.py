"""Unified CAD backend pipeline orchestrator."""
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
    backend: str = "occ",
    checkpoint_cad: Path | str | None = None,
    registry: dict[str, callable] | None = None,
    **mesh_kwargs,
) -> Any:
    """Unified API for generating a mesh from a list of entities.

    Args:
        entities: List of meshwell entities or their dictionary representations.
        dim: Dimension of the mesh to generate
        output_mesh: Optional path to save the generated mesh (.msh)
        backend: CAD backend to use ("occ" or "gmsh")
        checkpoint_cad: Optional path to save the CAD state (.xao)
        registry: Optional registry for OCC_entity function resolution
        **mesh_kwargs: Additional arguments for the mesh() function

    Returns:
        meshio.Mesh: The generated mesh object
    """
    # Deserialize entities if they are dictionaries
    entities = deserialize(entities, registry=registry)

    # Extract common backend arguments from mesh_kwargs
    backend_kwargs = {}
    if "n_threads" in mesh_kwargs:
        backend_kwargs["n_threads"] = mesh_kwargs["n_threads"]
    if "point_tolerance" in mesh_kwargs:
        backend_kwargs["point_tolerance"] = mesh_kwargs["point_tolerance"]

    if backend == "occ":
        from meshwell.backend_occ import OccBackend

        backend_obj = OccBackend(**backend_kwargs)
    elif backend == "gmsh":
        from meshwell.backend_gmsh import GmshBackend

        backend_obj = GmshBackend(**backend_kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'occ' or 'gmsh'.")

    # 1. Process CAD
    backend_obj.process_entities(entities)

    # 2. Save checkpoint if requested
    if checkpoint_cad:
        backend_obj.save_checkpoint(Path(checkpoint_cad))

    # 3. Mesh from memory
    # For GmshBackend, we reuse its model manager
    if backend == "gmsh":
        mm = backend_obj.processor.model_manager
    else:
        mm = ModelManager()
        backend_obj.to_gmsh_model(mm)

    return mesh(
        dim=dim,
        model=mm,
        output_file=Path(output_mesh) if output_mesh else None,
        **mesh_kwargs,
    )
