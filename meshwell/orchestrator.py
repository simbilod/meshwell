"""Unified CAD backend pipeline orchestrator."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from meshwell.backend_gmsh import GmshBackend
from meshwell.backend_occ import OccBackend
from meshwell.mesh import mesh
from meshwell.model import ModelManager


def generate_mesh(
    entities: list[Any],
    dim: int,
    output_mesh: Path | str | None = None,
    backend: str = "occ",
    checkpoint_cad: Path | str | None = None,
    **mesh_kwargs,
) -> Any:
    """Unified API for generating a mesh from a list of entities.

    Args:
        entities: List of meshwell entities (PolySurface, PolyPrism, etc.)
        dim: Dimension of the mesh to generate
        output_mesh: Optional path to save the generated mesh (.msh)
        backend: CAD backend to use ("occ" or "gmsh")
        checkpoint_cad: Optional path to save the CAD state (.xao)
        **mesh_kwargs: Additional arguments for the mesh() function

    Returns:
        meshio.Mesh: The generated mesh object
    """
    if backend == "occ":
        backend_obj = OccBackend()
    elif backend == "gmsh":
        backend_obj = GmshBackend()
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
