"""Mesh comparison utilities."""
from __future__ import annotations

from typing import Any
from difflib import unified_diff
from pathlib import Path

from meshwell.config import PATH


def compare_gmsh_files(meshfile: Path, other_meshfile: Path | None = None) -> None:
    """Compare two GMSH mesh files and assert they are identical.

    Args:
        meshfile: Path to the mesh file to compare
        other_meshfile: Optional path to compare against. If None, uses reference file
                       from PATH.references directory

    Raises:
        AssertionError: If the files differ
    """
    meshfile2 = meshfile
    if other_meshfile is None:
        meshfile1 = PATH.references / str(meshfile)
    else:
        meshfile1 = other_meshfile

    with meshfile1.open() as f:
        expected_lines = f.readlines()
    with meshfile2.open() as f:
        actual_lines = f.readlines()

    diff = list(unified_diff(expected_lines, actual_lines))
    if diff != []:
        raise ValueError(f"Mesh {meshfile!s} differs from its reference!")


def compare_mesh_headers(meshfile: Path, other_meshfile: Path | None = None) -> None:
    """Compare mesh file headers.

    Args:
        meshfile: Path to the mesh file whose header to compare
        other_meshfile: Optional path to compare against. If None, uses reference file
                       with .reference.msh extension from PATH.references directory
    """
    meshfile2 = meshfile
    if other_meshfile is None:
        meshfile1 = PATH.references / (str(meshfile.with_suffix("")) + ".reference.msh")
    else:
        meshfile1 = other_meshfile

    with meshfile1.open() as f:
        expected_lines = []
        for line in f:
            expected_lines.append(line)
            if line.startswith("$Entities"):
                # Get one more line after $Entities
                expected_lines.append(next(f))
                break

    with meshfile2.open() as f:
        actual_lines = []
        for line in f:
            actual_lines.append(line)
            if line.startswith("$Entities"):
                # Get one more line after $Entities
                actual_lines.append(next(f))
                break

    diff = list(unified_diff(expected_lines, actual_lines))
    if diff != []:
        raise ValueError(f"Mesh headers in {meshfile!s} differ from reference!")


def deserialize(data: Any, registry: dict[str, callable] | None = None) -> Any:
    """Reconstruct meshwell entities or resolution specs from dictionary representation.

    Args:
        data: Dictionary, list, or nested structure containing serialized data
        registry: Optional registry for OCC_entity function resolution

    Returns:
        Reconstructed objects
    """
    if isinstance(data, list):
        return [deserialize(item, registry=registry) for item in data]

    if isinstance(data, dict):
        if "type" in data:
            t = data["type"]
            if t == "GMSH_entity":
                from meshwell.gmsh_entity import GMSH_entity

                return GMSH_entity.from_dict(data)
            if t == "OCC_entity":
                from meshwell.occ_entity import OCC_entity

                return OCC_entity.from_dict(data, registry=registry)
            if t == "PolyLine":
                from meshwell.polyline import PolyLine

                return PolyLine.from_dict(data)
            if t == "PolySurface":
                from meshwell.polysurface import PolySurface

                return PolySurface.from_dict(data)
            if t == "PolyPrism":
                from meshwell.polyprism import PolyPrism

                return PolyPrism.from_dict(data)
            if t == "ResolutionSpec":
                import numpy as np

                from meshwell import resolution

                res_type = data["resolution_type"]
                params = data.copy()
                del params["type"]
                del params["resolution_type"]
                # Handle inf
                for k, v in params.items():
                    if v == "inf":
                        params[k] = np.inf

                res_class = getattr(resolution, res_type)
                return res_class.model_validate(params)

        # If it's a normal dict (like resolutions dict), recurse into values
        return {k: deserialize(v, registry=registry) for k, v in data.items()}

    return data
