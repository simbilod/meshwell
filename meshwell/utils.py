"""Mesh comparison utilities."""
from __future__ import annotations

import os
import shutil
from difflib import unified_diff
from pathlib import Path
from typing import Any

from meshwell.config import PATH


def _is_generating_references() -> bool:
    """Whether we are currently regenerating the reference fixtures.

    When set (by ``tests/generate_references.py``), the comparison helpers
    below switch from compare-mode to snapshot-mode: they copy the input
    file to the reference location instead of asserting equality. This
    lets the same test code drive both normal runs (byte-by-byte against
    a baseline produced from a previous commit) and reference-baseline
    refresh runs, without the tests having to know which mode is active.
    """
    return bool(os.environ.get("MESHWELL_GENERATING_REFERENCES"))


def compare_gmsh_files(meshfile: Path, other_meshfile: Path | None = None) -> None:
    """Compare a generated GMSH mesh file against its reference.

    Args:
        meshfile: Path to the generated mesh file to compare. May be an
                  absolute path (e.g. ``tmp_path / "foo.msh"``); only the
                  basename is used when deriving the reference path.
        other_meshfile: Optional explicit path to compare against. If None,
                       the reference is looked up in ``PATH.references`` by
                       the basename of ``meshfile``.

    Behaviour:
        * In normal runs: asserts that ``meshfile`` is byte-identical
          (line-diff) to the reference, raising ``ValueError`` otherwise.
        * When ``MESHWELL_GENERATING_REFERENCES`` is set: copies
          ``meshfile`` to the reference location instead of comparing.
          This is how :mod:`tests.generate_references` refreshes the
          baseline fixtures without bypassing the tests' own output paths.
    """
    meshfile2 = Path(meshfile)
    if other_meshfile is None:
        meshfile1 = PATH.references / meshfile2.name
    else:
        meshfile1 = other_meshfile

    if _is_generating_references():
        meshfile1.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(meshfile2, meshfile1)
        return

    with meshfile1.open() as f:
        expected_lines = f.readlines()
    with meshfile2.open() as f:
        actual_lines = f.readlines()

    diff = list(unified_diff(expected_lines, actual_lines))
    if diff != []:
        raise ValueError(f"Mesh {meshfile!s} differs from its reference!")


def compare_mesh_headers(meshfile: Path, other_meshfile: Path | None = None) -> None:
    """Compare a generated mesh file's header against its reference.

    Args:
        meshfile: Path to the generated mesh file whose header to compare. May
                  be an absolute path; only the basename stem is used when
                  deriving the reference path.
        other_meshfile: Optional explicit path to compare against. If None,
                       uses reference file with .reference.msh extension from
                       PATH.references directory.

    Behaviour mirrors :func:`compare_gmsh_files` -- in
    ``MESHWELL_GENERATING_REFERENCES`` mode, the input file is copied to
    the reference location (header-only excerpt is NOT extracted; the
    full file is copied so the comparison side can re-extract the header).
    """
    meshfile2 = Path(meshfile)
    if other_meshfile is None:
        meshfile1 = PATH.references / (meshfile2.stem + ".reference.msh")
    else:
        meshfile1 = other_meshfile

    if _is_generating_references():
        meshfile1.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(meshfile2, meshfile1)
        return

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

            raise ValueError(f"Unknown entity type: {t!r}")

        # If it's a normal dict (like resolutions dict), recurse into values
        return {k: deserialize(v, registry=registry) for k, v in data.items()}

    return data
