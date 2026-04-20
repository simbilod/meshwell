"""CAD processor facade - re-exports the OCC backend as the canonical CAD entry point.

``cad()`` is a back-compat shim: it accepts the legacy
``cad(entities_list=..., output_file=...)`` signature that ``cad_gmsh.cad``
exposed, and forwards to ``occ_to_xao(cad_occ(...), ...)``. For pure
in-memory OCC processing prefer ``cad_occ`` directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from meshwell.cad_occ import CAD_OCC, cad_occ

# Canonical names
CAD = CAD_OCC


def cad(
    entities_list: list[Any],
    output_file: str | Path | None = None,
    progress_bars: bool = False,
    **kwargs: Any,
) -> list[Any]:
    """Back-compat shim for the legacy ``cad_gmsh.cad`` signature.

    Runs the OCC fragmenter then optionally serialises to XAO.

    Args:
        entities_list: entities to process.
        output_file: optional XAO output path. If the caller omits the
            ``.xao`` suffix (as the legacy gmsh facade accepted),
            we append it.
        progress_bars: forwarded to both stages.
        **kwargs: forwarded to :func:`cad_occ`.
    """
    from meshwell.occ_to_gmsh import occ_to_xao

    occ_entities = cad_occ(entities_list, progress_bars=progress_bars, **kwargs)
    if output_file is not None:
        output_path = Path(output_file)
        if output_path.suffix != ".xao":
            output_path = output_path.with_suffix(".xao")
        occ_to_xao(occ_entities, output_path, progress_bars=progress_bars)
    return occ_entities


__all__ = ["CAD", "CAD_OCC", "cad", "cad_occ"]
